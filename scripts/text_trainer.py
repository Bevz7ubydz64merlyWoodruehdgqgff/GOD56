#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
import pathlib

import yaml
from transformers import AutoTokenizer


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import update_flash_attention
from core.dataset_utils import adapt_columns_for_dpo_dataset
from core.dataset_utils import adapt_columns_for_grpo_dataset
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
from miner.logic.job_handler import create_reward_funcs_file

from customized_config import customize_config, INSTRUCT, DPO, GRPO, get_available_gpu_count
from customized_trainer import WhenToEvalHandler, CustomEvalSaveCallback, GRPOCustomEvalSaveCallback
import torch.distributed as dist
from transformers.trainer_utils import is_main_process
from axolotl.train import Trainer
from datetime import datetime, timezone, timedelta

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

def setup_distributed():
    if get_available_gpu_count() > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://"
            )
            dist.barrier()


def patch_wandb_symlinks(base_dir:str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()


def patch_model_metadata(output_dir: str, base_model_id: str):
    try:
        adapter_config_path = os.path.join(output_dir, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                config = json.load(f)

            config["base_model_name_or_path"] = base_model_id

            with open(adapter_config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"Updated adapter_config.json with base_model: {base_model_id}", flush=True)
        else:
            print(" adapter_config.json not found", flush=True)

        readme_path = os.path.join(output_dir, "README.md")

        if os.path.exists(readme_path):
            with open(readme_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.strip().startswith("base_model:"):
                    new_lines.append(f"base_model: {base_model_id}\n")
                else:
                    new_lines.append(line)

            with open(readme_path, "w") as f:
                f.writelines(new_lines)

            print(f"Updated README.md with base_model: {base_model_id}", flush=True)
        else:
            print("README.md not found", flush=True)

    except Exception as e:
        print(f"Error updating metadata: {e}", flush=True)
        pass


def copy_dataset_to_axolotl_directories(dataset_path):
    dataset_filename = os.path.basename(dataset_path)
    data_path, root_path = train_paths.get_axolotl_dataset_paths(dataset_filename)
    shutil.copy(dataset_path, data_path)
    shutil.copy(dataset_path, root_path)

    return data_path


def create_config(task_id, model, dataset, dataset_type, file_format, output_dir, expected_repo_name=None, log_wandb=True):
    """Create the axolotl config file with appropriate settings."""
    config_path = train_paths.get_axolotl_base_config_path(dataset_type)

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = [create_dataset_entry(dataset, dataset_type, FileFormat(file_format))]
    model_path = str(train_paths.get_text_base_model_path(model))
    config["base_model"] = model_path
    config["mlflow_experiment_name"] = dataset
    os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = str(output_dir)

    if log_wandb:
        config["wandb_runid"] = f"{task_id}_{expected_repo_name}"
        config["wandb_name"] = f"{task_id}_{expected_repo_name}"
        config["wandb_mode"] = "offline"
        os.makedirs(train_cst.WANDB_LOGS_DIR, exist_ok=True)
    else:
        for key in list(config.keys()):
            if key.startswith("wandb"):
                config.pop(key)

    config = update_flash_attention(config, model)

    if isinstance(dataset_type, DpoDatasetType):
        config["rl"] = "dpo"
    elif isinstance(dataset_type, GrpoDatasetType):
        filename, reward_funcs_names = create_reward_funcs_file(
            [reward_function.reward_func for reward_function in dataset_type.reward_functions],
            task_id,
            destination_dir=train_cst.AXOLOTL_DIRECTORIES["src"],
        )
        config["trl"]["reward_funcs"] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
        config["trl"]["reward_weights"] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]

    if file_format != FileFormat.HF.value:
        for ds in config["datasets"]:
            ds["ds_type"] = "json"

            if "path" in ds:
                ds["path"] = train_cst.AXOLOTL_DIRECTORIES["data"]

            ds["data_files"] = [os.path.basename(dataset)]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        config["special_tokens"] = {"pad_token": tokenizer.eos_token}

    task_type = DPO if isinstance(dataset_type, DpoDatasetType) else GRPO if isinstance(dataset_type, GrpoDatasetType) else INSTRUCT
    customize_config(config, task_type, model_path, model)
    config["val_set_size"] = 0
    config_path = os.path.join(train_cst.AXOLOTL_DIRECTORIES["configs"], f"{task_id}.yml")
    save_config(config, config_path)
    return config_path


async def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF dataset name")
    parser.add_argument("--dataset-type", required=True, help="JSON string of dataset type config")
    parser.add_argument("--task-type", required=True, choices=["InstructTextTask", "DpoTask", "GrpoTask"], help="Type of task")
    parser.add_argument("--file-format", required=True, choices=["csv", "json", "hf", "s3"], help="File format")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()
    original_model_name = args.model
    original_task_type = args.task_type

    submission_dir = train_paths.get_checkpoints_output_path(args.task_id, args.expected_repo_name)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)
    output_dir = train_paths.get_checkpoints_output_path(args.task_id, f"{args.expected_repo_name}_temp")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(train_cst.AXOLOTL_DIRECTORIES["configs"], f"{args.task_id}.yml")

    if is_main_process(LOCAL_RANK):
        for directory in train_cst.AXOLOTL_DIRECTORIES.values():
            os.makedirs(directory, exist_ok=True)
        try:
            dataset_type_dict = json.loads(args.dataset_type)

            if args.task_type == TaskType.DPOTASK.value:
                dataset_type = DpoDatasetType(**dataset_type_dict)
            elif args.task_type == TaskType.INSTRUCTTEXTTASK.value:
                dataset_type = InstructTextDatasetType(**dataset_type_dict)
            elif args.task_type == TaskType.GRPOTASK.value:
                dataset_type = GrpoDatasetType(**dataset_type_dict)
            else:
                sys.exit(f"Unsupported task type: {args.task_type}")
        except Exception as e:
            sys.exit(f"Error creating dataset type object: {e}")

        dataset_path = train_paths.get_text_dataset_path(args.task_id)
        if args.task_type == TaskType.DPOTASK.value:
            adapt_columns_for_dpo_dataset(dataset_path, dataset_type, apply_formatting=True)
        elif args.task_type == TaskType.GRPOTASK.value:
            adapt_columns_for_grpo_dataset(dataset_path, dataset_type)

        dataset_path = copy_dataset_to_axolotl_directories(dataset_path)

        create_config(
            args.task_id,
            args.model,
            dataset_path,
            dataset_type,
            args.file_format,
            output_dir,
            args.expected_repo_name,
            log_wandb=True
        )


    setup_distributed()

    original_init = Trainer.__init__
    # set the value of end_time = current time in UTC + hours_to_complete
    end_time = datetime.now(timezone.utc) + timedelta(hours=args.hours_to_complete)
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print("end_time: ", end_time, flush=True)

    def patched_init(self, *args, **kwargs):
        callbacks = kwargs.get("callbacks", [])
        
        if original_task_type == TaskType.GRPOTASK.value:
            when_to_eval_handler = WhenToEvalHandler(end_time, save_before_remaining_time=5)
            callbacks.append(GRPOCustomEvalSaveCallback(when_to_eval_handler, submission_dir, output_dir, original_model_name))
        else:
            when_to_eval_handler = WhenToEvalHandler(end_time, save_before_remaining_time=5)
            callbacks.append(CustomEvalSaveCallback(when_to_eval_handler, submission_dir, output_dir, original_model_name))
        kwargs["callbacks"] = callbacks
        original_init(self, *args, **kwargs)

    Trainer.__init__ = patched_init
      
    # Load the config and call training directly instead of using CLI
    from axolotl.cli.train import do_cli
    
    # Call training directly (this will use the patched Trainer.__init__)
    do_cli(config=config_path)

    


if __name__ == "__main__":
    asyncio.run(main())
