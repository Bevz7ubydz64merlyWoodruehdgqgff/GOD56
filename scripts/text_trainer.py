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
from datetime import datetime, timezone, timedelta

import yaml
from transformers import AutoTokenizer


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import train_cst
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

from axolotl.train import Trainer
from transformers import TrainerCallback
from customized_trainer import WhenToEvalHandler, CustomEvalSaveCallback, GRPOCustomEvalSaveCallback

from axolotl.utils.dict import DictDefault
from axolotl.common.datasets import load_datasets
import training_paths as train_paths
from customized_config import customize_config, INSTRUCT, DPO, GRPO

CONFIG_DIR = "core/config/"

def create_reward_funcs_file(reward_funcs: list[str], task_id: str, destination_dir: str = CONFIG_DIR) -> list[str]:
    """
    Create a Python file with reward functions for GRPO training.

    Args:
        reward_funcs: List of strings containing Python reward function implementations
        task_id: Unique task identifier
    """
    filename = f"rewards_{task_id}"
    filepath = os.path.join(destination_dir, f"{filename}.py")

    func_names = []
    for reward_func in reward_funcs:
        if "def " in reward_func:
            func_name = reward_func.split("def ")[1].split("(")[0].strip()
            func_names.append(func_name)

    with open(filepath, "w") as f:
        f.write("# Auto-generated reward functions file\n\n")
        for reward_func in reward_funcs:
            f.write(f"{reward_func}\n\n")

    return filename, func_names


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


def get_data_size(dataset_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return len(data)


def create_config(task_id, model, dataset, dataset_type, file_format, output_dir, data_size, hours_to_complete, expected_repo_name=None,
                huggingface_username=None, huggingface_token=None, disable_upload=True):
    """Create the axolotl config file with appropriate settings."""
    dev_size = 200
    if isinstance(dataset_type, InstructTextDatasetType | DpoDatasetType):
        config_path = "/workspace/axolotl/scripts/yml_config/base.yml"
    elif isinstance(dataset_type, GrpoDatasetType):
        dev_size = 100 # as this is extremely slow now 
        config_path = "/workspace/axolotl/scripts/yml_config/base_grpo.yml"
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset_type)}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = [create_dataset_entry(dataset, dataset_type, FileFormat(file_format))]
    model_path = str(train_paths.get_text_base_model_path(model))
    config["base_model"] = model_path
    config["mlflow_experiment_name"] = dataset
    os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    # config = update_flash_attention(config, model)
    

    if isinstance(dataset_type, DpoDatasetType):
        config["rl"] = "dpo"
    elif isinstance(dataset_type, GrpoDatasetType):
        filename, reward_funcs_names = create_reward_funcs_file(
            [reward_function.reward_func for reward_function in dataset_type.reward_functions],
            task_id,
            destination_dir="/workspace/axolotl/src/",
        )
        config["trl"]["reward_funcs"] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
        config["trl"]["reward_weights"] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]

    if not disable_upload:
        hf_username = huggingface_username or os.environ.get("HUGGINGFACE_USERNAME", "rayonlabs")
        os.environ["HUGGINGFACE_USERNAME"] = hf_username

        repo_name = expected_repo_name or str(uuid.uuid4())
        config["hub_model_id"] = f"{hf_username}/{repo_name}"

        if huggingface_token:
            os.environ["HUGGINGFACE_TOKEN"] = huggingface_token
    else:
        for key in list(config.keys()):
            if key.startswith("wandb") or key.startswith("hub"):
                config.pop(key)

    if file_format != FileFormat.HF.value:
        for ds in config["datasets"]:
            ds["ds_type"] = "json"

            if "path" in ds:
                ds["path"] = "/workspace/axolotl/data"

            ds["data_files"] = [dataset] #[os.path.basename(dataset)]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        config["special_tokens"] = {"pad_token": tokenizer.eos_token}

    config_path = os.path.join("/workspace/axolotl/configs", f"{task_id}.yml")
    
    task_type = DPO if isinstance(dataset_type, DpoDatasetType) else GRPO if isinstance(dataset_type, GrpoDatasetType) else INSTRUCT
    customize_config(config, task_type, model_path, model, hours_to_complete)
    config["val_set_size"] = dev_size / data_size
    save_config(config, config_path)
    return config_path


def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF dataset name")
    parser.add_argument("--dataset-type", required=True, help="JSON string of dataset type config")
    parser.add_argument("--task-type", required=True, choices=["InstructTextTask", "DpoTask", "GrpoTask"], help="Type of task")
    parser.add_argument("--file-format", required=True, choices=["csv", "json", "hf", "s3"], help="File format")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    args = parser.parse_args()
    original_model_name = args.model
    original_task_type = args.task_type

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

    if args.file_format == FileFormat.S3.value and args.task_type == TaskType.DPOTASK.value:
        adapt_columns_for_dpo_dataset(dataset_path, dataset_type, apply_formatting=True)
    elif args.file_format == FileFormat.S3.value and args.task_type == TaskType.GRPOTASK.value:
        adapt_columns_for_grpo_dataset(dataset_path, dataset_type)

    dataset_path = copy_dataset_to_axolotl_directories(dataset_path)
    submission_dir = train_paths.get_checkpoints_output_path(args.task_id, args.expected_repo_name)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)

    output_dir = train_paths.get_training_temp_output_path(args.task_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("dataset_path: ", dataset_path, flush=True)
    print("submission_dir: ", submission_dir, flush=True)
    print("output_dir: ", output_dir, flush=True)   
    data_size = get_data_size(dataset_path)
    
    config_path = create_config(
        args.task_id,
        args.model,
        dataset_path,
        dataset_type,
        args.file_format,
        output_dir,
        data_size,
        args.hours_to_complete,
        args.expected_repo_name,
    )
    
    original_init = Trainer.__init__
    # set the value of end_time = current time in UTC + hours_to_complete
    end_time = datetime.now(timezone.utc) + timedelta(hours=args.hours_to_complete)
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print("end_time: ", end_time, flush=True)

    def patched_init(self, *args, **kwargs):
        print("************* patching Trainer.__init__", flush=True)
        callbacks = kwargs.get("callbacks", [])
        
        if original_task_type == TaskType.GRPOTASK.value:
            when_to_eval_handler = WhenToEvalHandler(end_time, save_before_remaining_time=15)
            callbacks.append(GRPOCustomEvalSaveCallback(when_to_eval_handler, submission_dir, output_dir, original_model_name))
        else:
            when_to_eval_handler = WhenToEvalHandler(end_time, save_before_remaining_time=5)
            callbacks.append(CustomEvalSaveCallback(when_to_eval_handler, submission_dir, output_dir, original_model_name))
        kwargs["callbacks"] = callbacks
        original_init(self, *args, **kwargs)

    Trainer.__init__ = patched_init
    
    print("config_path: ", config_path, flush=True)
    
    # Load the config and call training directly instead of using CLI
    from axolotl.cli.train import do_cli
    
    # Call training directly (this will use the patched Trainer.__init__)
    do_cli(config=config_path)
    
    # patch_model_metadata(output_dir, args.model)


if __name__ == "__main__":
    main()
