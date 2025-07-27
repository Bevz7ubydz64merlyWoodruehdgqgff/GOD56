from transformers import GenerationConfig
import datetime
from datetime import timezone
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
import re
import torch
from typing import Callable, Optional, Dict
import shutil
import json
from transformers.trainer_utils import is_main_process
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging, time, traceback
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import HfApi
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

class CustomEvalSaveCallback(TrainerCallback):
    def __init__(
        self,
        function_when_to_evaluate: Callable,
        submission_dir: str,
        output_dir: str,
        original_model_name: str,
    ):
        self.function_when_to_evaluate = function_when_to_evaluate
        self.submission_dir = submission_dir
        self.current_best_loss = None
        self.best_checkpoint_info = None
        self.update_best_checkpoint = False
        self.output_dir = output_dir
        self.original_model_name = original_model_name
        
    def compute_loss(self, state: TrainerState, metrics):
        return metrics.get("eval_loss", None)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Custom logic to decide whether to save or evaluate
        when_to_eval = self.function_when_to_evaluate(state.global_step)
        if when_to_eval["eval"]:
            print(f"Evaluating the model at step: {state.global_step} the reason: {when_to_eval['reason']}")
            control.should_evaluate = True
        if when_to_eval["save"]:
            control.should_save = True
            control.should_training_stop = True
        return control


    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs
    ):
        # Append eval_loss to file
        eval_loss = self.compute_loss(state, metrics)
        if state.global_step < 2:
            return 
        print(f"GO INTO CUSTOMIZED EVALUATE AT STEP: {state.global_step}")
        if self.best_checkpoint_info is None or eval_loss < self.best_checkpoint_info["loss"]:
            print(f"Updating the best checkpoint info at step: {state.global_step} with eval_loss: {eval_loss}")
            self.best_checkpoint_info = {
                "loss": eval_loss,
                "step": state.global_step
            }
            self.update_best_checkpoint = True
        else:
            if self.best_checkpoint_info is not None:
                print(f" At step: {state.global_step} The eval_loss: {eval_loss} is not smaller than the current best eval_loss: {self.best_checkpoint_info['loss']}, update_best_checkpoint={self.update_best_checkpoint}")

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if is_main_process(LOCAL_RANK):
            print(f"Copy the latest checkpoint to the submission directory at step: {state.global_step}")

            if os.path.exists(self.submission_dir):
                shutil.rmtree(self.submission_dir)

            checkpoint_dirs = [
                d for d in os.listdir(self.output_dir)
                if re.match(r'^checkpoint-\d+$', d)
            ]
            if not checkpoint_dirs:
                print(f"[ERROR] No checkpoint directories found in {self.output_dir}. Skipping copy.")
                return

            latest_checkpoint = max(
                checkpoint_dirs,
                key=lambda x: int(x.split("-")[1])
            )
            checkpoint_path = os.path.join(self.output_dir, latest_checkpoint)

            print(f"Latest checkpoint found: {latest_checkpoint}")
            try:
                print(f"Detected PEFT adapter: merging with base model...")
                original_model_name = str(train_paths.get_text_base_model_path(self.original_model_name))
                base_model = AutoModelForCausalLM.from_pretrained(original_model_name, trust_remote_code=True)

                tokenizer = AutoTokenizer.from_pretrained(original_model_name, trust_remote_code=True)
                tokenizer.save_pretrained(self.submission_dir)

                num_tokens = len(tokenizer)
                base_model.resize_token_embeddings(num_tokens)
                peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
                merged_model = peft_model.merge_and_unload()
                merged_model.save_pretrained(self.submission_dir, safe_serialization=True, max_shard_size="4GB")

                print(f"Merged full model saved to {self.submission_dir}")
            except Exception as e:
                print(f"Not a PEFT model or merge failed: {e}. Falling back to raw checkpoint copy.")
                if os.path.exists(self.submission_dir):
                    shutil.rmtree(self.submission_dir)
                shutil.copytree(checkpoint_path, self.submission_dir)

            patch_model_metadata(self.submission_dir, self.original_model_name)
            patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)

            self.update_best_checkpoint = False

class GRPOCustomEvalSaveCallback(CustomEvalSaveCallback):
    def compute_loss(self, state: TrainerState, metrics):
        eval_loss = None
        if state.log_history:
            last_log_entry = state.log_history[-1]
            eval_loss = last_log_entry.get("eval_reward", None)
            print(f"choose eval_loss ({eval_loss}) as eval_reward from: last_log_entry: {last_log_entry}; \n metrics: {metrics}")
        else:
            print(f"state.log_history is empty")
            
        if eval_loss is not None:
            eval_loss = - eval_loss
            
        return eval_loss
    
    def penalize_eval_loss(self, eval_loss: float):
        if eval_loss < 0:
            return eval_loss / 3
        else:
            return eval_loss * 3


def check_remaining_time_less_than_minutes(end_time: str, minutes: int) -> bool: 
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)  # Make end_time timezone-aware in UTC
    now = datetime.datetime.now(timezone.utc)
    time_diff = end_time - now
    result =  time_diff.total_seconds() < minutes * 60
    if result:
        print(f"*** current time: {now} end_time: {end_time} time_diff: {time_diff}")
    return result


class WhenToEvalHandler:
    def __init__(self, end_time: str, save_before_remaining_time: int = 3):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval_save = False
        self.end_time = end_time

    def __call__(self, global_step: int) -> dict:
        if self.save_before_remaining_time > 0 and not self.run_eval_save:
            if check_remaining_time_less_than_minutes(self.end_time, self.save_before_remaining_time):
                # the eval time might be higher than the end_time, so we need to let the pod not stop by setting a flag for this
                self.run_eval_save = True
                return {"save": True, "eval": False, "reason": "end_time"}

        return {"save": False, "eval": False, "reason": "none"}


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