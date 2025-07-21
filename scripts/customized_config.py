DPO = "dpo"
GRPO = "grpo"
INSTRUCT = "instruct"
import re 
from huggingface_hub import HfApi
from transformers import AutoConfig
from instruct_config import MODEL_CONFIG, modify_config
from dpo_config import modify_config as modify_dpo_config
from grpo_config import modify_config as modify_grpo_config

hf_api = HfApi()


def get_model_architecture(model_name: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_name)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except:
        return "Unknown"


def get_use_liger(architecture: str) -> str:
    if architecture.lower() in [
        "qwen2forcausallm",
        "llamaforcausallm",
        "gemma2forcausallm",
        "mixtralforcausallm",
        "mistralforcausallm",
        "qwen3forcausallm",
        "phi3forcausallm",
        "gemmaforcausallm",
    ]:
        return "True"
    else:
        return "False"


def get_model_num_params(model_id: str, model_path: str) -> int:
    if model_id in MODEL_CONFIG:
        return MODEL_CONFIG[model_id]["model_size"]
    try:
        model_info = hf_api.model_info(model_path)
        size = model_info.safetensors.total
        return size
    except Exception as e:
        print(f"Error getting model size from safetensors: {e}")
        try:
            model_size = re.search(r"(\d+)(?=[bB])", model_id)
            model_size = (
                int(model_size.group(1)) * 1_000_000_000 if model_size else None
            )
            print(f"Model size from regex: {model_size}")
            return model_size
        except Exception as e:
            print(f"Error getting model size from regex: {e}")
            return None


def disable_flash_attention(architecture: str, model: str) -> str:
    if model == "microsoft/phi-2":
        return "True"
    if "falcon-rw" in model.lower(): # ex, tiiuae/falcon-rw-1b
        return "True"
    #if model == "databricks/dolly-v2-3b":
    #    return "True"
    if architecture.strip().lower() in ["gptneoforcausallm", "bloomforcausallm"]:
        return "True"
    else:
        return "False"


def get_use_vllm(architecture: str, model: str) -> str:
    if model in ["Eurdem/Defne_llama3_2x8B", "heegyu/WizardVicuna-open-llama-3b-v2", "openlm-research/open_llama_3b", "TitanML/tiny-mixtral", "dunzhang/stella_en_1.5B_v5", "oopsung/llama2-7b-n-ox-test-v1", "microsoft/phi-2", "databricks/dolly-v2-3b"]:
        return False
    if "falcon-rw" in model.lower():
        return False
    
    if architecture in ["gptneoforcausallm", "bloomforcausallm"]:
        return False
    else:
        return True


def get_gradient_checkpointing(model: str) -> str:
    if "falcon-rw" in model.lower():
        return "False"
    return "True"
    

def customize_config(config: dict, task_type: str, model_path: str, model_name: str, hours_to_complete: float):
    model_architecture = get_model_architecture(model_path)
    param_nums = get_model_num_params(model_name, model_path)
    
    if task_type == INSTRUCT:
        modify_config(config, model_name, model_architecture, param_nums)
    elif task_type == DPO:
        modify_dpo_config(config, model_name, model_architecture, param_nums)
    elif task_type == GRPO:
        modify_grpo_config(config, model_name, model_architecture, param_nums)
