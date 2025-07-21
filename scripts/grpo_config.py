
INSTRUCT_CONFIG = {
    "0_1_b": {
        "lr": 0.0002,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 8,
    },
    "1_2_b": {
        "lr": 0.0002,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 10,
    },
    "2_4_b": {
        "lr": 0.0002,
        "distributed": "ddp",
        "gpu_count": 1,
        "batch_size": 8,
        "use_lora": True
    },
    "4_5_b": {
        "lr": 0.0002,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 8,
        "use_lora": True
    },
    "5_9_b": {
        "lr": 0.0002,
        "distributed": "ddp",
        "gpu_count": 2,
        "batch_size": 4,
        "use_lora": True
    },
    "9_12_b": {
        "lr": 0.0002,
        "distributed": "ddp",
        "gpu_count": 2,
        "use_lora": True,
        "batch_size": 4,
    },
    "12_15_b": {
        "lr": 0.0002,
        "distributed": "ddp",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 2,
    },
    "15_40_b": {
        "lr": 0.0002,
        "distributed": "ds",
        "gpu_count": 4,
        "use_lora": True,
        "batch_size": 1,
    },
    "40_80_b": {
        "lr": 0.0002,
        "distributed": "ds",
        "gpu_count": 8,
        "use_lora": True,
        "batch_size": 1,
    }        
}

for key in INSTRUCT_CONFIG:
    INSTRUCT_CONFIG[key]["label"] = key
    

def get_instruct_config(param_nums: int) -> dict:
    if param_nums < 1_000_000_000:
        return INSTRUCT_CONFIG["0_1_b"]
    elif param_nums < 2_000_000_000:
        return INSTRUCT_CONFIG["1_2_b"]
    elif param_nums < 4_000_000_000:
        return INSTRUCT_CONFIG["2_4_b"]
    elif param_nums < 5_000_000_000:
        return INSTRUCT_CONFIG["4_5_b"]
    elif param_nums < 9_000_000_000:
        return INSTRUCT_CONFIG["5_9_b"]
    elif param_nums < 12_000_000_000:
        return INSTRUCT_CONFIG["9_12_b"]
    elif param_nums < 15_000_000_000:  
        return INSTRUCT_CONFIG["12_15_b"]
    elif param_nums < 35_000_000_000:
        return INSTRUCT_CONFIG["15_40_b"]
    elif param_nums < 80_000_000_000:
        return INSTRUCT_CONFIG["40_80_b"]
    else:
        print(f"Model size {param_nums} is not supported")
        return {
            "lr": 4e-5,
            "distributed": "ds",
            "gpu_count": 8,
            "batch_size": 6,
            "use_lora": True
        }


def modify_config(axolotl_config: dict, model_name: str, model_architecture: str, param_nums: int) -> dict:
    config = get_instruct_config(param_nums)
    print(f"INSTRUCT CONFIG: {config}")
    axolotl_config["sample_packing"] = False
    axolotl_config["learning_rate"] = config["lr"]
    axolotl_config["micro_batch_size"] = config["batch_size"]
    
    if config.get("use_lora", False):
        axolotl_config["adapter"] = "lora"
        axolotl_config["lora_alpha"] = 512
        axolotl_config["lora_r"] = 128
        axolotl_config["lora_dropout"] = 0.1
        axolotl_config["lora_target_linear"] = True
    
    if config.get("distributed", "ddp") == "ds":
        axolotl_config["deepspeed"] = "/workspace/axolotl/scripts/yml_config/zero3.json"