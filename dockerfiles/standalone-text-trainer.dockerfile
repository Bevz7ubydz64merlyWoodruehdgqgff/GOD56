FROM axolotlai/axolotl:main-py3.11-cu124-2.5.1

# Install core dependencies from pyproject.toml
RUN pip install mlflow huggingface_hub aiohttp pydantic requests toml \
    "fiber @ git+https://github.com/rayonlabs/fiber.git@2.4.0" \
    fastapi uvicorn httpx loguru python-dotenv \
    scipy numpy datasets tenacity minio \
    transformers pandas==2.2.3 tiktoken==0.8.0 sentencepiece==0.2.0 peft Pillow==11.1.0 PyYAML \
    requests huggingface_hub textstat==0.7.7 langcheck detoxify

WORKDIR /workspace/axolotl/scripts
RUN mkdir -p /workspace/axolotl/configs \
    /workspace/axolotl/outputs \
    /workspace/axolotl/data \
    /workspace/input_data

ENV CONFIG_DIR="/workspace/axolotl/configs"
ENV OUTPUT_DIR="/workspace/axolotl/outputs"

# COPY core /workspace/core
# COPY miner /workspace/miner
# COPY trainer /workspace/trainer
# COPY scripts /workspace/scripts
# COPY core/config/base.yml /workspace/axolotl/base.yml
# COPY core/config/base_grpo.yml /workspace/axolotl/base_grpo.yml

# Create scripts directory for volume mounting
RUN mkdir -p /workspace/axolotl/scripts

# COPY scripts /workspace/axolotl/scripts
# Note: When running with volume mount, this COPY will be overridden
# but it's kept for cases when volume mount is not used
COPY scripts /workspace/axolotl/scripts

RUN python3 /workspace/axolotl/scripts/core/manual_reward_funcs.py

# Make scripts executable
RUN chmod +x /workspace/axolotl/scripts/run_train.sh

# RUN chmod +x /workspace/scripts/run_text_trainer.sh
# RUN chmod +x /workspace/scripts/text_trainer.py

# ENTRYPOINT ["/workspace/scripts/run_text_trainer.sh"]
# ENTRYPOINT ["tail", "-f", "/dev/null"]
ENTRYPOINT ["sh", "run_train.sh"]