#!/bin/bash

# Script to run the Docker container with scripts folder synced
# This allows live editing of scripts on the host machine

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build the Docker image
echo "Building Docker image..."
# docker build -f dockerfiles/standalone-text-trainer.dockerfile -t axolotl-text-trainer .

# Run the container with scripts folder mounted as a volume
echo "Running container with scripts folder synced..."
docker run -it --rm \
    -v "${SCRIPT_DIR}/scripts:/workspace/axolotl/scripts" \
    -p 8080:8080 \
    --gpus all \
    dev_dock 

echo "Container stopped." 