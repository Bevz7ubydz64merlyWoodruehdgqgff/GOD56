#!/bin/bash
set -e
accelerate launch /workspace/scripts/text_trainer.py "$@"