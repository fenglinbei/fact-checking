#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --dynamo_backend=inductor \
  --use_deepspeed \
  --deepspeed_config_file configs/deepspeed_zero3.json \
  scripts/train_llm_baseline_sft.py --config configs/baseline_b0.yaml
