#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes=4 \
  --num_machines=1 \
  --multi_gpu \
  --mixed_precision=bf16 \
  --dynamo_backend=inductor \
  --use_deepspeed \
  --deepspeed_config_file configs/deepspeed_zero3.json \
  scripts/train_llm_baseline_sft.py --config configs/baseline_b1.yaml
