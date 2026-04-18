#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export NCCL_CUMEM_HOST_ENABLE=0
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --use_deepspeed \
  --deepspeed_config_file configs/deepspeed_zero3.json \
  scripts/train_llm_baseline_sft.py --config configs/baseline_b0.yaml
