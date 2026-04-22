#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

export NCCL_CUMEM_HOST_ENABLE=0
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# 减少 CUDA 内存碎片
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 先查询 L20 的 capability；如果输出是 (8, 9)，再设 8.9
# python - <<'PY'
# import torch
# print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
# PY
export TORCH_CUDA_ARCH_LIST="8.9"
export FC_RUN_TIMESTAMP="${FC_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"

# 防止多个 rank 编译扩展时并发过高
export MAX_JOBS=4

# 训练输出将自动写入：
# outputs/liar-raw/llm_baseline/<baseline.variant 或时间戳>_<timestamp>/
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  --use_deepspeed \
  --deepspeed_config_file configs/deepspeed_zero3_v2.json \
  scripts/train_llm_baseline_sft.py --config configs/baseline_b1.yaml
