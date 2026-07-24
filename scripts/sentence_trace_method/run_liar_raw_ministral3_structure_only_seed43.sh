#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export MREC_POLICY_CONFIG="${MREC_POLICY_CONFIG:-configs/experiment/mrec_v0.2/learned_marginal_structure_only_fullpool_minmax5_10_seed43.yaml}"
export CHECKPOINTS="${CHECKPOINTS:-checkpoint-800}"
export PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
export ACCELERATE_BIN="${ACCELERATE_BIN:-$(dirname "$PYTHON_BIN")/accelerate}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29681}"
export SFT_TRAIN_MODULE="${SFT_TRAIN_MODULE:-sft.hami_cuda_bootstrap}"
export MREC_RUNTIME_CACHE_ROOT="${MREC_RUNTIME_CACHE_ROOT:-${ROOT_DIR}/outputs/cache/runtime/structure_only_seed43}"
export CUDA_DEVICE_MEMORY_SHARED_CACHE="${CUDA_DEVICE_MEMORY_SHARED_CACHE:-/tmp/lzj_mrec_vs_seed43_hami/cudevshr.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/lzj_mrec_vs_seed43_hami/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/lzj_mrec_vs_seed43_hami/torchinductor}"
mkdir -p "$(dirname "$CUDA_DEVICE_MEMORY_SHARED_CACHE")" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

printf '[structure-only-seed43] config=%s port=%s cache=%s\n' \
  "$MREC_POLICY_CONFIG" "$MAIN_PROCESS_PORT" "$CUDA_DEVICE_MEMORY_SHARED_CACHE"

exec bash scripts/sentence_trace_method/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh

