#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export MREC_POLICY_CONFIG="${MREC_POLICY_CONFIG:-configs/experiment/mrec_v0.2/learned_marginal_structure_only_no_map_fullpool_minmax5_10.yaml}"
export MAP_ABLATION_MODE="no_map"
export PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
export ACCELERATE_BIN="${ACCELERATE_BIN:-$(dirname "$PYTHON_BIN")/accelerate}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29671}"
export SFT_TRAIN_MODULE="${SFT_TRAIN_MODULE:-sft.hami_cuda_bootstrap}"
export MREC_RUNTIME_CACHE_ROOT="${MREC_RUNTIME_CACHE_ROOT:-${ROOT_DIR}/outputs/cache/runtime/structure_only_no_map}"
export CUDA_DEVICE_MEMORY_SHARED_CACHE="${CUDA_DEVICE_MEMORY_SHARED_CACHE:-/tmp/lzj_mrec_no_map_hami/cudevshr.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/lzj_mrec_no_map_hami/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/lzj_mrec_no_map_hami/torchinductor}"
mkdir -p "$(dirname "$CUDA_DEVICE_MEMORY_SHARED_CACHE")" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

printf '[structure-only-no-map] HAMI_SHARED_CACHE=%s TRITON_CACHE_DIR=%s TORCHINDUCTOR_CACHE_DIR=%s MAIN_PROCESS_PORT=%s SFT_TRAIN_MODULE=%s\n' \
  "$CUDA_DEVICE_MEMORY_SHARED_CACHE" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$MAIN_PROCESS_PORT" "$SFT_TRAIN_MODULE"

exec bash scripts/sentence_trace_method/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh
