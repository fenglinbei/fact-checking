#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

export MREC_POLICY_CONFIG="${MREC_POLICY_CONFIG:-configs/experiment/mrec_v0.2/rawfc_learned_marginal_structure_only_fullpool_minmax5_10_baseline20.yaml}"
export PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
export ACCELERATE_BIN="${ACCELERATE_BIN:-$(dirname "$PYTHON_BIN")/accelerate}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29651}"
export SFT_TRAIN_MODULE="${SFT_TRAIN_MODULE:-sft.hami_cuda_bootstrap}"
export MREC_RUNTIME_CACHE_ROOT="${MREC_RUNTIME_CACHE_ROOT:-${ROOT_DIR}/outputs/cache/runtime/rawfc_structure_only_baseline20}"
export CUDA_DEVICE_MEMORY_SHARED_CACHE="${CUDA_DEVICE_MEMORY_SHARED_CACHE:-/tmp/lzj_rawfc_structure_only_hami/cudevshr.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/lzj_rawfc_structure_only_hami/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/lzj_rawfc_structure_only_hami/torchinductor}"
mkdir -p "$(dirname "$CUDA_DEVICE_MEMORY_SHARED_CACHE")" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

MODE="${MODE:-full}"
case "$MODE" in
  build|full)
    env \
      PYTHON_BIN="$PYTHON_BIN" \
      DRY_RUN="${DRY_RUN:-false}" \
      ENSURE_WEIGHTS=true \
      bash scripts/phase5_selectors/run/run_rawfc_mrec_structure_only_weights_baseline20.sh
    ;;
  check|train|eval)
    env \
      PYTHON_BIN="$PYTHON_BIN" \
      DRY_RUN="${DRY_RUN:-false}" \
      ENSURE_WEIGHTS=false \
      bash scripts/phase5_selectors/run/run_rawfc_mrec_structure_only_weights_baseline20.sh
    ;;
  *)
    printf 'Unsupported MODE=%s. Use check, build, train, eval, or full.\n' "$MODE" >&2
    exit 2
    ;;
esac

printf '[rawfc-structure-only] MODE=%s config=%s main_port=%s train_module=%s hami_cache=%s\n' \
  "$MODE" "$MREC_POLICY_CONFIG" "$MAIN_PROCESS_PORT" "$SFT_TRAIN_MODULE" "$CUDA_DEVICE_MEMORY_SHARED_CACHE"

export MODE
exec bash scripts/sentence_trace_method/run_rawfc_ministral3_atom_anchor_v0_2_fullpool_policy_baseline20_lora_r16a32_d010_lr1e5_ep12_eval50.sh
