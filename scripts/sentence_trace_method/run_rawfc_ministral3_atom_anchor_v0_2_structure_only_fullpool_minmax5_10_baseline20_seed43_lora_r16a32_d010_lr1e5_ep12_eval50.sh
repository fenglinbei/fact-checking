#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

POLICY_CONFIG="configs/experiment/mrec_v0.2/rawfc_learned_marginal_structure_only_fullpool_minmax5_10_baseline20_seed43.yaml"
CANONICAL_WRAPPER="scripts/sentence_trace_method/run_rawfc_ministral3_atom_anchor_v0_2_structure_only_fullpool_minmax5_10_baseline20_lora_r16a32_d010_lr1e5_ep12_eval50.sh"

# These generic variables are consumed with ${VAR:-default} deep in the shared
# wrapper stack. Refuse them here so a caller cannot silently redirect this
# seed-only replicate into an older case or output tree.
polluting_vars=(
  CASE_NAME CASE_ROOT LORA_ROOT TRAIN_CASE_ROOT RUN_DIR CONFIG
  CONFIG_PATH BASE_CASE_NAME CASE_SUFFIX LORA_SUFFIX OUTPUT_ROOT
)
for var_name in "${polluting_vars[@]}"; do
  if [[ -v "$var_name" ]]; then
    printf '[rawfc-structure-only-seed43] refusing inherited %s; the seed43 path is config-derived\n' \
      "$var_name" >&2
    exit 2
  fi
done
if [[ -v MREC_POLICY_CONFIG && "$MREC_POLICY_CONFIG" != "$POLICY_CONFIG" ]]; then
  printf '[rawfc-structure-only-seed43] refusing MREC_POLICY_CONFIG=%s; expected %s\n' \
    "$MREC_POLICY_CONFIG" "$POLICY_CONFIG" >&2
  exit 2
fi

export MREC_POLICY_CONFIG="$POLICY_CONFIG"
export MAIN_PROCESS_PORT="${RAWFC_SEED43_MAIN_PROCESS_PORT:-29683}"
export MREC_RUNTIME_CACHE_ROOT="${RAWFC_SEED43_RUNTIME_CACHE_ROOT:-${ROOT_DIR}/outputs/cache/runtime/rawfc_structure_only_baseline20_seed43}"
export XDG_CACHE_HOME="${MREC_RUNTIME_CACHE_ROOT}/xdg"
export VLLM_CACHE_ROOT="${MREC_RUNTIME_CACHE_ROOT}/vllm"
export CUDA_DEVICE_MEMORY_SHARED_CACHE="${RAWFC_SEED43_HAMI_CACHE:-/tmp/lzj_rawfc_structure_only_seed43_hami/cudevshr.cache}"
export TRITON_CACHE_DIR="${RAWFC_SEED43_TRITON_CACHE:-/tmp/lzj_rawfc_structure_only_seed43_hami/triton}"
export TORCHINDUCTOR_CACHE_DIR="${RAWFC_SEED43_TORCHINDUCTOR_CACHE:-/tmp/lzj_rawfc_structure_only_seed43_hami/torchinductor}"
mkdir -p \
  "$XDG_CACHE_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$(dirname "$CUDA_DEVICE_MEMORY_SHARED_CACHE")" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR"

printf '[rawfc-structure-only-seed43] config=%s port=%s runtime=%s hami_cache=%s\n' \
  "$MREC_POLICY_CONFIG" "$MAIN_PROCESS_PORT" "$MREC_RUNTIME_CACHE_ROOT" \
  "$CUDA_DEVICE_MEMORY_SHARED_CACHE"

exec bash "$CANONICAL_WRAPPER"
