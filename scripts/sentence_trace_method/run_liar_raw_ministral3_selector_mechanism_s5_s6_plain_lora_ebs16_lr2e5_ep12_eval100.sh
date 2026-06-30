#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

SCRIPT_DIR="${ROOT_DIR}/scripts/sentence_trace_method"

export PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
export MODE="${SELECTOR_MECH_MODE:-${MODE:-full}}"
export EVAL_SPLITS="${SELECTOR_MECH_EVAL_SPLITS:-${EVAL_SPLITS:-val,test}}"
export CHECKPOINTS="${CHECKPOINTS:-best}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export FORCE_MREC_BUILD="${SELECTOR_MECH_FORCE_MREC_BUILD:-${FORCE_MREC_BUILD:-${FORCE_STAGE:-false}}}"
export FORCE_BUILD="${FORCE_BUILD:-${SELECTOR_MECH_FORCE_BUILD:-auto}}"
export FORCE_LORA_CONFIG="${FORCE_LORA_CONFIG:-true}"
export RUN_TAU_EVAL="${SELECTOR_MECH_RUN_TAU_EVAL:-${RUN_TAU_EVAL:-auto}}"
export LORA_SUFFIX="${LORA_SUFFIX:-_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"

SELECTORS="${SELECTOR_MECH_CASES:-selector_mech_s5_map_quality_greedy selector_mech_s6_learned_marginal_proxy_trace_shuffle}"

config_for_selector() {
  case "$1" in
    selector_mech_s5_map_quality_greedy)
      printf '%s\n' "configs/experiment/mrec_v0.2/selector_mech_s5_map_quality_greedy.yaml"
      ;;
    selector_mech_s6_learned_marginal_proxy_trace_shuffle)
      printf '%s\n' "configs/experiment/mrec_v0.2/selector_mech_s6_learned_marginal_proxy_trace_shuffle.yaml"
      ;;
    *)
      printf 'Unsupported SELECTOR_MECH case: %s\n' "$1" >&2
      exit 2
      ;;
  esac
}

printf '[liar-raw-ministral3-selector-mechanism-s5-s6] MODE=%s OUTPUT_ROOT=%s EVAL_SPLITS=%s RUN_TAU_EVAL=%s FORCE_MREC_BUILD=%s FORCE_BUILD=%s SELECTORS=%s LORA_SUFFIX=%s NCCL_CUMEM_HOST_ENABLE=%s OMP_NUM_THREADS=%s\n' \
  "$MODE" "$OUTPUT_ROOT" "$EVAL_SPLITS" "$RUN_TAU_EVAL" "$FORCE_MREC_BUILD" "$FORCE_BUILD" "$SELECTORS" "$LORA_SUFFIX" "$NCCL_CUMEM_HOST_ENABLE" "$OMP_NUM_THREADS"

for selector in ${SELECTORS}; do
  export SELECTOR_NAME="$selector"
  export MREC_POLICY_CONFIG="$(config_for_selector "$selector")"

  printf '\n[liar-raw-ministral3-selector-mechanism-s5-s6] SELECTOR_NAME=%s MREC_POLICY_CONFIG=%s MODE=%s EVAL_SPLITS=%s RUN_TAU_EVAL=%s FORCE_MREC_BUILD=%s FORCE_BUILD=%s LORA_SUFFIX=%s\n' \
    "$SELECTOR_NAME" "$MREC_POLICY_CONFIG" "$MODE" "$EVAL_SPLITS" "$RUN_TAU_EVAL" "$FORCE_MREC_BUILD" "$FORCE_BUILD" "$LORA_SUFFIX"

  bash "${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh"
done
