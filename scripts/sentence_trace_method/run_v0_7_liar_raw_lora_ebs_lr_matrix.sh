#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCRIPT_DIR="${ROOT_DIR}/scripts/sentence_trace_method"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
MODE="${MODE:-full}"
MATRIX_EVAL_SPLITS="${MATRIX_EVAL_SPLITS:-val}"
TAU_SPLITS="${TAU_SPLITS:-val}"
TAUS="${TAUS:-0,0.5,0.75}"
RUN_TAU_EVAL="${RUN_TAU_EVAL:-auto}"

export DATASETS="liar_raw"
export MODELS="llama31_8b"
export SELECTOR_NAME="${SELECTOR_NAME:-v0_7_budgeted_marginal_chain_adaptive3_10}"
export SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-evidence_chain_graph_v0_7}"
export SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-budgeted_marginal_v0_7}"
export EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-$SELECTOR_NAME}"
export CASE_SUFFIX="${CASE_SUFFIX:-__v0_7_bm_adaptive3_10}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-8}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-$SFT_EVAL_STEPS}"
export SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-8}"
export LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-v0-7-liar-lora-ebs-lr}"

lr_slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+}"
  printf '%s\n' "$value"
}

ga_for_ebs() {
  case "$1" in
    16) printf '%s\n' "4" ;;
    32) printf '%s\n' "8" ;;
    *) printf 'Unsupported EBS=%s\n' "$1" >&2; exit 2 ;;
  esac
}

deepspeed_for_ebs() {
  case "$1" in
    16) printf '%s\n' "configs/deepspeed_zero2_bsz1_ga4.json" ;;
    32) printf '%s\n' "configs/deepspeed_zero2_bsz1_ga8.json" ;;
    *) printf 'Unsupported EBS=%s\n' "$1" >&2; exit 2 ;;
  esac
}

should_run_tau_eval() {
  case "$RUN_TAU_EVAL" in
    true) return 0 ;;
    false) return 1 ;;
    auto)
      case "$MODE" in
        full|eval) return 0 ;;
        *) return 1 ;;
      esac
      ;;
    *) printf 'Unsupported RUN_TAU_EVAL=%s. Use true, false, or auto.\n' "$RUN_TAU_EVAL" >&2; exit 2 ;;
  esac
}

run_tau_eval() {
  local lora_suffix="$1"
  local case_root="${OUTPUT_ROOT}/liar_raw__llama31_8b${CASE_SUFFIX}${lora_suffix}"
  PYTHON_BIN="$PYTHON_BIN" \
    CASE_ROOT="$case_root" \
    SPLITS="$TAU_SPLITS" \
    CHECKPOINTS=best \
    TAUS="$TAUS" \
    LOGIT_ADJUST_MODE=on \
    bash "${SCRIPT_DIR}/run_lora_label_token_logit_adjust_eval_only.sh"
}

if [[ "${PREPARE_V0_7_SOURCES:-true}" == "true" ]]; then
  DATASETS=liar_raw \
    SPLITS=train,val,test \
    SELECTOR_NAME="$SELECTOR_NAME" \
    SELECTOR_GRAPH_VERSION="$SELECTOR_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$SELECTOR_ADAPTIVE_POLICY" \
    MIN_TOP_K=3 \
    MAX_TOP_K=10 \
    bash "${SCRIPT_DIR}/prepare_v0_7_sources.sh"
fi

IFS=',' read -r -a ebs_values <<< "${EBS_VALUES:-16,32}"
IFS=',' read -r -a lr_values <<< "${LR_VALUES:-1e-5,2e-5}"

for raw_ebs in "${ebs_values[@]}"; do
  ebs="${raw_ebs// /}"
  [[ -z "$ebs" ]] && continue
  ga="$(ga_for_ebs "$ebs")"
  ds_config="$(deepspeed_for_ebs "$ebs")"
  for raw_lr in "${lr_values[@]}"; do
    lr="${raw_lr// /}"
    [[ -z "$lr" ]] && continue
    lora_suffix="_lora_ebs${ebs}_lr$(lr_slug "$lr")_ep8_eval100_pat8_liarw"
    printf '\n[liar-v0.7-ebs-lr] CASE_SUFFIX=%s LORA_SUFFIX=%s EBS=%s SFT_GRADIENT_ACCUMULATION_STEPS=%s SFT_LEARNING_RATE=%s\n' \
      "$CASE_SUFFIX" "$lora_suffix" "$ebs" "$ga" "$lr"
    PYTHON_BIN="$PYTHON_BIN" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      MODE="$MODE" \
      EVAL_SPLITS="$MATRIX_EVAL_SPLITS" \
      CHECKPOINTS=best \
      LORA_SUFFIX="$lora_suffix" \
      DEEPSPEED_CONFIG="$ds_config" \
      SFT_GRADIENT_ACCUMULATION_STEPS="$ga" \
      SFT_LEARNING_RATE="$lr" \
      bash "${SCRIPT_DIR}/run_lora_matrix.sh"
    if should_run_tau_eval; then
      run_tau_eval "$lora_suffix"
    fi
  done
done
