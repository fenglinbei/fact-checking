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
PREPARE_SELECTOR_SOURCES="${PREPARE_SELECTOR_SOURCES:-true}"

export DATASETS="rawfc"
export MODELS="llama31_8b"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz1_ga4.json}"
export SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-4}"
export SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-8}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-$SFT_EVAL_STEPS}"
export SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-8}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-rawfc-selector-lora-lr}"

lr_slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+}"
  printf '%s\n' "$value"
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

selector_env() {
  local key="$1"
  case "$key" in
    old_adaptive5_10)
      SELECTOR_NAME="sentence_rule_step_adaptive5_10"
      SELECTOR_GRAPH_VERSION="sentence_evidence_chain_graph"
      SELECTOR_ADAPTIVE_POLICY="sentence_rule_step"
      CASE_SUFFIX="__old_adaptive5_10"
      MIN_TOP_K=""
      MAX_TOP_K=""
      ;;
    v0_7_bm_adaptive3_10)
      SELECTOR_NAME="v0_7_budgeted_marginal_chain_adaptive3_10"
      SELECTOR_GRAPH_VERSION="evidence_chain_graph_v0_7"
      SELECTOR_ADAPTIVE_POLICY="budgeted_marginal_v0_7"
      CASE_SUFFIX="__v0_7_bm_adaptive3_10"
      MIN_TOP_K="3"
      MAX_TOP_K="10"
      ;;
    v0_7_bm_adaptive5_10)
      SELECTOR_NAME="v0_7_budgeted_marginal_chain_adaptive5_10"
      SELECTOR_GRAPH_VERSION="evidence_chain_graph_v0_7"
      SELECTOR_ADAPTIVE_POLICY="budgeted_marginal_v0_7"
      CASE_SUFFIX="__v0_7_bm_adaptive5_10"
      MIN_TOP_K="5"
      MAX_TOP_K="10"
      ;;
    v0_7_bm_adaptive5_12)
      SELECTOR_NAME="v0_7_budgeted_marginal_chain_adaptive5_12"
      SELECTOR_GRAPH_VERSION="evidence_chain_graph_v0_7"
      SELECTOR_ADAPTIVE_POLICY="budgeted_marginal_v0_7"
      CASE_SUFFIX="__v0_7_bm_adaptive5_12"
      MIN_TOP_K="5"
      MAX_TOP_K="12"
      ;;
    *) printf 'Unsupported selector key=%s\n' "$key" >&2; exit 2 ;;
  esac
}

prepare_selector_sources() {
  if [[ "$PREPARE_SELECTOR_SOURCES" != "true" || -z "${MIN_TOP_K:-}" ]]; then
    return 0
  fi
  DATASETS=rawfc \
    SPLITS=train,val,test \
    SELECTOR_NAME="$SELECTOR_NAME" \
    SELECTOR_GRAPH_VERSION="$SELECTOR_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$SELECTOR_ADAPTIVE_POLICY" \
    MIN_TOP_K="$MIN_TOP_K" \
    MAX_TOP_K="$MAX_TOP_K" \
    bash "${SCRIPT_DIR}/prepare_v0_7_sources.sh"
}

run_tau_eval() {
  local case_suffix="$1"
  local lora_suffix="$2"
  local case_root="${OUTPUT_ROOT}/rawfc__llama31_8b${case_suffix}${lora_suffix}"
  PYTHON_BIN="$PYTHON_BIN" \
    CASE_ROOT="$case_root" \
    SPLITS="$TAU_SPLITS" \
    CHECKPOINTS=best \
    TAUS="$TAUS" \
    LOGIT_ADJUST_MODE=on \
    bash "${SCRIPT_DIR}/run_lora_label_token_logit_adjust_eval_only.sh"
}

IFS=',' read -r -a selector_keys <<< "${SELECTOR_KEYS:-old_adaptive5_10,v0_7_bm_adaptive3_10,v0_7_bm_adaptive5_10,v0_7_bm_adaptive5_12}"
IFS=',' read -r -a lr_values <<< "${LR_VALUES:-1e-5,5e-6}"

for raw_selector_key in "${selector_keys[@]}"; do
  selector_key="${raw_selector_key// /}"
  [[ -z "$selector_key" ]] && continue
  selector_env "$selector_key"
  prepare_selector_sources
  for raw_lr in "${lr_values[@]}"; do
    lr="${raw_lr// /}"
    [[ -z "$lr" ]] && continue
    lora_suffix="_lora_ebs16_lr$(lr_slug "$lr")_ep8_eval100_pat8_rawfc"
    printf '\n[rawfc-selector-lr] selector_key=%s SELECTOR_NAME=%s CASE_SUFFIX=%s LORA_SUFFIX=%s SFT_LEARNING_RATE=%s\n' \
      "$selector_key" "$SELECTOR_NAME" "$CASE_SUFFIX" "$lora_suffix" "$lr"
    PYTHON_BIN="$PYTHON_BIN" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      MODE="$MODE" \
      EVAL_SPLITS="$MATRIX_EVAL_SPLITS" \
      CHECKPOINTS=best \
      SELECTOR_NAME="$SELECTOR_NAME" \
      SELECTOR_GRAPH_VERSION="$SELECTOR_GRAPH_VERSION" \
      SELECTOR_ADAPTIVE_POLICY="$SELECTOR_ADAPTIVE_POLICY" \
      EXPECTED_SELECTOR_NAME="$SELECTOR_NAME" \
      CASE_SUFFIX="$CASE_SUFFIX" \
      LORA_SUFFIX="$lora_suffix" \
      SFT_LEARNING_RATE="$lr" \
      bash "${SCRIPT_DIR}/run_lora_matrix.sh"
    if should_run_tau_eval; then
      run_tau_eval "$CASE_SUFFIX" "$lora_suffix"
    fi
  done
done
