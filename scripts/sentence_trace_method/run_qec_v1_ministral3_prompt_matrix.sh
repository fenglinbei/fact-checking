#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCRIPT_DIR="${ROOT_DIR}/scripts/sentence_trace_method"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
MODE="${MODE:-full}"
EVAL_SPLITS="${EVAL_SPLITS:-val}"
CHECKPOINTS="${CHECKPOINTS:-best}"
PROMPT_STYLES="${PROMPT_STYLES:-qec_min,qec_map}"
RUN_LIAR_RAW="${RUN_LIAR_RAW:-true}"
RUN_RAWFC="${RUN_RAWFC:-true}"
RUN_LIAR_TAU_EVAL="${RUN_LIAR_TAU_EVAL:-auto}"
TAU_SPLITS="${TAU_SPLITS:-$EVAL_SPLITS}"
LIAR_TAUS="${LIAR_TAUS:-0.75}"

SELECTOR_NAME="${SELECTOR_NAME:-v0_7_budgeted_marginal_chain_adaptive5_10}"
SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-evidence_chain_graph_v0_7}"
SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-budgeted_marginal_v0_7}"
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-$SELECTOR_NAME}"

style_suffix() {
  case "$1" in
    qec_min) printf '%s\n' "__qec_min" ;;
    qec_map) printf '%s\n' "__qec_map" ;;
    *) printf 'Unsupported PROMPT_STYLE=%s. Use qec_min or qec_map.\n' "$1" >&2; exit 2 ;;
  esac
}

truthy() {
  case "$1" in
    true|1|yes|y) return 0 ;;
    false|0|no|n) return 1 ;;
    *) printf 'Unsupported boolean value: %s\n' "$1" >&2; exit 2 ;;
  esac
}

should_run_liar_tau_eval() {
  case "$RUN_LIAR_TAU_EVAL" in
    true) return 0 ;;
    false) return 1 ;;
    auto)
      case "$MODE" in
        full|eval) return 0 ;;
        *) return 1 ;;
      esac
      ;;
    *) printf 'Unsupported RUN_LIAR_TAU_EVAL=%s. Use true, false, or auto.\n' "$RUN_LIAR_TAU_EVAL" >&2; exit 2 ;;
  esac
}

run_liar_tau_eval() {
  local case_suffix="$1"
  local lora_suffix="$2"
  local case_root="${OUTPUT_ROOT}/liar_raw__ministral3_8b${case_suffix}${lora_suffix}"
  PYTHON_BIN="$PYTHON_BIN" \
    CASE_ROOT="$case_root" \
    SPLITS="$TAU_SPLITS" \
    CHECKPOINTS=best \
    TAUS="$LIAR_TAUS" \
    LOGIT_ADJUST_MODE=on \
    bash "${SCRIPT_DIR}/run_lora_label_token_logit_adjust_eval_only.sh"
}

prepare_v0_7_sources() {
  local dataset="$1"
  if [[ "${PREPARE_V0_7_SOURCES:-true}" != "true" ]]; then
    return 0
  fi
  DATASETS="$dataset" \
    SPLITS=train,val,test \
    SELECTOR_NAME="$SELECTOR_NAME" \
    SELECTOR_GRAPH_VERSION="$SELECTOR_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$SELECTOR_ADAPTIVE_POLICY" \
    MIN_TOP_K=5 \
    MAX_TOP_K=10 \
    bash "${SCRIPT_DIR}/prepare_v0_7_sources.sh"
}

run_liar_raw_qec() {
  local style="$1"
  local suffix="$2"
  local case_suffix="__v0_7_bm_adaptive5_10${suffix}"
  local lora_suffix="_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
  printf '\n[qec-v1-liar-raw] DATASETS=liar_raw MODELS=ministral3_8b TRACE_PROMPT_STYLE=%s CASE_SUFFIX=%s LORA_SUFFIX=%s EBS=16 DEEPSPEED_CONFIG=configs/deepspeed_zero2_bsz1_ga4.json SFT_GRADIENT_ACCUMULATION_STEPS=4 SFT_LEARNING_RATE=2e-5 SFT_NUM_TRAIN_EPOCHS=12 SFT_EVAL_STEPS=100 SFT_SAVE_STEPS=100 SFT_EARLY_STOPPING_PATIENCE=8 REQUIRE_PROMPT_INPUT_IDS=true TAU_POLICY=label_token_logit_adjust_tau0p75\n' \
    "$style" "$case_suffix" "$lora_suffix"
  PYTHON_BIN="$PYTHON_BIN" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    MODE="$MODE" \
    EVAL_SPLITS="$EVAL_SPLITS" \
    CHECKPOINTS="$CHECKPOINTS" \
    DATASETS=liar_raw \
    MODELS=ministral3_8b \
    SELECTOR_NAME="$SELECTOR_NAME" \
    SELECTOR_GRAPH_VERSION="$SELECTOR_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$SELECTOR_ADAPTIVE_POLICY" \
    EXPECTED_SELECTOR_NAME="$EXPECTED_SELECTOR_NAME" \
    CASE_SUFFIX="$case_suffix" \
    LORA_SUFFIX="$lora_suffix" \
    LORA_R=16 \
    LORA_ALPHA=32 \
    LORA_DROPOUT=0.05 \
    DEEPSPEED_CONFIG=configs/deepspeed_zero2_bsz1_ga4.json \
    SFT_GRADIENT_ACCUMULATION_STEPS=4 \
    SFT_LEARNING_RATE=2e-5 \
    SFT_NUM_TRAIN_EPOCHS=12 \
    SFT_EVAL_STEPS=100 \
    SFT_SAVE_STEPS=100 \
    SFT_EARLY_STOPPING_PATIENCE=8 \
    REQUIRE_PROMPT_INPUT_IDS=true \
    TRACE_PROMPT_STYLE="$style" \
    LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8}" \
    SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-qec-v1}" \
    bash "${SCRIPT_DIR}/run_lora_matrix.sh"
  if [[ "${DRY_RUN:-false}" != "true" ]] && should_run_liar_tau_eval; then
    run_liar_tau_eval "$case_suffix" "$lora_suffix"
  fi
}

run_rawfc_qec() {
  local style="$1"
  local suffix="$2"
  local case_suffix="__v0_7_bm_adaptive5_10${suffix}"
  local lora_suffix="_lora_ebs16_lr1em5_ep10_eval50_pat8_rawfc"
  printf '\n[qec-v1-rawfc] DATASETS=rawfc MODELS=ministral3_8b TRACE_PROMPT_STYLE=%s CASE_SUFFIX=%s LORA_SUFFIX=%s EBS=16 DEEPSPEED_CONFIG=configs/deepspeed_zero2_bsz1_ga4.json SFT_GRADIENT_ACCUMULATION_STEPS=4 SFT_LEARNING_RATE=1e-5 SFT_NUM_TRAIN_EPOCHS=10 SFT_EVAL_STEPS=50 SFT_SAVE_STEPS=50 SFT_EARLY_STOPPING_PATIENCE=8 REQUIRE_PROMPT_INPUT_IDS=true TAU_POLICY=label_token_main_only\n' \
    "$style" "$case_suffix" "$lora_suffix"
  PYTHON_BIN="$PYTHON_BIN" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    MODE="$MODE" \
    EVAL_SPLITS="$EVAL_SPLITS" \
    CHECKPOINTS="$CHECKPOINTS" \
    DATASETS=rawfc \
    MODELS=ministral3_8b \
    SELECTOR_NAME="$SELECTOR_NAME" \
    SELECTOR_GRAPH_VERSION="$SELECTOR_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$SELECTOR_ADAPTIVE_POLICY" \
    EXPECTED_SELECTOR_NAME="$EXPECTED_SELECTOR_NAME" \
    CASE_SUFFIX="$case_suffix" \
    LORA_SUFFIX="$lora_suffix" \
    LORA_R=16 \
    LORA_ALPHA=32 \
    LORA_DROPOUT=0.05 \
    DEEPSPEED_CONFIG=configs/deepspeed_zero2_bsz1_ga4.json \
    SFT_GRADIENT_ACCUMULATION_STEPS=4 \
    SFT_LEARNING_RATE=1e-5 \
    SFT_NUM_TRAIN_EPOCHS=10 \
    SFT_EVAL_STEPS=50 \
    SFT_SAVE_STEPS=50 \
    SFT_EARLY_STOPPING_PATIENCE=8 \
    REQUIRE_PROMPT_INPUT_IDS=true \
    TRACE_PROMPT_STYLE="$style" \
    SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-qec-v1}" \
    bash "${SCRIPT_DIR}/run_lora_matrix.sh"
}

if truthy "$RUN_LIAR_RAW"; then
  prepare_v0_7_sources liar_raw
fi
if truthy "$RUN_RAWFC"; then
  prepare_v0_7_sources rawfc
fi

IFS=',' read -r -a style_array <<< "$PROMPT_STYLES"
for raw_style in "${style_array[@]}"; do
  style="${raw_style// /}"
  [[ -z "$style" ]] && continue
  suffix="$(style_suffix "$style")"
  if truthy "$RUN_LIAR_RAW"; then
    run_liar_raw_qec "$style" "$suffix"
  fi
  if truthy "$RUN_RAWFC"; then
    run_rawfc_qec "$style" "$suffix"
  fi
done
