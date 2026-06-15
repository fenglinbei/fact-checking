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
TAU_SPLITS="${TAU_SPLITS:-val}"
TAUS="${TAUS:-0,0.5,0.75}"
RUN_TAU_EVAL="${RUN_TAU_EVAL:-auto}"

export PYTHON_BIN
export OUTPUT_ROOT
export MODE
export EVAL_SPLITS
export CHECKPOINTS
export DATASET="liar_raw"
export MODEL="ministral3_8b"
export SELECTOR_NAME="${SELECTOR_NAME:-v0_7_budgeted_marginal_chain_adaptive5_10}"
export SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-evidence_chain_graph_v0_7}"
export SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-budgeted_marginal_v0_7}"
export EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-$SELECTOR_NAME}"
export CASE_SUFFIX="${CASE_SUFFIX:-__v0_7_bm_adaptive5_10_fullft_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero3_bsz1_ga4_lowpeak.json}"
export SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-4}"
export SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-2e-5}"
export SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-12}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-$SFT_EVAL_STEPS}"
export SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-8}"
export SFT_WEIGHT_DECAY="${SFT_WEIGHT_DECAY:-0.01}"
export SFT_WARMUP_RATIO="${SFT_WARMUP_RATIO:-0.03}"
export SFT_MAX_GRAD_NORM="${SFT_MAX_GRAD_NORM:-1.0}"
export REQUIRE_PROMPT_INPUT_IDS="${REQUIRE_PROMPT_INPUT_IDS:-true}"
export LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-liar-ministral-fullft}"

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

if [[ "${PREPARE_V0_7_SOURCES:-true}" == "true" ]]; then
  DATASETS=liar_raw \
    SPLITS=train,val,test \
    SELECTOR_NAME="$SELECTOR_NAME" \
    SELECTOR_GRAPH_VERSION="$SELECTOR_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$SELECTOR_ADAPTIVE_POLICY" \
    MIN_TOP_K=5 \
    MAX_TOP_K=10 \
    bash "${SCRIPT_DIR}/prepare_v0_7_sources.sh"
fi

printf '[fullft-aligned] DATASET=%s MODEL=%s\n' "$DATASET" "$MODEL"
printf '[fullft-aligned] SELECTOR_NAME=%s\n' "$SELECTOR_NAME"
printf '[fullft-aligned] CASE_SUFFIX=%s\n' "$CASE_SUFFIX"
printf '[fullft-aligned] DEEPSPEED_CONFIG=%s\n' "$DEEPSPEED_CONFIG"
printf '[fullft-aligned] SFT_GRADIENT_ACCUMULATION_STEPS=%s\n' "$SFT_GRADIENT_ACCUMULATION_STEPS"
printf '[fullft-aligned] SFT_LEARNING_RATE=%s\n' "$SFT_LEARNING_RATE"
printf '[fullft-aligned] SFT_NUM_TRAIN_EPOCHS=%s\n' "$SFT_NUM_TRAIN_EPOCHS"
printf '[fullft-aligned] SFT_EVAL_STEPS=%s\n' "$SFT_EVAL_STEPS"
printf '[fullft-aligned] SFT_SAVE_STEPS=%s\n' "$SFT_SAVE_STEPS"
printf '[fullft-aligned] SFT_EARLY_STOPPING_PATIENCE=%s\n' "$SFT_EARLY_STOPPING_PATIENCE"
printf '[fullft-aligned] REQUIRE_PROMPT_INPUT_IDS=%s\n' "$REQUIRE_PROMPT_INPUT_IDS"
printf '[fullft-aligned] LIAR_CLASS_WEIGHTS=%s\n' "$LIAR_CLASS_WEIGHTS"

bash "${SCRIPT_DIR}/run_one.sh"

if [[ "${DRY_RUN:-false}" != "true" ]] && should_run_tau_eval; then
  PYTHON_BIN="$PYTHON_BIN" \
    CASE_ROOT="${OUTPUT_ROOT}/liar_raw__ministral3_8b${CASE_SUFFIX}" \
    SPLITS="$TAU_SPLITS" \
    CHECKPOINTS=best \
    TAUS="$TAUS" \
    LOGIT_ADJUST_MODE=on \
    bash "${SCRIPT_DIR}/run_lora_label_token_logit_adjust_eval_only.sh"
fi
