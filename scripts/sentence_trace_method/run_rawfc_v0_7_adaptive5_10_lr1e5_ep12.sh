#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCRIPT_DIR="${ROOT_DIR}/scripts/sentence_trace_method"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"
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
export DATASETS="rawfc"
export MODELS="llama31_8b"
export SELECTOR_NAME="${SELECTOR_NAME:-v0_7_budgeted_marginal_chain_adaptive5_10}"
export SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-evidence_chain_graph_v0_7}"
export SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-budgeted_marginal_v0_7}"
export EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-$SELECTOR_NAME}"
export CASE_SUFFIX="${CASE_SUFFIX:-__v0_7_bm_adaptive5_10}"
export LORA_SUFFIX="${LORA_SUFFIX:-_lora_ebs16_lr1em5_ep12_eval100_pat8_rawfc}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz1_ga4.json}"
export SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-4}"
export SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-5}"
export SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-12}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-$SFT_EVAL_STEPS}"
export SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-8}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-rawfc-selector-lora-lr}"

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

bash "${SCRIPT_DIR}/run_lora_matrix.sh"

if [[ "${DRY_RUN:-false}" != "true" ]] && should_run_tau_eval; then
  PYTHON_BIN="$PYTHON_BIN" \
    CASE_ROOT="${OUTPUT_ROOT}/rawfc__llama31_8b${CASE_SUFFIX}${LORA_SUFFIX}" \
    SPLITS="$TAU_SPLITS" \
    CHECKPOINTS=best \
    TAUS="$TAUS" \
    LOGIT_ADJUST_MODE=on \
    bash "${SCRIPT_DIR}/run_lora_label_token_logit_adjust_eval_only.sh"
fi
