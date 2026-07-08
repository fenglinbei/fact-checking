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
export MODE="${SELECTOR_MECH_MODE:-full}"
export EVAL_SPLITS="${SELECTOR_MECH_EVAL_SPLITS:-val,test}"
export CHECKPOINTS="${CHECKPOINTS:-best}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export DATASETS="liar_raw"
export MODELS="ministral3_8b"
export TRACE_PROMPT_STYLE="plain"
export SELECTOR_GRAPH_VERSION="selector_mechanism_ablation_v0"
export SELECTOR_ADAPTIVE_POLICY="fixed_top5"
export EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-d4cbf7c18126}"
export ALLOW_MULTI_SENTENCE_CANDIDATES="${ALLOW_MULTI_SENTENCE_CANDIDATES:-true}"
export FORCE_STAGE="${SELECTOR_MECH_FORCE_STAGE:-true}"

export LORA_SUFFIX="${LORA_SUFFIX:-_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export FORCE_LORA_CONFIG="${FORCE_LORA_CONFIG:-true}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/deepspeed_zero2_bsz1_ga4.json}"
export SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-4}"
export SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-2e-5}"
export SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-12}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-$SFT_EVAL_STEPS}"
export SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-8}"
export REQUIRE_PROMPT_INPUT_IDS="${REQUIRE_PROMPT_INPUT_IDS:-true}"
export LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-selector-mechanism}"

SELECTORS="${SELECTOR_MECH_CASES:-selector_mech_s0_no_evidence selector_mech_s1_claim_pool_random_top5 selector_mech_s2_claim_pool_hybrid_top5 selector_mech_s3_claim_pool_hybrid_mmr_top5 selector_mech_s4_atom_union_source_score_top5 selector_mech_s4b_atom_route_only_source_score_top5}"
SOURCE_BASE_ROOT="${SOURCE_BASE_ROOT:-outputs/selectors/selector_mechanism_ablation}"
CASE_SUFFIX_EXTRA="${CASE_SUFFIX_EXTRA:-}"
PREPARE_SELECTOR_MECH_TRACES="${PREPARE_SELECTOR_MECH_TRACES:-true}"
RUN_TAU_EVAL="${SELECTOR_MECH_RUN_TAU_EVAL:-${RUN_TAU_EVAL:-auto}}"
TAU_SPLITS="${TAU_SPLITS:-$EVAL_SPLITS}"
TAUS="${TAUS:-0.75}"

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
  local case_root="${OUTPUT_ROOT}/liar_raw__ministral3_8b${CASE_SUFFIX}${LORA_SUFFIX}"
  PYTHON_BIN="$PYTHON_BIN" \
    CASE_ROOT="$case_root" \
    SPLITS="$TAU_SPLITS" \
    CHECKPOINTS=best \
    TAUS="$TAUS" \
    LOGIT_ADJUST_MODE=on \
    bash "${SCRIPT_DIR}/run_lora_label_token_logit_adjust_eval_only.sh"
}

if [[ "$PREPARE_SELECTOR_MECH_TRACES" == "true" || "$PREPARE_SELECTOR_MECH_TRACES" == "1" ]]; then
  printf '[liar-raw-ministral3-selector-mechanism-s0-s4-plain] prepare traces: %s\n' \
    "scripts/phase5_selectors/run/run_liar_raw_selector_mechanism_s0_s4.sh"
  PYTHON_BIN="$PYTHON_BIN" \
    SPLITS="train val test" \
    SELECTORS="$SELECTORS" \
    SOURCE_BASE_ROOT="$SOURCE_BASE_ROOT" \
    CHUNK_MMR_FINGERPRINT="$EXPECTED_CHUNK_MMR_FINGERPRINT" \
    bash "${ROOT_DIR}/scripts/phase5_selectors/run/run_liar_raw_selector_mechanism_s0_s4.sh"
fi

printf '[liar-raw-ministral3-selector-mechanism-s0-s4-plain] DATASETS=%s MODELS=%s MODE=%s EVAL_SPLITS=%s RUN_TAU_EVAL=%s TRACE_PROMPT_STYLE=%s FORCE_STAGE=%s FORCE_LORA_CONFIG=%s SELECTORS=%s LORA_SUFFIX=%s EXPECTED_CHUNK_MMR_FINGERPRINT=%s NCCL_CUMEM_HOST_ENABLE=%s OMP_NUM_THREADS=%s\n' \
  "$DATASETS" "$MODELS" "$MODE" "$EVAL_SPLITS" "$RUN_TAU_EVAL" "$TRACE_PROMPT_STYLE" "$FORCE_STAGE" "$FORCE_LORA_CONFIG" "$SELECTORS" "$LORA_SUFFIX" "$EXPECTED_CHUNK_MMR_FINGERPRINT" "$NCCL_CUMEM_HOST_ENABLE" "$OMP_NUM_THREADS"

for selector in ${SELECTORS}; do
  export SELECTOR_NAME="$selector"
  export EXPECTED_SELECTOR_NAME="$selector"
  export SOURCE_ROOT="${SOURCE_BASE_ROOT}/liar_raw_${selector}"
  export CASE_SUFFIX="__${selector}_plain${CASE_SUFFIX_EXTRA}"
  if [[ "$selector" == "selector_mech_s0_no_evidence" ]]; then
    export ALLOW_EMPTY_EVIDENCE="true"
    export ALLOW_EMPTY_CANDIDATE_POOL="true"
  else
    export ALLOW_EMPTY_EVIDENCE="false"
    export ALLOW_EMPTY_CANDIDATE_POOL="false"
  fi

  printf '\n[liar-raw-ministral3-selector-mechanism-plain] SELECTOR_NAME=%s SOURCE_ROOT=%s CASE_SUFFIX=%s TRACE_PROMPT_STYLE=%s EVAL_SPLITS=%s RUN_TAU_EVAL=%s FORCE_STAGE=%s ALLOW_EMPTY_EVIDENCE=%s ALLOW_EMPTY_CANDIDATE_POOL=%s SFT_LEARNING_RATE=%s SFT_NUM_TRAIN_EPOCHS=%s SFT_EVAL_STEPS=%s REQUIRE_PROMPT_INPUT_IDS=%s LIAR_CLASS_WEIGHTS=%s\n' \
    "$SELECTOR_NAME" "$SOURCE_ROOT" "$CASE_SUFFIX" "$TRACE_PROMPT_STYLE" "$EVAL_SPLITS" "$RUN_TAU_EVAL" "$FORCE_STAGE" "$ALLOW_EMPTY_EVIDENCE" "$ALLOW_EMPTY_CANDIDATE_POOL" "$SFT_LEARNING_RATE" "$SFT_NUM_TRAIN_EPOCHS" "$SFT_EVAL_STEPS" "$REQUIRE_PROMPT_INPUT_IDS" "$LIAR_CLASS_WEIGHTS"

  bash "${SCRIPT_DIR}/run_lora_matrix.sh"

  if [[ "${DRY_RUN:-false}" != "true" ]] && should_run_tau_eval; then
    run_tau_eval
  fi
done
