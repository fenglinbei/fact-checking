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
export MODE="${MODE:-full}"
export EVAL_SPLITS="${EVAL_SPLITS:-val}"
export CHECKPOINTS="${CHECKPOINTS:-best}"
export DATASETS="liar_raw"
export MODELS="ministral3_8b"
export SELECTOR_NAME="${SELECTOR_NAME:-v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10}"
export SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-evidence_chain_graph_v0_7}"
export SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-budgeted_marginal_v0_7}"
export EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-$SELECTOR_NAME}"
export TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-qec_min}"
export CASE_SUFFIX="${CASE_SUFFIX:-__v0_7_atom_facts_abc_bm_adaptive5_10__${TRACE_PROMPT_STYLE}}"
export ALLOW_MULTI_SENTENCE_CANDIDATES="${ALLOW_MULTI_SENTENCE_CANDIDATES:-true}"

ATOM_FACTS_ABC_SOURCE_ROOT="${ATOM_FACTS_ABC_SOURCE_ROOT:-outputs/selectors/evidence_chain_graph/liar_raw_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10}"
export SOURCE_ROOT="${SOURCE_ROOT:-$ATOM_FACTS_ABC_SOURCE_ROOT}"
export EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-d4cbf7c18126}"

export LORA_SUFFIX="${LORA_SUFFIX:-_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz1_ga4.json}"
export SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-4}"
export SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-2e-5}"
export SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-12}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-$SFT_EVAL_STEPS}"
export SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-8}"
export REQUIRE_PROMPT_INPUT_IDS="${REQUIRE_PROMPT_INPUT_IDS:-true}"
export LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-qec-v1}"

RUN_TAU_EVAL="${RUN_TAU_EVAL:-auto}"
TAU_SPLITS="${TAU_SPLITS:-$EVAL_SPLITS}"
TAUS="${TAUS:-0.75}"
PREPARE_LIAR_RAW_ATOM_FACTS_ABC_SOURCES="${PREPARE_LIAR_RAW_ATOM_FACTS_ABC_SOURCES:-true}"
FORCE_ATOM_FACTS_ABC_STAGE="${FORCE_ATOM_FACTS_ABC_STAGE:-false}"
STAGE_SAMPLE_LIMIT="${STAGE_SAMPLE_LIMIT:-${SAMPLE_LIMIT:-0}}"

run_cmd() {
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
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
  local case_root="${OUTPUT_ROOT}/liar_raw__ministral3_8b${CASE_SUFFIX}${LORA_SUFFIX}"
  PYTHON_BIN="$PYTHON_BIN" \
    CASE_ROOT="$case_root" \
    SPLITS="$TAU_SPLITS" \
    CHECKPOINTS=best \
    TAUS="$TAUS" \
    LOGIT_ADJUST_MODE=on \
    bash "${SCRIPT_DIR}/run_lora_label_token_logit_adjust_eval_only.sh"
}

stage_force_args=()
if [[ "${FORCE_ATOM_FACTS_ABC_STAGE}" == "true" || "${FORCE_ATOM_FACTS_ABC_STAGE}" == "1" ]]; then
  stage_force_args=(--force)
fi

if [[ "${PREPARE_LIAR_RAW_ATOM_FACTS_ABC_SOURCES}" == "true" || "${PREPARE_LIAR_RAW_ATOM_FACTS_ABC_SOURCES}" == "1" ]]; then
  run_cmd "$PYTHON_BIN" scripts/sentence_trace_method/stage_sources.py \
    --dataset liar_raw \
    --output-root "$OUTPUT_ROOT" \
    --source-root "$SOURCE_ROOT" \
    --selector-name "$SELECTOR_NAME" \
    --graph-version "$SELECTOR_GRAPH_VERSION" \
    --adaptive-policy "$SELECTOR_ADAPTIVE_POLICY" \
    --expected-fingerprint "$EXPECTED_CHUNK_MMR_FINGERPRINT" \
    --sample-limit "$STAGE_SAMPLE_LIMIT" \
    --splits train,val,test \
    --allow-multi-sentence-candidates \
    "${stage_force_args[@]}"
fi

printf '\n[liar-raw-ministral3-atom-facts-abc-%s-lora] DATASETS=%s MODELS=%s TRACE_PROMPT_STYLE=%s CASE_SUFFIX=%s LORA_SUFFIX=%s SELECTOR_NAME=%s SOURCE_ROOT=%s EXPECTED_CHUNK_MMR_FINGERPRINT=%s EBS=16 DEEPSPEED_CONFIG=%s SFT_GRADIENT_ACCUMULATION_STEPS=%s SFT_LEARNING_RATE=%s SFT_NUM_TRAIN_EPOCHS=%s SFT_EVAL_STEPS=%s SFT_SAVE_STEPS=%s SFT_EARLY_STOPPING_PATIENCE=%s REQUIRE_PROMPT_INPUT_IDS=%s LIAR_CLASS_WEIGHTS=%s TAU_POLICY=label_token_logit_adjust_tau%s\n' \
  "$TRACE_PROMPT_STYLE" "$DATASETS" "$MODELS" "$TRACE_PROMPT_STYLE" "$CASE_SUFFIX" "$LORA_SUFFIX" "$SELECTOR_NAME" "$SOURCE_ROOT" "$EXPECTED_CHUNK_MMR_FINGERPRINT" "$DEEPSPEED_CONFIG" "$SFT_GRADIENT_ACCUMULATION_STEPS" "$SFT_LEARNING_RATE" "$SFT_NUM_TRAIN_EPOCHS" "$SFT_EVAL_STEPS" "$SFT_SAVE_STEPS" "$SFT_EARLY_STOPPING_PATIENCE" "$REQUIRE_PROMPT_INPUT_IDS" "$LIAR_CLASS_WEIGHTS" "$TAUS"

bash "${SCRIPT_DIR}/run_lora_matrix.sh"

if [[ "${DRY_RUN:-false}" != "true" ]] && should_run_tau_eval; then
  run_tau_eval
fi
