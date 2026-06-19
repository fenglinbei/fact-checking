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

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
MODE="${MODE:-full}"
EVAL_SPLITS="${EVAL_SPLITS:-val,test}"
CHECKPOINTS="${CHECKPOINTS:-best}"
RUN_LIAR_RAW="${RUN_LIAR_RAW:-true}"
RUN_RAWFC="${RUN_RAWFC:-true}"
TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-mrec_min}"
REQUIRE_PROMPT_INPUT_IDS="${REQUIRE_PROMPT_INPUT_IDS:-true}"
ALLOW_MULTI_SENTENCE_CANDIDATES="${ALLOW_MULTI_SENTENCE_CANDIDATES:-true}"
STAGE_SAMPLE_LIMIT="${STAGE_SAMPLE_LIMIT:-${SAMPLE_LIMIT:-0}}"

MREC_SELECTOR_NAME="${MREC_SELECTOR_NAME:-mrec_greedy_transition_v0_1}"
MREC_GRAPH_VERSION="${MREC_GRAPH_VERSION:-mrec_trace_v0_1}"
MREC_ADAPTIVE_POLICY="${MREC_ADAPTIVE_POLICY:-minimal_resolving_chain_v0_1}"
MREC_CANDIDATE_TOP_N="${MREC_CANDIDATE_TOP_N:-20}"
MREC_MAX_STEPS="${MREC_MAX_STEPS:-10}"
MREC_TOKEN_BUDGET="${MREC_TOKEN_BUDGET:-0}"
MREC_TARGET_RESOLVED_RATE="${MREC_TARGET_RESOLVED_RATE:-0.80}"
MREC_CONTINUE_AFTER_TARGET_FOR_CONTRAST="${MREC_CONTINUE_AFTER_TARGET_FOR_CONTRAST:-false}"
MREC_DISABLE_FALLBACK="${MREC_DISABLE_FALLBACK:-false}"
PREPARE_MREC_SOURCES="${PREPARE_MREC_SOURCES:-true}"
FORCE_MREC_BUILD="${FORCE_MREC_BUILD:-false}"
FORCE_MREC_STAGE="${FORCE_MREC_STAGE:-true}"
RUN_MREC_DIAGNOSTICS="${RUN_MREC_DIAGNOSTICS:-auto}"
MREC_DIAGNOSTIC_SPLITS="${MREC_DIAGNOSTIC_SPLITS:-train,val,test}"
MREC_DIAGNOSTIC_MAX_TRUNCATION_RATE="${MREC_DIAGNOSTIC_MAX_TRUNCATION_RATE:-0.02}"

LIAR_SOURCE_SELECTOR_NAME="${LIAR_SOURCE_SELECTOR_NAME:-v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10}"
LIAR_SOURCE_ROOT="${LIAR_SOURCE_ROOT:-outputs/selectors/evidence_chain_graph/liar_raw_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10}"
LIAR_EXPECTED_CHUNK_MMR_FINGERPRINT="${LIAR_EXPECTED_CHUNK_MMR_FINGERPRINT:-d4cbf7c18126}"
LIAR_CASE_SUFFIX="${LIAR_CASE_SUFFIX:-__mrec_min}"
LIAR_LORA_SUFFIX="${LIAR_LORA_SUFFIX:-_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8}"
LIAR_TAUS="${LIAR_TAUS:-0.75}"
RUN_LIAR_TAU_EVAL="${RUN_LIAR_TAU_EVAL:-auto}"
TAU_SPLITS="${TAU_SPLITS:-$EVAL_SPLITS}"

RAWFC_SOURCE_SELECTOR_NAME="${RAWFC_SOURCE_SELECTOR_NAME:-v0_7_atom_facts_abc_tight_budgeted_marginal_chain_adaptive5_10}"
RAWFC_SOURCE_ROOT="${RAWFC_SOURCE_ROOT:-outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_abc_tight_budgeted_marginal_adaptive5_10}"
RAWFC_CASE_SUFFIX="${RAWFC_CASE_SUFFIX:-__mrec_min_anchor_only}"
RAWFC_LORA_SUFFIX="${RAWFC_LORA_SUFFIX:-_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc}"

truthy() {
  case "$1" in
    true|1|yes|y) return 0 ;;
    false|0|no|n) return 1 ;;
    *) printf 'Unsupported boolean value: %s\n' "$1" >&2; exit 2 ;;
  esac
}

run_cmd() {
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
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

should_run_mrec_diagnostics() {
  case "$RUN_MREC_DIAGNOSTICS" in
    true) return 0 ;;
    false) return 1 ;;
    auto)
      case "$MODE" in
        build|full) return 0 ;;
        *) return 1 ;;
      esac
      ;;
    *) printf 'Unsupported RUN_MREC_DIAGNOSTICS=%s. Use true, false, or auto.\n' "$RUN_MREC_DIAGNOSTICS" >&2; exit 2 ;;
  esac
}

stage_base_sources() {
  local dataset="$1"
  local selector_name="$2"
  local source_root="$3"
  local expected_fingerprint="$4"
  local extra_args=()
  if [[ -n "$expected_fingerprint" ]]; then
    extra_args+=(--expected-fingerprint "$expected_fingerprint")
  fi
  if [[ "$ALLOW_MULTI_SENTENCE_CANDIDATES" == "true" || "$ALLOW_MULTI_SENTENCE_CANDIDATES" == "1" ]]; then
    extra_args+=(--allow-multi-sentence-candidates)
  fi
  run_cmd "$PYTHON_BIN" scripts/sentence_trace_method/stage_sources.py \
    --dataset "$dataset" \
    --output-root "$OUTPUT_ROOT" \
    --source-root "$source_root" \
    --selector-name "$selector_name" \
    --graph-version evidence_chain_graph_v0_7 \
    --adaptive-policy budgeted_marginal_v0_7 \
    --sample-limit "$STAGE_SAMPLE_LIMIT" \
    --splits train,val,test \
    "${extra_args[@]}"
}

prepare_mrec_sources_for() {
  local dataset="$1"
  local source_selector_name="$2"
  local expected_fingerprint="$3"
  if ! truthy "$PREPARE_MREC_SOURCES"; then
    return 0
  fi
  PYTHON_BIN="$PYTHON_BIN" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    DATASETS="$dataset" \
    SPLITS=train,val,test \
    SELECTOR_NAME="$MREC_SELECTOR_NAME" \
    SELECTOR_GRAPH_VERSION="$MREC_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$MREC_ADAPTIVE_POLICY" \
    SOURCE_SELECTOR_NAME="$source_selector_name" \
    CANDIDATE_TOP_N="$MREC_CANDIDATE_TOP_N" \
    MAX_STEPS="$MREC_MAX_STEPS" \
    TOKEN_BUDGET="$MREC_TOKEN_BUDGET" \
    TARGET_RESOLVED_RATE="$MREC_TARGET_RESOLVED_RATE" \
    CONTINUE_AFTER_TARGET_FOR_CONTRAST="$MREC_CONTINUE_AFTER_TARGET_FOR_CONTRAST" \
    DISABLE_FALLBACK="$MREC_DISABLE_FALLBACK" \
    EXPECTED_CHUNK_MMR_FINGERPRINT="$expected_fingerprint" \
    ALLOW_MULTI_SENTENCE_CANDIDATES="$ALLOW_MULTI_SENTENCE_CANDIDATES" \
    FORCE_MREC_BUILD="$FORCE_MREC_BUILD" \
    FORCE_STAGE="$FORCE_MREC_STAGE" \
    DRY_RUN="${DRY_RUN:-false}" \
    bash "${SCRIPT_DIR}/prepare_mrec_sources.sh"
}

run_liar_tau_eval() {
  local case_root="${OUTPUT_ROOT}/liar_raw__ministral3_8b${LIAR_CASE_SUFFIX}${LIAR_LORA_SUFFIX}"
  PYTHON_BIN="$PYTHON_BIN" \
    CASE_ROOT="$case_root" \
    SPLITS="$TAU_SPLITS" \
    CHECKPOINTS=best \
    TAUS="$LIAR_TAUS" \
    LOGIT_ADJUST_MODE=on \
    bash "${SCRIPT_DIR}/run_lora_label_token_logit_adjust_eval_only.sh"
}

run_mrec_diagnostics() {
  local dataset="$1"
  local case_root="$2"
  local evidence_text_mode="$3"
  if ! should_run_mrec_diagnostics; then
    return 0
  fi
  run_cmd "$PYTHON_BIN" scripts/sentence_trace_method/check_mrec_diagnostics.py \
    --dataset "$dataset" \
    --output-root "$OUTPUT_ROOT" \
    --case-root "$case_root" \
    --source-selector-name "$MREC_SELECTOR_NAME" \
    --splits "$MREC_DIAGNOSTIC_SPLITS" \
    --expected-trace-prompt-style "$TRACE_PROMPT_STYLE" \
    --expected-evidence-text-mode "$evidence_text_mode" \
    --max-truncation-rate "$MREC_DIAGNOSTIC_MAX_TRUNCATION_RATE" \
    --report-path "${case_root}/mrec_diagnostics_report.json"
}

run_liar_raw_mrec() {
  local mrec_source_root="outputs/selectors/mrec/liar_raw/${MREC_SELECTOR_NAME}"
  local case_root="${OUTPUT_ROOT}/liar_raw__ministral3_8b${LIAR_CASE_SUFFIX}"
  printf '\n[mrec-main-liar-raw] DATASETS=liar_raw MODELS=ministral3_8b TRACE_PROMPT_STYLE=%s CASE_SUFFIX=%s LORA_SUFFIX=%s SELECTOR_NAME=%s SOURCE_SELECTOR_NAME=%s SOURCE_ROOT=%s EBS=16 DEEPSPEED_CONFIG=configs/deepspeed_zero2_bsz1_ga4.json SFT_GRADIENT_ACCUMULATION_STEPS=4 SFT_LEARNING_RATE=2e-5 SFT_NUM_TRAIN_EPOCHS=12 SFT_EVAL_STEPS=100 SFT_SAVE_STEPS=100 SFT_EARLY_STOPPING_PATIENCE=8 SFT_EARLY_STOPPING_METRIC=macro_f1 REQUIRE_PROMPT_INPUT_IDS=%s TAU_POLICY=label_token_logit_adjust_tau%s\n' \
    "$TRACE_PROMPT_STYLE" "$LIAR_CASE_SUFFIX" "$LIAR_LORA_SUFFIX" "$MREC_SELECTOR_NAME" "$LIAR_SOURCE_SELECTOR_NAME" "$mrec_source_root" "$REQUIRE_PROMPT_INPUT_IDS" "$LIAR_TAUS"
  PYTHON_BIN="$PYTHON_BIN" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    MODE="$MODE" \
    EVAL_SPLITS="$EVAL_SPLITS" \
    CHECKPOINTS="$CHECKPOINTS" \
    DATASETS=liar_raw \
    MODELS=ministral3_8b \
    SELECTOR_NAME="$MREC_SELECTOR_NAME" \
    SELECTOR_GRAPH_VERSION="$MREC_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$MREC_ADAPTIVE_POLICY" \
    EXPECTED_SELECTOR_NAME="$MREC_SELECTOR_NAME" \
    SOURCE_ROOT="$mrec_source_root" \
    EXPECTED_CHUNK_MMR_FINGERPRINT="$LIAR_EXPECTED_CHUNK_MMR_FINGERPRINT" \
    CASE_SUFFIX="$LIAR_CASE_SUFFIX" \
    LORA_SUFFIX="$LIAR_LORA_SUFFIX" \
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
    SFT_EARLY_STOPPING_METRIC=macro_f1 \
    REQUIRE_PROMPT_INPUT_IDS="$REQUIRE_PROMPT_INPUT_IDS" \
    TRACE_PROMPT_STYLE="$TRACE_PROMPT_STYLE" \
    LIAR_CLASS_WEIGHTS="$LIAR_CLASS_WEIGHTS" \
    SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-mrec}" \
    ALLOW_MULTI_SENTENCE_CANDIDATES="$ALLOW_MULTI_SENTENCE_CANDIDATES" \
    FORCE_STAGE=false \
    bash "${SCRIPT_DIR}/run_lora_matrix.sh"
  run_mrec_diagnostics liar_raw "$case_root" full
  if [[ "${DRY_RUN:-false}" != "true" ]] && should_run_liar_tau_eval; then
    run_liar_tau_eval
  fi
}

run_rawfc_mrec() {
  local mrec_source_root="outputs/selectors/mrec/rawfc/${MREC_SELECTOR_NAME}"
  local case_root="${OUTPUT_ROOT}/rawfc__ministral3_8b${RAWFC_CASE_SUFFIX}"
  printf '\n[mrec-main-rawfc] DATASETS=rawfc MODELS=ministral3_8b TRACE_PROMPT_STYLE=%s EVIDENCE_TEXT_MODE=anchor_only CASE_SUFFIX=%s LORA_SUFFIX=%s SELECTOR_NAME=%s SOURCE_SELECTOR_NAME=%s SOURCE_ROOT=%s EBS=16 DEEPSPEED_CONFIG=configs/deepspeed_zero2_bsz1_ga4.json SFT_GRADIENT_ACCUMULATION_STEPS=4 SFT_LEARNING_RATE=1e-5 SFT_NUM_TRAIN_EPOCHS=12 SFT_EVAL_STEPS=50 SFT_SAVE_STEPS=50 SFT_EARLY_STOPPING_PATIENCE=8 SFT_EARLY_STOPPING_METRIC=macro_f1 REQUIRE_PROMPT_INPUT_IDS=%s\n' \
    "$TRACE_PROMPT_STYLE" "$RAWFC_CASE_SUFFIX" "$RAWFC_LORA_SUFFIX" "$MREC_SELECTOR_NAME" "$RAWFC_SOURCE_SELECTOR_NAME" "$mrec_source_root" "$REQUIRE_PROMPT_INPUT_IDS"
  PYTHON_BIN="$PYTHON_BIN" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    MODE="$MODE" \
    EVAL_SPLITS="$EVAL_SPLITS" \
    CHECKPOINTS="$CHECKPOINTS" \
    DATASETS=rawfc \
    MODELS=ministral3_8b \
    SELECTOR_NAME="$MREC_SELECTOR_NAME" \
    SELECTOR_GRAPH_VERSION="$MREC_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$MREC_ADAPTIVE_POLICY" \
    EXPECTED_SELECTOR_NAME="$MREC_SELECTOR_NAME" \
    SOURCE_ROOT="$mrec_source_root" \
    CASE_SUFFIX="$RAWFC_CASE_SUFFIX" \
    LORA_SUFFIX="$RAWFC_LORA_SUFFIX" \
    LORA_R=16 \
    LORA_ALPHA=32 \
    LORA_DROPOUT=0.10 \
    DEEPSPEED_CONFIG=configs/deepspeed_zero2_bsz1_ga4.json \
    SFT_GRADIENT_ACCUMULATION_STEPS=4 \
    SFT_LEARNING_RATE=1e-5 \
    SFT_NUM_TRAIN_EPOCHS=12 \
    SFT_EVAL_STEPS=50 \
    SFT_SAVE_STEPS=50 \
    SFT_EARLY_STOPPING_PATIENCE=8 \
    SFT_EARLY_STOPPING_METRIC=macro_f1 \
    REQUIRE_PROMPT_INPUT_IDS="$REQUIRE_PROMPT_INPUT_IDS" \
    TRACE_PROMPT_STYLE="$TRACE_PROMPT_STYLE" \
    EVIDENCE_TEXT_MODE=anchor_only \
    SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-mrec}" \
    ALLOW_MULTI_SENTENCE_CANDIDATES="$ALLOW_MULTI_SENTENCE_CANDIDATES" \
    FORCE_STAGE=false \
    bash "${SCRIPT_DIR}/run_lora_matrix.sh"
  run_mrec_diagnostics rawfc "$case_root" anchor_only
}

if truthy "$RUN_LIAR_RAW"; then
  stage_base_sources liar_raw "$LIAR_SOURCE_SELECTOR_NAME" "$LIAR_SOURCE_ROOT" "$LIAR_EXPECTED_CHUNK_MMR_FINGERPRINT"
  prepare_mrec_sources_for liar_raw "$LIAR_SOURCE_SELECTOR_NAME" "$LIAR_EXPECTED_CHUNK_MMR_FINGERPRINT"
  run_liar_raw_mrec
fi

if truthy "$RUN_RAWFC"; then
  stage_base_sources rawfc "$RAWFC_SOURCE_SELECTOR_NAME" "$RAWFC_SOURCE_ROOT" ""
  prepare_mrec_sources_for rawfc "$RAWFC_SOURCE_SELECTOR_NAME" ""
  run_rawfc_mrec
fi
