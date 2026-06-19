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
MODE="${MODE:-build}"
EVAL_SPLITS="${EVAL_SPLITS:-val}"
CHECKPOINTS="${CHECKPOINTS:-best}"
RUN_LIAR_RAW="${RUN_LIAR_RAW:-true}"
AA_QEC_STAGE3_CASES="${AA_QEC_STAGE3_CASES:-F1,F2,F3}"
PREPARE_LIAR_RAW_ATOM_FACTS_ABC_SOURCES="${PREPARE_LIAR_RAW_ATOM_FACTS_ABC_SOURCES:-true}"
PREPARE_AA_QEC_SOURCES="${PREPARE_AA_QEC_SOURCES:-true}"
FORCE_ATOM_FACTS_ABC_STAGE="${FORCE_ATOM_FACTS_ABC_STAGE:-false}"
FORCE_AA_QEC_BUILD="${FORCE_AA_QEC_BUILD:-true}"
FORCE_STAGE="${FORCE_STAGE:-true}"
STAGE_SAMPLE_LIMIT="${STAGE_SAMPLE_LIMIT:-${SAMPLE_LIMIT:-0}}"

TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-qec_min}"
SOURCE_SELECTOR_NAME="${SOURCE_SELECTOR_NAME:-v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10}"
SOURCE_GRAPH_VERSION="${SOURCE_GRAPH_VERSION:-evidence_chain_graph_v0_7}"
SOURCE_ADAPTIVE_POLICY="${SOURCE_ADAPTIVE_POLICY:-budgeted_marginal_v0_7}"
ATOM_FACTS_ABC_SOURCE_ROOT="${ATOM_FACTS_ABC_SOURCE_ROOT:-outputs/selectors/evidence_chain_graph/liar_raw_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10}"
EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-d4cbf7c18126}"
ALLOW_MULTI_SENTENCE_CANDIDATES="${ALLOW_MULTI_SENTENCE_CANDIDATES:-true}"
SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-atom_anchored_qec_v1}"
SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-aa_qec_full_atom_facts_abc}"
CANDIDATE_SCOPE="${CANDIDATE_SCOPE:-top20}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
RUN_STAGE3_BUILD_GATE="${RUN_STAGE3_BUILD_GATE:-auto}"
STAGE3_BUILD_GATE_REPORT_PATH="${STAGE3_BUILD_GATE_REPORT_PATH:-${OUTPUT_ROOT}/aa_qec_stage3_build_gate_report.json}"

LORA_SUFFIX="${LORA_SUFFIX:-_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz1_ga4.json}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-4}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-2e-5}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-12}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-$SFT_EVAL_STEPS}"
SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-8}"
REQUIRE_PROMPT_INPUT_IDS="${REQUIRE_PROMPT_INPUT_IDS:-true}"
LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-aa-qec-stage3}"

RUN_TAU_EVAL="${RUN_TAU_EVAL:-auto}"
TAU_SPLITS="${TAU_SPLITS:-$EVAL_SPLITS}"
TAUS="${TAUS:-0.75}"

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

should_run_build_gate() {
  case "$RUN_STAGE3_BUILD_GATE" in
    true) return 0 ;;
    false) return 1 ;;
    auto)
      case "$MODE" in
        build) return 0 ;;
        *) return 1 ;;
      esac
      ;;
    *) printf 'Unsupported RUN_STAGE3_BUILD_GATE=%s. Use true, false, or auto.\n' "$RUN_STAGE3_BUILD_GATE" >&2; exit 2 ;;
  esac
}

run_build_gate() {
  run_cmd "$PYTHON_BIN" scripts/sentence_trace_method/check_aa_qec_stage3_build_gate.py \
    --output-root "$OUTPUT_ROOT" \
    --graph-root outputs/selectors/atom_anchored_qec/liar_raw \
    --source-selector-name "$SOURCE_SELECTOR_NAME" \
    --model ministral3_8b \
    --lora-suffix "$LORA_SUFFIX" \
    --cases "$AA_QEC_STAGE3_CASES" \
    --splits train,val,test \
    --prompt-splits train,val,test \
    --report-path "$STAGE3_BUILD_GATE_REPORT_PATH"
}

case_enabled() {
  local case_id="$1"
  local normalized_cases=",${AA_QEC_STAGE3_CASES^^},"
  normalized_cases="${normalized_cases// /}"
  [[ "$normalized_cases" == *,ALL,* || "$normalized_cases" == *,"$case_id",* ]]
}

stage_liar_raw_atom_facts_abc_sources() {
  if ! truthy "$PREPARE_LIAR_RAW_ATOM_FACTS_ABC_SOURCES"; then
    return 0
  fi
  local force_args=()
  if truthy "$FORCE_ATOM_FACTS_ABC_STAGE"; then
    force_args=(--force)
  fi
  run_cmd "$PYTHON_BIN" scripts/sentence_trace_method/stage_sources.py \
    --dataset liar_raw \
    --output-root "$OUTPUT_ROOT" \
    --source-root "$ATOM_FACTS_ABC_SOURCE_ROOT" \
    --selector-name "$SOURCE_SELECTOR_NAME" \
    --graph-version "$SOURCE_GRAPH_VERSION" \
    --adaptive-policy "$SOURCE_ADAPTIVE_POLICY" \
    --expected-fingerprint "$EXPECTED_CHUNK_MMR_FINGERPRINT" \
    --sample-limit "$STAGE_SAMPLE_LIMIT" \
    --splits train,val,test \
    --allow-multi-sentence-candidates \
    "${force_args[@]}"
}

prepare_source() {
  local selector_name="$1"
  local selection_policy="$2"
  local min_chain_steps="$3"
  local max_chain_steps="$4"
  if ! truthy "$PREPARE_AA_QEC_SOURCES"; then
    return 0
  fi
  PYTHON_BIN="$PYTHON_BIN" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    DATASETS=liar_raw \
    SPLITS=train,val,test \
    SELECTOR_NAME="$selector_name" \
    SELECTOR_GRAPH_VERSION="$SELECTOR_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$SELECTOR_ADAPTIVE_POLICY" \
    SOURCE_SELECTOR_NAME="$SOURCE_SELECTOR_NAME" \
    SOURCE_GRAPH_VERSION="$SOURCE_GRAPH_VERSION" \
    SOURCE_ADAPTIVE_POLICY="$SOURCE_ADAPTIVE_POLICY" \
    SELECTION_POLICY="$selection_policy" \
    CANDIDATE_SCOPE="$CANDIDATE_SCOPE" \
    CANDIDATE_TOP_N="$CANDIDATE_TOP_N" \
    MIN_CHAIN_STEPS="$min_chain_steps" \
    MAX_CHAIN_STEPS="$max_chain_steps" \
    EXPECTED_CHUNK_MMR_FINGERPRINT="$EXPECTED_CHUNK_MMR_FINGERPRINT" \
    ALLOW_MULTI_SENTENCE_CANDIDATES="$ALLOW_MULTI_SENTENCE_CANDIDATES" \
    FORCE_AA_QEC_BUILD="$FORCE_AA_QEC_BUILD" \
    FORCE_STAGE="$FORCE_STAGE" \
    DRY_RUN="${DRY_RUN:-false}" \
    bash "${SCRIPT_DIR}/prepare_aa_qec_sources.sh"
}

run_tau_eval() {
  local case_suffix="$1"
  local case_root="${OUTPUT_ROOT}/liar_raw__ministral3_8b${case_suffix}${LORA_SUFFIX}"
  PYTHON_BIN="$PYTHON_BIN" \
    CASE_ROOT="$case_root" \
    SPLITS="$TAU_SPLITS" \
    CHECKPOINTS=best \
    TAUS="$TAUS" \
    LOGIT_ADJUST_MODE=on \
    bash "${SCRIPT_DIR}/run_lora_label_token_logit_adjust_eval_only.sh"
}

run_liar_raw_case() {
  local case_id="$1"
  local selector_name="$2"
  local selection_policy="$3"
  local case_suffix="$4"
  local min_chain_steps="$5"
  local max_chain_steps="$6"

  printf '\n[aa-qec-stage3-liar-raw-atom-facts-abc] ID=%s DATASETS=liar_raw MODELS=ministral3_8b MODE=%s EVAL_SPLITS=%s SELECTOR_NAME=%s SELECTOR_ADAPTIVE_POLICY=%s SOURCE_SELECTOR_NAME=%s SOURCE_ROOT=%s EXPECTED_CHUNK_MMR_FINGERPRINT=%s ALLOW_MULTI_SENTENCE_CANDIDATES=%s FORCE_AA_QEC_BUILD=%s FORCE_STAGE=%s CANDIDATE_SCOPE=%s CANDIDATE_TOP_N=%s SELECTION_POLICY=%s MIN_CHAIN_STEPS=%s MAX_CHAIN_STEPS=%s TRACE_PROMPT_STYLE=%s CASE_SUFFIX=%s LORA_SUFFIX=%s EBS=16 DEEPSPEED_CONFIG=%s SFT_GRADIENT_ACCUMULATION_STEPS=%s SFT_LEARNING_RATE=%s SFT_NUM_TRAIN_EPOCHS=%s SFT_EVAL_STEPS=%s SFT_SAVE_STEPS=%s SFT_EARLY_STOPPING_PATIENCE=%s REQUIRE_PROMPT_INPUT_IDS=%s LIAR_CLASS_WEIGHTS=%s TAU_POLICY=label_token_logit_adjust_tau%s\n' \
    "$case_id" "$MODE" "$EVAL_SPLITS" "$selector_name" "$SELECTOR_ADAPTIVE_POLICY" "$SOURCE_SELECTOR_NAME" "$ATOM_FACTS_ABC_SOURCE_ROOT" "$EXPECTED_CHUNK_MMR_FINGERPRINT" "$ALLOW_MULTI_SENTENCE_CANDIDATES" "$FORCE_AA_QEC_BUILD" "$FORCE_STAGE" "$CANDIDATE_SCOPE" "$CANDIDATE_TOP_N" "$selection_policy" "$min_chain_steps" "$max_chain_steps" "$TRACE_PROMPT_STYLE" "$case_suffix" "$LORA_SUFFIX" "$DEEPSPEED_CONFIG" "$SFT_GRADIENT_ACCUMULATION_STEPS" "$SFT_LEARNING_RATE" "$SFT_NUM_TRAIN_EPOCHS" "$SFT_EVAL_STEPS" "$SFT_SAVE_STEPS" "$SFT_EARLY_STOPPING_PATIENCE" "$REQUIRE_PROMPT_INPUT_IDS" "$LIAR_CLASS_WEIGHTS" "$TAUS"

  prepare_source "$selector_name" "$selection_policy" "$min_chain_steps" "$max_chain_steps"

  PYTHON_BIN="$PYTHON_BIN" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    MODE="$MODE" \
    EVAL_SPLITS="$EVAL_SPLITS" \
    CHECKPOINTS="$CHECKPOINTS" \
    DATASETS=liar_raw \
    MODELS=ministral3_8b \
    SELECTOR_NAME="$selector_name" \
    SELECTOR_GRAPH_VERSION="$SELECTOR_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$SELECTOR_ADAPTIVE_POLICY" \
    EXPECTED_SELECTOR_NAME="$selector_name" \
    SOURCE_ROOT="outputs/selectors/atom_anchored_qec/liar_raw/${selector_name}" \
    EXPECTED_CHUNK_MMR_FINGERPRINT="$EXPECTED_CHUNK_MMR_FINGERPRINT" \
    ALLOW_MULTI_SENTENCE_CANDIDATES="$ALLOW_MULTI_SENTENCE_CANDIDATES" \
    FORCE_STAGE="$FORCE_STAGE" \
    CASE_SUFFIX="$case_suffix" \
    LORA_SUFFIX="$LORA_SUFFIX" \
    LORA_R="$LORA_R" \
    LORA_ALPHA="$LORA_ALPHA" \
    LORA_DROPOUT="$LORA_DROPOUT" \
    DEEPSPEED_CONFIG="$DEEPSPEED_CONFIG" \
    SFT_GRADIENT_ACCUMULATION_STEPS="$SFT_GRADIENT_ACCUMULATION_STEPS" \
    SFT_LEARNING_RATE="$SFT_LEARNING_RATE" \
    SFT_NUM_TRAIN_EPOCHS="$SFT_NUM_TRAIN_EPOCHS" \
    SFT_EVAL_STEPS="$SFT_EVAL_STEPS" \
    SFT_SAVE_STEPS="$SFT_SAVE_STEPS" \
    SFT_EARLY_STOPPING_PATIENCE="$SFT_EARLY_STOPPING_PATIENCE" \
    REQUIRE_PROMPT_INPUT_IDS="$REQUIRE_PROMPT_INPUT_IDS" \
    LIAR_CLASS_WEIGHTS="$LIAR_CLASS_WEIGHTS" \
    TRACE_PROMPT_STYLE="$TRACE_PROMPT_STYLE" \
    SWANLAB_PROJECT="$SWANLAB_PROJECT" \
    DRY_RUN="${DRY_RUN:-false}" \
    bash "${SCRIPT_DIR}/run_lora_matrix.sh"

  if [[ "${DRY_RUN:-false}" != "true" ]] && should_run_tau_eval; then
    run_tau_eval "$case_suffix"
  fi
}

stage_liar_raw_atom_facts_abc_sources

if truthy "$RUN_LIAR_RAW"; then
  if case_enabled F1; then
    run_liar_raw_case \
      F1 \
      aa_qec_full_atom_facts_abc_primary_fallback_no_secondary_qd_prefer_top20_min5_10 \
      primary_fallback_min5_no_secondary \
      __aa_qec_f1_atom_facts_abc_primary_fallback_no_secondary \
      5 \
      10
  fi
  if case_enabled F2; then
    run_liar_raw_case \
      F2 \
      aa_qec_full_atom_facts_abc_primary_secondary_fallback_qd_prefer_top20_min5_10 \
      primary_secondary_fallback_min5 \
      __aa_qec_f2_atom_facts_abc_primary_secondary_fallback \
      5 \
      10
  fi
  if case_enabled F3; then
    run_liar_raw_case \
      F3 \
      aa_qec_full_atom_facts_abc_primary_secondary_dynamic_qd_prefer_top20 \
      primary_secondary \
      __aa_qec_f3_atom_facts_abc_primary_secondary_dynamic \
      0 \
      0
  fi
fi

if should_run_build_gate; then
  run_build_gate
fi
