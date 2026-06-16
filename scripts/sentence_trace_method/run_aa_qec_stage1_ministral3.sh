#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCRIPT_DIR="${ROOT_DIR}/scripts/sentence_trace_method"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
MODE="${MODE:-build}"
EVAL_SPLITS="${EVAL_SPLITS:-val}"
CHECKPOINTS="${CHECKPOINTS:-best}"
RUN_RAWFC="${RUN_RAWFC:-true}"
PREPARE_AA_QEC_SOURCES="${PREPARE_AA_QEC_SOURCES:-true}"

TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-qec_min}"
SOURCE_SELECTOR_NAME="${SOURCE_SELECTOR_NAME:-v0_7_budgeted_marginal_chain_adaptive5_10}"
SOURCE_GRAPH_VERSION="${SOURCE_GRAPH_VERSION:-evidence_chain_graph_v0_7}"
SOURCE_ADAPTIVE_POLICY="${SOURCE_ADAPTIVE_POLICY:-budgeted_marginal_v0_7}"
SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-atom_anchored_qec_v1}"
SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-aa_qec_view}"

truthy() {
  case "$1" in
    true|1|yes|y) return 0 ;;
    false|0|no|n) return 1 ;;
    *) printf 'Unsupported boolean value: %s\n' "$1" >&2; exit 2 ;;
  esac
}

prepare_source() {
  local selector_name="$1"
  local selection_policy="$2"
  local random_seed="$3"
  if ! truthy "$PREPARE_AA_QEC_SOURCES"; then
    return 0
  fi
  PYTHON_BIN="$PYTHON_BIN" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    DATASETS=rawfc \
    SPLITS=train,val,test \
    SELECTOR_NAME="$selector_name" \
    SELECTOR_GRAPH_VERSION="$SELECTOR_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$SELECTOR_ADAPTIVE_POLICY" \
    SOURCE_SELECTOR_NAME="$SOURCE_SELECTOR_NAME" \
    SOURCE_GRAPH_VERSION="$SOURCE_GRAPH_VERSION" \
    SOURCE_ADAPTIVE_POLICY="$SOURCE_ADAPTIVE_POLICY" \
    SELECTION_POLICY="$selection_policy" \
    RANDOM_SEED="$random_seed" \
    DRY_RUN="${DRY_RUN:-false}" \
    bash "${SCRIPT_DIR}/prepare_aa_qec_sources.sh"
}

run_rawfc_case() {
  local case_id="$1"
  local selector_name="$2"
  local selection_policy="$3"
  local case_suffix="$4"
  local random_seed="$5"
  local lora_suffix="_lora_ebs16_lr1em5_ep10_eval50_pat8_rawfc"

  printf '\n[aa-qec-stage1-rawfc] ID=%s SELECTOR_NAME=%s SELECTION_POLICY=%s TRACE_PROMPT_STYLE=%s CASE_SUFFIX=%s LORA_SUFFIX=%s EBS=16 DEEPSPEED_CONFIG=configs/deepspeed_zero2_bsz1_ga4.json SFT_GRADIENT_ACCUMULATION_STEPS=4 SFT_LEARNING_RATE=1e-5 SFT_NUM_TRAIN_EPOCHS=10 SFT_EVAL_STEPS=50 SFT_SAVE_STEPS=50 SFT_EARLY_STOPPING_PATIENCE=8 REQUIRE_PROMPT_INPUT_IDS=true\n' \
    "$case_id" "$selector_name" "$selection_policy" "$TRACE_PROMPT_STYLE" "$case_suffix" "$lora_suffix"

  prepare_source "$selector_name" "$selection_policy" "$random_seed"

  PYTHON_BIN="$PYTHON_BIN" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    MODE="$MODE" \
    EVAL_SPLITS="$EVAL_SPLITS" \
    CHECKPOINTS="$CHECKPOINTS" \
    DATASETS=rawfc \
    MODELS=ministral3_8b \
    SELECTOR_NAME="$selector_name" \
    SELECTOR_GRAPH_VERSION="$SELECTOR_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$SELECTOR_ADAPTIVE_POLICY" \
    EXPECTED_SELECTOR_NAME="$selector_name" \
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
    TRACE_PROMPT_STYLE="$TRACE_PROMPT_STYLE" \
    SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-aa-qec-stage1}" \
    DRY_RUN="${DRY_RUN:-false}" \
    bash "${SCRIPT_DIR}/run_lora_matrix.sh"
}

if truthy "$RUN_RAWFC"; then
  run_rawfc_case \
    O1 \
    aa_qec_view_keep_all_qd_prefer_selected_min5_10 \
    keep_all_reorder \
    __aa_qec_o1_view_atom_order \
    0
  run_rawfc_case \
    O2 \
    aa_qec_view_primary_secondary_order_qd_prefer_selected_min5_10 \
    primary_secondary_order \
    __aa_qec_o2_view_primary_secondary_order \
    0
  run_rawfc_case \
    O3 \
    aa_qec_view_shuffled_qd_prefer_selected_min5_10 \
    shuffled \
    __aa_qec_o3_view_shuffled \
    13
fi
