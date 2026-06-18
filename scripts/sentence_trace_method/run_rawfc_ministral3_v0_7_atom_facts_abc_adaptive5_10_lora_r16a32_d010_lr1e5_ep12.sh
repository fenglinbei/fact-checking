#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCRIPT_DIR="${ROOT_DIR}/scripts/sentence_trace_method"

export PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
export DATASETS="rawfc"
export MODELS="ministral3_8b"
export SELECTOR_NAME="${SELECTOR_NAME:-v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10}"
export SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-evidence_chain_graph_v0_7}"
export SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-budgeted_marginal_v0_7}"
export EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-${SELECTOR_NAME}}"
export CASE_SUFFIX="${CASE_SUFFIX:-__v0_7_atom_facts_abc_bm_adaptive5_10}"
export ALLOW_MULTI_SENTENCE_CANDIDATES="${ALLOW_MULTI_SENTENCE_CANDIDATES:-true}"

export LORA_SUFFIX="${LORA_SUFFIX:-_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.10}"
export SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-5}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-50}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-50}"

ATOM_FACTS_ABC_SOURCE_ROOT="${ATOM_FACTS_ABC_SOURCE_ROOT:-outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10}"
PREPARE_RAWFC_ATOM_FACTS_ABC_SOURCES="${PREPARE_RAWFC_ATOM_FACTS_ABC_SOURCES:-true}"
FORCE_ATOM_FACTS_ABC_STAGE="${FORCE_ATOM_FACTS_ABC_STAGE:-true}"
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

stage_force_args=()
if [[ "${FORCE_ATOM_FACTS_ABC_STAGE}" == "true" || "${FORCE_ATOM_FACTS_ABC_STAGE}" == "1" ]]; then
  stage_force_args=(--force)
fi

if [[ "${PREPARE_RAWFC_ATOM_FACTS_ABC_SOURCES}" == "true" || "${PREPARE_RAWFC_ATOM_FACTS_ABC_SOURCES}" == "1" ]]; then
  run_cmd "$PYTHON_BIN" scripts/sentence_trace_method/stage_sources.py \
    --dataset rawfc \
    --output-root "$OUTPUT_ROOT" \
    --source-root "$ATOM_FACTS_ABC_SOURCE_ROOT" \
    --selector-name "$SELECTOR_NAME" \
    --graph-version "$SELECTOR_GRAPH_VERSION" \
    --adaptive-policy "$SELECTOR_ADAPTIVE_POLICY" \
    --sample-limit "$STAGE_SAMPLE_LIMIT" \
    --splits train,val,test \
    --allow-multi-sentence-candidates \
    "${stage_force_args[@]}"
fi

export PREPARE_RAWFC_ATOM_FACTS_ABC_SOURCES=false
exec bash "${SCRIPT_DIR}/run_rawfc_ministral3_v0_7_adaptive5_10_lora_r16a32_d010_lr1e5_ep12.sh"
