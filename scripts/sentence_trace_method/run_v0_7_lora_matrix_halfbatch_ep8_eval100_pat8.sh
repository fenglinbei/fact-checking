#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SELECTOR_NAME="${SELECTOR_NAME:-v0_7_budgeted_marginal_chain_adaptive3_10}"
export SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-evidence_chain_graph_v0_7}"
export SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-budgeted_marginal_v0_7}"
export EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-$SELECTOR_NAME}"

export CASE_SUFFIX="${CASE_SUFFIX:-__v0_7_bm}"
export LORA_SUFFIX="${LORA_SUFFIX:-_lora_halfbatch_ep8_eval100_pat8}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz1_ga4.json}"
export SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-4}"
export SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-8}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-$SFT_EVAL_STEPS}"
export SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-8}"
export LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-v0-7-lora}"

if [[ "${PREPARE_V0_7_SOURCES:-true}" == "true" ]]; then
  bash "${SCRIPT_DIR}/prepare_v0_7_sources.sh"
fi

printf '[v0.7-lora] selector=%s suffix=%s\n' "$SELECTOR_NAME" "$LORA_SUFFIX"
printf '[v0.7-lora] datasets=%s models=%s mode=%s\n' "${DATASETS:-liar_raw,rawfc}" "${MODELS:-llama31_8b,qwen3_4b_2507}" "${MODE:-full}"

exec bash "${SCRIPT_DIR}/run_lora_matrix.sh"
