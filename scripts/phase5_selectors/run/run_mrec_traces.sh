#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT="${INPUT:-outputs/sentence_trace_method/_sources/liar_raw/v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10/${SPLIT}/selection_trace_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/mrec/liar_raw/mrec_greedy_transition_v0_1_${SPLIT}}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
MAX_STEPS="${MAX_STEPS:-10}"
TOKEN_BUDGET="${TOKEN_BUDGET:-0}"
TARGET_RESOLVED_RATE="${TARGET_RESOLVED_RATE:-0.80}"
CONTINUE_AFTER_TARGET_FOR_CONTRAST="${CONTINUE_AFTER_TARGET_FOR_CONTRAST:-false}"
DISABLE_FALLBACK="${DISABLE_FALLBACK:-false}"
SELECTOR_NAME="${SELECTOR_NAME:-mrec_greedy_transition_v0_1}"
SOURCE_SELECTOR_NAME="${SOURCE_SELECTOR_NAME:-v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10}"

SAMPLE_ARGS=()
if [[ "$SAMPLE_LIMIT" != "0" ]]; then
  SAMPLE_ARGS=(--sample-limit "$SAMPLE_LIMIT")
fi

EXTRA_ARGS=()
if [[ "$CONTINUE_AFTER_TARGET_FOR_CONTRAST" == "true" || "$CONTINUE_AFTER_TARGET_FOR_CONTRAST" == "1" ]]; then
  EXTRA_ARGS+=(--continue-after-target-for-contrast)
fi
if [[ "$DISABLE_FALLBACK" == "true" || "$DISABLE_FALLBACK" == "1" ]]; then
  EXTRA_ARGS+=(--disable-fallback)
fi

echo "[mrec] split             : ${SPLIT}"
echo "[mrec] input             : ${INPUT}"
echo "[mrec] output            : ${OUTPUT_DIR}"
echo "[mrec] selector          : ${SELECTOR_NAME}"
echo "[mrec] source selector   : ${SOURCE_SELECTOR_NAME}"
echo "[mrec] top_n/max_steps   : ${CANDIDATE_TOP_N}/${MAX_STEPS}"
echo "[mrec] token_budget      : ${TOKEN_BUDGET}"
echo "[mrec] target_resolved   : ${TARGET_RESOLVED_RATE}"

PYTHONPATH=src "$PYTHON_BIN" scripts/phase5_selectors/build/build_mrec_traces.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --candidate-top-n "$CANDIDATE_TOP_N" \
  --max-steps "$MAX_STEPS" \
  --token-budget "$TOKEN_BUDGET" \
  --target-resolved-rate "$TARGET_RESOLVED_RATE" \
  --selector-name "$SELECTOR_NAME" \
  --source-selector-name "$SOURCE_SELECTOR_NAME" \
  "${SAMPLE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

echo "[mrec] done: ${OUTPUT_DIR}"
