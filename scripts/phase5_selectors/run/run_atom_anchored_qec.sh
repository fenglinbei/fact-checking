#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT="${INPUT:-outputs/sentence_trace_method/_sources/rawfc/v0_7_budgeted_marginal_chain_adaptive5_10/${SPLIT}/selection_trace_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/atom_anchored_qec/stage1_${SPLIT}}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
MIN_CHAIN_STEPS="${MIN_CHAIN_STEPS:-5}"
MAX_CHAIN_STEPS="${MAX_CHAIN_STEPS:-10}"
CUE_POLICY="${CUE_POLICY:-qd_prefer}"
CANDIDATE_SCOPE="${CANDIDATE_SCOPE:-selected}"
SELECTION_POLICY="${SELECTION_POLICY:-keep_all_reorder}"
SOURCE_SELECTOR_NAME="${SOURCE_SELECTOR_NAME:-v0_7_budgeted_marginal_chain_adaptive5_10}"
RANDOM_SEED="${RANDOM_SEED:-0}"

SAMPLE_ARGS=()
if [[ "$SAMPLE_LIMIT" != "0" ]]; then
  SAMPLE_ARGS=(--sample-limit "$SAMPLE_LIMIT")
fi

echo "[aa-qec] split           : ${SPLIT}"
echo "[aa-qec] input           : ${INPUT}"
echo "[aa-qec] output          : ${OUTPUT_DIR}"
echo "[aa-qec] policy          : ${SELECTION_POLICY}"
echo "[aa-qec] cue/scope       : ${CUE_POLICY}/${CANDIDATE_SCOPE}"
echo "[aa-qec] top_n/min/max   : ${CANDIDATE_TOP_N}/${MIN_CHAIN_STEPS}/${MAX_CHAIN_STEPS}"
echo "[aa-qec] random_seed     : ${RANDOM_SEED}"

PYTHONPATH=src "$PYTHON_BIN" scripts/phase5_selectors/build/build_atom_anchored_qec.py \
  --input "$INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --candidate-top-n "$CANDIDATE_TOP_N" \
  --min-chain-steps "$MIN_CHAIN_STEPS" \
  --max-chain-steps "$MAX_CHAIN_STEPS" \
  --cue-policy "$CUE_POLICY" \
  --candidate-scope "$CANDIDATE_SCOPE" \
  --selection-policy "$SELECTION_POLICY" \
  --source-selector-name "$SOURCE_SELECTOR_NAME" \
  --random-seed "$RANDOM_SEED" \
  "${SAMPLE_ARGS[@]}"

echo "[aa-qec] done: ${OUTPUT_DIR}"
