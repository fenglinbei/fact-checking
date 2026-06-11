#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT="${INPUT:-outputs/selectors/evidence_map_selector/v0_6b_${SPLIT}/candidate_evidence_map_features_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/evidence_chain_graph/v0_7_budgeted_marginal_adaptive3_10_${SPLIT}}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
MIN_TOP_K="${MIN_TOP_K:-3}"
MAX_TOP_K="${MAX_TOP_K:-10}"
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
TARGET_COVERAGE="${TARGET_COVERAGE:-0.80}"
STOP_GAIN_THRESHOLD="${STOP_GAIN_THRESHOLD:-0.10}"
INSUFFICIENT_GAIN_THRESHOLD="${INSUFFICIENT_GAIN_THRESHOLD:-0.05}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

SAMPLE_ARGS=()
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  SAMPLE_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

echo "[evidence-chain-graph-v0.7] split       : ${SPLIT}"
echo "[evidence-chain-graph-v0.7] input       : ${INPUT}"
echo "[evidence-chain-graph-v0.7] output      : ${OUTPUT_DIR}"
echo "[evidence-chain-graph-v0.7] top_n/min/max: ${CANDIDATE_TOP_N}/${MIN_TOP_K}/${MAX_TOP_K}"
echo "[evidence-chain-graph-v0.7] fingerprint : ${CHUNK_MMR_FINGERPRINT}"
echo "[evidence-chain-graph-v0.7] stop        : ${TARGET_COVERAGE}/${STOP_GAIN_THRESHOLD}/${INSUFFICIENT_GAIN_THRESHOLD}"

PYTHONPATH=src "$PYTHON_BIN" scripts/phase5_selectors/build/build_evidence_chain_graph_v0_7.py \
  --input "${INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --candidate-top-n "${CANDIDATE_TOP_N}" \
  --min-top-k "${MIN_TOP_K}" \
  --max-top-k "${MAX_TOP_K}" \
  --chunk-mmr-fingerprint "${CHUNK_MMR_FINGERPRINT}" \
  --target-coverage "${TARGET_COVERAGE}" \
  --stop-gain-threshold "${STOP_GAIN_THRESHOLD}" \
  --insufficient-gain-threshold "${INSUFFICIENT_GAIN_THRESHOLD}" \
  "${SAMPLE_ARGS[@]}"

echo "[evidence-chain-graph-v0.7] done: ${OUTPUT_DIR}"
