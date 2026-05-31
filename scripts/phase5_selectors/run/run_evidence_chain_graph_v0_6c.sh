#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
INPUT="${INPUT:-outputs/selectors/evidence_map_selector/v0_6b_${SPLIT}/candidate_evidence_map_features_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/evidence_chain_graph/v0_6c_adaptive5_10_${SPLIT}}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
MIN_TOP_K="${MIN_TOP_K:-5}"
MAX_TOP_K="${MAX_TOP_K:-10}"
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

SAMPLE_ARGS=()
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  SAMPLE_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

echo "[evidence-chain-graph-v0.6c] split       : ${SPLIT}"
echo "[evidence-chain-graph-v0.6c] input       : ${INPUT}"
echo "[evidence-chain-graph-v0.6c] output      : ${OUTPUT_DIR}"
echo "[evidence-chain-graph-v0.6c] top_n/min/max: ${CANDIDATE_TOP_N}/${MIN_TOP_K}/${MAX_TOP_K}"
echo "[evidence-chain-graph-v0.6c] fingerprint : ${CHUNK_MMR_FINGERPRINT}"

PYTHONPATH=src python scripts/phase5_selectors/build/build_evidence_chain_graph_v0_6c.py \
  --input "${INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --candidate-top-n "${CANDIDATE_TOP_N}" \
  --min-top-k "${MIN_TOP_K}" \
  --max-top-k "${MAX_TOP_K}" \
  --chunk-mmr-fingerprint "${CHUNK_MMR_FINGERPRINT}" \
  "${SAMPLE_ARGS[@]}"

echo "[evidence-chain-graph-v0.6c] done: ${OUTPUT_DIR}"
