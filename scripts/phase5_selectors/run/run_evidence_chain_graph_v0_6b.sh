#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
INPUT="${INPUT:-outputs/selectors/evidence_map_selector/v0_6b_${SPLIT}/candidate_evidence_map_features_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/evidence_chain_graph/v0_6b_${SPLIT}}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
TOP_K="${TOP_K:-5}"
BEAM_SIZE="${BEAM_SIZE:-12}"
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

SAMPLE_ARGS=()
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  SAMPLE_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

echo "[evidence-chain-graph-v0.6b] split       : ${SPLIT}"
echo "[evidence-chain-graph-v0.6b] input       : ${INPUT}"
echo "[evidence-chain-graph-v0.6b] output      : ${OUTPUT_DIR}"
echo "[evidence-chain-graph-v0.6b] top_n/top_k : ${CANDIDATE_TOP_N}/${TOP_K}"
echo "[evidence-chain-graph-v0.6b] beam        : ${BEAM_SIZE}"
echo "[evidence-chain-graph-v0.6b] fingerprint : ${CHUNK_MMR_FINGERPRINT}"

PYTHONPATH=src python scripts/phase5_selectors/build/build_evidence_chain_graph_v0_6b.py \
  --input "${INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --candidate-top-n "${CANDIDATE_TOP_N}" \
  --top-k "${TOP_K}" \
  --beam-size "${BEAM_SIZE}" \
  --chunk-mmr-fingerprint "${CHUNK_MMR_FINGERPRINT}" \
  "${SAMPLE_ARGS[@]}"

echo "[evidence-chain-graph-v0.6b] done: ${OUTPUT_DIR}"
