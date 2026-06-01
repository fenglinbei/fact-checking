#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
INPUT="${INPUT:-outputs/selectors/evidence_map_selector/v0_6b_${SPLIT}/candidate_evidence_map_features_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/evidence_chain_graph/v0_6d_sufficiency_contradiction_${SPLIT}}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
MIN_TOP_K="${MIN_TOP_K:-5}"
MAX_TOP_K="${MAX_TOP_K:-10}"
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
IMPORTANT_ATOM_THRESHOLD="${IMPORTANT_ATOM_THRESHOLD:-0.50}"
SUFFICIENCY_WEIGHTED_COVERAGE_THRESHOLD="${SUFFICIENCY_WEIGHTED_COVERAGE_THRESHOLD:-0.80}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

SAMPLE_ARGS=()
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  SAMPLE_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

echo "[evidence-chain-graph-v0.6d] split       : ${SPLIT}"
echo "[evidence-chain-graph-v0.6d] input       : ${INPUT}"
echo "[evidence-chain-graph-v0.6d] output      : ${OUTPUT_DIR}"
echo "[evidence-chain-graph-v0.6d] top_n/min/max: ${CANDIDATE_TOP_N}/${MIN_TOP_K}/${MAX_TOP_K}"
echo "[evidence-chain-graph-v0.6d] fingerprint : ${CHUNK_MMR_FINGERPRINT}"
echo "[evidence-chain-graph-v0.6d] sufficiency : ${IMPORTANT_ATOM_THRESHOLD}/${SUFFICIENCY_WEIGHTED_COVERAGE_THRESHOLD}"

PYTHONPATH=src python scripts/phase5_selectors/build/build_evidence_chain_graph_v0_6d.py \
  --input "${INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --candidate-top-n "${CANDIDATE_TOP_N}" \
  --min-top-k "${MIN_TOP_K}" \
  --max-top-k "${MAX_TOP_K}" \
  --chunk-mmr-fingerprint "${CHUNK_MMR_FINGERPRINT}" \
  --important-atom-threshold "${IMPORTANT_ATOM_THRESHOLD}" \
  --sufficiency-weighted-coverage-threshold "${SUFFICIENCY_WEIGHTED_COVERAGE_THRESHOLD}" \
  "${SAMPLE_ARGS[@]}"

echo "[evidence-chain-graph-v0.6d] done: ${OUTPUT_DIR}"
