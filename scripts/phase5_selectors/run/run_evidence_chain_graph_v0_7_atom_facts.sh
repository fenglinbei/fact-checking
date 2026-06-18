#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT="${INPUT:-outputs/selectors/evidence_map_selector/v0_7_atom_facts_${SPLIT}/candidate_evidence_map_features_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/evidence_chain_graph/v0_7_atom_facts_budgeted_marginal_adaptive5_10_${SPLIT}}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
MIN_TOP_K="${MIN_TOP_K:-5}"
MAX_TOP_K="${MAX_TOP_K:-10}"
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
TARGET_COVERAGE="${TARGET_COVERAGE:-0.80}"
STOP_GAIN_THRESHOLD="${STOP_GAIN_THRESHOLD:-0.10}"
INSUFFICIENT_GAIN_THRESHOLD="${INSUFFICIENT_GAIN_THRESHOLD:-0.05}"
OBJECTIVE_COVERAGE="${OBJECTIVE_COVERAGE:-}"
OBJECTIVE_MAP_QUALITY="${OBJECTIVE_MAP_QUALITY:-}"
OBJECTIVE_BASE_SCORE="${OBJECTIVE_BASE_SCORE:-}"
OBJECTIVE_KEY_SPAN="${OBJECTIVE_KEY_SPAN:-}"
OBJECTIVE_COMPLEMENTS="${OBJECTIVE_COMPLEMENTS:-}"
OBJECTIVE_CORROBORATES="${OBJECTIVE_CORROBORATES:-}"
OBJECTIVE_CONDITIONAL_TENSION="${OBJECTIVE_CONDITIONAL_TENSION:-}"
OBJECTIVE_BRIDGE_CONTEXT="${OBJECTIVE_BRIDGE_CONTEXT:-}"
OBJECTIVE_DUPLICATE_REPEAT="${OBJECTIVE_DUPLICATE_REPEAT:-}"
OBJECTIVE_BACKGROUND_OR_IRRELEVANT="${OBJECTIVE_BACKGROUND_OR_IRRELEVANT:-}"
OBJECTIVE_SAME_SOURCE_EXCESS_AFTER_TWO="${OBJECTIVE_SAME_SOURCE_EXCESS_AFTER_TWO:-}"
OBJECTIVE_LENGTH="${OBJECTIVE_LENGTH:-}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

SAMPLE_ARGS=()
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  SAMPLE_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

OBJECTIVE_ARGS=()
if [[ -n "${OBJECTIVE_COVERAGE}" ]]; then
  OBJECTIVE_ARGS+=(--objective-coverage "${OBJECTIVE_COVERAGE}")
fi
if [[ -n "${OBJECTIVE_MAP_QUALITY}" ]]; then
  OBJECTIVE_ARGS+=(--objective-map-quality "${OBJECTIVE_MAP_QUALITY}")
fi
if [[ -n "${OBJECTIVE_BASE_SCORE}" ]]; then
  OBJECTIVE_ARGS+=(--objective-base-score "${OBJECTIVE_BASE_SCORE}")
fi
if [[ -n "${OBJECTIVE_KEY_SPAN}" ]]; then
  OBJECTIVE_ARGS+=(--objective-key-span "${OBJECTIVE_KEY_SPAN}")
fi
if [[ -n "${OBJECTIVE_COMPLEMENTS}" ]]; then
  OBJECTIVE_ARGS+=(--objective-complements "${OBJECTIVE_COMPLEMENTS}")
fi
if [[ -n "${OBJECTIVE_CORROBORATES}" ]]; then
  OBJECTIVE_ARGS+=(--objective-corroborates "${OBJECTIVE_CORROBORATES}")
fi
if [[ -n "${OBJECTIVE_CONDITIONAL_TENSION}" ]]; then
  OBJECTIVE_ARGS+=(--objective-conditional-tension "${OBJECTIVE_CONDITIONAL_TENSION}")
fi
if [[ -n "${OBJECTIVE_BRIDGE_CONTEXT}" ]]; then
  OBJECTIVE_ARGS+=(--objective-bridge-context "${OBJECTIVE_BRIDGE_CONTEXT}")
fi
if [[ -n "${OBJECTIVE_DUPLICATE_REPEAT}" ]]; then
  OBJECTIVE_ARGS+=(--objective-duplicate-repeat "${OBJECTIVE_DUPLICATE_REPEAT}")
fi
if [[ -n "${OBJECTIVE_BACKGROUND_OR_IRRELEVANT}" ]]; then
  OBJECTIVE_ARGS+=(--objective-background-or-irrelevant "${OBJECTIVE_BACKGROUND_OR_IRRELEVANT}")
fi
if [[ -n "${OBJECTIVE_SAME_SOURCE_EXCESS_AFTER_TWO}" ]]; then
  OBJECTIVE_ARGS+=(--objective-same-source-excess-after-two "${OBJECTIVE_SAME_SOURCE_EXCESS_AFTER_TWO}")
fi
if [[ -n "${OBJECTIVE_LENGTH}" ]]; then
  OBJECTIVE_ARGS+=(--objective-length "${OBJECTIVE_LENGTH}")
fi

echo "[evidence-chain-graph-v0.7-atom-facts] split        : ${SPLIT}"
echo "[evidence-chain-graph-v0.7-atom-facts] input        : ${INPUT}"
echo "[evidence-chain-graph-v0.7-atom-facts] output       : ${OUTPUT_DIR}"
echo "[evidence-chain-graph-v0.7-atom-facts] top_n/min/max: ${CANDIDATE_TOP_N}/${MIN_TOP_K}/${MAX_TOP_K}"
echo "[evidence-chain-graph-v0.7-atom-facts] fingerprint  : ${CHUNK_MMR_FINGERPRINT}"
echo "[evidence-chain-graph-v0.7-atom-facts] stop         : ${TARGET_COVERAGE}/${STOP_GAIN_THRESHOLD}/${INSUFFICIENT_GAIN_THRESHOLD}"
echo "[evidence-chain-graph-v0.7-atom-facts] objective    : background=${OBJECTIVE_BACKGROUND_OR_IRRELEVANT:-default} length=${OBJECTIVE_LENGTH:-default}"

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
  "${OBJECTIVE_ARGS[@]}" \
  "${SAMPLE_ARGS[@]}"

echo "[evidence-chain-graph-v0.7-atom-facts] trace: ${OUTPUT_DIR}/selection_trace_${SPLIT}.jsonl"
echo "[evidence-chain-graph-v0.7-atom-facts] done: ${OUTPUT_DIR}"
