#!/usr/bin/env bash
set -euo pipefail

# Evaluate candidate-pool recall for an oracle result directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

ORACLE_DIR="${ORACLE_DIR:?ORACLE_DIR is required, e.g. outputs/oracle_evidence/rawfc_qwen3_dense_fullpool_margin}"
RUN_NAME="${RUN_NAME:-$(basename "${ORACLE_DIR}")}"
SPLITS="${SPLITS:-val test}"
TOP_N="${TOP_N:-20 32}"
SELECTION_MODE="${SELECTION_MODE:-mmr}"
MMR_LAMBDA="${MMR_LAMBDA:-0.70}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/retrieval_signal_ablation}"

echo "[dense-recall] oracle dir    : ${ORACLE_DIR}"
echo "[dense-recall] run name      : ${RUN_NAME}"
echo "[dense-recall] splits        : ${SPLITS}"
echo "[dense-recall] top_n         : ${TOP_N}"
echo "[dense-recall] selection mode: ${SELECTION_MODE}"

for split in ${SPLITS}; do
  oracle_results="${ORACLE_DIR}/oracle_results_${split}.jsonl"
  if [[ ! -s "${oracle_results}" ]]; then
    echo "[dense-recall] missing oracle results for split=${split}: ${oracle_results}" >&2
    exit 1
  fi
  PYTHONPATH=src python scripts/phase3_oracle_evidence/evaluate_candidate_pool_recall.py \
    --oracle-results "${oracle_results}" \
    --top-n ${TOP_N} \
    --selection-mode "${SELECTION_MODE}" \
    --mmr-lambda "${MMR_LAMBDA}" \
    --output "${OUTPUT_ROOT}/${RUN_NAME}_${split}_${SELECTION_MODE}.json" \
    --csv-output "${OUTPUT_ROOT}/${RUN_NAME}_${split}_${SELECTION_MODE}.csv"
done

echo "[dense-recall] done"
