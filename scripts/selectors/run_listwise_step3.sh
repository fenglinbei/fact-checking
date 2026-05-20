#!/usr/bin/env bash
# Train and selection-only evaluate the Step3 Stage2 set-aware listwise selector.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-microsoft/deberta-v3-base}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/stage2_sentence_listwise/deberta_listwise}"
TRAIN_ORACLE_RESULTS="${TRAIN_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl}"
VAL_ORACLE_RESULTS="${VAL_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/eval_val}"
MAX_LENGTH="${MAX_LENGTH:-384}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-5}"
HEAD_LR="${HEAD_LR:-1e-4}"
FILTER_POLICY="${FILTER_POLICY:-all}"
SHUFFLE_PROBABILITY="${SHUFFLE_PROBABILITY:-0.0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

echo "[listwise-step3] model=${MODEL_NAME}"
echo "[listwise-step3] output=${OUTPUT_DIR}"
echo "[listwise-step3] train=${TRAIN_ORACLE_RESULTS}"
echo "[listwise-step3] val=${VAL_ORACLE_RESULTS}"
echo "[listwise-step3] nproc_per_node=${NPROC_PER_NODE}"

TRAIN_CMD=(
    scripts/selectors/train_listwise_selector.py
    --model-name "${MODEL_NAME}" \
    --train-oracle-results "${TRAIN_ORACLE_RESULTS}" \
    --val-oracle-results "${VAL_ORACLE_RESULTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --learning-rate "${LR}" \
    --head-learning-rate "${HEAD_LR}" \
    --filter-policy "${FILTER_POLICY}" \
    --shuffle-probability "${SHUFFLE_PROBABILITY}" \
    "$@"
)

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_CMD[@]}"
else
    python "${TRAIN_CMD[@]}"
fi

python scripts/selectors/eval_listwise_selector.py \
    --model-dir "${OUTPUT_DIR}" \
    --oracle-results "${VAL_ORACLE_RESULTS}" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --filter-policy "${FILTER_POLICY}"
