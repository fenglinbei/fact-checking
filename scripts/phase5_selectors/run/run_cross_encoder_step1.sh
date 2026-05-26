#!/usr/bin/env bash
# Train and selection-only evaluate the Step1 Stage2 cross-encoder selector.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-/data/models/bge-reranker-large}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/stage2_sentence_cross_encoder/bge_reranker_large_pairwise}"
TRAIN_ORACLE_RESULTS="${TRAIN_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl}"
VAL_ORACLE_RESULTS="${VAL_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/eval_val}"
MAX_LENGTH="${MAX_LENGTH:-384}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-5}"
FILTER_POLICY="${FILTER_POLICY:-all}"

echo "[cross-encoder-step1] model=${MODEL_NAME}"
echo "[cross-encoder-step1] output=${OUTPUT_DIR}"
echo "[cross-encoder-step1] train=${TRAIN_ORACLE_RESULTS}"
echo "[cross-encoder-step1] val=${VAL_ORACLE_RESULTS}"

python scripts/phase5_selectors/train/train_cross_encoder_pairwise.py \
    --model-name "${MODEL_NAME}" \
    --train-oracle-results "${TRAIN_ORACLE_RESULTS}" \
    --val-oracle-results "${VAL_ORACLE_RESULTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --learning-rate "${LR}" \
    --filter-policy "${FILTER_POLICY}" \
    "$@"

python scripts/phase5_selectors/eval/eval_cross_encoder_selector.py \
    --model-dir "${OUTPUT_DIR}" \
    --oracle-results "${VAL_ORACLE_RESULTS}" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --filter-policy "${FILTER_POLICY}"

