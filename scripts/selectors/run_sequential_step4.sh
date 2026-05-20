#!/usr/bin/env bash
# Train and selection-only evaluate the Step4 Stage2 sequential pointer selector.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-/data/models/deberta-v3-base/}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/stage2_sentence_sequential/deberta_sequential_deep}"
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
SEMANTIC_FEATURE_PROFILE="${SEMANTIC_FEATURE_PROFILE:-deep}"
TARGETED_FEATURE_PROFILE="${TARGETED_FEATURE_PROFILE:-none}"
SHALLOW_FEATURE_PROFILE="${SHALLOW_FEATURE_PROFILE:-off}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

echo "[sequential-step4] model=${MODEL_NAME}"
echo "[sequential-step4] output=${OUTPUT_DIR}"
echo "[sequential-step4] train=${TRAIN_ORACLE_RESULTS}"
echo "[sequential-step4] val=${VAL_ORACLE_RESULTS}"
echo "[sequential-step4] nproc_per_node=${NPROC_PER_NODE}"
echo "[sequential-step4] semantic_feature_profile=${SEMANTIC_FEATURE_PROFILE}"
echo "[sequential-step4] targeted_feature_profile=${TARGETED_FEATURE_PROFILE}"
echo "[sequential-step4] shallow_feature_profile=${SHALLOW_FEATURE_PROFILE}"

TRAIN_CMD=(
    scripts/selectors/train_sequential_selector.py
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
    --semantic-feature-profile "${SEMANTIC_FEATURE_PROFILE}" \
    --targeted-feature-profile "${TARGETED_FEATURE_PROFILE}" \
    --shallow-feature-profile "${SHALLOW_FEATURE_PROFILE}" \
    "$@"
)

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_CMD[@]}"
else
    python "${TRAIN_CMD[@]}"
fi

python scripts/selectors/eval_sequential_selector.py \
    --model-dir "${OUTPUT_DIR}" \
    --oracle-results "${VAL_ORACLE_RESULTS}" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --filter-policy "${FILTER_POLICY}"
