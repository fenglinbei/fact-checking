#!/usr/bin/env bash
# Train and selection-only evaluate the Step4 Stage2 sequential pointer selector.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_CUMEM_HOST_ENABLE=0

MODEL_NAME="${MODEL_NAME:-/data/models/deberta-v3-base/}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/stage2_sentence_sequential/deberta_sequential_deep}"
TRAIN_ORACLE_RESULTS="${TRAIN_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl}"
VAL_ORACLE_RESULTS="${VAL_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/eval_val}"
MAX_LENGTH="${MAX_LENGTH:-384}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-6}"
LR="${LR:-2e-5}"
HEAD_LR="${HEAD_LR:-1e-4}"
SEQ_LOSS_WEIGHT="${SEQ_LOSS_WEIGHT:-1.0}"
MASK_LOSS_WEIGHT="${MASK_LOSS_WEIGHT:-0.0}"
FILTER_POLICY="${FILTER_POLICY:-all}"
SEMANTIC_FEATURE_PROFILE="${SEMANTIC_FEATURE_PROFILE:-deep}"
TARGETED_FEATURE_PROFILE="${TARGETED_FEATURE_PROFILE:-none}"
SHALLOW_FEATURE_PROFILE="${SHALLOW_FEATURE_PROFILE:-off}"
PROJECTION_MODE="${PROJECTION_MODE:-linear}"
PROJECTION_HIDDEN_MULTIPLIER="${PROJECTION_HIDDEN_MULTIPLIER:-2}"
CLAIM_START_MODE="${CLAIM_START_MODE:-learned}"
CLAIM_FEATURE_MODE="${CLAIM_FEATURE_MODE:-off}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-stage2-sequential}"
SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-$(basename "${OUTPUT_DIR}")}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"
SWANLAB_MODE="${SWANLAB_MODE:-}"
SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-}"
SWANLAB_TAGS="${SWANLAB_TAGS:-selector,sequential,step4}"
SWANLAB_DESCRIPTION="${SWANLAB_DESCRIPTION:-}"
SWANLAB_DISABLED="${SWANLAB_DISABLED:-0}"

echo "[sequential-step4] model=${MODEL_NAME}"
echo "[sequential-step4] output=${OUTPUT_DIR}"
echo "[sequential-step4] train=${TRAIN_ORACLE_RESULTS}"
echo "[sequential-step4] val=${VAL_ORACLE_RESULTS}"
echo "[sequential-step4] nproc_per_node=${NPROC_PER_NODE}"
echo "[sequential-step4] semantic_feature_profile=${SEMANTIC_FEATURE_PROFILE}"
echo "[sequential-step4] targeted_feature_profile=${TARGETED_FEATURE_PROFILE}"
echo "[sequential-step4] shallow_feature_profile=${SHALLOW_FEATURE_PROFILE}"
echo "[sequential-step4] projection_mode=${PROJECTION_MODE}"
echo "[sequential-step4] projection_hidden_multiplier=${PROJECTION_HIDDEN_MULTIPLIER}"
echo "[sequential-step4] claim_start_mode=${CLAIM_START_MODE}"
echo "[sequential-step4] claim_feature_mode=${CLAIM_FEATURE_MODE}"
echo "[sequential-step4] seq_loss_weight=${SEQ_LOSS_WEIGHT}"
echo "[sequential-step4] mask_loss_weight=${MASK_LOSS_WEIGHT}"
echo "[sequential-step4] swanlab_project=${SWANLAB_PROJECT}"
echo "[sequential-step4] swanlab_experiment_name=${SWANLAB_EXPERIMENT_NAME}"

TRAIN_CMD=(
    scripts/phase5_selectors/train/train_sequential_selector.py
    --model-name "${MODEL_NAME}" \
    --train-oracle-results "${TRAIN_ORACLE_RESULTS}" \
    --val-oracle-results "${VAL_ORACLE_RESULTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --learning-rate "${LR}" \
    --head-learning-rate "${HEAD_LR}" \
    --seq-loss-weight "${SEQ_LOSS_WEIGHT}" \
    --mask-loss-weight "${MASK_LOSS_WEIGHT}" \
    --filter-policy "${FILTER_POLICY}" \
    --semantic-feature-profile "${SEMANTIC_FEATURE_PROFILE}" \
    --targeted-feature-profile "${TARGETED_FEATURE_PROFILE}" \
    --shallow-feature-profile "${SHALLOW_FEATURE_PROFILE}" \
    --projection-mode "${PROJECTION_MODE}" \
    --projection-hidden-multiplier "${PROJECTION_HIDDEN_MULTIPLIER}" \
    --claim-start-mode "${CLAIM_START_MODE}" \
    --claim-feature-mode "${CLAIM_FEATURE_MODE}" \
    --swanlab-project "${SWANLAB_PROJECT}" \
    --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME}" \
    --swanlab-tags "${SWANLAB_TAGS}" \
    "$@"
)

if [[ -n "${SWANLAB_WORKSPACE}" ]]; then
    TRAIN_CMD+=(--swanlab-workspace "${SWANLAB_WORKSPACE}")
fi
if [[ -n "${SWANLAB_MODE}" ]]; then
    TRAIN_CMD+=(--swanlab-mode "${SWANLAB_MODE}")
fi
if [[ -n "${SWANLAB_LOGDIR}" ]]; then
    TRAIN_CMD+=(--swanlab-logdir "${SWANLAB_LOGDIR}")
fi
if [[ -n "${SWANLAB_DESCRIPTION}" ]]; then
    TRAIN_CMD+=(--swanlab-description "${SWANLAB_DESCRIPTION}")
fi
if [[ "${SWANLAB_DISABLED}" == "1" || "${SWANLAB_DISABLED}" == "true" || "${SWANLAB_DISABLED}" == "TRUE" ]]; then
    TRAIN_CMD+=(--no-swanlab)
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_CMD[@]}"
else
    python "${TRAIN_CMD[@]}"
fi

python scripts/phase5_selectors/eval/eval_sequential_selector.py \
    --model-dir "${OUTPUT_DIR}" \
    --oracle-results "${VAL_ORACLE_RESULTS}" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --filter-policy "${FILTER_POLICY}"
