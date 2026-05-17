#!/usr/bin/env bash
set -euo pipefail

# Build V1b true-side-anchor pointwise supervision, train the lightweight
# selector, then run build+vLLM inference with an existing verifier checkpoint.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

ORACLE_TRAIN="${ORACLE_TRAIN:-outputs/oracle_evidence/20260517_041502/oracle_results_train.jsonl}"
ORACLE_VAL="${ORACLE_VAL:-outputs/oracle_evidence/20260516_135632/oracle_results_val.jsonl}"
CONFIG="${CONFIG:-configs/experiment/b3_mmr_topk_sweep_1024.yaml}"
PIPELINE_EXPERIMENT="${PIPELINE_EXPERIMENT:-b3_pointwise_oracle_selector_v1b_1024}"

CHUNK_MMR_TRAIN="${CHUNK_MMR_TRAIN:-outputs/cache/chunk_mmr/e0b01520364d/train.pkl}"
CHUNK_MMR_VAL="${CHUNK_MMR_VAL:-outputs/cache/chunk_mmr/e0b01520364d/val.pkl}"

DATA_DIR="${DATA_DIR:-outputs/oracle_pointwise/v1b/data}"
MODEL_DIR="${MODEL_DIR:-outputs/oracle_pointwise/v1b/logreg}"
EVAL_DIR="${EVAL_DIR:-outputs/oracle_pointwise/v1b/logreg/eval_val_v1b}"

MOSTLY_TRUE_ANCHOR_WEIGHT="${MOSTLY_TRUE_ANCHOR_WEIGHT:-0.25}"
TRUE_ANCHOR_WEIGHT="${TRUE_ANCHOR_WEIGHT:-0.10}"
MAX_TRUE_SIDE_ANCHORS_PER_LABEL="${MAX_TRUE_SIDE_ANCHORS_PER_LABEL:-0}"

TOP_K="${TOP_K:-5}"
FALLBACK_POOL_SIZE="${FALLBACK_POOL_SIZE:-15}"
EPOCHS="${EPOCHS:-800}"
LR="${LR:-0.05}"
PATIENCE="${PATIENCE:-80}"

VERIFIER_TRAIN_RUN_DIR="${VERIFIER_TRAIN_RUN_DIR:-outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train}"
INFER_SPLIT="${INFER_SPLIT:-val}"
INFER_CHECKPOINT="${INFER_CHECKPOINT:-best}"
INFER_PORT="${INFER_PORT:-35021}"
PIPELINE_OUTPUT_SUBDIR="${PIPELINE_OUTPUT_SUBDIR:-pointwise_oracle_v1b_eval_${INFER_SPLIT}}"

SKIP_DATASET="${SKIP_DATASET:-false}"
SKIP_TRAIN="${SKIP_TRAIN:-false}"
SKIP_SELECTION_EVAL="${SKIP_SELECTION_EVAL:-false}"
SKIP_VLLM_INFER="${SKIP_VLLM_INFER:-false}"

echo "[pointwise_v1b] DATA_DIR=${DATA_DIR}"
echo "[pointwise_v1b] MODEL_DIR=${MODEL_DIR}"
echo "[pointwise_v1b] INFER_SPLIT=${INFER_SPLIT}"
echo "[pointwise_v1b] VERIFIER_TRAIN_RUN_DIR=${VERIFIER_TRAIN_RUN_DIR}"

if [[ "${SKIP_DATASET}" != "true" ]]; then
  python scripts/selectors/build_pointwise_oracle_dataset.py \
    --oracle-results "${ORACLE_TRAIN}" \
    --config "${CONFIG}" \
    --split train \
    --chunk-mmr-cache "${CHUNK_MMR_TRAIN}" \
    --output-dir "${DATA_DIR}" \
    --top-k "${TOP_K}" \
    --filter-preset v1b \
    --fallback-pool-size "${FALLBACK_POOL_SIZE}" \
    --mostly-true-anchor-weight "${MOSTLY_TRUE_ANCHOR_WEIGHT}" \
    --true-anchor-weight "${TRUE_ANCHOR_WEIGHT}" \
    --max-true-side-anchors-per-label "${MAX_TRUE_SIDE_ANCHORS_PER_LABEL}"
fi

if [[ "${SKIP_TRAIN}" != "true" ]]; then
  python scripts/selectors/train_pointwise_oracle_selector.py \
    --train-jsonl "${DATA_DIR}/train_pointwise.jsonl" \
    --feature-schema "${DATA_DIR}/feature_schema.json" \
    --output-dir "${MODEL_DIR}" \
    --model logreg \
    --top-k "${TOP_K}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --patience "${PATIENCE}"
fi

if [[ "${SKIP_SELECTION_EVAL}" != "true" ]]; then
  python scripts/selectors/eval_pointwise_oracle_selector.py \
    --model-dir "${MODEL_DIR}" \
    --oracle-results "${ORACLE_VAL}" \
    --config "${CONFIG}" \
    --split val \
    --chunk-mmr-cache "${CHUNK_MMR_VAL}" \
    --output-dir "${EVAL_DIR}" \
    --filter-preset v1b \
    --top-k "${TOP_K}"
fi

if [[ "${SKIP_VLLM_INFER}" != "true" ]]; then
  python -m fact_checking.pipeline.run \
    "experiment=${PIPELINE_EXPERIMENT}" \
    "pipeline.steps=[build,infer]" \
    "pipeline.output_subdir=${PIPELINE_OUTPUT_SUBDIR}" \
    "build.retrieval.pointwise_oracle.model_dir=${MODEL_DIR}" \
    "train.run_dir=\"${VERIFIER_TRAIN_RUN_DIR}\"" \
    "infer.split=${INFER_SPLIT}" \
    "infer.checkpoint=${INFER_CHECKPOINT}" \
    "infer.port=${INFER_PORT}"
fi
