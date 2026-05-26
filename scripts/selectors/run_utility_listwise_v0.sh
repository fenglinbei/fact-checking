#!/usr/bin/env bash
# Train v0 frozen-encoder utility listwise scorer from step-0 VIG rows.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-/data/models/deberta-v3-base/}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/utility_listwise/deberta_v0_step0_static}"
TRAIN_VIG_CACHE="${TRAIN_VIG_CACHE:-outputs/selectors/vig_utility/saved_step_train/vig_records_train.jsonl}"
VAL_VIG_CACHE="${VAL_VIG_CACHE:-outputs/selectors/vig_utility/saved_step_val/vig_records_val.jsonl}"
TRAIN_ORACLE_RESULTS="${TRAIN_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl}"
VAL_ORACLE_RESULTS="${VAL_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
MAX_LENGTH="${MAX_LENGTH:-384}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-1.0}"
SOFT_CE_WEIGHT="${SOFT_CE_WEIGHT:-0.2}"
BCE_WEIGHT="${BCE_WEIGHT:-0.2}"
SOFT_TAU="${SOFT_TAU:-0.3}"
POSITIVE_BEST_MARGIN="${POSITIVE_BEST_MARGIN:-0.05}"
EVAL_EVERY="${EVAL_EVERY:-100}"
TRAIN_SAMPLE_LIMIT="${TRAIN_SAMPLE_LIMIT:-}"
VAL_SAMPLE_LIMIT="${VAL_SAMPLE_LIMIT:-}"
NO_PROGRESS="${NO_PROGRESS:-false}"

echo "[utility-listwise-v0] model=${MODEL_NAME}"
echo "[utility-listwise-v0] output=${OUTPUT_DIR}"
echo "[utility-listwise-v0] train_vig=${TRAIN_VIG_CACHE}"
echo "[utility-listwise-v0] val_vig=${VAL_VIG_CACHE}"
echo "[utility-listwise-v0] train_oracle=${TRAIN_ORACLE_RESULTS}"
echo "[utility-listwise-v0] val_oracle=${VAL_ORACLE_RESULTS}"
echo "[utility-listwise-v0] frozen_encoder=true"

cmd=(
  python scripts/selectors/train_utility_listwise_selector.py
  --model-name "${MODEL_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --train-vig-cache "${TRAIN_VIG_CACHE}"
  --val-vig-cache "${VAL_VIG_CACHE}"
  --train-oracle-results "${TRAIN_ORACLE_RESULTS}"
  --val-oracle-results "${VAL_ORACLE_RESULTS}"
  --max-length "${MAX_LENGTH}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --learning-rate "${LR}"
  --pairwise-weight "${PAIRWISE_WEIGHT}"
  --soft-ce-weight "${SOFT_CE_WEIGHT}"
  --bce-weight "${BCE_WEIGHT}"
  --soft-tau "${SOFT_TAU}"
  --positive-best-margin "${POSITIVE_BEST_MARGIN}"
  --eval-every "${EVAL_EVERY}"
)

if [[ -n "${TRAIN_SAMPLE_LIMIT}" ]]; then
  cmd+=(--train-sample-limit "${TRAIN_SAMPLE_LIMIT}")
fi
if [[ -n "${VAL_SAMPLE_LIMIT}" ]]; then
  cmd+=(--val-sample-limit "${VAL_SAMPLE_LIMIT}")
fi
if [[ "${NO_PROGRESS}" == "true" || "${NO_PROGRESS}" == "1" ]]; then
  cmd+=(--no-progress)
fi

"${cmd[@]}" "$@"

