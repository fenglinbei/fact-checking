#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
N_STANCE_BUCKETS="${N_STANCE_BUCKETS:-7}"
if [[ -z "${INPUT_BUCKET_FILE:-}" ]]; then
  if [[ "${N_STANCE_BUCKETS}" == "3" ]]; then
    INPUT_BUCKET_FILE="outputs/selectors/count_amplified_stance_bucket_selector/v0_2_${SPLIT}/candidate_stance_buckets_v02_${SPLIT}.jsonl"
  else
    INPUT_BUCKET_FILE="outputs/selectors/count_amplified_stance_bucket_selector/v0_2_${SPLIT}/candidate_stance_buckets_v02_n${N_STANCE_BUCKETS}_${SPLIT}.jsonl"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/oracle_likelihood_constrained_selector/v0_3_${SPLIT}}"

SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
FOLDS="${FOLDS:-5}"
CROSS_FIT_FOLDS="${CROSS_FIT_FOLDS:-${FOLDS}}"
FEATURE_SET="${FEATURE_SET:-all_features}"
OBJECTIVE="${OBJECTIVE:-pointwise}"
SEED="${SEED:-20260527}"
EPOCHS="${EPOCHS:-800}"
LR="${LR:-0.05}"
L2="${L2:-0.0001}"
PATIENCE="${PATIENCE:-80}"
EVAL_EVERY="${EVAL_EVERY:-10}"
DEV_FRACTION="${DEV_FRACTION:-0.1}"
TOP_K="${TOP_K:-5}"
ANCHOR_K="${ANCHOR_K:-2}"
SOURCE_PENALTY="${SOURCE_PENALTY:-0.10}"
STANCE_REGION_PENALTY="${STANCE_REGION_PENALTY:-0.04}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

SAMPLE_ARGS=()
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  SAMPLE_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

echo "[oracle-likelihood-v0.3] split       : ${SPLIT}"
echo "[oracle-likelihood-v0.3] buckets     : ${INPUT_BUCKET_FILE}"
echo "[oracle-likelihood-v0.3] output dir  : ${OUTPUT_DIR}"
echo "[oracle-likelihood-v0.3] folds       : ${CROSS_FIT_FOLDS}"
echo "[oracle-likelihood-v0.3] feature set : ${FEATURE_SET}"
echo "[oracle-likelihood-v0.3] objective   : ${OBJECTIVE}"
echo "[oracle-likelihood-v0.3] anchor_k    : ${ANCHOR_K}"
echo "[oracle-likelihood-v0.3] source pen  : ${SOURCE_PENALTY}"
echo "[oracle-likelihood-v0.3] stance pen  : ${STANCE_REGION_PENALTY}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${RUN_TRAIN}" == "1" || "${RUN_TRAIN}" == "true" || "${RUN_TRAIN}" == "True" ]]; then
  PYTHONPATH=src python scripts/phase5_selectors/train/train_oracle_likelihood_constrained_selector.py \
    --candidate-stance-buckets "${INPUT_BUCKET_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --cross-fit-folds "${CROSS_FIT_FOLDS}" \
    --feature-set "${FEATURE_SET}" \
    --objective "${OBJECTIVE}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --l2 "${L2}" \
    --patience "${PATIENCE}" \
    --eval-every "${EVAL_EVERY}" \
    --dev-fraction "${DEV_FRACTION}" \
    "${SAMPLE_ARGS[@]}"
fi

if [[ "${RUN_EVAL}" == "1" || "${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "True" ]]; then
  SCORED_FILE="${OUTPUT_DIR}/candidate_oracle_likelihood_scores_${SPLIT}.jsonl"
  if [[ ! -s "${SCORED_FILE}" ]]; then
    echo "[oracle-likelihood-v0.3] missing scored file: ${SCORED_FILE}" >&2
    exit 1
  fi
  PYTHONPATH=src python scripts/phase5_selectors/eval/eval_oracle_likelihood_constrained_selector.py \
    --scored-candidates "${SCORED_FILE}" \
    --output-dir "${OUTPUT_DIR}/eval" \
    --split "${SPLIT}" \
    --top-k "${TOP_K}" \
    --anchor-k "${ANCHOR_K}" \
    --source-penalty "${SOURCE_PENALTY}" \
    --stance-region-penalty "${STANCE_REGION_PENALTY}" \
    "${SAMPLE_ARGS[@]}"
fi

echo "[oracle-likelihood-v0.3] done: ${OUTPUT_DIR}"
