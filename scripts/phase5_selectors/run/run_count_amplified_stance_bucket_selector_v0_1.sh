#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
INPUT_DIR="${INPUT_DIR:-outputs/selectors/count_amplified_stance_bucket_selector/v0_${SPLIT}}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/count_amplified_stance_bucket_selector/v0_1_${SPLIT}}"

EVAL_N_BUCKETS="${EVAL_N_BUCKETS:-3 5 7}"
TOP_K="${TOP_K:-5}"
ALPHA="${ALPHA:-0.5}"
GAMMA_VALUES="${GAMMA_VALUES:-0.6,0.8,1.0}"
PRIMARY_GAMMA="${PRIMARY_GAMMA:-0.8}"
RHO="${RHO:-2.0}"
AMBIGUOUS_BUCKET_PENALTY="${AMBIGUOUS_BUCKET_PENALTY:-0.6}"
TAU_C="${TAU_C:-0.50}"
TAU_R="${TAU_R:-0.15}"
MIN_BUCKET_MEMBERSHIP="${MIN_BUCKET_MEMBERSHIP:-}"

MIN_BUCKET_ARGS=()
if [[ -n "${MIN_BUCKET_MEMBERSHIP}" ]]; then
  MIN_BUCKET_ARGS=(--min-bucket-membership "${MIN_BUCKET_MEMBERSHIP}")
fi

echo "[count-amplified-v0.1] split       : ${SPLIT}"
echo "[count-amplified-v0.1] input dir   : ${INPUT_DIR}"
echo "[count-amplified-v0.1] output dir  : ${OUTPUT_DIR}"
echo "[count-amplified-v0.1] gamma       : ${GAMMA_VALUES}"
echo "[count-amplified-v0.1] primary     : ${PRIMARY_GAMMA}"
echo "[count-amplified-v0.1] rho         : ${RHO}"
echo "[count-amplified-v0.1] amb penalty : ${AMBIGUOUS_BUCKET_PENALTY}"

mkdir -p "${OUTPUT_DIR}"

for n in ${EVAL_N_BUCKETS}; do
  if [[ "${n}" == "3" ]]; then
    BUCKET_FILE="${INPUT_DIR}/candidate_stance_buckets_${SPLIT}.jsonl"
  else
    BUCKET_FILE="${INPUT_DIR}/candidate_stance_buckets_n${n}_${SPLIT}.jsonl"
  fi
  if [[ ! -s "${BUCKET_FILE}" ]]; then
    echo "[count-amplified-v0.1] missing bucket file: ${BUCKET_FILE}" >&2
    exit 1
  fi
  EVAL_DIR="${OUTPUT_DIR}/eval_n${n}"
  PYTHONPATH=src python scripts/phase5_selectors/eval/eval_count_amplified_stance_bucket_selector.py \
    --candidate-stance-buckets "${BUCKET_FILE}" \
    --output-dir "${EVAL_DIR}" \
    --split "${SPLIT}" \
    --top-k "${TOP_K}" \
    --alpha "${ALPHA}" \
    --gamma-values "${GAMMA_VALUES}" \
    --primary-gamma "${PRIMARY_GAMMA}" \
    --rho "${RHO}" \
    --ambiguous-bucket-penalty "${AMBIGUOUS_BUCKET_PENALTY}" \
    --tau-c "${TAU_C}" \
    --tau-r "${TAU_R}" \
    "${MIN_BUCKET_ARGS[@]}"
done

echo "[count-amplified-v0.1] done: ${OUTPUT_DIR}"
