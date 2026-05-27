#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
INPUT_DIR="${INPUT_DIR:-outputs/selectors/count_amplified_stance_bucket_selector/v0_${SPLIT}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/count_amplified_stance_bucket_selector/v0_2_${SPLIT}}"

UNION_POOL="${UNION_POOL:-${INPUT_DIR}/union_analysis_candidate_pool_${SPLIT}.jsonl}"
QUALITY_LABELS="${QUALITY_LABELS:-${INPUT_DIR}/candidate_quality_labels_${SPLIT}.jsonl}"
ANNOTATIONS="${ANNOTATIONS:-${OUTPUT_DIR}/deepseek_teacher_annotations_v02_${SPLIT}.jsonl}"

SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
RUN_ANNOTATE="${RUN_ANNOTATE:-auto}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
NO_PROGRESS="${NO_PROGRESS:-0}"

TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://api.deepseek.com}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
TEACHER_CONCURRENCY="${TEACHER_CONCURRENCY:-8}"
TEACHER_RPM="${TEACHER_RPM:-120}"
TEACHER_TPM="${TEACHER_TPM:-200000}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-5}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-192}"
TEACHER_TOP_LOGPROBS="${TEACHER_TOP_LOGPROBS:-20}"
TEACHER_FALLBACK_TOP_LOGPROBS="${TEACHER_FALLBACK_TOP_LOGPROBS:-5}"
TEACHER_THINKING_TYPE="${TEACHER_THINKING_TYPE:-disabled}"

N_STANCE_BUCKETS="${N_STANCE_BUCKETS:-3,5,7}"
EVAL_N_BUCKETS="${EVAL_N_BUCKETS:-3 5 7}"
BUCKET_TAU="${BUCKET_TAU:-2.0}"
TOP_K="${TOP_K:-5}"
ALPHA="${ALPHA:-0.5}"
GAMMA_VALUES="${GAMMA_VALUES:-0.6,0.8,1.0}"
PRIMARY_GAMMA="${PRIMARY_GAMMA:-0.8}"
RHO="${RHO:-2.0}"
AMBIGUOUS_BUCKET_PENALTY="${AMBIGUOUS_BUCKET_PENALTY:-0.6}"
TAU_POLAR_READY="${TAU_POLAR_READY:-0.8}"
MAX_FORCED_POLAR_SLOTS="${MAX_FORCED_POLAR_SLOTS:-2}"
TAU_C="${TAU_C:-0.50}"
TAU_R="${TAU_R:-0.15}"

SAMPLE_ARGS=()
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  SAMPLE_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

PROGRESS_ARGS=()
if [[ "${NO_PROGRESS}" == "1" || "${NO_PROGRESS}" == "true" || "${NO_PROGRESS}" == "True" ]]; then
  PROGRESS_ARGS=(--no-progress)
fi

if [[ "${RUN_ANNOTATE}" == "auto" ]]; then
  if [[ -n "${!TEACHER_API_KEY_ENV:-}" ]]; then
    RUN_ANNOTATE="1"
  else
    RUN_ANNOTATE="0"
  fi
fi

echo "[count-amplified-v0.2] split       : ${SPLIT}"
echo "[count-amplified-v0.2] input dir   : ${INPUT_DIR}"
echo "[count-amplified-v0.2] output dir  : ${OUTPUT_DIR}"
echo "[count-amplified-v0.2] union pool  : ${UNION_POOL}"
echo "[count-amplified-v0.2] annotate    : ${RUN_ANNOTATE}"
echo "[count-amplified-v0.2] gamma       : ${GAMMA_VALUES}"
echo "[count-amplified-v0.2] rho         : ${RHO}"
echo "[count-amplified-v0.2] amb penalty : ${AMBIGUOUS_BUCKET_PENALTY}"
echo "[count-amplified-v0.2] polar ready : ${TAU_POLAR_READY}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${RUN_ANNOTATE}" == "1" || "${RUN_ANNOTATE}" == "true" || "${RUN_ANNOTATE}" == "True" ]]; then
  PYTHONPATH=src python scripts/phase5_selectors/build/annotate_stance_buckets_deepseek.py \
    --candidate-pool "${UNION_POOL}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --base-url "${TEACHER_BASE_URL}" \
    --model "${TEACHER_MODEL}" \
    --prompt-version stance_bucket_teacher_v02 \
    --api-key-env "${TEACHER_API_KEY_ENV}" \
    --concurrency "${TEACHER_CONCURRENCY}" \
    --requests-per-minute "${TEACHER_RPM}" \
    --tokens-per-minute "${TEACHER_TPM}" \
    --max-retries "${TEACHER_MAX_RETRIES}" \
    --max-tokens "${TEACHER_MAX_TOKENS}" \
    --top-logprobs "${TEACHER_TOP_LOGPROBS}" \
    --fallback-top-logprobs "${TEACHER_FALLBACK_TOP_LOGPROBS}" \
    --thinking-type "${TEACHER_THINKING_TYPE}" \
    "${SAMPLE_ARGS[@]}" \
    "${PROGRESS_ARGS[@]}"
else
  echo "[count-amplified-v0.2] annotation skipped; expecting ${ANNOTATIONS}"
fi

if [[ "${RUN_POSTPROCESS}" == "1" || "${RUN_POSTPROCESS}" == "true" || "${RUN_POSTPROCESS}" == "True" ]]; then
  if [[ ! -s "${ANNOTATIONS}" ]]; then
    echo "[count-amplified-v0.2] missing annotations: ${ANNOTATIONS}" >&2
    exit 1
  fi
  PYTHONPATH=src python scripts/phase5_selectors/build/postprocess_stance_scores_to_buckets.py \
    --candidate-pool "${QUALITY_LABELS}" \
    --teacher-annotations "${ANNOTATIONS}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --n-stance-buckets "${N_STANCE_BUCKETS}" \
    --bucket-tau "${BUCKET_TAU}" \
    --artifact-suffix v02
fi

if [[ "${RUN_EVAL}" == "1" || "${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "True" ]]; then
  for n in ${EVAL_N_BUCKETS}; do
    if [[ "${n}" == "3" ]]; then
      BUCKET_FILE="${OUTPUT_DIR}/candidate_stance_buckets_v02_${SPLIT}.jsonl"
    else
      BUCKET_FILE="${OUTPUT_DIR}/candidate_stance_buckets_v02_n${n}_${SPLIT}.jsonl"
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
      --use-directness-scoring \
      --adaptive-polar-quota \
      --tau-polar-ready "${TAU_POLAR_READY}" \
      --max-forced-polar-slots "${MAX_FORCED_POLAR_SLOTS}" \
      --tau-c "${TAU_C}" \
      --tau-r "${TAU_R}"
  done
fi

echo "[count-amplified-v0.2] done: ${OUTPUT_DIR}"
