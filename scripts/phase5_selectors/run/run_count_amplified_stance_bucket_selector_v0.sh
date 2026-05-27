#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/count_amplified_stance_bucket_selector/v0_${SPLIT}}"
ORACLE_RESULTS="${ORACLE_RESULTS:-}"
QD_UNION_POOL_JSONL="${QD_UNION_POOL_JSONL:-}"

if [[ -z "${ORACLE_RESULTS}" ]]; then
  if [[ "${SPLIT}" == "train" ]]; then
    ORACLE_RESULTS="outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl"
  else
    ORACLE_RESULTS="outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"
  fi
fi

if [[ -z "${QD_UNION_POOL_JSONL}" ]]; then
  if [[ "${SPLIT}" == "train" ]]; then
    QD_UNION_POOL_JSONL="outputs/selectors/question_decomp_retrieval/qwen_v0_train/union_candidate_pool_train.jsonl"
  else
    QD_UNION_POOL_JSONL="outputs/selectors/question_decomp_retrieval/qwen_v0_val/union_candidate_pool_val.jsonl"
  fi
fi

SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
ARTIFACT_SUFFIX="${ARTIFACT_SUFFIX:-}"
TEACHER_OUTPUT_SUFFIX="${TEACHER_OUTPUT_SUFFIX:-}"
RUN_BUILD="${RUN_BUILD:-1}"
RUN_ANNOTATE="${RUN_ANNOTATE:-auto}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
NO_PROGRESS="${NO_PROGRESS:-0}"

TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://api.deepseek.com}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
TEACHER_PROMPT_VERSION="${TEACHER_PROMPT_VERSION:-stance_bucket_teacher_v0}"
TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
TEACHER_CONCURRENCY="${TEACHER_CONCURRENCY:-8}"
TEACHER_RPM="${TEACHER_RPM:-120}"
TEACHER_TPM="${TEACHER_TPM:-200000}"
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-5}"
TEACHER_MAX_TOKENS="${TEACHER_MAX_TOKENS:-64}"
TEACHER_TOP_LOGPROBS="${TEACHER_TOP_LOGPROBS:-20}"
TEACHER_FALLBACK_TOP_LOGPROBS="${TEACHER_FALLBACK_TOP_LOGPROBS:-5}"
TEACHER_THINKING_TYPE="${TEACHER_THINKING_TYPE:-disabled}"

N_STANCE_BUCKETS="${N_STANCE_BUCKETS:-3,5,7}"
EVAL_N_BUCKETS="${EVAL_N_BUCKETS:-3 5 7}"
BUCKET_TAU="${BUCKET_TAU:-2.0}"
TOP_K="${TOP_K:-5}"
ALPHA="${ALPHA:-0.5}"
GAMMA_VALUES="${GAMMA_VALUES:-1.0,1.6,1.8}"
PRIMARY_GAMMA="${PRIMARY_GAMMA:-1.6}"
RHO="${RHO:-0.6}"
AMBIGUOUS_BUCKET_PENALTY="${AMBIGUOUS_BUCKET_PENALTY:-1.0}"
USE_DIRECTNESS_SCORING="${USE_DIRECTNESS_SCORING:-0}"
ADAPTIVE_POLAR_QUOTA="${ADAPTIVE_POLAR_QUOTA:-0}"
TAU_POLAR_READY="${TAU_POLAR_READY:-0.8}"
MAX_FORCED_POLAR_SLOTS="${MAX_FORCED_POLAR_SLOTS:-0}"
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

DIRECTNESS_ARGS=()
if [[ "${USE_DIRECTNESS_SCORING}" == "1" || "${USE_DIRECTNESS_SCORING}" == "true" || "${USE_DIRECTNESS_SCORING}" == "True" ]]; then
  DIRECTNESS_ARGS=(--use-directness-scoring)
fi

ADAPTIVE_ARGS=()
if [[ "${ADAPTIVE_POLAR_QUOTA}" == "1" || "${ADAPTIVE_POLAR_QUOTA}" == "true" || "${ADAPTIVE_POLAR_QUOTA}" == "True" ]]; then
  ADAPTIVE_ARGS=(--adaptive-polar-quota --tau-polar-ready "${TAU_POLAR_READY}" --max-forced-polar-slots "${MAX_FORCED_POLAR_SLOTS}")
fi

if [[ "${RUN_ANNOTATE}" == "auto" ]]; then
  if [[ -n "${!TEACHER_API_KEY_ENV:-}" ]]; then
    RUN_ANNOTATE="1"
  else
    RUN_ANNOTATE="0"
  fi
fi

UNION_POOL="${OUTPUT_DIR}/union_analysis_candidate_pool_${SPLIT}.jsonl"
QUALITY_LABELS="${OUTPUT_DIR}/candidate_quality_labels_${SPLIT}.jsonl"
ANNOTATIONS="${ANNOTATIONS:-${OUTPUT_DIR}/deepseek_teacher_annotations${TEACHER_OUTPUT_SUFFIX}_${SPLIT}.jsonl}"
ARTIFACT_SUFFIX_PART=""
if [[ -n "${ARTIFACT_SUFFIX}" ]]; then
  if [[ "${ARTIFACT_SUFFIX}" == _* ]]; then
    ARTIFACT_SUFFIX_PART="${ARTIFACT_SUFFIX}"
  else
    ARTIFACT_SUFFIX_PART="_${ARTIFACT_SUFFIX}"
  fi
fi

echo "[count-amplified-v0] split        : ${SPLIT}"
echo "[count-amplified-v0] oracle       : ${ORACLE_RESULTS}"
echo "[count-amplified-v0] qd union     : ${QD_UNION_POOL_JSONL}"
echo "[count-amplified-v0] output dir   : ${OUTPUT_DIR}"
echo "[count-amplified-v0] run build    : ${RUN_BUILD}"
echo "[count-amplified-v0] run annotate : ${RUN_ANNOTATE}"
echo "[count-amplified-v0] run postproc : ${RUN_POSTPROCESS}"
echo "[count-amplified-v0] run eval     : ${RUN_EVAL}"
echo "[count-amplified-v0] thinking     : ${TEACHER_THINKING_TYPE}"
echo "[count-amplified-v0] amb penalty  : ${AMBIGUOUS_BUCKET_PENALTY}"
echo "[count-amplified-v0] direct scorer: ${USE_DIRECTNESS_SCORING}"
echo "[count-amplified-v0] adaptive     : ${ADAPTIVE_POLAR_QUOTA}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${RUN_BUILD}" == "1" || "${RUN_BUILD}" == "true" || "${RUN_BUILD}" == "True" ]]; then
  PYTHONPATH=src python scripts/phase5_selectors/build/build_union_analysis_candidate_pool.py \
    --oracle-results "${ORACLE_RESULTS}" \
    --qd-union-pool-jsonl "${QD_UNION_POOL_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    "${SAMPLE_ARGS[@]}"
fi

if [[ "${RUN_ANNOTATE}" == "1" || "${RUN_ANNOTATE}" == "true" || "${RUN_ANNOTATE}" == "True" ]]; then
  PYTHONPATH=src python scripts/phase5_selectors/build/annotate_stance_buckets_deepseek.py \
    --candidate-pool "${UNION_POOL}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --base-url "${TEACHER_BASE_URL}" \
    --model "${TEACHER_MODEL}" \
    --prompt-version "${TEACHER_PROMPT_VERSION}" \
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
  echo "[count-amplified-v0] annotation skipped; expecting ${ANNOTATIONS}"
fi

if [[ "${RUN_POSTPROCESS}" == "1" || "${RUN_POSTPROCESS}" == "true" || "${RUN_POSTPROCESS}" == "True" ]]; then
  if [[ ! -s "${ANNOTATIONS}" ]]; then
    echo "[count-amplified-v0] missing annotations: ${ANNOTATIONS}" >&2
    exit 1
  fi
  PYTHONPATH=src python scripts/phase5_selectors/build/postprocess_stance_scores_to_buckets.py \
    --candidate-pool "${QUALITY_LABELS}" \
    --teacher-annotations "${ANNOTATIONS}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --n-stance-buckets "${N_STANCE_BUCKETS}" \
    --bucket-tau "${BUCKET_TAU}" \
    --artifact-suffix "${ARTIFACT_SUFFIX}"
fi

if [[ "${RUN_EVAL}" == "1" || "${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "True" ]]; then
  for n in ${EVAL_N_BUCKETS}; do
    if [[ "${n}" == "3" ]]; then
      BUCKET_FILE="${OUTPUT_DIR}/candidate_stance_buckets${ARTIFACT_SUFFIX_PART}_${SPLIT}.jsonl"
    else
      BUCKET_FILE="${OUTPUT_DIR}/candidate_stance_buckets${ARTIFACT_SUFFIX_PART}_n${n}_${SPLIT}.jsonl"
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
      "${DIRECTNESS_ARGS[@]}" \
      "${ADAPTIVE_ARGS[@]}"
  done
fi

echo "[count-amplified-v0] done: ${OUTPUT_DIR}"
