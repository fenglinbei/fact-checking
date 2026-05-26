#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

SPLIT="${SPLIT:-val}"
if [[ -z "${ORACLE_RESULTS:-}" ]]; then
  if [[ "${SPLIT}" == "train" ]]; then
    ORACLE_RESULTS="outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl"
  elif [[ "${SPLIT}" == "test" ]]; then
    ORACLE_RESULTS="outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"
  else
    ORACLE_RESULTS="outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"
  fi
fi

OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/question_decomp_retrieval/qwen_v0_${SPLIT}}"
QUESTION_CACHE_DIR="${QUESTION_CACHE_DIR:-outputs/selectors/question_decomp_retrieval/question_cache}"
QUESTION_BASE_URL="${QUESTION_BASE_URL:-http://127.0.0.1:8000/v1}"
QUESTION_MODEL="${QUESTION_MODEL:-/data/models/Qwen2.5-7B-Instruct}"
QUESTION_API_KEY_ENV="${QUESTION_API_KEY_ENV:-QUESTION_API_KEY}"
QUESTION_API_TIMEOUT="${QUESTION_API_TIMEOUT:-120}"
API_MAX_RETRIES="${API_MAX_RETRIES:-5}"
API_PARSE_MAX_RETRIES="${API_PARSE_MAX_RETRIES:-2}"
API_RETRY_INITIAL_DELAY="${API_RETRY_INITIAL_DELAY:-1.0}"
API_RETRY_MAX_DELAY="${API_RETRY_MAX_DELAY:-30.0}"

MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-20260526}"
GUIDED_JSON="${GUIDED_JSON:-1}"
RESUME_QUESTIONS="${RESUME_QUESTIONS:-true}"
RUN_QUESTION_GENERATION="${RUN_QUESTION_GENERATION:-true}"
RUN_RETRIEVAL="${RUN_RETRIEVAL:-false}"
RUN_VERIFIER="${RUN_VERIFIER:-false}"
NO_PROGRESS="${NO_PROGRESS:-false}"

RETRIEVAL_OUTPUT_DIR="${RETRIEVAL_OUTPUT_DIR:-${OUTPUT_DIR}}"
QUESTIONS_JSONL="${QUESTIONS_JSONL:-${OUTPUT_DIR}/questions_${SPLIT}.jsonl}"
CHUNK_CACHE_PATH="${CHUNK_CACHE_PATH:-outputs/cache/chunk_mmr/432dfc970e75/${SPLIT}.pkl}"
if [[ -z "${EMBEDDER_MODEL:-}" ]]; then
  if [[ -d "/data/models/bge-base-en-v1.5" ]]; then
    EMBEDDER_MODEL="/data/models/bge-base-en-v1.5"
  elif [[ -d "/home/fenglin/project/models/bge-base-en-v1.5" ]]; then
    EMBEDDER_MODEL="/home/fenglin/project/models/bge-base-en-v1.5"
  else
    EMBEDDER_MODEL="/data/models/bge-base-en-v1.5"
  fi
fi
DEVICE="${DEVICE:-cuda}"
EMBEDDER_MAX_LENGTH="${EMBEDDER_MAX_LENGTH:-256}"
EMBEDDER_BATCH_SIZE="${EMBEDDER_BATCH_SIZE:-64}"
PRECISION="${PRECISION:-fp32}"
PER_QUESTION_KEEP="${PER_QUESTION_KEEP:-20}"
MERGED_POOL_SIZE="${MERGED_POOL_SIZE:-15}"
SELECTOR_TOP_K="${SELECTOR_TOP_K:-5}"
RRF_K="${RRF_K:-60}"
Q1_WEIGHT="${Q1_WEIGHT:-1.2}"
OTHER_QUESTION_WEIGHT="${OTHER_QUESTION_WEIGHT:-1.0}"
MERGE_MMR_LAMBDA="${MERGE_MMR_LAMBDA:-0.70}"
ALPHA_DENSE="${ALPHA_DENSE:-0.70}"
ALPHA_LEXICAL="${ALPHA_LEXICAL:-0.20}"
ALPHA_BM25="${ALPHA_BM25:-0.10}"

SAMPLE_LIMIT_ARGS=()
if [[ -n "${SAMPLE_LIMIT:-}" ]]; then
  SAMPLE_LIMIT_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

QUESTION_CACHE_ID_ARGS=()
if [[ -n "${QUESTION_CACHE_ID:-}" ]]; then
  QUESTION_CACHE_ID_ARGS=(--question-cache-id "${QUESTION_CACHE_ID}")
fi

if [[ -z "${QUESTION_THINKING_TYPE+x}" ]]; then
  if [[ "${QUESTION_MODEL}" == deepseek-* ]]; then
    QUESTION_THINKING_TYPE="disabled"
  else
    QUESTION_THINKING_TYPE=""
  fi
fi
THINKING_ARGS=()
if [[ -n "${QUESTION_THINKING_TYPE}" ]]; then
  THINKING_ARGS=(--thinking-type "${QUESTION_THINKING_TYPE}")
fi

GUIDED_JSON_ARGS=()
if [[ "${GUIDED_JSON}" == "0" || "${GUIDED_JSON}" == "false" || "${GUIDED_JSON}" == "False" ]]; then
  GUIDED_JSON_ARGS=(--no-guided-json)
fi

RESUME_ARGS=(--resume-questions)
if [[ "${RESUME_QUESTIONS}" == "0" || "${RESUME_QUESTIONS}" == "false" || "${RESUME_QUESTIONS}" == "False" ]]; then
  RESUME_ARGS=(--no-resume-questions)
fi

PROGRESS_ARGS=()
if [[ "${NO_PROGRESS}" == "1" || "${NO_PROGRESS}" == "true" || "${NO_PROGRESS}" == "True" ]]; then
  PROGRESS_ARGS=(--no-progress)
fi

echo "[question-decomp] split           : ${SPLIT}"
echo "[question-decomp] oracle results  : ${ORACLE_RESULTS}"
echo "[question-decomp] output dir      : ${OUTPUT_DIR}"
echo "[question-decomp] question cache  : ${QUESTION_CACHE_DIR}"
echo "[question-decomp] question model  : ${QUESTION_MODEL}"
echo "[question-decomp] base url        : ${QUESTION_BASE_URL}"
echo "[question-decomp] run generation  : ${RUN_QUESTION_GENERATION}"
echo "[question-decomp] run retrieval   : ${RUN_RETRIEVAL}"
echo "[question-decomp] resume questions: ${RESUME_QUESTIONS}"
echo "[question-decomp] thinking type   : ${QUESTION_THINKING_TYPE:-none}"
echo "[question-decomp] max tokens      : ${MAX_TOKENS}"
echo "[question-decomp] no progress     : ${NO_PROGRESS}"

if [[ "${RUN_QUESTION_GENERATION}" == "1" || "${RUN_QUESTION_GENERATION}" == "true" || "${RUN_QUESTION_GENERATION}" == "True" ]]; then
  PYTHONPATH=src python scripts/selectors/generate_question_decomp_cache.py \
    --oracle-results "${ORACLE_RESULTS}" \
    --split "${SPLIT}" \
    --output-dir "${OUTPUT_DIR}" \
    --question-cache-dir "${QUESTION_CACHE_DIR}" \
    --question-base-url "${QUESTION_BASE_URL}" \
    --question-model "${QUESTION_MODEL}" \
    --question-api-key-env "${QUESTION_API_KEY_ENV}" \
    --api-timeout "${QUESTION_API_TIMEOUT}" \
    --api-max-retries "${API_MAX_RETRIES}" \
    --api-parse-max-retries "${API_PARSE_MAX_RETRIES}" \
    --retry-initial-delay "${API_RETRY_INITIAL_DELAY}" \
    --retry-max-delay "${API_RETRY_MAX_DELAY}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --seed "${SEED}" \
    "${GUIDED_JSON_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${PROGRESS_ARGS[@]}" \
    "${THINKING_ARGS[@]}" \
    "${QUESTION_CACHE_ID_ARGS[@]}" \
    "${SAMPLE_LIMIT_ARGS[@]}" \
    "$@"
fi

if [[ "${RUN_RETRIEVAL}" == "1" || "${RUN_RETRIEVAL}" == "true" || "${RUN_RETRIEVAL}" == "True" ]]; then
  echo "[question-decomp] retrieval output: ${RETRIEVAL_OUTPUT_DIR}"
  echo "[question-decomp] questions jsonl : ${QUESTIONS_JSONL}"
  echo "[question-decomp] chunk cache     : ${CHUNK_CACHE_PATH}"
  PYTHONPATH=src python scripts/selectors/build_question_decomp_retrieval.py \
    --oracle-results "${ORACLE_RESULTS}" \
    --split "${SPLIT}" \
    --questions-jsonl "${QUESTIONS_JSONL}" \
    --chunk-cache-path "${CHUNK_CACHE_PATH}" \
    --output-dir "${RETRIEVAL_OUTPUT_DIR}" \
    --embedder-model "${EMBEDDER_MODEL}" \
    --device "${DEVICE}" \
    --embedder-max-length "${EMBEDDER_MAX_LENGTH}" \
    --embedder-batch-size "${EMBEDDER_BATCH_SIZE}" \
    --precision "${PRECISION}" \
    --per-question-keep "${PER_QUESTION_KEEP}" \
    --merged-pool-size "${MERGED_POOL_SIZE}" \
    --selector-top-k "${SELECTOR_TOP_K}" \
    --rrf-k "${RRF_K}" \
    --q1-weight "${Q1_WEIGHT}" \
    --other-question-weight "${OTHER_QUESTION_WEIGHT}" \
    --merge-mmr-lambda "${MERGE_MMR_LAMBDA}" \
    --alpha-dense "${ALPHA_DENSE}" \
    --alpha-lexical "${ALPHA_LEXICAL}" \
    --alpha-bm25 "${ALPHA_BM25}" \
    "${PROGRESS_ARGS[@]}" \
    "${SAMPLE_LIMIT_ARGS[@]}"
fi

if [[ "${RUN_VERIFIER}" == "1" || "${RUN_VERIFIER}" == "true" || "${RUN_VERIFIER}" == "True" ]]; then
  echo "[question-decomp] RUN_VERIFIER=true requested, but verifier evaluation is not implemented in this v0 wrapper."
fi
