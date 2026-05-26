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
API_RETRY_INITIAL_DELAY="${API_RETRY_INITIAL_DELAY:-1.0}"
API_RETRY_MAX_DELAY="${API_RETRY_MAX_DELAY:-30.0}"

MAX_TOKENS="${MAX_TOKENS:-384}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-20260526}"
GUIDED_JSON="${GUIDED_JSON:-1}"
RESUME_QUESTIONS="${RESUME_QUESTIONS:-true}"
RUN_VERIFIER="${RUN_VERIFIER:-false}"

SAMPLE_LIMIT_ARGS=()
if [[ -n "${SAMPLE_LIMIT:-}" ]]; then
  SAMPLE_LIMIT_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

QUESTION_CACHE_ID_ARGS=()
if [[ -n "${QUESTION_CACHE_ID:-}" ]]; then
  QUESTION_CACHE_ID_ARGS=(--question-cache-id "${QUESTION_CACHE_ID}")
fi

GUIDED_JSON_ARGS=()
if [[ "${GUIDED_JSON}" == "0" || "${GUIDED_JSON}" == "false" || "${GUIDED_JSON}" == "False" ]]; then
  GUIDED_JSON_ARGS=(--no-guided-json)
fi

RESUME_ARGS=(--resume-questions)
if [[ "${RESUME_QUESTIONS}" == "0" || "${RESUME_QUESTIONS}" == "false" || "${RESUME_QUESTIONS}" == "False" ]]; then
  RESUME_ARGS=(--no-resume-questions)
fi

echo "[question-decomp] split           : ${SPLIT}"
echo "[question-decomp] oracle results  : ${ORACLE_RESULTS}"
echo "[question-decomp] output dir      : ${OUTPUT_DIR}"
echo "[question-decomp] question cache  : ${QUESTION_CACHE_DIR}"
echo "[question-decomp] question model  : ${QUESTION_MODEL}"
echo "[question-decomp] base url        : ${QUESTION_BASE_URL}"
echo "[question-decomp] resume questions: ${RESUME_QUESTIONS}"

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
  --retry-initial-delay "${API_RETRY_INITIAL_DELAY}" \
  --retry-max-delay "${API_RETRY_MAX_DELAY}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --seed "${SEED}" \
  "${GUIDED_JSON_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${QUESTION_CACHE_ID_ARGS[@]}" \
  "${SAMPLE_LIMIT_ARGS[@]}" \
  "$@"

if [[ "${RUN_VERIFIER}" == "1" || "${RUN_VERIFIER}" == "true" || "${RUN_VERIFIER}" == "True" ]]; then
  echo "[question-decomp] RUN_VERIFIER=true requested, but this wrapper currently implements question-cache generation only."
  echo "[question-decomp] Retrieval merge and verifier evaluation remain a follow-up stage."
fi
