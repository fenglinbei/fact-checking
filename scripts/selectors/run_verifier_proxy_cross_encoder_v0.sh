#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DIRECT_VERIFIER_RUN_DIR="${DIRECT_VERIFIER_RUN_DIR:-outputs/oracle_direct_verifier/stage2_sentence/train/b3_oracle_sentence_direct_verifier_1024_20260519-200709}"
VERIFIER_CHECKPOINT="${VERIFIER_CHECKPOINT:-best}"
VERIFIER_BASE_URL="${VERIFIER_BASE_URL:-http://127.0.0.1:8000/v1}"
VERIFIER_MODEL="${VERIFIER_MODEL:-fact-checking-sft}"
LABEL_PREFIX="${LABEL_PREFIX:-Label:}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/question_decomp_retrieval/verifier_proxy_cross_encoder/b3_oracle_direct_v0}"
CACHE_DIR="${CACHE_DIR:-outputs/selectors/question_decomp_retrieval/verifier_proxy_cross_encoder/verifier_score_cache}"
TRAIN_UNION_POOL_JSONL="${TRAIN_UNION_POOL_JSONL:-outputs/selectors/question_decomp_retrieval/qwen_v0_train/union_candidate_pool_train.jsonl}"
VAL_UNION_POOL_JSONL="${VAL_UNION_POOL_JSONL:-outputs/selectors/question_decomp_retrieval/qwen_v0_val/union_candidate_pool_val.jsonl}"
TRAIN_ORACLE_RESULTS="${TRAIN_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl}"
VAL_ORACLE_RESULTS="${VAL_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
TRAIN_RAW_JSON="${TRAIN_RAW_JSON:-data/raw/LIAR-RAW/train.json}"
VAL_RAW_JSON="${VAL_RAW_JSON:-data/raw/LIAR-RAW/val.json}"

RUN_LABELS="${RUN_LABELS:-true}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
RESUME_LABELS="${RESUME_LABELS:-true}"
NO_PROGRESS="${NO_PROGRESS:-false}"

API_TIMEOUT="${API_TIMEOUT:-120}"
API_MAX_RETRIES="${API_MAX_RETRIES:-5}"
RETRY_INITIAL_DELAY="${RETRY_INITIAL_DELAY:-1.0}"
RETRY_MAX_DELAY="${RETRY_MAX_DELAY:-30.0}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-1024}"

CROSS_ENCODER_MODEL="${CROSS_ENCODER_MODEL:-/data/models/bge-reranker-large}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-${OUTPUT_DIR}/cross_encoder}"
SELECTOR_TOP_K="${SELECTOR_TOP_K:-5}"
MAX_LENGTH="${MAX_LENGTH:-384}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SOFT_TAU="${SOFT_TAU:-0.3}"
SOFT_CE_WEIGHT="${SOFT_CE_WEIGHT:-0.2}"
REGRESSION_WEIGHT="${REGRESSION_WEIGHT:-0.2}"
BCE_WEIGHT="${BCE_WEIGHT:-0.1}"
UTILITY_EPSILON="${UTILITY_EPSILON:-0.0001}"
SEED="${SEED:-20260526}"
DEVICE="${DEVICE:-cuda}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-4}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[verifier-proxy-v0] missing ${label}: ${path}" >&2
    echo "[verifier-proxy-v0] sync this file before running; this wrapper will not fall back to DeepSeek or another verifier." >&2
    exit 1
  fi
}

if [[ "${VERIFIER_CHECKPOINT}" == "final" ]]; then
  echo "[verifier-proxy-v0] VERIFIER_CHECKPOINT=final is not allowed; use best or checkpoint-600." >&2
  exit 1
fi

VERIFIER_CHECKPOINT_DIR="${DIRECT_VERIFIER_RUN_DIR}/${VERIFIER_CHECKPOINT}"
require_file "${VERIFIER_CHECKPOINT_DIR}/adapter_config.json" "verifier adapter config"
require_file "${VERIFIER_CHECKPOINT_DIR}/adapter_model.safetensors" "verifier adapter weights"
require_file "${VERIFIER_CHECKPOINT_DIR}/tokenizer_config.json" "verifier tokenizer config"
require_file "${DIRECT_VERIFIER_RUN_DIR}/label_token_ce_meta.json" "label-token metadata"

PROGRESS_ARGS=()
if [[ "${NO_PROGRESS}" == "1" || "${NO_PROGRESS}" == "true" || "${NO_PROGRESS}" == "True" ]]; then
  PROGRESS_ARGS=(--no-progress)
fi

RESUME_ARGS=(--resume)
if [[ "${RESUME_LABELS}" == "0" || "${RESUME_LABELS}" == "false" || "${RESUME_LABELS}" == "False" ]]; then
  RESUME_ARGS=(--no-resume)
fi

TRAIN_SAMPLE_LIMIT_ARGS=()
if [[ -n "${TRAIN_SAMPLE_LIMIT:-}" ]]; then
  TRAIN_SAMPLE_LIMIT_ARGS=(--sample-limit "${TRAIN_SAMPLE_LIMIT}")
fi

VAL_SAMPLE_LIMIT_ARGS=()
if [[ -n "${VAL_SAMPLE_LIMIT:-}" ]]; then
  VAL_SAMPLE_LIMIT_ARGS=(--sample-limit "${VAL_SAMPLE_LIMIT}")
fi

TRAIN_XENC_LIMIT_ARGS=()
if [[ -n "${TRAIN_SAMPLE_LIMIT:-}" ]]; then
  TRAIN_XENC_LIMIT_ARGS=(--train-sample-limit "${TRAIN_SAMPLE_LIMIT}")
fi

VAL_XENC_LIMIT_ARGS=()
if [[ -n "${VAL_SAMPLE_LIMIT:-}" ]]; then
  VAL_XENC_LIMIT_ARGS=(--val-sample-limit "${VAL_SAMPLE_LIMIT}")
fi

echo "[verifier-proxy-v0] verifier run   : ${DIRECT_VERIFIER_RUN_DIR}"
echo "[verifier-proxy-v0] checkpoint     : ${VERIFIER_CHECKPOINT}"
echo "[verifier-proxy-v0] verifier api   : ${VERIFIER_BASE_URL} model=${VERIFIER_MODEL}"
echo "[verifier-proxy-v0] output dir     : ${OUTPUT_DIR}"
echo "[verifier-proxy-v0] cache dir      : ${CACHE_DIR}"
echo "[verifier-proxy-v0] run labels     : ${RUN_LABELS}"
echo "[verifier-proxy-v0] run train      : ${RUN_TRAIN}"
echo "[verifier-proxy-v0] run eval       : ${RUN_EVAL}"

if [[ "${RUN_LABELS}" == "1" || "${RUN_LABELS}" == "true" || "${RUN_LABELS}" == "True" ]]; then
  PYTHONPATH=src python scripts/selectors/build_verifier_proxy_candidate_labels.py \
    --split train \
    --union-pool-jsonl "${TRAIN_UNION_POOL_JSONL}" \
    --oracle-results "${TRAIN_ORACLE_RESULTS}" \
    --raw-split-json "${TRAIN_RAW_JSON}" \
    --output-dir "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --direct-verifier-run-dir "${DIRECT_VERIFIER_RUN_DIR}" \
    --verifier-checkpoint "${VERIFIER_CHECKPOINT}" \
    --verifier-base-url "${VERIFIER_BASE_URL}" \
    --verifier-model "${VERIFIER_MODEL}" \
    --api-timeout "${API_TIMEOUT}" \
    --api-max-retries "${API_MAX_RETRIES}" \
    --retry-initial-delay "${RETRY_INITIAL_DELAY}" \
    --retry-max-delay "${RETRY_MAX_DELAY}" \
    --prompt-logprobs "${PROMPT_LOGPROBS}" \
    --label-prefix "${LABEL_PREFIX}" \
    --prompt-max-length "${PROMPT_MAX_LENGTH}" \
    "${RESUME_ARGS[@]}" \
    "${PROGRESS_ARGS[@]}" \
    "${TRAIN_SAMPLE_LIMIT_ARGS[@]}"

  PYTHONPATH=src python scripts/selectors/build_verifier_proxy_candidate_labels.py \
    --split val \
    --union-pool-jsonl "${VAL_UNION_POOL_JSONL}" \
    --oracle-results "${VAL_ORACLE_RESULTS}" \
    --raw-split-json "${VAL_RAW_JSON}" \
    --output-dir "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --direct-verifier-run-dir "${DIRECT_VERIFIER_RUN_DIR}" \
    --verifier-checkpoint "${VERIFIER_CHECKPOINT}" \
    --verifier-base-url "${VERIFIER_BASE_URL}" \
    --verifier-model "${VERIFIER_MODEL}" \
    --api-timeout "${API_TIMEOUT}" \
    --api-max-retries "${API_MAX_RETRIES}" \
    --retry-initial-delay "${RETRY_INITIAL_DELAY}" \
    --retry-max-delay "${RETRY_MAX_DELAY}" \
    --prompt-logprobs "${PROMPT_LOGPROBS}" \
    --label-prefix "${LABEL_PREFIX}" \
    --prompt-max-length "${PROMPT_MAX_LENGTH}" \
    "${RESUME_ARGS[@]}" \
    "${PROGRESS_ARGS[@]}" \
    "${VAL_SAMPLE_LIMIT_ARGS[@]}"
fi

if [[ "${RUN_TRAIN}" == "1" || "${RUN_TRAIN}" == "true" || "${RUN_TRAIN}" == "True" ]]; then
  PYTHONPATH=src python scripts/selectors/train_verifier_proxy_cross_encoder.py \
    --train-labels-jsonl "${OUTPUT_DIR}/candidate_utility_train.jsonl" \
    --val-labels-jsonl "${OUTPUT_DIR}/candidate_utility_val.jsonl" \
    --train-oracle-results "${TRAIN_ORACLE_RESULTS}" \
    --val-oracle-results "${VAL_ORACLE_RESULTS}" \
    --output-dir "${MODEL_OUTPUT_DIR}" \
    --model-name "${CROSS_ENCODER_MODEL}" \
    --top-k "${SELECTOR_TOP_K}" \
    --max-length "${MAX_LENGTH}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --learning-rate "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --soft-tau "${SOFT_TAU}" \
    --soft-ce-weight "${SOFT_CE_WEIGHT}" \
    --regression-weight "${REGRESSION_WEIGHT}" \
    --bce-weight "${BCE_WEIGHT}" \
    --utility-epsilon "${UTILITY_EPSILON}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --eval-every "${EVAL_EVERY}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
    "${PROGRESS_ARGS[@]}" \
    "${TRAIN_XENC_LIMIT_ARGS[@]}" \
    "${VAL_XENC_LIMIT_ARGS[@]}"
elif [[ "${RUN_EVAL}" == "1" || "${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "True" ]]; then
  PYTHONPATH=src python scripts/selectors/train_verifier_proxy_cross_encoder.py \
    --eval-only \
    --model-dir "${MODEL_OUTPUT_DIR}" \
    --train-labels-jsonl "${OUTPUT_DIR}/candidate_utility_train.jsonl" \
    --val-labels-jsonl "${OUTPUT_DIR}/candidate_utility_val.jsonl" \
    --train-oracle-results "${TRAIN_ORACLE_RESULTS}" \
    --val-oracle-results "${VAL_ORACLE_RESULTS}" \
    --output-dir "${MODEL_OUTPUT_DIR}" \
    --top-k "${SELECTOR_TOP_K}" \
    --max-length "${MAX_LENGTH}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --device "${DEVICE}" \
    "${PROGRESS_ARGS[@]}" \
    "${TRAIN_XENC_LIMIT_ARGS[@]}" \
    "${VAL_XENC_LIMIT_ARGS[@]}"
fi
