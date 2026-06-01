#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

EXPERIMENT="${EXPERIMENT:-v0_6c_deepseek_v4_flash}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/b3_selector_trace_full_pipeline/v0_6c_rule_step_adaptive5_10_deepseek_v4_flash}"
CONFIG_PATH="${CONFIG_PATH:-outputs/selector_trace_verifier/stage2_sentence/v0_6c_rule_step_adaptive5_10/train.resolved.yaml}"
SPLIT="${SPLIT:-val}"
MODEL="${MODEL:-deepseek-v4-flash}"
BASE_URL="${BASE_URL:-https://api.deepseek.com}"
API_KEY_ENV="${API_KEY_ENV:-DEEPSEEK_API_KEY}"
CONCURRENCY="${CONCURRENCY:-1}"
REQUESTS_PER_MINUTE="${REQUESTS_PER_MINUTE:-60}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-120}"
MAX_RETRIES="${MAX_RETRIES:-4}"
RETRY_BASE_SLEEP="${RETRY_BASE_SLEEP:-2.0}"
RETRY_MAX_SLEEP="${RETRY_MAX_SLEEP:-60.0}"
RESUME="${RESUME:-true}"
FORCE="${FORCE:-false}"
RETRY_FAILED="${RETRY_FAILED:-true}"
PIPELINE_RESUME="${PIPELINE_RESUME:-true}"
PIPELINE_FORCE_INFER="${PIPELINE_FORCE_INFER:-false}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"

hydra_string_override() {
  local key="$1"
  local value="$2"
  value="${value//\\/\\\\}"
  value="${value//\'/\\\'}"
  printf "%s='%s'" "${key}" "${value}"
}

run_mode() {
  local checkpoint="$1"
  local mode_label="$2"
  local thinking_type="$3"
  local reasoning_effort="$4"
  local max_tokens="$5"
  local temperature="$6"
  local cmd=(
    python -m fact_checking.pipeline.run
    "experiment=${EXPERIMENT}"
    "pipeline.mode=infer"
    "pipeline.resume=${PIPELINE_RESUME}"
    "pipeline.force.infer=${PIPELINE_FORCE_INFER}"
    "$(hydra_string_override pipeline.run_dir "${RUN_ROOT}")"
    "$(hydra_string_override infer.config_path "${CONFIG_PATH}")"
    "$(hydra_string_override infer.split "${SPLIT}")"
    "$(hydra_string_override infer.checkpoint "${checkpoint}")"
    "$(hydra_string_override infer.model "${MODEL}")"
    "$(hydra_string_override infer.base_url "${BASE_URL}")"
    "$(hydra_string_override infer.api_key_env "${API_KEY_ENV}")"
    "$(hydra_string_override infer.mode_label "${mode_label}")"
    "$(hydra_string_override infer.thinking.type "${thinking_type}")"
    "infer.max_tokens=${max_tokens}"
    "infer.concurrency=${CONCURRENCY}"
    "infer.requests_per_minute=${REQUESTS_PER_MINUTE}"
    "infer.request_timeout_seconds=${REQUEST_TIMEOUT_SECONDS}"
    "infer.max_retries=${MAX_RETRIES}"
    "infer.retry_base_sleep=${RETRY_BASE_SLEEP}"
    "infer.retry_max_sleep=${RETRY_MAX_SLEEP}"
    "infer.resume=${RESUME}"
    "infer.force=${FORCE}"
    "infer.retry_failed=${RETRY_FAILED}"
  )
  if [[ -n "${reasoning_effort}" ]]; then
    cmd+=("$(hydra_string_override infer.reasoning_effort "${reasoning_effort}")")
  else
    cmd+=("infer.reasoning_effort=null")
  fi
  if [[ -n "${temperature}" ]]; then
    cmd+=("infer.temperature=${temperature}")
  else
    cmd+=("infer.temperature=null")
  fi
  if [[ -n "${SAMPLE_LIMIT}" ]]; then
    cmd+=("infer.sample_limit=${SAMPLE_LIMIT}")
  fi

  echo "[v0.6c-deepseek] running ${mode_label}: checkpoint=${checkpoint} max_tokens=${max_tokens}"
  "${cmd[@]}"
}

echo "[v0.6c-deepseek] run_root : ${RUN_ROOT}"
echo "[v0.6c-deepseek] config   : ${CONFIG_PATH}"
echo "[v0.6c-deepseek] split    : ${SPLIT}"
echo "[v0.6c-deepseek] model    : ${MODEL}"
echo "[v0.6c-deepseek] api env  : ${API_KEY_ENV}"

run_mode no_thinking no_thinking disabled "" 16 0.0
run_mode thinking_high thinking_high enabled high 1024 ""

python scripts/phase5_selectors/eval/summarize_v0_6c_deepseek_comparison.py \
  --run-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}" \
  --split "${SPLIT}" \
  --no-thinking-checkpoint no_thinking \
  --thinking-checkpoint thinking_high

echo "[v0.6c-deepseek] done: ${RUN_ROOT}"
