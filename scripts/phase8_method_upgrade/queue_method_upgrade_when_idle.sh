#!/usr/bin/env bash
set -euo pipefail

# Temporary LIAR-RAW method-upgrade queue:
# 1. test-eval the no-ordinal control,
# 2. test-eval the ordinal-aware alpha=0.2 run,
# 3. run the combined method-upgrade training with test eval.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

IDLE_SECONDS="${IDLE_SECONDS:-300}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
IDLE_CHECK_MODE="${IDLE_CHECK_MODE:-memory}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
GPU_MEMORY_IDLE_MAX_MB="${GPU_MEMORY_IDLE_MAX_MB:-0}"
RUN_ENV_BIN="${RUN_ENV_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin}"
LOG_DIR="${LOG_DIR:-outputs/selector_trace_verifier/liar_raw_v0_6c_method_upgrade/_logs}"
CHECKPOINTS="${CHECKPOINTS:-best}"
FORCE_LABEL_TOKEN_INFER="${FORCE_LABEL_TOKEN_INFER:-false}"
METHOD_EVAL_SPLITS="${METHOD_EVAL_SPLITS:-test}"
METHOD_UPGRADE_ARGS="${METHOD_UPGRADE_ARGS:---wd 0.01 --warmup 0.3 --restart 2 --calibrated}"
QUEUE_DRY_RUN="${QUEUE_DRY_RUN:-false}"

BLOCKING_PATTERN="${BLOCKING_PATTERN:-accelerate|deepspeed|torchrun|vllm|sft\\.label_token|sft\\.trainer|sft\\.infer|fact_checking\\.pipeline\\.run|run_method_upgrade|run_llama31_liar_raw|python.*(sft|fact_checking|vllm|torch)}"

TEST_TRACE="${TEST_TRACE:-outputs/selectors/evidence_chain_graph/v0_6c_adaptive5_10_test/selection_trace_test.jsonl}"
NOORD_RUN_DIR="${NOORD_RUN_DIR:-outputs/selector_trace_verifier/liar_raw_v0_6c_method_upgrade/v0_6c_liar6_rule_step_adaptive5_10_llama31_8b_fullft_noord_ctrl}"
ORD_RUN_DIR="${ORD_RUN_DIR:-outputs/selector_trace_verifier/liar_raw_v0_6c_method_upgrade/v0_6c_liar6_rule_step_adaptive5_10_llama31_8b_fullft_ord_abs_a02}"

mkdir -p "${LOG_DIR}"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
MAIN_LOG="${LOG_DIR}/idle_method_upgrade_queue_${RUN_ID}.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

is_true() {
  case "${1:-}" in
    true|1|True|TRUE|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[$(timestamp)] missing ${label}: ${path}" >&2
    exit 1
  fi
}

blocking_processes() {
  ps -eo pid=,ppid=,stat=,etime=,cmd= | awk \
    -v self="$$" \
    -v pattern="${BLOCKING_PATTERN}" '
      BEGIN { IGNORECASE = 1 }
      $1 == self { next }
      $2 == self { next }
      /queue_method_upgrade_when_idle/ { next }
      /codex-linux-sandbox|bwrap|awk -v self/ { next }
      $0 ~ pattern { print }
    '
}

gpu_memory_busy_report() {
  local output=""
  local gpu_filter=",${GPU_IDS//[[:space:]]/},"
  if ! output="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>&1)"; then
    echo "nvidia-smi failed; treating GPU as busy: ${output}"
    return 0
  fi
  printf '%s\n' "${output}" | awk \
    -v gpu_filter="${gpu_filter}" \
    -v max_mb="${GPU_MEMORY_IDLE_MAX_MB}" '
      BEGIN { any = 0 }
      /^[[:space:]]*$/ { next }
      {
        gpu_idx = $1
        used = $2
        gsub(/,/, "", gpu_idx)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", gpu_idx)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", used)
        if (gpu_filter != ",all," && gpu_filter !~ "," gpu_idx ",") {
          next
        }
        any = 1
        if (used + 0 > max_mb + 0) {
          printf("gpu=%s memory.used=%s MiB > %s MiB\n", gpu_idx, used, max_mb)
        }
      }
      END {
        if (!any) {
          printf("no matching GPU ids in nvidia-smi output for GPU_IDS=%s\n", gpu_filter)
        }
      }
    '
}

idle_report() {
  if [[ "${IDLE_CHECK_MODE}" == "none" ]]; then
    return 1
  fi
  if [[ "${IDLE_CHECK_MODE}" == "memory" || "${IDLE_CHECK_MODE}" == "nvidia" ]]; then
    gpu_memory_busy_report
    return 0
  fi
  blocking_processes
}

wait_until_idle() {
  local idle_since=0
  local now=0
  local elapsed=0
  local busy_report=""

  echo "[$(timestamp)] idle check mode=${IDLE_CHECK_MODE} idle_seconds=${IDLE_SECONDS} interval=${CHECK_INTERVAL_SECONDS}"
  echo "[$(timestamp)] gpu ids=${GPU_IDS} idle memory threshold=${GPU_MEMORY_IDLE_MAX_MB} MiB"
  echo "[$(timestamp)] blocking pattern=${BLOCKING_PATTERN}"
  while true; do
    now="$(date '+%s')"
    busy_report="$(idle_report || true)"
    if [[ -z "${busy_report}" ]]; then
      if [[ "${idle_since}" -eq 0 ]]; then
        idle_since="${now}"
        echo "[$(timestamp)] idle window started"
      fi
      elapsed=$((now - idle_since))
      if [[ "${elapsed}" -ge "${IDLE_SECONDS}" ]]; then
        echo "[$(timestamp)] idle for ${elapsed}s; starting queue"
        return 0
      fi
      echo "[$(timestamp)] idle for ${elapsed}s/${IDLE_SECONDS}s"
    else
      if [[ "${idle_since}" -ne 0 ]]; then
        echo "[$(timestamp)] idle window reset"
      fi
      idle_since=0
      echo "[$(timestamp)] busy; blockers:"
      printf '%s\n' "${busy_report}" | sed 's/^/[blocker] /'
    fi
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

run_task() {
  local name="$1"
  shift
  local task_log="${LOG_DIR}/${name}_${RUN_ID}.log"

  echo "[$(timestamp)] start ${name}; log=${task_log}"
  if is_true "${QUEUE_DRY_RUN}"; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@" > "${task_log}" 2>&1
  echo "[$(timestamp)] done ${name}"
}

require_path "${TEST_TRACE}" "LIAR-RAW test trace"
require_path "${NOORD_RUN_DIR}/train/best" "no-ordinal best checkpoint"
require_path "${ORD_RUN_DIR}/train/best" "ordinal best checkpoint"
if [[ -n "${RUN_ENV_BIN}" ]]; then
  require_path "${RUN_ENV_BIN}/python" "run environment python"
  export PATH="${RUN_ENV_BIN}:${PATH}"
fi

echo "[$(timestamp)] main log=${MAIN_LOG}"
echo "[$(timestamp)] test trace=${TEST_TRACE}"
echo "[$(timestamp)] method args=${METHOD_UPGRADE_ARGS}"
echo "[$(timestamp)] method eval splits=${METHOD_EVAL_SPLITS}"

wait_until_idle

read -r -a METHOD_UPGRADE_ARGV <<< "${METHOD_UPGRADE_ARGS}"

run_task "test_eval_noord_ctrl" \
  env \
  RUN_TRAIN=false \
  RUN_INFER=false \
  RUN_API_INFER=false \
  RUN_LABEL_TOKEN_INFER=true \
  INFER_SPLIT=test \
  CHECKPOINTS="${CHECKPOINTS}" \
  FORCE_LABEL_TOKEN_INFER="${FORCE_LABEL_TOKEN_INFER}" \
  bash "${SCRIPT_DIR}/run_llama31_liar_raw_noord_ctrl.sh"

run_task "test_eval_ord_abs_a02" \
  env \
  RUN_TRAIN=false \
  RUN_INFER=false \
  RUN_API_INFER=false \
  RUN_LABEL_TOKEN_INFER=true \
  INFER_SPLIT=test \
  CHECKPOINTS="${CHECKPOINTS}" \
  FORCE_LABEL_TOKEN_INFER="${FORCE_LABEL_TOKEN_INFER}" \
  bash "${SCRIPT_DIR}/run_llama31_liar_raw.sh"

run_task "train_method_upgrade_combo" \
  env \
  EVAL_SPLITS="${METHOD_EVAL_SPLITS}" \
  bash "${SCRIPT_DIR}/run_method_upgrade.sh" "${METHOD_UPGRADE_ARGV[@]}"

echo "[$(timestamp)] queue completed"
