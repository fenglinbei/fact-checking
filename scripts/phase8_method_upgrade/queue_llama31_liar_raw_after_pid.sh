#!/usr/bin/env bash
set -euo pipefail

# Wait for an existing process to exit, then run the LIAR-RAW Llama-3.1
# no-ordinal control followed by the ordinal-aware variant.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

WAIT_PID="${WAIT_PID:?Set WAIT_PID to the currently running process id.}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
RUN_ENV_BIN="${RUN_ENV_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin}"
EVAL_SPLITS="${EVAL_SPLITS:-${INFER_SPLIT:-val,test}}"
RUN_API_INFER="${RUN_API_INFER:-false}"
RUN_LABEL_TOKEN_INFER="${RUN_LABEL_TOKEN_INFER:-true}"
LOG_DIR="${LOG_DIR:-outputs/selector_trace_verifier/liar_raw_v0_6c_method_upgrade/_logs}"

mkdir -p "${LOG_DIR}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

wait_for_pid() {
  echo "[$(timestamp)] waiting for pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
  echo "[$(timestamp)] pid=${WAIT_PID} exited; starting queued LIAR-RAW runs"
}

run_step() {
  local name="$1"
  local script="$2"
  local split="$3"
  local log_path="${LOG_DIR}/${name}.log"

  echo "[$(timestamp)] start ${name}; log=${log_path}"
  PATH="${RUN_ENV_BIN}:${PATH}" \
  INFER_SPLIT="${split}" \
  RUN_API_INFER="${RUN_API_INFER}" \
  RUN_LABEL_TOKEN_INFER="${RUN_LABEL_TOKEN_INFER}" \
  bash "${script}" > "${log_path}" 2>&1
  echo "[$(timestamp)] done ${name}"
}

wait_for_pid
IFS=',' read -r -a EVAL_SPLIT_ARRAY <<< "${EVAL_SPLITS}"
for EVAL_SPLIT in "${EVAL_SPLIT_ARRAY[@]}"; do
  EVAL_SPLIT="${EVAL_SPLIT//[[:space:]]/}"
  if [[ -z "${EVAL_SPLIT}" ]]; then
    continue
  fi
  run_step "llama31_liar_raw_noord_ctrl_${EVAL_SPLIT}" "${SCRIPT_DIR}/run_llama31_liar_raw_noord_ctrl.sh" "${EVAL_SPLIT}"
  run_step "llama31_liar_raw_ord_abs_a02_${EVAL_SPLIT}" "${SCRIPT_DIR}/run_llama31_liar_raw.sh" "${EVAL_SPLIT}"
done

echo "[$(timestamp)] queued LIAR-RAW runs completed"
