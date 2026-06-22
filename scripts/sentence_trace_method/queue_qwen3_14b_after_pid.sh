#!/usr/bin/env bash
set -euo pipefail

# Queue Qwen3-14B no-sibling full pipeline after a currently running PID.
#
# Usage (foreground):
#   WAIT_PID=1338115 bash scripts/sentence_trace_method/queue_qwen3_14b_after_pid.sh
#
# Env vars:
#   WAIT_PID           PID to wait for (required)
#   DEADLINE_DATE       deadline date, default: next day (2026-06-23)
#   DEADLINE_TIME       deadline time, default: 10:00:00
#   POLL_SECONDS        check interval, default: 60
#   TERM_GRACE_SECONDS  grace period after SIGTERM before SIGKILL, default: 600
#   NPROC_PER_NODE      GPU count for the queued job, default: 4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MEDFACT_ROOT="${MEDFACT_ROOT:-/data/liaozijie/MedFact}"

WAIT_PID="${WAIT_PID:?Set WAIT_PID to the currently running process id.}"
DEADLINE_DATE="${DEADLINE_DATE:-2026-06-23}"
DEADLINE_TIME="${DEADLINE_TIME:-10:00:00}"
DEADLINE_AT="${DEADLINE_AT:-${DEADLINE_DATE} ${DEADLINE_TIME}}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-600}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

TARGET_SCRIPT="${MEDFACT_ROOT}/scripts/run_qwen3_14b_no_sibling_full_pipeline.sh"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/sentence_trace_method/_queue_logs}"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/qwen3_14b_queue_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

deadline_epoch() {
  date -d "${DEADLINE_AT}" +%s
}

now_epoch() {
  date +%s
}

deadline_reached() {
  [[ "$(now_epoch)" -ge "$(deadline_epoch)" ]]
}

process_alive() {
  kill -0 "$1" 2>/dev/null
}

children_of() {
  ps -eo pid=,ppid= | awk -v parent="$1" '$2 == parent {print $1}'
}

collect_tree() {
  local pid="$1"
  local child=""
  printf '%s\n' "${pid}"
  while read -r child; do
    [[ -z "${child}" ]] && continue
    collect_tree "${child}"
  done < <(children_of "${pid}")
}

any_alive() {
  local pid=""
  for pid in "$@"; do
    if process_alive "${pid}"; then
      return 0
    fi
  done
  return 1
}

terminate_tree() {
  local root_pid="$1"
  local label="$2"
  local hard_deadline=0
  local pid=""
  local pids=()

  if ! process_alive "${root_pid}"; then
    log "${label}: root pid ${root_pid} is already gone"
    return 0
  fi

  mapfile -t pids < <(collect_tree "${root_pid}" | awk '!seen[$0]++' | sort -rn)
  log "${label}: TERM process tree root=${root_pid} count=${#pids[@]}"
  for pid in "${pids[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done

  hard_deadline=$(( $(now_epoch) + TERM_GRACE_SECONDS ))
  while any_alive "${pids[@]}" && [[ "$(now_epoch)" -lt "${hard_deadline}" ]]; do
    sleep 5
  done

  if any_alive "${pids[@]}"; then
    log "${label}: KILL remaining processes after ${TERM_GRACE_SECONDS}s grace"
    for pid in "${pids[@]}"; do
      kill -KILL "${pid}" 2>/dev/null || true
    done
  else
    log "${label}: stopped after TERM"
  fi
}

monitor_until_exit_or_deadline() {
  local pid="$1"
  local label="$2"

  while process_alive "${pid}"; do
    if deadline_reached; then
      log "${label}: deadline ${DEADLINE_AT} reached"
      terminate_tree "${pid}" "${label}"
      return 124
    fi
    sleep "${POLL_SECONDS}"
  done
  log "${label}: pid ${pid} exited before deadline"
  return 0
}

main() {
  log "============================================"
  log "queue: Qwen3-14B no-sibling full pipeline"
  log "log path:  ${LOG_PATH}"
  log "wait pid:  ${WAIT_PID}"
  log "deadline:  ${DEADLINE_AT} (epoch $(deadline_epoch))"
  log "poll:      ${POLL_SECONDS}s"
  log "grace:     ${TERM_GRACE_SECONDS}s"
  log "target:    ${TARGET_SCRIPT}"
  log "medfact:   ${MEDFACT_ROOT}"
  log "NPROC:     ${NPROC_PER_NODE}"
  log "============================================"

  # Phase 1: wait for the current training process
  if process_alive "${WAIT_PID}"; then
    log "phase1: waiting for pid=${WAIT_PID} to exit"
    monitor_until_exit_or_deadline "${WAIT_PID}" "wait-pid" || exit $?
  else
    log "phase1: pid=${WAIT_PID} is not alive; starting immediately"
  fi

  # Phase 2: check deadline before launching
  if deadline_reached; then
    log "phase2: deadline ${DEADLINE_AT} already reached; exiting"
    exit 124
  fi

  # Phase 3: launch the queued job
  if [[ ! -x "${TARGET_SCRIPT}" ]]; then
    log "phase3: ERROR target script not found or not executable: ${TARGET_SCRIPT}"
    exit 1
  fi

  log "phase3: launching qwen3-14b no-sibling pipeline in ${MEDFACT_ROOT}"
  (
    cd "${MEDFACT_ROOT}"
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
      bash scripts/run_qwen3_14b_no_sibling_full_pipeline.sh
  ) &
  local job_pid="$!"

  log "phase3: job pid=${job_pid}"

  # Phase 4: monitor the job with deadline
  if monitor_until_exit_or_deadline "${job_pid}" "qwen3-14b"; then
    wait "${job_pid}" || true
    local job_status=$?
    log "phase4: job exited with status ${job_status}"
    exit "${job_status}"
  else
    exit $?
  fi
}

main "$@"
