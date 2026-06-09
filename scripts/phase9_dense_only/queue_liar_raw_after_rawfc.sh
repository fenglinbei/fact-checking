#!/usr/bin/env bash
set -euo pipefail

# Queue LIAR-RAW dense-only backbones after the currently running RAWFC job.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_BIN="${CONDA_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin}"
export PATH="${CONDA_BIN}:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DEADLINE_DATE="${DEADLINE_DATE:-$(date +%F)}"
DEADLINE_TIME="${DEADLINE_TIME:-08:30:00}"
DEADLINE_AT="${DEADLINE_AT:-${DEADLINE_DATE} ${DEADLINE_TIME}}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-600}"

LIAR_MODE="${LIAR_MODE:-full}"
LIAR_BACKBONES="${LIAR_BACKBONES:-qwen3_4b_2507,llama31_8b}"
SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE:-true}"
RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE:-true}"

LOG_DIR="${LOG_DIR:-outputs/runs/phase9_dense_only_queue}"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/liar_raw_after_rawfc_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_PATH}") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
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

find_rawfc_pid() {
  ps -eo pid=,ppid=,cmd= \
    | awk '$3 == "bash" && $4 == "scripts/phase9_dense_only/run_rawfc_dense_only_backbones.sh" {print $1}' \
    | sort -n \
    | tail -n 1
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
  local rawfc_pid="${RAWFC_PID:-}"
  local liar_pid=""
  local liar_status=0

  log "queue log: ${LOG_PATH}"
  log "deadline: ${DEADLINE_AT} ($(date -d "${DEADLINE_AT}" '+%Y-%m-%d %H:%M:%S %Z'))"
  log "PATH head: ${CONDA_BIN}"

  if [[ -z "${rawfc_pid}" ]]; then
    rawfc_pid="$(find_rawfc_pid || true)"
  fi

  if [[ -n "${rawfc_pid}" ]] && process_alive "${rawfc_pid}"; then
    log "watching RAWFC pid=${rawfc_pid}"
    monitor_until_exit_or_deadline "${rawfc_pid}" "rawfc-dense" || exit $?
  else
    log "no running RAWFC pid found; LIAR-RAW can start immediately"
  fi

  if deadline_reached; then
    log "deadline already reached before LIAR-RAW launch; exiting"
    exit 124
  fi

  log "launching LIAR-RAW: MODE=${LIAR_MODE} BACKBONES=${LIAR_BACKBONES}"
  (
    MODE="${LIAR_MODE}" \
    BACKBONES="${LIAR_BACKBONES}" \
    SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE}" \
    RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE}" \
    bash scripts/phase9_dense_only/run_liar_raw_dense_only_backbones.sh
  ) &
  liar_pid="$!"

  if monitor_until_exit_or_deadline "${liar_pid}" "liar-raw-dense"; then
    wait "${liar_pid}" || liar_status=$?
    log "LIAR-RAW exited with status ${liar_status}"
    exit "${liar_status}"
  else
    exit $?
  fi
}

main "$@"
