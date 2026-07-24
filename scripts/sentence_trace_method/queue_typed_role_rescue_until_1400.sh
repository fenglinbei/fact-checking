#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
QUEUE_DEADLINE="${QUEUE_DEADLINE:-$(date +%F) 13:45:00}"
HARD_STOP_AT="${HARD_STOP_AT:-$(date +%F) 14:00:00}"
QUEUE_ID="${QUEUE_ID:-typed_role_rescue_$(date +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${QUEUE_ROOT:-outputs/sentence_trace_method/queues}"
QUEUE_DIR="${QUEUE_ROOT}/${QUEUE_ID}"
QUEUE_LOG="${QUEUE_DIR}/queue.log"
STATUS_FILE="${QUEUE_DIR}/status.txt"
LOCK_FILE="${QUEUE_ROOT}/typed_role_rescue.lock"
RUN_FROZEN_MATRIX="${RUN_FROZEN_MATRIX:-true}"
CONTINUE_ON_CELL_ERROR="${CONTINUE_ON_CELL_ERROR:-true}"
CELLS="${CELLS:-full random retr cor opp ctx r_only}"
CELL_SCRIPT="${CELL_SCRIPT:-scripts/sentence_trace_method/run_typed_role_rescue_lora_cell.sh}"
FROZEN_SCRIPT="${FROZEN_SCRIPT:-scripts/phase5_selectors/eval/run_typed_role_rescue_frozen_matrix.sh}"

mkdir -p "$QUEUE_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'Another typed role-rescue queue holds %s\n' "$LOCK_FILE" >&2
  exit 2
fi
exec > >(tee -a "$QUEUE_LOG") 2>&1

deadline_epoch="$(date -d "$QUEUE_DEADLINE" +%s)"
hard_epoch="$(date -d "$HARD_STOP_AT" +%s)"
if (( hard_epoch <= deadline_epoch )); then
  printf 'HARD_STOP_AT must be later than QUEUE_DEADLINE\n' >&2
  exit 2
fi

current_pgid=""
stop_requested=false

timestamp() { date '+%F %T %Z %z'; }
status() { printf '[%s] %s\n' "$(timestamp)" "$1" | tee -a "$STATUS_FILE"; }
process_group_alive() { [[ -n "$1" ]] && kill -0 "-$1" 2>/dev/null; }

descendant_pids() {
  local root_pid="$1" frontier="$1" all_pids="" next children pid
  while [[ -n "$frontier" ]]; do
    next=""
    for pid in $frontier; do
      children="$(ps --ppid "$pid" -o pid= 2>/dev/null || true)"
      all_pids="${all_pids} ${children}"
      next="${next} ${children}"
    done
    frontier="$next"
  done
  echo "$all_pids"
}

trainer_pids_for_tree() {
  local root_pid="$1" pid cmd
  for pid in $(descendant_pids "$root_pid"); do
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    [[ "$cmd" == *"sft.label_token_trainer"* ]] && echo "$pid"
  done
}

stop_current() {
  local reason="$1" trainers force_wait
  if ! process_group_alive "$current_pgid"; then
    return 0
  fi
  trainers="$(trainer_pids_for_tree "$current_pgid")"
  if [[ -n "$trainers" ]]; then
    status "sending SIGTERM to trainer ranks: ${reason}; pgid=${current_pgid}; pids=$(echo "$trainers" | tr '\n' ' ')"
    kill -TERM $trainers 2>/dev/null || true
  else
    status "sending SIGTERM to process group: ${reason}; pgid=${current_pgid}"
    kill -TERM "-${current_pgid}" 2>/dev/null || true
  fi
  while process_group_alive "$current_pgid"; do
    if [[ -z "$(trainer_pids_for_tree "$current_pgid")" ]]; then
      kill -TERM "-${current_pgid}" 2>/dev/null || true
    fi
    if (( $(date +%s) >= hard_epoch )); then
      status "hard deadline reached; SIGKILL pgid=${current_pgid}"
      kill -KILL "-${current_pgid}" 2>/dev/null || true
      break
    fi
    force_wait=$(( hard_epoch - $(date +%s) ))
    (( force_wait > 10 )) && force_wait=10
    (( force_wait < 1 )) && force_wait=1
    sleep "$force_wait"
  done
}

handle_stop() {
  stop_requested=true
  status "queue received stop signal"
  [[ -n "$current_pgid" ]] && stop_current "queue stop signal"
}
trap handle_stop TERM INT

run_stage() {
  local stage="$1"; shift
  local now rc=0 remaining sleep_seconds
  now="$(date +%s)"
  if (( now >= deadline_epoch )); then
    status "deadline reached before ${stage}; stage not started"
    return 124
  fi
  status "starting ${stage}; remaining_seconds=$(( deadline_epoch - now )); command=$*"
  setsid env \
    CUDA_VISIBLE_DEVICES="$GPUS" \
    PYTHON_BIN="$PYTHON_BIN" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    SAVE_LATEST_TRAIN_STATE=true \
    RESUME_LATEST_TRAIN_STATE=true \
    "$@" &
  current_pgid="$!"
  status "${stage} process_group=${current_pgid}"
  while process_group_alive "$current_pgid"; do
    if [[ "$stop_requested" == "true" ]]; then
      stop_current "queue stop requested"
      wait "$current_pgid" || true
      current_pgid=""
      return 130
    fi
    now="$(date +%s)"
    if (( now >= deadline_epoch )); then
      stop_current "graceful deadline ${QUEUE_DEADLINE} reached"
      wait "$current_pgid" || true
      current_pgid=""
      return 124
    fi
    remaining=$(( deadline_epoch - now ))
    sleep_seconds=30
    (( remaining < sleep_seconds )) && sleep_seconds="$remaining"
    (( sleep_seconds < 1 )) && sleep_seconds=1
    sleep "$sleep_seconds"
  done
  wait "$current_pgid" || rc="$?"
  status "${stage} exited code=${rc}"
  current_pgid=""
  return "$rc"
}

status "queue_id=${QUEUE_ID}"
status "gpu=${GPUS} nproc=${NPROC_PER_NODE}"
status "graceful_deadline=${QUEUE_DEADLINE}; hard_stop=${HARD_STOP_AT}"
status "cells=${CELLS}"
status "log=${QUEUE_LOG}"

if (( $(date +%s) >= deadline_epoch )); then
  status "deadline is not in the future; exiting"
  exit 2
fi

if [[ "$RUN_FROZEN_MATRIX" == "true" ]]; then
  matrix_rc=0
  run_stage frozen_shared_verifier bash "$FROZEN_SCRIPT" || matrix_rc="$?"
  if [[ "$matrix_rc" != "0" ]]; then
    status "frozen shared-verifier stage failed/stopped; no policy-matched training will start"
    exit "$matrix_rc"
  fi
fi

completed=0
failed=0
for cell in $CELLS; do
  cell_rc=0
  run_stage "lora_${cell}" bash "$CELL_SCRIPT" "$cell" || cell_rc="$?"
  if [[ "$cell_rc" == "124" || "$cell_rc" == "130" ]]; then
    status "queue stopped at cell=${cell}; rerun the same queue script to resume"
    exit "$cell_rc"
  fi
  if [[ "$cell_rc" != "0" ]]; then
    failed=$((failed + 1))
    status "cell=${cell} failed code=${cell_rc}"
    if [[ "$CONTINUE_ON_CELL_ERROR" != "true" ]]; then
      exit "$cell_rc"
    fi
    continue
  fi
  completed=$((completed + 1))
done

status "queue completed; completed_cells=${completed}; failed_cells=${failed}"
(( failed == 0 ))
