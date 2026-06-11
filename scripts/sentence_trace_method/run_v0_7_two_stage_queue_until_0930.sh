#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"
QUEUE_DEADLINE="${QUEUE_DEADLINE:-$(date +%F) 09:30:00}"
QUEUE_GRACE_SECONDS="${QUEUE_GRACE_SECONDS:-1800}"
QUEUE_ID="${QUEUE_ID:-v0_7_two_stage_lora_$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG_ROOT="${QUEUE_LOG_ROOT:-outputs/sentence_trace_method/queues}"
GROUP1_SCRIPT="${GROUP1_SCRIPT:-scripts/sentence_trace_method/run_v0_7_lora_matrix_halfbatch_ep8_eval100_pat8.sh}"
GROUP2_SCRIPT="${GROUP2_SCRIPT:-scripts/sentence_trace_method/run_v0_7_coverage_lora_matrix_halfbatch_ep8_eval100_pat8.sh}"

mkdir -p "$QUEUE_LOG_ROOT"
QUEUE_DIR="${QUEUE_LOG_ROOT}/${QUEUE_ID}"
mkdir -p "$QUEUE_DIR"
QUEUE_LOG="${QUEUE_DIR}/queue.log"
STATUS_FILE="${QUEUE_DIR}/status.txt"
LOCK_FILE="${QUEUE_LOG_ROOT}/v0_7_two_stage_lora.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another v0.7 two-stage LoRA queue is already running. Lock: $LOCK_FILE" >&2
  exit 2
fi

exec > >(tee -a "$QUEUE_LOG") 2>&1

deadline_epoch="$(date -d "$QUEUE_DEADLINE" +%s)"
current_child_pid=""
stop_requested=false

timestamp() {
  date '+%F %T %Z %z'
}

status() {
  local message="$1"
  printf '[%s] %s\n' "$(timestamp)" "$message" | tee -a "$STATUS_FILE"
}

process_group_alive() {
  local pgid="$1"
  kill -0 "-${pgid}" 2>/dev/null
}

descendant_pids() {
  local root_pid="$1"
  local frontier="$root_pid"
  local all_pids=""
  local next_frontier children pid
  while [[ -n "$frontier" ]]; do
    next_frontier=""
    for pid in $frontier; do
      children="$(ps --ppid "$pid" -o pid= 2>/dev/null || true)"
      if [[ -n "$children" ]]; then
        all_pids="${all_pids} ${children}"
        next_frontier="${next_frontier} ${children}"
      fi
    done
    frontier="$next_frontier"
  done
  echo "$all_pids"
}

trainer_pids_for_tree() {
  local root_pid="$1"
  local pid ppid cmd parent_cmd
  for pid in $(descendant_pids "$root_pid"); do
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    [[ "$cmd" == *"sft.label_token_trainer"* ]] || continue
    ppid="$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ' || true)"
    parent_cmd="$(ps -p "$ppid" -o args= 2>/dev/null || true)"
    if [[ "$parent_cmd" == *"accelerate launch"* ]]; then
      echo "$pid"
    fi
  done
}

terminate_process_group() {
  local pgid="$1"
  local reason="$2"
  if ! process_group_alive "$pgid"; then
    return 0
  fi

  local trainer_pids
  trainer_pids="$(trainer_pids_for_tree "$pgid")"
  if [[ -n "$trainer_pids" ]]; then
    status "sending SIGTERM to trainer process(es) in group ${pgid}: ${reason}; pids=$(echo "$trainer_pids" | tr '\n' ' ')"
    # Signal the Python trainer workers first so their SIGTERM handler can save latest_state.
    kill -TERM $trainer_pids 2>/dev/null || true
  else
    status "sending SIGTERM to process group ${pgid}: ${reason}"
    kill -TERM "-${pgid}" 2>/dev/null || true
  fi

  local force_epoch=$(( $(date +%s) + QUEUE_GRACE_SECONDS ))
  while process_group_alive "$pgid"; do
    if [[ -z "$(trainer_pids_for_tree "$pgid")" ]]; then
      kill -TERM "-${pgid}" 2>/dev/null || true
    fi
    if (( $(date +%s) >= force_epoch )); then
      status "process group ${pgid} still alive after ${QUEUE_GRACE_SECONDS}s; sending SIGKILL"
      kill -KILL "-${pgid}" 2>/dev/null || true
      break
    fi
    sleep 10
  done
}

handle_stop_signal() {
  stop_requested=true
  status "queue received stop signal"
  if [[ -n "$current_child_pid" ]]; then
    terminate_process_group "$current_child_pid" "queue stop signal"
  fi
}

trap handle_stop_signal TERM INT

run_stage() {
  local name="$1"
  local script_path="$2"
  local prepare_sources="$3"
  local now
  now="$(date +%s)"
  if (( now >= deadline_epoch )); then
    status "deadline already reached before ${name}; not starting ${script_path}"
    return 124
  fi

  status "starting ${name}: ${script_path}"
  status "deadline: ${QUEUE_DEADLINE}; remaining_seconds=$(( deadline_epoch - now ))"
  setsid env \
    PYTHON_BIN="$PYTHON_BIN" \
    SAVE_LATEST_TRAIN_STATE=true \
    RESUME_LATEST_TRAIN_STATE=true \
    PREPARE_V0_7_SOURCES="$prepare_sources" \
    bash "$script_path" &
  current_child_pid="$!"
  status "${name} process_group=${current_child_pid}"

  while process_group_alive "$current_child_pid"; do
    if [[ "$stop_requested" == "true" ]]; then
      terminate_process_group "$current_child_pid" "queue stop requested"
      wait "$current_child_pid" || true
      current_child_pid=""
      return 130
    fi
    now="$(date +%s)"
    if (( now >= deadline_epoch )); then
      terminate_process_group "$current_child_pid" "deadline reached at ${QUEUE_DEADLINE}"
      wait "$current_child_pid" || true
      current_child_pid=""
      return 124
    fi
    local sleep_seconds=30
    local remaining=$(( deadline_epoch - now ))
    if (( remaining < sleep_seconds )); then
      sleep_seconds="$remaining"
    fi
    if (( sleep_seconds < 1 )); then
      sleep_seconds=1
    fi
    sleep "$sleep_seconds"
  done

  local rc=0
  wait "$current_child_pid" || rc="$?"
  status "${name} exited with code ${rc}"
  current_child_pid=""
  return "$rc"
}

status "queue_id=${QUEUE_ID}"
status "cwd=${ROOT_DIR}"
status "python=${PYTHON_BIN}"
status "deadline=${QUEUE_DEADLINE} epoch=${deadline_epoch}"
status "grace_seconds=${QUEUE_GRACE_SECONDS}"
status "log=${QUEUE_LOG}"

if (( $(date +%s) >= deadline_epoch )); then
  status "deadline is not in the future; exiting without starting experiments"
  exit 2
fi

group1_rc=0
run_stage "group1_v0_7_lora" "$GROUP1_SCRIPT" "${PREPARE_V0_7_SOURCES_FIRST:-true}" || group1_rc="$?"
if [[ "$group1_rc" != "0" ]]; then
  status "group1 did not complete cleanly; group2 will not start"
  exit "$group1_rc"
fi

group2_rc=0
run_stage "group2_v0_7_coverage_lora" "$GROUP2_SCRIPT" "${PREPARE_V0_7_SOURCES_SECOND:-false}" || group2_rc="$?"
if [[ "$group2_rc" != "0" ]]; then
  status "group2 did not complete cleanly"
  exit "$group2_rc"
fi

status "queue completed successfully"
