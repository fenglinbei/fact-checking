#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"
QUEUE_DEADLINE="${QUEUE_DEADLINE:-$(date +%F) 10:00:00}"
QUEUE_GRACE_SECONDS="${QUEUE_GRACE_SECONDS:-1800}"
QUEUE_ID="${QUEUE_ID:-rawfc_then_liar_until_1000_$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG_ROOT="${QUEUE_LOG_ROOT:-outputs/sentence_trace_method/queues}"
RAWFC_SCRIPT="${RAWFC_SCRIPT:-scripts/sentence_trace_method/run_rawfc_lora_selector_lr_matrix.sh}"
LIAR_SCRIPT="${LIAR_SCRIPT:-scripts/sentence_trace_method/run_v0_7_liar_raw_lora_ebs_lr_matrix.sh}"
RAWFC_POLL_SECONDS="${RAWFC_POLL_SECONDS:-60}"
VERIFY_RAWFC_COMPLETE="${VERIFY_RAWFC_COMPLETE:-true}"

mkdir -p "$QUEUE_LOG_ROOT"
QUEUE_DIR="${QUEUE_LOG_ROOT}/${QUEUE_ID}"
mkdir -p "$QUEUE_DIR"
QUEUE_LOG="${QUEUE_DIR}/queue.log"
STATUS_FILE="${QUEUE_DIR}/status.txt"
LOCK_FILE="${QUEUE_LOG_ROOT}/rawfc_then_liar_until_1000.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another RAWFC -> LIAR queue is already running. Lock: $LOCK_FILE" >&2
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

rawfc_pids() {
  pgrep -f 'run_rawfc_lora_selector_lr_matrix[.]sh' || true
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
    kill -TERM $trainer_pids 2>/dev/null || true
  else
    status "sending SIGTERM to process group ${pgid}: ${reason}"
    kill -TERM "-${pgid}" 2>/dev/null || true
  fi

  local force_epoch
  force_epoch=$(( $(date +%s) + QUEUE_GRACE_SECONDS ))
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

wait_for_rawfc() {
  local pids now remaining
  pids="$(rawfc_pids)"
  if [[ -z "$pids" ]]; then
    status "no active RAWFC matrix process found"
    return 0
  fi

  status "waiting for active RAWFC matrix process(es): $(echo "$pids" | tr '\n' ' ')"
  while [[ -n "$pids" ]]; do
    if [[ "$stop_requested" == "true" ]]; then
      status "queue stop requested while waiting for RAWFC; RAWFC process is left untouched"
      return 130
    fi
    now="$(date +%s)"
    remaining=$(( deadline_epoch - now ))
    status "RAWFC still running; deadline_remaining_seconds=${remaining}; pids=$(echo "$pids" | tr '\n' ' ')"
    sleep "$RAWFC_POLL_SECONDS"
    pids="$(rawfc_pids)"
  done
  status "RAWFC matrix process finished"
}

verify_rawfc_complete() {
  [[ "$VERIFY_RAWFC_COMPLETE" == "true" ]] || return 0

  local selectors lrs selector lr run missing=()
  selectors=(
    "__old_adaptive5_10"
    "__v0_7_bm_adaptive3_10"
    "__v0_7_bm_adaptive5_10"
    "__v0_7_bm_adaptive5_12"
  )
  lrs=(
    "_lora_ebs16_lr1em5_ep8_eval100_pat8_rawfc"
    "_lora_ebs16_lr5em6_ep8_eval100_pat8_rawfc"
  )

  for selector in "${selectors[@]}"; do
    for lr in "${lrs[@]}"; do
      run="outputs/sentence_trace_method/rawfc__llama31_8b${selector}${lr}"
      if ! grep -Eq '"completed"[[:space:]]*:[[:space:]]*true' "${run}/train/training_complete.json" 2>/dev/null; then
        missing+=("${run}/train/training_complete.json")
      fi
      if [[ ! -f "${run}/eval/val/best/label_token/metrics.json" ]]; then
        missing+=("${run}/eval/val/best/label_token/metrics.json")
      fi
      if [[ ! -f "${run}/eval/val/best/label_token_logit_adjust_tau0/metrics.json" ]]; then
        missing+=("${run}/eval/val/best/label_token_logit_adjust_tau0/metrics.json")
      fi
      if [[ ! -f "${run}/eval/val/best/label_token_logit_adjust_tau0p5/metrics.json" ]]; then
        missing+=("${run}/eval/val/best/label_token_logit_adjust_tau0p5/metrics.json")
      fi
      if [[ ! -f "${run}/eval/val/best/label_token_logit_adjust_tau0p75/metrics.json" ]]; then
        missing+=("${run}/eval/val/best/label_token_logit_adjust_tau0p75/metrics.json")
      fi
    done
  done

  if ((${#missing[@]})); then
    status "RAWFC matrix is not fully complete; LIAR-RAW will not start"
    printf '%s\n' "${missing[@]}" | sed 's/^/[missing] /'
    return 1
  fi
  status "RAWFC matrix completion check passed"
}

run_liar_until_deadline() {
  local now remaining rc=0
  now="$(date +%s)"
  if (( now >= deadline_epoch )); then
    status "deadline already reached before LIAR-RAW start; not starting ${LIAR_SCRIPT}"
    return 124
  fi

  remaining=$(( deadline_epoch - now ))
  status "starting LIAR-RAW matrix: ${LIAR_SCRIPT}"
  status "deadline: ${QUEUE_DEADLINE}; remaining_seconds=${remaining}"
  setsid env \
    PYTHON_BIN="$PYTHON_BIN" \
    SAVE_LATEST_TRAIN_STATE=true \
    RESUME_LATEST_TRAIN_STATE=true \
    bash "$LIAR_SCRIPT" &
  current_child_pid="$!"
  status "LIAR-RAW process_group=${current_child_pid}"

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
    remaining=$(( deadline_epoch - now ))
    if (( remaining < 30 )); then
      sleep "$remaining"
    else
      sleep 30
    fi
  done

  wait "$current_child_pid" || rc="$?"
  status "LIAR-RAW matrix exited with code ${rc}"
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

wait_for_rawfc
verify_rawfc_complete
run_liar_until_deadline
status "queue completed successfully"
