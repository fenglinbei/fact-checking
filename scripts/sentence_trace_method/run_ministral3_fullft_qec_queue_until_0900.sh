#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
QUEUE_DEADLINE="${QUEUE_DEADLINE:-$(date +%F) 09:00:00}"
QUEUE_GRACE_SECONDS="${QUEUE_GRACE_SECONDS:-1800}"
QUEUE_ID="${QUEUE_ID:-ministral3_fullft_qec_until_0900_$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG_ROOT="${QUEUE_LOG_ROOT:-outputs/sentence_trace_method/queues}"

RAWFC_SCRIPT="${RAWFC_SCRIPT:-scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_lr1e5_ep12_fullft_aligned.sh}"
QEC_SCRIPT="${QEC_SCRIPT:-scripts/sentence_trace_method/run_qec_v1_ministral3_prompt_matrix.sh}"
LIAR_SCRIPT="${LIAR_SCRIPT:-scripts/sentence_trace_method/run_liar_raw_ministral3_v0_7_adaptive5_10_lr2e5_ep12_fullft_aligned.sh}"

RAWFC_PATTERN="${RAWFC_PATTERN:-run_rawfc_ministral3_v0_7_adaptive5_10_lr1e5_ep12_fullft_aligned[.]sh}"
QEC_PATTERN="${QEC_PATTERN:-run_qec_v1_ministral3_prompt_matrix[.]sh}"
LIAR_PATTERN="${LIAR_PATTERN:-run_liar_raw_ministral3_v0_7_adaptive5_10_lr2e5_ep12_fullft_aligned[.]sh}"

RAWFC_CASE_ROOT="${RAWFC_CASE_ROOT:-outputs/sentence_trace_method/rawfc__ministral3_8b__v0_7_bm_adaptive5_10_fullft_ebs16_lr1em5_ep12_eval100_pat8_rawfc}"
RAWFC_TAUS="${RAWFC_TAUS:-0 0.5 0.75}"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "$QUEUE_LOG_ROOT"
QUEUE_DIR="${QUEUE_LOG_ROOT}/${QUEUE_ID}"
mkdir -p "$QUEUE_DIR"
QUEUE_LOG="${QUEUE_DIR}/queue.log"
STATUS_FILE="${QUEUE_DIR}/status.txt"
LOCK_FILE="${QUEUE_LOG_ROOT}/ministral3_fullft_qec_until_0900.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another Ministral fullFT/QEC queue is already running. Lock: $LOCK_FILE" >&2
  exit 2
fi

exec > >(tee -a "$QUEUE_LOG") 2>&1

deadline_epoch="$(date -d "$QUEUE_DEADLINE" +%s)"
current_child_pid=""
current_external_pgids=""
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

pgid_for_pid() {
  ps -p "$1" -o pgid= 2>/dev/null | tr -d ' ' || true
}

unique_pgids_for_pids() {
  local pid pgid
  for pid in "$@"; do
    pgid="$(pgid_for_pid "$pid")"
    [[ -n "$pgid" ]] && printf '%s\n' "$pgid"
  done | sort -u
}

stage_pids() {
  local pattern="$1"
  pgrep -f "$pattern" 2>/dev/null || true
}

terminate_external_groups() {
  local reason="$1"
  local pgid
  for pgid in $current_external_pgids; do
    terminate_process_group "$pgid" "$reason"
  done
}

handle_stop_signal() {
  stop_requested=true
  status "queue received stop signal"
  if [[ -n "$current_child_pid" ]]; then
    terminate_process_group "$current_child_pid" "queue stop signal"
  fi
  if [[ -n "$current_external_pgids" ]]; then
    terminate_external_groups "queue stop signal"
  fi
}

trap handle_stop_signal TERM INT

rawfc_fullft_complete() {
  local marker="${RAWFC_CASE_ROOT}/train/training_complete.json"
  if ! grep -Eq '"completed"[[:space:]]*:[[:space:]]*true' "$marker" 2>/dev/null; then
    status "RAWFC fullFT completion marker missing or incomplete: ${marker}"
    return 1
  fi
  if [[ ! -f "${RAWFC_CASE_ROOT}/eval/val/best/label_token/metrics.json" ]]; then
    status "RAWFC fullFT eval metrics missing: ${RAWFC_CASE_ROOT}/eval/val/best/label_token/metrics.json"
    return 1
  fi

  local tau suffix metrics_path
  for tau in $RAWFC_TAUS; do
    suffix="${tau/./p}"
    metrics_path="${RAWFC_CASE_ROOT}/eval/val/best/label_token_logit_adjust_tau${suffix}/metrics.json"
    if [[ ! -f "$metrics_path" ]]; then
      status "RAWFC fullFT tau eval metrics missing: ${metrics_path}"
      return 1
    fi
  done
}

monitor_external_stage() {
  local name="$1"
  local pattern="$2"
  local pids pgids now remaining

  pids="$(stage_pids "$pattern")"
  if [[ -z "$pids" ]]; then
    return 1
  fi

  current_external_pgids="$(unique_pgids_for_pids $pids)"
  status "waiting for active ${name} process(es): pids=$(echo "$pids" | tr '\n' ' ') pgids=$(echo "$current_external_pgids" | tr '\n' ' ')"
  while [[ -n "$(stage_pids "$pattern")" ]]; do
    if [[ "$stop_requested" == "true" ]]; then
      terminate_external_groups "${name} queue stop requested"
      current_external_pgids=""
      return 130
    fi
    now="$(date +%s)"
    if (( now >= deadline_epoch )); then
      terminate_external_groups "${name} deadline reached at ${QUEUE_DEADLINE}"
      current_external_pgids=""
      return 124
    fi
    remaining=$(( deadline_epoch - now ))
    status "${name} still running; deadline_remaining_seconds=${remaining}; pids=$(echo "$(stage_pids "$pattern")" | tr '\n' ' ')"
    if (( remaining < POLL_SECONDS )); then
      sleep "$remaining"
    else
      sleep "$POLL_SECONDS"
    fi
  done

  current_external_pgids=""
  status "active ${name} process finished"
  return 0
}

run_stage_until_deadline() {
  local name="$1"
  local script_path="$2"
  local now remaining rc=0

  now="$(date +%s)"
  if (( now >= deadline_epoch )); then
    status "deadline already reached before ${name}; not starting ${script_path}"
    return 124
  fi

  remaining=$(( deadline_epoch - now ))
  status "starting ${name}: ${script_path}"
  status "deadline: ${QUEUE_DEADLINE}; remaining_seconds=${remaining}"
  setsid env \
    PYTHON_BIN="$PYTHON_BIN" \
    SAVE_LATEST_TRAIN_STATE=true \
    RESUME_LATEST_TRAIN_STATE=true \
    bash "$script_path" &
  current_child_pid="$!"
  status "${name} process_group=${current_child_pid}"

  while process_group_alive "$current_child_pid"; do
    if [[ "$stop_requested" == "true" ]]; then
      terminate_process_group "$current_child_pid" "${name} queue stop requested"
      wait "$current_child_pid" || true
      current_child_pid=""
      return 130
    fi
    now="$(date +%s)"
    if (( now >= deadline_epoch )); then
      terminate_process_group "$current_child_pid" "${name} deadline reached at ${QUEUE_DEADLINE}"
      wait "$current_child_pid" || true
      current_child_pid=""
      return 124
    fi
    remaining=$(( deadline_epoch - now ))
    if (( remaining < POLL_SECONDS )); then
      sleep "$remaining"
    else
      sleep "$POLL_SECONDS"
    fi
  done

  wait "$current_child_pid" || rc="$?"
  status "${name} exited with code ${rc}"
  current_child_pid=""
  return "$rc"
}

run_or_wait_stage() {
  local name="$1"
  local script_path="$2"
  local pattern="$3"
  local rc=0

  if monitor_external_stage "$name" "$pattern"; then
    return 0
  fi
  rc="$?"
  if [[ "$rc" != "1" ]]; then
    return "$rc"
  fi
  run_stage_until_deadline "$name" "$script_path"
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

rawfc_rc=0
run_or_wait_stage "rawfc_fullft" "$RAWFC_SCRIPT" "$RAWFC_PATTERN" || rawfc_rc="$?"
if [[ "$rawfc_rc" != "0" ]]; then
  status "rawfc_fullft did not complete cleanly; later stages will not start"
  exit "$rawfc_rc"
fi
rawfc_fullft_complete

qec_rc=0
run_or_wait_stage "qec_prompt_matrix" "$QEC_SCRIPT" "$QEC_PATTERN" || qec_rc="$?"
if [[ "$qec_rc" != "0" ]]; then
  status "qec_prompt_matrix did not complete cleanly; LIAR fullFT will not start"
  exit "$qec_rc"
fi

liar_rc=0
run_or_wait_stage "liar_raw_fullft" "$LIAR_SCRIPT" "$LIAR_PATTERN" || liar_rc="$?"
if [[ "$liar_rc" != "0" ]]; then
  status "liar_raw_fullft did not complete cleanly"
  exit "$liar_rc"
fi

status "queue completed successfully"
