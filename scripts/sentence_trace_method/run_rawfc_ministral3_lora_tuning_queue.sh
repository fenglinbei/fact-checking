#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
QUEUE_ID="${QUEUE_ID:-rawfc_ministral3_lora_tuning_$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG_ROOT="${QUEUE_LOG_ROOT:-outputs/sentence_trace_method/queues}"
QUEUE_GRACE_SECONDS="${QUEUE_GRACE_SECONDS:-1800}"
POLL_SECONDS="${POLL_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-false}"

STAGE_1_SCRIPT="${STAGE_1_SCRIPT:-scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_lora_r32a64_d005_lr1e5_ep12.sh}"
STAGE_2_SCRIPT="${STAGE_2_SCRIPT:-scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_lora_r16a32_d010_lr1e5_ep12.sh}"
STAGE_3_SCRIPT="${STAGE_3_SCRIPT:-scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_lora_r16a32_d005_lr5e6_ep12.sh}"

STAGE_1_RUN_ROOT="${STAGE_1_RUN_ROOT:-outputs/sentence_trace_method/rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r32a64_d005_ebs16_lr1em5_ep12_eval50_pat8_rawfc}"
STAGE_2_RUN_ROOT="${STAGE_2_RUN_ROOT:-outputs/sentence_trace_method/rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc}"
STAGE_3_RUN_ROOT="${STAGE_3_RUN_ROOT:-outputs/sentence_trace_method/rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r16a32_d005_ebs16_lr5em6_ep12_eval50_pat8_rawfc}"

mkdir -p "$QUEUE_LOG_ROOT"
QUEUE_DIR="${QUEUE_LOG_ROOT}/${QUEUE_ID}"
mkdir -p "$QUEUE_DIR"
QUEUE_LOG="${QUEUE_DIR}/queue.log"
STATUS_FILE="${QUEUE_DIR}/status.txt"
LOCK_FILE="${QUEUE_LOG_ROOT}/rawfc_ministral3_lora_tuning.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another RAWFC Ministral3 LoRA tuning queue is already running. Lock: $LOCK_FILE" >&2
  exit 2
fi

exec > >(tee -a "$QUEUE_LOG") 2>&1

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

training_complete() {
  local run_root="$1"
  grep -Eq '"completed"[[:space:]]*:[[:space:]]*true' "${run_root}/train/training_complete.json" 2>/dev/null
}

active_pids_for_run() {
  local script_path="$1"
  local run_root="$2"
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  {
    pgrep -f "$script_path" 2>/dev/null || true
    pgrep -f "$run_root" 2>/dev/null || true
  } | sort -nu
}

wait_for_active_stage() {
  local name="$1"
  local script_path="$2"
  local run_root="$3"
  local pids

  pids="$(active_pids_for_run "$script_path" "$run_root")"
  if [[ -z "$pids" ]]; then
    return 1
  fi

  status "waiting for externally active ${name}: pids=$(echo "$pids" | tr '\n' ' ')"
  while [[ -n "$pids" ]]; do
    if [[ "$stop_requested" == "true" ]]; then
      status "queue stop requested while waiting for externally active ${name}; external process is left untouched"
      return 130
    fi
    sleep "$POLL_SECONDS"
    pids="$(active_pids_for_run "$script_path" "$run_root")"
    if [[ -n "$pids" ]]; then
      status "${name} still externally active: pids=$(echo "$pids" | tr '\n' ' ')"
    fi
  done
  status "externally active ${name} finished"
  return 0
}

run_stage() {
  local name="$1"
  local script_path="$2"
  local run_root="$3"
  local rc=0
  local wait_rc=0

  if [[ "$DRY_RUN" != "true" ]] && training_complete "$run_root"; then
    status "${name} already has completed marker; skipping: ${run_root}/train/training_complete.json"
    return 0
  fi

  wait_for_active_stage "$name" "$script_path" "$run_root" || wait_rc="$?"
  if [[ "$wait_rc" == "0" ]]; then
    if training_complete "$run_root"; then
      status "${name} external run completed: ${run_root}/train/training_complete.json"
      return 0
    fi
    status "${name} external process ended without completed marker; later stages will not start"
    return 1
  fi
  if [[ "$wait_rc" != "1" ]]; then
    return "$wait_rc"
  fi

  status "starting ${name}: ${script_path}"
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
    sleep "$POLL_SECONDS"
  done

  wait "$current_child_pid" || rc="$?"
  status "${name} exited with code ${rc}"
  current_child_pid=""
  if [[ "$rc" != "0" ]]; then
    return "$rc"
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    status "${name} dry-run completed; skipping completed-marker check"
    return 0
  fi
  if ! training_complete "$run_root"; then
    status "${name} completed process but marker is missing or incomplete: ${run_root}/train/training_complete.json"
    return 1
  fi
  status "${name} completed: ${run_root}/train/training_complete.json"
}

status "queue_id=${QUEUE_ID}"
status "cwd=${ROOT_DIR}"
status "python=${PYTHON_BIN}"
status "poll_seconds=${POLL_SECONDS}"
status "grace_seconds=${QUEUE_GRACE_SECONDS}"
status "dry_run=${DRY_RUN}"
status "log=${QUEUE_LOG}"

run_stage "rawfc_lora_r32a64_d005_lr1e5" "$STAGE_1_SCRIPT" "$STAGE_1_RUN_ROOT"
run_stage "rawfc_lora_r16a32_d010_lr1e5" "$STAGE_2_SCRIPT" "$STAGE_2_RUN_ROOT"
run_stage "rawfc_lora_r16a32_d005_lr5e6" "$STAGE_3_SCRIPT" "$STAGE_3_RUN_ROOT"

status "queue completed successfully"
