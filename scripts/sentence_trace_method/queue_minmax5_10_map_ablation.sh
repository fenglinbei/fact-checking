#!/usr/bin/env bash
# Queue: wait for the running HoVer verifier training (PID 1596978) to finish,
# then launch the minmax5_10 map-ablation wrapper, and stop it safely by 12:00.
#
# Behavior:
#   1. Poll PID 1596978 every 60s until it exits (or 12:00, whichever first).
#   2. Once the GPU is free, start the map-ablation wrapper in its own session
#      (setsid) so we can kill the whole process tree cleanly.
#   3. If 12:00 is reached while map-ablation is running, send SIGTERM to the
#      entire process group, wait 30s for graceful shutdown, then SIGKILL.
#   4. If the HoVer training is STILL running at 12:00, do not launch at all.
#
# Resume next day is manual: just re-run the wrapper (it skips existing
# weights/traces by default).
#
# Usage (run in a tmux session so it survives shell disconnect):
#   tmux new -s mrec-queue
#   bash scripts/sentence_trace_method/queue_minmax5_10_map_ablation.sh
#   # then Ctrl-B d to detach; tail -f the log to watch progress.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

# --- config (override via env) ---
BLOCKING_PID="${BLOCKING_PID:-1596978}"          # HoVer verifier training main PID
DEADLINE="${DEADLINE:-12:00}"                    # stop-by time today (HH:MM)
POLL_INTERVAL="${POLL_INTERVAL:-60}"             # seconds between PID checks
LOG="${LOG:-${ROOT_DIR}/outputs/logs/mrec_queue_$(date +%Y%m%d_%H%M%S).log}"
WRAPPER="${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_minmax5_10_map_ablation_lora_ebs16_lr2e5_ep12_eval100.sh"

mkdir -p "$(dirname "$LOG")"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

# Compute the deadline as a Unix timestamp today.
deadline_ts() {
  date -d "today ${DEADLINE}" +%s
}
now_ts() { date +%s; }

seconds_until_deadline() {
  local d n
  d=$(deadline_ts); n=$(now_ts)
  echo $(( d - n ))
}

pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

kill_tree() {
  # Kill an entire process group, gracefully then forcefully.
  local pgid="$1" sig="$2"
  # Negative PGID kills the whole group.
  kill -"$sig" -- -"$pgid" 2>/dev/null || true
}

# --- sanity checks ---
if [[ ! -f "$WRAPPER" ]]; then
  log "ERROR: wrapper not found: $WRAPPER"
  exit 2
fi
if ! deadline_ts >/dev/null 2>&1; then
  log "ERROR: invalid DEADLINE='$DEADLINE' (use HH:MM)"
  exit 2
fi

log "=========================================================="
log "MREC queue started"
log "  blocking PID : $BLOCKING_PID"
log "  deadline     : today $DEADLINE ($(date -d "today $DEADLINE" '+%H:%M:%S'))"
log "  poll interval: ${POLL_INTERVAL}s"
log "  wrapper      : $WRAPPER"
log "  log file     : $LOG"
log "=========================================================="

# --- phase 1: wait for the blocking PID to finish (or deadline) ---
if pid_alive "$BLOCKING_PID"; then
  log "Blocking PID $BLOCKING_PID still running. Waiting for it to finish..."
  while pid_alive "$BLOCKING_PID"; do
    remaining=$(seconds_until_deadline)
    if [[ "$remaining" -le 0 ]]; then
      log "Deadline $DEADLINE reached while STILL waiting for PID $BLOCKING_PID."
      log "Map-ablation was NOT launched. The blocking job is left untouched."
      log "To resume later: re-run this script (or the wrapper directly) once GPU is free."
      exit 0
    fi
    log "  PID $BLOCKING_PID alive; ${remaining}s left until $DEADLINE. Sleeping ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
  done
  log "Blocking PID $BLOCKING_PID has finished."
  # Give the GPU a moment to fully release (VRAM teardown).
  log "Waiting 30s for GPU resources to release..."
  sleep 30
else
  log "Blocking PID $BLOCKING_PID not running. Proceeding directly."
fi

# --- phase 2: launch the wrapper in its own session/process-group ---
remaining=$(seconds_until_deadline)
if [[ "$remaining" -le 300 ]]; then
  log "Only ${remaining}s left until $DEADLINE — not enough time to start. Aborting."
  log "Run the wrapper manually later: bash $WRAPPER"
  exit 0
fi

WRAPPER_LOG="${ROOT_DIR}/outputs/logs/map_ablation_minmax5_10_$(date +%Y%m%d_%H%M%S).log"
log "Launching map-ablation wrapper. Output -> $WRAPPER_LOG"

# setsid creates a new session so the wrapper + all its children (accelerate,
# vllm, label_token_trainer) share one process group = the setsid child's PID.
setsid bash "$WRAPPER" >"$WRAPPER_LOG" 2>&1 &
QUEUE_CHILD=$!
WRAPPER_PGID=$QUEUE_CHILD   # setsid child is the process-group leader
log "Wrapper launched. queue child PID=$QUEUE_CHILD PGID=$WRAPPER_PGID"

# --- phase 3: monitor; kill at deadline if still running ---
while true; do
  if ! kill -0 "$QUEUE_CHILD" 2>/dev/null; then
    log "Wrapper finished on its own before the deadline."
    wait "$QUEUE_CHILD" 2>/dev/null || true
    rc=$?
    log "Wrapper exit code: $rc"
    log "Done. Metrics should be under the paths printed at the end of $WRAPPER_LOG"
    exit "$rc"
  fi
  remaining=$(seconds_until_deadline)
  if [[ "$remaining" -le 0 ]]; then
    break
  fi
  # Sleep in small increments so we react to wrapper completion promptly.
  step=$(( remaining < POLL_INTERVAL ? remaining : POLL_INTERVAL ))
  step=${step/#-/0}; step=$(( step < 10 ? 10 : step ))
  sleep "$step"
done

# --- phase 4: deadline reached, wrapper still running -> safe stop ---
log "Deadline $DEADLINE reached. Stopping wrapper process tree (PGID=$WRAPPER_PGID)..."
log "Sending SIGTERM (graceful, 30s window)..."
kill_tree "$WRAPPER_PGID" TERM
for _ in $(seq 1 30); do
  if ! kill -0 "$QUEUE_CHILD" 2>/dev/null; then
    break
  fi
  sleep 1
done

if kill -0 "$QUEUE_CHILD" 2>/dev/null; then
  # Some children (vllm server, accelerate workers) may linger; force-kill.
  log "Still alive after 30s. Sending SIGKILL to process group..."
  kill_tree "$WRAPPER_PGID" KILL
  sleep 3
fi

# Final sweep: kill any stray accelerator/vllm/python children of this group.
log "Sweeping any stray child processes..."
pkill -KILL -g "$WRAPPER_PGID" 2>/dev/null || true

wait "$QUEUE_CHILD" 2>/dev/null || true
log "Wrapper stopped. Partial results may exist for the in-progress variant."
log "GPU should now be free. Check: nvidia-smi"
log "To resume: bash $WRAPPER  (it skips existing weights/traces automatically)"
log "Queue finished (stopped at deadline)."
