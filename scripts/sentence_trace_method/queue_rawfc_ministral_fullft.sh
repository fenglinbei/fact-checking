#!/usr/bin/env bash
# Queue: wait for the SciFact abc cache build (PID 273410) to finish, then launch
# the Ministral-3-8B RAWFC baseline20 FullFT migrecipe training.
#
# This fills the missing cell in the 2×2 (backbone × training-mode) matrix so
# the main comparison table can report each backbone's best config without
# cherry-picking concerns.
#
# Usage (in tmux):
#   tmux new -s ministral-fullft-queue
#   bash scripts/sentence_trace_method/queue_rawfc_ministral_fullft.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

BLOCKING_PID="${BLOCKING_PID:-273410}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
LOG="${LOG:-${ROOT_DIR}/outputs/logs/queue_rawfc_ministral_fullft_$(date +%Y%m%d_%H%M%S).log}"
WRAPPER="${SCRIPT_DIR}/run_rawfc_ministral3_atom_anchor_v0_2_fullpool_minmax5_10_baseline20_fullft_migrecipe_ebs8_lr2e6_ep5_eval25.sh"

mkdir -p "$(dirname "$LOG")"
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

if [[ ! -f "$WRAPPER" ]]; then
  log "ERROR: wrapper not found: $WRAPPER"
  exit 2
fi

log "=========================================================="
log "Ministral FullFT migrecipe queue started"
log "  blocking PID : $BLOCKING_PID (SciFact abc cache)"
log "  poll interval: ${POLL_INTERVAL}s"
log "  wrapper      : $WRAPPER"
log "  log file     : $LOG"
log "=========================================================="

pid_alive() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null; }

# Phase 1: wait for blocking PID
if pid_alive "$BLOCKING_PID"; then
  log "Blocking PID $BLOCKING_PID still running. Waiting..."
  while pid_alive "$BLOCKING_PID"; do
    log "  PID $BLOCKING_PID alive. Sleeping ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
  done
  log "Blocking PID $BLOCKING_PID has finished."
  log "Waiting 15s for GPU resources to release..."
  sleep 15
else
  log "Blocking PID $BLOCKING_PID not running. Proceeding directly."
fi

# Phase 2: launch
WRAPPER_LOG="${ROOT_DIR}/outputs/logs/rawfc_ministral_fullft_migrecipe_$(date +%Y%m%d_%H%M%S).log"
log "Launching Ministral FullFT migrecipe. Output -> $WRAPPER_LOG"
setsid bash "$WRAPPER" >"$WRAPPER_LOG" 2>&1 &
CHILD=$!
log "Wrapper launched. PID=$CHILD"

# Phase 3: wait for completion (no deadline this time)
wait "$CHILD"
rc=$?
log "Wrapper finished. exit code: $rc"
log "Expected metrics: outputs/sentence_trace_method/rawfc__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10_baseline20_fullft_ebs8_lr2em6_ep5_eval25_pat8_migrecipe_rawfc/eval/test/best/label_token/metrics.json"
log "Queue done."
exit "$rc"
