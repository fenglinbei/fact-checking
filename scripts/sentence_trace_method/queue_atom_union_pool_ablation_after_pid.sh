#!/usr/bin/env bash
# Wait for an existing GPU job to finish, then run the complete LIAR-RAW
# Atom-Union pool ablation (build, fixed-checkpoint eval, and summary).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

BLOCKING_PID="${BLOCKING_PID:?Set BLOCKING_PID to the active wrapper PID.}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
LOG="${LOG:-${ROOT_DIR}/outputs/logs/atom_union_pool_ablation_after_${BLOCKING_PID}_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
pid_alive() { kill -0 "$1" 2>/dev/null; }

log "Atom-Union pool ablation queue started"
log "blocking PID: $BLOCKING_PID"
log "log file: $LOG"

while pid_alive "$BLOCKING_PID"; do
  log "PID $BLOCKING_PID is still running; sleeping ${POLL_INTERVAL}s"
  sleep "$POLL_INTERVAL"
done

log "PID $BLOCKING_PID finished; waiting 20s for GPU resources to release"
sleep 20
log "Starting full Atom-Union pool ablation"

set +e
MODE=full bash scripts/sentence_trace_method/run_liar_raw_atom_union_pool_ablation.sh 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e

if [[ "$rc" -eq 0 ]]; then
  log "Atom-Union pool ablation completed successfully"
else
  log "Atom-Union pool ablation failed with exit code $rc"
fi
exit "$rc"
