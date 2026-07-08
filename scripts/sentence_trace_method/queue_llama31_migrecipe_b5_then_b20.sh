#!/usr/bin/env bash
# Queue script: wait for the currently-running baseline5 migrecipe fullft job
# to finish, then automatically launch the baseline20 migrecipe fullft job.
#
# Why a poll loop instead of `&&`: the baseline5 job is ALREADY running in
# another shell (started before this queue). This script detects it, waits for
# it to exit, then starts baseline20. Both jobs use all 4 GPUs (~45GB each),
# so they must NOT overlap.
#
# Usage:
#   bash scripts/sentence_trace_method/queue_llama31_migrecipe_b5_then_b20.sh
#   bash scripts/sentence_trace_method/queue_llama31_migrecipe_b5_then_b20.sh status
#
# Logs to: outputs/logs/queue_llama31_migrecipe_b5_then_b20.log
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCRIPT_DIR="${ROOT_DIR}/scripts/sentence_trace_method"
LOG_DIR="${ROOT_DIR}/outputs/logs"
mkdir -p "$LOG_DIR"
QUEUE_LOG="${LOG_DIR}/queue_llama31_migrecipe_b5_then_b20.log"
QUEUE_STATE="${ROOT_DIR}/.queue_state_llama31_migrecipe.txt"

STEP1_SCRIPT="${SCRIPT_DIR}/run_rawfc_llama31_atom_anchor_v0_2_fullpool_minmax5_10_fullft_migrecipe_ebs8_lr2e6_ep5_eval25.sh"
STEP1_NAME="run_rawfc_llama31_atom_anchor_v0_2_fullpool_minmax5_10_fullft_migrecipe_ebs8_lr2e6_ep5_eval25.sh"
STEP2_SCRIPT="${SCRIPT_DIR}/run_rawfc_llama31_atom_anchor_v0_2_fullpool_minmax5_10_baseline20_fullft_migrecipe_ebs8_lr2e6_ep5_eval25.sh"
STEP2_NAME="run_rawfc_llama31_atom_anchor_v0_2_fullpool_minmax5_10_baseline20_fullft_migrecipe_ebs8_lr2e6_ep5_eval25.sh"

POLL_INTERVAL=60  # seconds between checks while waiting

log() {
    local msg
    msg="$(printf '[%s] [queue] %s' "$(date '+%Y-%m-%d %H:%M:%S')" "$*")"
    printf '%s\n' "$msg" | tee -a "$QUEUE_LOG"
}

# Count running instances of a wrapper script (match on the script basename
# via the bash invocation, excluding this queue script and grep itself).
running_count() {
    local script_name="$1"
    ps -eo pid,cmd | grep -F "$script_name" \
        | grep -v -F "queue_llama31_migrecipe_b5_then_b20" \
        | grep -v -F "grep" \
        | grep -c -F "$script_name" || true
}

# --- status subcommand: report what is running and queue progress ---
if [[ "${1:-}" == "status" ]]; then
    b5=$(running_count "$(basename "$STEP1_SCRIPT")")
    b20=$(running_count "$(basename "$STEP2_SCRIPT")")
    printf 'queue log:      %s\n' "$QUEUE_LOG"
    printf 'queue state:    %s\n' "$QUEUE_STATE"
    [[ -f "$QUEUE_STATE" ]] && cat "$QUEUE_STATE" || printf '(no state file yet)\n'
    printf 'baseline5 running: %s\n' "$b5"
    printf 'baseline20 running: %s\n' "$b20"
    exit 0
fi

log "===== llama31 migrecipe queue started (PID $$) ====="
log "step1 (wait/ensure): $STEP1_NAME"
log "step2 (queued):      $STEP2_NAME"
log "poll interval: ${POLL_INTERVAL}s   log: $QUEUE_LOG"

# ---- STEP 1: wait for the already-running baseline5 job to finish ----
# We do NOT launch step1 here — it is already running in another shell.
# We only wait. If it is somehow not running, we skip straight to step2
# (so the queue still makes progress) but warn loudly.
b5_running=$(running_count "$(basename "$STEP1_SCRIPT")")
if [[ "$b5_running" -eq 0 ]]; then
    log "WARN: baseline5 job ('$STEP1_NAME') is not currently running."
    log "      Assuming it already finished (or was never started). Proceeding to step2."
    echo "step1=skipped_not_running ts=$(date +%s)" > "$QUEUE_STATE"
else
    log "baseline5 job detected ($b5_running instance(s)). Waiting for it to finish..."
    echo "step1=waiting ts=$(date +%s)" > "$QUEUE_STATE"
    while true; do
        b5_running=$(running_count "$(basename "$STEP1_SCRIPT")")
        if [[ "$b5_running" -eq 0 ]]; then
            break
        fi
        sleep "$POLL_INTERVAL"
    done
    log "baseline5 job finished."
    echo "step1=done ts=$(date +%s)" > "$QUEUE_STATE"
fi

# Brief settle delay: let the previous job release GPU memory and close files
# (training/eval writes can still be flushing when the wrapper process exits).
log "settling 30s to let GPU memory and file handles release..."
sleep 30

# Sanity check: no leftover GPU-hungry training processes before we start.
# `label_token_trainer` / `label_token_infer` are the actual GPU consumers.
leftover=$(ps -eo pid,cmd | grep -E "label_token_trainer|label_token_infer" | grep -v grep || true)
if [[ -n "$leftover" ]]; then
    log "WARN: GPU training process still alive after baseline5 exited:"
    printf '%s\n' "$leftover" | tee -a "$QUEUE_LOG"
    log "      Waiting additional 60s for it to exit..."
    sleep 60
    leftover=$(ps -eo pid,cmd | grep -E "label_token_trainer|label_token_infer" | grep -v grep || true)
    if [[ -n "$leftover" ]]; then
        log "ERROR: training process still running after grace period. Aborting step2 to avoid GPU OOM."
        printf '%s\n' "$leftover" | tee -a "$QUEUE_LOG"
        echo "step2=aborted_leftover_training ts=$(date +%s)" >> "$QUEUE_STATE"
        exit 3
    fi
fi

# ---- STEP 2: launch baseline20 ----
log "launching step2 (baseline20 migrecipe fullft)..."
echo "step2=starting ts=$(date +%s)" >> "$QUEUE_STATE"
# Run step2 in the foreground so its stdout/stderr stream to this terminal too,
# and tee a copy to the queue log for post-mortem.
set +e
bash "$STEP2_SCRIPT" 2>&1 | tee -a "$QUEUE_LOG"
step2_rc=${PIPESTATUS[0]}
set -e
if [[ "$step2_rc" -eq 0 ]]; then
    log "step2 (baseline20) completed successfully (rc=0)."
    echo "step2=done rc=0 ts=$(date +%s)" >> "$QUEUE_STATE"
else
    log "step2 (baseline20) FAILED with rc=$step2_rc."
    echo "step2=failed rc=$step2_rc ts=$(date +%s)" >> "$QUEUE_STATE"
fi

log "===== queue finished ====="
