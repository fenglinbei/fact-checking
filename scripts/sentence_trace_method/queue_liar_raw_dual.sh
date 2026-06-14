#!/usr/bin/env bash
# Queue script: runs two training scripts sequentially with 8:30 AM deadline.
#
# Order:
#   1. run_v0_7_liar_raw_lora_ebs_lr_matrix.sh    (llama31_8b, EBS×LR grid)
#   2. run_liar_raw_ministral3_v0_7_adaptive5_10_lr2e5_transfer.sh  (ministral3_8b transfer)
#
# At 8:30 AM, the running script receives SIGTERM and saves training state
# to latest_state/ before exiting. Re-run this queue script to resume —
# completed work is automatically skipped.
#
# Usage:
#   bash scripts/sentence_trace_method/queue_liar_raw_dual.sh
#   bash scripts/sentence_trace_method/queue_liar_raw_dual.sh status   # check progress
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# ---- config ----
export PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"

SCRIPT_DIR="${ROOT_DIR}/scripts/sentence_trace_method"
QUEUE_STATE="${ROOT_DIR}/.queue_state_liar_raw_dual.txt"

DEADLINE_HOUR=8
DEADLINE_MINUTE=30
GRACE_PERIOD=300  # seconds to wait after SIGTERM before SIGKILL

STEP1_SCRIPT="${SCRIPT_DIR}/run_v0_7_liar_raw_lora_ebs_lr_matrix.sh"
STEP1_NAME="run_v0_7_liar_raw_lora_ebs_lr_matrix.sh"

STEP2_SCRIPT="${SCRIPT_DIR}/run_liar_raw_ministral3_v0_7_adaptive5_10_lr2e5_transfer.sh"
STEP2_NAME="run_liar_raw_ministral3_v0_7_adaptive5_10_lr2e5_transfer.sh"

# ---- helpers ----
log() {
    printf '[%s] [queue] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

# Compute the next 8:30 AM deadline as a Unix timestamp.
next_deadline_ts() {
    local now_sec today_ymd dl_sec
    now_sec=$(date +%s)
    today_ymd=$(date +%Y-%m-%d)
    dl_sec=$(date -d "${today_ymd} ${DEADLINE_HOUR}:${DEADLINE_MINUTE}:00" +%s 2>/dev/null || echo '')
    if [[ -z "$dl_sec" ]]; then
        log "ERROR: GNU date required (the -d flag is not supported by this system's date)." >&2
        exit 2
    fi
    if [[ "$now_sec" -ge "$dl_sec" ]]; then
        # Already past today's 8:30 — use tomorrow
        dl_sec=$(date -d "tomorrow ${DEADLINE_HOUR}:${DEADLINE_MINUTE}:00" +%s)
    fi
    printf '%s' "$dl_sec"
}

# Check for leftover training processes from a previous run that may still
# be saving state.  Running two training jobs simultaneously will OOM the GPUs
# and corrupt checkpoints.
check_leftover_processes() {
    local procs
    procs=$(pgrep -f 'label_token_trainer' 2>/dev/null || true)
    if [[ -n "$procs" ]]; then
        log "WARNING: Found running training process(es):"
        ps -o pid,etime,args -p $procs 2>/dev/null || true
        log "This usually means a previous run's trainer is still saving state."
        log "Wait for it to finish or kill it manually before re-running the queue."
        log "  To wait:  while pgrep -f label_token_trainer >/dev/null; do sleep 5; done"
        log "  To kill:  pkill -f label_token_trainer"
        exit 1
    fi
}

# ---- status check (--status / status) ----
show_status() {
    echo "=== Queue Status ==="
    echo "Script 1: $STEP1_NAME"
    echo "Script 2: $STEP2_NAME"
    echo "Deadline: ${DEADLINE_HOUR}:$(printf '%02d' "$DEADLINE_MINUTE") (daily)"
    echo ""
    if [[ -f "$QUEUE_STATE" ]]; then
        local s
        s=$(head -1 "$QUEUE_STATE" | tr -d '[:space:]')
        echo "State file: step $s"
    else
        echo "State file: not started"
    fi

    # Quick check: does script 1's expected LoRA training dir exist?
    local root="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
    echo ""
    echo "--- Script 1 (llama31_8b) checkpoints ---"
    for d in "$root"/liar_raw__llama31_8b__v0_7_bm_adaptive3_10_lora_ebs*/train/; do
        if [[ -d "$d" ]]; then
            local status="incomplete"
            [[ -f "${d}training_complete.json" ]] && status="COMPLETE"
            printf '  %-80s [%s]\n' "$(basename "$(dirname "$d")")" "$status"
        fi
    done

    echo ""
    echo "--- Script 2 (ministral3_8b) checkpoints ---"
    for d in "$root"/liar_raw__ministral3_8b__v0_7_bm_adaptive5_10_lora_ebs*/train/; do
        if [[ -d "$d" ]]; then
            local status="incomplete"
            [[ -f "${d}training_complete.json" ]] && status="COMPLETE"
            printf '  %-80s [%s]\n' "$(basename "$(dirname "$d")")" "$status"
        fi
    done
}

if [[ "${1:-}" == "status" || "${1:-}" == "--status" ]]; then
    show_status
    exit 0
fi

# ---- run one step with deadline enforcement ----
run_step() {
    local name="$1" script_path="$2" step_num="$3"

    echo "$step_num" > "$QUEUE_STATE"
    log "========== Step ${step_num}/2: ${name} =========="

    local dl_sec remain
    dl_sec=$(next_deadline_ts)
    remain=$(( dl_sec - $(date +%s) ))
    if [[ "$remain" -le 0 ]]; then
        log "Deadline is in the past (${remain}s). Exiting — re-run later to resume."
        exit 0
    fi
    log "Deadline: $(date -d "@${dl_sec}" '+%Y-%m-%d %H:%M:%S') — max runtime: ${remain}s"

    # timeout(1) runs the command in a new process group (default behaviour).
    # When the limit expires it sends SIGTERM to the whole group, then
    # SIGKILL after --kill-after seconds.  The training code traps SIGTERM
    # and saves latest_state/ before exiting, so the run is resumable.
    # timeout(1) behaviour recap:
    #   - Without --foreground: creates a new process group, sends SIGTERM to
    #     the whole group at expiry, then SIGKILL after --kill-after seconds.
    #   - Exit codes from timeout itself: 124 = killed by timeout's signal;
    #     137 = killed by timeout's follow-up SIGKILL.
    #   - The training code traps SIGTERM, saves latest_state/, and exits via
    #     SystemExit(143).  When the process catches the signal and exits
    #     "cleanly" (WIFEXITED, not WIFSIGNALED), timeout passes through the
    #     child's exit code, so we also treat 143 as a deadline stop.
    timeout \
        --signal=TERM \
        --kill-after="${GRACE_PERIOD}" \
        "${remain}s" \
        bash "$script_path" || {
        local ec=$?
        case "$ec" in
            124|143)
                # 124: timeout sent SIGTERM and child died from it
                # 143: trainer caught SIGTERM, saved state, exited via SystemExit
                log "Deadline reached — training state saved (exit=${ec})."
                log "Re-run this queue script to resume from step ${step_num}."
                exit 0
                ;;
            137)
                # timeout sent SIGKILL after grace period expired
                log "Deadline reached + SIGKILL after ${GRACE_PERIOD}s grace (exit=137)."
                log "Training may not have saved its final step — check latest_state/."
                exit 0
                ;;
            *)
                log "FATAL: step ${step_num} failed with exit code ${ec}"
                exit "$ec"
                ;;
        esac
    }

    log "Step ${step_num} completed within deadline."
}

# ---- main ----
check_leftover_processes

current=1
if [[ -f "$QUEUE_STATE" ]]; then
    current=$(head -1 "$QUEUE_STATE" | tr -d '[:space:]')
    case "$current" in
        1|2) ;;
        *)
            log "Invalid state file content '${current}'; starting from step 1."
            current=1
            ;;
    esac
    log "Resuming queue from step ${current}"
else
    log "Starting fresh queue"
fi

log "PYTHON_BIN=${PYTHON_BIN}"
log "OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/sentence_trace_method}"

if [[ "$current" -le 1 ]]; then
    run_step "$STEP1_NAME" "$STEP1_SCRIPT" 1
fi

if [[ "$current" -le 2 ]]; then
    run_step "$STEP2_NAME" "$STEP2_SCRIPT" 2
fi

rm -f "$QUEUE_STATE"
log "All steps completed!"
