#!/usr/bin/env bash
# Queue: wait for the current Ministral RAWFC FullFT training (PID 537386) to
# finish, then run three tasks sequentially:
#   1. SciFact atom-union fullpool LoRA train+eval
#   2. LIAR-RAW Llama-3.1-8B LoRA (minmax5_10)
#   3. LIAR-RAW Llama-3.1-8B FullFT migrecipe (minmax5_10)
#
# Usage (in tmux):
#   tmux new -s llama-queue
#   bash scripts/sentence_trace_method/queue_llama31_liar_3tasks.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

BLOCKING_PID="${BLOCKING_PID:-537386}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
LOG="${LOG:-${ROOT_DIR}/outputs/logs/queue_llama31_liar_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }
pid_alive() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null; }

TASKS=(
  "MODE=full bash scripts/phase13_scifact/05_train_eval_scifact_atom_union_fullpool_lora.sh"
  "bash scripts/sentence_trace_method/run_liar_raw_llama31_atom_anchor_v0_2_fullpool_minmax5_10_lora_ebs16_lr2e5_ep12_eval100.sh"
  "bash scripts/sentence_trace_method/run_liar_raw_llama31_atom_anchor_v0_2_fullpool_minmax5_10_fullft_migrecipe_ebs8_lr2e6_ep5_eval25.sh"
)

log "=========================================================="
log "Llama-31 LIAR + SciFact queue started"
log "  blocking PID : $BLOCKING_PID (Ministral RAWFC FullFT)"
log "  tasks        : ${#TASKS[@]}"
log "  poll interval: ${POLL_INTERVAL}s"
log "  log file     : $LOG"
log "=========================================================="

# --- Phase 1: wait for blocking PID ---
if pid_alive "$BLOCKING_PID"; then
  log "Blocking PID $BLOCKING_PID still running. Waiting..."
  while pid_alive "$BLOCKING_PID"; do
    log "  PID $BLOCKING_PID alive. Sleeping ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
  done
  log "Blocking PID $BLOCKING_PID has finished."
  log "Waiting 20s for GPU resources to release..."
  sleep 20
else
  log "Blocking PID $BLOCKING_PID not running. Proceeding directly."
fi

# --- Phase 2: run tasks sequentially ---
for i in "${!TASKS[@]}"; do
  i0=$((i + 1))
  task="${TASKS[$i]}"
  log ""
  log "========== Task ${i0}/${#TASKS[@]}: ${task} =========="
  # Run the task; use setsid so we can identify its process group if needed.
  # Eval allows MODE=full bash ... to work for task 1.
  set +e
  setsid bash -c "cd '$ROOT_DIR' && $task" > >(tee -a "$LOG") 2>&1 &
  TASK_PID=$!
  wait "$TASK_PID"
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    log "Task ${i0} FAILED (exit code $rc). Continuing to next task."
  else
    log "Task ${i0} completed successfully."
  fi
done

log ""
log "=========================================================="
log "All ${#TASKS[@]} tasks finished."
log "Expected metrics:"
log "  SciFact:  outputs/runs/scifact_atom_union_fullpool_lora/.../metrics.json"
log "  Llama LoRA:  outputs/sentence_trace_method/liar_raw__llama31_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/eval/test/best/label_token/metrics.json"
log "  Llama FullFT: outputs/sentence_trace_method/liar_raw__llama31_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10_fullft_ebs8_lr2em6_ep5_eval25_pat8_migrecipe_liar/eval/test/best/label_token/metrics.json"
log "=========================================================="
