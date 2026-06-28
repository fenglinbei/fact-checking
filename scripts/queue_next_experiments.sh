#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Queue script: wait for current training, then run 2 jobs in sequence
# ============================================================

WATCH_PIDS=(3520632 3521351 3521353 3521355 3521357)
CHECK_INTERVAL=30  # seconds between checks
LOG_FILE="/data/liaozijie/fact-checking/scripts/queue_next_experiments.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

wait_for_pids() {
  log "Watching PIDs: ${WATCH_PIDS[*]}"
  while true; do
    local all_dead=true
    for pid in "${WATCH_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        all_dead=false
        break
      fi
    done
    if $all_dead; then
      log "All watched processes have exited."
      return 0
    fi
    log "Processes still running, sleeping ${CHECK_INTERVAL}s..."
    sleep "$CHECK_INTERVAL"
  done
}

run_job1() {
  log "===== Job 1: atom-anchor v0.2 learned-marginal-proxy budget1024 ====="
  cd /data/liaozijie/fact-checking
  EXPECTED_WEIGHT_FINGERPRINT=73e064c851af \
    MODE=full \
    FORCE_MREC_BUILD=true \
    bash scripts/sentence_trace_method/run_liar_raw_ministral3_atom_anchor_v0_2_learned_marginal_proxy_budget1024_lora_ebs16_lr2e5_ep12_eval100.sh
  log "===== Job 1 completed ====="
}

run_job2() {
  log "===== Job 2: MedFact qwen3-14b full pipeline ====="
  cd /data/liaozijie/MedFact
  NPROC_PER_NODE=4 bash scripts/run_qwen3_14b_no_sibling_full_pipeline.sh
  log "===== Job 2 completed ====="
}

main() {
  log "Queue started. Will execute after current training finishes."
  wait_for_pids
  run_job1
  run_job2
  log "All jobs completed."
}

main "$@"
