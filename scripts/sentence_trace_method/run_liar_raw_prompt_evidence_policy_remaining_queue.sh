#!/usr/bin/env bash
# Run the five missing LIAR-RAW prompt-evidence policy cells against the shared
# learned-marginal fullpool trace.
#
# Usage:
#   bash scripts/sentence_trace_method/run_liar_raw_prompt_evidence_policy_remaining_queue.sh
#   MODE=build bash scripts/sentence_trace_method/run_liar_raw_prompt_evidence_policy_remaining_queue.sh
#   ONLY_STAGE=budget512 bash scripts/sentence_trace_method/run_liar_raw_prompt_evidence_policy_remaining_queue.sh
#   START_FROM=minmax5_12 bash scripts/sentence_trace_method/run_liar_raw_prompt_evidence_policy_remaining_queue.sh
#   DRY_RUN=true bash scripts/sentence_trace_method/run_liar_raw_prompt_evidence_policy_remaining_queue.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python}"
CORE_WRAPPER="${CORE_WRAPPER:-scripts/sentence_trace_method/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh}"
QUEUE_ID="${QUEUE_ID:-liar_raw_prompt_evidence_remaining_$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG_ROOT="${QUEUE_LOG_ROOT:-outputs/sentence_trace_method/queues}"
MODE="${MODE:-full}"
DRY_RUN="${DRY_RUN:-false}"
ONLY_STAGE="${ONLY_STAGE:-}"
START_FROM="${START_FROM:-}"
FORCE_BUILD="${FORCE_BUILD:-auto}"
FORCE_MREC_BUILD="${FORCE_MREC_BUILD:-false}"
FORCE_WEIGHT_TRAIN="${FORCE_WEIGHT_TRAIN:-false}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"
FORCE_EVAL="${FORCE_EVAL:-false}"
SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE:-true}"
RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE:-true}"

STAGES=(fixed3 minmax3_8 minmax5_12 budget512 budget768)
CONFIGS=(
  configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_minmax3_3.yaml
  configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_minmax3_8.yaml
  configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_minmax5_12.yaml
  configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_budget512.yaml
  configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_budget768.yaml
)
VARIANTS=(minmax3_3 minmax3_8 minmax5_12 budget512 budget768)

stage_is_known() {
  local wanted="$1" stage
  for stage in "${STAGES[@]}"; do
    [[ "$stage" == "$wanted" ]] && return 0
  done
  return 1
}

if [[ -n "$ONLY_STAGE" && -n "$START_FROM" ]]; then
  printf 'Set only one of ONLY_STAGE or START_FROM.\n' >&2
  exit 2
fi
if [[ -z "$ONLY_STAGE" && ( "$FORCE_MREC_BUILD" == "true" || "$FORCE_WEIGHT_TRAIN" == "true" ) ]]; then
  printf 'FORCE_MREC_BUILD/FORCE_WEIGHT_TRAIN may overwrite the shared trace or weights repeatedly; use ONLY_STAGE for a forced rebuild.\n' >&2
  exit 2
fi
if [[ -n "$ONLY_STAGE" ]] && ! stage_is_known "$ONLY_STAGE"; then
  printf 'Unknown ONLY_STAGE=%s. Expected one of: %s\n' "$ONLY_STAGE" "${STAGES[*]}" >&2
  exit 2
fi
if [[ -n "$START_FROM" ]] && ! stage_is_known "$START_FROM"; then
  printf 'Unknown START_FROM=%s. Expected one of: %s\n' "$START_FROM" "${STAGES[*]}" >&2
  exit 2
fi
if [[ ! -f "$CORE_WRAPPER" ]]; then
  printf 'Missing core wrapper: %s\n' "$CORE_WRAPPER" >&2
  exit 2
fi
for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    printf 'Missing policy config: %s\n' "$config" >&2
    exit 2
  fi
done

mkdir -p "$QUEUE_LOG_ROOT"
QUEUE_DIR="${QUEUE_LOG_ROOT}/${QUEUE_ID}"
mkdir -p "$QUEUE_DIR"
QUEUE_LOG="${QUEUE_DIR}/queue.log"
LOCK_FILE="${QUEUE_LOG_ROOT}/liar_raw_prompt_evidence_remaining.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'Another LIAR-RAW prompt-evidence remaining queue is running. Lock: %s\n' "$LOCK_FILE" >&2
  exit 2
fi

exec > >(tee -a "$QUEUE_LOG") 2>&1

timestamp() {
  date '+%F %T %Z %z'
}

log() {
  printf '[prompt-evidence-remaining %s] %s\n' "$(timestamp)" "$*"
}

experiment_complete() {
  local run_root="$1"
  local marker="${run_root}/train/training_complete.json"
  [[ -f "$marker" ]] || return 1
  grep -Eq '"completed"[[:space:]]*:[[:space:]]*true' "$marker" || return 1
  [[ -f "${run_root}/eval/val/best/label_token/metrics.json" ]] || return 1
  [[ -f "${run_root}/eval/test/best/label_token/metrics.json" ]] || return 1
  [[ -f "${run_root}/eval/val/best/label_token_logit_adjust_tau0p75/metrics.json" ]] || return 1
  [[ -f "${run_root}/eval/test/best/label_token_logit_adjust_tau0p75/metrics.json" ]] || return 1
}

force_requested() {
  [[ "$FORCE_BUILD" == "true" \
    || "$FORCE_MREC_BUILD" == "true" \
    || "$FORCE_WEIGHT_TRAIN" == "true" \
    || "$FORCE_TRAIN" == "true" \
    || "$FORCE_EVAL" == "true" ]]
}

run_stage() {
  local stage="$1"
  local config="$2"
  local variant="$3"
  local base="liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_${variant}"
  local run_root="outputs/sentence_trace_method/${base}_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
  local stage_log="${QUEUE_DIR}/${stage}.log"

  if [[ "$DRY_RUN" != "true" && "$MODE" == "full" ]] \
    && ! force_requested \
    && experiment_complete "$run_root"; then
    log "${stage} already complete; skipping: ${run_root}"
    return 0
  fi

  log "starting ${stage}: config=${config} run_root=${run_root}"
  if env \
    PYTHON_BIN="$PYTHON_BIN" \
    MREC_POLICY_CONFIG="$config" \
    MODE="$MODE" \
    FINETUNE_MODE=lora \
    DRY_RUN="$DRY_RUN" \
    FORCE_BUILD="$FORCE_BUILD" \
    FORCE_MREC_BUILD="$FORCE_MREC_BUILD" \
    FORCE_WEIGHT_TRAIN="$FORCE_WEIGHT_TRAIN" \
    FORCE_TRAIN="$FORCE_TRAIN" \
    FORCE_EVAL="$FORCE_EVAL" \
    SAVE_LATEST_TRAIN_STATE="$SAVE_LATEST_TRAIN_STATE" \
    RESUME_LATEST_TRAIN_STATE="$RESUME_LATEST_TRAIN_STATE" \
    REQUIRE_PROMPT_INPUT_IDS=true \
    SAMPLE_LIMIT=0 \
    bash "$CORE_WRAPPER" 2>&1 | tee "$stage_log"; then
    :
  else
    local rc=${PIPESTATUS[0]}
    log "${stage} failed with exit=${rc}; see ${stage_log}"
    return "$rc"
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    log "${stage} dry-run completed"
    return 0
  fi
  if [[ "$MODE" == "full" ]] && ! experiment_complete "$run_root"; then
    log "${stage} exited successfully but completion artifacts are incomplete: ${run_root}"
    return 1
  fi
  log "${stage} completed"
}

log "queue_id=${QUEUE_ID}"
log "mode=${MODE} dry_run=${DRY_RUN} only_stage=${ONLY_STAGE:-<none>} start_from=${START_FROM:-<first>}"
log "shared_trace=outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy_fullpool"
log "log=${QUEUE_LOG}"

start_reached=false
[[ -z "$START_FROM" ]] && start_reached=true
for index in "${!STAGES[@]}"; do
  stage="${STAGES[$index]}"
  if [[ -n "$ONLY_STAGE" && "$stage" != "$ONLY_STAGE" ]]; then
    continue
  fi
  if [[ "$start_reached" != "true" ]]; then
    if [[ "$stage" == "$START_FROM" ]]; then
      start_reached=true
    else
      log "skipping ${stage} before START_FROM=${START_FROM}"
      continue
    fi
  fi
  run_stage "$stage" "${CONFIGS[$index]}" "${VARIANTS[$index]}"
done

log "queue completed successfully"
