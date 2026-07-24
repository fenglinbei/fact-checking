#!/usr/bin/env bash
# Run the corrected verifier-visible prompt-budget sweep against the shared
# learned-marginal fullpool selector trace.
#
# Usage:
#   bash scripts/sentence_trace_method/run_liar_raw_prompt_budget_corrected_queue.sh
#   MODE=build bash scripts/sentence_trace_method/run_liar_raw_prompt_budget_corrected_queue.sh
#   ONLY_STAGE=promptbudget512 bash scripts/sentence_trace_method/run_liar_raw_prompt_budget_corrected_queue.sh
#   START_FROM=promptbudget768 bash scripts/sentence_trace_method/run_liar_raw_prompt_budget_corrected_queue.sh
#   DRY_RUN=true bash scripts/sentence_trace_method/run_liar_raw_prompt_budget_corrected_queue.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python}"
CORE_WRAPPER="${CORE_WRAPPER:-scripts/sentence_trace_method/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh}"
QUEUE_ID="${QUEUE_ID:-liar_raw_prompt_budget_corrected_$(date +%Y%m%d_%H%M%S)}"
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

STAGES=(promptbudget512 promptbudget768 promptbudget1024)
CONFIGS=(
  configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_promptbudget512.yaml
  configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_promptbudget768.yaml
  configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_promptbudget1024.yaml
)
VARIANTS=(promptbudget512 promptbudget768 promptbudget1024)
BUDGETS=(512 768 1024)

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
    printf 'Missing prompt-budget config: %s\n' "$config" >&2
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
  printf 'Another LIAR-RAW prompt-evidence queue is running. Lock: %s\n' "$LOCK_FILE" >&2
  exit 2
fi

exec > >(tee -a "$QUEUE_LOG") 2>&1

timestamp() {
  date '+%F %T %Z %z'
}

log() {
  printf '[prompt-budget-corrected %s] %s\n' "$(timestamp)" "$*"
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

validate_prompt_budget_build() {
  local case_root="$1"
  local budget="$2"
  "$PYTHON_BIN" - "$case_root" "$budget" <<'PY'
import json
import sys
from pathlib import Path

case_root = Path(sys.argv[1])
budget = int(sys.argv[2])
build_dir = case_root / "build"
paths = sorted(build_dir.glob("build_*.jsonl"))
if not paths:
    raise SystemExit(f"missing build JSONL files: {build_dir}")

row_count = 0
partial_count = 0
max_prompt_tokens = 0
for path in paths:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_count += 1
            if row.get("prompt_evidence_policy") != "prompt_budget":
                raise SystemExit(f"{path}:{line_number}: unexpected policy={row.get('prompt_evidence_policy')!r}")
            if int(row.get("prompt_evidence_prompt_token_budget") or 0) != budget:
                raise SystemExit(f"{path}:{line_number}: wrong prompt budget metadata")
            prompt_ids = row.get("prompt_input_ids")
            if not isinstance(prompt_ids, list) or not prompt_ids:
                raise SystemExit(f"{path}:{line_number}: missing prompt_input_ids")
            prompt_count = int(row.get("prompt_token_count") or 0)
            if len(prompt_ids) != prompt_count:
                raise SystemExit(
                    f"{path}:{line_number}: len(prompt_input_ids)={len(prompt_ids)} != prompt_token_count={prompt_count}"
                )
            if prompt_count > budget:
                raise SystemExit(f"{path}:{line_number}: prompt_token_count={prompt_count} > budget={budget}")
            effective_budget = int(row.get("prompt_evidence_effective_prompt_token_budget") or 0)
            if effective_budget <= 0 or prompt_count > effective_budget:
                raise SystemExit(
                    f"{path}:{line_number}: prompt_token_count={prompt_count} > effective_budget={effective_budget}"
                )
            selected = ((row.get("selector_trace") or {}).get("selected_indices") or [])
            evidence_count = int(row.get("evidence_count") or 0)
            candidates = row.get("candidates") or []
            if not (len(selected) == evidence_count == len(candidates)):
                raise SystemExit(
                    f"{path}:{line_number}: prefix alignment failed: selected={len(selected)}, "
                    f"evidence_count={evidence_count}, candidates={len(candidates)}"
                )
            mrec_steps = row.get("mrec_prompt_steps")
            if isinstance(mrec_steps, list) and len(mrec_steps) != evidence_count:
                raise SystemExit(f"{path}:{line_number}: MREC step/evidence alignment failed")
            partial = bool(row.get("prompt_evidence_partial_evidence"))
            if partial:
                partial_count += 1
                if not bool(row.get("was_truncated")) or not bool(row.get("evidence_text_truncated")):
                    raise SystemExit(f"{path}:{line_number}: partial evidence is not explicitly marked")
            max_prompt_tokens = max(max_prompt_tokens, prompt_count)

print(
    f"prompt-budget audit ok: case_root={case_root} budget={budget} "
    f"rows={row_count} max_prompt_tokens={max_prompt_tokens} partial_evidence_rows={partial_count}"
)
PY
}

run_stage() {
  local stage="$1"
  local config="$2"
  local variant="$3"
  local budget="$4"
  local base="liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_${variant}"
  local case_root="outputs/sentence_trace_method/${base}"
  local run_root="${case_root}_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
  local stage_log="${QUEUE_DIR}/${stage}.log"

  if [[ "$DRY_RUN" != "true" && "$MODE" == "full" ]] \
    && ! force_requested \
    && experiment_complete "$run_root"; then
    validate_prompt_budget_build "$case_root" "$budget"
    log "${stage} already complete and build audit passed; skipping: ${run_root}"
    return 0
  fi

  log "starting ${stage}: config=${config} budget=${budget} case_root=${case_root}"
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
  validate_prompt_budget_build "$case_root" "$budget"
  if [[ "$MODE" == "full" ]] && ! experiment_complete "$run_root"; then
    log "${stage} exited successfully but completion artifacts are incomplete: ${run_root}"
    return 1
  fi
  log "${stage} completed and prompt-budget audit passed"
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
  run_stage \
    "$stage" \
    "${CONFIGS[$index]}" \
    "${VARIANTS[$index]}" \
    "${BUDGETS[$index]}"
done

log "queue completed successfully"
