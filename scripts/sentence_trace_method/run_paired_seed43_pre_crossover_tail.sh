#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
DRY_RUN="${DRY_RUN:-false}"
HOOK_PHASE="${HOOK_PHASE:-pre}"
TAIL_BASE_DIR="${TAIL_BASE_DIR:-outputs/sentence_trace_method/queues/mrec_vo_crossover_20260717_0311_ctxfix}"
TAIL_SENTINEL="${TAIL_SENTINEL:-${TAIL_BASE_DIR}/enable_seed43_pre_crossover_tail}"
TAIL_CONSUMED_SENTINEL="${TAIL_CONSUMED_SENTINEL:-${TAIL_SENTINEL}.consumed}"
TAIL_LOCK_FILE="${TAIL_LOCK_FILE:-${TAIL_BASE_DIR}/seed43_pre_crossover_tail.lock}"
TAIL_EVENTS_FILE="${TAIL_EVENTS_FILE:-${TAIL_BASE_DIR}/seed43_pre_crossover_tail_events.tsv}"
TAIL_COMPLETE_FILE="${TAIL_COMPLETE_FILE:-${TAIL_BASE_DIR}/seed43_pre_crossover_tail_complete.json}"
TAIL_FAILURE_MARKER="${TAIL_FAILURE_MARKER:-${TAIL_BASE_DIR}/seed43_pre_crossover_tail_failure.json}"
QUEUE_WRAPPER="${QUEUE_WRAPPER:-scripts/sentence_trace_method/run_structure_only_reservation_queue.sh}"
AUDIT_WRAPPER="${AUDIT_WRAPPER:-scripts/phase5_selectors/analyze/run_structure_only_clean_results_audit.sh}"
CLEAN_AUDIT_OUTPUT_ROOT="${CLEAN_AUDIT_OUTPUT_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/clean_results_audit}"

SEED43_S_TRAIN_DIR="${SEED43_S_TRAIN_DIR:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_fullpool_minmax5_10_seed43_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/train}"
SEED43_O_TRAIN_DIR="${SEED43_O_TRAIN_DIR:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_one_shot_fullpool_minmax5_10_seed43_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/train}"
RAWFC_RUN_ROOT="${RAWFC_RUN_ROOT:-outputs/sentence_trace_method/rawfc__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_fullpool_minmax5_10_baseline20_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc}"
SCIFACT_RUN_ROOT="${SCIFACT_RUN_ROOT:-outputs/sentence_trace_method/scifact__ministral3_8b__atom_union_structure_only_fullpool_minmax9_9_lora_ebs16_lr2em5_ep12_eval100_pat8}"
CROSSOVER_OUTPUT_ROOT="${CROSSOVER_OUTPUT_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/matched_verifier_crossover_seed43_step800_val}"

HARD_STOP_ISO="${HARD_STOP_ISO:-2026-07-17T11:40:00+08:00}"
RAWFC_SAFE_START_ISO="${RAWFC_SAFE_START_ISO:-2026-07-17T10:20:00+08:00}"
SCIFACT_SAFE_START_ISO="${SCIFACT_SAFE_START_ISO:-2026-07-17T10:55:00+08:00}"
VR_BLOCK_UNTIL_ISO="${VR_BLOCK_UNTIL_ISO:-2026-07-17T09:00:00+08:00}"
CROSSOVER_MIN_REMAINING="${CROSSOVER_MIN_REMAINING:-900}"
QUEUE_CUDA_VISIBLE_DEVICES="${QUEUE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
POLL_SECONDS="${POLL_SECONDS:-20}"

mkdir -p "$TAIL_BASE_DIR" "$(dirname "$TAIL_LOCK_FILE")" "$(dirname "$TAIL_EVENTS_FILE")"
if [[ "${TAIL_HOOK_LOCK_HELD:-false}" != "true" ]]; then
  exec 9>"$TAIL_LOCK_FILE"
  if ! flock -n 9; then
    printf '[seed43-tail] blocked: another tail hook holds %s\n' "$TAIL_LOCK_FILE" >&2
    exit 73
  fi
fi

now_epoch() {
  if [[ -n "${TAIL_NOW_EPOCH:-}" ]]; then
    printf '%s\n' "$TAIL_NOW_EPOCH"
  else
    date +%s
  fi
}

timestamp() {
  date --iso-8601=seconds
}

event() {
  local stage="$1" status="$2" detail="$3"
  detail="${detail//$'\t'/ }"
  detail="${detail//$'\n'/ }"
  if [[ ! -s "$TAIL_EVENTS_FILE" ]]; then
    printf 'timestamp\tphase\tstage\tstatus\tdetail\n' > "$TAIL_EVENTS_FILE"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(timestamp)" "$HOOK_PHASE" "$stage" "$status" "$detail" >> "$TAIL_EVENTS_FILE"
  printf '[seed43-tail] phase=%s stage=%s status=%s %s\n' \
    "$HOOK_PHASE" "$stage" "$status" "$detail"
}

epoch_from_iso() {
  local value="$1" label="$2" epoch
  epoch="$(date -d "$value" +%s 2>/dev/null || true)"
  if [[ ! "$epoch" =~ ^[0-9]+$ ]]; then
    printf '[seed43-tail] invalid %s=%s\n' "$label" "$value" >&2
    exit 2
  fi
  printf '%s\n' "$epoch"
}

HARD_STOP_EPOCH="$(epoch_from_iso "$HARD_STOP_ISO" HARD_STOP_ISO)"
RAWFC_SAFE_START_EPOCH="$(epoch_from_iso "$RAWFC_SAFE_START_ISO" RAWFC_SAFE_START_ISO)"
SCIFACT_SAFE_START_EPOCH="$(epoch_from_iso "$SCIFACT_SAFE_START_ISO" SCIFACT_SAFE_START_ISO)"
VR_BLOCK_UNTIL_EPOCH="$(epoch_from_iso "$VR_BLOCK_UNTIL_ISO" VR_BLOCK_UNTIL_ISO)"
[[ "$CROSSOVER_MIN_REMAINING" =~ ^[1-9][0-9]*$ ]] || {
  printf '[seed43-tail] CROSSOVER_MIN_REMAINING must be a positive integer\n' >&2
  exit 2
}
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  printf '[seed43-tail] POLL_SECONDS must be a positive integer\n' >&2
  exit 2
}

stable_sha256() {
  local path="$1" before after actual
  before="$(stat -c '%s:%y' "$path")"
  [[ "${before%%:*}" -gt 0 ]] || return 1
  actual="$(sha256sum "$path" | awk '{print $1}')"
  after="$(stat -c '%s:%y' "$path")"
  [[ "$before" == "$after" ]] || return 1
  printf '%s\n' "$actual"
}

seed43_contract() {
  local train_dir="$1" marker checkpoint adapter_config outer_config
  marker="${train_dir}/training_complete.json"
  checkpoint="${train_dir}/checkpoint-800/adapter_model.safetensors"
  adapter_config="${train_dir}/checkpoint-800/adapter_config.json"
  outer_config="${train_dir%/train}/train.resolved.yaml"
  [[ -s "$marker" ]] &&
    jq -e '.completed == true and (.global_step | numbers) >= 800' "$marker" >/dev/null 2>&1 &&
    [[ -s "$checkpoint" ]] &&
    [[ -s "$adapter_config" ]] &&
    jq -e '
      (.base_model_name_or_path | type == "string" and length > 0)
      and (.peft_type | type == "string" and length > 0)
    ' "$adapter_config" >/dev/null 2>&1 &&
    [[ -s "${train_dir}/config.resolved.yaml" ]] &&
    [[ -s "$outer_config" ]] &&
    "$PYTHON_BIN" -c '
import sys, yaml
payload = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
seed = payload.get("sft_train", {}).get("seed") if isinstance(payload, dict) else None
raise SystemExit(0 if type(seed) is int and seed == 43 else 1)
' "$outer_config" >/dev/null 2>&1
}

wait_for_seed_pair() {
  while ! seed43_contract "$SEED43_S_TRAIN_DIR" || ! seed43_contract "$SEED43_O_TRAIN_DIR"; do
    if [[ "$DRY_RUN" == "true" ]]; then
      event seed43_pair invalid "dry-run requires both strict training/checkpoint-800 contracts"
      return 4
    fi
    if (( $(now_epoch) >= HARD_STOP_EPOCH )); then
      event seed43_pair cutoff "hard stop reached before both contracts became valid"
      return 3
    fi
    event seed43_pair waiting "S/O completion or checkpoint-800 contract is not ready"
    sleep "$POLL_SECONDS"
  done
  S_ADAPTER_SHA256="$(stable_sha256 "${SEED43_S_TRAIN_DIR}/checkpoint-800/adapter_model.safetensors")" || return 4
  O_ADAPTER_SHA256="$(stable_sha256 "${SEED43_O_TRAIN_DIR}/checkpoint-800/adapter_model.safetensors")" || return 4
  export S_ADAPTER_SHA256 O_ADAPTER_SHA256
  event seed43_pair ready "S_SHA=${S_ADAPTER_SHA256} O_SHA=${O_ADAPTER_SHA256}"
}

training_marker_complete() {
  local root="$1"
  [[ -s "$root/train/training_complete.json" ]] &&
    jq -e '.completed == true and (.global_step | numbers) > 0' \
      "$root/train/training_complete.json" >/dev/null 2>&1
}

rawfc_complete() {
  local split metrics
  training_marker_complete "$RAWFC_RUN_ROOT" || return 1
  for split in val test; do
    metrics="$RAWFC_RUN_ROOT/eval/$split/best/label_token/metrics.json"
    [[ -s "$metrics" ]] || return 1
    jq -e '
      .num_samples == 200
      and (.accuracy | numbers)
      and (.macro_precision | numbers)
      and (.macro_recall | numbers)
      and (.macro_f1 | numbers)
    ' "$metrics" >/dev/null || return 1
  done
}

scifact_complete() {
  local val_metrics test_manifest official val_submission test_submission
  training_marker_complete "$SCIFACT_RUN_ROOT" || return 1
  val_metrics="$SCIFACT_RUN_ROOT/eval/val/best/label_token/metrics.json"
  test_manifest="$SCIFACT_RUN_ROOT/eval/test/best/label_token/prediction_manifest.json"
  official="$SCIFACT_RUN_ROOT/submission/scifact_official_style_metrics_val.json"
  val_submission="$SCIFACT_RUN_ROOT/submission/scifact_submission_val.jsonl"
  test_submission="$SCIFACT_RUN_ROOT/submission/scifact_submission_test.jsonl"
  [[ -s "$val_metrics" && -s "$test_manifest" && -s "$official" ]] || return 1
  jq -e '.num_samples == 300 and (.macro_f1 | numbers)' "$val_metrics" >/dev/null || return 1
  jq -e '
    .split == "test" and .num_samples == 300 and .prediction_only == true
  ' "$test_manifest" >/dev/null || return 1
  jq -e '
    (.sentence_selection_only.f1 | numbers)
    and (.sentence.f1 | numbers)
    and (.abstract_label_only.f1 | numbers)
    and (.abstract.f1 | numbers)
  ' "$official" >/dev/null || return 1
  [[ -s "$val_submission" && -s "$test_submission" ]] || return 1
  [[ "$(wc -l < "$val_submission")" -eq 300 ]] || return 1
  [[ "$(wc -l < "$test_submission")" -eq 300 ]] || return 1
}

task_contract_complete() {
  case "$1" in
    rawfc_clean) rawfc_complete ;;
    scifact_clean) scifact_complete ;;
    *) return 2 ;;
  esac
}

record_task_failure() {
  local task="$1" rc="$2" tmp
  if [[ "$DRY_RUN" == "true" ]]; then
    event failure_marker dry_run "would record task=${task} rc=${rc}"
    return 0
  fi
  tmp="${TAIL_FAILURE_MARKER}.tmp.$$"
  if [[ -s "$TAIL_FAILURE_MARKER" ]] && jq -e '
    .schema_version == "paired-seed43-tail-failure-v0.1"
    and (.failed_tasks | type == "object")
  ' "$TAIL_FAILURE_MARKER" >/dev/null 2>&1; then
    jq \
      --arg task "$task" \
      --arg failed_at "$(timestamp)" \
      --argjson rc "$rc" '
        .status = "degraded"
        | .updated_at = $failed_at
        | .failed_tasks[$task] = {
            last_exit_code: $rc,
            last_failed_at: $failed_at
          }
      ' "$TAIL_FAILURE_MARKER" > "$tmp"
  else
    jq -n \
      --arg task "$task" \
      --arg failed_at "$(timestamp)" \
      --argjson rc "$rc" '
        {
          schema_version: "paired-seed43-tail-failure-v0.1",
          status: "degraded",
          updated_at: $failed_at,
          failed_tasks: {
            ($task): {
              last_exit_code: $rc,
              last_failed_at: $failed_at
            }
          }
        }
      ' > "$tmp"
  fi
  mv "$tmp" "$TAIL_FAILURE_MARKER"
  event failure_marker recorded "task=${task} rc=${rc} marker=${TAIL_FAILURE_MARKER}"
}

reconcile_failure_marker() {
  local tmp rawfc_ok=false scifact_ok=false remaining resolved_path
  [[ -s "$TAIL_FAILURE_MARKER" ]] || return 0
  if ! jq -e '
    .schema_version == "paired-seed43-tail-failure-v0.1"
    and (.failed_tasks | type == "object")
  ' "$TAIL_FAILURE_MARKER" >/dev/null 2>&1; then
    event failure_marker invalid "preserving malformed marker=${TAIL_FAILURE_MARKER}"
    return 1
  fi
  rawfc_complete && rawfc_ok=true
  scifact_complete && scifact_ok=true
  tmp="${TAIL_FAILURE_MARKER}.tmp.$$"
  jq \
    --argjson rawfc_ok "$rawfc_ok" \
    --argjson scifact_ok "$scifact_ok" \
    --arg updated_at "$(timestamp)" '
      if $rawfc_ok then del(.failed_tasks.rawfc_clean) else . end
      | if $scifact_ok then del(.failed_tasks.scifact_clean) else . end
      | .updated_at = $updated_at
    ' "$TAIL_FAILURE_MARKER" > "$tmp"
  remaining="$(jq -er '.failed_tasks | length' "$tmp")"
  if (( remaining == 0 )); then
    resolved_path="${TAIL_FAILURE_MARKER}.resolved.$(date +%Y%m%d_%H%M%S)_$$"
    mv "$TAIL_FAILURE_MARKER" "$resolved_path"
    rm -f "$tmp"
    event failure_marker resolved "all failed task contracts are now valid; archived=${resolved_path}"
  else
    mv "$tmp" "$TAIL_FAILURE_MARKER"
    event failure_marker pending "remaining_tasks=$(jq -r '.failed_tasks | keys | join(",")' "$TAIL_FAILURE_MARKER")"
  fi
}

run_with_hard_stop() {
  local remaining
  remaining=$((HARD_STOP_EPOCH - $(now_epoch)))
  if (( remaining <= 0 )); then
    return 3
  fi
  timeout --signal=INT --kill-after=120 "${remaining}s" "$@"
}

run_full_task() {
  local task="$1" queue_name queue_dir task_deadline
  case "$task" in
    rawfc_clean)
      if rawfc_complete; then
        event "$task" skipped_complete "strict full-result contract already valid"
        return 0
      fi
      task_deadline="$RAWFC_SAFE_START_EPOCH"
      ;;
    scifact_clean)
      if scifact_complete; then
        event "$task" skipped_complete "strict full-result contract already valid"
        return 0
      fi
      task_deadline="$SCIFACT_SAFE_START_EPOCH"
      ;;
    *)
      event "$task" invalid "unsupported tail task"
      return 2
      ;;
  esac
  if (( $(now_epoch) >= task_deadline )); then
    event "$task" skipped_deadline "safe-start deadline passed"
    return 10
  fi

  queue_name="seed43_tail_${task}_$(date +%Y%m%d_%H%M%S)_$$"
  queue_dir="${TAIL_BASE_DIR}/pre_crossover_tail_attempts/${queue_name}"
  event "$task" starting "queue=${queue_dir} cutoff=${HARD_STOP_ISO}"
  local -a queue_cmd=(env
    "DRY_RUN=$DRY_RUN"
    "QUEUE_ID=$queue_name"
    "QUEUE_RUN_DIR=$queue_dir"
    "QUEUE_CUTOFF=$HARD_STOP_ISO"
    "QUEUE_CUDA_VISIBLE_DEVICES=$QUEUE_CUDA_VISIBLE_DEVICES"
    "QUEUE_RAWFC_CLEAN_RUN_ROOT=$RAWFC_RUN_ROOT"
    "QUEUE_SCIFACT_CLEAN_RUN_ROOT=$SCIFACT_RUN_ROOT")
  if [[ -n "${TAIL_NOW_EPOCH:-}" ]]; then
    queue_cmd+=("QUEUE_NOW_EPOCH=$TAIL_NOW_EPOCH")
  fi
  queue_cmd+=(bash "$QUEUE_WRAPPER" "${task}:full")

  local rc
  if run_with_hard_stop "${queue_cmd[@]}"; then
    rc=0
  else
    rc=$?
  fi
  if (( rc != 0 )); then
    event "$task" failed "queue rc=${rc}; later stages remain eligible"
    return "$rc"
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    event "$task" dry_run "queue command emitted"
    return 0
  fi
  if [[ "$task" == "rawfc_clean" ]]; then
    rawfc_complete || {
      event "$task" invalid "queue exited zero but strict full-result contract failed"
      return 5
    }
  else
    scifact_complete || {
      event "$task" invalid "queue exited zero but strict full-result contract failed"
      return 5
    }
  fi
  event "$task" complete "strict full-result contract passed"
}

attempt_full_task() {
  local task="$1" rc
  if run_full_task "$task"; then
    rc=0
  else
    rc=$?
  fi
  if (( rc == 0 )); then
    if [[ "$DRY_RUN" != "true" ]]; then
      reconcile_failure_marker || true
    fi
    return 0
  fi
  if (( rc == 10 )); then
    return 0
  fi
  record_task_failure "$task" "$rc"
  return 0
}

wait_until_vr_is_blocked() {
  local now remaining sleep_for
  now="$(now_epoch)"
  if (( now >= VR_BLOCK_UNTIL_EPOCH )); then
    return 0
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    event v_r_guard dry_run "would wait until ${VR_BLOCK_UNTIL_ISO}; V_R is never queued by this hook"
    return 0
  fi
  while (( now < VR_BLOCK_UNTIL_EPOCH )); do
    remaining=$((HARD_STOP_EPOCH - now))
    (( remaining > 0 )) || return 3
    sleep_for=$((VR_BLOCK_UNTIL_EPOCH - now))
    (( sleep_for > POLL_SECONDS )) && sleep_for="$POLL_SECONDS"
    sleep "$sleep_for"
    now="$(now_epoch)"
  done
  event v_r_guard complete "parent safe-start deadline is now closed; V_R was not queued"
}

refresh_clean_audit() {
  if [[ "$DRY_RUN" == "true" ]]; then
    event clean_results_audit dry_run "would run ${AUDIT_WRAPPER}"
    return 0
  fi
  PYTHON_BIN="$PYTHON_BIN" OUTPUT_ROOT="$CLEAN_AUDIT_OUTPUT_ROOT" bash "$AUDIT_WRAPPER"
  jq -e '
    .schema_version == "structure-only-clean-results-audit-summary-v0.1"
    and .coverage.total == 6
    and .coverage.invalid == 0
    and .provenance_policy.explicit_roots_only == true
    and .provenance_policy.fallback_used == false
  ' "$CLEAN_AUDIT_OUTPUT_ROOT/summary.json" >/dev/null
  event clean_results_audit complete "summary=${CLEAN_AUDIT_OUTPUT_ROOT}/summary.json"
}

crossover_complete() {
  local summary="${CROSSOVER_OUTPUT_ROOT}/summary.json"
  [[ -s "$summary" ]] || return 1
  jq -e \
    --arg s_sha "$S_ADAPTER_SHA256" \
    --arg o_sha "$O_ADAPTER_SHA256" '
      .schema_version == "structure-only-matched-verifier-crossover-summary-v0.1"
      and .status == "complete"
      and .scope == "frozen_val_only_fixed_k5_common_support"
      and .split == "val"
      and .checkpoint == "checkpoint-800"
      and .event_count == 1234
      and .event_id_sequence_sha256 == "65038f1f222b7d990642970ebf7281434abdb17fe61ec1e14ed0c937e8ee6549"
      and .verifiers.V_S.adapter_sha256 == $s_sha
      and .verifiers.V_O.adapter_sha256 == $o_sha
      and .verifiers.V_S.metrics.one_shot__fixed5.num_samples == 1234
      and .verifiers.V_S.metrics.stateful__fixed5.num_samples == 1234
      and .verifiers.V_O.metrics.one_shot__fixed5.num_samples == 1234
      and .verifiers.V_O.metrics.stateful__fixed5.num_samples == 1234
    ' "$summary" >/dev/null
}

consume_sentinel() {
  local status="$1" tmp
  if [[ "$DRY_RUN" == "true" ]]; then
    event sentinel dry_run "would consume ${TAIL_SENTINEL} status=${status}"
    return 0
  fi
  tmp="${TAIL_COMPLETE_FILE}.tmp.$$"
  printf '{\n  "status": "%s",\n  "completed_at": "%s",\n  "sentinel": "%s"\n}\n' \
    "$status" "$(timestamp)" "$TAIL_SENTINEL" > "$tmp"
  mv "$tmp" "$TAIL_COMPLETE_FILE"
  mv -f "$TAIL_SENTINEL" "$TAIL_CONSUMED_SENTINEL"
  event sentinel consumed "status=${status} consumed=${TAIL_CONSUMED_SENTINEL}"
}

record_degraded_completion() {
  local status="$1" tmp
  if [[ "$DRY_RUN" == "true" ]]; then
    event sentinel dry_run "would preserve ${TAIL_SENTINEL} status=${status}"
    return 0
  fi
  tmp="${TAIL_COMPLETE_FILE}.tmp.$$"
  printf '{\n  "status": "%s",\n  "completed_at": "%s",\n  "sentinel": "%s",\n  "failure_marker": "%s"\n}\n' \
    "$status" "$(timestamp)" "$TAIL_SENTINEL" "$TAIL_FAILURE_MARKER" > "$tmp"
  mv "$tmp" "$TAIL_COMPLETE_FILE"
  event sentinel preserved "status=${status} failure_marker=${TAIL_FAILURE_MARKER}"
}

finish_or_preserve_sentinel() {
  local success_status="$1" degraded_status="$2"
  reconcile_failure_marker || true
  if [[ -s "$TAIL_FAILURE_MARKER" ]]; then
    record_degraded_completion "$degraded_status"
  else
    consume_sentinel "$success_status"
  fi
}

if [[ ! -f "$TAIL_SENTINEL" ]]; then
  event sentinel missing "hook is disabled; expected ${TAIL_SENTINEL}"
  exit 2
fi

case "$HOOK_PHASE" in
  pre)
    wait_for_seed_pair
    if [[ "$DRY_RUN" != "true" ]]; then
      reconcile_failure_marker || true
    fi
    now="$(now_epoch)"
    if (( now < RAWFC_SAFE_START_EPOCH )); then
      attempt_full_task rawfc_clean
      if (( $(now_epoch) < SCIFACT_SAFE_START_EPOCH )); then
        attempt_full_task scifact_clean
      fi
    elif (( now < SCIFACT_SAFE_START_EPOCH )); then
      attempt_full_task scifact_clean
    else
      event full_training skipped_deadline "now is at or after ${SCIFACT_SAFE_START_ISO}; no complete training is queued"
    fi
    wait_until_vr_is_blocked
    if crossover_complete; then
      event seed43_crossover skipped_complete "strict fixed-checkpoint-800 summary already matches both seed43 adapters"
      refresh_clean_audit
      finish_or_preserve_sentinel complete degraded_complete
      exit 76
    fi
    remaining=$((HARD_STOP_EPOCH - $(now_epoch)))
    if (( remaining < CROSSOVER_MIN_REMAINING )); then
      event seed43_crossover skipped_budget "remaining=${remaining}s required=${CROSSOVER_MIN_REMAINING}s"
      refresh_clean_audit
      finish_or_preserve_sentinel crossover_skipped_budget degraded_no_crossover_budget
      exit 75
    fi
    event seed43_crossover ready "remaining=${remaining}s required=${CROSSOVER_MIN_REMAINING}s; return to parent wrapper"
    ;;
  finalize)
    wait_for_seed_pair
    if [[ "$DRY_RUN" != "true" ]]; then
      crossover_complete || {
        event seed43_crossover invalid "strict fixed-checkpoint-800 summary contract failed"
        exit 5
      }
    fi
    refresh_clean_audit
    finish_or_preserve_sentinel complete degraded_complete
    ;;
  *)
    event hook invalid "HOOK_PHASE must be pre or finalize; got ${HOOK_PHASE}"
    exit 2
    ;;
esac
