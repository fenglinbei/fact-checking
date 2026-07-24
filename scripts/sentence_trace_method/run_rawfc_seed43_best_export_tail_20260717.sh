#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
INFER_BIN="${INFER_BIN:-$PYTHON_BIN}"
CONTRACT_HELPER="${CONTRACT_HELPER:-scripts/sentence_trace_method/rawfc_seed43_tail_contract.py}"

OLD_QUEUE_BASE="${OLD_QUEUE_BASE:-outputs/sentence_trace_method/queues/mrec_vo_crossover_20260717_0311_ctxfix}"
OLD_DECISION_LOCK="${OLD_DECISION_LOCK:-${OLD_QUEUE_BASE}/decision_queue.lock}"
TAIL_RUN_DIR="${TAIL_RUN_DIR:-outputs/sentence_trace_method/queues/rawfc_seed43_best_export_20260717}"
TAIL_LOCK="${TAIL_LOCK:-${TAIL_RUN_DIR}/rawfc_seed43_best_export.lock}"
EVENTS_FILE="${EVENTS_FILE:-${TAIL_RUN_DIR}/events.tsv}"
FINAL_MANIFEST="${FINAL_MANIFEST:-${TAIL_RUN_DIR}/manifest.json}"
VAL_LOG="${VAL_LOG:-${TAIL_RUN_DIR}/val.log}"
TEST_LOG="${TEST_LOG:-${TAIL_RUN_DIR}/test.log}"

SEED43_RUN_ROOT="${SEED43_RUN_ROOT:-outputs/sentence_trace_method/rawfc__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_fullpool_minmax5_10_baseline20_seed43_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc}"
EXPECTED_ADAPTER_SHA="${EXPECTED_ADAPTER_SHA:-beb673526d7ddf339ffba18b132fa7222ed64e3c0fec3edfaa51de298e80fe49}"
EXPECTED_NUM_SAMPLES="${EXPECTED_NUM_SAMPLES:-200}"

HARD_STOP_ISO="${HARD_STOP_ISO:-2026-07-17T11:40:00+08:00}"
EXPORT_FINISH_MARGIN_SECONDS="${EXPORT_FINISH_MARGIN_SECONDS:-60}"
MIN_EXPORT_BUDGET_SECONDS="${MIN_EXPORT_BUDGET_SECONDS:-600}"
EXPORT_CUDA_VISIBLE_DEVICES="${EXPORT_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
RAWFC_SEED43_PROC_ROOT="${RAWFC_SEED43_PROC_ROOT:-/proc}"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "$TAIL_RUN_DIR" "$(dirname "$OLD_DECISION_LOCK")" "$(dirname "$TAIL_LOCK")"

timestamp() {
  date --iso-8601=seconds
}

now_epoch() {
  if [[ -n "${TAIL_NOW_EPOCH:-}" ]]; then
    printf '%s\n' "$TAIL_NOW_EPOCH"
  else
    date +%s
  fi
}

event() {
  local stage="$1" status="$2" detail="$3"
  detail="${detail//$'\t'/ }"
  detail="${detail//$'\n'/ }"
  if [[ ! -s "$EVENTS_FILE" ]]; then
    printf 'timestamp\tstage\tstatus\tdetail\n' > "$EVENTS_FILE"
  fi
  printf '%s\t%s\t%s\t%s\n' "$(timestamp)" "$stage" "$status" "$detail" >> "$EVENTS_FILE"
  printf '[rawfc-seed43-best-export] stage=%s status=%s %s\n' "$stage" "$status" "$detail"
}

PREFLIGHT_JSON='null'
EXPORTS_JSON='{}'
ACCELERATE_AUDITS_JSON='[]'
INFER_EXIT_CODES_JSON='{}'
FINALIZED=false
EXPORT_DEADLINE_ISO='unresolved'
LATEST_START_ISO='unresolved'

write_manifest() {
  local status="$1" exit_code="$2" detail="$3" tmp
  tmp="${FINAL_MANIFEST}.tmp.$$"
  jq -n \
    --arg status "$status" \
    --arg detail "$detail" \
    --arg completed_at "$(timestamp)" \
    --arg hard_stop "$HARD_STOP_ISO" \
    --arg export_deadline "$EXPORT_DEADLINE_ISO" \
    --arg latest_start "$LATEST_START_ISO" \
    --arg old_decision_lock "$OLD_DECISION_LOCK" \
    --arg events "$EVENTS_FILE" \
    --arg root "$SEED43_RUN_ROOT" \
    --arg expected_adapter_sha "$EXPECTED_ADAPTER_SHA" \
    --arg val_log "$VAL_LOG" \
    --arg test_log "$TEST_LOG" \
    --arg cuda_visible_devices "$EXPORT_CUDA_VISIBLE_DEVICES" \
    --argjson exit_code "$exit_code" \
    --arg expected_num_samples "$EXPECTED_NUM_SAMPLES" \
    --arg finish_margin_seconds "$EXPORT_FINISH_MARGIN_SECONDS" \
    --arg min_export_budget_seconds "$MIN_EXPORT_BUDGET_SECONDS" \
    --argjson preflight "$PREFLIGHT_JSON" \
    --argjson exports "$EXPORTS_JSON" \
    --argjson accelerate_audits "$ACCELERATE_AUDITS_JSON" \
    --argjson infer_exit_codes "$INFER_EXIT_CODES_JSON" '
      {
        schema_version: "rawfc-seed43-best-export-tail-v0.1",
        status: $status,
        exit_code: $exit_code,
        detail: $detail,
        completed_at: $completed_at,
        seed: 43,
        hard_stop: $hard_stop,
        export_deadline: $export_deadline,
        latest_start: $latest_start,
        finish_margin_seconds: ($finish_margin_seconds | tonumber? // null),
        min_export_budget_seconds: ($min_export_budget_seconds | tonumber? // null),
        old_decision_lock: $old_decision_lock,
        events: $events,
        root: $root,
        expected_adapter_sha256: $expected_adapter_sha,
        expected_num_samples: ($expected_num_samples | tonumber? // null),
        logs: {val: $val_log, test: $test_log},
        cuda_visible_devices: $cuda_visible_devices,
        preflight: $preflight,
        exports: $exports,
        infer_exit_codes: $infer_exit_codes,
        accelerate_audits: $accelerate_audits
      }
    ' > "$tmp"
  mv "$tmp" "$FINAL_MANIFEST"
}

finish() {
  local status="$1" exit_code="$2" detail="$3"
  event orchestrator "$status" "$detail"
  write_manifest "$status" "$exit_code" "$detail"
  FINALIZED=true
  exit "$exit_code"
}

on_exit() {
  local exit_code=$?
  if [[ "$FINALIZED" != "true" ]]; then
    set +e
    event orchestrator unexpected_failure "exit_code=${exit_code}"
    write_manifest unexpected_failure "$exit_code" "unhandled shell failure"
  fi
}
trap on_exit EXIT

epoch_from_iso() {
  local value="$1" label="$2" epoch
  epoch="$(date -d "$value" +%s 2>/dev/null || true)"
  if [[ ! "$epoch" =~ ^[0-9]+$ ]]; then
    finish invalid_config 2 "invalid ${label}=${value}"
  fi
  printf '%s\n' "$epoch"
}

[[ "$EXPECTED_ADAPTER_SHA" =~ ^[0-9a-f]{64}$ ]] || \
  finish invalid_config 2 "EXPECTED_ADAPTER_SHA must be a lowercase SHA256"
[[ "$EXPECTED_NUM_SAMPLES" =~ ^[1-9][0-9]*$ ]] || \
  finish invalid_config 2 "EXPECTED_NUM_SAMPLES must be positive"
[[ "$EXPORT_FINISH_MARGIN_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
  finish invalid_config 2 "EXPORT_FINISH_MARGIN_SECONDS must be positive"
[[ "$MIN_EXPORT_BUDGET_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
  finish invalid_config 2 "MIN_EXPORT_BUDGET_SECONDS must be positive"
[[ "$DRY_RUN" == "true" || "$DRY_RUN" == "false" ]] || \
  finish invalid_config 2 "DRY_RUN must be true or false"
[[ -f "$CONTRACT_HELPER" ]] || finish invalid_config 2 "missing contract helper=${CONTRACT_HELPER}"
[[ -x "$INFER_BIN" ]] || finish invalid_config 2 "inference binary is not executable=${INFER_BIN}"

HARD_STOP_EPOCH="$(epoch_from_iso "$HARD_STOP_ISO" HARD_STOP_ISO)"
EXPORT_DEADLINE_EPOCH=$((HARD_STOP_EPOCH - EXPORT_FINISH_MARGIN_SECONDS))
LATEST_START_EPOCH=$((EXPORT_DEADLINE_EPOCH - MIN_EXPORT_BUDGET_SECONDS))
EXPORT_DEADLINE_ISO="$(date -d "@${EXPORT_DEADLINE_EPOCH}" --iso-8601=seconds)"
LATEST_START_ISO="$(date -d "@${LATEST_START_EPOCH}" --iso-8601=seconds)"

exec 9>"$TAIL_LOCK"
if ! flock -n 9; then
  finish blocked_tail_lock 73 "another RAWFC seed43 best-export tail owns ${TAIL_LOCK}"
fi

current_epoch="$(now_epoch)"
[[ "$current_epoch" =~ ^[0-9]+$ ]] || finish invalid_config 2 "invalid current epoch=${current_epoch}"
if (( current_epoch > LATEST_START_EPOCH )); then
  finish skipped_insufficient_window 0 \
    "less than ${MIN_EXPORT_BUDGET_SECONDS}s remains before export deadline=${EXPORT_DEADLINE_ISO}; no inference invoked"
fi

exec 8>"$OLD_DECISION_LOCK"
wait_seconds=$((LATEST_START_EPOCH - current_epoch + 1))
event old_decision_lock waiting \
  "lock=${OLD_DECISION_LOCK} wait_budget=${wait_seconds}s latest_start=${LATEST_START_ISO}"
if [[ "$DRY_RUN" == "true" ]]; then
  if ! flock -n 8; then
    finish blocked_live_lock 73 \
      "dry-run will not wait on live OLD_DECISION_LOCK=${OLD_DECISION_LOCK}"
  fi
else
  if ! flock -w "$wait_seconds" 8; then
    finish skipped_insufficient_window 0 \
      "OLD_DECISION_LOCK was not released by latest safe start=${LATEST_START_ISO}"
  fi
fi
event old_decision_lock acquired "exclusive lease acquired and held through terminal export audit"

current_epoch="$(now_epoch)"
if (( current_epoch > LATEST_START_EPOCH )); then
  finish skipped_insufficient_window 0 \
    "lock released after latest safe start=${LATEST_START_ISO}; no inference invoked"
fi

audit_accelerate() {
  local stage="$1" payload rc count
  set +e
  payload="$("$PYTHON_BIN" "$CONTRACT_HELPER" accelerate --proc-root "$RAWFC_SEED43_PROC_ROOT")"
  rc=$?
  set -e
  if [[ -z "$payload" ]]; then
    payload='{"status":"invalid","error":"empty accelerate audit"}'
  fi
  ACCELERATE_AUDITS_JSON="$(jq -n \
    --argjson audits "$ACCELERATE_AUDITS_JSON" \
    --arg stage "$stage" \
    --argjson audit "$payload" '$audits + [{stage:$stage,audit:$audit}]')"
  (( rc == 0 )) || return 2
  count="$(jq -er '.count | numbers' <<< "$payload" 2>/dev/null || true)"
  [[ "$count" =~ ^[0-9]+$ ]] || return 2
  (( count == 0 ))
}

audit_preflight() {
  local payload rc
  set +e
  payload="$("$PYTHON_BIN" "$CONTRACT_HELPER" best-adapter \
    --root "$SEED43_RUN_ROOT" \
    --expected-seed 43 \
    --expected-adapter-sha "$EXPECTED_ADAPTER_SHA")"
  rc=$?
  set -e
  [[ -n "$payload" ]] || payload='{"status":"invalid","error":"empty best-adapter audit"}'
  PREFLIGHT_JSON="$payload"
  (( rc == 0 ))
}

if ! audit_preflight; then
  finish blocked_preflight 6 \
    "best SHA/seed43/training_complete-missing contract failed; no inference invoked"
fi
event preflight ready \
  "best SHA=${EXPECTED_ADAPTER_SHA}; resolved seed=43; training_complete missing"

if ! audit_accelerate before_export; then
  finish blocked_accelerate 5 "one or more accelerate launchers exist; no inference invoked"
fi
event accelerate ready "global accelerate launcher count=0 before export"

audit_export() {
  local split="$1" payload rc
  set +e
  payload="$("$PYTHON_BIN" "$CONTRACT_HELPER" export \
    --root "$SEED43_RUN_ROOT" \
    --split "$split" \
    --expected-num-samples "$EXPECTED_NUM_SAMPLES")"
  rc=$?
  set -e
  [[ -n "$payload" ]] || payload="$(jq -n --arg split "$split" \
    '{status:"incomplete",split:$split,error:"empty export audit"}')"
  EXPORTS_JSON="$(jq -n \
    --argjson exports "$EXPORTS_JSON" \
    --arg split "$split" \
    --argjson audit "$payload" '$exports + {($split):$audit}')"
  (( rc == 0 ))
}

val_ready=false
test_ready=false
if audit_export val; then val_ready=true; fi
if audit_export test; then test_ready=true; fi
if [[ "$val_ready" == "true" && "$test_ready" == "true" ]]; then
  finish skipped_complete 0 \
    "val/test best exports already satisfy n=200, parse_failures=0, predictions=200"
fi

if [[ "$DRY_RUN" == "true" ]]; then
  for split in val test; do
    printf '+ env CUDA_VISIBLE_DEVICES=%q PYTHONPATH=src %q -m sft.label_token_infer --run-dir %q --checkpoint best --split %q --config %q\n' \
      "$EXPORT_CUDA_VISIBLE_DEVICES" "$INFER_BIN" "$SEED43_RUN_ROOT/train" "$split" \
      "$SEED43_RUN_ROOT/train.resolved.yaml"
  done
  finish dry_run 0 \
    "all non-mutating gates passed; would export val then test before ${EXPORT_DEADLINE_ISO}"
fi

run_with_export_deadline() {
  local remaining
  remaining=$((EXPORT_DEADLINE_EPOCH - $(now_epoch)))
  (( remaining > 0 )) || return 124
  timeout --signal=INT --kill-after=30 "${remaining}s" "$@"
}

export_pythonpath="src"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export_pythonpath="src:${PYTHONPATH}"
fi

for split in val test; do
  ready=false
  if audit_export "$split"; then ready=true; fi
  if [[ "$ready" == "true" ]]; then
    event "$split" reused \
      "existing best export satisfies n=200, parse_failures=0, predictions=200"
    continue
  fi

  if ! audit_preflight; then
    finish blocked_preflight 6 \
      "best adapter contract changed immediately before ${split} export"
  fi
  if ! audit_accelerate "before_${split}"; then
    finish blocked_accelerate 5 \
      "accelerate launcher appeared immediately before ${split} export"
  fi
  current_epoch="$(now_epoch)"
  if (( current_epoch >= EXPORT_DEADLINE_EPOCH )); then
    finish cutoff 124 "export deadline=${EXPORT_DEADLINE_ISO} reached before ${split}"
  fi

  log_var="${split^^}_LOG"
  log_path="${!log_var}"
  event "$split" starting \
    "direct label_token_infer checkpoint=best deadline=${EXPORT_DEADLINE_ISO} log=${log_path}"
  set +e
  run_with_export_deadline env \
    CUDA_VISIBLE_DEVICES="$EXPORT_CUDA_VISIBLE_DEVICES" \
    PYTHONPATH="$export_pythonpath" \
    "$INFER_BIN" -m sft.label_token_infer \
    --run-dir "$SEED43_RUN_ROOT/train" \
    --checkpoint best \
    --split "$split" \
    --config "$SEED43_RUN_ROOT/train.resolved.yaml" >> "$log_path" 2>&1
  infer_rc=$?
  set -e
  INFER_EXIT_CODES_JSON="$(jq -n \
    --argjson codes "$INFER_EXIT_CODES_JSON" \
    --arg split "$split" \
    --argjson rc "$infer_rc" '$codes + {($split):$rc}')"
  if (( infer_rc != 0 )); then
    if (( infer_rc == 124 || infer_rc == 137 )); then
      finish cutoff "$infer_rc" \
        "${split} inference reached export deadline=${EXPORT_DEADLINE_ISO}; hard stop remains ${HARD_STOP_ISO}"
    fi
    finish export_failed "$infer_rc" "${split} direct inference exited rc=${infer_rc}"
  fi
  if ! audit_export "$split"; then
    finish export_contract_failed 7 \
      "${split} inference exited zero but strict artifact contract failed"
  fi
  event "$split" complete \
    "n=200 parse_failures=0 predictions=200; metrics/prediction SHA recorded"
done

if ! audit_preflight; then
  finish blocked_preflight 6 "best adapter contract changed after exports"
fi
if ! audit_accelerate after_export; then
  finish blocked_accelerate 5 "exports returned but an accelerate launcher remains"
fi
event accelerate ready "global accelerate launcher count=0 after export"

finish complete 0 \
  "best adapter SHA matched; val/test n=200 parse_failures=0 predictions=200; metrics/adapter SHA recorded; completed before 11:40 hard stop"
