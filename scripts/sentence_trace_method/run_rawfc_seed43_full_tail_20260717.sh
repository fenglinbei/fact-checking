#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
CONTRACT_HELPER="${CONTRACT_HELPER:-scripts/sentence_trace_method/rawfc_seed43_tail_contract.py}"
SEED43_WRAPPER="${SEED43_WRAPPER:-scripts/sentence_trace_method/run_rawfc_ministral3_atom_anchor_v0_2_structure_only_fullpool_minmax5_10_baseline20_seed43_lora_r16a32_d010_lr1e5_ep12_eval50.sh}"

OLD_QUEUE_BASE="${OLD_QUEUE_BASE:-outputs/sentence_trace_method/queues/mrec_vo_crossover_20260717_0311_ctxfix}"
OLD_DECISION_LOCK="${OLD_DECISION_LOCK:-${OLD_QUEUE_BASE}/decision_queue.lock}"
TAIL_RUN_DIR="${TAIL_RUN_DIR:-outputs/sentence_trace_method/queues/rawfc_seed43_full_tail_20260717}"
TAIL_LOCK="${TAIL_LOCK:-${TAIL_RUN_DIR}/rawfc_seed43_tail.lock}"
EVENTS_FILE="${EVENTS_FILE:-${TAIL_RUN_DIR}/events.tsv}"
FINAL_MANIFEST="${FINAL_MANIFEST:-${TAIL_RUN_DIR}/manifest.json}"
BUILD_LOG="${BUILD_LOG:-${TAIL_RUN_DIR}/build.log}"
FULL_LOG="${FULL_LOG:-${TAIL_RUN_DIR}/full.log}"

CANONICAL_BASE_ROOT="${CANONICAL_BASE_ROOT:-outputs/sentence_trace_method/rawfc__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_fullpool_minmax5_10_baseline20}"
CANONICAL_RUN_ROOT="${CANONICAL_RUN_ROOT:-${CANONICAL_BASE_ROOT}_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc}"
SEED43_BASE_ROOT="${SEED43_BASE_ROOT:-outputs/sentence_trace_method/rawfc__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_fullpool_minmax5_10_baseline20_seed43}"
SEED43_RUN_ROOT="${SEED43_RUN_ROOT:-${SEED43_BASE_ROOT}_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc}"

SAFE_START_ISO="${SAFE_START_ISO:-2026-07-17T10:20:00+08:00}"
HARD_STOP_ISO="${HARD_STOP_ISO:-2026-07-17T11:40:00+08:00}"
TAIL_CUDA_VISIBLE_DEVICES="${TAIL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
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
  printf '[rawfc-seed43-tail] stage=%s status=%s %s\n' "$stage" "$status" "$detail"
}

CANONICAL_CONTRACT_JSON='null'
SEED43_CONTRACT_JSON='null'
BUILD_CONTRACT_JSON='null'
ACCELERATE_AUDITS_JSON='[]'
WRAPPER_EXIT_CODE=-1
FINALIZED=false

write_manifest() {
  local status="$1" exit_code="$2" detail="$3" tmp
  tmp="${FINAL_MANIFEST}.tmp.$$"
  jq -n \
    --arg status "$status" \
    --arg detail "$detail" \
    --arg completed_at "$(timestamp)" \
    --arg safe_start "$SAFE_START_ISO" \
    --arg hard_stop "$HARD_STOP_ISO" \
    --arg old_decision_lock "$OLD_DECISION_LOCK" \
    --arg events "$EVENTS_FILE" \
    --arg canonical_base_root "$CANONICAL_BASE_ROOT" \
    --arg canonical_run_root "$CANONICAL_RUN_ROOT" \
    --arg seed43_base_root "$SEED43_BASE_ROOT" \
    --arg seed43_run_root "$SEED43_RUN_ROOT" \
    --arg wrapper "$SEED43_WRAPPER" \
    --arg build_log "$BUILD_LOG" \
    --arg full_log "$FULL_LOG" \
    --arg cuda_visible_devices "$TAIL_CUDA_VISIBLE_DEVICES" \
    --argjson exit_code "$exit_code" \
    --argjson wrapper_exit_code "$WRAPPER_EXIT_CODE" \
    --argjson canonical_contract "$CANONICAL_CONTRACT_JSON" \
    --argjson seed43_contract "$SEED43_CONTRACT_JSON" \
    --argjson build_contract "$BUILD_CONTRACT_JSON" \
    --argjson accelerate_audits "$ACCELERATE_AUDITS_JSON" '
      {
        schema_version: "rawfc-seed43-full-tail-v0.1",
        status: $status,
        exit_code: $exit_code,
        detail: $detail,
        completed_at: $completed_at,
        seed: 43,
        safe_start: $safe_start,
        hard_stop: $hard_stop,
        old_decision_lock: $old_decision_lock,
        events: $events,
        roots: {
          canonical_base: $canonical_base_root,
          canonical_run: $canonical_run_root,
          seed43_base: $seed43_base_root,
          seed43_run: $seed43_run_root
        },
        wrapper: $wrapper,
        logs: {build: $build_log, full: $full_log},
        cuda_visible_devices: $cuda_visible_devices,
        wrapper_exit_code: $wrapper_exit_code,
        canonical_contract: $canonical_contract,
        seed43_contract: $seed43_contract,
        build_contract: $build_contract,
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

SAFE_START_EPOCH="$(epoch_from_iso "$SAFE_START_ISO" SAFE_START_ISO)"
HARD_STOP_EPOCH="$(epoch_from_iso "$HARD_STOP_ISO" HARD_STOP_ISO)"
[[ -f "$CONTRACT_HELPER" ]] || finish invalid_config 2 "missing contract helper=${CONTRACT_HELPER}"
[[ -f "$SEED43_WRAPPER" ]] || finish invalid_config 2 "missing seed43 wrapper=${SEED43_WRAPPER}"
[[ "$DRY_RUN" == "true" || "$DRY_RUN" == "false" ]] || \
  finish invalid_config 2 "DRY_RUN must be true or false"

polluting_vars=(
  CASE_NAME CASE_ROOT LORA_ROOT TRAIN_CASE_ROOT RUN_DIR CONFIG
  CONFIG_PATH BASE_CASE_NAME CASE_SUFFIX LORA_SUFFIX OUTPUT_ROOT MREC_POLICY_CONFIG
)
for var_name in "${polluting_vars[@]}"; do
  if [[ -v "$var_name" ]]; then
    finish blocked_environment 2 "refusing inherited ${var_name}; seed43 roots must remain config-derived"
  fi
done

exec 9>"$TAIL_LOCK"
if ! flock -n 9; then
  finish blocked_tail_lock 73 "another RAWFC seed43 tail owns ${TAIL_LOCK}"
fi

current_epoch="$(now_epoch)"
[[ "$current_epoch" =~ ^[0-9]+$ ]] || finish invalid_config 2 "invalid current epoch=${current_epoch}"
if (( current_epoch > SAFE_START_EPOCH )); then
  finish skipped_deadline 0 "safe-start deadline=${SAFE_START_ISO} already passed; no wrapper invoked"
fi
if (( current_epoch >= HARD_STOP_EPOCH )); then
  finish skipped_hard_stop 0 "hard stop=${HARD_STOP_ISO} already reached; no wrapper invoked"
fi

exec 8>"$OLD_DECISION_LOCK"
wait_seconds=$((SAFE_START_EPOCH - current_epoch + 1))
event old_decision_lock waiting "lock=${OLD_DECISION_LOCK} wait_budget=${wait_seconds}s"
if [[ "$DRY_RUN" == "true" ]]; then
  if ! flock -n 8; then
    finish blocked_live_lock 73 "dry-run will not wait on live OLD_DECISION_LOCK=${OLD_DECISION_LOCK}"
  fi
else
  if ! flock -w "$wait_seconds" 8; then
    finish skipped_deadline 0 "OLD_DECISION_LOCK was not released by safe-start deadline=${SAFE_START_ISO}"
  fi
fi
event old_decision_lock acquired "exclusive lease acquired and held through terminal audit"

current_epoch="$(now_epoch)"
if (( current_epoch > SAFE_START_EPOCH )); then
  finish skipped_deadline 0 "lock released after safe-start deadline=${SAFE_START_ISO}; no wrapper invoked"
fi

set +e
CANONICAL_CONTRACT_JSON="$("$PYTHON_BIN" "$CONTRACT_HELPER" full \
  --root "$CANONICAL_RUN_ROOT" \
  --expected-seed 42 \
  --allow-missing-seed \
  --expected-num-samples 200)"
canonical_rc=$?
set -e
if (( canonical_rc != 0 )); then
  finish skipped_canonical_incomplete 0 \
    "canonical seed42 strict full contract is incomplete; no seed43 wrapper invoked"
fi
event canonical ready "training_complete=true seed42(default-or-explicit) val/test n=200 builds=train,val,test"

audit_accelerate() {
  local stage="$1" payload rc count
  set +e
  payload="$("$PYTHON_BIN" "$CONTRACT_HELPER" accelerate --proc-root "$RAWFC_SEED43_PROC_ROOT")"
  rc=$?
  set -e
  if (( rc != 0 )); then
    ACCELERATE_AUDITS_JSON="$(jq -n \
      --argjson audits "$ACCELERATE_AUDITS_JSON" \
      --arg stage "$stage" \
      --argjson audit "$payload" '$audits + [{stage:$stage,audit:$audit}]')"
    return 2
  fi
  count="$(jq -er '.count | numbers' <<< "$payload" 2>/dev/null || true)"
  ACCELERATE_AUDITS_JSON="$(jq -n \
    --argjson audits "$ACCELERATE_AUDITS_JSON" \
    --arg stage "$stage" \
    --argjson audit "$payload" '$audits + [{stage:$stage,audit:$audit}]')"
  [[ "$count" =~ ^[0-9]+$ ]] || return 2
  (( count == 0 ))
}

if ! audit_accelerate after_lock; then
  finish blocked_accelerate 5 \
    "one or more accelerate launchers remain after OLD_DECISION_LOCK release"
fi
event accelerate ready "global accelerate launcher count=0 after lock acquisition"

set +e
SEED43_CONTRACT_JSON="$("$PYTHON_BIN" "$CONTRACT_HELPER" full \
  --root "$SEED43_RUN_ROOT" \
  --expected-seed 43 \
  --expected-num-samples 200 \
  --canonical-build-root "$CANONICAL_RUN_ROOT")"
seed43_complete_rc=$?
set -e
if (( seed43_complete_rc == 0 )); then
  finish skipped_complete 0 \
    "seed43 strict full contract already passes; training/eval were not repeated"
fi

stale_paths=()
for path in \
  "$SEED43_BASE_ROOT/train" \
  "$SEED43_BASE_ROOT/eval" \
  "$SEED43_RUN_ROOT/train" \
  "$SEED43_RUN_ROOT/eval"; do
  if [[ -e "$path" ]]; then
    stale_paths+=("$path")
  fi
done
if (( ${#stale_paths[@]} > 0 )); then
  finish blocked_stale 4 \
    "incomplete seed43 root contains stale train/eval paths: ${stale_paths[*]}"
fi
event seed43_root clean "no stale train/eval path exists; prior build-only artifacts remain auditable"

if [[ "$DRY_RUN" == "true" ]]; then
  event build dry_run \
    "would run MODE=build on CPU, then require base and LoRA train/val/test SHA equality with canonical"
  printf '+ env CUDA_VISIBLE_DEVICES= MODE=build DRY_RUN=false bash %q\n' "$SEED43_WRAPPER"
  event full dry_run \
    "would recheck <=${SAFE_START_ISO}, require zero accelerate launchers, then dynamically timeout at ${HARD_STOP_ISO}"
  printf '+ env CUDA_VISIBLE_DEVICES=%q MODE=full DRY_RUN=false bash %q\n' \
    "$TAIL_CUDA_VISIBLE_DEVICES" "$SEED43_WRAPPER"
  finish dry_run 0 "all non-mutating safety gates passed; no wrapper invoked"
fi

run_with_hard_stop() {
  local remaining
  remaining=$((HARD_STOP_EPOCH - $(now_epoch)))
  (( remaining > 0 )) || return 124
  timeout --signal=INT --kill-after=120 "${remaining}s" "$@"
}

event build starting "CPU build/config preparation; log=${BUILD_LOG}"
set +e
run_with_hard_stop env \
  CUDA_VISIBLE_DEVICES= \
  MODE=build \
  DRY_RUN=false \
  FORCE_BUILD=auto \
  FORCE_LORA_CONFIG=false \
  RAWFC_SEED43_BASE_ROOT="$SEED43_BASE_ROOT" \
  RAWFC_SEED43_RUN_ROOT="$SEED43_RUN_ROOT" \
  RAWFC_SEED43_CANONICAL_BASE_ROOT="$CANONICAL_BASE_ROOT" \
  RAWFC_SEED43_CANONICAL_RUN_ROOT="$CANONICAL_RUN_ROOT" \
  bash "$SEED43_WRAPPER" >> "$BUILD_LOG" 2>&1
build_rc=$?
set -e
if (( build_rc != 0 )); then
  WRAPPER_EXIT_CODE=$build_rc
  finish build_failed "$build_rc" "seed43 build wrapper exited rc=${build_rc}"
fi

set +e
base_build_json="$("$PYTHON_BIN" "$CONTRACT_HELPER" builds \
  --candidate-root "$SEED43_BASE_ROOT" \
  --canonical-root "$CANONICAL_BASE_ROOT")"
base_build_rc=$?
lora_build_json="$("$PYTHON_BIN" "$CONTRACT_HELPER" builds \
  --candidate-root "$SEED43_RUN_ROOT" \
  --canonical-root "$CANONICAL_RUN_ROOT")"
lora_build_rc=$?
base_seed_json="$("$PYTHON_BIN" "$CONTRACT_HELPER" config \
  --config "$SEED43_BASE_ROOT/train.resolved.yaml" --expected-seed 43)"
base_seed_rc=$?
lora_seed_json="$("$PYTHON_BIN" "$CONTRACT_HELPER" config \
  --config "$SEED43_RUN_ROOT/train.resolved.yaml" --expected-seed 43)"
lora_seed_rc=$?
set -e
BUILD_CONTRACT_JSON="$(jq -n \
  --argjson base "$base_build_json" \
  --argjson lora "$lora_build_json" \
  --argjson base_seed "$base_seed_json" \
  --argjson lora_seed "$lora_seed_json" \
  '{base:$base,lora:$lora,resolved_seed:{base:$base_seed,lora:$lora_seed}}')"
if (( base_build_rc != 0 || lora_build_rc != 0 || base_seed_rc != 0 || lora_seed_rc != 0 )); then
  finish build_contract_failed 6 \
    "seed43 build SHA or resolved seed contract failed; GPU training was not launched"
fi
event build verified \
  "base and LoRA train/val/test build SHA match canonical exactly; resolved seed=43"

current_epoch="$(now_epoch)"
if (( current_epoch > SAFE_START_EPOCH )); then
  finish skipped_deadline_after_build 0 \
    "build verified after safe-start deadline=${SAFE_START_ISO}; GPU training was not launched"
fi
if ! audit_accelerate before_full; then
  finish blocked_accelerate 5 \
    "accelerate launcher appeared between build verification and full launch"
fi
event accelerate ready "global accelerate launcher count=0 immediately before full launch"

remaining=$((HARD_STOP_EPOCH - current_epoch))
if (( remaining <= 0 )); then
  finish skipped_hard_stop 0 "no time remains before hard stop=${HARD_STOP_ISO}"
fi
event full starting \
  "seed43 full train+val/test eval; timeout=${remaining}s hard_stop=${HARD_STOP_ISO} log=${FULL_LOG}"
set +e
run_with_hard_stop env \
  CUDA_VISIBLE_DEVICES="$TAIL_CUDA_VISIBLE_DEVICES" \
  MODE=full \
  DRY_RUN=false \
  FORCE_BUILD=auto \
  FORCE_LORA_CONFIG=false \
  FORCE_TRAIN=false \
  FORCE_EVAL=false \
  SAVE_LATEST_TRAIN_STATE=true \
  RESUME_LATEST_TRAIN_STATE=true \
  RAWFC_SEED43_BASE_ROOT="$SEED43_BASE_ROOT" \
  RAWFC_SEED43_RUN_ROOT="$SEED43_RUN_ROOT" \
  RAWFC_SEED43_CANONICAL_BASE_ROOT="$CANONICAL_BASE_ROOT" \
  RAWFC_SEED43_CANONICAL_RUN_ROOT="$CANONICAL_RUN_ROOT" \
  bash "$SEED43_WRAPPER" >> "$FULL_LOG" 2>&1
WRAPPER_EXIT_CODE=$?
set -e

if ! audit_accelerate after_full; then
  finish blocked_accelerate 5 \
    "full wrapper returned but an accelerate launcher still exists"
fi
event accelerate ready "global accelerate launcher count=0 after full wrapper returned"

if (( WRAPPER_EXIT_CODE != 0 )); then
  if (( WRAPPER_EXIT_CODE == 124 || WRAPPER_EXIT_CODE == 137 )); then
    finish cutoff "$WRAPPER_EXIT_CODE" \
      "dynamic timeout reached hard stop=${HARD_STOP_ISO}; no complete claim was written"
  fi
  finish full_failed "$WRAPPER_EXIT_CODE" \
    "seed43 full wrapper exited rc=${WRAPPER_EXIT_CODE}"
fi

set +e
SEED43_CONTRACT_JSON="$("$PYTHON_BIN" "$CONTRACT_HELPER" full \
  --root "$SEED43_RUN_ROOT" \
  --expected-seed 43 \
  --expected-num-samples 200 \
  --canonical-build-root "$CANONICAL_RUN_ROOT")"
seed43_complete_rc=$?
set -e
if (( seed43_complete_rc != 0 )); then
  finish completion_contract_failed 7 \
    "wrapper exited zero but strict training/val/test/seed/build contract is incomplete"
fi

finish complete 0 \
  "training_complete=true; val/test best label_token metrics n=200; seed=43; build SHA matched; accelerate count=0"
