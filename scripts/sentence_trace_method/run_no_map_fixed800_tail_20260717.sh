#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
TRAIN_ACCELERATE_BIN="${TRAIN_ACCELERATE_BIN:-$(dirname "$PYTHON_BIN")/accelerate}"
CONTRACT_HELPER="${CONTRACT_HELPER:-scripts/sentence_trace_method/no_map_fixed800_tail_contract.py}"
PROCESS_HELPER="${PROCESS_HELPER:-scripts/sentence_trace_method/fixed_step_salvage_process.py}"
INPUT_CONTRACT="${INPUT_CONTRACT:-configs/validation/no_map_fixed800_tail_inputs_v0_1.json}"
TRAIN_WRAPPER="${TRAIN_WRAPPER:-scripts/sentence_trace_method/run_liar_raw_ministral3_structure_only_no_map.sh}"
CROSSOVER_WRAPPER="${CROSSOVER_WRAPPER:-scripts/phase5_selectors/eval/run_no_map_structure_fixed5_crossover_step800.sh}"

OLD_QUEUE_BASE="${OLD_QUEUE_BASE:-outputs/sentence_trace_method/queues/mrec_vo_crossover_20260717_0311_ctxfix}"
OLD_DECISION_LOCK="${OLD_DECISION_LOCK:-${OLD_QUEUE_BASE}/decision_queue.lock}"
TAIL_DIR="${TAIL_DIR:-outputs/sentence_trace_method/queues/no_map_fixed800_tail_20260717}"
TAIL_LOCK="${TAIL_LOCK:-${TAIL_DIR}/no_map_fixed800_tail.lock}"
EVENTS_FILE="${EVENTS_FILE:-${TAIL_DIR}/events.tsv}"
FINAL_MANIFEST="${FINAL_MANIFEST:-${TAIL_DIR}/manifest.json}"
INPUT_AUDIT="${INPUT_AUDIT:-${TAIL_DIR}/input_audit.json}"
CAPPED_MANIFEST="${CAPPED_MANIFEST:-${TAIL_DIR}/no_map_checkpoint800_capped.json}"
TRAIN_LOG="${TRAIN_LOG:-${TAIL_DIR}/train.log}"
DIAGNOSTIC_LOG="${DIAGNOSTIC_LOG:-${TAIL_DIR}/diagnostic.log}"

NO_MAP_BASE_ROOT="${NO_MAP_BASE_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_no_map_fullpool_minmax5_10}"
NO_MAP_RUN_ROOT="${NO_MAP_RUN_ROOT:-${NO_MAP_BASE_ROOT}_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
NO_MAP_TRACE_ROOT="${NO_MAP_TRACE_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_structure_only_no_map_fullpool}"
NO_MAP_POLICY_CONFIG="${NO_MAP_POLICY_CONFIG:-configs/experiment/mrec_v0.2/learned_marginal_structure_only_no_map_fullpool_minmax5_10.yaml}"
S_RUN_ROOT="${S_RUN_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_fullpool_minmax5_10_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
S_EXPECTED_ADAPTER_SHA256="${S_EXPECTED_ADAPTER_SHA256:-7b7512cd8f5a37d7087be935c3d768db04a29dd3bd479131bd1c5c7681b9374a}"
SOURCE_STRUCTURE_MATRIX="${SOURCE_STRUCTURE_MATRIX:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/frozen_matrix_val}"
DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/no_map_structure_fixed5_matched_verifier_crossover_step800_val}"
MATRIX_ROOT="${MATRIX_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/no_map_structure_fixed5_matrix_val}"
N_FIXED5_SOURCE_ROOT="${N_FIXED5_SOURCE_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/no_map_fixed5_source_val}"

SAFE_START_ISO="${SAFE_START_ISO:-2026-07-17T10:35:00+08:00}"
CHECKPOINT_DEADLINE_ISO="${CHECKPOINT_DEADLINE_ISO:-2026-07-17T11:31:00+08:00}"
CLEANUP_ISO="${CLEANUP_ISO:-2026-07-17T11:39:00+08:00}"
HARD_STOP_ISO="${HARD_STOP_ISO:-2026-07-17T11:40:00+08:00}"
MIN_START_REMAINING_SECONDS="${MIN_START_REMAINING_SECONDS:-3900}"
MIN_DIAGNOSTIC_REMAINING_SECONDS="${MIN_DIAGNOSTIC_REMAINING_SECONDS:-480}"
CHECKPOINT_STABLE_SECONDS="${CHECKPOINT_STABLE_SECONDS:-10}"
POLL_SECONDS="${POLL_SECONDS:-5}"
TAIL_CUDA_VISIBLE_DEVICES="${TAIL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
PROC_ROOT="${NO_MAP_TAIL_PROC_ROOT:-/proc}"
TRAIN_MODULE="${TRAIN_MODULE:-sft.hami_cuda_bootstrap}"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "$TAIL_DIR" "$(dirname "$OLD_DECISION_LOCK")"

timestamp() { date --iso-8601=seconds; }
now_epoch() { [[ -n "${TAIL_NOW_EPOCH:-}" ]] && printf '%s\n' "$TAIL_NOW_EPOCH" || date +%s; }
epoch() { date -d "$1" +%s 2>/dev/null || true; }

event() {
  local stage="$1" status="$2" detail="$3"
  detail="${detail//$'\t'/ }"; detail="${detail//$'\n'/ }"
  [[ -s "$EVENTS_FILE" ]] || printf 'timestamp\tstage\tstatus\tdetail\n' > "$EVENTS_FILE"
  printf '%s\t%s\t%s\t%s\n' "$(timestamp)" "$stage" "$status" "$detail" >> "$EVENTS_FILE"
  printf '[no-map-fixed800-tail] stage=%s status=%s %s\n' "$stage" "$status" "$detail"
}

SAFE_START_EPOCH="$(epoch "$SAFE_START_ISO")"
CHECKPOINT_DEADLINE_EPOCH="$(epoch "$CHECKPOINT_DEADLINE_ISO")"
CLEANUP_EPOCH="$(epoch "$CLEANUP_ISO")"
HARD_STOP_EPOCH="$(epoch "$HARD_STOP_ISO")"
for value in "$SAFE_START_EPOCH" "$CHECKPOINT_DEADLINE_EPOCH" "$CLEANUP_EPOCH" "$HARD_STOP_EPOCH" "$MIN_START_REMAINING_SECONDS" "$MIN_DIAGNOSTIC_REMAINING_SECONDS" "$CHECKPOINT_STABLE_SECONDS" "$POLL_SECONDS"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { printf 'invalid numeric/deadline contract\n' >&2; exit 2; }
done
(( SAFE_START_EPOCH < CHECKPOINT_DEADLINE_EPOCH && CHECKPOINT_DEADLINE_EPOCH < CLEANUP_EPOCH && CLEANUP_EPOCH < HARD_STOP_EPOCH )) || { printf 'deadline order invalid\n' >&2; exit 2; }
[[ -x "$PYTHON_BIN" && -x "$TRAIN_ACCELERATE_BIN" ]] || { printf 'python/accelerate binary missing\n' >&2; exit 2; }
[[ -f "$CONTRACT_HELPER" && -f "$PROCESS_HELPER" && -f "$INPUT_CONTRACT" && -f "$TRAIN_WRAPPER" && -f "$CROSSOVER_WRAPPER" ]] || { printf 'required tail file missing\n' >&2; exit 2; }

STATUS="initializing"
DETAIL=""
EXIT_CODE=1
INPUT_JSON='null'
CAP_JSON='null'
SUMMARY_JSON='null'
ROOT_PID=0
ROOT_STARTTIME=0
ACCEL_PID=0
ACCEL_STARTTIME=0
WRAPPER_RC=-1
FINALIZED=false

write_manifest() {
  local tmp="${FINAL_MANIFEST}.tmp.$$"
  jq -n --arg status "$STATUS" --arg detail "$DETAIL" --arg completed_at "$(timestamp)" \
    --arg safe_start "$SAFE_START_ISO" --arg checkpoint_deadline "$CHECKPOINT_DEADLINE_ISO" \
    --arg cleanup "$CLEANUP_ISO" --arg hard_stop "$HARD_STOP_ISO" --arg old_lock "$OLD_DECISION_LOCK" \
    --arg no_map_root "$NO_MAP_RUN_ROOT" --arg matrix_root "$MATRIX_ROOT" --arg diagnostic_root "$DIAGNOSTIC_ROOT" \
    --arg capped_manifest "$CAPPED_MANIFEST" --argjson exit_code "$EXIT_CODE" --argjson wrapper_rc "$WRAPPER_RC" \
    --argjson input "$INPUT_JSON" --argjson cap "$CAP_JSON" --argjson summary "$SUMMARY_JSON" '
      {schema_version:"no-map-fixed800-tail-v0.1",status:$status,exit_code:$exit_code,detail:$detail,completed_at:$completed_at,
       deadlines:{safe_start:$safe_start,checkpoint:$checkpoint_deadline,cleanup:$cleanup,hard_stop:$hard_stop},old_decision_lock:$old_lock,
       run_root:$no_map_root,matrix_root:$matrix_root,diagnostic_root:$diagnostic_root,capped_manifest:$capped_manifest,
       wrapper_exit_code:$wrapper_rc,input_audit:$input,cap_contract:$cap,diagnostic_summary:$summary,
       standard_clean_results_audit_slot_mutated:false}' > "$tmp"
  mv "$tmp" "$FINAL_MANIFEST"
}

finish() {
  STATUS="$1"; EXIT_CODE="$2"; DETAIL="$3"
  if (( ROOT_PID > 0 )) && kill -0 "$ROOT_PID" 2>/dev/null; then
    cleanup_training
    wait_training_exit
  fi
  event orchestrator "$STATUS" "$DETAIL"
  write_manifest
  FINALIZED=true
  exit "$EXIT_CODE"
}

identify_accelerate() {
  "$PYTHON_BIN" "$PROCESS_HELPER" --proc-root "$PROC_ROOT" identify \
    --root-pid "$ROOT_PID" --root-starttime "$ROOT_STARTTIME" \
    --config "$NO_MAP_RUN_ROOT/train.resolved.yaml" --module "$TRAIN_MODULE"
}

signal_accelerate() {
  "$PYTHON_BIN" "$PROCESS_HELPER" --proc-root "$PROC_ROOT" signal \
    --root-pid "$ROOT_PID" --root-starttime "$ROOT_STARTTIME" \
    --config "$NO_MAP_RUN_ROOT/train.resolved.yaml" --module "$TRAIN_MODULE" \
    --pid "$ACCEL_PID" --starttime "$ACCEL_STARTTIME"
}

cleanup_training() {
  (( ROOT_PID > 0 )) || return 0
  kill -0 "$ROOT_PID" 2>/dev/null || return 0
  local payload rc
  set +e; payload="$(identify_accelerate 2>/dev/null)"; rc=$?; set -e
  if (( rc == 0 )); then
    ACCEL_PID="$(jq -er '.pid' <<< "$payload")"
    ACCEL_STARTTIME="$(jq -er '.starttime' <<< "$payload")"
    set +e; signal_accelerate >/dev/null 2>&1; set -e
  fi
}

wait_training_exit() {
  (( ROOT_PID > 0 )) || return 0
  local current payload rc pid start root_identity root_rc
  while kill -0 "$ROOT_PID" 2>/dev/null; do
    current="$(date +%s)"
    if (( current >= CLEANUP_EPOCH )); then
      set +e; payload="$(identify_accelerate 2>/dev/null)"; rc=$?; set -e
      if (( rc == 0 )); then
        pid="$(jq -er '.pid' <<< "$payload")"; start="$(jq -er '.starttime' <<< "$payload")"
        ACCEL_PID="$pid"; ACCEL_STARTTIME="$start"
        kill -TERM "$pid" 2>/dev/null || true
      fi
    fi
    if (( current >= HARD_STOP_EPOCH )); then
      set +e; payload="$(identify_accelerate 2>/dev/null)"; rc=$?; set -e
      if (( rc == 0 )); then
        pid="$(jq -er '.pid' <<< "$payload")"; start="$(jq -er '.starttime' <<< "$payload")"
        ACCEL_PID="$pid"; ACCEL_STARTTIME="$start"
        kill -KILL "$pid" 2>/dev/null || true
      fi
      set +e; root_identity="$($PYTHON_BIN "$PROCESS_HELPER" --proc-root "$PROC_ROOT" identity --pid "$ROOT_PID" 2>/dev/null)"; root_rc=$?; set -e
      if (( root_rc == 0 )) && [[ "$(jq -er '.starttime' <<< "$root_identity")" == "$ROOT_STARTTIME" ]]; then
        kill -KILL "$ROOT_PID" 2>/dev/null || true
      fi
    fi
    sleep 1
  done
  set +e; wait "$ROOT_PID"; WRAPPER_RC=$?; set -e
}

on_exit() {
  local rc=$?
  if [[ "$FINALIZED" != "true" ]]; then
    set +e; cleanup_training; STATUS=unexpected_failure; EXIT_CODE="$rc"; DETAIL="unhandled shell failure"; write_manifest; set -e
  fi
}
trap on_exit EXIT

polluting=(CASE_NAME CASE_ROOT LORA_ROOT TRAIN_CASE_ROOT RUN_DIR CONFIG CONFIG_PATH BASE_CASE_NAME CASE_SUFFIX LORA_SUFFIX OUTPUT_ROOT MREC_POLICY_CONFIG MAP_ABLATION_MODE)
unset_env=()
for name in "${polluting[@]}"; do [[ -v "$name" ]] && finish blocked_environment 2 "refusing inherited ${name}"; done
for name in "${polluting[@]}"; do unset_env+=(-u "$name"); done

exec 9>"$TAIL_LOCK"
flock -n 9 || finish blocked_tail_lock 73 "another no-map tail owns ${TAIL_LOCK}"
current="$(now_epoch)"
(( current <= SAFE_START_EPOCH )) || finish skipped_deadline 0 "safe-start deadline passed"
(( HARD_STOP_EPOCH - current >= MIN_START_REMAINING_SECONDS )) || finish skipped_insufficient_window 0 "less than ${MIN_START_REMAINING_SECONDS}s remains"

exec 8>"$OLD_DECISION_LOCK"
wait_budget=$((SAFE_START_EPOCH - current + 1))
event old_decision_lock waiting "lock=${OLD_DECISION_LOCK} wait_budget=${wait_budget}s"
if [[ "$DRY_RUN" == "true" ]]; then
  flock -n 8 || finish blocked_live_lock 73 "dry-run will not wait on a live old decision lock"
else
  flock -w "$wait_budget" 8 || finish skipped_deadline 0 "old decision lock not released by safe-start"
fi
event old_decision_lock acquired "exclusive lease held through terminal state"
current="$(now_epoch)"
(( current <= SAFE_START_EPOCH && HARD_STOP_EPOCH - current >= MIN_START_REMAINING_SECONDS )) || finish skipped_insufficient_window 0 "lock released too late for 65-minute contract"

ACCEL_AUDIT="$($PYTHON_BIN "$CONTRACT_HELPER" accelerate --proc-root "$PROC_ROOT")" || finish blocked_accelerate 5 "global accelerate audit failed"
[[ "$(jq -er '.count' <<< "$ACCEL_AUDIT")" == "0" ]] || finish blocked_accelerate 5 "accelerate launcher count is nonzero"

stale=()
for path in "$NO_MAP_BASE_ROOT/train" "$NO_MAP_BASE_ROOT/eval" "$NO_MAP_RUN_ROOT/train" "$NO_MAP_RUN_ROOT/eval"; do [[ -e "$path" ]] && stale+=("$path"); done
(( ${#stale[@]} == 0 )) || finish blocked_stale 4 "no-map root has stale train/eval: ${stale[*]}"

set +e; INPUT_JSON="$($PYTHON_BIN "$CONTRACT_HELPER" inputs --contract "$INPUT_CONTRACT")"; input_rc=$?; set -e
(( input_rc == 0 )) || finish input_contract_failed 6 "weights/traces/build/seed clean contract failed"
printf '%s\n' "$INPUT_JSON" | jq . > "$INPUT_AUDIT"
event inputs ready "structure-only no-map weights/traces and natural minmax(5,10) builds frozen; seed=42"

if [[ "$DRY_RUN" == "true" ]]; then
  event training dry_run "would MODE=train from zero, cap stable checkpoint-800 with exact accelerate SIGINT"
  event diagnostic dry_run "would run only with >=8m: render N_fixed5 then V_N/V_S x N_fixed5/S_fixed5"
  finish dry_run 0 "all non-mutating gates passed; no wrapper invoked"
fi

event training starting "MODE=train from zero; deadline=${CHECKPOINT_DEADLINE_ISO} log=${TRAIN_LOG}"
env "${unset_env[@]}" \
  PYTHON_BIN="$PYTHON_BIN" ACCELERATE_BIN="$TRAIN_ACCELERATE_BIN" \
  CUDA_VISIBLE_DEVICES="$TAIL_CUDA_VISIBLE_DEVICES" NPROC_PER_NODE=4 FINETUNE_MODE=lora \
  SFT_TRAIN_MODULE="$TRAIN_MODULE" MODE=train DRY_RUN=false FORCE_TRAIN=false FORCE_LORA_CONFIG=false RUN_TAU_EVAL=false \
  SAVE_LATEST_TRAIN_STATE=true RESUME_LATEST_TRAIN_STATE=false MAIN_PROCESS_PORT=29689 \
  MREC_POLICY_CONFIG="$NO_MAP_POLICY_CONFIG" MAP_ABLATION_MODE=no_map \
  MREC_RUNTIME_CACHE_ROOT="${ROOT_DIR}/outputs/cache/runtime/no_map_fixed800_tail" \
  CUDA_DEVICE_MEMORY_SHARED_CACHE=/tmp/lzj_no_map_fixed800_tail_hami/cudevshr.cache \
  TRITON_CACHE_DIR=/tmp/lzj_no_map_fixed800_tail_hami/triton \
  TORCHINDUCTOR_CACHE_DIR=/tmp/lzj_no_map_fixed800_tail_hami/torchinductor \
  bash "$TRAIN_WRAPPER" >> "$TRAIN_LOG" 2>&1 &
ROOT_PID=$!
ROOT_ID="$($PYTHON_BIN "$PROCESS_HELPER" --proc-root "$PROC_ROOT" identity --pid "$ROOT_PID")" || { wait "$ROOT_PID" || true; finish process_identity_failed 7 "wrapper root identity missing"; }
ROOT_STARTTIME="$(jq -er '.starttime' <<< "$ROOT_ID")"

candidate=""
while kill -0 "$ROOT_PID" 2>/dev/null; do
  current="$(now_epoch)"
  if (( current >= CHECKPOINT_DEADLINE_EPOCH )); then
    cleanup_training
    wait_training_exit
    finish checkpoint_deadline 0 "checkpoint-800 was not stably ready by ${CHECKPOINT_DEADLINE_ISO}"
  fi
  set +e; candidate="$(identify_accelerate 2>/dev/null)"; identify_rc=$?; set -e
  if (( identify_rc == 0 )); then
    ACCEL_PID="$(jq -er '.pid' <<< "$candidate")"; ACCEL_STARTTIME="$(jq -er '.starttime' <<< "$candidate")"
  elif (( identify_rc == 4 || identify_rc == 5 || identify_rc == 6 )); then
    cleanup_training; wait "$ROOT_PID" || true
    finish process_identity_failed 7 "exact accelerate descendant is ambiguous or root identity changed"
  fi
  set +e; first="$($PYTHON_BIN "$CONTRACT_HELPER" checkpoint --run-root "$NO_MAP_RUN_ROOT" --contract "$INPUT_CONTRACT" --step 800 2>/dev/null)"; checkpoint_rc=$?; set -e
  if (( checkpoint_rc == 0 && ACCEL_PID > 0 )); then
    sleep "$CHECKPOINT_STABLE_SECONDS"
    set +e; second="$($PYTHON_BIN "$CONTRACT_HELPER" checkpoint --run-root "$NO_MAP_RUN_ROOT" --contract "$INPUT_CONTRACT" --step 800 2>/dev/null)"; stable_rc=$?; set -e
    if (( stable_rc == 0 )) && [[ "$(jq -r '.adapter.sha256,.adapter.size,.adapter.mtime_ns,.runtime_config.sha256' <<< "$first")" == "$(jq -r '.adapter.sha256,.adapter.size,.adapter.mtime_ns,.runtime_config.sha256' <<< "$second")" ]]; then
      signal_payload="$(signal_accelerate)" || finish signal_failed 8 "exact checkpoint-800 accelerate SIGINT failed"
      event training sigint "pid=${ACCEL_PID} starttime=${ACCEL_STARTTIME} adapter_sha=$(jq -r '.adapter.sha256' <<< "$second")"
      CAP_JSON="$second"
      break
    fi
  fi
  sleep "$POLL_SECONDS"
done

wait_training_exit
set +e; CAP_JSON="$($PYTHON_BIN "$CONTRACT_HELPER" checkpoint --run-root "$NO_MAP_RUN_ROOT" --contract "$INPUT_CONTRACT" --step 800)"; cap_rc=$?; set -e
(( cap_rc == 0 )) || finish cap_contract_failed 9 "post-SIGINT checkpoint contract failed"
[[ ! -e "$NO_MAP_RUN_ROOT/train/training_complete.json" ]] || finish cap_contract_failed 9 "training_complete appeared after fixed-step cap"
tmp="${CAPPED_MANIFEST}.tmp.$$"
jq -n --arg capped_at "$(timestamp)" --arg run_root "$NO_MAP_RUN_ROOT" --arg signal "SIGINT" \
  --argjson root_pid "$ROOT_PID" --argjson root_starttime "$ROOT_STARTTIME" --argjson accel_pid "$ACCEL_PID" --argjson accel_starttime "$ACCEL_STARTTIME" \
  --argjson wrapper_rc "$WRAPPER_RC" --argjson contract "$CAP_JSON" '
   {schema_version:"no-map-fixed800-cap-v0.1",status:"capped",role:"V_N",seed:42,run_root:$run_root,checkpoint:"checkpoint-800",checkpoint_step:800,
    signal:$signal,training_complete_present:false,capped_at:$capped_at,process_identity:{wrapper_root_pid:$root_pid,wrapper_root_starttime:$root_starttime,accelerate_pid:$accel_pid,accelerate_starttime:$accel_starttime},wrapper_exit_code:$wrapper_rc,contract:$contract,
    note:"Fixed-step diagnostic artifact; not a completed training run."}' > "$tmp"
mv "$tmp" "$CAPPED_MANIFEST"
event training capped "checkpoint-800 stable; no training_complete; manifest=${CAPPED_MANIFEST}"

current="$(now_epoch)"
diagnostic_remaining=$((CLEANUP_EPOCH - current))
if (( diagnostic_remaining < MIN_DIAGNOSTIC_REMAINING_SECONDS )); then
  finish capped_no_diagnostic 0 "checkpoint capped; only ${diagnostic_remaining}s remains before cleanup (<8m)"
fi
if [[ -e "$N_FIXED5_SOURCE_ROOT" || -e "$MATRIX_ROOT" || -e "$DIAGNOSTIC_ROOT" ]]; then
  finish blocked_diagnostic_stale 10 "diagnostic output root exists but no strict summary was accepted"
fi

event diagnostic starting "remaining=${diagnostic_remaining}s; rendering independent N_fixed5 input"
set +e
timeout --signal=INT --kill-after=30 "${diagnostic_remaining}s" env PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON_BIN" scripts/phase5_selectors/build/build_trace_verifier_data.py \
    --config "$NO_MAP_POLICY_CONFIG" --val-trace "$NO_MAP_TRACE_ROOT/selection_trace_val.jsonl" \
    --val-raw data/raw/LIAR-RAW/val.json --dataset liar_raw --label-schema liar6 --output-dir "$N_FIXED5_SOURCE_ROOT" \
    --selection-mode trace --trace-order-field display_ordered_indices --trace-prompt-style mrec_min --evidence-text-mode full \
    --expected-selector-name mrec_greedy_transition_v0_2_learned_marginal_structure_only_no_map_fullpool \
    --expected-chunk-mmr-fingerprint "" \
    --top-k 5 --prompt-evidence-policy fixed_topk --prompt-evidence-min-count 5 --prompt-evidence-max-count 5 \
    --prompt-model-name-or-path /data/models/Ministral-3-8B-Instruct-2512 --train-model-name-or-path /data/models/Ministral-3-8B-Instruct-2512 \
    --val-only --no-progress >> "$DIAGNOSTIC_LOG" 2>&1
build_diag_rc=$?
set -e
(( build_diag_rc == 0 )) || finish diagnostic_failed "$build_diag_rc" "N_fixed5 rendering failed or hit cleanup"

"$PYTHON_BIN" scripts/phase5_selectors/build/prepare_no_map_structure_fixed5_matrix.py \
  --no-map-build "$N_FIXED5_SOURCE_ROOT/build/build_val.jsonl" --no-map-build-report "$N_FIXED5_SOURCE_ROOT/build/build_report.json" \
  --structure-build "$SOURCE_STRUCTURE_MATRIX/stateful__fixed5/build/build_val.jsonl" \
  --source-matrix-manifest "$SOURCE_STRUCTURE_MATRIX/manifest.json" --output-dir "$MATRIX_ROOT" >> "$DIAGNOSTIC_LOG" 2>&1

current="$(now_epoch)"; diagnostic_remaining=$((CLEANUP_EPOCH - current))
(( diagnostic_remaining > 0 )) || finish diagnostic_failed 124 "no time remains for crossover"
set +e
timeout --signal=INT --kill-after=30 "${diagnostic_remaining}s" env CUDA_VISIBLE_DEVICES="$TAIL_CUDA_VISIBLE_DEVICES" \
  MATRIX_ROOT="$MATRIX_ROOT" MATRIX_MANIFEST="$MATRIX_ROOT/manifest.json" OUTPUT_ROOT="$DIAGNOSTIC_ROOT" \
  N_RUN_DIR="$NO_MAP_RUN_ROOT/train" S_RUN_DIR="$S_RUN_ROOT/train" \
  N_EXPECTED_ADAPTER_SHA256="$(jq -r '.adapter.sha256' <<< "$CAP_JSON")" S_EXPECTED_ADAPTER_SHA256="$S_EXPECTED_ADAPTER_SHA256" \
  PHASES=prepare,infer,fanout,summarize bash "$CROSSOVER_WRAPPER" >> "$DIAGNOSTIC_LOG" 2>&1
diag_rc=$?
set -e
(( diag_rc == 0 )) || finish diagnostic_failed "$diag_rc" "2x2 diagnostic failed or hit 11:39 cleanup"
SUMMARY_JSON="$(jq -c . "$DIAGNOSTIC_ROOT/summary.json")"
jq -e --arg n_sha "$(jq -r '.adapter.sha256' <<< "$CAP_JSON")" --arg s_sha "$S_EXPECTED_ADAPTER_SHA256" '
 .schema_version == "no-map-structure-fixed5-crossover-summary-v0.1" and .status == "complete" and .checkpoint == "checkpoint-800"
 and .event_count == 1234 and .prompt_cells == {N:"N_fixed5",S:"S_fixed5"}
 and .verifiers.V_N.adapter_sha256 == $n_sha and .verifiers.V_S.adapter_sha256 == $s_sha
 and ([.verifiers[].metrics[].num_samples] | all(. == 1234))
 and .interpretation_contract.standard_clean_results_audit_slot_mutated == false
' "$DIAGNOSTIC_ROOT/summary.json" >/dev/null || finish diagnostic_contract_failed 11 "summary contract failed"
finish complete 0 "checkpoint-800 capped and N/S fixed-K 2x2 diagnostic complete"
