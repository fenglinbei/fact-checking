#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
PROCESS_HELPER="${PROCESS_HELPER:-scripts/sentence_trace_method/fixed_step_salvage_process.py}"
QUEUE_WRAPPER="${QUEUE_WRAPPER:-scripts/sentence_trace_method/run_structure_only_reservation_queue.sh}"
CROSSOVER_WRAPPER="${CROSSOVER_WRAPPER:-scripts/phase5_selectors/eval/run_structure_only_matched_verifier_crossover_step800.sh}"
AUDIT_WRAPPER="${AUDIT_WRAPPER:-scripts/phase5_selectors/analyze/run_structure_only_clean_results_audit.sh}"
O_WRAPPER="${O_WRAPPER:-scripts/sentence_trace_method/run_liar_raw_ministral3_structure_only_one_shot_seed43.sh}"

OLD_QUEUE_BASE="${OLD_QUEUE_BASE:-outputs/sentence_trace_method/queues/mrec_vo_crossover_20260717_0311_ctxfix}"
OLD_DECISION_LOCK="${OLD_DECISION_LOCK:-${OLD_QUEUE_BASE}/decision_queue.lock}"
OLD_TAIL_SENTINEL="${OLD_TAIL_SENTINEL:-${OLD_QUEUE_BASE}/enable_seed43_pre_crossover_tail}"
SUPERSEDED_TAIL_SENTINEL="${SUPERSEDED_TAIL_SENTINEL:-${OLD_TAIL_SENTINEL}.superseded_by_fixed_step_salvage_20260717}"
SALVAGE_RUN_DIR="${SALVAGE_RUN_DIR:-outputs/sentence_trace_method/queues/fixed_step_salvage_20260717}"
SALVAGE_LOCK="${SALVAGE_LOCK:-${SALVAGE_RUN_DIR}/fixed_step_salvage.lock}"
EVENTS_FILE="${EVENTS_FILE:-${SALVAGE_RUN_DIR}/events.tsv}"
FINAL_MANIFEST="${FINAL_MANIFEST:-${SALVAGE_RUN_DIR}/salvage_manifest.json}"
S_CAPPED_MANIFEST="${S_CAPPED_MANIFEST:-${SALVAGE_RUN_DIR}/seed43_s_checkpoint800_capped.json}"
O_CAPPED_MANIFEST="${O_CAPPED_MANIFEST:-${SALVAGE_RUN_DIR}/seed43_o_checkpoint800_capped.json}"
O_LOG="${O_LOG:-${SALVAGE_RUN_DIR}/seed43_o_train.log}"

S_RUN_ROOT="${S_RUN_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_fullpool_minmax5_10_seed43_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
O_RUN_ROOT="${O_RUN_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_one_shot_fullpool_minmax5_10_seed43_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
S_EXPECTED_ADAPTER_SHA256="${S_EXPECTED_ADAPTER_SHA256:-9b81751e7ad4dd59740149a92ee1bc8c979be93ebd1260c936de2059a44fa089}"
S_MIN_PROGRESS_STEP="${S_MIN_PROGRESS_STEP:-1200}"
CAP_CHECKPOINT_STEP="${CAP_CHECKPOINT_STEP:-800}"
TRAIN_MODULE="${TRAIN_MODULE:-sft.hami_cuda_bootstrap}"

CROSSOVER_OUTPUT_ROOT="${CROSSOVER_OUTPUT_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/matched_verifier_crossover_seed43_step800_val}"
CLEAN_AUDIT_OUTPUT_ROOT="${CLEAN_AUDIT_OUTPUT_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/clean_results_audit}"
QUEUE_CUDA_VISIBLE_DEVICES="${QUEUE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
HARD_STOP_ISO="${HARD_STOP_ISO:-2026-07-17T11:40:00+08:00}"
SCIFACT_SAFE_START_ISO="${SCIFACT_SAFE_START_ISO:-2026-07-17T10:55:00+08:00}"
RAWFC_SAFE_START_ISO="${RAWFC_SAFE_START_ISO:-2026-07-17T10:20:00+08:00}"
POLL_SECONDS="${POLL_SECONDS:-5}"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "$SALVAGE_RUN_DIR" "$(dirname "$OLD_DECISION_LOCK")"

timestamp() {
  date --iso-8601=seconds
}

now_epoch() {
  if [[ -n "${SALVAGE_NOW_EPOCH:-}" ]]; then
    printf '%s\n' "$SALVAGE_NOW_EPOCH"
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
  printf '[fixed-step-salvage] stage=%s status=%s %s\n' "$stage" "$status" "$detail"
}

die() {
  event orchestrator failed "$*"
  exit 2
}

HARD_STOP_EPOCH="$(date -d "$HARD_STOP_ISO" +%s 2>/dev/null || true)"
SCIFACT_SAFE_START_EPOCH="$(date -d "$SCIFACT_SAFE_START_ISO" +%s 2>/dev/null || true)"
RAWFC_SAFE_START_EPOCH="$(date -d "$RAWFC_SAFE_START_ISO" +%s 2>/dev/null || true)"
[[ "$HARD_STOP_EPOCH" =~ ^[0-9]+$ ]] || die "invalid HARD_STOP_ISO=${HARD_STOP_ISO}"
[[ "$SCIFACT_SAFE_START_EPOCH" =~ ^[0-9]+$ ]] || die "invalid SCIFACT_SAFE_START_ISO=${SCIFACT_SAFE_START_ISO}"
[[ "$RAWFC_SAFE_START_EPOCH" =~ ^[0-9]+$ ]] || die "invalid RAWFC_SAFE_START_ISO=${RAWFC_SAFE_START_ISO}"
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || die "POLL_SECONDS must be a positive integer"
[[ "$S_MIN_PROGRESS_STEP" =~ ^[1-9][0-9]*$ ]] || die "S_MIN_PROGRESS_STEP must be positive"
[[ "$CAP_CHECKPOINT_STEP" == "800" ]] || die "fixed-step contract is frozen to step 800"
[[ -f "$PROCESS_HELPER" ]] || die "missing process helper=${PROCESS_HELPER}"
[[ -f "$QUEUE_WRAPPER" && -f "$CROSSOVER_WRAPPER" && -f "$AUDIT_WRAPPER" && -f "$O_WRAPPER" ]] || \
  die "one or more required wrappers are missing"

exec 9>"$SALVAGE_LOCK"
if ! flock -n 9; then
  die "another fixed-step salvage orchestrator holds ${SALVAGE_LOCK}"
fi

remaining=$((HARD_STOP_EPOCH - $(now_epoch)))
(( remaining > 0 )) || die "hard stop already reached: ${HARD_STOP_ISO}"
exec 8>"$OLD_DECISION_LOCK"
event old_decision_lock waiting "lock=${OLD_DECISION_LOCK} remaining=${remaining}s"
if [[ "$DRY_RUN" == "true" ]]; then
  flock -n 8 || die "dry-run will not wait on a live old decision lock"
else
  flock -w "$remaining" 8 || die "old decision lock was not released before hard stop"
fi
event old_decision_lock acquired "exclusive GPU lease transferred to salvage orchestrator"

supersede_old_tail_sentinel() {
  if [[ -e "$OLD_TAIL_SENTINEL" ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
      event old_tail_sentinel dry_run "would atomically rename ${OLD_TAIL_SENTINEL} to ${SUPERSEDED_TAIL_SENTINEL}"
    else
      mv -f -- "$OLD_TAIL_SENTINEL" "$SUPERSEDED_TAIL_SENTINEL"
      event old_tail_sentinel superseded "atomic_rename=${OLD_TAIL_SENTINEL}->${SUPERSEDED_TAIL_SENTINEL}"
    fi
  elif [[ -e "$SUPERSEDED_TAIL_SENTINEL" ]]; then
    event old_tail_sentinel skipped_complete "already superseded=${SUPERSEDED_TAIL_SENTINEL}"
  else
    event old_tail_sentinel absent "no enabled sentinel remained after old decision lock release"
  fi
}

supersede_old_tail_sentinel

stable_sha256() {
  local path="$1" before after actual
  before="$(stat -c '%s:%y' "$path")" || return 1
  [[ "${before%%:*}" -gt 0 ]] || return 1
  actual="$(sha256sum "$path" | awk '{print $1}')"
  after="$(stat -c '%s:%y' "$path")" || return 1
  [[ "$before" == "$after" ]] || return 1
  printf '%s\n' "$actual"
}

progress_step() {
  local train_dir="$1" marker value
  for marker in \
    "$train_dir/latest_state/trainer_state.json" \
    "$train_dir/training_complete.json"; do
    if [[ -s "$marker" ]]; then
      value="$(jq -er '.global_step | numbers' "$marker" 2>/dev/null || true)"
      if [[ "$value" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$value"
        return 0
      fi
    fi
  done
  return 1
}

seed43_config() {
  local path="$1"
  [[ -s "$path" ]] && "$PYTHON_BIN" -c '
import sys, yaml
payload = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
seed = payload.get("sft_train", {}).get("seed") if isinstance(payload, dict) else None
raise SystemExit(0 if type(seed) is int and seed == 43 else 1)
' "$path" >/dev/null 2>&1
}

adapter_config_ready() {
  local path="$1"
  [[ -s "$path" ]] && jq -e '
    (.base_model_name_or_path | type == "string" and length > 0)
    and (.peft_type | type == "string" and length > 0)
  ' "$path" >/dev/null 2>&1
}

no_complete_marker() {
  local train_dir="$1"
  [[ ! -e "$train_dir/training_complete.json" ]]
}

validate_fixed_checkpoint() {
  local run_root="$1" min_progress="$2" expected_sha="$3"
  local train_dir="$run_root/train" checkpoint_dir step actual_sha
  checkpoint_dir="$train_dir/checkpoint-${CAP_CHECKPOINT_STEP}"
  no_complete_marker "$train_dir" || return 1
  [[ -s "$checkpoint_dir/adapter_model.safetensors" ]] || return 1
  adapter_config_ready "$checkpoint_dir/adapter_config.json" || return 1
  [[ -s "$train_dir/config.resolved.yaml" ]] || return 1
  seed43_config "$run_root/train.resolved.yaml" || return 1
  step="$(progress_step "$train_dir")" || return 1
  [[ "$step" =~ ^[0-9]+$ && "$step" -ge "$min_progress" ]] || return 1
  actual_sha="$(stable_sha256 "$checkpoint_dir/adapter_model.safetensors")" || return 1
  [[ -z "$expected_sha" || "$actual_sha" == "$expected_sha" ]] || return 1
  printf '%s\t%s\n' "$step" "$actual_sha"
}

S_CONTRACT="$(validate_fixed_checkpoint "$S_RUN_ROOT" "$S_MIN_PROGRESS_STEP" "$S_EXPECTED_ADAPTER_SHA256")" || \
  die "seed43-S fixed-step contract failed (requires no training_complete, progress>=${S_MIN_PROGRESS_STEP}, seed=43, exact checkpoint-800 SHA)"
S_PROGRESS_STEP="${S_CONTRACT%%$'\t'*}"
S_ADAPTER_SHA256="${S_CONTRACT##*$'\t'}"
write_s_capped_manifest() {
  local tmp
  if [[ "$DRY_RUN" == "true" ]]; then
    event seed43_s_manifest dry_run "would write fixed-step-not-complete manifest=${S_CAPPED_MANIFEST}"
    return 0
  fi
  tmp="${S_CAPPED_MANIFEST}.tmp.$$"
  jq -n \
    --arg capped_at "$(timestamp)" \
    --arg run_root "$S_RUN_ROOT" \
    --arg sha "$S_ADAPTER_SHA256" \
    --argjson progress "$S_PROGRESS_STEP" '
      {
        schema_version: "fixed-step-salvage-cap-v0.1",
        status: "capped",
        role: "V_S",
        seed: 43,
        run_root: $run_root,
        checkpoint: "checkpoint-800",
        checkpoint_step: 800,
        progress_step: $progress,
        adapter_sha256: $sha,
        signal: "SIGINT",
        cap_source: "external_before_orchestrator",
        training_complete_present: false,
        capped_at: $capped_at,
        note: "Fixed-step diagnostic artifact; not a completed training run."
      }
    ' > "$tmp"
  mv "$tmp" "$S_CAPPED_MANIFEST"
  jq -e \
    --arg sha "$S_ADAPTER_SHA256" \
    --argjson progress "$S_PROGRESS_STEP" '
      .schema_version == "fixed-step-salvage-cap-v0.1"
      and .status == "capped"
      and .role == "V_S"
      and .progress_step == $progress
      and .adapter_sha256 == $sha
      and .training_complete_present == false
    ' "$S_CAPPED_MANIFEST" >/dev/null
}
write_s_capped_manifest
event seed43_s ready "progress_step=${S_PROGRESS_STEP} checkpoint=checkpoint-800 adapter_sha256=${S_ADAPTER_SHA256} capped_external=true"

process_match_count() {
  local config="$1" output
  output="$($PYTHON_BIN "$PROCESS_HELPER" match \
    --config "$config" \
    --module "$TRAIN_MODULE")" || return 1
  jq -er '.count | numbers' <<< "$output"
}

wait_for_gpu_release() {
  local count_s count_o
  while true; do
    count_s="$(process_match_count "$S_RUN_ROOT/train.resolved.yaml")" || return 1
    count_o="$(process_match_count "$O_RUN_ROOT/train.resolved.yaml")" || return 1
    if (( count_s == 0 && count_o == 0 )); then
      event gpu_lease ready "no seed43 S/O training command remains after old decision lock release"
      return 0
    fi
    if (( $(now_epoch) >= HARD_STOP_EPOCH )); then
      return 3
    fi
    event gpu_lease waiting "S_matching_processes=${count_s} O_matching_processes=${count_o}"
    sleep "$POLL_SECONDS"
  done
}

if [[ "$DRY_RUN" == "true" ]]; then
  event gpu_lease dry_run "would require zero exact S/O training process matches"
else
  wait_for_gpu_release || die "seed43 training processes remained after GPU lease transfer"
fi

o_capped_complete() {
  local contract step actual_sha manifest_sha
  [[ -s "$O_CAPPED_MANIFEST" ]] || return 1
  contract="$(validate_fixed_checkpoint "$O_RUN_ROOT" "$CAP_CHECKPOINT_STEP" "")" || return 1
  step="${contract%%$'\t'*}"
  actual_sha="${contract##*$'\t'}"
  manifest_sha="$(jq -er '.adapter_sha256' "$O_CAPPED_MANIFEST" 2>/dev/null || true)"
  jq -e \
    --arg sha "$actual_sha" \
    --argjson progress "$step" '
      .schema_version == "fixed-step-salvage-cap-v0.1"
      and .status == "capped"
      and .role == "V_O"
      and .seed == 43
      and .checkpoint == "checkpoint-800"
      and .checkpoint_step == 800
      and .progress_step == $progress
      and .adapter_sha256 == $sha
      and .signal == "SIGINT"
      and .training_complete_present == false
    ' "$O_CAPPED_MANIFEST" >/dev/null 2>&1 && [[ "$manifest_sha" == "$actual_sha" ]]
}

write_o_capped_manifest() {
  local progress="$1" sha="$2" root_pid="$3" root_starttime="$4"
  local accelerate_pid="$5" accelerate_starttime="$6" wrapper_rc="$7" tmp
  tmp="${O_CAPPED_MANIFEST}.tmp.$$"
  jq -n \
    --arg capped_at "$(timestamp)" \
    --arg run_root "$O_RUN_ROOT" \
    --arg sha "$sha" \
    --argjson progress "$progress" \
    --argjson root_pid "$root_pid" \
    --argjson root_starttime "$root_starttime" \
    --argjson accelerate_pid "$accelerate_pid" \
    --argjson accelerate_starttime "$accelerate_starttime" \
    --argjson wrapper_exit_code "$wrapper_rc" '
      {
        schema_version: "fixed-step-salvage-cap-v0.1",
        status: "capped",
        role: "V_O",
        seed: 43,
        run_root: $run_root,
        checkpoint: "checkpoint-800",
        checkpoint_step: 800,
        progress_step: $progress,
        adapter_sha256: $sha,
        signal: "SIGINT",
        training_complete_present: false,
        capped_at: $capped_at,
        process_identity: {
          wrapper_root_pid: $root_pid,
          wrapper_root_starttime: $root_starttime,
          accelerate_pid: $accelerate_pid,
          accelerate_starttime: $accelerate_starttime
        },
        wrapper_exit_code: $wrapper_exit_code,
        note: "Fixed-step diagnostic artifact; not a completed training run."
      }
    ' > "$tmp"
  mv "$tmp" "$O_CAPPED_MANIFEST"
}

cap_seed43_o() {
  local remaining root_pid identity root_starttime candidate candidate_pid candidate_starttime
  local signal_result contract progress sha wrapper_rc
  if o_capped_complete; then
    O_ADAPTER_SHA256="$(jq -er '.adapter_sha256' "$O_CAPPED_MANIFEST")"
    O_PROGRESS_STEP="$(jq -er '.progress_step' "$O_CAPPED_MANIFEST")"
    export O_ADAPTER_SHA256 O_PROGRESS_STEP
    event seed43_o skipped_complete "strict capped manifest already matches checkpoint-800 sha=${O_ADAPTER_SHA256}"
    return 0
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    event seed43_o dry_run "would launch ${O_WRAPPER} MODE=train, wait for checkpoint-800 plus latest_state>=800, identify one exact accelerate descendant, send SIGINT, and write ${O_CAPPED_MANIFEST}"
    O_ADAPTER_SHA256="DRY_RUN_O_CHECKPOINT800_SHA256"
    O_PROGRESS_STEP=800
    export O_ADAPTER_SHA256 O_PROGRESS_STEP
    return 0
  fi
  remaining=$((HARD_STOP_EPOCH - $(now_epoch)))
  (( remaining > 0 )) || return 3
  event seed43_o starting "wrapper=${O_WRAPPER} log=${O_LOG} hard_stop=${HARD_STOP_ISO}"
  timeout --signal=INT --kill-after=120 "${remaining}s" \
    env \
      CUDA_VISIBLE_DEVICES="$QUEUE_CUDA_VISIBLE_DEVICES" \
      MODE=train \
      DRY_RUN=false \
      FORCE_TRAIN=false \
      SAVE_LATEST_TRAIN_STATE=true \
      RESUME_LATEST_TRAIN_STATE=true \
      bash "$O_WRAPPER" >> "$O_LOG" 2>&1 &
  root_pid=$!
  identity="$($PYTHON_BIN "$PROCESS_HELPER" identity --pid "$root_pid")" || {
    wait "$root_pid" || true
    return 5
  }
  root_starttime="$(jq -er '.starttime' <<< "$identity")"
  event seed43_o launched "wrapper_root_pid=${root_pid} wrapper_root_starttime=${root_starttime}"

  candidate=""
  while kill -0 "$root_pid" 2>/dev/null; do
    if (( $(now_epoch) >= HARD_STOP_EPOCH )); then
      event seed43_o cutoff "hard stop reached before fixed checkpoint cap completed"
      break
    fi
    if contract="$(validate_fixed_checkpoint "$O_RUN_ROOT" "$CAP_CHECKPOINT_STEP" "" 2>/dev/null)"; then
      set +e
      candidate="$($PYTHON_BIN "$PROCESS_HELPER" identify \
        --root-pid "$root_pid" \
        --root-starttime "$root_starttime" \
        --config "$O_RUN_ROOT/train.resolved.yaml" \
        --module "$TRAIN_MODULE")"
      identify_rc=$?
      set -e
      if (( identify_rc == 4 || identify_rc == 5 )); then
        event seed43_o identity_failed "identify_rc=${identify_rc} payload=${candidate}"
        break
      fi
      if (( identify_rc == 0 )); then
        candidate_pid="$(jq -er '.pid' <<< "$candidate")"
        candidate_starttime="$(jq -er '.starttime' <<< "$candidate")"
        signal_result="$($PYTHON_BIN "$PROCESS_HELPER" signal \
          --root-pid "$root_pid" \
          --root-starttime "$root_starttime" \
          --config "$O_RUN_ROOT/train.resolved.yaml" \
          --module "$TRAIN_MODULE" \
          --pid "$candidate_pid" \
          --starttime "$candidate_starttime")" || {
            event seed43_o signal_failed "payload=${signal_result}"
            break
          }
        event seed43_o sigint "accelerate_pid=${candidate_pid} accelerate_starttime=${candidate_starttime} payload=${signal_result}"
        break
      fi
    fi
    sleep "$POLL_SECONDS"
  done

  set +e
  wait "$root_pid"
  wrapper_rc=$?
  set -e
  [[ -n "$candidate" && "$(jq -r '.status // empty' <<< "$candidate" 2>/dev/null || true)" == "ready" ]] || return 6
  contract="$(validate_fixed_checkpoint "$O_RUN_ROOT" "$CAP_CHECKPOINT_STEP" "")" || return 7
  progress="${contract%%$'\t'*}"
  sha="${contract##*$'\t'}"
  no_complete_marker "$O_RUN_ROOT/train" || return 8
  write_o_capped_manifest \
    "$progress" "$sha" "$root_pid" "$root_starttime" \
    "$candidate_pid" "$candidate_starttime" "$wrapper_rc"
  o_capped_complete || return 9
  O_ADAPTER_SHA256="$sha"
  O_PROGRESS_STEP="$progress"
  export O_ADAPTER_SHA256 O_PROGRESS_STEP
  event seed43_o capped "progress_step=${progress} checkpoint=checkpoint-800 adapter_sha256=${sha} wrapper_rc=${wrapper_rc}; no training_complete written"
}

cap_seed43_o || die "seed43-O fixed-step cap failed rc=$?"

crossover_complete() {
  local summary="$CROSSOVER_OUTPUT_ROOT/summary.json"
  [[ -s "$summary" ]] && jq -e \
    --arg s_sha "$S_ADAPTER_SHA256" \
    --arg o_sha "$O_ADAPTER_SHA256" '
      .schema_version == "structure-only-matched-verifier-crossover-summary-v0.1"
      and .status == "complete"
      and .split == "val"
      and .checkpoint == "checkpoint-800"
      and .event_count == 1234
      and .verifiers.V_S.adapter_sha256 == $s_sha
      and .verifiers.V_O.adapter_sha256 == $o_sha
      and ([.verifiers[].metrics[].num_samples] | all(. == 1234))
    ' "$summary" >/dev/null 2>&1
}

run_with_hard_stop() {
  local remaining
  remaining=$((HARD_STOP_EPOCH - $(now_epoch)))
  (( remaining > 0 )) || return 3
  timeout --signal=INT --kill-after=120 "${remaining}s" "$@"
}

stage_failures=()
run_crossover() {
  if [[ "$DRY_RUN" == "true" ]]; then
    event seed43_crossover dry_run "would run fixed val/K5/checkpoint-800 crossover with SEED43_TAIL_HOOK_ACTIVE=true"
    return 0
  fi
  if crossover_complete; then
    event seed43_crossover skipped_complete "strict summary already matches S/O checkpoint-800 adapters"
    return 0
  fi
  event seed43_crossover starting "output=${CROSSOVER_OUTPUT_ROOT} S_SHA=${S_ADAPTER_SHA256} O_SHA=${O_ADAPTER_SHA256}"
  set +e
  run_with_hard_stop env \
    CUDA_VISIBLE_DEVICES="$QUEUE_CUDA_VISIBLE_DEVICES" \
    SEED43_TAIL_HOOK_ACTIVE=true \
    S_RUN_DIR="$S_RUN_ROOT/train" \
    O_RUN_DIR="$O_RUN_ROOT/train" \
    S_CONFIG="$S_RUN_ROOT/train/config.resolved.yaml" \
    O_CONFIG="$O_RUN_ROOT/train/config.resolved.yaml" \
    S_EXPECTED_ADAPTER_SHA256="$S_ADAPTER_SHA256" \
    O_EXPECTED_ADAPTER_SHA256="$O_ADAPTER_SHA256" \
    OUTPUT_ROOT="$CROSSOVER_OUTPUT_ROOT" \
    PHASES=prepare,infer,fanout,summarize \
    bash "$CROSSOVER_WRAPPER"
  crossover_rc=$?
  set -e
  if (( crossover_rc != 0 )) || ! crossover_complete; then
    event seed43_crossover failed "rc=${crossover_rc}; continuing to cross-dataset tasks"
    stage_failures+=("seed43_crossover:${crossover_rc}")
    return 0
  fi
  event seed43_crossover complete "summary=${CROSSOVER_OUTPUT_ROOT}/summary.json"
}

run_cross_dataset_task() {
  local task="$1" queue_dir queue_id rc safe_start_epoch safe_start_iso
  queue_id="fixed_step_salvage_${task}_$(date +%Y%m%d_%H%M%S)_$$"
  queue_dir="$SALVAGE_RUN_DIR/attempts/$queue_id"
  case "$task" in
    scifact_clean)
      safe_start_epoch="$SCIFACT_SAFE_START_EPOCH"
      safe_start_iso="$SCIFACT_SAFE_START_ISO"
      ;;
    rawfc_clean)
      safe_start_epoch="$RAWFC_SAFE_START_EPOCH"
      safe_start_iso="$RAWFC_SAFE_START_ISO"
      ;;
    *)
      event "$task" failed "missing safe-start policy"
      stage_failures+=("${task}:invalid")
      return 0
      ;;
  esac
  if (( $(now_epoch) >= safe_start_epoch )); then
    event "$task" skipped_deadline "safe-start deadline=${safe_start_iso} passed; task was not launched"
    stage_failures+=("${task}:safe_start_deadline")
    return 0
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    event "$task" dry_run "would run reservation queue ${task}:full queue_dir=${queue_dir}"
    return 0
  fi
  if (( $(now_epoch) >= HARD_STOP_EPOCH )); then
    event "$task" cutoff "not started after hard stop"
    stage_failures+=("${task}:cutoff")
    return 0
  fi
  event "$task" starting "reservation queue full mode; queue_dir=${queue_dir}"
  set +e
  run_with_hard_stop env \
    QUEUE_ID="$queue_id" \
    QUEUE_CUTOFF="$HARD_STOP_ISO" \
    QUEUE_RUN_DIR="$queue_dir" \
    QUEUE_CUDA_VISIBLE_DEVICES="$QUEUE_CUDA_VISIBLE_DEVICES" \
    bash "$QUEUE_WRAPPER" "${task}:full"
  rc=$?
  set -e
  if (( rc != 0 )); then
    event "$task" failed "rc=${rc}; continuing to the other cross-dataset task"
    stage_failures+=("${task}:${rc}")
    return 0
  fi
  event "$task" complete "strict reservation-queue completion contract passed"
}

run_crossover
run_cross_dataset_task scifact_clean
run_cross_dataset_task rawfc_clean

refresh_audit() {
  if [[ "$DRY_RUN" == "true" ]]; then
    event clean_results_audit dry_run "would run ${AUDIT_WRAPPER} after all attempted stages"
    return 0
  fi
  event clean_results_audit starting "output=${CLEAN_AUDIT_OUTPUT_ROOT}"
  set +e
  PYTHON_BIN="$PYTHON_BIN" OUTPUT_ROOT="$CLEAN_AUDIT_OUTPUT_ROOT" bash "$AUDIT_WRAPPER"
  audit_rc=$?
  set -e
  if (( audit_rc != 0 )) || [[ ! -s "$CLEAN_AUDIT_OUTPUT_ROOT/summary.json" ]]; then
    event clean_results_audit failed "rc=${audit_rc}"
    stage_failures+=("clean_results_audit:${audit_rc}")
    return 0
  fi
  event clean_results_audit complete "summary=${CLEAN_AUDIT_OUTPUT_ROOT}/summary.json"
}

refresh_audit

if (( ${#stage_failures[@]} == 0 )); then
  final_status="complete"
  failures_json='[]'
else
  final_status="degraded"
  failures_json="$(printf '%s\n' "${stage_failures[@]}" | jq -R . | jq -s .)"
fi
if [[ "$DRY_RUN" == "true" ]]; then
  event orchestrator dry_run_complete "stage_order=seed43-O,crossover,SciFact,RAWFC,audit"
else
  tmp="${FINAL_MANIFEST}.tmp.$$"
  jq -n \
    --arg status "$final_status" \
    --arg completed_at "$(timestamp)" \
    --arg hard_stop "$HARD_STOP_ISO" \
    --arg s_sha "$S_ADAPTER_SHA256" \
    --arg o_sha "$O_ADAPTER_SHA256" \
    --arg s_cap_manifest "$S_CAPPED_MANIFEST" \
    --arg o_cap_manifest "$O_CAPPED_MANIFEST" \
    --arg audit_summary "$CLEAN_AUDIT_OUTPUT_ROOT/summary.json" \
    --argjson failures "$failures_json" '
      {
        schema_version: "fixed-step-salvage-orchestrator-v0.1",
        status: $status,
        completed_at: $completed_at,
        hard_stop: $hard_stop,
        fixed_checkpoint: "checkpoint-800",
        seed: 43,
        adapters: {V_S: $s_sha, V_O: $o_sha},
        s_capped_manifest: $s_cap_manifest,
        o_capped_manifest: $o_cap_manifest,
        cross_dataset_order: ["scifact_clean", "rawfc_clean"],
        clean_audit_summary: $audit_summary,
        failures: $failures
      }
    ' > "$tmp"
  mv "$tmp" "$FINAL_MANIFEST"
  event orchestrator "$final_status" "manifest=${FINAL_MANIFEST} failures=${#stage_failures[@]}"
fi
