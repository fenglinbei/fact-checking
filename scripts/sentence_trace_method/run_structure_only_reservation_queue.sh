#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc-gemma4/bin/python" ]]; then
    PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-gemma4/bin/python"
  elif [[ -x "/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python" ]]; then
    PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

DRY_RUN="${DRY_RUN:-false}"
QUEUE_CUTOFF="${QUEUE_CUTOFF:-2026-07-17T11:40:00+08:00}"
QUEUE_ID="${QUEUE_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
QUEUE_RUN_DIR="${QUEUE_RUN_DIR:-outputs/sentence_trace_method/queues/structure_only_reservation_${QUEUE_ID}}"
QUEUE_CUDA_VISIBLE_DEVICES="${QUEUE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
TASK_LIST_RAW="${*:-${TASK_LIST:-}}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sentence_trace_method/run_structure_only_reservation_queue.sh \
    rawfc_clean:full scifact_clean:full v_r:full

Task names: rawfc_clean, scifact_clean, v_r, no_map, seed43_s, seed43_o
Task modes: train, eval, full

The mode is part of each task token. v_r and no_map are mutually exclusive in
one queue unless ALLOW_BOTH_GATE_BRANCHES=true is set explicitly. The seed43
tasks must appear exactly once each, use the same mode, and be adjacent in the
order seed43_s -> seed43_o. Set ALLOW_SEED43_PAIR_OVERRIDE=true only for a
targeted test or recovery operation.

Useful environment variables:
  DRY_RUN=true
  QUEUE_CUTOFF=2026-07-17T11:40:00+08:00
  QUEUE_RUN_DIR=outputs/sentence_trace_method/queues/<queue-id>
  QUEUE_CUDA_VISIBLE_DEVICES=0,1,2,3
  ALLOW_SEED43_PAIR_OVERRIDE=false
EOF
}

if [[ -z "${TASK_LIST_RAW//[[:space:],]/}" ]]; then
  usage >&2
  exit 2
fi

if [[ -e "$QUEUE_RUN_DIR" ]]; then
  shopt -s nullglob dotglob
  existing_queue_files=("$QUEUE_RUN_DIR"/*)
  if (( ${#existing_queue_files[@]} > 0 )); then
    printf '[reservation-queue] refusing to overwrite non-empty queue directory: %s\n' "$QUEUE_RUN_DIR" >&2
    exit 2
  fi
fi
mkdir -p "$QUEUE_RUN_DIR/logs"

EVENTS_FILE="$QUEUE_RUN_DIR/events.tsv"
PLAN_FILE="$QUEUE_RUN_DIR/plan.tsv"
printf 'timestamp\tindex\ttask\tmode\tstatus\texit_code\tdetail\tlog\n' > "$EVENTS_FILE"
printf 'index\ttask\tmode\tconfig\twrapper\trun_root\tcompletion_contract\n' > "$PLAN_FILE"

timestamp() {
  date --iso-8601=seconds
}

append_event() {
  local index="$1" task="$2" mode="$3" status="$4" exit_code="$5" detail="$6" log_path="$7"
  detail="${detail//$'\t'/ }"
  detail="${detail//$'\n'/ }"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(timestamp)" "$index" "$task" "$mode" "$status" "$exit_code" "$detail" "$log_path" \
    >> "$EVENTS_FILE"
}

cutoff_epoch="$(date -d "$QUEUE_CUTOFF" +%s 2>/dev/null || true)"
if [[ ! "$cutoff_epoch" =~ ^[0-9]+$ ]]; then
  printf '[reservation-queue] invalid QUEUE_CUTOFF=%s\n' "$QUEUE_CUTOFF" >&2
  exit 2
fi

now_epoch() {
  if [[ -n "${QUEUE_NOW_EPOCH:-}" ]]; then
    printf '%s\n' "$QUEUE_NOW_EPOCH"
  else
    date +%s
  fi
}

derive_run_root() {
  local config="$1"
  (
    eval "$("$PYTHON_BIN" scripts/sentence_trace_method/mrec_policy_config.py --config "$config")"
    printf '%s/%s%s%s\n' "$OUTPUT_ROOT" "$BASE_CASE_NAME" "$CASE_SUFFIX" "$LORA_SUFFIX"
  )
}

resolve_task() {
  local task="$1" override_var override_value
  TASK_CHECKPOINTS_OVERRIDE=""
  case "$task" in
    rawfc_clean)
      TASK_CONFIG="configs/experiment/mrec_v0.2/rawfc_learned_marginal_structure_only_fullpool_minmax5_10_baseline20.yaml"
      TASK_WRAPPER="scripts/sentence_trace_method/run_rawfc_ministral3_atom_anchor_v0_2_structure_only_fullpool_minmax5_10_baseline20_lora_r16a32_d010_lr1e5_ep12_eval50.sh"
      TASK_CHECKPOINT="best"
      TASK_EVAL_SPLITS="val,test"
      TASK_SPECIAL="standard"
      override_var="QUEUE_RAWFC_CLEAN_RUN_ROOT"
      ;;
    scifact_clean)
      TASK_CONFIG="configs/experiment/mrec_v0.2/scifact_atom_union_structure_only_fullpool_minmax9_9.yaml"
      TASK_WRAPPER="scripts/phase13_scifact/05_train_eval_scifact_structure_only_fullpool_lora.sh"
      TASK_CHECKPOINT="best"
      TASK_EVAL_SPLITS="val"
      TASK_SPECIAL="scifact"
      override_var="QUEUE_SCIFACT_CLEAN_RUN_ROOT"
      ;;
    v_r)
      TASK_CONFIG="configs/experiment/mrec_v0.2/retrieval_order_matched_fullpool_minmax5_10.yaml"
      TASK_WRAPPER="scripts/sentence_trace_method/run_liar_raw_ministral3_retrieval_order_matched_verifier.sh"
      TASK_CHECKPOINT="checkpoint-800"
      TASK_EVAL_SPLITS="val"
      TASK_SPECIAL="standard"
      override_var="QUEUE_V_R_RUN_ROOT"
      ;;
    no_map)
      TASK_CONFIG="configs/experiment/mrec_v0.2/learned_marginal_structure_only_no_map_fullpool_minmax5_10.yaml"
      TASK_WRAPPER="scripts/sentence_trace_method/run_liar_raw_ministral3_structure_only_no_map.sh"
      TASK_CHECKPOINT="best"
      TASK_EVAL_SPLITS="val,test"
      TASK_SPECIAL="standard"
      override_var="QUEUE_NO_MAP_RUN_ROOT"
      ;;
    seed43_s)
      TASK_CONFIG="configs/experiment/mrec_v0.2/learned_marginal_structure_only_fullpool_minmax5_10_seed43.yaml"
      TASK_WRAPPER="scripts/sentence_trace_method/run_liar_raw_ministral3_structure_only_seed43.sh"
      TASK_CHECKPOINT="best"
      TASK_CHECKPOINTS_OVERRIDE="best"
      TASK_EVAL_SPLITS="val"
      TASK_SPECIAL="seed43"
      override_var="QUEUE_SEED43_S_RUN_ROOT"
      ;;
    seed43_o)
      TASK_CONFIG="configs/experiment/mrec_v0.2/learned_marginal_structure_only_one_shot_fullpool_minmax5_10_seed43.yaml"
      TASK_WRAPPER="scripts/sentence_trace_method/run_liar_raw_ministral3_structure_only_one_shot_seed43.sh"
      TASK_CHECKPOINT="best"
      TASK_CHECKPOINTS_OVERRIDE="best"
      TASK_EVAL_SPLITS="val"
      TASK_SPECIAL="seed43"
      override_var="QUEUE_SEED43_O_RUN_ROOT"
      ;;
    *)
      printf '[reservation-queue] unsupported task name: %s\n' "$task" >&2
      return 2
      ;;
  esac

  override_value="${!override_var:-}"
  if [[ -n "$override_value" ]]; then
    TASK_RUN_ROOT="$override_value"
  else
    TASK_RUN_ROOT="$(derive_run_root "$TASK_CONFIG")"
  fi

  local wrapper_override_var
  wrapper_override_var="QUEUE_${task^^}_WRAPPER"
  wrapper_override_var="${wrapper_override_var//-/_}"
  if [[ -n "${!wrapper_override_var:-}" ]]; then
    TASK_WRAPPER="${!wrapper_override_var}"
  fi
}

training_complete() {
  local marker="$1/train/training_complete.json"
  [[ -s "$marker" ]] && jq -e '.completed == true' "$marker" >/dev/null 2>&1
}

seed43_training_complete() {
  local root="$1"
  local checkpoint_dir="$root/train/checkpoint-800"
  local resolved_config="$root/train.resolved.yaml"
  training_complete "$root" &&
    [[ -s "$checkpoint_dir/adapter_model.safetensors" ]] &&
    [[ -s "$checkpoint_dir/adapter_config.json" ]] &&
    [[ -s "$resolved_config" ]] &&
    "$PYTHON_BIN" -c \
      'import sys, yaml; payload = yaml.safe_load(open(sys.argv[1], encoding="utf-8")); seed = payload["sft_train"].get("seed") if isinstance(payload, dict) and isinstance(payload.get("sft_train"), dict) else None; raise SystemExit(0 if type(seed) is int and seed == 43 else 1)' \
      "$resolved_config" >/dev/null 2>&1
}

task_training_complete() {
  if [[ "$TASK_SPECIAL" == "seed43" ]]; then
    seed43_training_complete "$TASK_RUN_ROOT"
  else
    training_complete "$TASK_RUN_ROOT"
  fi
}

standard_eval_complete() {
  local root="$1" splits="$2" checkpoint="$3" split
  IFS=',' read -r -a split_array <<< "$splits"
  for split in "${split_array[@]}"; do
    split="${split// /}"
    [[ -z "$split" ]] && continue
    if [[ ! -s "$root/eval/$split/$checkpoint/label_token/metrics.json" ]]; then
      return 1
    fi
  done
  return 0
}

scifact_eval_complete() {
  local root="$1"
  [[ -s "$root/eval/val/best/label_token/metrics.json" ]] &&
    [[ -s "$root/eval/test/best/label_token/prediction_manifest.json" ]] &&
    [[ -s "$root/submission/scifact_official_style_metrics_val.json" ]] &&
    [[ -s "$root/submission/scifact_submission_val.jsonl" ]] &&
    [[ -s "$root/submission/scifact_submission_test.jsonl" ]]
}

eval_complete() {
  if [[ "$TASK_SPECIAL" == "scifact" ]]; then
    scifact_eval_complete "$TASK_RUN_ROOT"
  else
    standard_eval_complete "$TASK_RUN_ROOT" "$TASK_EVAL_SPLITS" "$TASK_CHECKPOINT"
  fi
}

task_complete() {
  local mode="$1"
  case "$mode" in
    train) task_training_complete ;;
    eval) task_training_complete && eval_complete ;;
    full) task_training_complete && eval_complete ;;
    *) return 1 ;;
  esac
}

completion_contract() {
  local mode="$1" contract split
  contract="$TASK_RUN_ROOT/train/training_complete.json:.completed=true"
  if [[ "$TASK_SPECIAL" == "seed43" ]]; then
    contract+=";$TASK_RUN_ROOT/train/checkpoint-800/adapter_model.safetensors"
    contract+=";$TASK_RUN_ROOT/train/checkpoint-800/adapter_config.json"
    contract+=";$TASK_RUN_ROOT/train.resolved.yaml:sft_train.seed=43"
  fi
  if [[ "$mode" == "train" ]]; then
    printf '%s\n' "$contract"
    return 0
  fi
  if [[ "$TASK_SPECIAL" == "scifact" ]]; then
    contract+=";$TASK_RUN_ROOT/eval/val/best/label_token/metrics.json"
    contract+=";$TASK_RUN_ROOT/eval/test/best/label_token/prediction_manifest.json"
    contract+=";$TASK_RUN_ROOT/submission/scifact_official_style_metrics_val.json"
    contract+=";$TASK_RUN_ROOT/submission/scifact_submission_val.jsonl"
    contract+=";$TASK_RUN_ROOT/submission/scifact_submission_test.jsonl"
  else
    IFS=',' read -r -a split_array <<< "$TASK_EVAL_SPLITS"
    for split in "${split_array[@]}"; do
      split="${split// /}"
      [[ -z "$split" ]] && continue
      contract+=";$TASK_RUN_ROOT/eval/$split/$TASK_CHECKPOINT/label_token/metrics.json"
    done
  fi
  printf '%s\n' "$contract"
}

TASK_LIST_NORMALIZED="${TASK_LIST_RAW//,/ }"
read -r -a task_specs <<< "$TASK_LIST_NORMALIZED"

declare -A seen_specs=()
has_v_r=false
has_no_map=false
seed43_s_count=0
seed43_o_count=0
seed43_s_index=-1
seed43_o_index=-1
seed43_s_mode=""
seed43_o_mode=""
spec_index=0
for spec in "${task_specs[@]}"; do
  [[ -z "$spec" ]] && continue
  spec_index=$((spec_index + 1))
  if [[ ! "$spec" =~ ^(rawfc_clean|scifact_clean|v_r|no_map|seed43_s|seed43_o):(train|eval|full)$ ]]; then
    printf '[reservation-queue] invalid task token: %s\n' "$spec" >&2
    usage >&2
    exit 2
  fi
  if [[ -n "${seen_specs[$spec]:-}" ]]; then
    printf '[reservation-queue] duplicate task token: %s\n' "$spec" >&2
    exit 2
  fi
  seen_specs[$spec]=1
  [[ "$spec" == v_r:* ]] && has_v_r=true
  [[ "$spec" == no_map:* ]] && has_no_map=true
  if [[ "$spec" == seed43_s:* ]]; then
    seed43_s_count=$((seed43_s_count + 1))
    seed43_s_index="$spec_index"
    seed43_s_mode="${spec##*:}"
  fi
  if [[ "$spec" == seed43_o:* ]]; then
    seed43_o_count=$((seed43_o_count + 1))
    seed43_o_index="$spec_index"
    seed43_o_mode="${spec##*:}"
  fi
done
if [[ "$has_v_r" == "true" && "$has_no_map" == "true" && "${ALLOW_BOTH_GATE_BRANCHES:-false}" != "true" ]]; then
  printf '[reservation-queue] v_r and no_map are mutually exclusive gate branches; choose one.\n' >&2
  exit 2
fi
if [[ "${ALLOW_SEED43_PAIR_OVERRIDE:-false}" != "true" ]] && \
    (( seed43_s_count > 0 || seed43_o_count > 0 )); then
  if (( seed43_s_count != 1 || seed43_o_count != 1 )); then
    printf '[reservation-queue] seed43 tasks must be paired exactly once as seed43_s -> seed43_o; set ALLOW_SEED43_PAIR_OVERRIDE=true only for test/recovery.\n' >&2
    exit 2
  fi
  if (( seed43_o_index != seed43_s_index + 1 )); then
    printf '[reservation-queue] seed43 pair must be adjacent and ordered seed43_s -> seed43_o; set ALLOW_SEED43_PAIR_OVERRIDE=true only for test/recovery.\n' >&2
    exit 2
  fi
  if [[ "$seed43_s_mode" != "$seed43_o_mode" ]]; then
    printf '[reservation-queue] seed43 pair must use the same mode; got seed43_s:%s seed43_o:%s.\n' "$seed43_s_mode" "$seed43_o_mode" >&2
    exit 2
  fi
fi

printf '[reservation-queue] id=%s cutoff=%s cutoff_epoch=%s dry_run=%s cuda=%s tasks=%s output=%s\n' \
  "$QUEUE_ID" "$QUEUE_CUTOFF" "$cutoff_epoch" "$DRY_RUN" "$QUEUE_CUDA_VISIBLE_DEVICES" \
  "$TASK_LIST_NORMALIZED" "$QUEUE_RUN_DIR"

task_index=0
for spec in "${task_specs[@]}"; do
  [[ -z "$spec" ]] && continue
  task_index=$((task_index + 1))
  task="${spec%%:*}"
  mode="${spec##*:}"
  resolve_task "$task"

  if [[ ! -f "$TASK_CONFIG" ]]; then
    append_event "$task_index" "$task" "$mode" "preflight_failed" 2 "missing config: $TASK_CONFIG" ""
    printf '[reservation-queue] missing config: %s\n' "$TASK_CONFIG" >&2
    exit 2
  fi
  if [[ ! -f "$TASK_WRAPPER" ]]; then
    append_event "$task_index" "$task" "$mode" "preflight_failed" 2 "missing wrapper: $TASK_WRAPPER" ""
    printf '[reservation-queue] missing wrapper: %s\n' "$TASK_WRAPPER" >&2
    exit 2
  fi

  log_path="$QUEUE_RUN_DIR/logs/$(printf '%02d' "$task_index")_${task}_${mode}.log"
  task_completion_contract="$(completion_contract "$mode")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$task_index" "$task" "$mode" "$TASK_CONFIG" "$TASK_WRAPPER" "$TASK_RUN_ROOT" \
    "$task_completion_contract" >> "$PLAN_FILE"

  if task_complete "$mode"; then
    printf '[reservation-queue] SKIP complete task=%s mode=%s root=%s\n' "$task" "$mode" "$TASK_RUN_ROOT" | tee "$log_path"
    append_event "$task_index" "$task" "$mode" "skipped_complete" 0 "completion markers already valid" "$log_path"
    continue
  fi

  current_epoch="$(now_epoch)"
  if [[ ! "$current_epoch" =~ ^[0-9]+$ ]]; then
    append_event "$task_index" "$task" "$mode" "preflight_failed" 2 "invalid current epoch: $current_epoch" "$log_path"
    printf '[reservation-queue] invalid QUEUE_NOW_EPOCH/current epoch: %s\n' "$current_epoch" >&2
    exit 2
  fi
  if (( current_epoch >= cutoff_epoch )); then
    printf '[reservation-queue] CUTOFF task=%s mode=%s now=%s cutoff=%s; not launched\n' \
      "$task" "$mode" "$current_epoch" "$cutoff_epoch" | tee "$log_path"
    append_event "$task_index" "$task" "$mode" "cutoff_blocked" 3 "startup cutoff reached" "$log_path"
    exit 3
  fi

  if [[ "$mode" == "eval" && "$DRY_RUN" != "true" ]] && ! task_training_complete; then
    printf '[reservation-queue] PRECONDITION task=%s mode=eval missing valid training_complete.json\n' "$task" | tee "$log_path"
    append_event "$task_index" "$task" "$mode" "preflight_failed" 4 "eval requires completed training" "$log_path"
    exit 4
  fi

  runtime_root="/tmp/lzj_structure_queue_${QUEUE_ID}_${task_index}_${task}_${mode}"
  printf '[reservation-queue] START task=%s mode=%s root=%s wrapper=%s log=%s runtime=%s\n' \
    "$task" "$mode" "$TASK_RUN_ROOT" "$TASK_WRAPPER" "$log_path" "$runtime_root" | tee "$log_path"
  append_event "$task_index" "$task" "$mode" "planned" 0 "startup checks passed" "$log_path"

  task_env=(
    "CUDA_VISIBLE_DEVICES=$QUEUE_CUDA_VISIBLE_DEVICES"
    "MODE=$mode"
    "DRY_RUN=false"
    "FORCE_TRAIN=false"
    "FORCE_EVAL=false"
    "CUDA_DEVICE_MEMORY_SHARED_CACHE=$runtime_root/cudevshr.cache"
    "TRITON_CACHE_DIR=$runtime_root/triton"
    "TORCHINDUCTOR_CACHE_DIR=$runtime_root/torchinductor"
    "QUEUE_TASK_RUN_ROOT=$TASK_RUN_ROOT"
  )
  if [[ -n "$TASK_CHECKPOINTS_OVERRIDE" ]]; then
    task_env+=("CHECKPOINTS=$TASK_CHECKPOINTS_OVERRIDE")
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    {
      printf '+ env'
      printf ' %q' "${task_env[@]}"
      printf ' bash %q\n' "$TASK_WRAPPER"
    } | tee -a "$log_path"
    append_event "$task_index" "$task" "$mode" "dry_run" 0 "command emitted; wrapper not invoked" "$log_path"
    continue
  fi

  mkdir -p "$runtime_root/triton" "$runtime_root/torchinductor"
  set +e
  env "${task_env[@]}" bash "$TASK_WRAPPER" 2>&1 | tee -a "$log_path"
  task_rc=${PIPESTATUS[0]}
  set -e

  if (( task_rc != 0 )); then
    append_event "$task_index" "$task" "$mode" "failed" "$task_rc" "wrapper exited non-zero; queue stopped" "$log_path"
    printf '[reservation-queue] FAIL task=%s mode=%s rc=%s; queue stopped\n' "$task" "$mode" "$task_rc" >&2
    exit "$task_rc"
  fi
  if ! task_complete "$mode"; then
    append_event "$task_index" "$task" "$mode" "marker_missing" 5 "wrapper exited zero but completion markers are incomplete" "$log_path"
    printf '[reservation-queue] FAIL task=%s mode=%s: completion marker audit failed\n' "$task" "$mode" >&2
    exit 5
  fi

  append_event "$task_index" "$task" "$mode" "completed" 0 "wrapper and completion-marker checks passed" "$log_path"
  printf '[reservation-queue] DONE task=%s mode=%s\n' "$task" "$mode"
done

printf 'completed_at\tqueue_id\ttask_count\tdry_run\n%s\t%s\t%s\t%s\n' \
  "$(timestamp)" "$QUEUE_ID" "$task_index" "$DRY_RUN" > "$QUEUE_RUN_DIR/queue_complete.tsv"
printf '[reservation-queue] COMPLETE tasks=%s dry_run=%s audit=%s\n' "$task_index" "$DRY_RUN" "$QUEUE_RUN_DIR"
