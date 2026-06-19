#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
PYTHON_SELECT_BIN="${PYTHON_SELECT_BIN:-$PYTHON_BIN}"
CASE_ROOT="${CASE_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__mrec_min_lora}"
RUN_DIR="${RUN_DIR:-${CASE_ROOT}/train}"
CONFIG="${CONFIG:-${CASE_ROOT}/train.resolved.yaml}"
EVAL_SPLITS="${EVAL_SPLITS:-val,test}"
TAU_GRID="${TAU_GRID:-0,0.25,0.5,0.75,1}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
FORCE_EVAL="${FORCE_EVAL:-false}"
DRY_RUN="${DRY_RUN:-false}"
LOG_PREDICTIONS="${LOG_PREDICTIONS:-0}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-}"

if [[ -z "${ACCELERATE_BIN:-}" ]]; then
  py_dir="$(dirname "$PYTHON_BIN")"
  if [[ -x "${py_dir}/accelerate" ]]; then
    ACCELERATE_BIN="${py_dir}/accelerate"
  else
    ACCELERATE_BIN="accelerate"
  fi
fi

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

use_distributed_eval() {
  if ! [[ "$EVAL_NPROC_PER_NODE" =~ ^[0-9]+$ ]]; then
    printf 'EVAL_NPROC_PER_NODE must be a positive integer: %s\n' "$EVAL_NPROC_PER_NODE" >&2
    exit 2
  fi
  [[ "$EVAL_NPROC_PER_NODE" -gt 1 ]]
}

select_checkpoint() {
  local policy="$1"
  "$PYTHON_SELECT_BIN" scripts/sentence_trace_method/select_checkpoint_for_eval.py checkpoint \
    --case-root "$CASE_ROOT" \
    --policy "$policy"
}

build_plan() {
  local checkpoint="$1"
  local plan_kind="$2"
  "$PYTHON_SELECT_BIN" scripts/sentence_trace_method/select_checkpoint_for_eval.py plan \
    --case-root "$CASE_ROOT" \
    --checkpoint "$checkpoint" \
    --plan-kind "$plan_kind" \
    --eval-splits "$EVAL_SPLITS" \
    --tau-grid "$TAU_GRID"
}

merge_plan_json() {
  "$PYTHON_SELECT_BIN" -c 'import json, sys; out = []; [out.extend(json.loads(arg)) for arg in sys.argv[1:]]; print(json.dumps(out, sort_keys=True))' "$@"
}

append_eval_options() {
  if [[ -n "$PER_DEVICE_EVAL_BATCH_SIZE" ]]; then
    cmd+=(--per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE")
  fi
  if [[ -n "$DATALOADER_NUM_WORKERS" ]]; then
    cmd+=(--dataloader-num-workers "$DATALOADER_NUM_WORKERS")
  fi
}

print_plan_kind_summary() {
  local checkpoint="$1"
  local plan_kind="$2"
  case "$plan_kind" in
    current_macro_f1)
      printf '\n[checkpoint-gap-matrix] EXPERIMENT=%s CHECKPOINT=%s LOGIT_ADJUST=on TAU=%s SPLITS=%s EVAL_NPROC_PER_NODE=%s\n' \
        "E0_current_macro_f1_tau1" "$checkpoint" "1.0" "$EVAL_SPLITS" "$EVAL_NPROC_PER_NODE"
      ;;
    macro_f1)
      printf '\n[checkpoint-gap-matrix] EXPERIMENT=%s CHECKPOINT=%s LOGIT_ADJUST=on TAU=%s SPLITS=%s EVAL_NPROC_PER_NODE=%s\n' \
        "E1_val_macro_f1_tau1" "$checkpoint" "1.0" "$EVAL_SPLITS" "$EVAL_NPROC_PER_NODE"
      printf '[checkpoint-gap-matrix] EXPERIMENT=%s CHECKPOINT=%s LOGIT_ADJUST=on TAU=%s SPLITS=%s EVAL_NPROC_PER_NODE=%s\n' \
        "E2_val_macro_f1_tau0" "$checkpoint" "0.0" "$EVAL_SPLITS" "$EVAL_NPROC_PER_NODE"
      printf '[checkpoint-gap-matrix] EXPERIMENT=%s CHECKPOINT=%s LOGIT_ADJUST=on TAU_POLICY=val_macro_f1 TAU_GRID=%s EVAL_NPROC_PER_NODE=%s\n' \
        "E3_val_macro_f1_val_selected_tau" "$checkpoint" "$TAU_GRID" "$EVAL_NPROC_PER_NODE"
      ;;
    one_standard_error)
      printf '\n[checkpoint-gap-matrix] EXPERIMENT=%s CHECKPOINT=%s LOGIT_ADJUST=on TAU=%s SPLITS=%s EVAL_NPROC_PER_NODE=%s\n' \
        "E4_one_se_tau1" "$checkpoint" "1.0" "$EVAL_SPLITS" "$EVAL_NPROC_PER_NODE"
      printf '[checkpoint-gap-matrix] EXPERIMENT=%s CHECKPOINT=%s LOGIT_ADJUST=on TAU=%s SPLITS=%s EVAL_NPROC_PER_NODE=%s\n' \
        "E5_one_se_tau0" "$checkpoint" "0.0" "$EVAL_SPLITS" "$EVAL_NPROC_PER_NODE"
      printf '[checkpoint-gap-matrix] EXPERIMENT=%s CHECKPOINT=%s LOGIT_ADJUST=on TAU_POLICY=val_macro_f1 TAU_GRID=%s EVAL_NPROC_PER_NODE=%s\n' \
        "E6_one_se_val_selected_tau" "$checkpoint" "$TAU_GRID" "$EVAL_NPROC_PER_NODE"
      ;;
    *)
      printf 'Unsupported plan kind: %s\n' "$plan_kind" >&2
      exit 2
      ;;
  esac
}

run_multi_eval_checkpoint() {
  local checkpoint="$1"
  local plan_kinds_csv="$2"
  if [[ "$DRY_RUN" != "true" ]]; then
    [[ -d "$RUN_DIR" ]] || { printf 'Run directory not found: %s\n' "$RUN_DIR" >&2; exit 2; }
    [[ -f "$CONFIG" ]] || { printf 'Config not found: %s\n' "$CONFIG" >&2; exit 2; }
    if [[ "$checkpoint" != "best" ]]; then
      [[ -d "${RUN_DIR}/${checkpoint}" ]] || { printf 'Checkpoint directory not found: %s\n' "${RUN_DIR}/${checkpoint}" >&2; exit 2; }
    fi
  fi
  IFS=',' read -r -a plan_kind_array <<< "$plan_kinds_csv"
  plan_json_args=()
  for plan_kind in "${plan_kind_array[@]}"; do
    [[ -z "$plan_kind" ]] && continue
    print_plan_kind_summary "$checkpoint" "$plan_kind"
    plan_json_args+=("$(build_plan "$checkpoint" "$plan_kind")")
  done
  plan_json="$(merge_plan_json "${plan_json_args[@]}")"
  infer_args=(-m sft.label_token_multi_infer
    --run-dir "$RUN_DIR"
    --checkpoint "$checkpoint"
    --config "$CONFIG"
    --plan-json "$plan_json"
    --log-predictions "$LOG_PREDICTIONS")
  if [[ "$FORCE_EVAL" == "true" ]]; then
    infer_args+=(--force-eval)
  fi
  if use_distributed_eval; then
    cmd=("$ACCELERATE_BIN" launch
      --num_processes "$EVAL_NPROC_PER_NODE"
      --num_machines "$NUM_MACHINES"
      --mixed_precision "$MIXED_PRECISION"
      "${infer_args[@]}")
  else
    cmd=("$PYTHON_BIN" "${infer_args[@]}")
  fi
  append_eval_options
  run_cmd "${cmd[@]}"
}

declare -a checkpoint_order=()
declare -A plan_kinds_by_checkpoint=()

add_plan_kind() {
  local checkpoint="$2"
  local plan_kind="$1"
  if [[ -z "${plan_kinds_by_checkpoint[$checkpoint]+set}" ]]; then
    checkpoint_order+=("$checkpoint")
    plan_kinds_by_checkpoint[$checkpoint]="$plan_kind"
  else
    plan_kinds_by_checkpoint[$checkpoint]="${plan_kinds_by_checkpoint[$checkpoint]},${plan_kind}"
  fi
}

macro_checkpoint="$(select_checkpoint macro_f1)"
one_se_checkpoint="$(select_checkpoint one_standard_error)"

add_plan_kind current_macro_f1 "$macro_checkpoint"
add_plan_kind macro_f1 "$macro_checkpoint"
add_plan_kind one_standard_error "$one_se_checkpoint"

for checkpoint in "${checkpoint_order[@]}"; do
  run_multi_eval_checkpoint "$checkpoint" "${plan_kinds_by_checkpoint[$checkpoint]}"
done
