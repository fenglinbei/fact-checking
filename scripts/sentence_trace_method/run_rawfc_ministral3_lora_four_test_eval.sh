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
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
SPLIT="${SPLIT:-test}"
CHECKPOINT="${CHECKPOINT:-best}"
FORCE_EVAL="${FORCE_EVAL:-true}"
DRY_RUN="${DRY_RUN:-false}"
LOG_PREDICTIONS="${LOG_PREDICTIONS:-0}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-}"

RUN_NAMES=(
  "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_ebs16_lr1em5_ep12_eval100_pat8_rawfc"
  "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r32a64_d005_ebs16_lr1em5_ep12_eval50_pat8_rawfc"
  "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r16a32_d010_ebs16_lr1em5_ep12_eval50_pat8_rawfc"
  "rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_r16a32_d005_ebs16_lr5em6_ep12_eval50_pat8_rawfc"
)

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

append_eval_options() {
  if [[ -n "$PER_DEVICE_EVAL_BATCH_SIZE" ]]; then
    cmd+=(--per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE")
  fi
  if [[ -n "$DATALOADER_NUM_WORKERS" ]]; then
    cmd+=(--dataloader-num-workers "$DATALOADER_NUM_WORKERS")
  fi
}

run_plain_eval() {
  local case_root="$1"
  local run_dir="${case_root}/train"
  local config="${case_root}/train.resolved.yaml"
  local output_dir="${case_root}/eval/${SPLIT}/${CHECKPOINT}/label_token"
  local metrics_path="${output_dir}/metrics.json"

  if [[ "$DRY_RUN" != "true" ]]; then
    [[ -d "$run_dir" ]] || { printf 'Run directory not found: %s\n' "$run_dir" >&2; exit 2; }
    [[ -f "$config" ]] || { printf 'Config not found: %s\n' "$config" >&2; exit 2; }
  fi
  if [[ -f "$metrics_path" && "$FORCE_EVAL" != "true" ]]; then
    printf 'Test eval already exists: %s; set FORCE_EVAL=true to rerun.\n' "$metrics_path"
    return 0
  fi

  cmd=("$PYTHON_BIN" -m sft.label_token_infer
    --run-dir "$run_dir"
    --checkpoint "$CHECKPOINT"
    --split "$SPLIT"
    --config "$config"
    --output-dir "$output_dir"
    --logit-adjust off
    --log-predictions "$LOG_PREDICTIONS")
  append_eval_options
  run_cmd "${cmd[@]}"
}

for run_name in "${RUN_NAMES[@]}"; do
  case_root="${OUTPUT_ROOT}/${run_name}"
  printf '\n== test eval: %s split=%s checkpoint=%s ==\n' "$run_name" "$SPLIT" "$CHECKPOINT"
  run_plain_eval "$case_root"
done
