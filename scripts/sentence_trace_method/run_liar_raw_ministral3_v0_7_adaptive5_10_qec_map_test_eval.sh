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
RUN_NAME="${RUN_NAME:-liar_raw__ministral3_8b__v0_7_bm_adaptive5_10__qec_map_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
SPLIT="${SPLIT:-test}"
CHECKPOINT="${CHECKPOINT:-best}"
FORCE_EVAL="${FORCE_EVAL:-true}"
DRY_RUN="${DRY_RUN:-false}"
LOG_PREDICTIONS="${LOG_PREDICTIONS:-0}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-}"

CASE_ROOT="${CASE_ROOT:-${OUTPUT_ROOT}/${RUN_NAME}}"
RUN_DIR="${RUN_DIR:-${CASE_ROOT}/train}"
CONFIG="${CONFIG:-${CASE_ROOT}/train.resolved.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${CASE_ROOT}/eval/${SPLIT}/${CHECKPOINT}/label_token}"
METRICS_PATH="${OUTPUT_DIR}/metrics.json"

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

if [[ "$DRY_RUN" != "true" ]]; then
  [[ -d "$RUN_DIR" ]] || { printf 'Run directory not found: %s\n' "$RUN_DIR" >&2; exit 2; }
  [[ -f "$CONFIG" ]] || { printf 'Config not found: %s\n' "$CONFIG" >&2; exit 2; }
fi

if [[ -f "$METRICS_PATH" && "$FORCE_EVAL" != "true" ]]; then
  printf 'Test eval already exists: %s; set FORCE_EVAL=true to rerun.\n' "$METRICS_PATH"
  exit 0
fi

printf '\n[liar-raw-ministral3-qec-map-test-eval] RUN_NAME=%s SPLIT=%s CHECKPOINT=%s OUTPUT_DIR=%s LOGIT_ADJUST=off\n' \
  "$RUN_NAME" "$SPLIT" "$CHECKPOINT" "$OUTPUT_DIR"

cmd=("$PYTHON_BIN" -m sft.label_token_infer
  --run-dir "$RUN_DIR"
  --checkpoint "$CHECKPOINT"
  --split "$SPLIT"
  --config "$CONFIG"
  --output-dir "$OUTPUT_DIR"
  --logit-adjust off
  --log-predictions "$LOG_PREDICTIONS")
append_eval_options
run_cmd "${cmd[@]}"
