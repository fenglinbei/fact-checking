#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CASE_ROOT="${CASE_ROOT:-outputs/sentence_trace_method/liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw}"
CONFIG="${CONFIG:-${CASE_ROOT}/train.resolved.yaml}"
RUN_DIR="${RUN_DIR:-${CASE_ROOT}/train}"
SPLITS="${SPLITS:-val,test}"
CHECKPOINTS="${CHECKPOINTS:-best}"
TAUS="${TAUS:-1.0}"
LOGIT_ADJUST_MODE="${LOGIT_ADJUST_MODE:-on}"
FORCE_EVAL="${FORCE_EVAL:-false}"
DRY_RUN="${DRY_RUN:-false}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-}"
LOG_PREDICTIONS="${LOG_PREDICTIONS:-0}"

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

tau_tag() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+}"
  printf '%s\n' "$value"
}

if [[ ! -d "$RUN_DIR" ]]; then
  printf 'Run directory not found: %s\n' "$RUN_DIR" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  printf 'Config not found: %s\n' "$CONFIG" >&2
  exit 2
fi

IFS=',' read -r -a split_array <<< "$SPLITS"
IFS=',' read -r -a checkpoint_array <<< "$CHECKPOINTS"
IFS=',' read -r -a tau_array <<< "$TAUS"

for raw_tau in "${tau_array[@]}"; do
  tau="${raw_tau// /}"
  [[ -z "$tau" ]] && continue
  tag="$(tau_tag "$tau")"
  for raw_split in "${split_array[@]}"; do
    split="${raw_split// /}"
    [[ -z "$split" ]] && continue
    for raw_checkpoint in "${checkpoint_array[@]}"; do
      checkpoint="${raw_checkpoint// /}"
      [[ -z "$checkpoint" ]] && continue
      output_dir="${CASE_ROOT}/eval/${split}/${checkpoint}/label_token_logit_adjust_tau${tag}"
      metrics_path="${output_dir}/metrics.json"
      if [[ -f "$metrics_path" && "$FORCE_EVAL" != "true" ]]; then
        printf 'Logit-adjust eval already exists: %s; set FORCE_EVAL=true to rerun.\n' "$metrics_path"
        continue
      fi

      cmd=("$PYTHON_BIN" -m sft.label_token_infer
        --run-dir "$RUN_DIR"
        --checkpoint "$checkpoint"
        --split "$split"
        --config "$CONFIG"
        --output-dir "$output_dir"
        --logit-adjust "$LOGIT_ADJUST_MODE"
        --logit-adjust-tau "$tau"
        --log-predictions "$LOG_PREDICTIONS")
      if [[ -n "$PER_DEVICE_EVAL_BATCH_SIZE" ]]; then
        cmd+=(--per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE")
      fi
      if [[ -n "$DATALOADER_NUM_WORKERS" ]]; then
        cmd+=(--dataloader-num-workers "$DATALOADER_NUM_WORKERS")
      fi
      run_cmd "${cmd[@]}"
    done
  done
done
