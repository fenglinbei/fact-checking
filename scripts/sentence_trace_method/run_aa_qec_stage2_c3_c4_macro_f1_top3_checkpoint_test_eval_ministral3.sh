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
PYTHON_SELECT_BIN="${PYTHON_SELECT_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
SPLITS="${SPLITS:-test}"
METRIC="${METRIC:-macro_f1}"
TOP_K="${TOP_K:-3}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-1}"
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

CASES=(
  "C3|liar_raw__ministral3_8b__aa_qec_c3_atom_facts_abc_primary_secondary_fallback_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
  "C4|liar_raw__ministral3_8b__aa_qec_c4_atom_facts_abc_primary_fallback_no_secondary_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
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

use_distributed_eval() {
  if ! [[ "$EVAL_NPROC_PER_NODE" =~ ^[0-9]+$ ]]; then
    printf 'EVAL_NPROC_PER_NODE must be a positive integer: %s\n' "$EVAL_NPROC_PER_NODE" >&2
    exit 2
  fi
  [[ "$EVAL_NPROC_PER_NODE" -gt 1 ]]
}

select_top_checkpoints() {
  local case_root="$1"
  CASE_ROOT="$case_root" METRIC="$METRIC" TOP_K="$TOP_K" "$PYTHON_SELECT_BIN" - <<'PY'
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


case_root = Path(os.environ["CASE_ROOT"])
metric = os.environ["METRIC"]
try:
    top_k = int(os.environ["TOP_K"])
except ValueError as exc:
    raise SystemExit(f"TOP_K must be an integer: {os.environ['TOP_K']}") from exc

eval_root = case_root / "eval"
candidates: list[tuple[float, int]] = []
for metrics_path in sorted(eval_root.glob("step-*/metrics.json")):
    match = re.fullmatch(r"step-(\d+)", metrics_path.parent.name)
    if not match:
        continue
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid metrics JSON: {metrics_path}: {exc}") from exc
    value = metrics.get(metric)
    if value is None:
        continue
    try:
        score = float(value)
    except (TypeError, ValueError):
        continue
    candidates.append((score, int(match.group(1))))

if not candidates:
    raise SystemExit(f"No eval/step-*/metrics.json files with metric={metric!r} under {eval_root}")

selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[:top_k]
if len(selected) < top_k:
    print(
        f"warning: requested top_k={top_k}, found only {len(selected)} metric-bearing checkpoints under {eval_root}",
        file=sys.stderr,
    )

print(",".join(f"checkpoint-{step}" for _, step in selected))
PY
}

IFS=',' read -r -a split_array <<< "$SPLITS"

for case_spec in "${CASES[@]}"; do
  IFS='|' read -r case_id run_name <<< "$case_spec"
  case_root="${OUTPUT_ROOT}/${run_name}"
  run_dir="${case_root}/train"
  config="${case_root}/train.resolved.yaml"
  checkpoints="$(select_top_checkpoints "$case_root")"

  printf '\n[aa-qec-stage2-c3-c4-macro-f1-top3-test-eval] CASE=%s RUN_NAME=%s METRIC=%s TOP_K=%s CHECKPOINTS=%s SPLITS=%s LOGIT_ADJUST=off EVAL_NPROC_PER_NODE=%s NUM_MACHINES=%s MIXED_PRECISION=%s\n' \
    "$case_id" "$run_name" "$METRIC" "$TOP_K" "$checkpoints" "$SPLITS" "$EVAL_NPROC_PER_NODE" "$NUM_MACHINES" "$MIXED_PRECISION"

  if [[ "$DRY_RUN" != "true" ]]; then
    [[ -d "$run_dir" ]] || { printf 'Run directory not found: %s\n' "$run_dir" >&2; exit 2; }
    [[ -f "$config" ]] || { printf 'Config not found: %s\n' "$config" >&2; exit 2; }
  fi

  IFS=',' read -r -a checkpoint_array <<< "$checkpoints"
  for raw_split in "${split_array[@]}"; do
    split="${raw_split// /}"
    [[ -z "$split" ]] && continue
    for raw_checkpoint in "${checkpoint_array[@]}"; do
      checkpoint="${raw_checkpoint// /}"
      [[ -z "$checkpoint" ]] && continue
      output_dir="${case_root}/eval/${split}/${checkpoint}/label_token"
      metrics_path="${output_dir}/metrics.json"
      if [[ "$DRY_RUN" != "true" ]]; then
        [[ -d "${run_dir}/${checkpoint}" ]] || { printf 'Checkpoint directory not found: %s\n' "${run_dir}/${checkpoint}" >&2; exit 2; }
      fi
      if [[ -f "$metrics_path" && "$FORCE_EVAL" != "true" ]]; then
        printf 'Checkpoint test eval already exists: %s; set FORCE_EVAL=true to rerun.\n' "$metrics_path"
        continue
      fi
      infer_args=(-m sft.label_token_infer
        --run-dir "$run_dir"
        --checkpoint "$checkpoint"
        --split "$split"
        --config "$config"
        --output-dir "$output_dir"
        --logit-adjust off
        --log-predictions "$LOG_PREDICTIONS")
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
    done
  done
done
