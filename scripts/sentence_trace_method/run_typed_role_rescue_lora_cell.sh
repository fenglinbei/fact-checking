#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CELL="${1:-}"
case "$CELL" in
  r_only|cor|opp|ctx|retr|random|full) ;;
  *)
    printf 'Usage: %s {r_only|cor|opp|ctx|retr|random|full}\n' "$0" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$(dirname "$PYTHON_BIN")/accelerate}"
DATA_ROOT="${DATA_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__typed_role_rescue_v0_1}"
SOURCE_ROOT="${SOURCE_ROOT:-${DATA_ROOT}/${CELL}}"
LORA_SUFFIX="${LORA_SUFFIX:-__lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
LORA_ROOT="${LORA_ROOT:-${DATA_ROOT}/${CELL}${LORA_SUFFIX}}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/deepspeed_zero2_bsz1_ga4.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29631}"
EVAL_PROCESS_PORT="${EVAL_PROCESS_PORT:-29632}"
FORCE_CONFIG="${FORCE_CONFIG:-false}"
FORCE_EVAL="${FORCE_EVAL:-false}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

RUNTIME_ROOT="${RUNTIME_ROOT:-outputs/cache/runtime/typed_role_rescue_v0_1/${CELL}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_ROOT}/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUNTIME_ROOT}/torchinductor}"
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

SOURCE_CONFIG="${SOURCE_ROOT}/train.resolved.yaml"
if [[ ! -f "$SOURCE_CONFIG" ]]; then
  printf 'Missing role-rescue source config: %s\n' "$SOURCE_CONFIG" >&2
  exit 2
fi

prepare_args=(
  "$PYTHON_BIN" scripts/sentence_trace_method/prepare_lora_config.py
  --source-config "$SOURCE_CONFIG"
  --output-root "$LORA_ROOT"
  --experiment-name "liar_raw__ministral3_8b__typed_role_rescue_v0_1__${CELL}${LORA_SUFFIX}"
  --swanlab-project fact-checking-typed-role-rescue-lora
  --r 16
  --alpha 32
  --dropout 0.1
  --bias none
  --deepspeed-config "$DEEPSPEED_CONFIG"
  --gradient-accumulation-steps 4
  --learning-rate 2e-5
  --num-train-epochs 12
  --eval-steps 100
  --save-steps 100
  --early-stopping-patience 8
  --early-stopping-metric macro_f1
)
if [[ "$FORCE_CONFIG" == "true" ]]; then
  prepare_args+=(--force)
fi
"${prepare_args[@]}"

"$PYTHON_BIN" - "$CELL" "$LORA_ROOT" "$DEEPSPEED_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import yaml

cell, raw_root, expected_ds = sys.argv[1:]
root = Path(raw_root)
cfg = yaml.safe_load((root / "train.resolved.yaml").read_text(encoding="utf-8"))
train = cfg["sft_train"]
assert cfg["model_name_or_path"] == "/data/models/Ministral-3-8B-Instruct-2512"
assert cfg["label_schema"] == "liar6" and train["label_schema"] == "liar6"
assert train["lora"]["enabled"] is True
assert train["lora"]["r"] == 16 and train["lora"]["alpha"] == 32
assert abs(float(train["lora"]["dropout"]) - 0.1) < 1e-12
assert train["per_device_train_batch_size"] == 1
assert train["per_device_eval_batch_size"] == 1
assert train["gradient_accumulation_steps"] == 4
assert abs(float(train["learning_rate"]) - 2e-5) < 1e-12
assert float(train["num_train_epochs"]) == 12.0
assert train["eval_steps"] == 100 and train["save_steps"] == 100
assert train["early_stopping_patience"] == 8
assert train["max_length"] == 1024
assert train["lr_scheduler_type"] == "cosine_with_restarts"
assert train["lr_scheduler_kwargs"] == {"num_cycles": 2}
assert train["logit_adjust"]["enabled"] is False
label_ce = train["label_token_ce"]
assert label_ce["early_stopping_metric"] == "macro_f1"
assert label_ce["class_weights"] == {
    "pants-fire": 1.2,
    "false": 1.0,
    "barely-true": 1.5,
    "half-true": 1.0,
    "mostly-true": 1.0,
    "true": 1.8,
}
ordinal = label_ce["ordinal_loss"]
assert ordinal["enabled"] is True
assert abs(float(ordinal["alpha"]) - 0.2) < 1e-12
assert ordinal["normalize_distance"] is True
assert cfg["train"]["deepspeed_config"] == expected_ds
for split, expected in (("train", 10065), ("val", 1274), ("test", 1251)):
    path = Path(cfg["data"][f"{split}_candidates"])
    assert path.is_file(), (cell, split, path)
    with path.open(encoding="utf-8") as handle:
        count = sum(1 for line in handle if line.strip())
    assert count == expected, (cell, split, count, expected)
print(f"{cell}: LoRA config/data gate PASS")
PY

TRAIN_MARKER="${LORA_ROOT}/train/training_complete.json"
if [[ ! -f "$TRAIN_MARKER" ]] || ! grep -Eq '"completed"[[:space:]]*:[[:space:]]*true' "$TRAIN_MARKER"; then
  printf '[role-rescue-cell] training cell=%s root=%s nproc=%s\n' "$CELL" "$LORA_ROOT" "$NPROC_PER_NODE"
  env \
    SAVE_LATEST_TRAIN_STATE=true \
    RESUME_LATEST_TRAIN_STATE=true \
    "$ACCELERATE_BIN" launch \
      --num_processes "$NPROC_PER_NODE" \
      --num_machines "$NUM_MACHINES" \
      --main_process_port "$MAIN_PROCESS_PORT" \
      --mixed_precision "$MIXED_PRECISION" \
      --use_deepspeed \
      --deepspeed_config_file "$DEEPSPEED_CONFIG" \
      -m sft.label_token_trainer \
      --config "${LORA_ROOT}/train.resolved.yaml"
else
  printf '[role-rescue-cell] reuse completed training: %s\n' "$TRAIN_MARKER"
fi

if [[ ! -f "$TRAIN_MARKER" ]] || ! grep -Eq '"completed"[[:space:]]*:[[:space:]]*true' "$TRAIN_MARKER"; then
  printf 'Training did not produce a completed marker: %s\n' "$TRAIN_MARKER" >&2
  exit 3
fi

plan_json="$("$PYTHON_BIN" - "$LORA_ROOT" "$CELL" <<'PY'
import json
import sys

root, cell = sys.argv[1:]
print(json.dumps([
    {
        "type": "fixed",
        "experiment": f"typed_role_rescue_{cell}_val",
        "split": "val",
        "logit_adjust": "off",
        "output_dir": f"{root}/eval/val/best/label_token",
    },
    {
        "type": "fixed",
        "experiment": f"typed_role_rescue_{cell}_test",
        "split": "test",
        "logit_adjust": "off",
        "output_dir": f"{root}/eval/test/best/label_token",
    },
]))
PY
)"

eval_args=(
  -m sft.label_token_multi_infer
  --run-dir "${LORA_ROOT}/train"
  --checkpoint best
  --config "${LORA_ROOT}/train.resolved.yaml"
  --plan-json "$plan_json"
  --per-device-eval-batch-size 1
  --dataloader-num-workers 4
)
if [[ "$FORCE_EVAL" == "true" ]]; then
  eval_args+=(--force-eval)
fi

printf '[role-rescue-cell] evaluating cell=%s val,test\n' "$CELL"
"$ACCELERATE_BIN" launch \
  --multi_gpu \
  --num_processes "$NPROC_PER_NODE" \
  --num_machines "$NUM_MACHINES" \
  --main_process_port "$EVAL_PROCESS_PORT" \
  --mixed_precision "$MIXED_PRECISION" \
  "${eval_args[@]}"

for split in val test; do
  metrics="${LORA_ROOT}/eval/${split}/best/label_token/metrics.json"
  [[ -f "$metrics" ]] || { printf 'Missing %s metrics: %s\n' "$split" "$metrics" >&2; exit 4; }
done
printf '[role-rescue-cell] complete cell=%s root=%s\n' "$CELL" "$LORA_ROOT"
