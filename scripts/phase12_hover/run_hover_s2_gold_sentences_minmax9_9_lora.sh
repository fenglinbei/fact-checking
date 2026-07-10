#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python" ]]; then
    PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python"
  elif [[ -x "/data/liaozijie/conda/accelerate-fc-gemma4/bin/python" ]]; then
    PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-gemma4/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ -z "${ACCELERATE_BIN:-}" ]]; then
  if [[ "$PYTHON_BIN" == */python ]]; then
    ACCELERATE_BIN="$(dirname "$PYTHON_BIN")/accelerate"
  else
    ACCELERATE_BIN="accelerate"
  fi
fi

MODE="${MODE:-full}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_BUILD="${FORCE_BUILD:-false}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"

TRAIN_RAW="${TRAIN_RAW:-data/raw/HoVer/train.json}"
VAL_RAW="${VAL_RAW:-data/raw/HoVer/val.json}"
WIKI_ROOT="${WIKI_ROOT:-data/raw/HoVer/wiki}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sentence_trace_method/hover__ministral3_8b__gold_sentences_minmax9_9}"
MODEL_PATH="${MODEL_PATH:-/data/models/Ministral-3-8B-Instruct-2512}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/deepspeed_zero2_bsz1_ga4.json}"

EVIDENCE_MODE="${EVIDENCE_MODE:-gold_sentences}"
SENTENCE_WINDOW="${SENTENCE_WINDOW:-0}"
MAX_DOC_SENTENCES="${MAX_DOC_SENTENCES:-20}"
MISSING_POLICY="${MISSING_POLICY:-error}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE:-1}"
RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE:-1}"

HOVER_RUNTIME_CACHE_ROOT="${HOVER_RUNTIME_CACHE_ROOT:-${ROOT_DIR}/outputs/cache/runtime/hover_s2_gold_sentences}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOVER_RUNTIME_CACHE_ROOT}/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOVER_RUNTIME_CACHE_ROOT}/vllm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${HOVER_RUNTIME_CACHE_ROOT}/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${HOVER_RUNTIME_CACHE_ROOT}/triton}"
if [[ "$DRY_RUN" != "true" ]]; then
  mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
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

require_path() {
  local path="$1"
  local label="$2"
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  if [[ ! -e "$path" ]]; then
    printf 'Missing %s: %s\n' "$label" "$path" >&2
    exit 2
  fi
}

training_complete() {
  [[ -f "${OUTPUT_DIR}/train/training_complete.json" ]]
}

build_data() {
  if [[ "$DRY_RUN" != "true" && "$FORCE_BUILD" != "true" \
      && -s "${OUTPUT_DIR}/build/build_train.jsonl" \
      && -s "${OUTPUT_DIR}/build/build_val.jsonl" \
      && -s "${OUTPUT_DIR}/train.resolved.yaml" \
      && -s "${OUTPUT_DIR}/build_report.json" ]]; then
    printf '[hover-s2] reuse build artifacts: %s\n' "$OUTPUT_DIR"
    return 0
  fi

  require_path "$TRAIN_RAW" "HoVer train split"
  require_path "$VAL_RAW" "HoVer val split"
  require_path "$WIKI_ROOT" "HoVer HotpotQA processed Wikipedia corpus"

  local cmd=(
    "$PYTHON_BIN" scripts/phase12_hover/build_hover_gold_sentence_verifier_data.py
    --train-raw "$TRAIN_RAW"
    --val-raw "$VAL_RAW"
    --wiki-root "$WIKI_ROOT"
    --output-dir "$OUTPUT_DIR"
    --model-name-or-path "$MODEL_PATH"
    --deepspeed-config "$DEEPSPEED_CONFIG"
    --evidence-mode "$EVIDENCE_MODE"
    --sentence-window "$SENTENCE_WINDOW"
    --max-doc-sentences "$MAX_DOC_SENTENCES"
    --missing-policy "$MISSING_POLICY"
  )
  if [[ "$SAMPLE_LIMIT" != "0" ]]; then
    cmd+=(--sample-limit "$SAMPLE_LIMIT")
  fi
  run_cmd "${cmd[@]}"
}

check_ready() {
  require_path "${OUTPUT_DIR}/train.resolved.yaml" "HoVer S2 train config"
  require_path "${OUTPUT_DIR}/build/build_train.jsonl" "HoVer S2 build_train.jsonl"
  require_path "${OUTPUT_DIR}/build/build_val.jsonl" "HoVer S2 build_val.jsonl"
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  "$PYTHON_BIN" - <<'PY' "$OUTPUT_DIR"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for split in ("train", "val"):
    path = root / "build" / f"build_{split}.jsonl"
    with path.open("r", encoding="utf-8") as f:
        row = json.loads(next(f))
    if row.get("label_schema") != "hover2":
        raise SystemExit(f"{path}: label_schema={row.get('label_schema')!r}")
    if not row.get("prompt"):
        raise SystemExit(f"{path}: missing prompt")
    if not row.get("target"):
        raise SystemExit(f"{path}: missing target")
    if int(row.get("evidence_count") or 0) <= 0:
        raise SystemExit(f"{path}: empty evidence")
print(f"[hover-s2] checked build rows under {root}")
PY
}

train_model() {
  require_path "${OUTPUT_DIR}/train.resolved.yaml" "HoVer S2 train config"
  if training_complete && [[ "$FORCE_TRAIN" != "true" ]]; then
    printf '[hover-s2] training already complete: %s/train; set FORCE_TRAIN=true to rerun.\n' "$OUTPUT_DIR"
    return 0
  fi
  run_cmd env \
    SAVE_LATEST_TRAIN_STATE="$SAVE_LATEST_TRAIN_STATE" \
    RESUME_LATEST_TRAIN_STATE="$RESUME_LATEST_TRAIN_STATE" \
    "$ACCELERATE_BIN" launch \
    --num_processes "$NPROC_PER_NODE" \
    --num_machines "$NUM_MACHINES" \
    --mixed_precision "$MIXED_PRECISION" \
    --use_deepspeed \
    --deepspeed_config_file "$DEEPSPEED_CONFIG" \
    -m sft.label_token_trainer \
    --config "${OUTPUT_DIR}/train.resolved.yaml"
}

printf '[hover-s2] MODE=%s DRY_RUN=%s OUTPUT_DIR=%s WIKI_ROOT=%s EVIDENCE_MODE=%s SENTENCE_WINDOW=%s MODEL_PATH=%s\n' \
  "$MODE" "$DRY_RUN" "$OUTPUT_DIR" "$WIKI_ROOT" "$EVIDENCE_MODE" "$SENTENCE_WINDOW" "$MODEL_PATH"

case "$MODE" in
  build)
    build_data
    check_ready
    ;;
  check)
    check_ready
    ;;
  train)
    check_ready
    train_model
    ;;
  full)
    build_data
    check_ready
    train_model
    ;;
  *)
    printf 'Unsupported MODE=%s. Use build, check, train, or full.\n' "$MODE" >&2
    exit 2
    ;;
esac
