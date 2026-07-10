#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH=".:src:${PYTHONPATH}"
else
  export PYTHONPATH=".:src"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc-gemma4/bin/python" ]]; then
    PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-gemma4/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

MODE="${MODE:-build}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_INDEX="${FORCE_INDEX:-false}"
FORCE_BUILD="${FORCE_BUILD:-false}"
SPLITS="${SPLITS:-train,val}"
RETRIEVAL_STAGE="${RETRIEVAL_STAGE:-sentences}"
PAGE_QUERY_MODE="${PAGE_QUERY_MODE:-text}"
NUM_WORKERS="${NUM_WORKERS:-1}"

TRAIN_RAW="${TRAIN_RAW:-data/raw/HoVer/train.json}"
VAL_RAW="${VAL_RAW:-data/raw/HoVer/val.json}"
WIKI_ROOT="${WIKI_ROOT:-data/raw/HoVer/wiki}"
INDEX_DB="${INDEX_DB:-outputs/cache/hover/wiki_index/wiki_fts.db}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sentence_trace_method/hover__ministral3_8b__bm25_page_mmr_sentence_minmax9_9}"
MODEL_PATH="${MODEL_PATH:-/data/models/Ministral-3-8B-Instruct-2512}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed/deepspeed_zero2_bsz1_ga4.json}"

PAGE_TOP_K="${PAGE_TOP_K:-100}"
SENTENCE_POOL_K="${SENTENCE_POOL_K:-128}"
TOP_K="${TOP_K:-9}"
MMR_LAMBDA="${MMR_LAMBDA:-0.70}"
PAGE_CACHE_SIZE="${PAGE_CACHE_SIZE:-50000}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
INDEX_LIMIT="${INDEX_LIMIT:-0}"

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

split_requested() {
  local split="$1"
  [[ ",${SPLITS}," == *",${split},"* ]]
}

artifact_exists() {
  local path="$1"
  [[ -s "$path" ]]
}

artifacts_ready() {
  artifact_exists "${OUTPUT_DIR}/build_report.json" || return 1
  IFS=',' read -r -a requested_splits <<< "$SPLITS"
  local split
  for split in "${requested_splits[@]}"; do
    split="${split//[[:space:]]/}"
    [[ -n "$split" ]] || continue
    if [[ "$RETRIEVAL_STAGE" == "pages" ]]; then
      artifact_exists "${OUTPUT_DIR}/retrieval/page_retrieval_${split}.jsonl" || return 1
    else
      artifact_exists "${OUTPUT_DIR}/retrieval/retrieval_${split}.jsonl" || return 1
      artifact_exists "${OUTPUT_DIR}/build/build_${split}.jsonl" || return 1
    fi
  done
  if [[ "$RETRIEVAL_STAGE" == "sentences" ]] && split_requested train && split_requested val; then
    artifact_exists "${OUTPUT_DIR}/train.resolved.yaml" || return 1
  fi
}

build_data() {
  if [[ "$DRY_RUN" != "true" && "$FORCE_BUILD" != "true" ]] && artifacts_ready; then
    printf '[hover-s3] reuse build artifacts: %s\n' "$OUTPUT_DIR"
    return 0
  fi

  if split_requested train; then
    require_path "$TRAIN_RAW" "HoVer train split"
  fi
  if split_requested val; then
    require_path "$VAL_RAW" "HoVer val split"
  fi
  require_path "$WIKI_ROOT" "HoVer Wikipedia corpus"

  local cmd=(
    "$PYTHON_BIN" scripts/phase12_hover/build_hover_s3_retrieval_baseline.py
    --train-raw "$TRAIN_RAW"
    --val-raw "$VAL_RAW"
    --wiki-root "$WIKI_ROOT"
    --index-db "$INDEX_DB"
    --output-dir "$OUTPUT_DIR"
    --model-name-or-path "$MODEL_PATH"
    --deepspeed-config "$DEEPSPEED_CONFIG"
    --page-top-k "$PAGE_TOP_K"
    --sentence-pool-k "$SENTENCE_POOL_K"
    --top-k "$TOP_K"
    --mmr-lambda "$MMR_LAMBDA"
    --splits "$SPLITS"
    --retrieval-stage "$RETRIEVAL_STAGE"
    --page-query-mode "$PAGE_QUERY_MODE"
    --page-cache-size "$PAGE_CACHE_SIZE"
    --num-workers "$NUM_WORKERS"
  )
  if [[ "$FORCE_INDEX" == "true" ]]; then
    cmd+=(--force-index)
  fi
  if [[ "$SAMPLE_LIMIT" != "0" ]]; then
    cmd+=(--sample-limit "$SAMPLE_LIMIT")
  fi
  if [[ "$INDEX_LIMIT" != "0" ]]; then
    cmd+=(--index-limit "$INDEX_LIMIT")
  fi
  run_cmd "${cmd[@]}"
}

check_ready() {
  require_path "${OUTPUT_DIR}/build_report.json" "HoVer S3 build report"
  IFS=',' read -r -a requested_splits <<< "$SPLITS"
  local split
  for split in "${requested_splits[@]}"; do
    split="${split//[[:space:]]/}"
    [[ -n "$split" ]] || continue
    if [[ "$RETRIEVAL_STAGE" == "pages" ]]; then
      require_path "${OUTPUT_DIR}/retrieval/page_retrieval_${split}.jsonl" "HoVer S3 page retrieval ${split}"
    else
      require_path "${OUTPUT_DIR}/retrieval/retrieval_${split}.jsonl" "HoVer S3 retrieval ${split}"
      require_path "${OUTPUT_DIR}/build/build_${split}.jsonl" "HoVer S3 build ${split}"
    fi
  done
  if [[ "$RETRIEVAL_STAGE" == "sentences" ]] && split_requested train && split_requested val; then
    require_path "${OUTPUT_DIR}/train.resolved.yaml" "HoVer S3 train config"
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  "$PYTHON_BIN" -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); report=json.load(open(root/"build_report.json")); assert report["status"]=="completed"; print("[hover-s3] checked", root)' "$OUTPUT_DIR"
}

printf '[hover-s3] MODE=%s DRY_RUN=%s STAGE=%s SPLITS=%s PAGE_QUERY_MODE=%s WORKERS=%s OUTPUT_DIR=%s WIKI_ROOT=%s INDEX_DB=%s PAGE_TOP_K=%s TOP_K=%s CACHE=%s\n' \
  "$MODE" "$DRY_RUN" "$RETRIEVAL_STAGE" "$SPLITS" "$PAGE_QUERY_MODE" "$NUM_WORKERS" "$OUTPUT_DIR" "$WIKI_ROOT" "$INDEX_DB" "$PAGE_TOP_K" "$TOP_K" "$PAGE_CACHE_SIZE"

case "$MODE" in
  build|full)
    build_data
    check_ready
    ;;
  check)
    check_ready
    ;;
  *)
    printf 'Unsupported MODE=%s. Use build, check, or full.\n' "$MODE" >&2
    exit 2
    ;;
esac
