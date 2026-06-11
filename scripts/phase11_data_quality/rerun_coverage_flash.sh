#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="${DATASETS:-liar_raw rawfc}"
SPLITS="${SPLITS:-train val test}"

COVERAGE_VERSION="${COVERAGE_VERSION:-source_coverage_v2_flash}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/data_quality/source_coverage_flash}"
PROCESSED_ROOT="${PROCESSED_ROOT:-data/processed/coverage/${COVERAGE_VERSION}}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-/data/models/bge-base-en-v1.5}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
EMBEDDING_PRECISION="${EMBEDDING_PRECISION:-bf16}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"
EMBEDDING_MAX_LENGTH="${EMBEDDING_MAX_LENGTH:-256}"

LLM_BASE_URL="${LLM_BASE_URL:-https://api.deepseek.com}"
LLM_MODEL="${LLM_MODEL:-deepseek-v4-flash}"
LLM_TIMEOUT="${LLM_TIMEOUT:-120}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-512}"
LLM_MIN_CONFIDENCE="${LLM_MIN_CONFIDENCE:-0.65}"
LLM_WORKERS="${LLM_WORKERS:-4}"
LLM_RETRIES="${LLM_RETRIES:-3}"
LLM_RETRY_BACKOFF="${LLM_RETRY_BACKOFF:-2.0}"
LLM_THINKING="${LLM_THINKING:-disabled}"

TOP_K="${TOP_K:-12}"
SENTENCE_SOURCE="${SENTENCE_SOURCE:-content}"
MATERIALIZE="${MATERIALIZE:-1}"
COMPARE_ORIGINAL="${COMPARE_ORIGINAL:-1}"
ALLOW_LLM_ERRORS="${ALLOW_LLM_ERRORS:-0}"
NO_PROGRESS="${NO_PROGRESS:-0}"
RESUME="${RESUME:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
KEEP_LLM_ERRORS="${KEEP_LLM_ERRORS:-0}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]] && ! grep -q '^DEEPSEEK_API_KEY=' .env 2>/dev/null; then
  echo "ERROR: DEEPSEEK_API_KEY is not set and was not found in .env" >&2
  exit 1
fi

if [[ ! -d "$EMBEDDING_MODEL" ]]; then
  echo "ERROR: embedding model directory not found: $EMBEDDING_MODEL" >&2
  exit 1
fi

TAG_COMMON_ARGS=(
  --coverage-version "$COVERAGE_VERSION"
  --embedding-model "$EMBEDDING_MODEL"
  --embedding-device "$EMBEDDING_DEVICE"
  --embedding-precision "$EMBEDDING_PRECISION"
  --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
  --embedding-max-length "$EMBEDDING_MAX_LENGTH"
  --llm-base-url "$LLM_BASE_URL"
  --llm-model "$LLM_MODEL"
  --llm-timeout "$LLM_TIMEOUT"
  --llm-max-tokens "$LLM_MAX_TOKENS"
  --llm-min-confidence "$LLM_MIN_CONFIDENCE"
  --llm-workers "$LLM_WORKERS"
  --llm-retries "$LLM_RETRIES"
  --llm-retry-backoff "$LLM_RETRY_BACKOFF"
  --llm-thinking "$LLM_THINKING"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --top-k "$TOP_K"
  --sentence-source "$SENTENCE_SOURCE"
)

if [[ "$NO_PROGRESS" == "1" ]]; then
  TAG_COMMON_ARGS+=(--no-progress)
fi
if [[ "$RESUME" == "0" ]]; then
  TAG_COMMON_ARGS+=(--no-resume)
fi
if [[ "$KEEP_LLM_ERRORS" == "1" ]]; then
  TAG_COMMON_ARGS+=(--keep-llm-errors)
fi

run_one_split() {
  local dataset="$1"
  local split="$2"
  local output_dir="$OUTPUT_BASE/$dataset"
  local args=(--dataset "$dataset" --split "$split" --output-dir "$output_dir")

  if [[ "$dataset" == "rawfc" ]]; then
    args+=(--label-schema rawfc3)
  fi

  echo "==> tagging dataset=$dataset split=$split model=$LLM_MODEL output=$output_dir"
  PYTHONPATH=src "$PYTHON_BIN" scripts/phase11_data_quality/tag_source_coverage.py \
    "${args[@]}" \
    "${TAG_COMMON_ARGS[@]}"
}

for dataset in $DATASETS; do
  case "$dataset" in
    liar_raw|rawfc) ;;
    *)
      echo "ERROR: unsupported dataset: $dataset" >&2
      exit 1
      ;;
  esac
  for split in $SPLITS; do
    case "$split" in
      train|val|test) ;;
      *)
        echo "ERROR: unsupported split: $split" >&2
        exit 1
        ;;
    esac
    run_one_split "$dataset" "$split"
  done
done

echo "==> checking LLM review status"
"$PYTHON_BIN" - "$OUTPUT_BASE" "$ALLOW_LLM_ERRORS" "$DATASETS" "$SPLITS" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
allow_errors = sys.argv[2] == "1"
datasets = sys.argv[3].split()
splits = sys.argv[4].split()
bad = []

for dataset in datasets:
    for split in splits:
        path = root / dataset / f"source_coverage_{split}.jsonl"
        if not path.exists():
            bad.append((path.as_posix(), "", "missing_file", "coverage sidecar was not created"))
            continue
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        review_needed = 0
        status_counts = Counter()
        model_counts = Counter()
        for row in rows:
            judgment = row.get("llm_judgment") if isinstance(row.get("llm_judgment"), dict) else {}
            if judgment.get("review_needed"):
                review_needed += 1
                status = str(judgment.get("status") or "missing")
                status_counts[status] += 1
                model = judgment.get("model")
                if model:
                    model_counts[str(model)] += 1
                if status != "ok":
                    bad.append((path.as_posix(), row.get("event_id"), status, str(judgment.get("error") or "")[:200]))
        print(
            f"{path}: rows={len(rows)} review_needed={review_needed} "
            f"status={dict(status_counts)} models={dict(model_counts)}"
        )

if bad and not allow_errors:
    print("\nERROR: LLM review had non-ok statuses. First examples:", file=sys.stderr)
    for item in bad[:20]:
        print(item, file=sys.stderr)
    raise SystemExit(1)
PY

if [[ "$MATERIALIZE" == "1" ]]; then
  for dataset in $DATASETS; do
    echo "==> materializing dataset=$dataset coverage_dir=$OUTPUT_BASE/$dataset output_root=$PROCESSED_ROOT"
    "$PYTHON_BIN" scripts/phase11_data_quality/materialize_coverage_datasets.py \
      --dataset "$dataset" \
      --coverage-version "$COVERAGE_VERSION" \
      --coverage-dir "$OUTPUT_BASE/$dataset" \
      --output-root "$PROCESSED_ROOT" \
      --splits $SPLITS
  done
fi

if [[ "$COMPARE_ORIGINAL" == "1" ]]; then
  for dataset in $DATASETS; do
    echo "==> comparing original vs coverage dataset=$dataset coverage_dir=$OUTPUT_BASE/$dataset"
    "$PYTHON_BIN" scripts/phase11_data_quality/compare_coverage_to_original.py \
      --dataset "$dataset" \
      --coverage-version "$COVERAGE_VERSION" \
      --coverage-dir "$OUTPUT_BASE/$dataset" \
      --processed-root "$PROCESSED_ROOT" \
      --output-dir "$OUTPUT_BASE/$dataset/original_diff" \
      --splits $SPLITS
  done
fi

echo "Done."
echo "Coverage sidecars: $OUTPUT_BASE/{liar_raw,rawfc}/"
if [[ "$MATERIALIZE" == "1" ]]; then
  echo "Materialized datasets: $PROCESSED_ROOT/{liar_raw,rawfc}/{all,covered,covered_weak}/"
fi
if [[ "$COMPARE_ORIGINAL" == "1" ]]; then
  echo "Original diff reports: $OUTPUT_BASE/{liar_raw,rawfc}/original_diff/"
fi
