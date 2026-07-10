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
FORCE_BUILD="${FORCE_BUILD:-false}"

TRAIN_RAW="${TRAIN_RAW:-data/raw/HoVer/train.json}"
VAL_RAW="${VAL_RAW:-data/raw/HoVer/val.json}"
S3_OUTPUT_DIR="${S3_OUTPUT_DIR:-outputs/sentence_trace_method/hover__ministral3_8b__bm25_page_mmr_sentence_minmax9_9}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/atom_anchor/hover_abc_v0_1}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

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

build_sources() {
  if [[ "$DRY_RUN" != "true" && "$FORCE_BUILD" != "true" \
      && -s "${OUTPUT_DIR}/manifest.json" \
      && -s "${OUTPUT_DIR}/04_evidence_map/train.jsonl" \
      && -s "${OUTPUT_DIR}/04_evidence_map/val.jsonl" ]]; then
    printf '[hover-s4-prep] reuse source artifacts: %s\n' "$OUTPUT_DIR"
    return 0
  fi
  require_path "$TRAIN_RAW" "HoVer train split"
  require_path "$VAL_RAW" "HoVer val split"
  require_path "${S3_OUTPUT_DIR}/retrieval/retrieval_train.jsonl" "HoVer S3 train retrieval"
  require_path "${S3_OUTPUT_DIR}/retrieval/retrieval_val.jsonl" "HoVer S3 val retrieval"

  local cmd=(
    "$PYTHON_BIN" scripts/phase12_hover/prepare_hover_s4_mrec_sources.py
    --train-raw "$TRAIN_RAW"
    --val-raw "$VAL_RAW"
    --s3-output-dir "$S3_OUTPUT_DIR"
    --output-dir "$OUTPUT_DIR"
  )
  if [[ "$SAMPLE_LIMIT" != "0" ]]; then
    cmd+=(--sample-limit "$SAMPLE_LIMIT")
  fi
  run_cmd "${cmd[@]}"
}

check_ready() {
  require_path "${OUTPUT_DIR}/manifest.json" "HoVer S4 prep manifest"
  require_path "${OUTPUT_DIR}/01_claim_atoms/train.jsonl" "HoVer S4 claim atoms train"
  require_path "${OUTPUT_DIR}/02_candidate_pool/train.jsonl" "HoVer S4 candidate pool train"
  require_path "${OUTPUT_DIR}/04_evidence_map/train.jsonl" "HoVer S4 evidence map train"
  require_path "${OUTPUT_DIR}/05_mrec_v0_2_learned_marginal_proxy_fullpool/train_proxy_pairs.jsonl" "HoVer S4 proxy pairs train"
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  "$PYTHON_BIN" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"manifest.json"; m=json.load(open(p)); assert m["status"]=="completed"; print("[hover-s4-prep] checked", p.parent)' "$OUTPUT_DIR"
}

printf '[hover-s4-prep] MODE=%s DRY_RUN=%s OUTPUT_DIR=%s S3_OUTPUT_DIR=%s\n' \
  "$MODE" "$DRY_RUN" "$OUTPUT_DIR" "$S3_OUTPUT_DIR"

case "$MODE" in
  build|full)
    build_sources
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
