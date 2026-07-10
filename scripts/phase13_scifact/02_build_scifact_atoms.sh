#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python" ]]; then
    export PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python"
  else
    export PYTHON_BIN="python"
  fi
fi
DRY_RUN="${DRY_RUN:-false}"
DATA_ROOT="${DATA_ROOT:-data/raw/SciFact}"
ATOM_ANCHOR_ROOT="${ATOM_ANCHOR_ROOT:-outputs/selectors/scifact_atom_anchor}"
CLAIM_ATOM_ROOT="${CLAIM_ATOM_ROOT:-${ATOM_ANCHOR_ROOT}/01_claim_atoms}"
SPLITS="${SPLITS:-train val test}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
FORCE_CLAIM_ATOMS="${FORCE_CLAIM_ATOMS:-false}"
MOCK_ATOMS="${MOCK_ATOMS:-false}"
NO_PROGRESS="${NO_PROGRESS:-false}"

ATOM_BASE_URL="${ATOM_BASE_URL:-https://api.deepseek.com}"
ATOM_MODEL="${ATOM_MODEL:-deepseek-v4-flash}"
ATOM_API_KEY_ENV="${ATOM_API_KEY_ENV:-DEEPSEEK_API_KEY}"
ATOM_API_CONCURRENCY="${ATOM_API_CONCURRENCY:-128}"
ATOM_MAX_TOKENS="${ATOM_MAX_TOKENS:-2048}"
ATOM_THINKING_TYPE="${ATOM_THINKING_TYPE:-disabled}"

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

raw_path_for_split() {
  case "$1" in
    train) printf '%s\n' "${DATA_ROOT}/claims_train.jsonl" ;;
    val) printf '%s\n' "${DATA_ROOT}/claims_dev.jsonl" ;;
    test) printf '%s\n' "${DATA_ROOT}/claims_test.jsonl" ;;
    *) printf 'Unsupported split=%s\n' "$1" >&2; exit 2 ;;
  esac
}

progress_args=()
if [[ "$NO_PROGRESS" == "true" || "$NO_PROGRESS" == "1" || "$NO_PROGRESS" == "True" ]]; then
  progress_args=(--no-progress)
fi

printf '[scifact-02] ATOM_ANCHOR_ROOT=%s CLAIM_ATOM_ROOT=%s SPLITS=%s MODEL=%s MOCK_ATOMS=%s NO_PROGRESS=%s\n' \
  "$ATOM_ANCHOR_ROOT" "$CLAIM_ATOM_ROOT" "$SPLITS" "$ATOM_MODEL" "$MOCK_ATOMS" "$NO_PROGRESS"

for split in $SPLITS; do
  raw_path="$(raw_path_for_split "$split")"
  claim_atoms="${CLAIM_ATOM_ROOT}/claim_atoms_${split}.jsonl"
  require_path "$raw_path" "${split} SciFact raw split"
  if [[ -s "$claim_atoms" && "$FORCE_CLAIM_ATOMS" != "true" ]]; then
    printf '[scifact-02] reuse claim atoms: %s\n' "$claim_atoms"
    continue
  fi
  sample_args=()
  if [[ "$SAMPLE_LIMIT" != "0" ]]; then
    sample_args=(--sample-limit "$SAMPLE_LIMIT")
  fi
  mock_args=()
  if [[ "$MOCK_ATOMS" == "true" ]]; then
    mock_args=(--mock-atoms)
  fi
  run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/generate_claim_atom_cache.py \
    --input-mode raw_split \
    --dataset scifact \
    --label-schema scifact3 \
    --raw-path "$raw_path" \
    --output-dir "$CLAIM_ATOM_ROOT" \
    --split "$split" \
    --atom-cache-dir "${CLAIM_ATOM_ROOT}/cache" \
    --atom-base-url "$ATOM_BASE_URL" \
    --atom-model "$ATOM_MODEL" \
    --atom-api-key-env "$ATOM_API_KEY_ENV" \
    --api-concurrency "$ATOM_API_CONCURRENCY" \
    --max-tokens "$ATOM_MAX_TOKENS" \
    --thinking-type "$ATOM_THINKING_TYPE" \
    "${progress_args[@]}" \
    "${mock_args[@]}" \
    "${sample_args[@]}"
done
