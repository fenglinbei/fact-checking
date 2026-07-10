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
PROCESSED_ROOT="${PROCESSED_ROOT:-data/processed/SciFact}"
ATOM_ANCHOR_ROOT="${ATOM_ANCHOR_ROOT:-outputs/selectors/scifact_atom_anchor}"
SCIFACT_URL="${SCIFACT_URL:-https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-false}"
FORCE_INDEX="${FORCE_INDEX:-false}"

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

args=(
  scripts/phase13_scifact/download_and_index_scifact.py
  --data-root "$DATA_ROOT"
  --processed-root "$PROCESSED_ROOT"
  --output-manifest "${ATOM_ANCHOR_ROOT}/00_data/download_manifest.json"
  --url "$SCIFACT_URL"
)
if [[ "$FORCE_DOWNLOAD" == "true" ]]; then
  args+=(--force-download)
fi
if [[ "$FORCE_INDEX" == "true" ]]; then
  args+=(--force-index)
fi

printf '[scifact-01] DATA_ROOT=%s PROCESSED_ROOT=%s ATOM_ANCHOR_ROOT=%s URL=%s\n' \
  "$DATA_ROOT" "$PROCESSED_ROOT" "$ATOM_ANCHOR_ROOT" "$SCIFACT_URL"
run_cmd "$PYTHON_BIN" "${args[@]}"
