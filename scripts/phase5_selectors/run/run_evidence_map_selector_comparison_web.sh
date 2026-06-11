#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
BASE_PATH="${BASE_PATH:-/evidence-map}"
SPLITS="${SPLITS:-val}"
MAX_CANDIDATES="${MAX_CANDIDATES:-20}"

if [[ -z "${EVIDENCE_MAP_TOKEN:-}" ]]; then
  echo "ERROR: EVIDENCE_MAP_TOKEN must be set." >&2
  exit 2
fi

PYTHONPATH=.:src python scripts/phase5_selectors/visualize/serve_evidence_map_selector_comparison.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --base-path "${BASE_PATH}" \
  --splits "${SPLITS}" \
  --max-candidates "${MAX_CANDIDATES}" \
  --token "${EVIDENCE_MAP_TOKEN}"
