#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

load_env_file() {
  local env_file="${1:-.env}"
  [[ -f "${env_file}" ]] || return 0
  local line key value
  while IFS= read -r line || [[ -n "${line}" ]]; do
    if [[ "${line}" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      if [[ -n "${!key+x}" ]]; then
        continue
      fi
      value="${BASH_REMATCH[2]}"
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
        value="${value:1:${#value}-2}"
      fi
      export "${key}=${value}"
    fi
  done < "${env_file}"
}

load_env_file "${ENV_FILE:-.env}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
BASE_PATH="${BASE_PATH:-/evidence-map}"
SPLITS="${SPLITS:-val}"
MAX_CANDIDATES="${MAX_CANDIDATES:-20}"
ENABLE_LIVE_TRANSLATION="${ENABLE_LIVE_TRANSLATION:-1}"

if [[ -z "${EVIDENCE_MAP_TOKEN:-}" ]]; then
  echo "ERROR: EVIDENCE_MAP_TOKEN must be set." >&2
  exit 2
fi

args=(
  --host "${HOST}"
  --port "${PORT}"
  --base-path "${BASE_PATH}"
  --splits "${SPLITS}"
  --max-candidates "${MAX_CANDIDATES}"
  --token "${EVIDENCE_MAP_TOKEN}"
)

if [[ "${ENABLE_LIVE_TRANSLATION}" == "1" || "${ENABLE_LIVE_TRANSLATION}" == "true" ]]; then
  args+=(--enable-live-translation)
fi

PYTHONPATH=.:src python scripts/phase5_selectors/visualize/serve_evidence_map_selector_comparison.py "${args[@]}"
