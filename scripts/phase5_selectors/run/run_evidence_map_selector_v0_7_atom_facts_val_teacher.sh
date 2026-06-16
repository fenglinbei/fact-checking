#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

LOAD_ENV="${LOAD_ENV:-true}"
if [[ "${LOAD_ENV}" == "true" || "${LOAD_ENV}" == "1" ]]; then
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
  fi
fi

SPLIT="${SPLIT:-val}"
PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"
RUN_TEACHER="${RUN_TEACHER:-true}"
BUILD_VERIFIER_DATA="${BUILD_VERIFIER_DATA:-false}"
MOCK_EVIDENCE_MAPS="${MOCK_EVIDENCE_MAPS:-false}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
CONCURRENCY="${CONCURRENCY:-128}"
REQUESTS_PER_MINUTE="${REQUESTS_PER_MINUTE:-60000}"
THINKING_TYPE="${THINKING_TYPE:-disabled}"
PROMPT_VERSION="${PROMPT_VERSION:-evidence_map_v0_7_atom_facts}"

export SPLIT
export PYTHON_BIN
export RUN_TEACHER
export BUILD_VERIFIER_DATA
export MOCK_EVIDENCE_MAPS
export TEACHER_MODEL
export CONCURRENCY
export REQUESTS_PER_MINUTE
export THINKING_TYPE
export PROMPT_VERSION

echo "[v0.7-atom-facts-val-teacher] repo        : ${REPO_ROOT}"
echo "[v0.7-atom-facts-val-teacher] split       : ${SPLIT}"
echo "[v0.7-atom-facts-val-teacher] python      : ${PYTHON_BIN}"
echo "[v0.7-atom-facts-val-teacher] model       : ${TEACHER_MODEL}"
echo "[v0.7-atom-facts-val-teacher] concurrency : ${CONCURRENCY}"
echo "[v0.7-atom-facts-val-teacher] rpm         : ${REQUESTS_PER_MINUTE}"
echo "[v0.7-atom-facts-val-teacher] thinking    : ${THINKING_TYPE}"

cd "${REPO_ROOT}"
exec bash "${SCRIPT_DIR}/run_evidence_map_selector_v0_7_atom_facts.sh"
