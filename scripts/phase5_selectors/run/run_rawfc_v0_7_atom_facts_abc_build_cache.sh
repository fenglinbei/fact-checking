#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

LOAD_ENV="${LOAD_ENV:-true}"
if [[ "${LOAD_ENV}" == "true" || "${LOAD_ENV}" == "1" ]]; then
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"
EXPERIMENT="${EXPERIMENT:-v0_6c_rawfc3_rule_step_adaptive5_10_abc_chunking}"
CONFIG="${CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_abc_chunking.yaml}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
FORCE_BUILD="${FORCE_BUILD:-false}"

FP_ARGS=(--config "${CONFIG}")
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  FP_ARGS+=(--sample-limit "${SAMPLE_LIMIT}")
fi
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-$(PYTHONPATH=src "${PYTHON_BIN}" scripts/phase5_selectors/build/print_chunk_mmr_fingerprint.py "${FP_ARGS[@]}")}"

echo "[rawfc-v0.7-atom-facts-abc-cache] experiment  : ${EXPERIMENT}"
echo "[rawfc-v0.7-atom-facts-abc-cache] config      : ${CONFIG}"
echo "[rawfc-v0.7-atom-facts-abc-cache] sample_limit: ${SAMPLE_LIMIT}"
echo "[rawfc-v0.7-atom-facts-abc-cache] force_build : ${FORCE_BUILD}"
echo "[rawfc-v0.7-atom-facts-abc-cache] fingerprint : ${CHUNK_MMR_FINGERPRINT}"

PYTHONPATH=src "${PYTHON_BIN}" -m fact_checking.pipeline.run \
  "experiment=${EXPERIMENT}" \
  "pipeline.mode=build" \
  "pipeline.force.build=${FORCE_BUILD}" \
  "build.data.sample_limit=${SAMPLE_LIMIT}"

echo "[rawfc-v0.7-atom-facts-abc-cache] train cache: outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/train.pkl"
echo "[rawfc-v0.7-atom-facts-abc-cache] val cache  : outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/val.pkl"
echo "[rawfc-v0.7-atom-facts-abc-cache] test cache : outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/test.pkl"
