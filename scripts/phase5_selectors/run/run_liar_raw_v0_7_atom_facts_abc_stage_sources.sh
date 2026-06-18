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
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
CONFIG="${CONFIG:-configs/experiment/v0_7_liar_raw_atom_facts_abc_chunking.yaml}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/selectors/evidence_chain_graph/liar_raw_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10}"
SELECTOR_NAME="${SELECTOR_NAME:-v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10}"
SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-evidence_chain_graph_v0_7}"
SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-budgeted_marginal_v0_7}"
SPLITS="${SPLITS:-train,val,test}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
STAGE_SAMPLE_LIMIT="${STAGE_SAMPLE_LIMIT:-${SAMPLE_LIMIT}}"
FORCE_STAGE="${FORCE_STAGE:-true}"
DRY_RUN="${DRY_RUN:-false}"

STAGE_SPLITS="${STAGE_SPLITS:-${SPLITS}}"
STAGE_SPLITS="${STAGE_SPLITS// /,}"

FP_ARGS=(--config "${CONFIG}")
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  FP_ARGS+=(--sample-limit "${SAMPLE_LIMIT}")
fi
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-$(PYTHONPATH=src "${PYTHON_BIN}" scripts/phase5_selectors/build/print_chunk_mmr_fingerprint.py "${FP_ARGS[@]}")}"

run_cmd() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

stage_force_args=()
if [[ "${FORCE_STAGE}" == "true" || "${FORCE_STAGE}" == "1" ]]; then
  stage_force_args=(--force)
fi

echo "[liar-raw-v0.7-atom-facts-abc-stage] source root : ${SOURCE_ROOT}"
echo "[liar-raw-v0.7-atom-facts-abc-stage] output root : ${OUTPUT_ROOT}"
echo "[liar-raw-v0.7-atom-facts-abc-stage] selector    : ${SELECTOR_NAME}"
echo "[liar-raw-v0.7-atom-facts-abc-stage] splits      : ${STAGE_SPLITS}"
echo "[liar-raw-v0.7-atom-facts-abc-stage] sample_limit: ${STAGE_SAMPLE_LIMIT}"
echo "[liar-raw-v0.7-atom-facts-abc-stage] fingerprint : ${CHUNK_MMR_FINGERPRINT}"

run_cmd "${PYTHON_BIN}" scripts/sentence_trace_method/stage_sources.py \
  --dataset liar_raw \
  --output-root "${OUTPUT_ROOT}" \
  --source-root "${SOURCE_ROOT}" \
  --selector-name "${SELECTOR_NAME}" \
  --graph-version "${SELECTOR_GRAPH_VERSION}" \
  --adaptive-policy "${SELECTOR_ADAPTIVE_POLICY}" \
  --expected-fingerprint "${CHUNK_MMR_FINGERPRINT}" \
  --sample-limit "${STAGE_SAMPLE_LIMIT}" \
  --splits "${STAGE_SPLITS}" \
  --allow-multi-sentence-candidates \
  "${stage_force_args[@]}"

echo "[liar-raw-v0.7-atom-facts-abc-stage] done"
