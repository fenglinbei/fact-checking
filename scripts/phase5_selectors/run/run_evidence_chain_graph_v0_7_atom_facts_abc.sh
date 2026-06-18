#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SPLIT="${SPLIT:-val}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_abc_chunking.yaml}"
INPUT="${INPUT:-outputs/selectors/evidence_map_selector/v0_7_atom_facts_abc_${SPLIT}/candidate_evidence_map_features_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/evidence_chain_graph/v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10_${SPLIT}}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

if [[ -z "${CHUNK_MMR_FINGERPRINT+x}" ]]; then
  FP_ARGS=(--config "${CONFIG}")
  if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
    FP_ARGS+=(--sample-limit "${SAMPLE_LIMIT}")
  fi
  CHUNK_MMR_FINGERPRINT="$(PYTHONPATH=src "${PYTHON_BIN}" scripts/phase5_selectors/build/print_chunk_mmr_fingerprint.py "${FP_ARGS[@]}")"
fi

echo "[evidence-chain-graph-v0.7-atom-facts-abc] split      : ${SPLIT}"
echo "[evidence-chain-graph-v0.7-atom-facts-abc] input      : ${INPUT}"
echo "[evidence-chain-graph-v0.7-atom-facts-abc] output     : ${OUTPUT_DIR}"
echo "[evidence-chain-graph-v0.7-atom-facts-abc] config     : ${CONFIG}"
echo "[evidence-chain-graph-v0.7-atom-facts-abc] fingerprint: ${CHUNK_MMR_FINGERPRINT}"

SPLIT="${SPLIT}" \
PYTHON_BIN="${PYTHON_BIN}" \
INPUT="${INPUT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
SAMPLE_LIMIT="${SAMPLE_LIMIT}" \
bash "${SCRIPT_DIR}/run_evidence_chain_graph_v0_7_atom_facts.sh"
