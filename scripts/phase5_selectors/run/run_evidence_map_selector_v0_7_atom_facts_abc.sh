#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SPLIT="${SPLIT:-val}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QD_UNION_POOL_FILE="${QD_UNION_POOL_FILE:-outputs/selectors/question_decomp_retrieval/rawfc_deepseek_v0_abc_${SPLIT}/union_candidate_pool_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/evidence_map_selector/v0_7_atom_facts_abc_${SPLIT}}"
PROMPT_VERSION="${PROMPT_VERSION:-evidence_map_v0_7_atom_facts_abc}"
CONFIG="${CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_abc_chunking.yaml}"

if [[ "${SPLIT}" == "train" ]]; then
  RAW_PATH="${RAW_PATH:-data/raw/RAWFC/train.json}"
elif [[ "${SPLIT}" == "test" ]]; then
  RAW_PATH="${RAW_PATH:-data/raw/RAWFC/test.json}"
else
  RAW_PATH="${RAW_PATH:-data/raw/RAWFC/val.json}"
fi
ORACLE_RESULTS="${ORACLE_RESULTS:-}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
CHILD_SAMPLE_LIMIT=""
if [[ -n "${SAMPLE_LIMIT}" && "${SAMPLE_LIMIT}" != "0" ]]; then
  CHILD_SAMPLE_LIMIT="${SAMPLE_LIMIT}"
fi

echo "[evidence-map-v0.7-atom-facts-abc] split    : ${SPLIT}"
echo "[evidence-map-v0.7-atom-facts-abc] qd union : ${QD_UNION_POOL_FILE}"
echo "[evidence-map-v0.7-atom-facts-abc] output   : ${OUTPUT_DIR}"
echo "[evidence-map-v0.7-atom-facts-abc] prompt   : ${PROMPT_VERSION}"

SPLIT="${SPLIT}" \
PYTHON_BIN="${PYTHON_BIN}" \
QD_UNION_POOL_FILE="${QD_UNION_POOL_FILE}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
PROMPT_VERSION="${PROMPT_VERSION}" \
CONFIG="${CONFIG}" \
RAW_PATH="${RAW_PATH}" \
ORACLE_RESULTS="${ORACLE_RESULTS}" \
SAMPLE_LIMIT="${CHILD_SAMPLE_LIMIT}" \
bash "${SCRIPT_DIR}/run_evidence_map_selector_v0_7_atom_facts.sh"
