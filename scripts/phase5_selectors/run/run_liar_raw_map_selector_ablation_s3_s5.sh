#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLITS="${SPLITS:-train val test}"
SELECTORS="${SELECTORS:-map_selector_s3_weighted_set_cover_top5 map_selector_s4_minimal_evidence_group_top5 map_selector_s5_fixed_budget_marginal_greedy_top5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selectors/evidence_chain_graph}"
TOP_K="${TOP_K:-5}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-d4cbf7c18126}"

run_cmd() {
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

for split in ${SPLITS}; do
  input="outputs/selectors/evidence_map_selector/liar_raw_v0_7_atom_facts_abc_${split}/candidate_evidence_map_features_${split}.jsonl"
  for selector in ${SELECTORS}; do
    if [[ "${OUTPUT_ROOT}" == "outputs/selectors/evidence_chain_graph" ]]; then
      output_dir="outputs/selectors/evidence_chain_graph/liar_raw_${selector}_${split}"
    else
      output_dir="${OUTPUT_ROOT}/liar_raw_${selector}_${split}"
    fi
    echo "[liar-raw-map-selector-ablation-s3-s5] split=${split} selector=${selector}"
    echo "[liar-raw-map-selector-ablation-s3-s5] input=${input}"
    echo "[liar-raw-map-selector-ablation-s3-s5] output=${output_dir}"
    run_cmd "${PYTHON_BIN}" scripts/phase5_selectors/build/build_map_selector_ablation_traces.py \
      --input "${input}" \
      --output-dir "${output_dir}" \
      --split "${split}" \
      --sample-limit "${SAMPLE_LIMIT}" \
      --selector-name "${selector}" \
      --top-k "${TOP_K}" \
      --candidate-top-n "${CANDIDATE_TOP_N}" \
      --chunk-mmr-fingerprint "${CHUNK_MMR_FINGERPRINT}"
  done
done
