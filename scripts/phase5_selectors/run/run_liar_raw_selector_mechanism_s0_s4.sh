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
SELECTORS="${SELECTORS:-selector_mech_s0_no_evidence selector_mech_s1_claim_pool_random_top5 selector_mech_s2_claim_pool_hybrid_top5 selector_mech_s3_claim_pool_hybrid_mmr_top5 selector_mech_s4_atom_union_source_score_top5 selector_mech_s4b_atom_route_only_source_score_top5}"
ATOM_ANCHOR_ROOT="${ATOM_ANCHOR_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1}"
CHUNK_CACHE_ROOT="${CHUNK_CACHE_ROOT:-outputs/cache/chunk_mmr/d4cbf7c18126}"
SOURCE_BASE_ROOT="${SOURCE_BASE_ROOT:-outputs/selectors/selector_mechanism_ablation}"
TOP_K="${TOP_K:-5}"
CLAIM_POOL_TOP_N="${CLAIM_POOL_TOP_N:-20}"
RANDOM_SEED="${RANDOM_SEED:-0}"
MERGE_MMR_LAMBDA="${MERGE_MMR_LAMBDA:-0.70}"
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

require_path() {
  local path="$1"
  local label="$2"
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    return 0
  fi
  if [[ ! -e "$path" ]]; then
    printf 'Missing %s: %s\n' "$label" "$path" >&2
    exit 2
  fi
}

for split in ${SPLITS}; do
  chunk_cache="${CHUNK_CACHE_ROOT}/${split}.pkl"
  union_jsonl="${ATOM_ANCHOR_ROOT}/03_atom_union/atom_union_candidate_pool_${split}.jsonl"
  require_path "$chunk_cache" "${split} chunk cache"
  for selector in ${SELECTORS}; do
    if [[ "$selector" == "selector_mech_s4_atom_union_source_score_top5" ]] \
      || [[ "$selector" == "selector_mech_s4b_atom_route_only_source_score_top5" ]]; then
      require_path "$union_jsonl" "${split} atom union candidate pool"
    fi
    output_dir="${SOURCE_BASE_ROOT}/liar_raw_${selector}_${split}"
    echo "[liar-raw-selector-mechanism] split=${split} selector=${selector}"
    echo "[liar-raw-selector-mechanism] chunk_cache=${chunk_cache}"
    echo "[liar-raw-selector-mechanism] atom_union=${union_jsonl}"
    echo "[liar-raw-selector-mechanism] output=${output_dir}"
    run_cmd "${PYTHON_BIN}" scripts/phase5_selectors/build/build_selector_mechanism_ablation_traces.py \
      --chunk-cache-path "${chunk_cache}" \
      --atom-union-jsonl "${union_jsonl}" \
      --output-dir "${output_dir}" \
      --split "${split}" \
      --sample-limit "${SAMPLE_LIMIT}" \
      --selector-name "${selector}" \
      --top-k "${TOP_K}" \
      --claim-pool-top-n "${CLAIM_POOL_TOP_N}" \
      --random-seed "${RANDOM_SEED}" \
      --merge-mmr-lambda "${MERGE_MMR_LAMBDA}" \
      --chunk-mmr-fingerprint "${CHUNK_MMR_FINGERPRINT}"
  done
done
