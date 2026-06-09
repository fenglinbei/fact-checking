#!/usr/bin/env bash
set -euo pipefail

# Run the RAWFC chunking-granularity ablation matrix:
#   evidence units: raw report, semantic chunk, sentence
#
# Common defaults are Qwen3-4B LoRA, adaptive5_10 prompt-budget matched MMR
# (min5, pool32, max20), the original hybrid retrieval weights
# (0.70 dense / 0.20 lexical / 0.10 BM25-like), auto truncation, and
# 1024-token prompts through build -> train followed by local label-token
# evaluation.
#
# Usage:
#   bash scripts/phase10_chunking_ablation/run_all_chunking_ablation.sh
#   MODE=build DRY_RUN=true bash scripts/phase10_chunking_ablation/run_all_chunking_ablation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASETS="${DATASETS:-rawfc}"
CHUNKINGS="${CHUNKINGS:-raw,semantic,sentence}"
SELECTION_METHOD="${SELECTION_METHOD:-mmr_prompt_budget}"
CANDIDATE_POOL_K="${CANDIDATE_POOL_K:-32}"
PROMPT_BUDGET_MIN_K="${PROMPT_BUDGET_MIN_K:-${TOP_K:-5}}"
PROMPT_BUDGET_MAX_K="${PROMPT_BUDGET_MAX_K:-20}"
PROMPT_BUDGET_OVERSHOOT_TOLERANCE_TOKENS="${PROMPT_BUDGET_OVERSHOOT_TOLERANCE_TOKENS:-32}"
PROMPT_BUDGET_TARGET_FIELD="${PROMPT_BUDGET_TARGET_FIELD:-prompt_token_count}"
PROMPT_BUDGET_MISSING_REFERENCE="${PROMPT_BUDGET_MISSING_REFERENCE:-error}"

echo "[chunking-ablation-all] datasets : ${DATASETS}"
echo "[chunking-ablation-all] chunkings: ${CHUNKINGS}"
echo "[chunking-ablation-all] selection: ${SELECTION_METHOD}"
if [[ "${SELECTION_METHOD}" == "mmr_prompt_budget" || "${SELECTION_METHOD}" == "prompt_budget_mmr" || "${SELECTION_METHOD}" == "adaptive_budget_mmr" ]]; then
  echo "[chunking-ablation-all] budget   : min=${PROMPT_BUDGET_MIN_K} pool=${CANDIDATE_POOL_K} max=${PROMPT_BUDGET_MAX_K} target=${PROMPT_BUDGET_TARGET_FIELD} +${PROMPT_BUDGET_OVERSHOOT_TOLERANCE_TOKENS}"
fi

IFS=',' read -r -a dataset_array <<< "${DATASETS}"
IFS=',' read -r -a chunking_array <<< "${CHUNKINGS}"

for raw_dataset in "${dataset_array[@]}"; do
  dataset="${raw_dataset//[[:space:]]/}"
  if [[ -z "${dataset}" ]]; then
    continue
  fi
  for raw_chunking in "${chunking_array[@]}"; do
    chunking="${raw_chunking//[[:space:]]/}"
    if [[ -z "${chunking}" ]]; then
      continue
    fi
    echo "[chunking-ablation-all] running DATASET=${dataset} CHUNKING=${chunking}"
    DATASET="${dataset}" \
    CHUNKING="${chunking}" \
    SELECTION_METHOD="${SELECTION_METHOD}" \
    CANDIDATE_POOL_K="${CANDIDATE_POOL_K}" \
    PROMPT_BUDGET_MIN_K="${PROMPT_BUDGET_MIN_K}" \
    PROMPT_BUDGET_MAX_K="${PROMPT_BUDGET_MAX_K}" \
    PROMPT_BUDGET_OVERSHOOT_TOLERANCE_TOKENS="${PROMPT_BUDGET_OVERSHOOT_TOLERANCE_TOKENS}" \
    PROMPT_BUDGET_TARGET_FIELD="${PROMPT_BUDGET_TARGET_FIELD}" \
    PROMPT_BUDGET_MISSING_REFERENCE="${PROMPT_BUDGET_MISSING_REFERENCE}" \
    bash scripts/phase10_chunking_ablation/run_one_chunking_ablation.sh "$@"
  done
done

echo "[chunking-ablation-all] done"
