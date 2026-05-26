#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

ORACLE_RESULTS="${ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
SPLIT="${SPLIT:-val}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/vig_utility/saved_step_${SPLIT}}"

TOP_K="${TOP_K:-5}"
MAX_CANDIDATES="${MAX_CANDIDATES:-15}"
FILTER_POLICY="${FILTER_POLICY:-all}"
MIN_MARGIN="${MIN_MARGIN:-0.25}"
EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
RESUME="${RESUME:-true}"
NO_PROGRESS="${NO_PROGRESS:-false}"

RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
TEST_FRACTION="${TEST_FRACTION:-0.25}"
SEED="${SEED:-20260522}"
ONLY_ANALYZE="${ONLY_ANALYZE:-false}"
ANALYZE="${ANALYZE:-true}"

SAMPLE_LIMIT_ARGS=()
if [[ -n "${SAMPLE_LIMIT:-}" ]]; then
  SAMPLE_LIMIT_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

RESUME_ARGS=(--resume)
if [[ "${RESUME}" == "0" || "${RESUME}" == "false" || "${RESUME}" == "False" ]]; then
  RESUME_ARGS=(--no-resume)
fi

PROGRESS_ARGS=()
if [[ "${NO_PROGRESS}" == "1" || "${NO_PROGRESS}" == "true" || "${NO_PROGRESS}" == "True" ]]; then
  PROGRESS_ARGS=(--no-progress)
fi

echo "[saved-step] oracle results : ${ORACLE_RESULTS}"
echo "[saved-step] output dir     : ${OUTPUT_DIR}"
echo "[saved-step] split          : ${SPLIT}"
echo "[saved-step] shard          : ${SHARD_INDEX}/${NUM_SHARDS}"
echo "[saved-step] no-vLLM        : true"

if [[ "${ONLY_ANALYZE}" != "1" && "${ONLY_ANALYZE}" != "true" && "${ONLY_ANALYZE}" != "True" ]]; then
  python scripts/phase5_selectors/build/generate_oracle_saved_step_utility.py \
    --oracle-results "${ORACLE_RESULTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --expected-chunk-mmr-fingerprint "${EXPECTED_CHUNK_MMR_FINGERPRINT}" \
    --top-k "${TOP_K}" \
    --max-candidates "${MAX_CANDIDATES}" \
    --filter-policy "${FILTER_POLICY}" \
    --min-margin "${MIN_MARGIN}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}" \
    "${RESUME_ARGS[@]}" \
    "${PROGRESS_ARGS[@]}" \
    "${SAMPLE_LIMIT_ARGS[@]}" \
    "$@"
fi

if [[ "${ANALYZE}" == "1" || "${ANALYZE}" == "true" || "${ANALYZE}" == "True" ]]; then
  if [[ "${NUM_SHARDS}" == "1" ]]; then
    VIG_CACHE="${OUTPUT_DIR}/vig_records_${SPLIT}.jsonl"
    FINAL_CACHE="${OUTPUT_DIR}/vig_final_counterfactuals_${SPLIT}.jsonl"
  else
    VIG_CACHE="${OUTPUT_DIR}/vig_records_${SPLIT}.shard-*-of-*.jsonl"
    FINAL_CACHE="${OUTPUT_DIR}/vig_final_counterfactuals_${SPLIT}.shard-*-of-*.jsonl"
  fi
  python scripts/phase5_selectors/eval/analyze_oracle_vig_utility.py \
    --vig-cache "${VIG_CACHE}" \
    --final-counterfactuals "${FINAL_CACHE}" \
    --output-dir "${OUTPUT_DIR}/analysis" \
    --split "${SPLIT}" \
    --ridge-alpha "${RIDGE_ALPHA}" \
    --test-fraction "${TEST_FRACTION}" \
    --seed "${SEED}"
fi
