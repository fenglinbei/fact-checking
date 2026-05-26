#!/usr/bin/env bash
# Offline selection-gate eval for no-vLLM saved-score utility rankers.
set -euo pipefail

VIG_CACHE="${VIG_CACHE:-outputs/selectors/vig_utility/saved_step_val/vig_records_val.jsonl}"
TRAIN_VIG_CACHE="${TRAIN_VIG_CACHE:-}"
ORACLE_RESULTS="${ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/vig_utility/saved_step_val/ranker_eval}"
SPLIT="${SPLIT:-val}"
EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
MAX_CANDIDATES="${MAX_CANDIDATES:-15}"
TOP_K="${TOP_K:-5}"
FILTER_POLICY="${FILTER_POLICY:-all}"
MIN_MARGIN="${MIN_MARGIN:-0.25}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
TEST_FRACTION="${TEST_FRACTION:-0.25}"
SEED="${SEED:-20260522}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
NO_PROGRESS="${NO_PROGRESS:-false}"
WRITE_ALL_TRACES="${WRITE_ALL_TRACES:-false}"

echo "[saved-score-ranker] vig_cache=${VIG_CACHE}"
if [[ -n "${TRAIN_VIG_CACHE}" ]]; then
  echo "[saved-score-ranker] train_vig_cache=${TRAIN_VIG_CACHE}"
fi
echo "[saved-score-ranker] oracle=${ORACLE_RESULTS}"
echo "[saved-score-ranker] output=${OUTPUT_DIR}"
echo "[saved-score-ranker] split=${SPLIT}"
echo "[saved-score-ranker] fingerprint=${EXPECTED_CHUNK_MMR_FINGERPRINT}"

cmd=(
  python scripts/phase5_selectors/eval/eval_saved_score_utility_ranker.py
  --vig-cache "${VIG_CACHE}"
  --oracle-results "${ORACLE_RESULTS}"
  --output-dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --expected-chunk-mmr-fingerprint "${EXPECTED_CHUNK_MMR_FINGERPRINT}"
  --max-candidates "${MAX_CANDIDATES}"
  --top-k "${TOP_K}"
  --filter-policy "${FILTER_POLICY}"
  --min-margin "${MIN_MARGIN}"
  --ridge-alpha "${RIDGE_ALPHA}"
  --test-fraction "${TEST_FRACTION}"
  --seed "${SEED}"
)

if [[ -n "${TRAIN_VIG_CACHE}" ]]; then
  cmd+=(--train-vig-cache "${TRAIN_VIG_CACHE}")
fi
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  cmd+=(--sample-limit "${SAMPLE_LIMIT}")
fi
if [[ "${NO_PROGRESS}" == "true" ]]; then
  cmd+=(--no-progress)
fi
if [[ "${WRITE_ALL_TRACES}" == "true" ]]; then
  cmd+=(--write-all-traces)
fi

PYTHONPATH=src "${cmd[@]}"
