#!/usr/bin/env bash
# Evaluate the learned-lambda MLP predictor against oracle lambda labels.
#
# Usage:
#   bash scripts/learned_lambda/run_evaluate_predictor.sh
#   SPLIT_NAME=val bash scripts/learned_lambda/run_evaluate_predictor.sh
#   MODEL=outputs/learned_lambda/predictor.pt FEATURE_STATS=outputs/learned_lambda/feature_stats.json bash scripts/learned_lambda/run_evaluate_predictor.sh
#   ORACLE_LAMBDAS=outputs/learned_lambda/oracle_lambda_train.jsonl bash scripts/learned_lambda/run_evaluate_predictor.sh
#   PREMMR_CACHE=outputs/cache/pre_mmr/<fingerprint>/train.pkl bash scripts/learned_lambda/run_evaluate_predictor.sh
#   PREMMR_CACHE_FINGERPRINT=68c6d9f97eee PROGRESS=false bash scripts/learned_lambda/run_evaluate_predictor.sh
#   FIXED_LAMBDA_GRID="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" bash scripts/learned_lambda/run_evaluate_predictor.sh
#
# Extra CLI args are forwarded to evaluate_predictor.py, for example:
#   bash scripts/learned_lambda/run_evaluate_predictor.sh --alpha-dense 0.75 --alpha-lexical 0.15

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

SPLIT_NAME="${SPLIT_NAME:-train}"
MODEL="${MODEL:-outputs/learned_lambda/predictor.pt}"
FEATURE_STATS="${FEATURE_STATS:-outputs/learned_lambda/feature_stats.json}"
ORACLE_LAMBDAS="${ORACLE_LAMBDAS:-outputs/learned_lambda/oracle_lambda_${SPLIT_NAME}.jsonl}"
PREMMR_CACHE_ROOT="${PREMMR_CACHE_ROOT:-outputs/cache/pre_mmr}"
PREMMR_CACHE_FINGERPRINT="${PREMMR_CACHE_FINGERPRINT:-68c6d9f97eee}"
PREMMR_CACHE="${PREMMR_CACHE:-}"

HIDDEN_DIM="${HIDDEN_DIM:-}"
DROPOUT="${DROPOUT:-}"
ALPHA_DENSE="${ALPHA_DENSE:-0.70}"
ALPHA_LEXICAL="${ALPHA_LEXICAL:-0.20}"
ALPHA_BM25="${ALPHA_BM25:-0.10}"
FIXED_LAMBDA_GRID="${FIXED_LAMBDA_GRID:-auto}"
PROGRESS="${PROGRESS:-true}"

if [[ -z "${PREMMR_CACHE}" ]]; then
  if [[ -n "${PREMMR_CACHE_FINGERPRINT}" ]]; then
    PREMMR_CACHE="${PREMMR_CACHE_ROOT}/${PREMMR_CACHE_FINGERPRINT}/${SPLIT_NAME}.pkl"
  else
    shopt -s nullglob
    matches=("${PREMMR_CACHE_ROOT}"/*/"${SPLIT_NAME}.pkl")
    shopt -u nullglob
    if [[ "${#matches[@]}" -eq 1 ]]; then
      PREMMR_CACHE="${matches[0]}"
    elif [[ "${#matches[@]}" -eq 0 ]]; then
      echo "[run_evaluate_predictor] No PreMMR cache found for split=${SPLIT_NAME} under ${PREMMR_CACHE_ROOT}" >&2
      echo "[run_evaluate_predictor] Run scripts/learned_lambda/run_generate_oracle_prompts.sh first, or set PREMMR_CACHE." >&2
      exit 1
    else
      echo "[run_evaluate_predictor] Multiple PreMMR caches found for split=${SPLIT_NAME} under ${PREMMR_CACHE_ROOT}" >&2
      printf '[run_evaluate_predictor]   %s\n' "${matches[@]}" >&2
      echo "[run_evaluate_predictor] Set PREMMR_CACHE or PREMMR_CACHE_FINGERPRINT to choose one." >&2
      exit 1
    fi
  fi
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "[run_evaluate_predictor] Predictor model not found: ${MODEL}" >&2
  echo "[run_evaluate_predictor] Run scripts/learned_lambda/run_train_predictor.sh first, or set MODEL." >&2
  exit 1
fi

if [[ ! -f "${FEATURE_STATS}" ]]; then
  echo "[run_evaluate_predictor] Feature stats not found: ${FEATURE_STATS}" >&2
  echo "[run_evaluate_predictor] Run scripts/learned_lambda/run_train_predictor.sh first, or set FEATURE_STATS." >&2
  exit 1
fi

if [[ ! -f "${ORACLE_LAMBDAS}" ]]; then
  echo "[run_evaluate_predictor] Oracle lambda file not found: ${ORACLE_LAMBDAS}" >&2
  echo "[run_evaluate_predictor] Run scripts/learned_lambda/run_compute_oracle_lambda.sh first, or set ORACLE_LAMBDAS." >&2
  exit 1
fi

if [[ ! -f "${PREMMR_CACHE}" ]]; then
  echo "[run_evaluate_predictor] PreMMR cache not found: ${PREMMR_CACHE}" >&2
  echo "[run_evaluate_predictor] Set PREMMR_CACHE=/path/to/{train,val,test}.pkl and rerun." >&2
  exit 1
fi

echo "[run_evaluate_predictor] split_name=${SPLIT_NAME}"
echo "[run_evaluate_predictor] model=${MODEL}"
echo "[run_evaluate_predictor] feature_stats=${FEATURE_STATS}"
echo "[run_evaluate_predictor] oracle_lambdas=${ORACLE_LAMBDAS}"
echo "[run_evaluate_predictor] premmr_cache=${PREMMR_CACHE}"
echo "[run_evaluate_predictor] hidden_dim=${HIDDEN_DIM:-from_feature_stats}"
echo "[run_evaluate_predictor] dropout=${DROPOUT:-from_feature_stats}"
echo "[run_evaluate_predictor] alpha_dense=${ALPHA_DENSE}"
echo "[run_evaluate_predictor] alpha_lexical=${ALPHA_LEXICAL}"
echo "[run_evaluate_predictor] alpha_bm25=${ALPHA_BM25}"
echo "[run_evaluate_predictor] fixed_lambda_grid=${FIXED_LAMBDA_GRID}"
echo "[run_evaluate_predictor] progress=${PROGRESS}"

cmd=(
  python scripts/learned_lambda/evaluate_predictor.py
  --model "${MODEL}"
  --feature-stats "${FEATURE_STATS}"
  --oracle-lambdas "${ORACLE_LAMBDAS}"
  --premmr-cache "${PREMMR_CACHE}"
  --alpha-dense "${ALPHA_DENSE}"
  --alpha-lexical "${ALPHA_LEXICAL}"
  --alpha-bm25 "${ALPHA_BM25}"
  --fixed-lambda-grid "${FIXED_LAMBDA_GRID}"
)

if [[ -n "${HIDDEN_DIM}" ]]; then
  cmd+=(--hidden-dim "${HIDDEN_DIM}")
fi

if [[ -n "${DROPOUT}" ]]; then
  cmd+=(--dropout "${DROPOUT}")
fi

if [[ "${PROGRESS}" == "false" ]]; then
  cmd+=(--no-progress)
fi

cmd+=("$@")

"${cmd[@]}"
