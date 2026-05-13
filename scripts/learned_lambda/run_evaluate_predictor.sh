#!/usr/bin/env bash
# Evaluate the learned-lambda predictor against oracle lambda labels.
#
# Usage:
#   bash scripts/learned_lambda/run_evaluate_predictor.sh
#   SPLIT_NAME=val bash scripts/learned_lambda/run_evaluate_predictor.sh
#   MODEL=outputs/learned_lambda/predictor.pt FEATURE_STATS=outputs/learned_lambda/feature_stats.json bash scripts/learned_lambda/run_evaluate_predictor.sh
#   ORACLE_LAMBDAS=outputs/learned_lambda/oracle_lambda_train.jsonl bash scripts/learned_lambda/run_evaluate_predictor.sh
#   CHUNK_MMR_CACHE=outputs/cache/chunk_mmr/<fingerprint>/train.pkl bash scripts/learned_lambda/run_evaluate_predictor.sh
#   CHUNK_MMR_CACHE_FINGERPRINT=<fingerprint> PROGRESS=false bash scripts/learned_lambda/run_evaluate_predictor.sh
#   CANDIDATE_TOP_K=16 bash scripts/learned_lambda/run_evaluate_predictor.sh
#   FIXED_LAMBDA_GRID="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" bash scripts/learned_lambda/run_evaluate_predictor.sh
#
# Extra CLI args are forwarded to evaluate_predictor.py, for example:
#   bash scripts/learned_lambda/run_evaluate_predictor.sh --alpha-dense 0.75 --alpha-lexical 0.15

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

SPLIT_NAME="${SPLIT_NAME:-train}"
EXPERIMENT="${EXPERIMENT:-b3_mmr_topk_sweep_1024}"
CONFIG_OVERRIDES="${CONFIG_OVERRIDES:-}"
MODEL="${MODEL:-outputs/learned_lambda/predictor.pt}"
FEATURE_STATS="${FEATURE_STATS:-outputs/learned_lambda/feature_stats.json}"
ORACLE_LAMBDAS="${ORACLE_LAMBDAS:-outputs/learned_lambda/oracle_lambda_${SPLIT_NAME}.jsonl}"
CHUNK_MMR_CACHE_ROOT="${CHUNK_MMR_CACHE_ROOT:-outputs/cache/chunk_mmr}"
CHUNK_MMR_CACHE_FINGERPRINT="${CHUNK_MMR_CACHE_FINGERPRINT:-}"
CHUNK_MMR_CACHE="${CHUNK_MMR_CACHE:-}"

HIDDEN_DIM="${HIDDEN_DIM:-256}"
DROPOUT="${DROPOUT:-0.1}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-}"
ALPHA_DENSE="${ALPHA_DENSE:-}"
ALPHA_LEXICAL="${ALPHA_LEXICAL:-}"
ALPHA_BM25="${ALPHA_BM25:-}"
FIXED_LAMBDA_GRID="${FIXED_LAMBDA_GRID:-auto}"
PROGRESS="${PROGRESS:-true}"

if [[ -z "${CHUNK_MMR_CACHE}" && -n "${CHUNK_MMR_CACHE_FINGERPRINT}" ]]; then
  CHUNK_MMR_CACHE="${CHUNK_MMR_CACHE_ROOT}/${CHUNK_MMR_CACHE_FINGERPRINT}/${SPLIT_NAME}.pkl"
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

if [[ -n "${CHUNK_MMR_CACHE}" && ! -f "${CHUNK_MMR_CACHE}" ]]; then
  echo "[run_evaluate_predictor] chunk-MMR cache not found: ${CHUNK_MMR_CACHE}" >&2
  echo "[run_evaluate_predictor] Set CHUNK_MMR_CACHE=/path/to/{train,val,test}.pkl and rerun." >&2
  exit 1
fi

echo "[run_evaluate_predictor] split_name=${SPLIT_NAME}"
echo "[run_evaluate_predictor] experiment=${EXPERIMENT}"
echo "[run_evaluate_predictor] model=${MODEL}"
echo "[run_evaluate_predictor] feature_stats=${FEATURE_STATS}"
echo "[run_evaluate_predictor] oracle_lambdas=${ORACLE_LAMBDAS}"
echo "[run_evaluate_predictor] chunk_mmr_cache=${CHUNK_MMR_CACHE:-auto_by_experiment}"
echo "[run_evaluate_predictor] chunk_mmr_cache_root=${CHUNK_MMR_CACHE_ROOT}"
echo "[run_evaluate_predictor] candidate_top_k=${CANDIDATE_TOP_K:-from_experiment}"
echo "[run_evaluate_predictor] hidden_dim=${HIDDEN_DIM:-from_feature_stats}"
echo "[run_evaluate_predictor] dropout=${DROPOUT:-from_feature_stats}"
echo "[run_evaluate_predictor] alpha_dense=${ALPHA_DENSE:-from_experiment}"
echo "[run_evaluate_predictor] alpha_lexical=${ALPHA_LEXICAL:-from_experiment}"
echo "[run_evaluate_predictor] alpha_bm25=${ALPHA_BM25:-from_experiment}"
echo "[run_evaluate_predictor] fixed_lambda_grid=${FIXED_LAMBDA_GRID}"
echo "[run_evaluate_predictor] progress=${PROGRESS}"
if [[ -n "${CONFIG_OVERRIDES}" ]]; then
  echo "[run_evaluate_predictor] config_overrides=${CONFIG_OVERRIDES}"
fi

cmd=(
  python scripts/learned_lambda/evaluate_predictor.py
  --model "${MODEL}"
  --feature-stats "${FEATURE_STATS}"
  --oracle-lambdas "${ORACLE_LAMBDAS}"
  --experiment "${EXPERIMENT}"
  --split-name "${SPLIT_NAME}"
  --chunk-mmr-cache-root "${CHUNK_MMR_CACHE_ROOT}"
  --fixed-lambda-grid "${FIXED_LAMBDA_GRID}"
)

if [[ -n "${CHUNK_MMR_CACHE}" ]]; then
  cmd+=(--chunk-mmr-cache "${CHUNK_MMR_CACHE}")
fi

if [[ -n "${CANDIDATE_TOP_K}" ]]; then
  cmd+=(--candidate-top-k "${CANDIDATE_TOP_K}")
fi

if [[ -n "${ALPHA_DENSE}" ]]; then
  cmd+=(--alpha-dense "${ALPHA_DENSE}")
fi

if [[ -n "${ALPHA_LEXICAL}" ]]; then
  cmd+=(--alpha-lexical "${ALPHA_LEXICAL}")
fi

if [[ -n "${ALPHA_BM25}" ]]; then
  cmd+=(--alpha-bm25 "${ALPHA_BM25}")
fi

if [[ -n "${CONFIG_OVERRIDES}" ]]; then
  # shellcheck disable=SC2206
  overrides=(${CONFIG_OVERRIDES})
  cmd+=(--config-overrides "${overrides[@]}")
fi

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
