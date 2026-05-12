#!/usr/bin/env bash
# Train the learned-lambda MLP predictor from oracle lambda labels and PreMMR features.
#
# Usage:
#   bash scripts/learned_lambda/run_train_predictor.sh
#   SPLIT_NAME=val bash scripts/learned_lambda/run_train_predictor.sh
#   ORACLE_LAMBDAS=outputs/learned_lambda/oracle_lambda_train.jsonl bash scripts/learned_lambda/run_train_predictor.sh
#   PREMMR_CACHE=outputs/cache/pre_mmr/<fingerprint>/train.pkl bash scripts/learned_lambda/run_train_predictor.sh
#   PREMMR_CACHE_FINGERPRINT=68c6d9f97eee bash scripts/learned_lambda/run_train_predictor.sh
#   EPOCHS=100 BATCH_SIZE=128 PROGRESS=false bash scripts/learned_lambda/run_train_predictor.sh
#
# Extra CLI args are forwarded to train_predictor.py, for example:
#   bash scripts/learned_lambda/run_train_predictor.sh --lr 5e-4 --patience 20

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

SPLIT_NAME="${SPLIT_NAME:-train}"
ORACLE_LAMBDAS="${ORACLE_LAMBDAS:-outputs/learned_lambda/oracle_lambda_${SPLIT_NAME}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/learned_lambda}"
PREMMR_CACHE_ROOT="${PREMMR_CACHE_ROOT:-outputs/cache/pre_mmr}"
PREMMR_CACHE_FINGERPRINT="${PREMMR_CACHE_FINGERPRINT:-68c6d9f97eee}"
PREMMR_CACHE="${PREMMR_CACHE:-}"

HIDDEN_DIM="${HIDDEN_DIM:-256}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PATIENCE="${PATIENCE:-30}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
ALPHA_DENSE="${ALPHA_DENSE:-0.70}"
ALPHA_LEXICAL="${ALPHA_LEXICAL:-0.20}"
ALPHA_BM25="${ALPHA_BM25:-0.10}"
SEED="${SEED:-42}"
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
      echo "[run_train_predictor] No PreMMR cache found for split=${SPLIT_NAME} under ${PREMMR_CACHE_ROOT}" >&2
      echo "[run_train_predictor] Run scripts/learned_lambda/run_generate_oracle_prompts.sh first, or set PREMMR_CACHE." >&2
      exit 1
    else
      echo "[run_train_predictor] Multiple PreMMR caches found for split=${SPLIT_NAME} under ${PREMMR_CACHE_ROOT}" >&2
      printf '[run_train_predictor]   %s\n' "${matches[@]}" >&2
      echo "[run_train_predictor] Set PREMMR_CACHE or PREMMR_CACHE_FINGERPRINT to choose one." >&2
      exit 1
    fi
  fi
fi

if [[ ! -f "${ORACLE_LAMBDAS}" ]]; then
  echo "[run_train_predictor] Oracle lambda file not found: ${ORACLE_LAMBDAS}" >&2
  echo "[run_train_predictor] Run scripts/learned_lambda/run_compute_oracle_lambda.sh first, or set ORACLE_LAMBDAS." >&2
  exit 1
fi

if [[ ! -f "${PREMMR_CACHE}" ]]; then
  echo "[run_train_predictor] PreMMR cache not found: ${PREMMR_CACHE}" >&2
  echo "[run_train_predictor] Set PREMMR_CACHE=/path/to/{train,val,test}.pkl and rerun." >&2
  exit 1
fi

echo "[run_train_predictor] split_name=${SPLIT_NAME}"
echo "[run_train_predictor] oracle_lambdas=${ORACLE_LAMBDAS}"
echo "[run_train_predictor] premmr_cache=${PREMMR_CACHE}"
echo "[run_train_predictor] output_dir=${OUTPUT_DIR}"
echo "[run_train_predictor] hidden_dim=${HIDDEN_DIM}"
echo "[run_train_predictor] dropout=${DROPOUT}"
echo "[run_train_predictor] epochs=${EPOCHS}"
echo "[run_train_predictor] lr=${LR}"
echo "[run_train_predictor] weight_decay=${WEIGHT_DECAY}"
echo "[run_train_predictor] batch_size=${BATCH_SIZE}"
echo "[run_train_predictor] patience=${PATIENCE}"
echo "[run_train_predictor] val_fraction=${VAL_FRACTION}"
echo "[run_train_predictor] alpha_dense=${ALPHA_DENSE}"
echo "[run_train_predictor] alpha_lexical=${ALPHA_LEXICAL}"
echo "[run_train_predictor] alpha_bm25=${ALPHA_BM25}"
echo "[run_train_predictor] seed=${SEED}"
echo "[run_train_predictor] progress=${PROGRESS}"

cmd=(
  python scripts/learned_lambda/train_predictor.py
  --oracle-lambdas "${ORACLE_LAMBDAS}"
  --premmr-cache "${PREMMR_CACHE}"
  --output-dir "${OUTPUT_DIR}"
  --hidden-dim "${HIDDEN_DIM}"
  --dropout "${DROPOUT}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --batch-size "${BATCH_SIZE}"
  --patience "${PATIENCE}"
  --val-fraction "${VAL_FRACTION}"
  --alpha-dense "${ALPHA_DENSE}"
  --alpha-lexical "${ALPHA_LEXICAL}"
  --alpha-bm25 "${ALPHA_BM25}"
  --seed "${SEED}"
)

if [[ "${PROGRESS}" == "false" ]]; then
  cmd+=(--no-progress)
fi

cmd+=("$@")

"${cmd[@]}"
