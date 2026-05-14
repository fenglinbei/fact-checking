#!/usr/bin/env bash
# Train the learned-lambda predictor on the high-margin subset only.
#
# Only claims with oracle logprob margin >= MARGIN_THRESHOLD are used for
# training. Low-margin claims fall back to the default λ=0.7 at inference time.
#
# Usage:
#   bash scripts/learned_lambda/run_train_predictor_high_margin.sh
#   MARGIN_THRESHOLD=0.03 bash scripts/learned_lambda/run_train_predictor_high_margin.sh  # alternate threshold
#
# Extra CLI args are forwarded to train_predictor.py, for example:
#   bash scripts/learned_lambda/run_train_predictor_high_margin.sh --lr 5e-4

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

SPLIT_NAME="${SPLIT_NAME:-train}"
EXPERIMENT="${EXPERIMENT:-b3_mmr_topk_sweep_1024}"
CONFIG_OVERRIDES="${CONFIG_OVERRIDES:-build.retrieval.top_k=5}"
ORACLE_LAMBDAS="${ORACLE_LAMBDAS:-outputs/learned_lambda/oracle_lambda_${SPLIT_NAME}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/learned_lambda_high_margin}"
CHUNK_MMR_CACHE_ROOT="${CHUNK_MMR_CACHE_ROOT:-outputs/cache/chunk_mmr}"
CHUNK_MMR_CACHE_FINGERPRINT="${CHUNK_MMR_CACHE_FINGERPRINT:-}"
CHUNK_MMR_CACHE="${CHUNK_MMR_CACHE:-}"

MARGIN_THRESHOLD="${MARGIN_THRESHOLD:-0.05}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
DROPOUT="${DROPOUT:-0.2}"
EPOCHS="${EPOCHS:-150}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PATIENCE="${PATIENCE:-20}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
OBJECTIVE="${OBJECTIVE:-regression}"
REGRESSION_LOSS="${REGRESSION_LOSS:-mse}"
HUBER_DELTA="${HUBER_DELTA:-0.1}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-}"
ALPHA_DENSE="${ALPHA_DENSE:-}"
ALPHA_LEXICAL="${ALPHA_LEXICAL:-}"
ALPHA_BM25="${ALPHA_BM25:-}"
SEED="${SEED:-42}"
PROGRESS="${PROGRESS:-true}"

if [[ -z "${CHUNK_MMR_CACHE}" && -n "${CHUNK_MMR_CACHE_FINGERPRINT}" ]]; then
  CHUNK_MMR_CACHE="${CHUNK_MMR_CACHE_ROOT}/${CHUNK_MMR_CACHE_FINGERPRINT}/${SPLIT_NAME}.pkl"
fi

if [[ ! -f "${ORACLE_LAMBDAS}" ]]; then
  echo "[run_train_predictor_high_margin] Oracle lambda file not found: ${ORACLE_LAMBDAS}" >&2
  echo "[run_train_predictor_high_margin] Run scripts/learned_lambda/run_compute_oracle_lambda.sh first, or set ORACLE_LAMBDAS." >&2
  exit 1
fi

if [[ -n "${CHUNK_MMR_CACHE}" && ! -f "${CHUNK_MMR_CACHE}" ]]; then
  echo "[run_train_predictor_high_margin] chunk-MMR cache not found: ${CHUNK_MMR_CACHE}" >&2
  echo "[run_train_predictor_high_margin] Set CHUNK_MMR_CACHE=/path/to/{train,val,test}.pkl and rerun." >&2
  exit 1
fi

echo "[run_train_predictor_high_margin] split_name=${SPLIT_NAME}"
echo "[run_train_predictor_high_margin] experiment=${EXPERIMENT}"
echo "[run_train_predictor_high_margin] oracle_lambdas=${ORACLE_LAMBDAS}"
echo "[run_train_predictor_high_margin] chunk_mmr_cache=${CHUNK_MMR_CACHE:-auto_by_experiment}"
echo "[run_train_predictor_high_margin] chunk_mmr_cache_root=${CHUNK_MMR_CACHE_ROOT}"
echo "[run_train_predictor_high_margin] output_dir=${OUTPUT_DIR}"
echo "[run_train_predictor_high_margin] margin_threshold=${MARGIN_THRESHOLD}"
echo "[run_train_predictor_high_margin] candidate_top_k=${CANDIDATE_TOP_K:-full_chunk_pool}"
echo "[run_train_predictor_high_margin] hidden_dim=${HIDDEN_DIM}"
echo "[run_train_predictor_high_margin] dropout=${DROPOUT}"
echo "[run_train_predictor_high_margin] epochs=${EPOCHS}"
echo "[run_train_predictor_high_margin] lr=${LR}"
echo "[run_train_predictor_high_margin] weight_decay=${WEIGHT_DECAY}"
echo "[run_train_predictor_high_margin] batch_size=${BATCH_SIZE}"
echo "[run_train_predictor_high_margin] patience=${PATIENCE}"
echo "[run_train_predictor_high_margin] val_fraction=${VAL_FRACTION}"
echo "[run_train_predictor_high_margin] objective=${OBJECTIVE}"
echo "[run_train_predictor_high_margin] regression_loss=${REGRESSION_LOSS}"
echo "[run_train_predictor_high_margin] huber_delta=${HUBER_DELTA}"
echo "[run_train_predictor_high_margin] alpha_dense=${ALPHA_DENSE:-from_experiment}"
echo "[run_train_predictor_high_margin] alpha_lexical=${ALPHA_LEXICAL:-from_experiment}"
echo "[run_train_predictor_high_margin] alpha_bm25=${ALPHA_BM25:-from_experiment}"
echo "[run_train_predictor_high_margin] seed=${SEED}"
echo "[run_train_predictor_high_margin] progress=${PROGRESS}"
if [[ -n "${CONFIG_OVERRIDES}" ]]; then
  echo "[run_train_predictor_high_margin] config_overrides=${CONFIG_OVERRIDES}"
fi

cmd=(
  python scripts/learned_lambda/train_predictor.py
  --oracle-lambdas "${ORACLE_LAMBDAS}"
  --experiment "${EXPERIMENT}"
  --split-name "${SPLIT_NAME}"
  --chunk-mmr-cache-root "${CHUNK_MMR_CACHE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --margin-threshold "${MARGIN_THRESHOLD}"
  --hidden-dim "${HIDDEN_DIM}"
  --dropout "${DROPOUT}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --batch-size "${BATCH_SIZE}"
  --patience "${PATIENCE}"
  --val-fraction "${VAL_FRACTION}"
  --objective "${OBJECTIVE}"
  --regression-loss "${REGRESSION_LOSS}"
  --huber-delta "${HUBER_DELTA}"
  --seed "${SEED}"
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

if [[ "${PROGRESS}" == "false" ]]; then
  cmd+=(--no-progress)
fi

cmd+=("$@")

"${cmd[@]}"
