#!/usr/bin/env bash
# Train a 3-bin coarse lambda classifier.
#
# Maps the 21-value oracle lambda grid to 3 coarse bins:
#   diversity  [0.0, 0.3] -> center 0.15
#   balanced   (0.3, 0.7) -> center 0.50
#   relevance  [0.7, 1.0] -> center 0.85
#
# Usage:
#   bash scripts/phase2_learned_lambda/run_train_predictor_coarse.sh
#   OBJECTIVE=soft_classification bash scripts/phase2_learned_lambda/run_train_predictor_coarse.sh
#   LAMBDA_GRID="0.1,0.5,0.9" bash scripts/phase2_learned_lambda/run_train_predictor_coarse.sh  # custom grid
#
# Extra CLI args are forwarded to train_predictor.py, for example:
#   bash scripts/phase2_learned_lambda/run_train_predictor_coarse.sh --softmax-temperature 3.0

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

SPLIT_NAME="${SPLIT_NAME:-train}"
EXPERIMENT="${EXPERIMENT:-b3_mmr_topk_sweep_1024}"
CONFIG_OVERRIDES="${CONFIG_OVERRIDES:-build.retrieval.top_k=5}"
ORACLE_LAMBDAS="${ORACLE_LAMBDAS:-outputs/learned_lambda/oracle_lambda_${SPLIT_NAME}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/learned_lambda_coarse}"
CHUNK_MMR_CACHE_ROOT="${CHUNK_MMR_CACHE_ROOT:-outputs/cache/chunk_mmr}"
CHUNK_MMR_CACHE_FINGERPRINT="${CHUNK_MMR_CACHE_FINGERPRINT:-}"
CHUNK_MMR_CACHE="${CHUNK_MMR_CACHE:-}"

OBJECTIVE="${OBJECTIVE:-classification}"
LAMBDA_GRID="${LAMBDA_GRID:-0.15,0.50,0.85}"
SOFTMAX_TEMPERATURE="${SOFTMAX_TEMPERATURE:-2.0}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
PATIENCE="${PATIENCE:-30}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
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
  echo "[run_train_predictor_coarse] Oracle lambda file not found: ${ORACLE_LAMBDAS}" >&2
  echo "[run_train_predictor_coarse] Run scripts/phase2_learned_lambda/run_compute_oracle_lambda.sh first, or set ORACLE_LAMBDAS." >&2
  exit 1
fi

if [[ -n "${CHUNK_MMR_CACHE}" && ! -f "${CHUNK_MMR_CACHE}" ]]; then
  echo "[run_train_predictor_coarse] chunk-MMR cache not found: ${CHUNK_MMR_CACHE}" >&2
  echo "[run_train_predictor_coarse] Set CHUNK_MMR_CACHE=/path/to/{train,val,test}.pkl and rerun." >&2
  exit 1
fi

echo "[run_train_predictor_coarse] split_name=${SPLIT_NAME}"
echo "[run_train_predictor_coarse] experiment=${EXPERIMENT}"
echo "[run_train_predictor_coarse] oracle_lambdas=${ORACLE_LAMBDAS}"
echo "[run_train_predictor_coarse] chunk_mmr_cache=${CHUNK_MMR_CACHE:-auto_by_experiment}"
echo "[run_train_predictor_coarse] chunk_mmr_cache_root=${CHUNK_MMR_CACHE_ROOT}"
echo "[run_train_predictor_coarse] output_dir=${OUTPUT_DIR}"
echo "[run_train_predictor_coarse] objective=${OBJECTIVE}"
echo "[run_train_predictor_coarse] lambda_grid=${LAMBDA_GRID}"
echo "[run_train_predictor_coarse] softmax_temperature=${SOFTMAX_TEMPERATURE}"
echo "[run_train_predictor_coarse] candidate_top_k=${CANDIDATE_TOP_K:-full_chunk_pool}"
echo "[run_train_predictor_coarse] hidden_dim=${HIDDEN_DIM}"
echo "[run_train_predictor_coarse] dropout=${DROPOUT}"
echo "[run_train_predictor_coarse] epochs=${EPOCHS}"
echo "[run_train_predictor_coarse] lr=${LR}"
echo "[run_train_predictor_coarse] weight_decay=${WEIGHT_DECAY}"
echo "[run_train_predictor_coarse] batch_size=${BATCH_SIZE}"
echo "[run_train_predictor_coarse] patience=${PATIENCE}"
echo "[run_train_predictor_coarse] val_fraction=${VAL_FRACTION}"
echo "[run_train_predictor_coarse] alpha_dense=${ALPHA_DENSE:-from_experiment}"
echo "[run_train_predictor_coarse] alpha_lexical=${ALPHA_LEXICAL:-from_experiment}"
echo "[run_train_predictor_coarse] alpha_bm25=${ALPHA_BM25:-from_experiment}"
echo "[run_train_predictor_coarse] seed=${SEED}"
echo "[run_train_predictor_coarse] progress=${PROGRESS}"
if [[ -n "${CONFIG_OVERRIDES}" ]]; then
  echo "[run_train_predictor_coarse] config_overrides=${CONFIG_OVERRIDES}"
fi

cmd=(
  python scripts/phase2_learned_lambda/train_predictor.py
  --oracle-lambdas "${ORACLE_LAMBDAS}"
  --experiment "${EXPERIMENT}"
  --split-name "${SPLIT_NAME}"
  --chunk-mmr-cache-root "${CHUNK_MMR_CACHE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --objective "${OBJECTIVE}"
  --lambda-grid "${LAMBDA_GRID}"
  --hidden-dim "${HIDDEN_DIM}"
  --dropout "${DROPOUT}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --batch-size "${BATCH_SIZE}"
  --patience "${PATIENCE}"
  --val-fraction "${VAL_FRACTION}"
  --seed "${SEED}"
)

if [[ "${OBJECTIVE}" == "soft_classification" ]]; then
  cmd+=(--softmax-temperature "${SOFTMAX_TEMPERATURE}")
fi

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
