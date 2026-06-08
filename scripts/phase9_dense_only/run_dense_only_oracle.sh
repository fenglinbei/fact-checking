#!/usr/bin/env bash
set -euo pipefail

# Generic dense-only full-pool oracle search wrapper.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET="${DATASET:-rawfc}"  # rawfc | liar_raw
BACKBONE="${BACKBONE:-manual}"
SPLITS="${SPLITS:-val test}"
VERIFIER_MODEL="${VERIFIER_MODEL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/oracle_evidence/${DATASET}_${BACKBONE}_dense_fullpool_margin}"

case "${DATASET}" in
  rawfc)
    CONFIG="${CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_eval25.yaml}"
    LABEL_SCHEMA="${LABEL_SCHEMA:-rawfc3}"
    ;;
  liar_raw)
    CONFIG="${CONFIG:-configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml}"
    LABEL_SCHEMA="${LABEL_SCHEMA:-liar6}"
    ;;
  *)
    echo "[dense-oracle] DATASET must be rawfc or liar_raw, got: ${DATASET}" >&2
    exit 2
    ;;
esac

if [[ -z "${VERIFIER_MODEL}" ]]; then
  echo "[dense-oracle] VERIFIER_MODEL must point to the trained checkpoint, e.g. .../train/best" >&2
  exit 2
fi
if [[ ! -d "${VERIFIER_MODEL}" ]]; then
  echo "[dense-oracle] verifier model not found: ${VERIFIER_MODEL}" >&2
  exit 1
fi

ALPHA_OVERRIDES="build.retrieval.alpha_dense=1.0,build.retrieval.alpha_lexical=0.0,build.retrieval.alpha_bm25=0.0"
PROMPT_OVERRIDE="build.prompt.model_name_or_path=${VERIFIER_MODEL}"
if [[ -n "${CONFIG_OVERRIDES:-}" ]]; then
  CONFIG_OVERRIDES="${ALPHA_OVERRIDES},${PROMPT_OVERRIDE},${CONFIG_OVERRIDES}"
else
  CONFIG_OVERRIDES="${ALPHA_OVERRIDES},${PROMPT_OVERRIDE}"
fi

TOP_K="${TOP_K:-5}"
SEARCH_METHOD="${SEARCH_METHOD:-greedy}"
SEARCH_OBJECTIVE="${SEARCH_OBJECTIVE:-margin}"
TWO_STAGE="${TWO_STAGE:-false}"
MAX_CANDIDATE_POOL_SIZE="${MAX_CANDIDATE_POOL_SIZE:-0}"
SAVE_CANDIDATE_POOL="${SAVE_CANDIDATE_POOL:-true}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
DTYPE="${DTYPE:-bfloat16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
RESUME="${RESUME:-true}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/data/models/}"

echo "[dense-oracle] dataset        : ${DATASET}"
echo "[dense-oracle] backbone       : ${BACKBONE}"
echo "[dense-oracle] config         : ${CONFIG}"
echo "[dense-oracle] verifier       : ${VERIFIER_MODEL}"
echo "[dense-oracle] output         : ${OUTPUT_DIR}"
echo "[dense-oracle] splits         : ${SPLITS}"
echo "[dense-oracle] overrides      : ${CONFIG_OVERRIDES}"

for split in ${SPLITS}; do
  echo "[dense-oracle] running split=${split}"
  CONFIG="${CONFIG}" \
  CONFIG_OVERRIDES="${CONFIG_OVERRIDES}" \
  VERIFIER_MODEL="${VERIFIER_MODEL}" \
  SPLIT="${split}" \
  TOP_K="${TOP_K}" \
  SEARCH_METHOD="${SEARCH_METHOD}" \
  SEARCH_OBJECTIVE="${SEARCH_OBJECTIVE}" \
  TWO_STAGE="${TWO_STAGE}" \
  MAX_CANDIDATE_POOL_SIZE="${MAX_CANDIDATE_POOL_SIZE}" \
  SAVE_CANDIDATE_POOL="${SAVE_CANDIDATE_POOL}" \
  TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  DTYPE="${DTYPE}" \
  SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING}" \
  RESUME="${RESUME}" \
  MODEL_BASE_PATH="${MODEL_BASE_PATH}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  bash scripts/phase3_oracle_evidence/run_search.sh
done

echo "[dense-oracle] done: ${OUTPUT_DIR}"
