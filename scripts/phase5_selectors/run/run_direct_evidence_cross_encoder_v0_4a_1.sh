#!/usr/bin/env bash
set -euo pipefail

NCCL_CUMEM_HOST_ENABLE=0

SPLIT="${SPLIT:-val}"
PROMPT_MODE="${PROMPT_MODE:-direct_evidence_custom}"
PROMPT_VERSION="${PROMPT_VERSION:-direct_evidence_ce_v0_4a_1}"
INPUT_BUCKET_FILE="${INPUT_BUCKET_FILE:-outputs/selectors/count_amplified_stance_bucket_selector/v0_2_${SPLIT}/candidate_stance_buckets_v02_n7_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/direct_evidence_cross_encoder/v0_4a_1_${SPLIT}_${PROMPT_MODE}}"
BASE_MODEL="${BASE_MODEL:-/data/models/Qwen3-Reranker-8B}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
NUM_SHARDS="${NUM_SHARDS:-4}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
SOURCE_PENALTY="${SOURCE_PENALTY:-0.05}"
TOP_K="${TOP_K:-5}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
MOCK_SCORES="${MOCK_SCORES:-0}"
RESUME="${RESUME:-0}"
RUN_SCORE="${RUN_SCORE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
SKIP_CANARY_CHECK="${SKIP_CANARY_CHECK:-0}"
SKIP_SCORE_SANITY_CHECK="${SKIP_SCORE_SANITY_CHECK:-0}"
DISABLE_INPUT_ID_DTYPE_REPAIR="${DISABLE_INPUT_ID_DTYPE_REPAIR:-0}"
MIN_SCORE_STD="${MIN_SCORE_STD:-0.0001}"
MIN_UNIQUE_SCORES="${MIN_UNIQUE_SCORES:-3}"
MAX_EVENT_ALL_TIE_RATE="${MAX_EVENT_ALL_TIE_RATE:-0.5}"
V03_REFERENCE_SCORED="${V03_REFERENCE_SCORED:-outputs/selectors/oracle_likelihood_constrained_selector/v0_3_1_${SPLIT}/pointwise_all_features/candidate_oracle_likelihood_scores_${SPLIT}.jsonl}"

SAMPLE_ARGS=()
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  SAMPLE_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

MOCK_ARGS=()
if [[ "${MOCK_SCORES}" == "1" || "${MOCK_SCORES}" == "true" || "${MOCK_SCORES}" == "True" ]]; then
  MOCK_ARGS=(--mock-scores)
fi

RESUME_ARGS=()
if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "True" ]]; then
  RESUME_ARGS=(--resume)
fi

CANARY_ARGS=()
if [[ "${SKIP_CANARY_CHECK}" == "1" || "${SKIP_CANARY_CHECK}" == "true" || "${SKIP_CANARY_CHECK}" == "True" ]]; then
  CANARY_ARGS=(--skip-canary-check)
fi

SANITY_ARGS=()
if [[ "${SKIP_SCORE_SANITY_CHECK}" == "1" || "${SKIP_SCORE_SANITY_CHECK}" == "true" || "${SKIP_SCORE_SANITY_CHECK}" == "True" ]]; then
  SANITY_ARGS=(--skip-score-sanity-check)
fi

DTYPE_REPAIR_ARGS=()
if [[ "${DISABLE_INPUT_ID_DTYPE_REPAIR}" == "1" || "${DISABLE_INPUT_ID_DTYPE_REPAIR}" == "true" || "${DISABLE_INPUT_ID_DTYPE_REPAIR}" == "True" ]]; then
  DTYPE_REPAIR_ARGS=(--disable-input-id-dtype-repair)
fi

echo "[direct-ce-v0.4a.1] split       : ${SPLIT}"
echo "[direct-ce-v0.4a.1] input       : ${INPUT_BUCKET_FILE}"
echo "[direct-ce-v0.4a.1] output dir  : ${OUTPUT_DIR}"
echo "[direct-ce-v0.4a.1] base model  : ${BASE_MODEL}"
echo "[direct-ce-v0.4a.1] prompt mode : ${PROMPT_MODE}"
echo "[direct-ce-v0.4a.1] shards      : ${NUM_SHARDS}"
echo "[direct-ce-v0.4a.1] cuda devices: ${CUDA_DEVICES}"
echo "[direct-ce-v0.4a.1] mock scores : ${MOCK_SCORES}"
echo "[direct-ce-v0.4a.1] resume      : ${RESUME}"
echo "[direct-ce-v0.4a.1] dtype repair disabled: ${DISABLE_INPUT_ID_DTYPE_REPAIR}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${RUN_SCORE}" == "1" || "${RUN_SCORE}" == "true" || "${RUN_SCORE}" == "True" ]]; then
  IFS=',' read -r -a DEVICE_LIST <<< "${CUDA_DEVICES}"
  if [[ "${#DEVICE_LIST[@]}" -lt "${NUM_SHARDS}" && "${MOCK_SCORES}" != "1" && "${MOCK_SCORES}" != "true" && "${MOCK_SCORES}" != "True" ]]; then
    echo "[direct-ce-v0.4a.1] NUM_SHARDS=${NUM_SHARDS} but only ${#DEVICE_LIST[@]} CUDA devices were listed" >&2
    exit 1
  fi
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    if [[ "${MOCK_SCORES}" == "1" || "${MOCK_SCORES}" == "true" || "${MOCK_SCORES}" == "True" ]]; then
      shard_device="auto"
    else
      shard_device="cuda:${shard}"
    fi
    echo "[direct-ce-v0.4a.1] launch shard=${shard}/${NUM_SHARDS} device=${shard_device}"
    PYTHONPATH=src python scripts/phase5_selectors/eval/score_direct_evidence_cross_encoder_v0_4a.py \
      --candidate-stance-buckets "${INPUT_BUCKET_FILE}" \
      --output-dir "${OUTPUT_DIR}" \
      --split "${SPLIT}" \
      --model-name "${BASE_MODEL}" \
      --prompt-version "${PROMPT_VERSION}" \
      --prompt-mode "${PROMPT_MODE}" \
      --max-length "${MAX_LENGTH}" \
      --batch-size "${BATCH_SIZE}" \
      --device "${shard_device}" \
      --torch-dtype "${TORCH_DTYPE}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}" \
      --min-score-std "${MIN_SCORE_STD}" \
      --min-unique-scores "${MIN_UNIQUE_SCORES}" \
      --max-event-all-tie-rate "${MAX_EVENT_ALL_TIE_RATE}" \
      "${RESUME_ARGS[@]}" \
      "${MOCK_ARGS[@]}" \
      "${CANARY_ARGS[@]}" \
      "${SANITY_ARGS[@]}" \
      "${DTYPE_REPAIR_ARGS[@]}" \
      "${SAMPLE_ARGS[@]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  PYTHONPATH=src python scripts/phase5_selectors/eval/score_direct_evidence_cross_encoder_v0_4a.py \
    --candidate-stance-buckets "${INPUT_BUCKET_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --num-shards "${NUM_SHARDS}" \
    --prompt-mode "${PROMPT_MODE}" \
    --min-score-std "${MIN_SCORE_STD}" \
    --min-unique-scores "${MIN_UNIQUE_SCORES}" \
    --max-event-all-tie-rate "${MAX_EVENT_ALL_TIE_RATE}" \
    --merge-shards \
    "${SANITY_ARGS[@]}" \
    "${SAMPLE_ARGS[@]}"
fi

if [[ "${RUN_EVAL}" == "1" || "${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "True" ]]; then
  PYTHONPATH=src python scripts/phase5_selectors/eval/eval_direct_evidence_cross_encoder_v0_4a.py \
    --scored-candidates "${OUTPUT_DIR}/direct_ce_scored_candidates_${SPLIT}.jsonl" \
    --output-dir "${OUTPUT_DIR}/eval" \
    --split "${SPLIT}" \
    --top-k "${TOP_K}" \
    --source-penalty "${SOURCE_PENALTY}" \
    --v03-reference-scored-candidates "${V03_REFERENCE_SCORED}" \
    "${SAMPLE_ARGS[@]}"
fi

echo "[direct-ce-v0.4a.1] done: ${OUTPUT_DIR}"
