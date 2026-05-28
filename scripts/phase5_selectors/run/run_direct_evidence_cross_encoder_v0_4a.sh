#!/usr/bin/env bash
set -euo pipefail

NCCL_CUMEM_HOST_ENABLE=0

SPLIT="${SPLIT:-val}"
INPUT_BUCKET_FILE="${INPUT_BUCKET_FILE:-outputs/selectors/count_amplified_stance_bucket_selector/v0_2_${SPLIT}/candidate_stance_buckets_v02_n7_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/direct_evidence_cross_encoder/v0_4a_${SPLIT}}"
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
RESUME="${RESUME:-1}"
RUN_SCORE="${RUN_SCORE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
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

echo "[direct-ce-v0.4a] split       : ${SPLIT}"
echo "[direct-ce-v0.4a] input       : ${INPUT_BUCKET_FILE}"
echo "[direct-ce-v0.4a] output dir  : ${OUTPUT_DIR}"
echo "[direct-ce-v0.4a] base model  : ${BASE_MODEL}"
echo "[direct-ce-v0.4a] shards      : ${NUM_SHARDS}"
echo "[direct-ce-v0.4a] cuda devices: ${CUDA_DEVICES}"
echo "[direct-ce-v0.4a] mock scores : ${MOCK_SCORES}"

mkdir -p "${OUTPUT_DIR}"

if [[ "${RUN_SCORE}" == "1" || "${RUN_SCORE}" == "true" || "${RUN_SCORE}" == "True" ]]; then
  IFS=',' read -r -a DEVICE_LIST <<< "${CUDA_DEVICES}"
  if [[ "${#DEVICE_LIST[@]}" -lt "${NUM_SHARDS}" && "${MOCK_SCORES}" != "1" && "${MOCK_SCORES}" != "true" && "${MOCK_SCORES}" != "True" ]]; then
    echo "[direct-ce-v0.4a] NUM_SHARDS=${NUM_SHARDS} but only ${#DEVICE_LIST[@]} CUDA devices were listed" >&2
    exit 1
  fi
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    if [[ "${MOCK_SCORES}" == "1" || "${MOCK_SCORES}" == "true" || "${MOCK_SCORES}" == "True" ]]; then
      shard_device="auto"
    else
      shard_device="cuda:${shard}"
    fi
    echo "[direct-ce-v0.4a] launch shard=${shard}/${NUM_SHARDS} device=${shard_device}"
    PYTHONPATH=src python scripts/phase5_selectors/eval/score_direct_evidence_cross_encoder_v0_4a.py \
      --candidate-stance-buckets "${INPUT_BUCKET_FILE}" \
      --output-dir "${OUTPUT_DIR}" \
      --split "${SPLIT}" \
      --model-name "${BASE_MODEL}" \
      --max-length "${MAX_LENGTH}" \
      --batch-size "${BATCH_SIZE}" \
      --device "${shard_device}" \
      --torch-dtype "${TORCH_DTYPE}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}" \
      "${RESUME_ARGS[@]}" \
      "${MOCK_ARGS[@]}" \
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
    --merge-shards \
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

echo "[direct-ce-v0.4a] done: ${OUTPUT_DIR}"
