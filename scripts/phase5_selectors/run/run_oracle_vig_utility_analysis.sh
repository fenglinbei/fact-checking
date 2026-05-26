#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

ORACLE_RESULTS="${ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
SPLIT="${SPLIT:-val}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/vig_utility/stage2_margin_${SPLIT}}"

CONFIG="${CONFIG:-configs/experiment/b3_mmr_topk_sweep_1024.yaml}"
VERIFIER_MODEL="${VERIFIER_MODEL:-/data/models/Qwen2.5-7B-Instruct/}"
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-outputs/runs/b3_label_token_ce_1024/label_token_ce_stage1__0ee9b55f}"
LORA_ADAPTER="${LORA_ADAPTER:-${STAGE1_RUN_DIR}/train/best}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/data/models/}"

GPU_DEVICES="${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -n "${GPU_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
fi

if [[ -n "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}"
elif [[ -n "${GPU_DEVICES}" ]]; then
  _gpu_list="${GPU_DEVICES//,/ }"
  # shellcheck disable=SC2206
  _gpu_array=(${_gpu_list})
  TENSOR_PARALLEL_SIZE="${#_gpu_array[@]}"
else
  TENSOR_PARALLEL_SIZE="4"
fi
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1032}"
DTYPE="${DTYPE:-auto}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-256}"

TOP_K="${TOP_K:-5}"
MAX_CANDIDATES="${MAX_CANDIDATES:-15}"
FILTER_POLICY="${FILTER_POLICY:-all}"
MIN_MARGIN="${MIN_MARGIN:-0.25}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
RESUME="${RESUME:-true}"
INCLUDE_FINAL_COUNTERFACTUALS="${INCLUDE_FINAL_COUNTERFACTUALS:-true}"

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

FINAL_ARGS=(--include-final-counterfactuals)
if [[ "${INCLUDE_FINAL_COUNTERFACTUALS}" == "0" || "${INCLUDE_FINAL_COUNTERFACTUALS}" == "false" || "${INCLUDE_FINAL_COUNTERFACTUALS}" == "False" ]]; then
  FINAL_ARGS=(--no-include-final-counterfactuals)
fi

LORA_ARGS=()
if [[ -n "${LORA_ADAPTER}" ]]; then
  LORA_ARGS=(--lora-adapter "${LORA_ADAPTER}")
fi

echo "[vig] oracle results  : ${ORACLE_RESULTS}"
echo "[vig] output dir      : ${OUTPUT_DIR}"
echo "[vig] verifier model  : ${VERIFIER_MODEL}"
echo "[vig] lora adapter    : ${LORA_ADAPTER:-none}"
echo "[vig] gpu devices     : ${CUDA_VISIBLE_DEVICES:-all visible}"
echo "[vig] tp size         : ${TENSOR_PARALLEL_SIZE}"
echo "[vig] shard           : ${SHARD_INDEX}/${NUM_SHARDS}"

if [[ "${ONLY_ANALYZE}" != "1" && "${ONLY_ANALYZE}" != "true" && "${ONLY_ANALYZE}" != "True" ]]; then
  python scripts/phase5_selectors/build/generate_oracle_vig_cache.py \
    --oracle-results "${ORACLE_RESULTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --config "${CONFIG}" \
    --model-base-path "${MODEL_BASE_PATH}" \
    --verifier-model "${VERIFIER_MODEL}" \
    "${LORA_ARGS[@]}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype "${DTYPE}" \
    --score-batch-size "${SCORE_BATCH_SIZE}" \
    --top-k "${TOP_K}" \
    --max-candidates "${MAX_CANDIDATES}" \
    --filter-policy "${FILTER_POLICY}" \
    --min-margin "${MIN_MARGIN}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}" \
    "${RESUME_ARGS[@]}" \
    "${FINAL_ARGS[@]}" \
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
