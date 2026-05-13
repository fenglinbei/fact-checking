#!/usr/bin/env bash
# Compute per-claim oracle lambda labels with vLLM batch inference.
#
# Usage:
#   bash scripts/learned_lambda/run_compute_oracle_lambda.sh
#   SPLIT_NAME=val bash scripts/learned_lambda/run_compute_oracle_lambda.sh
#   MODEL=/data/models/Qwen2.5-7B-Instruct TENSOR_PARALLEL_SIZE=4 bash scripts/learned_lambda/run_compute_oracle_lambda.sh
#   MODEL=/data/models/Qwen2.5-7B-Instruct LORA_ADAPTER=outputs/runs/.../train/best bash scripts/learned_lambda/run_compute_oracle_lambda.sh
#   PROMPTS_DIR=outputs/learned_lambda/prompts OUTPUT=outputs/learned_lambda/oracle_lambda_train.jsonl bash scripts/learned_lambda/run_compute_oracle_lambda.sh
#   PROGRESS=false bash scripts/learned_lambda/run_compute_oracle_lambda.sh
#
# Extra CLI args are forwarded to compute_oracle_lambda.py, for example:
#   bash scripts/learned_lambda/run_compute_oracle_lambda.sh --default-lambda 0.5

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

SPLIT_NAME="${SPLIT_NAME:-train}"
PROMPTS_DIR="${PROMPTS_DIR:-outputs/learned_lambda/prompts}"
MODEL="${MODEL:-/data/models/Qwen2.5-7B-Instruct}"
TOKENIZER="${TOKENIZER:-}"
LORA_ADAPTER="${LORA_ADAPTER:-}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
OUTPUT="${OUTPUT:-outputs/learned_lambda/oracle_lambda_${SPLIT_NAME}.jsonl}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
DTYPE="${DTYPE:-auto}"
DEFAULT_LAMBDA="${DEFAULT_LAMBDA:-0.7}"
PROGRESS="${PROGRESS:-true}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

if [[ ! -d "${PROMPTS_DIR}" ]]; then
  echo "[run_compute_oracle_lambda] Prompts directory not found: ${PROMPTS_DIR}" >&2
  echo "[run_compute_oracle_lambda] Run scripts/learned_lambda/run_generate_oracle_prompts.sh first, or set PROMPTS_DIR." >&2
  exit 1
fi

if ! compgen -G "${PROMPTS_DIR}/lambda_*_${SPLIT_NAME}.jsonl" > /dev/null; then
  echo "[run_compute_oracle_lambda] No prompt files found for split=${SPLIT_NAME} in ${PROMPTS_DIR}" >&2
  echo "[run_compute_oracle_lambda] Expected files like: ${PROMPTS_DIR}/lambda_0.70_${SPLIT_NAME}.jsonl" >&2
  exit 1
fi

echo "[run_compute_oracle_lambda] split_name=${SPLIT_NAME}"
echo "[run_compute_oracle_lambda] prompts_dir=${PROMPTS_DIR}"
echo "[run_compute_oracle_lambda] model=${MODEL}"
echo "[run_compute_oracle_lambda] tokenizer=${TOKENIZER:-auto}"
echo "[run_compute_oracle_lambda] lora_adapter=${LORA_ADAPTER:-none}"
echo "[run_compute_oracle_lambda] max_lora_rank=${MAX_LORA_RANK}"
echo "[run_compute_oracle_lambda] output=${OUTPUT}"
echo "[run_compute_oracle_lambda] tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
echo "[run_compute_oracle_lambda] gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo "[run_compute_oracle_lambda] max_model_len=${MAX_MODEL_LEN}"
echo "[run_compute_oracle_lambda] dtype=${DTYPE}"
echo "[run_compute_oracle_lambda] default_lambda=${DEFAULT_LAMBDA}"
echo "[run_compute_oracle_lambda] progress=${PROGRESS}"
echo "[run_compute_oracle_lambda] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

cmd=(
  python scripts/learned_lambda/compute_oracle_lambda.py
  --prompts-dir "${PROMPTS_DIR}"
  --model "${MODEL}"
  --output "${OUTPUT}"
  --split-name "${SPLIT_NAME}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --dtype "${DTYPE}"
  --default-lambda "${DEFAULT_LAMBDA}"
)

if [[ -n "${TOKENIZER}" ]]; then
  cmd+=(--tokenizer "${TOKENIZER}")
fi

if [[ -n "${LORA_ADAPTER}" ]]; then
  cmd+=(--lora-adapter "${LORA_ADAPTER}" --max-lora-rank "${MAX_LORA_RANK}")
fi

if [[ "${PROGRESS}" == "false" ]]; then
  cmd+=(--no-progress)
fi

cmd+=("$@")

"${cmd[@]}"
