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
LORA_ADAPTER="${LORA_ADAPTER:-outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train/best}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
OUTPUT="${OUTPUT:-outputs/learned_lambda/oracle_lambda_${SPLIT_NAME}.jsonl}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
DTYPE="${DTYPE:-auto}"
DEFAULT_LAMBDA="${DEFAULT_LAMBDA:-0.7}"
LABEL_PREFIX="${LABEL_PREFIX:-Label:}"
SCORING_BACKEND="${SCORING_BACKEND:-vllm_hybrid}"
TOP_LOGPROBS="${TOP_LOGPROBS:-20}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_LOGPROBS="${MAX_LOGPROBS:-}"
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
echo "[run_compute_oracle_lambda] label_prefix=${LABEL_PREFIX}"
echo "[run_compute_oracle_lambda] scoring_backend=${SCORING_BACKEND}"
echo "[run_compute_oracle_lambda] top_logprobs=${TOP_LOGPROBS}"
echo "[run_compute_oracle_lambda] score_batch_size=${SCORE_BATCH_SIZE}"
echo "[run_compute_oracle_lambda] max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-auto}"
echo "[run_compute_oracle_lambda] max_num_seqs=${MAX_NUM_SEQS:-auto}"
echo "[run_compute_oracle_lambda] max_logprobs=${MAX_LOGPROBS:-auto}"
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
  --label-prefix "${LABEL_PREFIX}"
  --scoring-backend "${SCORING_BACKEND}"
  --top-logprobs "${TOP_LOGPROBS}"
  --score-batch-size "${SCORE_BATCH_SIZE}"
)

if [[ -n "${TOKENIZER}" ]]; then
  cmd+=(--tokenizer "${TOKENIZER}")
fi

if [[ -n "${LORA_ADAPTER}" ]]; then
  cmd+=(--lora-adapter "${LORA_ADAPTER}" --max-lora-rank "${MAX_LORA_RANK}")
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  cmd+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi

if [[ -n "${MAX_NUM_SEQS}" ]]; then
  cmd+=(--max-num-seqs "${MAX_NUM_SEQS}")
fi

if [[ -n "${MAX_LOGPROBS}" ]]; then
  cmd+=(--max-logprobs "${MAX_LOGPROBS}")
fi

if [[ "${PROGRESS}" == "false" ]]; then
  cmd+=(--no-progress)
fi

cmd+=("$@")

"${cmd[@]}"
