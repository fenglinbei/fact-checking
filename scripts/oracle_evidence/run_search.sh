#!/usr/bin/env bash
# Run oracle evidence selection search.
#
# Usage:
#   bash scripts/oracle_evidence/run_search.sh
#
# Environment variable overrides:
#   CONFIG              - Experiment config path
#   VERIFIER_MODEL      - Trained verifier model path (required)
#   LORA_ADAPTER        - LoRA adapter path (optional)
#   TOP_K               - Target evidence set size (default: 5)
#   SEARCH_METHOD       - greedy | exhaustive | beam (default: greedy)
#   SPLIT               - Data split (default: val)
#   MAX_SAMPLES         - Max samples to process (default: 0 = all)
#   MODEL_BASE_PATH     - Override /data/models/ prefix for local models
#   TWO_STAGE           - Enable/disable two-stage pruning (default: true)
#   TENSOR_PARALLEL_SIZE - Number of GPUs for vLLM (default: 4)
#   GPU_MEMORY_UTILIZATION - GPU memory utilization (default: 0.95)
#   MAX_MODEL_LEN       - Max model sequence length (default: 1024)
#   SCORE_BATCH_SIZE    - Batch size for vLLM scoring calls (default: 512)

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/experiment/b3_mmr_topk_sweep_1024.yaml}"
VERIFIER_MODEL="${VERIFIER_MODEL:?VERIFIER_MODEL must be set}"
LORA_ADAPTER="${LORA_ADAPTER:-}"
TOP_K="${TOP_K:-5}"
SEARCH_METHOD="${SEARCH_METHOD:-greedy}"
SPLIT="${SPLIT:-val}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-}"
TWO_STAGE="${TWO_STAGE:-true}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-512}"

echo "============================================"
echo "Oracle Evidence Selection Search"
echo "============================================"
echo "Config:            $CONFIG"
echo "Verifier model:    $VERIFIER_MODEL"
echo "LoRA adapter:      ${LORA_ADAPTER:-none}"
echo "Top-K:             $TOP_K"
echo "Search method:     $SEARCH_METHOD"
echo "Split:             $SPLIT"
echo "Max samples:       $MAX_SAMPLES"
echo "Tensor parallel:   $TENSOR_PARALLEL_SIZE"
echo "GPU mem util:      $GPU_MEMORY_UTILIZATION"
echo "Max model len:     $MAX_MODEL_LEN"
echo "Score batch size:  $SCORE_BATCH_SIZE"
echo "Two-stage:         $TWO_STAGE"
echo "Model base path:   ${MODEL_BASE_PATH:-auto}"
echo "============================================"

# Build optional args
LORA_ARG=()
if [ -n "$LORA_ADAPTER" ]; then
    LORA_ARG=(--lora-adapter "$LORA_ADAPTER")
fi

MODEL_PATH_ARG=()
if [ -n "$MODEL_BASE_PATH" ]; then
    MODEL_PATH_ARG=(--model-base-path "$MODEL_BASE_PATH")
fi

TWO_STAGE_ARG=()
if [ "$TWO_STAGE" = "false" ]; then
    TWO_STAGE_ARG=(--no-two-stage)
fi

python scripts/oracle_evidence/search_optimal_evidence.py \
    --config "$CONFIG" \
    --verifier-model "$VERIFIER_MODEL" \
    "${LORA_ARG[@]}" \
    --top-k "$TOP_K" \
    --search-method "$SEARCH_METHOD" \
    --split "$SPLIT" \
    --max-samples "$MAX_SAMPLES" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --score-batch-size "$SCORE_BATCH_SIZE" \
    "${MODEL_PATH_ARG[@]}" \
    "${TWO_STAGE_ARG[@]}" \
    "$@"
