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
#   SAVE_CANDIDATE_POOL - Save full candidate_pool/candidate_scores (default: true)
#   SAVE_SEARCH_STEP_SCORES - Save per-step oracle logprobs (default: false)

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
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1032}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-256}"
SAVE_CANDIDATE_POOL="${SAVE_CANDIDATE_POOL:-true}"
SAVE_SEARCH_STEP_SCORES="${SAVE_SEARCH_STEP_SCORES:-false}"

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
echo "Save candidates:   $SAVE_CANDIDATE_POOL"
echo "Save step scores:  $SAVE_SEARCH_STEP_SCORES"
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

CANDIDATE_POOL_ARG=()
if [ "$SAVE_CANDIDATE_POOL" = "false" ]; then
    CANDIDATE_POOL_ARG=(--no-save-candidate-pool)
fi

STEP_SCORES_ARG=()
if [ "$SAVE_SEARCH_STEP_SCORES" = "true" ]; then
    STEP_SCORES_ARG=(--save-search-step-scores)
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
    "${CANDIDATE_POOL_ARG[@]}" \
    "${STEP_SCORES_ARG[@]}" \
    "$@"
