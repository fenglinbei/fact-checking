#!/usr/bin/env bash
# Run oracle evidence selection search.
#
# Usage:
#   bash scripts/phase3_oracle_evidence/run_search.sh
#
# Environment variable overrides:
#   CONFIG              - Experiment config path
#   CONFIG_OVERRIDES    - Extra comma-separated config overrides
#   VERIFIER_MODEL      - Trained verifier model path (required)
#   LORA_ADAPTER        - LoRA adapter path (optional)
#   TOP_K               - Target evidence set size (default: 5)
#   SEARCH_METHOD       - greedy | exhaustive | beam (default: greedy)
#   SEARCH_OBJECTIVE    - gold_logprob | margin (default: gold_logprob)
#   SPLIT               - Data split (default: val)
#   MAX_SAMPLES         - Max samples to process (default: 0 = all)
#   OUTPUT_DIR          - Output directory (default: timestamped under outputs/oracle_evidence)
#   MODEL_BASE_PATH     - Override /data/models/ prefix for local models
#   PYTHON_BIN          - Python executable (default: python)
#   TWO_STAGE           - Enable/disable two-stage pruning (default: true)
#   TENSOR_PARALLEL_SIZE - Number of GPUs for vLLM (default: 4)
#   GPU_MEMORY_UTILIZATION - GPU memory utilization (default: 0.90)
#   MAX_MODEL_LEN       - Max model sequence length (default: 1032)
#   DTYPE               - vLLM dtype (default: auto)
#   SCORE_BATCH_SIZE    - Batch size for vLLM scoring calls (default: 256)
#   MAX_NUM_BATCHED_TOKENS - vLLM scheduler max_num_batched_tokens (default: 0 = vLLM default)
#   MAX_NUM_SEQS        - vLLM scheduler max_num_seqs (default: 0 = vLLM default)
#   ENABLE_PREFIX_CACHING - Enable vLLM prefix caching when supported (default: true)
#   SAVE_CANDIDATE_POOL - Save full candidate_pool/candidate_scores (default: true)
#   SAVE_SEARCH_STEP_SCORES - Save per-step oracle logprobs (default: false)
#   MAX_CANDIDATE_POOL_SIZE - Hard cap after dedup/two-stage (default: 0 = full pool)
#   NUM_SHARDS          - Number of deterministic event_id shards (default: 1)
#   SHARD_INDEX         - Shard index to run, in [0, NUM_SHARDS) (default: 0)
#   RESUME              - Skip event_ids already present in shard JSONL (default: true)

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/experiment/b3_mmr_topk_sweep_1024.yaml}"
CONFIG_OVERRIDES="${CONFIG_OVERRIDES:-}"
VERIFIER_MODEL="${VERIFIER_MODEL:?VERIFIER_MODEL must be set}"
LORA_ADAPTER="${LORA_ADAPTER:-}"
TOP_K="${TOP_K:-5}"
SEARCH_METHOD="${SEARCH_METHOD:-greedy}"
SEARCH_OBJECTIVE="${SEARCH_OBJECTIVE:-gold_logprob}"
SPLIT="${SPLIT:-val}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TWO_STAGE="${TWO_STAGE:-true}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1032}"
DTYPE="${DTYPE:-auto}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
SAVE_CANDIDATE_POOL="${SAVE_CANDIDATE_POOL:-true}"
SAVE_SEARCH_STEP_SCORES="${SAVE_SEARCH_STEP_SCORES:-false}"
MAX_CANDIDATE_POOL_SIZE="${MAX_CANDIDATE_POOL_SIZE:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
RESUME="${RESUME:-true}"

echo "============================================"
echo "Oracle Evidence Selection Search"
echo "============================================"
echo "Config:            $CONFIG"
echo "Config overrides:  ${CONFIG_OVERRIDES:-none}"
echo "Verifier model:    $VERIFIER_MODEL"
echo "LoRA adapter:      ${LORA_ADAPTER:-none}"
echo "Top-K:             $TOP_K"
echo "Search method:     $SEARCH_METHOD"
echo "Search objective:  $SEARCH_OBJECTIVE"
echo "Split:             $SPLIT"
echo "Max samples:       $MAX_SAMPLES"
echo "Output dir:        ${OUTPUT_DIR:-auto timestamp}"
echo "Tensor parallel:   $TENSOR_PARALLEL_SIZE"
echo "GPU mem util:      $GPU_MEMORY_UTILIZATION"
echo "Max model len:     $MAX_MODEL_LEN"
echo "Dtype:             $DTYPE"
echo "Score batch size:  $SCORE_BATCH_SIZE"
echo "Max batched tokens:${MAX_NUM_BATCHED_TOKENS}"
echo "Max num seqs:      ${MAX_NUM_SEQS}"
echo "Prefix caching:    ${ENABLE_PREFIX_CACHING}"
echo "Two-stage:         $TWO_STAGE"
echo "Max pool cap:      ${MAX_CANDIDATE_POOL_SIZE}"
echo "Save candidates:   $SAVE_CANDIDATE_POOL"
echo "Save step scores:  $SAVE_SEARCH_STEP_SCORES"
echo "Shard:             $SHARD_INDEX/$NUM_SHARDS"
echo "Resume:            $RESUME"
echo "Model base path:   ${MODEL_BASE_PATH:-auto}"
echo "Python:            ${PYTHON_BIN}"
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

CONFIG_OVERRIDES_ARG=()
if [ -n "$CONFIG_OVERRIDES" ]; then
    CONFIG_OVERRIDES_ARG=(--config-overrides "$CONFIG_OVERRIDES")
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

PREFIX_CACHING_ARG=()
if [ "$ENABLE_PREFIX_CACHING" = "false" ]; then
    PREFIX_CACHING_ARG=(--disable-prefix-caching)
fi

RESUME_ARG=()
if [ "$RESUME" = "false" ]; then
    RESUME_ARG=(--no-resume)
fi

OUTPUT_DIR_ARG=()
if [ -n "$OUTPUT_DIR" ]; then
    OUTPUT_DIR_ARG=(--output-dir "$OUTPUT_DIR")
fi

"${PYTHON_BIN}" scripts/phase3_oracle_evidence/search_optimal_evidence.py \
    --config "$CONFIG" \
    "${CONFIG_OVERRIDES_ARG[@]}" \
    --verifier-model "$VERIFIER_MODEL" \
    "${LORA_ARG[@]}" \
    --top-k "$TOP_K" \
    --search-method "$SEARCH_METHOD" \
    --objective "$SEARCH_OBJECTIVE" \
    --split "$SPLIT" \
    --max-samples "$MAX_SAMPLES" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE" \
    --score-batch-size "$SCORE_BATCH_SIZE" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-candidate-pool-size "$MAX_CANDIDATE_POOL_SIZE" \
    "${MODEL_PATH_ARG[@]}" \
    "${TWO_STAGE_ARG[@]}" \
    "${CANDIDATE_POOL_ARG[@]}" \
    "${STEP_SCORES_ARG[@]}" \
    "${PREFIX_CACHING_ARG[@]}" \
    "${RESUME_ARG[@]}" \
    "${OUTPUT_DIR_ARG[@]}" \
    "$@"
