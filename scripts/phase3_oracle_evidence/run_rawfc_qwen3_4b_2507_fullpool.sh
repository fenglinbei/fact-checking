#!/usr/bin/env bash
# Search RAWFC oracle evidence with the Qwen3-4B-Instruct-2507 verifier over
# the full deduplicated evidence pool for train/val/test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -x "/data/liaozijie/conda/accelerate-fc/bin/python" ]; then
        PYTHON_BIN="/data/liaozijie/conda/accelerate-fc/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

if [ -n "${GPU_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
fi

infer_tensor_parallel_size() {
    if [ -n "${TENSOR_PARALLEL_SIZE:-}" ]; then
        printf "%s" "${TENSOR_PARALLEL_SIZE}"
        return
    fi
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        "${PYTHON_BIN}" - "$CUDA_VISIBLE_DEVICES" <<'PY'
import sys
devices = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
print(max(len(devices), 1))
PY
        return
    fi
    printf "%s" "1"
}

CONFIG="${CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10.yaml}"
BASE_MODEL="${BASE_MODEL:-/data/models/Qwen3-4B-Instruct-2507}"
RUN_ROOT="${RUN_ROOT:-outputs/selector_trace_verifier/rawfc_v0_6c_eval25_backbone}"
CASE_NAME="${CASE_NAME:-v0_6c_rawfc3_rule_step_adaptive5_10_eval25_qwen3_4b_2507}"
FULLFT_MODEL="${FULLFT_MODEL:-${RUN_ROOT}/${CASE_NAME}_fullft/train/best}"
LORA_ADAPTER_DEFAULT="${LORA_ADAPTER_DEFAULT:-${RUN_ROOT}/${CASE_NAME}_lora/train/best}"
FINETUNE_MODE="${FINETUNE_MODE:-fullft}"  # fullft | lora | base

if [ -z "${VERIFIER_MODEL:-}" ]; then
    case "${FINETUNE_MODE}" in
        fullft)
            VERIFIER_MODEL="${FULLFT_MODEL}"
            LORA_ADAPTER="${LORA_ADAPTER:-}"
            ;;
        lora)
            VERIFIER_MODEL="${BASE_MODEL}"
            LORA_ADAPTER="${LORA_ADAPTER:-${LORA_ADAPTER_DEFAULT}}"
            ;;
        base)
            VERIFIER_MODEL="${BASE_MODEL}"
            LORA_ADAPTER="${LORA_ADAPTER:-}"
            ;;
        *)
            echo "[rawfc-qwen3-oracle] FINETUNE_MODE must be one of: fullft, lora, base" >&2
            exit 1
            ;;
    esac
else
    LORA_ADAPTER="${LORA_ADAPTER:-}"
fi

if [ ! -d "${VERIFIER_MODEL}" ]; then
    echo "[rawfc-qwen3-oracle] VERIFIER_MODEL not found: ${VERIFIER_MODEL}" >&2
    echo "[rawfc-qwen3-oracle] Set VERIFIER_MODEL, or set FINETUNE_MODE=lora/base if that is intended." >&2
    exit 1
fi

if [ -n "${LORA_ADAPTER}" ] && [ ! -d "${LORA_ADAPTER}" ]; then
    echo "[rawfc-qwen3-oracle] LORA_ADAPTER not found: ${LORA_ADAPTER}" >&2
    exit 1
fi

CONFIG_OVERRIDES="${CONFIG_OVERRIDES:-build.prompt.model_name_or_path=${VERIFIER_MODEL}}"
SPLITS="${SPLITS:-train val test}"
RUN_NAME="${RUN_NAME:-rawfc_qwen3_4b_2507_fullpool_margin}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/oracle_evidence/${RUN_NAME}}"

TOP_K="${TOP_K:-5}"
SEARCH_METHOD="${SEARCH_METHOD:-greedy}"
SEARCH_OBJECTIVE="${SEARCH_OBJECTIVE:-margin}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TWO_STAGE="${TWO_STAGE:-false}"
MAX_CANDIDATE_POOL_SIZE="${MAX_CANDIDATE_POOL_SIZE:-0}"
TWO_STAGE_MULTIPLIER="${TWO_STAGE_MULTIPLIER:-3}"
TENSOR_PARALLEL_SIZE="$(infer_tensor_parallel_size)"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
DTYPE="${DTYPE:-bfloat16}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
SAVE_CANDIDATE_POOL="${SAVE_CANDIDATE_POOL:-true}"
SAVE_SEARCH_STEP_SCORES="${SAVE_SEARCH_STEP_SCORES:-false}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
RESUME="${RESUME:-true}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/data/models/}"
VERIFY_CONFIG_ONLY="${VERIFY_CONFIG_ONLY:-false}"

VERIFY_ARGS=()
if [ "${VERIFY_CONFIG_ONLY}" = "true" ]; then
    VERIFY_ARGS=(--verify-config-only)
fi

echo "============================================"
echo "RAWFC Qwen3-4B Full-pool Oracle Search"
echo "============================================"
echo "Config:            ${CONFIG}"
echo "Config overrides:  ${CONFIG_OVERRIDES}"
echo "Verifier model:    ${VERIFIER_MODEL}"
echo "LoRA adapter:      ${LORA_ADAPTER:-none}"
echo "Splits:            ${SPLITS}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "Objective:         ${SEARCH_OBJECTIVE}"
echo "Search method:     ${SEARCH_METHOD}"
echo "Two-stage:         ${TWO_STAGE}"
echo "Max pool cap:      ${MAX_CANDIDATE_POOL_SIZE} (0 = full dedup evidence pool)"
echo "Score batch size:  ${SCORE_BATCH_SIZE}"
echo "vLLM max tokens:   ${MAX_NUM_BATCHED_TOKENS}"
echo "vLLM max seqs:     ${MAX_NUM_SEQS}"
echo "Tensor parallel:   ${TENSOR_PARALLEL_SIZE}"
echo "CUDA devices:      ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Python:            ${PYTHON_BIN}"
echo "============================================"

for split in ${SPLITS}; do
    echo "[rawfc-qwen3-oracle] running split=${split}"
    CONFIG="${CONFIG}" \
    CONFIG_OVERRIDES="${CONFIG_OVERRIDES}" \
    VERIFIER_MODEL="${VERIFIER_MODEL}" \
    LORA_ADAPTER="${LORA_ADAPTER}" \
    SPLIT="${split}" \
    TOP_K="${TOP_K}" \
    SEARCH_METHOD="${SEARCH_METHOD}" \
    SEARCH_OBJECTIVE="${SEARCH_OBJECTIVE}" \
    MAX_SAMPLES="${MAX_SAMPLES}" \
    TWO_STAGE="${TWO_STAGE}" \
    TWO_STAGE_MULTIPLIER="${TWO_STAGE_MULTIPLIER}" \
    MAX_CANDIDATE_POOL_SIZE="${MAX_CANDIDATE_POOL_SIZE}" \
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    DTYPE="${DTYPE}" \
    SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE}" \
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING}" \
    SAVE_CANDIDATE_POOL="${SAVE_CANDIDATE_POOL}" \
    SAVE_SEARCH_STEP_SCORES="${SAVE_SEARCH_STEP_SCORES}" \
    NUM_SHARDS="${NUM_SHARDS}" \
    SHARD_INDEX="${SHARD_INDEX}" \
    RESUME="${RESUME}" \
    MODEL_BASE_PATH="${MODEL_BASE_PATH}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    bash scripts/phase3_oracle_evidence/run_search.sh \
        --two-stage-multiplier "${TWO_STAGE_MULTIPLIER}" \
        "${VERIFY_ARGS[@]}" \
        "$@"
done

echo "[rawfc-qwen3-oracle] done: ${OUTPUT_DIR}"
