#!/usr/bin/env bash
# Run Stage 2 calibration-aware re-oracle with the Stage 1 verifier.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-outputs/runs/b3_label_token_ce_1024/label_token_ce_stage1__0ee9b55f}"
CONFIG="${CONFIG:-configs/experiment/b3_label_token_ce_1024.yaml}"
VERIFIER_MODEL="${VERIFIER_MODEL:-/data/models/Qwen2.5-7B-Instruct/}"
LORA_ADAPTER="${LORA_ADAPTER:-${STAGE1_RUN_DIR}/train/best}"
SPLIT="${SPLIT:-val}"
TOP_K="${TOP_K:-5}"
SEARCH_METHOD="${SEARCH_METHOD:-greedy}"
SEARCH_OBJECTIVE="${SEARCH_OBJECTIVE:-margin}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TWO_STAGE="${TWO_STAGE:-true}"
TWO_STAGE_MULTIPLIER="${TWO_STAGE_MULTIPLIER:-3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1032}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-128}"
SAVE_CANDIDATE_POOL="${SAVE_CANDIDATE_POOL:-true}"
SAVE_SEARCH_STEP_SCORES="${SAVE_SEARCH_STEP_SCORES:-true}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/data/models/}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/oracle_evidence/stage2_margin_${SPLIT}_${RUN_STAMP}}"

if [ -n "${LORA_ADAPTER}" ] && [ ! -d "${LORA_ADAPTER}" ]; then
    echo "[stage2] LORA_ADAPTER not found: ${LORA_ADAPTER}" >&2
    echo "[stage2] Set STAGE1_RUN_DIR or LORA_ADAPTER to the Stage 1 train/best adapter." >&2
    exit 1
fi

echo "============================================"
echo "Stage 2 Calibration-aware Re-Oracle"
echo "============================================"
echo "Stage1 run dir:    ${STAGE1_RUN_DIR}"
echo "Config:            ${CONFIG}"
echo "Verifier model:    ${VERIFIER_MODEL}"
echo "LoRA adapter:      ${LORA_ADAPTER:-none}"
echo "Split:             ${SPLIT}"
echo "Top-K:             ${TOP_K}"
echo "Search method:     ${SEARCH_METHOD}"
echo "Search objective:  ${SEARCH_OBJECTIVE}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "============================================"

CONFIG="${CONFIG}" \
VERIFIER_MODEL="${VERIFIER_MODEL}" \
LORA_ADAPTER="${LORA_ADAPTER}" \
SPLIT="${SPLIT}" \
TOP_K="${TOP_K}" \
SEARCH_METHOD="${SEARCH_METHOD}" \
SEARCH_OBJECTIVE="${SEARCH_OBJECTIVE}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
TWO_STAGE="${TWO_STAGE}" \
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE}" \
SAVE_CANDIDATE_POOL="${SAVE_CANDIDATE_POOL}" \
SAVE_SEARCH_STEP_SCORES="${SAVE_SEARCH_STEP_SCORES}" \
MODEL_BASE_PATH="${MODEL_BASE_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash scripts/oracle_evidence/run_search.sh \
    --two-stage-multiplier "${TWO_STAGE_MULTIPLIER}" \
    "$@"
