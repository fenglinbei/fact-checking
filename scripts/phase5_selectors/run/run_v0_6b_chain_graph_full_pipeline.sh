#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:-configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml}"
INFER_EXPERIMENT="${INFER_EXPERIMENT:-b3_oracle_sentence_direct_verifier_1024}"
TRAIN_RAW="${TRAIN_RAW:-data/raw/LIAR-RAW/train.json}"
VAL_RAW="${VAL_RAW:-data/raw/LIAR-RAW/val.json}"
TEST_RAW="${TEST_RAW:-data/raw/LIAR-RAW/test.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_trace_verifier/stage2_sentence}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/b3_selector_trace_full_pipeline}"

ORACLE_TRAIN="${ORACLE_TRAIN:-outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl}"
ORACLE_VAL="${ORACLE_VAL:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/data/models/}"
PROMPT_MODEL_NAME_OR_PATH="${PROMPT_MODEL_NAME_OR_PATH:-}"
TRAIN_MODEL_NAME_OR_PATH="${TRAIN_MODEL_NAME_OR_PATH:-}"

CASE_NAME="${CASE_NAME:-v0_6b_chain_graph_top5}"
SOURCE_TYPE="${SOURCE_TYPE:-trace}"
TRAIN_SOURCE="${TRAIN_SOURCE:-outputs/selectors/evidence_chain_graph/v0_6b_train/selection_trace_train.jsonl}"
VAL_SOURCE="${VAL_SOURCE:-outputs/selectors/evidence_chain_graph/v0_6b_val/selection_trace_val.jsonl}"
TEST_SOURCE="${TEST_SOURCE:-}"
TRACE_SELECTION_MODE="${TRACE_SELECTION_MODE:-trace}"
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-v0_6b_chain_graph_top5}"
RANDOM_SEED="${RANDOM_SEED:-0}"
RANDOM_SEEDS="${RANDOM_SEEDS:-${RANDOM_SEED}}"
CASE_SPECS="${CASE_SPECS:-}"

TOP_K="${TOP_K:-5}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
CHECKPOINTS="${CHECKPOINTS:-best}"
SPLIT="${SPLIT:-val}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_BUILD="${FORCE_BUILD:-false}"
RUN_TRAIN="${RUN_TRAIN:-true}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"
RUN_INFER="${RUN_INFER:-true}"
FORCE_INFER="${FORCE_INFER:-true}"
PIPELINE_RESUME="${PIPELINE_RESUME:-true}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz8_ga1.json}"
TRAIN_BACKEND="${TRAIN_BACKEND:-accelerate_deepspeed}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
PORT_BASE="${PORT_BASE:-35300}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-120}"
STOP_AFTER_INFER="${STOP_AFTER_INFER:-true}"

MERGE_LORA_CACHE="${MERGE_LORA_CACHE:-true}"
MERGE_LORA_CACHE_DIR="${MERGE_LORA_CACHE_DIR:-outputs/cache/merged_lora}"
MERGE_LORA_CACHE_FORCE_REBUILD="${MERGE_LORA_CACHE_FORCE_REBUILD:-false}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export CONFIG
export INFER_EXPERIMENT
export TRAIN_RAW
export VAL_RAW
export TEST_RAW
export OUTPUT_ROOT
export RUN_ROOT
export ORACLE_TRAIN
export ORACLE_VAL
export EXPECTED_CHUNK_MMR_FINGERPRINT
export MODEL_BASE_PATH
export PROMPT_MODEL_NAME_OR_PATH
export TRAIN_MODEL_NAME_OR_PATH
export CASE_NAME
export SOURCE_TYPE
export TRAIN_SOURCE
export VAL_SOURCE
export TEST_SOURCE
export TRACE_SELECTION_MODE
export EXPECTED_SELECTOR_NAME
export RANDOM_SEED
export RANDOM_SEEDS
export CASE_SPECS
export TOP_K
export SAMPLE_LIMIT
export CHECKPOINTS
export SPLIT
export DRY_RUN
export FORCE_BUILD
export RUN_TRAIN
export FORCE_TRAIN
export RUN_INFER
export FORCE_INFER
export PIPELINE_RESUME
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export NUM_MACHINES
export MIXED_PRECISION
export DEEPSPEED_CONFIG
export TRAIN_BACKEND
export TENSOR_PARALLEL_SIZE
export GPU_MEMORY_UTILIZATION
export PORT_BASE
export WAIT_SECONDS
export REQUEST_TIMEOUT_SECONDS
export STOP_AFTER_INFER
export MERGE_LORA_CACHE
export MERGE_LORA_CACHE_DIR
export MERGE_LORA_CACHE_FORCE_REBUILD
export TOKENIZERS_PARALLELISM

echo "[v0.6b-full-pipeline] config       : ${CONFIG}"
echo "[v0.6b-full-pipeline] infer exp    : ${INFER_EXPERIMENT}"
echo "[v0.6b-full-pipeline] case         : ${CASE_NAME}"
echo "[v0.6b-full-pipeline] finetune     : ${FINETUNE_MODE:-lora/peft}"
echo "[v0.6b-full-pipeline] source_type  : ${SOURCE_TYPE}"
echo "[v0.6b-full-pipeline] train src    : ${TRAIN_SOURCE}"
echo "[v0.6b-full-pipeline] val src      : ${VAL_SOURCE}"
echo "[v0.6b-full-pipeline] test src     : ${TEST_SOURCE:-none}"
echo "[v0.6b-full-pipeline] mode/top_k   : ${TRACE_SELECTION_MODE}/${TOP_K}"
echo "[v0.6b-full-pipeline] selector     : ${EXPECTED_SELECTOR_NAME}"
echo "[v0.6b-full-pipeline] fingerprint : ${EXPECTED_CHUNK_MMR_FINGERPRINT}"
echo "[v0.6b-full-pipeline] sample       : ${SAMPLE_LIMIT}"
echo "[v0.6b-full-pipeline] train/infer  : ${RUN_TRAIN}/${RUN_INFER}"
echo "[v0.6b-full-pipeline] force flags  : build=${FORCE_BUILD} train=${FORCE_TRAIN} infer=${FORCE_INFER}"
echo "[v0.6b-full-pipeline] train backend: ${TRAIN_BACKEND} nproc=${NPROC_PER_NODE} precision=${MIXED_PRECISION}"
echo "[v0.6b-full-pipeline] deepspeed    : ${DEEPSPEED_CONFIG}"
echo "[v0.6b-full-pipeline] cuda/tp      : ${CUDA_VISIBLE_DEVICES} / ${TENSOR_PARALLEL_SIZE}"
echo "[v0.6b-full-pipeline] merge lora   : ${MERGE_LORA_CACHE}"
echo "[v0.6b-full-pipeline] output_root  : ${OUTPUT_ROOT}"
echo "[v0.6b-full-pipeline] run_root     : ${RUN_ROOT}"

exec bash scripts/phase5_selectors/run/run_selector_trace_full_pipeline.sh
