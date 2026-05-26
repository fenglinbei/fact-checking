#!/usr/bin/env bash
# Build verifier-ready prompts from sentence-level Stage2 oracle evidence.
# Set RUN_TRAIN=true to launch label-token CE training after data construction.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml}"
ORACLE_TRAIN="${ORACLE_TRAIN:-outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl}"
ORACLE_VAL="${ORACLE_VAL:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
ORACLE_TEST="${ORACLE_TEST:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/oracle_direct_verifier/stage2_sentence}"
EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/data/models/}"
PROMPT_MODEL_NAME_OR_PATH="${PROMPT_MODEL_NAME_OR_PATH:-}"
TRAIN_MODEL_NAME_OR_PATH="${TRAIN_MODEL_NAME_OR_PATH:-}"
ORDER="${ORDER:-oracle}"
ORDER_SEED="${ORDER_SEED:-0}"
FILTER="${FILTER:-all}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
RUN_TRAIN="${RUN_TRAIN:-false}"

args=(
  --config "${CONFIG}"
  --train-oracle-results "${ORACLE_TRAIN}"
  --val-oracle-results "${ORACLE_VAL}"
  --output-dir "${OUTPUT_DIR}"
  --expected-chunk-mmr-fingerprint "${EXPECTED_CHUNK_MMR_FINGERPRINT}"
  --model-base-path "${MODEL_BASE_PATH}"
  --order "${ORDER}"
  --order-seed "${ORDER_SEED}"
  --filter "${FILTER}"
)

if [ -n "${ORACLE_TEST}" ]; then
  args+=(--test-oracle-results "${ORACLE_TEST}")
fi
if [ -n "${PROMPT_MODEL_NAME_OR_PATH}" ]; then
  args+=(--prompt-model-name-or-path "${PROMPT_MODEL_NAME_OR_PATH}")
fi
if [ -n "${TRAIN_MODEL_NAME_OR_PATH}" ]; then
  args+=(--train-model-name-or-path "${TRAIN_MODEL_NAME_OR_PATH}")
fi
if [ "${SAMPLE_LIMIT}" != "0" ]; then
  args+=(--sample-limit "${SAMPLE_LIMIT}")
fi

python scripts/phase3_oracle_evidence/build_oracle_direct_verifier_data.py "${args[@]}"

if [ "${RUN_TRAIN}" = "true" ]; then
  accelerate launch \
    --num_processes="${NPROC_PER_NODE:-4}" \
    --num_machines="${NUM_MACHINES:-1}" \
    --mixed_precision="${MIXED_PRECISION:-bf16}" \
    --use_deepspeed \
    --deepspeed_config_file "${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz8_ga1.json}" \
    -m sft.label_token_trainer \
    --config "${OUTPUT_DIR}/train.resolved.yaml"
fi
