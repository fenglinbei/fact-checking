#!/usr/bin/env bash
# ==============================================================================
# Soft-label RL-MMR full pipeline: build -> train -> infer
#
# Usage:
#   bash scripts/rl_mmr/run_soft_label_full.sh [INFERENCE_MODE] [EXPERIMENT] [MODE]
#
# Examples:
#   bash scripts/rl_mmr/run_soft_label_full.sh argmax
#   MODEL_PATH=outputs/rl_mmr/soft_label/mlp bash scripts/rl_mmr/run_soft_label_full.sh expected
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

INFERENCE_MODE="${1:-${INFERENCE_MODE:-argmax}}"
EXPERIMENT="${2:-${EXPERIMENT:-mmr_soft_label}}"
MODE="${3:-${MODE:-full}}"
MODEL_PATH="${MODEL_PATH:-outputs/rl_mmr/soft_label/lightgbm}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.5}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-soft_label_${INFERENCE_MODE}}"

echo "============================================"
echo "  Soft-Label RL-MMR Pipeline"
echo "  experiment      : ${EXPERIMENT}"
echo "  mode            : ${MODE}"
echo "  inference_mode  : ${INFERENCE_MODE}"
echo "  model_path      : ${MODEL_PATH}"
echo "  output_subdir   : ${OUTPUT_SUBDIR}"
echo "============================================"
echo ""

python -m fact_checking.pipeline.run \
    "experiment=${EXPERIMENT}" \
    "pipeline.mode=${MODE}" \
    "pipeline.output_subdir=${OUTPUT_SUBDIR}" \
    "build.retrieval.learned_lambda.soft_label.model_path=${MODEL_PATH}" \
    "build.retrieval.learned_lambda.soft_label.inference_mode=${INFERENCE_MODE}" \
    "build.retrieval.learned_lambda.soft_label.sample_temperature=${SAMPLE_TEMPERATURE}"
