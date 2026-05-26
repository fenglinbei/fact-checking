#!/usr/bin/env bash
# Run the full build -> train -> infer pipeline with the experimental
# Stage-2 sentence-cache pointwise selector.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

EXPERIMENT="${EXPERIMENT:-b3_pointwise_stage2_sentence_1024}"
PIPELINE_MODE="${PIPELINE_MODE:-full}"
PIPELINE_OUTPUT_SUBDIR="${PIPELINE_OUTPUT_SUBDIR:-stage2_sentence_pointwise_full}"
POINTWISE_MODEL_DIR="${POINTWISE_MODEL_DIR:-outputs/oracle_pointwise/stage2_margin_sentence/logreg}"

if [ ! -f "${POINTWISE_MODEL_DIR}/model.npz" ]; then
    echo "[pointwise-stage2] Missing selector model: ${POINTWISE_MODEL_DIR}/model.npz" >&2
    echo "[pointwise-stage2] Train it first with scripts/phase5_selectors/train/train_pointwise_oracle_selector.py." >&2
    exit 1
fi

python -m fact_checking.pipeline.run \
    experiment="${EXPERIMENT}" \
    pipeline.mode="${PIPELINE_MODE}" \
    pipeline.output_subdir="${PIPELINE_OUTPUT_SUBDIR}" \
    build.retrieval.pointwise_oracle.model_dir="${POINTWISE_MODEL_DIR}" \
    "$@"
