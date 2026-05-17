#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PIPELINE_MODE="${PIPELINE_MODE:-full}"
INFER_SPLIT="${INFER_SPLIT:-val}"
INFER_PORT="${INFER_PORT:-35031}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-label_token_ce_stage1}"

python -m fact_checking.pipeline.run \
  experiment=b3_label_token_ce_1024 \
  pipeline.mode="${PIPELINE_MODE}" \
  pipeline.output_subdir="${OUTPUT_SUBDIR}" \
  infer.split="${INFER_SPLIT}" \
  infer.port="${INFER_PORT}" \
  "$@"
