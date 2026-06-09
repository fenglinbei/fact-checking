#!/usr/bin/env bash
set -euo pipefail

# Run RAWFC dense-only traces through phase7 backbone migration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

BACKBONES="${BACKBONES:-qwen3_4b_2507,llama31_8b}"
MODE="${MODE:-full}"
FINETUNE="${FINETUNE:-fullft}"
TRACE_ROOT="${TRACE_ROOT:-outputs/selectors/evidence_chain_graph/rawfc_dense_v0_6c_adaptive5_10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_trace_verifier/rawfc_dense_v0_6c_eval25_backbone}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/rawfc_dense_v0_6c_eval25_backbone}"
CONFIG_CACHE_ROOT="${CONFIG_CACHE_ROOT:-outputs/cache/dense_only/rawfc_backbone_configs}"
SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE:-true}"
RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE:-true}"

echo "[rawfc-dense-backbones] backbones : ${BACKBONES}"
echo "[rawfc-dense-backbones] mode/ft   : ${MODE}/${FINETUNE}"
echo "[rawfc-dense-backbones] trace root: ${TRACE_ROOT}"
echo "[rawfc-dense-backbones] output    : ${OUTPUT_ROOT}"
echo "[rawfc-dense-backbones] latest state: save=${SAVE_LATEST_TRAIN_STATE} resume=${RESUME_LATEST_TRAIN_STATE}"

IFS=',' read -r -a backbone_array <<< "${BACKBONES}"
for raw_backbone in "${backbone_array[@]}"; do
  backbone="${raw_backbone//[[:space:]]/}"
  if [[ -z "${backbone}" ]]; then
    continue
  fi
  echo "[rawfc-dense-backbones] running ${backbone}"
  BACKBONE="${backbone}" \
  MODE="${MODE}" \
  FINETUNE="${FINETUNE}" \
  TRACE_ROOT="${TRACE_ROOT}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  CONFIG_CACHE_ROOT="${CONFIG_CACHE_ROOT}" \
  SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE}" \
  RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE}" \
  bash scripts/phase7_backbone_migration/run_one_backbone.sh
done

echo "[rawfc-dense-backbones] done"
