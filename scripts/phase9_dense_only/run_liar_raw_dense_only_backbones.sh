#!/usr/bin/env bash
set -euo pipefail

# Run LIAR-RAW dense-only traces for multiple backbones.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

BACKBONES="${BACKBONES:-qwen3_4b_2507,llama31_8b}"
SPLITS="${SPLITS:-test}"
SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE:-true}"
RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE:-true}"
FORCE_BUILD="${FORCE_BUILD:-false}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"
FORCE_LABEL_TOKEN_INFER="${FORCE_LABEL_TOKEN_INFER:-false}"
FORCE_INFER="${FORCE_INFER:-true}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_trace_verifier/liar_raw_dense_v0_6c_backbone}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/liar_raw_dense_v0_6c_backbone}"
CONFIG_CACHE_ROOT="${CONFIG_CACHE_ROOT:-outputs/cache/dense_only/liar_raw_backbone_configs}"

echo "[liar-dense-backbones] backbones: ${BACKBONES}"
echo "[liar-dense-backbones] splits   : ${SPLITS}"
echo "[liar-dense-backbones] output  : ${OUTPUT_ROOT}"
echo "[liar-dense-backbones] run root: ${RUN_ROOT}"
echo "[liar-dense-backbones] config  : ${CONFIG_CACHE_ROOT}"
echo "[liar-dense-backbones] latest state: save=${SAVE_LATEST_TRAIN_STATE} resume=${RESUME_LATEST_TRAIN_STATE}"
echo "[liar-dense-backbones] force   : build=${FORCE_BUILD} train=${FORCE_TRAIN} label_token=${FORCE_LABEL_TOKEN_INFER} infer=${FORCE_INFER}"

IFS=',' read -r -a backbone_array <<< "${BACKBONES}"
IFS=',' read -r -a split_array <<< "${SPLITS}"

for raw_backbone in "${backbone_array[@]}"; do
  backbone="${raw_backbone//[[:space:]]/}"
  if [[ -z "${backbone}" ]]; then
    continue
  fi
  for raw_split in "${split_array[@]}"; do
    split="${raw_split//[[:space:]]/}"
    if [[ -z "${split}" ]]; then
      continue
    fi
    echo "[liar-dense-backbones] running backbone=${backbone} split=${split}"
    BACKBONE="${backbone}" \
    SPLIT="${split}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    CONFIG_CACHE_ROOT="${CONFIG_CACHE_ROOT}" \
    SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE}" \
    RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE}" \
    FORCE_BUILD="${FORCE_BUILD}" \
    FORCE_TRAIN="${FORCE_TRAIN}" \
    FORCE_LABEL_TOKEN_INFER="${FORCE_LABEL_TOKEN_INFER}" \
    FORCE_INFER="${FORCE_INFER}" \
    bash scripts/phase9_dense_only/run_liar_raw_dense_only_backbone.sh
  done
done

echo "[liar-dense-backbones] done"
