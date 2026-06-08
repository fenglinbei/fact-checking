#!/usr/bin/env bash
set -euo pipefail

# Run LIAR-RAW dense-only traces for multiple backbones.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

BACKBONES="${BACKBONES:-qwen3_4b_2507,llama31_8b}"
SPLITS="${SPLITS:-test}"

echo "[liar-dense-backbones] backbones: ${BACKBONES}"
echo "[liar-dense-backbones] splits   : ${SPLITS}"

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
    bash scripts/phase9_dense_only/run_liar_raw_dense_only_backbone.sh
  done
done

echo "[liar-dense-backbones] done"
