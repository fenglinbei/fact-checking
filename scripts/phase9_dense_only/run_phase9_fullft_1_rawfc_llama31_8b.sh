#!/usr/bin/env bash
set -euo pipefail

# 1. RAWFC dense-only / llama31_8b FullFT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/phase9_dense_only/phase9_fullft_env.sh
source "${SCRIPT_DIR}/phase9_fullft_env.sh"

MODE="${MODE:-full}" \
FINETUNE="${FINETUNE:-fullft}" \
BACKBONES=llama31_8b \
OUTPUT_ROOT="${RAWFC_OUTPUT_ROOT}" \
RUN_ROOT="${RAWFC_RUN_ROOT}" \
CONFIG_CACHE_ROOT="${RAWFC_CONFIG_CACHE_ROOT}" \
SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE}" \
RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE}" \
bash scripts/phase9_dense_only/run_rawfc_dense_only_backbones.sh
