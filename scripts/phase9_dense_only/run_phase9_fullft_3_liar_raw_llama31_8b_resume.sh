#!/usr/bin/env bash
set -euo pipefail

# 3. LIAR-RAW dense-only / llama31_8b FullFT resume with bsz1/ga8.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/phase9_dense_only/phase9_fullft_env.sh
source "${SCRIPT_DIR}/phase9_fullft_env.sh"

MODE="${MODE:-full}" \
BACKBONES=llama31_8b \
OUTPUT_ROOT="${LIAR_OUTPUT_ROOT}" \
RUN_ROOT="${LIAR_RUN_ROOT}" \
CONFIG_CACHE_ROOT="${LIAR_CONFIG_CACHE_ROOT}" \
FORCE_BUILD=true \
SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE}" \
RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE}" \
bash scripts/phase9_dense_only/run_liar_raw_dense_only_backbones.sh
