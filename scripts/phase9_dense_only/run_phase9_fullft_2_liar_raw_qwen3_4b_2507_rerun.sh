#!/usr/bin/env bash
set -euo pipefail

# 2. LIAR-RAW dense-only / qwen3_4b_2507 FullFT rerun with bsz2/ga4.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/phase9_dense_only/phase9_fullft_env.sh
source "${SCRIPT_DIR}/phase9_fullft_env.sh"

MODE="${MODE:-full}" \
BACKBONES=qwen3_4b_2507 \
OUTPUT_ROOT="${LIAR_OUTPUT_ROOT}" \
RUN_ROOT="${LIAR_RUN_ROOT}" \
CONFIG_CACHE_ROOT="${LIAR_CONFIG_CACHE_ROOT}" \
FORCE_BUILD=true \
FORCE_TRAIN=true \
FORCE_LABEL_TOKEN_INFER=true \
SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE}" \
RESUME_LATEST_TRAIN_STATE=false \
bash scripts/phase9_dense_only/run_liar_raw_dense_only_backbones.sh
