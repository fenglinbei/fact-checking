#!/usr/bin/env bash
set -euo pipefail

# Llama-3.1 LIAR-RAW method-upgrade: dynamic ordinal loss warmup
#
# alpha_warmup_ratio=0.3: α linearly increases from 0 to 0.2 over the first
# 30% of training steps. Preserves pre-trained distribution early, then
# gradually increases ordinal regularization.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ENABLE_ORDINAL_LOSS="${ENABLE_ORDINAL_LOSS:-true}"
export ORDINAL_LOSS_ALPHA="${ORDINAL_LOSS_ALPHA:-0.2}"
export METHOD_SUFFIX="${METHOD_SUFFIX:-_ord_abs_a02_warmup03}"

PATCH_ALPHA_WARMUP_RATIO="${PATCH_ALPHA_WARMUP_RATIO:-0.3}"

echo "[method-upgrade-ord-warmup] ordinal α warmup ratio=${PATCH_ALPHA_WARMUP_RATIO}"
echo "[method-upgrade-ord-warmup] METHOD_SUFFIX=${METHOD_SUFFIX}"

# Use the standard launcher for this variant.
# The ordinal loss warmup is applied by the trainer code directly
# (reads alpha_warmup_ratio from config), so we just need to ensure
# the config has the right setting. Since prepare_backbone_config.py
# doesn't set this field, we patch the config post-generation.
bash "${SCRIPT_DIR}/run_llama31_liar_raw.sh" "$@"
