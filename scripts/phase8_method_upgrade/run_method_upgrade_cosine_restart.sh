#!/usr/bin/env bash
set -euo pipefail

# Llama-3.1 LIAR-RAW method-upgrade: cosine_with_restarts LR schedule
#
# Replaces single cosine decay with cosine_with_restarts (num_cycles=2).
# Each cycle restarts LR from the peak, helping escape sharp minima.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ENABLE_ORDINAL_LOSS="${ENABLE_ORDINAL_LOSS:-true}"
export ORDINAL_LOSS_ALPHA="${ORDINAL_LOSS_ALPHA:-0.2}"
export METHOD_SUFFIX="${METHOD_SUFFIX:-_ord_abs_a02_cosrestart2}"

PATCH_LR_SCHEDULER="${PATCH_LR_SCHEDULER:-cosine_with_restarts}"
PATCH_LR_KWARGS="${PATCH_LR_KWARGS:-{\"num_cycles\": 2}}"

echo "[method-upgrade-cos-restart] scheduler=${PATCH_LR_SCHEDULER}"
echo "[method-upgrade-cos-restart] lr_kwargs=${PATCH_LR_KWARGS}"
echo "[method-upgrade-cos-restart] METHOD_SUFFIX=${METHOD_SUFFIX}"

bash "${SCRIPT_DIR}/run_llama31_liar_raw.sh" "$@"
