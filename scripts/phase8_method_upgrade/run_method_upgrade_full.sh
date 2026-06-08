#!/usr/bin/env bash
set -euo pipefail

# Llama-3.1 LIAR-RAW method-upgrade: full combo
#
# Applies all 4 improvements:
#   1. weight_decay = 0.01
#   2. α warmup ratio = 0.3 (ordinal loss)
#   3. cosine_with_restarts (num_cycles=2)
#   4. early_stopping_metric = macro_f1_plus_true_side_plus_mae

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ENABLE_ORDINAL_LOSS="${ENABLE_ORDINAL_LOSS:-true}"
export ORDINAL_LOSS_ALPHA="${ORDINAL_LOSS_ALPHA:-0.2}"
export METHOD_SUFFIX="${METHOD_SUFFIX:-_ord_abs_a02_full_upgrade}"

PATCH_WEIGHT_DECAY="${PATCH_WEIGHT_DECAY:-0.01}"
PATCH_ALPHA_WARMUP_RATIO="${PATCH_ALPHA_WARMUP_RATIO:-0.3}"
PATCH_LR_SCHEDULER="${PATCH_LR_SCHEDULER:-cosine_with_restarts}"
PATCH_LR_KWARGS="${PATCH_LR_KWARGS:-{\"num_cycles\": 2}}"
PATCH_EARLY_STOPPING_METRIC="${PATCH_EARLY_STOPPING_METRIC:-macro_f1_plus_true_side_plus_mae}"
PATCH_MAE_METRIC_WEIGHT="${PATCH_MAE_METRIC_WEIGHT:-0.3}"

post_patch_config() {
  local config_path="$1"
  echo "[method-upgrade-full] patching config: ${config_path}"
  python "${SCRIPT_DIR}/patch_config_for_upgrade.py" \
    --config "${config_path}" \
    --weight-decay "${PATCH_WEIGHT_DECAY}" \
    --alpha-warmup-ratio "${PATCH_ALPHA_WARMUP_RATIO}" \
    --lr-scheduler "${PATCH_LR_SCHEDULER}" \
    --lr-kwargs "${PATCH_LR_KWARGS}" \
    --early-stopping-metric "${PATCH_EARLY_STOPPING_METRIC}" \
    --mae-metric-weight "${PATCH_MAE_METRIC_WEIGHT}"
}

echo "[method-upgrade-full] applying all upgrades:"
echo "  weight_decay=${PATCH_WEIGHT_DECAY}"
echo "  alpha_warmup_ratio=${PATCH_ALPHA_WARMUP_RATIO}"
echo "  lr_scheduler=${PATCH_LR_SCHEDULER} kwargs=${PATCH_LR_KWARGS}"
echo "  early_stopping_metric=${PATCH_EARLY_STOPPING_METRIC}"
echo "  mae_metric_weight=${PATCH_MAE_METRIC_WEIGHT}"
echo "[method-upgrade-full] METHOD_SUFFIX=${METHOD_SUFFIX}"

# Export post_patch_config so the pipeline can call it after config generation
export -f post_patch_config

bash "${SCRIPT_DIR}/run_llama31_liar_raw.sh" "$@"
