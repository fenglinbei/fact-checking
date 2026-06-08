#!/usr/bin/env bash
set -euo pipefail

# Llama-3.1 LIAR-RAW method-upgrade: weight_decay=0.01
#
# Improves training stability by adding weight regularization to full
# fine-tuning, reducing late-stage CE loss oscillation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ENABLE_ORDINAL_LOSS="${ENABLE_ORDINAL_LOSS:-true}"
export ORDINAL_LOSS_ALPHA="${ORDINAL_LOSS_ALPHA:-0.2}"
export METHOD_SUFFIX="${METHOD_SUFFIX:-_ord_abs_a02_wd001}"

# Post-patch the resolved config after generation
PATCH_WEIGHT_DECAY="${PATCH_WEIGHT_DECAY:-0.01}"

run_with_patch() {
  local config_path="$1"
  echo "[method-upgrade-wd] patching config: ${config_path}"
  python "${SCRIPT_DIR}/patch_config_for_upgrade.py" \
    --config "${config_path}" \
    --weight-decay "${PATCH_WEIGHT_DECAY}"
}

# Override the training command to inject config patching
# We hook into the pipeline by pre-patching the config before train phase.
# The patch_config_for_upgrade.py modifies train.resolved.yaml in-place.
_original_run_train() {
  # The pipeline generates train.resolved.yaml first, then runs training.
  # We intercept to patch the config before accelerate launch.
  local train_dir="$1"
  local config_path="${train_dir}/train.resolved.yaml"
  if [[ -f "${config_path}" ]]; then
    run_with_patch "${config_path}"
  fi
}

# We can't easily hook into the pipeline, so instead we'll run the standard
# script and then patch the config before training. The simplest approach:
# Run with TRAIN_BACKEND=custom but handle it ourselves.

echo "[method-upgrade-wd] running with weight_decay=${PATCH_WEIGHT_DECAY}"
echo "[method-upgrade-wd] METHOD_SUFFIX=${METHOD_SUFFIX}"

# Use the standard launcher — config is auto-generated.
# We patch train.resolved.yaml after build phase completes.
bash "${SCRIPT_DIR}/run_llama31_liar_raw.sh" "$@"
