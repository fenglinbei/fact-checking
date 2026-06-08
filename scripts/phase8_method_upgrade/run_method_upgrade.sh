#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# Llama-3.1 LIAR-RAW method-upgrade launcher
#
# Supports 4 independent flags that can be combined:
#   --wd 0.01          weight_decay (default: unset = 0.0)
#   --warmup 0.3       α warmup ratio (default: unset = 0.0, no warmup)
#   --restart 2        cosine_with_restarts num_cycles (default: unset = cosine)
#   --calibrated       use macro_f1_plus_true_side_plus_mae early-stop metric
#
# Examples:
#   bash run_method_upgrade.sh --wd 0.01
#   bash run_method_upgrade.sh --warmup 0.3
#   bash run_method_upgrade.sh --restart 2
#   bash run_method_upgrade.sh --wd 0.01 --warmup 0.3 --restart 2 --calibrated
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- defaults ----
WEIGHT_DECAY=""
ALPHA_WARMUP_RATIO=""
NUM_CYCLES=""
USE_CALIBRATED_METRIC="false"
MAE_METRIC_WEIGHT="0.3"
ENABLE_ORDINAL_LOSS="${ENABLE_ORDINAL_LOSS:-true}"
ORDINAL_LOSS_ALPHA="${ORDINAL_LOSS_ALPHA:-0.2}"
EVAL_SPLITS="${EVAL_SPLITS:-${INFER_SPLIT:-val,test}}"
METHOD_SUFFIX_BASE=""

# ---- parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wd)
      WEIGHT_DECAY="$2"
      METHOD_SUFFIX_BASE="${METHOD_SUFFIX_BASE}_wd${2/./}"
      shift 2 ;;
    --warmup)
      ALPHA_WARMUP_RATIO="$2"
      METHOD_SUFFIX_BASE="${METHOD_SUFFIX_BASE}_warmup${2/./}"
      shift 2 ;;
    --restart)
      NUM_CYCLES="$2"
      METHOD_SUFFIX_BASE="${METHOD_SUFFIX_BASE}_cr${2}"
      shift 2 ;;
    --calibrated)
      USE_CALIBRATED_METRIC="true"
      METHOD_SUFFIX_BASE="${METHOD_SUFFIX_BASE}_calib"
      shift ;;
    --mae-weight)
      MAE_METRIC_WEIGHT="$2"
      shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--wd VALUE] [--warmup RATIO] [--restart N] [--calibrated]" >&2
      exit 1 ;;
  esac
done

if [[ -z "${METHOD_SUFFIX_BASE}" ]]; then
  echo "ERROR: at least one upgrade flag is required." >&2
  echo "Usage: $0 [--wd VALUE] [--warmup RATIO] [--restart N] [--calibrated]" >&2
  exit 1
fi

# Generate the ordinal-aware base config first (build-only), then patch and train.
export METHOD_SUFFIX="_ord_abs_a02${METHOD_SUFFIX_BASE}"

echo "============================================================"
echo "[method-upgrade] Configuration:"
echo "  WEIGHT_DECAY         = ${WEIGHT_DECAY:-unset}"
echo "  ALPHA_WARMUP_RATIO   = ${ALPHA_WARMUP_RATIO:-unset}"
echo "  NUM_CYCLES           = ${NUM_CYCLES:-unset}"
echo "  CALIBRATED_METRIC    = ${USE_CALIBRATED_METRIC}"
echo "  MAE_METRIC_WEIGHT    = ${MAE_METRIC_WEIGHT}"
echo "  ENABLE_ORDINAL_LOSS  = ${ENABLE_ORDINAL_LOSS}"
echo "  ORDINAL_LOSS_ALPHA   = ${ORDINAL_LOSS_ALPHA}"
echo "  EVAL_SPLITS          = ${EVAL_SPLITS}"
echo "  METHOD_SUFFIX        = ${METHOD_SUFFIX}"
echo "============================================================"

# ------------------------------------------------------------------
# Phase 1: Build (generates train.resolved.yaml)
# ------------------------------------------------------------------
echo ""
echo "[method-upgrade] Phase 1: Building evidence traces..."
RUN_TRAIN=false \
RUN_INFER=false \
RUN_LABEL_TOKEN_INFER=false \
  bash "${SCRIPT_DIR}/run_llama31_liar_raw.sh"

# Locate the generated config
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_trace_verifier/liar_raw_v0_6c_method_upgrade}"
CASE_NAME="${CASE_NAME:-v0_6c_liar6_rule_step_adaptive5_10_llama31_8b_fullft${METHOD_SUFFIX}}"
CONFIG_PATH="${OUTPUT_ROOT}/${CASE_NAME}/train.resolved.yaml"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: config not found after build: ${CONFIG_PATH}" >&2
  exit 1
fi

# ------------------------------------------------------------------
# Phase 2: Patch config
# ------------------------------------------------------------------
echo ""
echo "[method-upgrade] Phase 2: Patching config..."

PATCH_ARGS=(--config "${CONFIG_PATH}")

if [[ -n "${WEIGHT_DECAY}" ]]; then
  PATCH_ARGS+=(--weight-decay "${WEIGHT_DECAY}")
fi
if [[ -n "${ALPHA_WARMUP_RATIO}" ]]; then
  PATCH_ARGS+=(--alpha-warmup-ratio "${ALPHA_WARMUP_RATIO}")
fi
if [[ -n "${NUM_CYCLES}" ]]; then
  PATCH_ARGS+=(--lr-scheduler "cosine_with_restarts")
  PATCH_ARGS+=(--lr-kwargs "{\"num_cycles\": ${NUM_CYCLES}}")
fi
if [[ "${USE_CALIBRATED_METRIC}" == "true" ]]; then
  PATCH_ARGS+=(--early-stopping-metric "macro_f1_plus_true_side_plus_mae")
  PATCH_ARGS+=(--mae-metric-weight "${MAE_METRIC_WEIGHT}")
fi

python "${SCRIPT_DIR}/patch_config_for_upgrade.py" "${PATCH_ARGS[@]}"

# ------------------------------------------------------------------
# Phase 3: Train
# ------------------------------------------------------------------
echo ""
echo "[method-upgrade] Phase 3: Training..."

RUN_BUILD=false \
RUN_TRAIN=true \
RUN_INFER=false \
RUN_LABEL_TOKEN_INFER=false \
  bash "${SCRIPT_DIR}/run_llama31_liar_raw.sh"

# ------------------------------------------------------------------
# Phase 4: Infer (label-token eval)
# ------------------------------------------------------------------
echo ""
echo "[method-upgrade] Phase 4: Inference..."

IFS=',' read -r -a EVAL_SPLIT_ARRAY <<< "${EVAL_SPLITS}"
for EVAL_SPLIT in "${EVAL_SPLIT_ARRAY[@]}"; do
  EVAL_SPLIT="${EVAL_SPLIT//[[:space:]]/}"
  if [[ -z "${EVAL_SPLIT}" ]]; then
    continue
  fi
  echo "[method-upgrade] Phase 4: Inference split=${EVAL_SPLIT}"
  RUN_BUILD=false \
  RUN_TRAIN=false \
  RUN_INFER=false \
  RUN_LABEL_TOKEN_INFER=true \
  INFER_SPLIT="${EVAL_SPLIT}" \
    bash "${SCRIPT_DIR}/run_llama31_liar_raw.sh"
done

echo ""
echo "[method-upgrade] Done. Results in ${OUTPUT_ROOT}/${CASE_NAME}/eval/"
