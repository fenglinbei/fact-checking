#!/usr/bin/env bash
# Step4.1-A: train the deep sequential pointer selector with selected-mask BCE.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/stage2_sentence_sequential/deberta_sequential_deep_mask02}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/eval_val}"
export SEQ_LOSS_WEIGHT="${SEQ_LOSS_WEIGHT:-1.0}"
export MASK_LOSS_WEIGHT="${MASK_LOSS_WEIGHT:-0.2}"
export SEMANTIC_FEATURE_PROFILE="${SEMANTIC_FEATURE_PROFILE:-deep}"
export TARGETED_FEATURE_PROFILE="${TARGETED_FEATURE_PROFILE:-none}"
export SHALLOW_FEATURE_PROFILE="${SHALLOW_FEATURE_PROFILE:-off}"
export SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-$(basename "${OUTPUT_DIR}")}"
export SWANLAB_TAGS="${SWANLAB_TAGS:-selector,sequential,step4.1,mask_bce}"
export SWANLAB_DESCRIPTION="${SWANLAB_DESCRIPTION:-Step4.1-A deep sequential pointer selector with selected-mask BCE.}"

exec "${SCRIPT_DIR}/run_sequential_step4.sh" "$@"
