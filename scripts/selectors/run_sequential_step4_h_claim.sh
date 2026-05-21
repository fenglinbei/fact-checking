#!/usr/bin/env bash
# Train the Step4 strategy-2 sequential selector with per-step claim-only h_claim features.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/stage2_sentence_sequential/deberta_sequential_deep_hclaim}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/eval_val}"
export PROJECTION_MODE="${PROJECTION_MODE:-linear}"
export CLAIM_START_MODE="${CLAIM_START_MODE:-learned}"
export CLAIM_FEATURE_MODE="${CLAIM_FEATURE_MODE:-claim_only}"
export SEQ_LOSS_WEIGHT="${SEQ_LOSS_WEIGHT:-1.0}"
export MASK_LOSS_WEIGHT="${MASK_LOSS_WEIGHT:-0.0}"
export SEMANTIC_FEATURE_PROFILE="${SEMANTIC_FEATURE_PROFILE:-deep}"
export TARGETED_FEATURE_PROFILE="${TARGETED_FEATURE_PROFILE:-none}"
export SHALLOW_FEATURE_PROFILE="${SHALLOW_FEATURE_PROFILE:-off}"
export SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-$(basename "${OUTPUT_DIR}")}"
export SWANLAB_TAGS="${SWANLAB_TAGS:-selector,sequential,step4,h_claim}"
export SWANLAB_DESCRIPTION="${SWANLAB_DESCRIPTION:-Step4 strategy-2 deep sequential pointer selector with per-step claim-only h_claim feature.}"

exec "${SCRIPT_DIR}/run_sequential_step4.sh" "$@"
