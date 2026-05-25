#!/usr/bin/env bash
# Phase 3 utility-ranking smoke for the Qwen LLM action selector.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/llm_action_selector/qwen25_3b_utility_pairwise_v1_smoke}"
export TARGET_MODE="${TARGET_MODE:-utility}"
export BUILD_BAD_PREFIX_DATA="${BUILD_BAD_PREFIX_DATA:-false}"
export HARD_LOSS_WEIGHT="${HARD_LOSS_WEIGHT:-0.2}"
export PAIRWISE_LOSS_WEIGHT="${PAIRWISE_LOSS_WEIGHT:-0.2}"
export SOFT_LOSS_WEIGHT="${SOFT_LOSS_WEIGHT:-0.05}"
export SOFT_TAU="${SOFT_TAU:-0.3}"
export SET_LOSS_WEIGHT="${SET_LOSS_WEIGHT:-0.02}"
export BEST_SELECTION_METRIC="${BEST_SELECTION_METRIC:-oracle_rank_ndcg@5}"
export TRAIN_SAMPLE_LIMIT="${TRAIN_SAMPLE_LIMIT:-2048}"
export VAL_SAMPLE_LIMIT="${VAL_SAMPLE_LIMIT:-1024}"
export EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT:-512}"
export EPOCHS="${EPOCHS:-10}"
export SWANLAB_TAGS="${SWANLAB_TAGS:-selector,llm_action,utility_pairwise}"

export RUN_TRAIN="${RUN_TRAIN:-true}"
export RUN_EVAL="${RUN_EVAL:-${RUN_TRAIN}}"

exec bash "${SCRIPT_DIR}/run_llm_action_selector_vig_soft.sh"
