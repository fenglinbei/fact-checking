#!/usr/bin/env bash
# v0.2 diagnostic: utility-only pairwise ranking with text-only candidate features.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/utility_listwise/deberta_v0_2_rank_only_text}"
export FEATURE_ABLATION="${FEATURE_ABLATION:-text_only}"
export USE_RANK_EMBEDDING="${USE_RANK_EMBEDDING:-false}"
export FREEZE_PAIR_ENCODER="${FREEZE_PAIR_ENCODER:-true}"
export SHUFFLE_PROBABILITY="${SHUFFLE_PROBABILITY:-1.0}"
export PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-1.0}"
export SOFT_CE_WEIGHT="${SOFT_CE_WEIGHT:-0.0}"
export BCE_WEIGHT="${BCE_WEIGHT:-0.0}"
export EARLY_STOPPING_METRIC="${EARLY_STOPPING_METRIC:-jaccard@5}"
export SELECTOR_NAME="${SELECTOR_NAME:-utility_listwise_v0_2_rank_only_text}"

exec bash "${SCRIPT_DIR}/run_utility_listwise_v0.sh" "$@"
