#!/usr/bin/env bash
# v0.1 diagnostic: frozen encoder with rank/position priors disabled and train-time candidate shuffle.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/utility_listwise/deberta_v0_1_no_rank_shuffle}"
export FEATURE_ABLATION="${FEATURE_ABLATION:-no_rank_prior}"
export USE_RANK_EMBEDDING="${USE_RANK_EMBEDDING:-false}"
export SHUFFLE_PROBABILITY="${SHUFFLE_PROBABILITY:-1.0}"
export SELECTOR_NAME="${SELECTOR_NAME:-utility_listwise_v0_1_no_rank_shuffle}"

exec bash "${SCRIPT_DIR}/run_utility_listwise_v0.sh" "$@"
