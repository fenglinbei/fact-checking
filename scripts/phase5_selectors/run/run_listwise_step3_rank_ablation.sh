#!/usr/bin/env bash
# Closing Step3 rank-prior ablation: keep hybrid_score as the only retrieval prior.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/stage2_sentence_listwise/deberta_listwise_rank_ablation}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/eval_val}"
export FILTER_POLICY="${FILTER_POLICY:-all}"
export SHUFFLE_PROBABILITY="${SHUFFLE_PROBABILITY:-0.3}"
export FEATURE_ABLATION="${FEATURE_ABLATION:-hybrid_score_only_prior}"

exec "${SCRIPT_DIR}/run_listwise_step3.sh" "$@"
