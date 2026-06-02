#!/usr/bin/env bash
set -euo pipefail

export RUN_LORA="${RUN_LORA:-false}"
export RUN_FULLFT="${RUN_FULLFT:-true}"
export FULLFT_EXPERIMENT="${FULLFT_EXPERIMENT:-v0_6c_rawfc3_rule_step_adaptive5_10_eval25_fullft_half12_1024}"
export FULLFT_CONFIG="${FULLFT_CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_eval25_fullft_half12_1024.yaml}"
export FULLFT_DEEPSPEED_CONFIG="${FULLFT_DEEPSPEED_CONFIG:-configs/deepspeed_zero3_bsz2_ga4.json}"

exec bash scripts/phase5_selectors/run/run_rawfc_v0_6c_rule_step_adaptive5_10_eval25_all_pipelines.sh
