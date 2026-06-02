#!/usr/bin/env bash
set -euo pipefail

export MIN_TOP_K="${MIN_TOP_K:-3}"
export MAX_TOP_K="${MAX_TOP_K:-8}"
export RUN_LORA="${RUN_LORA:-false}"
export RUN_FULLFT="${RUN_FULLFT:-true}"
export FULLFT_EXPERIMENT="${FULLFT_EXPERIMENT:-v0_6c_rawfc3_rule_step_adaptive3_8_eval25_fullft_maxlen1536_bsz1_ga8}"
export FULLFT_CONFIG="${FULLFT_CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive3_8_eval25_fullft_maxlen1536_bsz1_ga8.yaml}"
export FULLFT_CASE_TUNING_SUFFIX="${FULLFT_CASE_TUNING_SUFFIX:-_maxlen1536_bsz1_ga8}"
export FULLFT_DEEPSPEED_CONFIG="${FULLFT_DEEPSPEED_CONFIG:-configs/deepspeed_zero3.json}"

exec bash scripts/phase5_selectors/run/run_rawfc_v0_6c_rule_step_adaptive5_10_eval25_all_pipelines.sh
