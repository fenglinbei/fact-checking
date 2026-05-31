#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/experiment/b3_oracle_sentence_direct_verifier_1024_fullft.yaml}"
export INFER_EXPERIMENT="${INFER_EXPERIMENT:-b3_oracle_sentence_direct_verifier_1024_fullft}"
export CASE_NAME="${CASE_NAME:-v0_6b_chain_graph_top5_fullft}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero3_bsz2_ga4.json}"
export MERGE_LORA_CACHE="${MERGE_LORA_CACHE:-false}"
export FINETUNE_MODE="${FINETUNE_MODE:-full-parameter}"

exec bash scripts/phase5_selectors/run/run_v0_6b_chain_graph_full_pipeline.sh
