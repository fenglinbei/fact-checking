#!/usr/bin/env bash
# LIAR-RAW main method with Llama-3.1-8B backbone (FullFT migrecipe).
# Applies the same optimized training recipe used on RAWFC (lr 2e-6, 5 epochs,
# plain cosine, weight_decay 0.0, eval_steps 25, ordinal_loss disabled,
# ZeRO-3 bsz2 ga4) to LIAR-RAW 6-way classification.
#
# Usage:
#   bash scripts/sentence_trace_method/run_liar_raw_llama31_atom_anchor_v0_2_fullpool_minmax5_10_fullft_migrecipe_ebs8_lr2e6_ep5_eval25.sh
set -euo pipefail

export MREC_POLICY_CONFIG="${MREC_POLICY_CONFIG:-configs/experiment/mrec_v0.2/liar_raw_llama31_learned_marginal_proxy_fullpool_minmax5_10_fullft_migrecipe.yaml}"
export FINETUNE_MODE="${FINETUNE_MODE:-fullft}"
export REQUIRE_PROMPT_INPUT_IDS="${REQUIRE_PROMPT_INPUT_IDS:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh"
