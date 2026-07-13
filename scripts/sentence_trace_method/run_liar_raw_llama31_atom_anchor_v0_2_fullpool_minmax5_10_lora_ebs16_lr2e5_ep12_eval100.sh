#!/usr/bin/env bash
# LIAR-RAW main method with Llama-3.1-8B backbone (LoRA).
# Same learned marginal proxy selector + minmax(5,10) as the Ministral main
# method, only swaps the verifier backbone to Meta-Llama-3.1-8B-Instruct.
#
# Usage:
#   bash scripts/sentence_trace_method/run_liar_raw_llama31_atom_anchor_v0_2_fullpool_minmax5_10_lora_ebs16_lr2e5_ep12_eval100.sh
set -euo pipefail

export MREC_POLICY_CONFIG="${MREC_POLICY_CONFIG:-configs/experiment/mrec_v0.2/liar_raw_llama31_learned_marginal_proxy_fullpool_minmax5_10_lora.yaml}"

# Llama tokenizer is not a MistralCommon tokenizer, so build rows do not carry
# prompt_input_ids. Disable the strict check that the Ministral pipeline enables.
export REQUIRE_PROMPT_INPUT_IDS="${REQUIRE_PROMPT_INPUT_IDS:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh"
