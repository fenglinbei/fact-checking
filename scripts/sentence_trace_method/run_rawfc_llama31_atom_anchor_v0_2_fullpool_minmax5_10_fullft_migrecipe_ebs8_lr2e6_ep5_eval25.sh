#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc/bin/python" ]]; then
    export PYTHON_BIN="/data/liaozijie/conda/accelerate-fc/bin/python"
  else
    export PYTHON_BIN="python"
  fi
fi

# FullFT on baseline5 atom-anchor evidence, training recipe aligned to the
# phase7 backbone-migration run (lr 2e-6, ep5, cosine, wd 0, eval25,
# ordinal_loss off, ZeRO-3 bsz2 ga4).
export FINETUNE_MODE="${FINETUNE_MODE:-fullft}"
export MREC_POLICY_CONFIG="${MREC_POLICY_CONFIG:-configs/experiment/mrec_v0.2/rawfc_llama31_learned_marginal_proxy_fullpool_minmax5_10_fullft_migrecipe.yaml}"
export REQUIRE_PROMPT_INPUT_IDS="${REQUIRE_PROMPT_INPUT_IDS:-false}"

bash "${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh"
