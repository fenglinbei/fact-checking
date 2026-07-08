#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python" ]]; then
    export PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python"
  else
    export PYTHON_BIN="python"
  fi
fi

# LoRA on Ministral-3-8B over baseline20 atom-anchor evidence, training recipe
# adapted from the phase7 migration recipe: lr 1e-5 (raised from 2e-6 for LoRA),
# plain cosine (no restarts), weight_decay 0.0, ordinal_loss off, eval_steps 25,
# extended to 12 epochs / patience 12 so the adapter can converge under the
# gentler cosine schedule. Direct LoRA counterpart of the ministral3 fullft
# migrecipe run.
export FINETUNE_MODE="${FINETUNE_MODE:-lora}"
export MREC_POLICY_CONFIG="${MREC_POLICY_CONFIG:-configs/experiment/mrec_v0.2/rawfc_learned_marginal_proxy_fullpool_minmax5_10_baseline20_lora_migrecipe.yaml}"
export REQUIRE_PROMPT_INPUT_IDS="${REQUIRE_PROMPT_INPUT_IDS:-false}"

bash "${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh"
