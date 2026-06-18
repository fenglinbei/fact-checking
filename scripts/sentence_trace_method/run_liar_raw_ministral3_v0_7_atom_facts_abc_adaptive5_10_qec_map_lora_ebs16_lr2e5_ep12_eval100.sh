#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-qec_map}"
export CASE_SUFFIX="${CASE_SUFFIX:-__v0_7_atom_facts_abc_bm_adaptive5_10__qec_map}"

exec bash "${SCRIPT_DIR}/run_liar_raw_ministral3_v0_7_atom_facts_abc_adaptive5_10_qec_min_lora_ebs16_lr2e5_ep12_eval100.sh"
