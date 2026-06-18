#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CASE_SUFFIX="${CASE_SUFFIX:-__v0_7_atom_facts_abc_tight_anchor_only_bm_adaptive5_10__qec_min}"
export EVIDENCE_TEXT_MODE="${EVIDENCE_TEXT_MODE:-anchor_only}"

bash "${SCRIPT_DIR}/run_rawfc_ministral3_v0_7_atom_facts_abc_tight_adaptive5_10_qec_min_lora_r16a32_d010_lr1e5_ep12_eval50.sh"
