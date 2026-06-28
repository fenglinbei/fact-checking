#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ATOM_ANCHOR_ROOT="${ATOM_ANCHOR_ROOT:-outputs/selectors/atom_anchor/rawfc_abc_v0_1_baseline20}"
export BASELINE_TOP_K="${BASELINE_TOP_K:-20}"
export SELECTOR_TOP_K="${SELECTOR_TOP_K:-5}"

bash "${SCRIPT_DIR}/run_rawfc_atom_anchor_abc_v0_1_all_splits.sh"
