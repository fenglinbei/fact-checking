#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCRIPT_DIR="${ROOT_DIR}/scripts/sentence_trace_method"

export MODE="${AA_QEC_STAGE2_MODE:-full}"
export RUN_TAU_EVAL="${AA_QEC_STAGE2_RUN_TAU_EVAL:-auto}"
export AA_QEC_STAGE2_CASES="${AA_QEC_STAGE2_CASES:-C3,C4}"
export FORCE_AA_QEC_BUILD="${FORCE_AA_QEC_BUILD:-false}"
export FORCE_STAGE="${FORCE_STAGE:-false}"
export FORCE_ATOM_FACTS_ABC_STAGE="${FORCE_ATOM_FACTS_ABC_STAGE:-false}"
export PREPARE_LIAR_RAW_ATOM_FACTS_ABC_SOURCES="${PREPARE_LIAR_RAW_ATOM_FACTS_ABC_SOURCES:-true}"
export PREPARE_AA_QEC_SOURCES="${PREPARE_AA_QEC_SOURCES:-true}"

printf '[aa-qec-stage2-c3-c4-full] AA_QEC_STAGE2_CASES=%s MODE=%s RUN_TAU_EVAL=%s FORCE_AA_QEC_BUILD=%s FORCE_STAGE=%s FORCE_ATOM_FACTS_ABC_STAGE=%s PREPARE_AA_QEC_SOURCES=%s\n' \
  "$AA_QEC_STAGE2_CASES" "$MODE" "$RUN_TAU_EVAL" "$FORCE_AA_QEC_BUILD" "$FORCE_STAGE" "$FORCE_ATOM_FACTS_ABC_STAGE" "$PREPARE_AA_QEC_SOURCES"

bash "${SCRIPT_DIR}/run_aa_qec_stage2_liar_raw_atom_facts_abc_ministral3.sh"
