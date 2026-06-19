#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCRIPT_DIR="${ROOT_DIR}/scripts/sentence_trace_method"

export PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
export LORA_SUFFIX="${LORA_SUFFIX:-_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
export MODE="${AA_QEC_STAGE3_MODE:-full}"
export RUN_TAU_EVAL="${AA_QEC_STAGE3_RUN_TAU_EVAL:-auto}"
export EVAL_SPLITS="${EVAL_SPLITS:-val,test}"
export AA_QEC_STAGE3_CASES="${AA_QEC_STAGE3_CASES:-F1,F2,F3}"
export SOURCE_SELECTOR_NAME="${SOURCE_SELECTOR_NAME:-v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10}"
export FORCE_AA_QEC_BUILD="${FORCE_AA_QEC_BUILD:-false}"
export FORCE_STAGE="${FORCE_STAGE:-false}"
export FORCE_ATOM_FACTS_ABC_STAGE="${FORCE_ATOM_FACTS_ABC_STAGE:-false}"
export PREPARE_LIAR_RAW_ATOM_FACTS_ABC_SOURCES="${PREPARE_LIAR_RAW_ATOM_FACTS_ABC_SOURCES:-true}"
export PREPARE_AA_QEC_SOURCES="${PREPARE_AA_QEC_SOURCES:-true}"
RUN_STAGE3_BUILD_GATE="${RUN_STAGE3_BUILD_GATE:-true}"
STAGE3_BUILD_GATE_REPORT_PATH="${STAGE3_BUILD_GATE_REPORT_PATH:-${OUTPUT_ROOT}/aa_qec_stage3_build_gate_report.json}"
FULL_PRECHECK_BUILD_GATE="$RUN_STAGE3_BUILD_GATE"

run_cmd() {
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ "$FULL_PRECHECK_BUILD_GATE" == "true" || "$FULL_PRECHECK_BUILD_GATE" == "1" ]]; then
  run_cmd "$PYTHON_BIN" scripts/sentence_trace_method/check_aa_qec_stage3_build_gate.py \
    --output-root "$OUTPUT_ROOT" \
    --graph-root outputs/selectors/atom_anchored_qec/liar_raw \
    --source-selector-name "$SOURCE_SELECTOR_NAME" \
    --model ministral3_8b \
    --lora-suffix "$LORA_SUFFIX" \
    --cases "$AA_QEC_STAGE3_CASES" \
    --splits train,val,test \
    --prompt-splits train,val,test \
    --report-path "$STAGE3_BUILD_GATE_REPORT_PATH"
elif [[ "$FULL_PRECHECK_BUILD_GATE" != "false" && "$FULL_PRECHECK_BUILD_GATE" != "0" ]]; then
  printf 'Unsupported RUN_STAGE3_BUILD_GATE=%s. Use true or false.\n' "$FULL_PRECHECK_BUILD_GATE" >&2
  exit 2
fi

export RUN_STAGE3_BUILD_GATE=false

printf '[aa-qec-stage3-f1-f3-full] AA_QEC_STAGE3_CASES=%s MODE=%s EVAL_SPLITS=%s RUN_TAU_EVAL=%s RUN_STAGE3_BUILD_GATE=%s FORCE_AA_QEC_BUILD=%s FORCE_STAGE=%s FORCE_ATOM_FACTS_ABC_STAGE=%s PREPARE_AA_QEC_SOURCES=%s\n' \
  "$AA_QEC_STAGE3_CASES" "$MODE" "$EVAL_SPLITS" "$RUN_TAU_EVAL" "$FULL_PRECHECK_BUILD_GATE" "$FORCE_AA_QEC_BUILD" "$FORCE_STAGE" "$FORCE_ATOM_FACTS_ABC_STAGE" "$PREPARE_AA_QEC_SOURCES"

bash "${SCRIPT_DIR}/run_aa_qec_stage3_liar_raw_atom_facts_abc_ministral3.sh"
