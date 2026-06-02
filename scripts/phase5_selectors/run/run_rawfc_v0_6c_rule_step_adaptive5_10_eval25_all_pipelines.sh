#!/usr/bin/env bash
set -euo pipefail

export LORA_EXPERIMENT="${LORA_EXPERIMENT:-v0_6c_rawfc3_rule_step_adaptive5_10_eval25}"
export FULLFT_EXPERIMENT="${FULLFT_EXPERIMENT:-v0_6c_rawfc3_rule_step_adaptive5_10_eval25_fullft}"
export LORA_CONFIG="${LORA_CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_eval25.yaml}"
export FULLFT_CONFIG="${FULLFT_CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_eval25_fullft.yaml}"
MIN_TOP_K="${MIN_TOP_K:-5}"
MAX_TOP_K="${MAX_TOP_K:-10}"
GRAPH_BUDGET_SLUG="adaptive${MIN_TOP_K}_${MAX_TOP_K}"
TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-plain}"
export MIN_TOP_K
export MAX_TOP_K
export TRACE_PROMPT_STYLE

infer_case_suffix() {
  local config_path="$1"
  local prefix="$2"
  local stem
  local suffix
  stem="$(basename "${config_path}")"
  stem="${stem%.yaml}"
  suffix="${stem#"${prefix}"}"
  if [[ "${suffix}" == "${stem}" ]]; then
    suffix=""
  fi
  printf '%s' "${suffix}"
}

LORA_CASE_TUNING_SUFFIX="${LORA_CASE_TUNING_SUFFIX:-$(infer_case_suffix "${LORA_CONFIG}" "v0_6c_rawfc3_rule_step_adaptive5_10_eval25")}"
FULLFT_CASE_TUNING_SUFFIX="${FULLFT_CASE_TUNING_SUFFIX:-$(infer_case_suffix "${FULLFT_CONFIG}" "v0_6c_rawfc3_rule_step_adaptive5_10_eval25_fullft")}"
TRACE_CASE_SUFFIX=""
if [[ "${TRACE_PROMPT_STYLE}" != "plain" ]]; then
  TRACE_CASE_SUFFIX="_${TRACE_PROMPT_STYLE}"
fi

export LORA_CASE_NAME="${LORA_CASE_NAME:-v0_6c_rawfc3_rule_step_${GRAPH_BUDGET_SLUG}_eval25${LORA_CASE_TUNING_SUFFIX}${TRACE_CASE_SUFFIX}}"
export FULLFT_CASE_NAME="${FULLFT_CASE_NAME:-v0_6c_rawfc3_rule_step_${GRAPH_BUDGET_SLUG}_eval25_fullft${FULLFT_CASE_TUNING_SUFFIX}${TRACE_CASE_SUFFIX}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_trace_verifier/rawfc_v0_6c_eval25}"
export RUN_ROOT="${RUN_ROOT:-outputs/runs/rawfc_v0_6c_eval25_selector_trace_full_pipeline}"
export RUN_LABEL_TOKEN_INFER="${RUN_LABEL_TOKEN_INFER:-true}"
export RUN_API_INFER="${RUN_API_INFER:-false}"

exec bash scripts/phase5_selectors/run/run_rawfc_v0_6c_rule_step_adaptive5_10_all_pipelines.sh
