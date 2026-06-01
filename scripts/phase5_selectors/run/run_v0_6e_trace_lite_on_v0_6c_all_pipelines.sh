#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export TRACE_PROMPT_STYLE=trace_lite
export LORA_CASE_NAME=v0_6e_trace_lite_on_v0_6c
export FULLFT_CASE_NAME=v0_6e_trace_lite_on_v0_6c_fullft

export SOURCE_TYPE="${SOURCE_TYPE:-trace}"
export TRACE_SELECTION_MODE="${TRACE_SELECTION_MODE:-trace}"
export MAX_TOP_K="${MAX_TOP_K:-10}"
export EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-v0_6c_rule_step_adaptive5_10}"
export TRAIN_TRACE="${TRAIN_TRACE:-outputs/selectors/evidence_chain_graph/v0_6c_adaptive5_10_train/selection_trace_train.jsonl}"
export VAL_TRACE="${VAL_TRACE:-outputs/selectors/evidence_chain_graph/v0_6c_adaptive5_10_val/selection_trace_val.jsonl}"

echo "[v0.6e-trace-lite-on-v0.6c] lora case  : ${LORA_CASE_NAME}"
echo "[v0.6e-trace-lite-on-v0.6c] fullft case: ${FULLFT_CASE_NAME}"
echo "[v0.6e-trace-lite-on-v0.6c] train trace : ${TRAIN_TRACE}"
echo "[v0.6e-trace-lite-on-v0.6c] val trace   : ${VAL_TRACE}"

exec bash scripts/phase5_selectors/run/run_v0_6c_rule_step_adaptive5_10_all_pipelines.sh
