#!/usr/bin/env bash
set -euo pipefail

CASE_NAME="${CASE_NAME:-v0_6b_chain_graph_top5}"
SOURCE_TYPE="${SOURCE_TYPE:-trace}"
TRAIN_SOURCE="${TRAIN_SOURCE:-outputs/selectors/evidence_chain_graph/v0_6b_train/selection_trace_train.jsonl}"
VAL_SOURCE="${VAL_SOURCE:-outputs/selectors/evidence_chain_graph/v0_6b_val/selection_trace_val.jsonl}"
TRACE_SELECTION_MODE="${TRACE_SELECTION_MODE:-trace}"
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-v0_6b_chain_graph_top5}"

export CASE_NAME
export SOURCE_TYPE
export TRAIN_SOURCE
export VAL_SOURCE
export TRACE_SELECTION_MODE
export EXPECTED_SELECTOR_NAME

echo "[v0.6b-full-pipeline] case      : ${CASE_NAME}"
echo "[v0.6b-full-pipeline] train src : ${TRAIN_SOURCE}"
echo "[v0.6b-full-pipeline] val src   : ${VAL_SOURCE}"
echo "[v0.6b-full-pipeline] mode      : ${TRACE_SELECTION_MODE}"

exec bash scripts/phase5_selectors/run/run_selector_trace_full_pipeline.sh
