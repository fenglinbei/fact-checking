#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_GRAPH_BUILD="${RUN_GRAPH_BUILD:-true}"
RUN_LORA="${RUN_LORA:-true}"
RUN_FULLFT="${RUN_FULLFT:-true}"

CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
MIN_TOP_K="${MIN_TOP_K:-5}"
MAX_TOP_K="${MAX_TOP_K:-10}"
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-plain}"

TRAIN_TRACE="${TRAIN_TRACE:-outputs/selectors/evidence_chain_graph/v0_6c_adaptive5_10_train/selection_trace_train.jsonl}"
VAL_TRACE="${VAL_TRACE:-outputs/selectors/evidence_chain_graph/v0_6c_adaptive5_10_val/selection_trace_val.jsonl}"
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-v0_6c_rule_step_adaptive5_10}"

LORA_CASE_NAME="${LORA_CASE_NAME:-v0_6c_rule_step_adaptive5_10}"
FULLFT_CASE_NAME="${FULLFT_CASE_NAME:-v0_6c_rule_step_adaptive5_10_fullft}"
LORA_CONFIG="${LORA_CONFIG:-configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml}"
FULLFT_CONFIG="${FULLFT_CONFIG:-configs/experiment/b3_oracle_sentence_direct_verifier_1024_fullft.yaml}"
FULLFT_DEEPSPEED_CONFIG="${FULLFT_DEEPSPEED_CONFIG:-configs/deepspeed_zero3_bsz2_ga4.json}"
LORA_INFER_EXPERIMENT="${LORA_INFER_EXPERIMENT:-b3_oracle_sentence_direct_verifier_1024}"
FULLFT_INFER_EXPERIMENT="${FULLFT_INFER_EXPERIMENT:-b3_oracle_sentence_direct_verifier_1024_fullft}"

echo "[v0.6c-all] graph build : ${RUN_GRAPH_BUILD}"
echo "[v0.6c-all] lora/fullft : ${RUN_LORA}/${RUN_FULLFT}"
echo "[v0.6c-all] top_n/min/max: ${CANDIDATE_TOP_N}/${MIN_TOP_K}/${MAX_TOP_K}"
echo "[v0.6c-all] trace prompt: ${TRACE_PROMPT_STYLE}"
echo "[v0.6c-all] train trace : ${TRAIN_TRACE}"
echo "[v0.6c-all] val trace   : ${VAL_TRACE}"

if [[ "${RUN_GRAPH_BUILD}" == "true" ]]; then
  SPLIT=train \
  CANDIDATE_TOP_N="${CANDIDATE_TOP_N}" \
  MIN_TOP_K="${MIN_TOP_K}" \
  MAX_TOP_K="${MAX_TOP_K}" \
  CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
  SAMPLE_LIMIT="${SAMPLE_LIMIT}" \
  bash scripts/phase5_selectors/run/run_evidence_chain_graph_v0_6c.sh

  SPLIT=val \
  CANDIDATE_TOP_N="${CANDIDATE_TOP_N}" \
  MIN_TOP_K="${MIN_TOP_K}" \
  MAX_TOP_K="${MAX_TOP_K}" \
  CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
  SAMPLE_LIMIT="${SAMPLE_LIMIT}" \
  bash scripts/phase5_selectors/run/run_evidence_chain_graph_v0_6c.sh
fi

if [[ "${RUN_LORA}" == "true" ]]; then
  echo "[v0.6c-all] running LoRA full pipeline: ${LORA_CASE_NAME}"
  CONFIG="${LORA_CONFIG}" \
  INFER_EXPERIMENT="${LORA_INFER_EXPERIMENT}" \
  CASE_NAME="${LORA_CASE_NAME}" \
  SOURCE_TYPE=trace \
  TRAIN_SOURCE="${TRAIN_TRACE}" \
  VAL_SOURCE="${VAL_TRACE}" \
  TRACE_SELECTION_MODE=trace \
  TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE}" \
  TOP_K="${MAX_TOP_K}" \
  EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME}" \
  EXPECTED_CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
  bash scripts/phase5_selectors/run/run_selector_trace_full_pipeline.sh
fi

if [[ "${RUN_FULLFT}" == "true" ]]; then
  echo "[v0.6c-all] running FullFT full pipeline: ${FULLFT_CASE_NAME}"
  CONFIG="${FULLFT_CONFIG}" \
  INFER_EXPERIMENT="${FULLFT_INFER_EXPERIMENT}" \
  CASE_NAME="${FULLFT_CASE_NAME}" \
  SOURCE_TYPE=trace \
  TRAIN_SOURCE="${TRAIN_TRACE}" \
  VAL_SOURCE="${VAL_TRACE}" \
  TRACE_SELECTION_MODE=trace \
  TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE}" \
  TOP_K="${MAX_TOP_K}" \
  EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME}" \
  EXPECTED_CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
  DEEPSPEED_CONFIG="${FULLFT_DEEPSPEED_CONFIG}" \
  MERGE_LORA_CACHE=false \
  FINETUNE_MODE=full-parameter \
  bash scripts/phase5_selectors/run/run_selector_trace_full_pipeline.sh
fi

echo "[v0.6c-all] done"
