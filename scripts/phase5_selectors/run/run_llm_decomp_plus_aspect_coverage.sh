#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

ORACLE_RESULTS="${ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
SPLIT="${SPLIT:-val}"

DECOMP_OUTPUT_DIR="${DECOMP_OUTPUT_DIR:-outputs/selectors/aspect_coverage/llm_decomp_plus_qwen25_7b_${SPLIT}}"
COVERAGE_OUTPUT_DIR="${COVERAGE_OUTPUT_DIR:-outputs/selectors/aspect_coverage/llm_decomp_plus_qwen25_7b_${SPLIT}_coverage}"

QWEN_MODEL="${QWEN_MODEL:-/data/models/Qwen2.5-7B-Instruct}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-128}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MIN_SUBCLAIMS="${MIN_SUBCLAIMS:-2}"
MAX_SUBCLAIMS="${MAX_SUBCLAIMS:-5}"
GUIDED_JSON="${GUIDED_JSON:-1}"

ASPECT_ENCODER="${ASPECT_ENCODER:-/data/models/bge-base-en-v1.5}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-128}"
ENCODER_MAX_LENGTH="${ENCODER_MAX_LENGTH:-128}"
DEVICE="${DEVICE:-cuda}"

SAMPLE_LIMIT_ARGS=()
if [[ -n "${SAMPLE_LIMIT:-}" ]]; then
  SAMPLE_LIMIT_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

echo "[llm-decomp] oracle results     : ${ORACLE_RESULTS}"
echo "[llm-decomp] decomp output      : ${DECOMP_OUTPUT_DIR}"
echo "[llm-decomp] coverage output    : ${COVERAGE_OUTPUT_DIR}"
echo "[llm-decomp] qwen model         : ${QWEN_MODEL}"
echo "[llm-decomp] aspect encoder     : ${ASPECT_ENCODER}"

GUIDED_JSON_ARGS=()
if [[ "${GUIDED_JSON}" == "0" || "${GUIDED_JSON}" == "false" || "${GUIDED_JSON}" == "False" ]]; then
  GUIDED_JSON_ARGS=(--no-guided-json)
fi

PYTHONPATH=src python scripts/phase5_selectors/build/generate_llm_claim_decomp_aspects.py \
  --oracle-results "${ORACLE_RESULTS}" \
  --split "${SPLIT}" \
  --output-dir "${DECOMP_OUTPUT_DIR}" \
  --model "${QWEN_MODEL}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  --max-tokens "${MAX_TOKENS}" \
  --min-subclaims "${MIN_SUBCLAIMS}" \
  --max-subclaims "${MAX_SUBCLAIMS}" \
  "${GUIDED_JSON_ARGS[@]}" \
  "${SAMPLE_LIMIT_ARGS[@]}" \
  "$@"

PYTHONPATH=src python scripts/phase5_selectors/eval/analyze_oracle_aspect_coverage.py \
  --oracle-results "${ORACLE_RESULTS}" \
  --split "${SPLIT}" \
  --output-dir "${COVERAGE_OUTPUT_DIR}" \
  --claim-aspects-input "${DECOMP_OUTPUT_DIR}/claim_aspects.jsonl" \
  --model-name "${ASPECT_ENCODER}" \
  --batch-size "${ENCODER_BATCH_SIZE}" \
  --max-length "${ENCODER_MAX_LENGTH}" \
  --device "${DEVICE}" \
  "${SAMPLE_LIMIT_ARGS[@]}"
