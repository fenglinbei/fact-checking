#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
INPUT_FUSION_FILE="${INPUT_FUSION_FILE:-outputs/selectors/direct_evidence_cross_encoder/v0_4d_${SPLIT}_default_query_fusion/candidate_fusion_scores_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/evidence_map_selector/v0_5a_${SPLIT}}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
TOP_K="${TOP_K:-5}"
RUN_TEACHER="${RUN_TEACHER:-false}"
MOCK_EVIDENCE_MAPS="${MOCK_EVIDENCE_MAPS:-false}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://api.deepseek.com}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
CONCURRENCY="${CONCURRENCY:-4}"
REQUESTS_PER_MINUTE="${REQUESTS_PER_MINUTE:-6000}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
THINKING_TYPE="${THINKING_TYPE:-disabled}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
CASE_IDS="${CASE_IDS:-}"
BUILD_VERIFIER_DATA="${BUILD_VERIFIER_DATA:-true}"
CONFIG="${CONFIG:-configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-}"
PROMPT_MODEL_NAME_OR_PATH="${PROMPT_MODEL_NAME_OR_PATH:-}"
TRAIN_MODEL_NAME_OR_PATH="${TRAIN_MODEL_NAME_OR_PATH:-}"

if [[ "${SPLIT}" == "train" ]]; then
  RAW_PATH="${RAW_PATH:-data/raw/LIAR-RAW/train.json}"
elif [[ "${SPLIT}" == "test" ]]; then
  RAW_PATH="${RAW_PATH:-data/raw/LIAR-RAW/test.json}"
else
  RAW_PATH="${RAW_PATH:-data/raw/LIAR-RAW/val.json}"
fi

SAMPLE_ARGS=()
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  SAMPLE_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

PROMPT_MODEL_ARGS=()
if [[ -n "${PROMPT_MODEL_NAME_OR_PATH}" ]]; then
  PROMPT_MODEL_ARGS+=(--prompt-model-name-or-path "${PROMPT_MODEL_NAME_OR_PATH}")
fi
if [[ -n "${TRAIN_MODEL_NAME_OR_PATH}" ]]; then
  PROMPT_MODEL_ARGS+=(--train-model-name-or-path "${TRAIN_MODEL_NAME_OR_PATH}")
fi
if [[ -n "${MODEL_BASE_PATH}" ]]; then
  PROMPT_MODEL_ARGS+=(--model-base-path "${MODEL_BASE_PATH}")
fi

CANDIDATE_POOL="${OUTPUT_DIR}/evidence_map_candidate_pool_${SPLIT}.jsonl"
ANNOTATIONS="${OUTPUT_DIR}/deepseek_evidence_map_annotations_${SPLIT}.jsonl"
FEATURES="${OUTPUT_DIR}/candidate_evidence_map_features_${SPLIT}.jsonl"
TRACE="${OUTPUT_DIR}/selection_trace_${SPLIT}.jsonl"

echo "[evidence-map-v0.5a] split       : ${SPLIT}"
echo "[evidence-map-v0.5a] input       : ${INPUT_FUSION_FILE}"
echo "[evidence-map-v0.5a] output      : ${OUTPUT_DIR}"
echo "[evidence-map-v0.5a] top_n/top_k : ${CANDIDATE_TOP_N}/${TOP_K}"
echo "[evidence-map-v0.5a] run teacher : ${RUN_TEACHER}"
echo "[evidence-map-v0.5a] mock maps   : ${MOCK_EVIDENCE_MAPS}"

PYTHONPATH=src python scripts/phase5_selectors/build/prepare_evidence_map_candidate_pool.py \
  --input-fusion-file "${INPUT_FUSION_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --candidate-top-n "${CANDIDATE_TOP_N}" \
  "${SAMPLE_ARGS[@]}"

if [[ "${RUN_TEACHER}" == "true" ]]; then
  PYTHONPATH=src python scripts/phase5_selectors/build/annotate_evidence_maps_deepseek.py \
    --candidate-pool "${CANDIDATE_POOL}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --base-url "${TEACHER_BASE_URL}" \
    --model "${TEACHER_MODEL}" \
    --api-key-env "${TEACHER_API_KEY_ENV}" \
    --concurrency "${CONCURRENCY}" \
    --requests-per-minute "${REQUESTS_PER_MINUTE}" \
    --max-tokens "${MAX_TOKENS}" \
    --thinking-type "${THINKING_TYPE}" \
    "${SAMPLE_ARGS[@]}"
elif [[ "${MOCK_EVIDENCE_MAPS}" == "true" || "${MOCK_EVIDENCE_MAPS}" == "1" ]]; then
  PYTHONPATH=src python scripts/phase5_selectors/build/annotate_evidence_maps_deepseek.py \
    --candidate-pool "${CANDIDATE_POOL}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --model "mock-evidence-map" \
    --thinking-type "${THINKING_TYPE}" \
    --mock-maps \
    "${SAMPLE_ARGS[@]}"
elif [[ ! -s "${ANNOTATIONS}" ]]; then
  echo "[evidence-map-v0.5a] missing annotations and RUN_TEACHER/MOCK_EVIDENCE_MAPS are false: ${ANNOTATIONS}" >&2
  exit 1
else
  echo "[evidence-map-v0.5a] reusing annotations: ${ANNOTATIONS}"
fi

PYTHONPATH=src python scripts/phase5_selectors/build/postprocess_evidence_maps.py \
  --candidate-pool "${CANDIDATE_POOL}" \
  --annotations "${ANNOTATIONS}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  "${SAMPLE_ARGS[@]}"

PYTHONPATH=src python scripts/phase5_selectors/eval/eval_evidence_map_selector_v0_5a.py \
  --candidate-features "${FEATURES}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --top-k "${TOP_K}" \
  --case-ids "${CASE_IDS}" \
  "${SAMPLE_ARGS[@]}"

if [[ "${BUILD_VERIFIER_DATA}" == "true" ]]; then
  PYTHONPATH=src python scripts/phase5_selectors/build/build_evidence_map_verifier_data.py \
    --selection-trace "${TRACE}" \
    --output-dir "${OUTPUT_DIR}/verifier_data" \
    --split "${SPLIT}" \
    --raw-path "${RAW_PATH}" \
    --config "${CONFIG}" \
    "${PROMPT_MODEL_ARGS[@]}" \
    "${SAMPLE_ARGS[@]}"
fi

echo "[evidence-map-v0.5a] done: ${OUTPUT_DIR}"
