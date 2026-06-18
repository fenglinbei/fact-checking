#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QD_UNION_POOL_FILE="${QD_UNION_POOL_FILE:-outputs/selectors/question_decomp_retrieval/qwen_v0_${SPLIT}/union_candidate_pool_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/evidence_map_selector/v0_7_atom_facts_${SPLIT}}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
TOP_K="${TOP_K:-5}"
SELECTOR_BASE_WEIGHT="${SELECTOR_BASE_WEIGHT:-}"
SELECTOR_ATOM_COVERAGE_WEIGHT="${SELECTOR_ATOM_COVERAGE_WEIGHT:-}"
SELECTOR_DIRECTNESS_WEIGHT="${SELECTOR_DIRECTNESS_WEIGHT:-}"
SELECTOR_POLAR_RELATION_WEIGHT="${SELECTOR_POLAR_RELATION_WEIGHT:-}"
SELECTOR_DUPLICATE_PENALTY="${SELECTOR_DUPLICATE_PENALTY:-}"
SELECTOR_SOURCE_PENALTY="${SELECTOR_SOURCE_PENALTY:-}"
SELECTOR_BACKGROUND_PENALTY="${SELECTOR_BACKGROUND_PENALTY:-}"
RUN_TEACHER="${RUN_TEACHER:-false}"
MOCK_EVIDENCE_MAPS="${MOCK_EVIDENCE_MAPS:-false}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://api.deepseek.com}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
CONCURRENCY="${CONCURRENCY:-128}"
REQUESTS_PER_MINUTE="${REQUESTS_PER_MINUTE:-60000}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
THINKING_TYPE="${THINKING_TYPE:-disabled}"
PROMPT_VERSION="${PROMPT_VERSION:-evidence_map_v0_7_atom_facts}"
MAX_EVIDENCE_CHARS="${MAX_EVIDENCE_CHARS:-700}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
CASE_IDS="${CASE_IDS:-}"
BUILD_VERIFIER_DATA="${BUILD_VERIFIER_DATA:-false}"
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

if [[ -z "${ORACLE_RESULTS+x}" ]]; then
  if [[ "${SPLIT}" == "train" ]]; then
    ORACLE_RESULTS="outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl"
  elif [[ "${SPLIT}" == "val" ]]; then
    ORACLE_RESULTS="outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"
  else
    ORACLE_RESULTS=""
  fi
fi

SAMPLE_ARGS=()
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  SAMPLE_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

SELECTOR_ARGS=()
if [[ -n "${SELECTOR_BASE_WEIGHT}" ]]; then
  SELECTOR_ARGS+=(--selector-base-weight "${SELECTOR_BASE_WEIGHT}")
fi
if [[ -n "${SELECTOR_ATOM_COVERAGE_WEIGHT}" ]]; then
  SELECTOR_ARGS+=(--selector-atom-coverage-weight "${SELECTOR_ATOM_COVERAGE_WEIGHT}")
fi
if [[ -n "${SELECTOR_DIRECTNESS_WEIGHT}" ]]; then
  SELECTOR_ARGS+=(--selector-directness-weight "${SELECTOR_DIRECTNESS_WEIGHT}")
fi
if [[ -n "${SELECTOR_POLAR_RELATION_WEIGHT}" ]]; then
  SELECTOR_ARGS+=(--selector-polar-relation-weight "${SELECTOR_POLAR_RELATION_WEIGHT}")
fi
if [[ -n "${SELECTOR_DUPLICATE_PENALTY}" ]]; then
  SELECTOR_ARGS+=(--selector-duplicate-penalty "${SELECTOR_DUPLICATE_PENALTY}")
fi
if [[ -n "${SELECTOR_SOURCE_PENALTY}" ]]; then
  SELECTOR_ARGS+=(--selector-source-penalty "${SELECTOR_SOURCE_PENALTY}")
fi
if [[ -n "${SELECTOR_BACKGROUND_PENALTY}" ]]; then
  SELECTOR_ARGS+=(--selector-background-penalty "${SELECTOR_BACKGROUND_PENALTY}")
fi

ORACLE_ARGS=()
if [[ -n "${ORACLE_RESULTS}" ]]; then
  ORACLE_ARGS=(--oracle-results "${ORACLE_RESULTS}")
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

echo "[evidence-map-v0.7-atom-facts] split       : ${SPLIT}"
echo "[evidence-map-v0.7-atom-facts] qd union    : ${QD_UNION_POOL_FILE}"
echo "[evidence-map-v0.7-atom-facts] oracle meta : ${ORACLE_RESULTS:-none}"
echo "[evidence-map-v0.7-atom-facts] output      : ${OUTPUT_DIR}"
echo "[evidence-map-v0.7-atom-facts] top_n/top_k : ${CANDIDATE_TOP_N}/${TOP_K}"
echo "[evidence-map-v0.7-atom-facts] prompt      : ${PROMPT_VERSION}"
echo "[evidence-map-v0.7-atom-facts] max chars   : ${MAX_EVIDENCE_CHARS}"
echo "[evidence-map-v0.7-atom-facts] selector    : directness=${SELECTOR_DIRECTNESS_WEIGHT:-default} background=${SELECTOR_BACKGROUND_PENALTY:-default}"
echo "[evidence-map-v0.7-atom-facts] run teacher : ${RUN_TEACHER}"
echo "[evidence-map-v0.7-atom-facts] model       : ${TEACHER_MODEL}"
echo "[evidence-map-v0.7-atom-facts] concurrency : ${CONCURRENCY}"
echo "[evidence-map-v0.7-atom-facts] rpm         : ${REQUESTS_PER_MINUTE}"
echo "[evidence-map-v0.7-atom-facts] thinking    : ${THINKING_TYPE}"
echo "[evidence-map-v0.7-atom-facts] mock maps   : ${MOCK_EVIDENCE_MAPS}"

PYTHONPATH=src "$PYTHON_BIN" scripts/phase5_selectors/build/prepare_evidence_map_candidate_pool.py \
  --input-candidate-file "${QD_UNION_POOL_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --candidate-source qd_union \
  --candidate-top-n "${CANDIDATE_TOP_N}" \
  "${ORACLE_ARGS[@]}" \
  "${SAMPLE_ARGS[@]}"

if [[ "${RUN_TEACHER}" == "true" ]]; then
  PYTHONPATH=src "$PYTHON_BIN" scripts/phase5_selectors/build/annotate_evidence_maps_deepseek.py \
    --candidate-pool "${CANDIDATE_POOL}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --prompt-version "${PROMPT_VERSION}" \
    --max-evidence-chars "${MAX_EVIDENCE_CHARS}" \
    --base-url "${TEACHER_BASE_URL}" \
    --model "${TEACHER_MODEL}" \
    --api-key-env "${TEACHER_API_KEY_ENV}" \
    --concurrency "${CONCURRENCY}" \
    --requests-per-minute "${REQUESTS_PER_MINUTE}" \
    --max-tokens "${MAX_TOKENS}" \
    --thinking-type "${THINKING_TYPE}" \
    "${SAMPLE_ARGS[@]}"
elif [[ "${MOCK_EVIDENCE_MAPS}" == "true" || "${MOCK_EVIDENCE_MAPS}" == "1" ]]; then
  PYTHONPATH=src "$PYTHON_BIN" scripts/phase5_selectors/build/annotate_evidence_maps_deepseek.py \
    --candidate-pool "${CANDIDATE_POOL}" \
    --output-dir "${OUTPUT_DIR}" \
    --split "${SPLIT}" \
    --prompt-version "${PROMPT_VERSION}" \
    --max-evidence-chars "${MAX_EVIDENCE_CHARS}" \
    --model "mock-evidence-map" \
    --thinking-type "${THINKING_TYPE}" \
    --mock-maps \
    "${SAMPLE_ARGS[@]}"
elif [[ ! -s "${ANNOTATIONS}" ]]; then
  echo "[evidence-map-v0.7-atom-facts] missing annotations and RUN_TEACHER/MOCK_EVIDENCE_MAPS are false: ${ANNOTATIONS}" >&2
  exit 1
else
  echo "[evidence-map-v0.7-atom-facts] reusing annotations: ${ANNOTATIONS}"
fi

PYTHONPATH=src "$PYTHON_BIN" scripts/phase5_selectors/build/postprocess_evidence_maps.py \
  --candidate-pool "${CANDIDATE_POOL}" \
  --annotations "${ANNOTATIONS}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  "${SAMPLE_ARGS[@]}"

PYTHONPATH=src "$PYTHON_BIN" scripts/phase5_selectors/eval/eval_evidence_map_selector_v0_5a.py \
  --candidate-features "${FEATURES}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --top-k "${TOP_K}" \
  --case-ids "${CASE_IDS}" \
  "${SELECTOR_ARGS[@]}" \
  "${SAMPLE_ARGS[@]}"

if [[ "${BUILD_VERIFIER_DATA}" == "true" ]]; then
  PYTHONPATH=src "$PYTHON_BIN" scripts/phase5_selectors/build/build_evidence_map_verifier_data.py \
    --selection-trace "${TRACE}" \
    --output-dir "${OUTPUT_DIR}/verifier_data" \
    --split "${SPLIT}" \
    --raw-path "${RAW_PATH}" \
    --config "${CONFIG}" \
    "${PROMPT_MODEL_ARGS[@]}" \
    "${SAMPLE_ARGS[@]}"
fi

echo "[evidence-map-v0.7-atom-facts] features: ${FEATURES}"
echo "[evidence-map-v0.7-atom-facts] atom quality: ${OUTPUT_DIR}/atom_quality_summary.json"
echo "[evidence-map-v0.7-atom-facts] done: ${OUTPUT_DIR}"
