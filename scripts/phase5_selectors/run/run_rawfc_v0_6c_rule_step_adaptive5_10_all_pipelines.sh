#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

LORA_EXPERIMENT="${LORA_EXPERIMENT:-v0_6c_rawfc3_rule_step_adaptive5_10}"
FULLFT_EXPERIMENT="${FULLFT_EXPERIMENT:-v0_6c_rawfc3_rule_step_adaptive5_10_fullft}"
LORA_CONFIG="${LORA_CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10.yaml}"
FULLFT_CONFIG="${FULLFT_CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_fullft.yaml}"

RAW_DATASET="${RAW_DATASET:-rawfc}"
LABEL_SCHEMA="${LABEL_SCHEMA:-rawfc3}"
TRAIN_RAW="${TRAIN_RAW:-data/raw/RAWFC/train.json}"
VAL_RAW="${VAL_RAW:-data/raw/RAWFC/val.json}"
TEST_RAW="${TEST_RAW:-data/raw/RAWFC/test.json}"

RUN_CACHE_BUILD="${RUN_CACHE_BUILD:-true}"
RUN_QD="${RUN_QD:-true}"
RUN_EVIDENCE_MAP="${RUN_EVIDENCE_MAP:-true}"
RUN_GRAPH_BUILD="${RUN_GRAPH_BUILD:-true}"
RUN_LORA="${RUN_LORA:-true}"
RUN_FULLFT="${RUN_FULLFT:-false}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_INFER="${RUN_INFER:-true}"
RUN_API_INFER="${RUN_API_INFER:-${RUN_INFER}}"
RUN_LABEL_TOKEN_INFER="${RUN_LABEL_TOKEN_INFER:-false}"

SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
MIN_TOP_K="${MIN_TOP_K:-5}"
MAX_TOP_K="${MAX_TOP_K:-10}"
GRAPH_BUDGET_SLUG="adaptive${MIN_TOP_K}_${MAX_TOP_K}"
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-v0_6c_rule_step_${GRAPH_BUDGET_SLUG}}"
TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-plain}"
INFER_SPLIT="${INFER_SPLIT:-test}"

MOCK_EVIDENCE_MAPS="${MOCK_EVIDENCE_MAPS:-false}"
if [[ -z "${RUN_TEACHER+x}" ]]; then
  if [[ "${MOCK_EVIDENCE_MAPS}" == "true" || "${MOCK_EVIDENCE_MAPS}" == "1" || "${MOCK_EVIDENCE_MAPS}" == "True" ]]; then
    RUN_TEACHER=false
  else
    RUN_TEACHER=true
  fi
fi
if [[ -z "${MOCK_QUESTIONS+x}" ]]; then
  if [[ "${MOCK_EVIDENCE_MAPS}" == "true" && "${RUN_TRAIN}" != "true" && "${RUN_INFER}" != "true" ]]; then
    MOCK_QUESTIONS=true
  else
    MOCK_QUESTIONS=false
  fi
fi

QUESTION_OUTPUT_ROOT="${QUESTION_OUTPUT_ROOT:-outputs/selectors/question_decomp_retrieval/rawfc_deepseek_v0}"
QUESTION_CACHE_DIR="${QUESTION_CACHE_DIR:-outputs/selectors/question_decomp_retrieval/rawfc_question_cache}"
EVIDENCE_MAP_ROOT="${EVIDENCE_MAP_ROOT:-outputs/selectors/evidence_map_selector/rawfc_v0_6b}"
GRAPH_ROOT="${GRAPH_ROOT:-outputs/selectors/evidence_chain_graph/rawfc_v0_6c_${GRAPH_BUDGET_SLUG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_trace_verifier/rawfc_v0_6c}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/rawfc_v0_6c_selector_trace_full_pipeline}"

QUESTION_BASE_URL="${QUESTION_BASE_URL:-https://api.deepseek.com}"
QUESTION_MODEL="${QUESTION_MODEL:-deepseek-v4-flash}"
QUESTION_API_KEY_ENV="${QUESTION_API_KEY_ENV:-QUESTION_API_KEY}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://api.deepseek.com}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
QUESTION_API_TIMEOUT="${QUESTION_API_TIMEOUT:-120}"
API_MAX_RETRIES="${API_MAX_RETRIES:-5}"
API_CONCURRENCY="${API_CONCURRENCY:-64}"
API_PARSE_MAX_RETRIES="${API_PARSE_MAX_RETRIES:-2}"
API_RETRY_INITIAL_DELAY="${API_RETRY_INITIAL_DELAY:-1.0}"
API_RETRY_MAX_DELAY="${API_RETRY_MAX_DELAY:-30.0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-20260526}"
GUIDED_JSON="${GUIDED_JSON:-1}"
RESUME_QUESTIONS="${RESUME_QUESTIONS:-true}"
NO_PROGRESS="${NO_PROGRESS:-false}"

EMBEDDER_MODEL="${EMBEDDER_MODEL:-/data/models/bge-base-en-v1.5}"
DEVICE="${DEVICE:-cuda}"
EMBEDDER_MAX_LENGTH="${EMBEDDER_MAX_LENGTH:-256}"
EMBEDDER_BATCH_SIZE="${EMBEDDER_BATCH_SIZE:-64}"
PRECISION="${PRECISION:-bf16}"
PER_QUESTION_KEEP="${PER_QUESTION_KEEP:-20}"
MERGED_POOL_SIZE="${MERGED_POOL_SIZE:-15}"
SELECTOR_TOP_K="${SELECTOR_TOP_K:-5}"
RRF_K="${RRF_K:-60}"
Q1_WEIGHT="${Q1_WEIGHT:-1.2}"
OTHER_QUESTION_WEIGHT="${OTHER_QUESTION_WEIGHT:-1.0}"
MERGE_MMR_LAMBDA="${MERGE_MMR_LAMBDA:-0.70}"
ALPHA_DENSE="${ALPHA_DENSE:-0.70}"
ALPHA_LEXICAL="${ALPHA_LEXICAL:-0.20}"
ALPHA_BM25="${ALPHA_BM25:-0.10}"
BASELINE_BONUS="${BASELINE_BONUS:-0.04}"
BASELINE_RANK_WEIGHT="${BASELINE_RANK_WEIGHT:-0.01}"
QD_RRF_WEIGHT="${QD_RRF_WEIGHT:-1.0}"
QD_QUESTION_HIT_WEIGHT="${QD_QUESTION_HIT_WEIGHT:-0.004}"
QD_MAX_HYBRID_WEIGHT="${QD_MAX_HYBRID_WEIGHT:-0.01}"

LORA_CASE_NAME="${LORA_CASE_NAME:-v0_6c_rawfc3_rule_step_${GRAPH_BUDGET_SLUG}}"
FULLFT_CASE_NAME="${FULLFT_CASE_NAME:-v0_6c_rawfc3_rule_step_${GRAPH_BUDGET_SLUG}_fullft}"
FULLFT_DEEPSPEED_CONFIG="${FULLFT_DEEPSPEED_CONFIG:-configs/deepspeed_zero3_bsz2_ga4.json}"

SAMPLE_SUFFIX=""
CHILD_SAMPLE_LIMIT=""
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  SAMPLE_SUFFIX="_sample${SAMPLE_LIMIT}"
  CHILD_SAMPLE_LIMIT="${SAMPLE_LIMIT}"
fi

if [[ -z "${CHUNK_MMR_FINGERPRINT+x}" ]]; then
  FP_ARGS=(--config "${LORA_CONFIG}")
  if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
    FP_ARGS+=(--sample-limit "${SAMPLE_LIMIT}")
  fi
  CHUNK_MMR_FINGERPRINT="$(python scripts/phase5_selectors/build/print_chunk_mmr_fingerprint.py "${FP_ARGS[@]}")"
fi

sample_args=()
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  sample_args=(--sample-limit "${SAMPLE_LIMIT}")
fi

progress_args=()
if [[ "${NO_PROGRESS}" == "1" || "${NO_PROGRESS}" == "true" || "${NO_PROGRESS}" == "True" ]]; then
  progress_args=(--no-progress)
fi

guided_json_args=()
if [[ "${GUIDED_JSON}" == "0" || "${GUIDED_JSON}" == "false" || "${GUIDED_JSON}" == "False" ]]; then
  guided_json_args=(--no-guided-json)
fi

resume_args=(--resume-questions)
if [[ "${RESUME_QUESTIONS}" == "0" || "${RESUME_QUESTIONS}" == "false" || "${RESUME_QUESTIONS}" == "False" ]]; then
  resume_args=(--no-resume-questions)
fi

mock_question_args=()
if [[ "${MOCK_QUESTIONS}" == "1" || "${MOCK_QUESTIONS}" == "true" || "${MOCK_QUESTIONS}" == "True" ]]; then
  mock_question_args=(--mock-questions)
fi

raw_path_for_split() {
  local split="$1"
  if [[ "${split}" == "train" ]]; then
    printf "%s" "${TRAIN_RAW}"
  elif [[ "${split}" == "test" ]]; then
    printf "%s" "${TEST_RAW}"
  else
    printf "%s" "${VAL_RAW}"
  fi
}

qd_dir_for_split() {
  local split="$1"
  printf "%s_%s%s" "${QUESTION_OUTPUT_ROOT}" "${split}" "${SAMPLE_SUFFIX}"
}

evidence_map_dir_for_split() {
  local split="$1"
  printf "%s_%s%s" "${EVIDENCE_MAP_ROOT}" "${split}" "${SAMPLE_SUFFIX}"
}

graph_dir_for_split() {
  local split="$1"
  printf "%s_%s%s" "${GRAPH_ROOT}" "${split}" "${SAMPLE_SUFFIX}"
}

echo "[rawfc-v0.6c] sample_limit       : ${SAMPLE_LIMIT}"
echo "[rawfc-v0.6c] label_schema       : ${LABEL_SCHEMA}"
echo "[rawfc-v0.6c] chunk fingerprint  : ${CHUNK_MMR_FINGERPRINT}"
echo "[rawfc-v0.6c] cache/qd/map/graph : ${RUN_CACHE_BUILD}/${RUN_QD}/${RUN_EVIDENCE_MAP}/${RUN_GRAPH_BUILD}"
echo "[rawfc-v0.6c] lora/fullft        : ${RUN_LORA}/${RUN_FULLFT}"
echo "[rawfc-v0.6c] train/infer        : ${RUN_TRAIN}/${RUN_INFER}"
echo "[rawfc-v0.6c] api/label infer    : ${RUN_API_INFER}/${RUN_LABEL_TOKEN_INFER}"
echo "[rawfc-v0.6c] teacher/mock maps  : ${RUN_TEACHER}/${MOCK_EVIDENCE_MAPS}"
echo "[rawfc-v0.6c] teacher model/env  : ${TEACHER_MODEL}/${TEACHER_API_KEY_ENV}"
echo "[rawfc-v0.6c] mock questions     : ${MOCK_QUESTIONS}"

if [[ "${RUN_CACHE_BUILD}" == "true" || "${RUN_CACHE_BUILD}" == "1" ]]; then
  python -m fact_checking.pipeline.run \
    "experiment=${LORA_EXPERIMENT}" \
    "pipeline.mode=build" \
    "pipeline.force.build=false" \
    "build.data.sample_limit=${SAMPLE_LIMIT}"
fi

build_split_upstream() {
  local split="$1"
  local raw_path qd_dir em_dir graph_dir chunk_cache question_cache_id
  raw_path="$(raw_path_for_split "${split}")"
  qd_dir="$(qd_dir_for_split "${split}")"
  em_dir="$(evidence_map_dir_for_split "${split}")"
  graph_dir="$(graph_dir_for_split "${split}")"
  chunk_cache="outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/${split}.pkl"

  if [[ "${MOCK_QUESTIONS}" == "1" || "${MOCK_QUESTIONS}" == "true" || "${MOCK_QUESTIONS}" == "True" ]]; then
    question_cache_id="rawfc3_mock_${split}${SAMPLE_SUFFIX}"
  else
    question_cache_id="${QUESTION_CACHE_ID:-}"
  fi

  if [[ "${RUN_QD}" == "true" || "${RUN_QD}" == "1" ]]; then
    qcache_args=()
    if [[ -n "${question_cache_id}" ]]; then
      qcache_args=(--question-cache-id "${question_cache_id}")
    fi
    python scripts/phase5_selectors/build/generate_question_decomp_cache.py \
      --input-mode raw_split \
      --raw-path "${raw_path}" \
      --dataset "${RAW_DATASET}" \
      --label-schema "${LABEL_SCHEMA}" \
      --split "${split}" \
      --output-dir "${qd_dir}" \
      --question-cache-dir "${QUESTION_CACHE_DIR}" \
      --question-base-url "${QUESTION_BASE_URL}" \
      --question-model "${QUESTION_MODEL}" \
      --question-api-key-env "${QUESTION_API_KEY_ENV}" \
      --api-timeout "${QUESTION_API_TIMEOUT}" \
      --api-max-retries "${API_MAX_RETRIES}" \
      --api-concurrency "${API_CONCURRENCY}" \
      --api-parse-max-retries "${API_PARSE_MAX_RETRIES}" \
      --retry-initial-delay "${API_RETRY_INITIAL_DELAY}" \
      --retry-max-delay "${API_RETRY_MAX_DELAY}" \
      --max-tokens "${MAX_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --top-p "${TOP_P}" \
      --seed "${SEED}" \
      "${guided_json_args[@]}" \
      "${resume_args[@]}" \
      "${progress_args[@]}" \
      "${mock_question_args[@]}" \
      "${qcache_args[@]}" \
      "${sample_args[@]}"

    python scripts/phase5_selectors/build/build_question_decomp_retrieval.py \
      --oracle-results "" \
      --split "${split}" \
      --questions-jsonl "${qd_dir}/questions_${split}.jsonl" \
      --chunk-cache-path "${chunk_cache}" \
      --output-dir "${qd_dir}" \
      --embedder-model "${EMBEDDER_MODEL}" \
      --device "${DEVICE}" \
      --embedder-max-length "${EMBEDDER_MAX_LENGTH}" \
      --embedder-batch-size "${EMBEDDER_BATCH_SIZE}" \
      --precision "${PRECISION}" \
      --per-question-keep "${PER_QUESTION_KEEP}" \
      --merged-pool-size "${MERGED_POOL_SIZE}" \
      --selector-top-k "${SELECTOR_TOP_K}" \
      --rrf-k "${RRF_K}" \
      --q1-weight "${Q1_WEIGHT}" \
      --other-question-weight "${OTHER_QUESTION_WEIGHT}" \
      --merge-mmr-lambda "${MERGE_MMR_LAMBDA}" \
      --alpha-dense "${ALPHA_DENSE}" \
      --alpha-lexical "${ALPHA_LEXICAL}" \
      --alpha-bm25 "${ALPHA_BM25}" \
      "${progress_args[@]}" \
      "${sample_args[@]}"

    python scripts/phase5_selectors/build/build_question_decomp_union.py \
      --oracle-results "" \
      --split "${split}" \
      --baseline-jsonl "${qd_dir}/baseline_claim_mmr_selected_${split}.jsonl" \
      --qd-pool-jsonl "${qd_dir}/merged_candidate_pool_${split}.jsonl" \
      --output-dir "${qd_dir}" \
      --selector-top-k "${SELECTOR_TOP_K}" \
      --baseline-bonus "${BASELINE_BONUS}" \
      --baseline-rank-weight "${BASELINE_RANK_WEIGHT}" \
      --qd-rrf-weight "${QD_RRF_WEIGHT}" \
      --qd-question-hit-weight "${QD_QUESTION_HIT_WEIGHT}" \
      --qd-max-hybrid-weight "${QD_MAX_HYBRID_WEIGHT}" \
      "${sample_args[@]}"
  fi

  if [[ "${RUN_EVIDENCE_MAP}" == "true" || "${RUN_EVIDENCE_MAP}" == "1" ]]; then
    SPLIT="${split}" \
    RAW_PATH="${raw_path}" \
    QD_UNION_POOL_FILE="${qd_dir}/union_candidate_pool_${split}.jsonl" \
    OUTPUT_DIR="${em_dir}" \
    CANDIDATE_TOP_N="${CANDIDATE_TOP_N}" \
    TOP_K="${MIN_TOP_K}" \
    ORACLE_RESULTS="" \
    RUN_TEACHER="${RUN_TEACHER}" \
    MOCK_EVIDENCE_MAPS="${MOCK_EVIDENCE_MAPS}" \
    TEACHER_BASE_URL="${TEACHER_BASE_URL}" \
    TEACHER_MODEL="${TEACHER_MODEL}" \
    TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV}" \
    BUILD_VERIFIER_DATA=false \
    SAMPLE_LIMIT="${CHILD_SAMPLE_LIMIT}" \
    CONFIG="${LORA_CONFIG}" \
    bash scripts/phase5_selectors/run/run_evidence_map_selector_v0_6b.sh
  fi

  if [[ "${RUN_GRAPH_BUILD}" == "true" || "${RUN_GRAPH_BUILD}" == "1" ]]; then
    SPLIT="${split}" \
    INPUT="${em_dir}/candidate_evidence_map_features_${split}.jsonl" \
    OUTPUT_DIR="${graph_dir}" \
    CANDIDATE_TOP_N="${CANDIDATE_TOP_N}" \
    MIN_TOP_K="${MIN_TOP_K}" \
    MAX_TOP_K="${MAX_TOP_K}" \
    CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
    SAMPLE_LIMIT="${SAMPLE_LIMIT}" \
    bash scripts/phase5_selectors/run/run_evidence_chain_graph_v0_6c.sh
  fi
}

for split in train val test; do
  build_split_upstream "${split}"
done

TRAIN_TRACE="${TRAIN_TRACE:-$(graph_dir_for_split train)/selection_trace_train.jsonl}"
VAL_TRACE="${VAL_TRACE:-$(graph_dir_for_split val)/selection_trace_val.jsonl}"
TEST_TRACE="${TEST_TRACE:-$(graph_dir_for_split test)/selection_trace_test.jsonl}"

run_trace_pipeline() {
  local case_name="$1"
  local config_path="$2"
  local infer_experiment="$3"
  shift 3
  env \
    "$@" \
    CONFIG="${config_path}" \
    INFER_EXPERIMENT="${infer_experiment}" \
    CASE_NAME="${case_name}" \
    SOURCE_TYPE=trace \
    TRAIN_SOURCE="${TRAIN_TRACE}" \
    VAL_SOURCE="${VAL_TRACE}" \
    TEST_SOURCE="${TEST_TRACE}" \
    TRAIN_RAW="${TRAIN_RAW}" \
    VAL_RAW="${VAL_RAW}" \
    TEST_RAW="${TEST_RAW}" \
    RAW_DATASET="${RAW_DATASET}" \
    LABEL_SCHEMA="${LABEL_SCHEMA}" \
    TRACE_SELECTION_MODE=trace \
    TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE}" \
    TOP_K="${MAX_TOP_K}" \
    EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME}" \
    EXPECTED_CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
    SAMPLE_LIMIT="${SAMPLE_LIMIT}" \
    SPLIT="${INFER_SPLIT}" \
    RUN_TRAIN="${RUN_TRAIN}" \
    RUN_INFER="${RUN_INFER}" \
    RUN_API_INFER="${RUN_API_INFER}" \
    RUN_LABEL_TOKEN_INFER="${RUN_LABEL_TOKEN_INFER}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    bash scripts/phase5_selectors/run/run_selector_trace_full_pipeline.sh
}

if [[ "${RUN_LORA}" == "true" || "${RUN_LORA}" == "1" ]]; then
  echo "[rawfc-v0.6c] running LoRA selector-trace verifier: ${LORA_CASE_NAME}"
  run_trace_pipeline "${LORA_CASE_NAME}" "${LORA_CONFIG}" "${LORA_EXPERIMENT}"
fi

if [[ "${RUN_FULLFT}" == "true" || "${RUN_FULLFT}" == "1" ]]; then
  echo "[rawfc-v0.6c] running FullFT selector-trace verifier: ${FULLFT_CASE_NAME}"
  run_trace_pipeline "${FULLFT_CASE_NAME}" "${FULLFT_CONFIG}" "${FULLFT_EXPERIMENT}" \
    DEEPSPEED_CONFIG="${FULLFT_DEEPSPEED_CONFIG}" \
    MERGE_LORA_CACHE=false \
    FINETUNE_MODE=full-parameter
fi

echo "[rawfc-v0.6c] done"
