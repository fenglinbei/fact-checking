#!/usr/bin/env bash
set -euo pipefail

# Build dense-only question-decomposition, evidence-map, and evidence-chain
# traces for RAWFC or LIAR-RAW.  This wrapper keeps old hybrid outputs intact
# by writing to dense-specific output roots.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

is_true() {
  case "${1:-}" in
    true|1|True|TRUE|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

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

DATASET="${DATASET:-rawfc}"  # rawfc | liar_raw
MIN_TOP_K="${MIN_TOP_K:-5}"
MAX_TOP_K="${MAX_TOP_K:-10}"
GRAPH_BUDGET_SLUG="adaptive${MIN_TOP_K}_${MAX_TOP_K}"
SPLITS="${SPLITS:-train val test}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
SELECTOR_TOP_K="${SELECTOR_TOP_K:-5}"

ALPHA_DENSE="${ALPHA_DENSE:-1.0}"
ALPHA_LEXICAL="${ALPHA_LEXICAL:-0.0}"
ALPHA_BM25="${ALPHA_BM25:-0.0}"

RUN_CACHE_BUILD="${RUN_CACHE_BUILD:-false}"
RUN_QD="${RUN_QD:-true}"
RUN_EVIDENCE_MAP="${RUN_EVIDENCE_MAP:-true}"
RUN_GRAPH_BUILD="${RUN_GRAPH_BUILD:-true}"

MOCK_EVIDENCE_MAPS="${MOCK_EVIDENCE_MAPS:-false}"
if [[ -z "${RUN_TEACHER+x}" ]]; then
  if is_true "${MOCK_EVIDENCE_MAPS}"; then
    RUN_TEACHER=false
  else
    RUN_TEACHER=true
  fi
fi
MOCK_QUESTIONS="${MOCK_QUESTIONS:-false}"

QUESTION_BASE_URL="${QUESTION_BASE_URL:-https://api.deepseek.com}"
QUESTION_MODEL="${QUESTION_MODEL:-deepseek-v4-flash}"
QUESTION_API_KEY_ENV="${QUESTION_API_KEY_ENV:-QUESTION_API_KEY}"
QUESTION_API_TIMEOUT="${QUESTION_API_TIMEOUT:-120}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://api.deepseek.com}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
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
RRF_K="${RRF_K:-60}"
Q1_WEIGHT="${Q1_WEIGHT:-1.2}"
OTHER_QUESTION_WEIGHT="${OTHER_QUESTION_WEIGHT:-1.0}"
MERGE_MMR_LAMBDA="${MERGE_MMR_LAMBDA:-0.70}"
BASELINE_BONUS="${BASELINE_BONUS:-0.04}"
BASELINE_RANK_WEIGHT="${BASELINE_RANK_WEIGHT:-0.01}"
QD_RRF_WEIGHT="${QD_RRF_WEIGHT:-1.0}"
QD_QUESTION_HIT_WEIGHT="${QD_QUESTION_HIT_WEIGHT:-0.004}"
QD_MAX_HYBRID_WEIGHT="${QD_MAX_HYBRID_WEIGHT:-0.01}"

case "${DATASET}" in
  rawfc)
    CONFIG="${CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_eval25.yaml}"
    EXPERIMENT="${EXPERIMENT:-v0_6c_rawfc3_rule_step_adaptive5_10_eval25}"
    RAW_DATASET="${RAW_DATASET:-rawfc}"
    LABEL_SCHEMA="${LABEL_SCHEMA:-rawfc3}"
    TRAIN_RAW="${TRAIN_RAW:-data/raw/RAWFC/train.json}"
    VAL_RAW="${VAL_RAW:-data/raw/RAWFC/val.json}"
    TEST_RAW="${TEST_RAW:-data/raw/RAWFC/test.json}"
    QUESTION_OUTPUT_ROOT="${QUESTION_OUTPUT_ROOT:-outputs/selectors/question_decomp_retrieval/rawfc_dense_v0}"
    QUESTION_CACHE_DIR="${QUESTION_CACHE_DIR:-outputs/selectors/question_decomp_retrieval/rawfc_question_cache}"
    EVIDENCE_MAP_ROOT="${EVIDENCE_MAP_ROOT:-outputs/selectors/evidence_map_selector/rawfc_dense_v0_6b}"
    GRAPH_ROOT="${GRAPH_ROOT:-outputs/selectors/evidence_chain_graph/rawfc_dense_v0_6c_${GRAPH_BUDGET_SLUG}}"
    ;;
  liar_raw)
    CONFIG="${CONFIG:-configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml}"
    EXPERIMENT="${EXPERIMENT:-b3_oracle_sentence_direct_verifier_1024}"
    RAW_DATASET="${RAW_DATASET:-liar_raw}"
    LABEL_SCHEMA="${LABEL_SCHEMA:-liar6}"
    TRAIN_RAW="${TRAIN_RAW:-data/raw/LIAR-RAW/train.json}"
    VAL_RAW="${VAL_RAW:-data/raw/LIAR-RAW/val.json}"
    TEST_RAW="${TEST_RAW:-data/raw/LIAR-RAW/test.json}"
    QUESTION_OUTPUT_ROOT="${QUESTION_OUTPUT_ROOT:-outputs/selectors/question_decomp_retrieval/liar_raw_dense_v0}"
    QUESTION_CACHE_DIR="${QUESTION_CACHE_DIR:-outputs/selectors/question_decomp_retrieval/question_cache}"
    EVIDENCE_MAP_ROOT="${EVIDENCE_MAP_ROOT:-outputs/selectors/evidence_map_selector/liar_raw_dense_v0_6b}"
    GRAPH_ROOT="${GRAPH_ROOT:-outputs/selectors/evidence_chain_graph/liar_raw_dense_v0_6c_${GRAPH_BUDGET_SLUG}}"
    ;;
  *)
    echo "[dense-only-traces] DATASET must be rawfc or liar_raw, got: ${DATASET}" >&2
    exit 2
    ;;
esac

SAMPLE_SUFFIX=""
CHILD_SAMPLE_LIMIT=""
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  SAMPLE_SUFFIX="_sample${SAMPLE_LIMIT}"
  CHILD_SAMPLE_LIMIT="${SAMPLE_LIMIT}"
fi

if [[ -z "${CHUNK_MMR_FINGERPRINT+x}" ]]; then
  FP_ARGS=(--config "${CONFIG}")
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
if is_true "${NO_PROGRESS}"; then
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
if is_true "${MOCK_QUESTIONS}"; then
  mock_question_args=(--mock-questions)
fi

echo "[dense-only-traces] dataset          : ${DATASET}"
echo "[dense-only-traces] config/experiment: ${CONFIG} / ${EXPERIMENT}"
echo "[dense-only-traces] label schema     : ${LABEL_SCHEMA}"
echo "[dense-only-traces] splits           : ${SPLITS}"
echo "[dense-only-traces] alpha            : ${ALPHA_DENSE}/${ALPHA_LEXICAL}/${ALPHA_BM25}"
echo "[dense-only-traces] chunk fingerprint: ${CHUNK_MMR_FINGERPRINT}"
echo "[dense-only-traces] qd/map/graph     : ${QUESTION_OUTPUT_ROOT} / ${EVIDENCE_MAP_ROOT} / ${GRAPH_ROOT}"
echo "[dense-only-traces] run cache/qd/map/graph: ${RUN_CACHE_BUILD}/${RUN_QD}/${RUN_EVIDENCE_MAP}/${RUN_GRAPH_BUILD}"

if is_true "${RUN_CACHE_BUILD}"; then
  python -m fact_checking.pipeline.run \
    "experiment=${EXPERIMENT}" \
    "pipeline.mode=build" \
    "pipeline.force.build=false" \
    "build.data.sample_limit=${SAMPLE_LIMIT}" \
    "build.retrieval.alpha_dense=${ALPHA_DENSE}" \
    "build.retrieval.alpha_lexical=${ALPHA_LEXICAL}" \
    "build.retrieval.alpha_bm25=${ALPHA_BM25}"
fi

for split in ${SPLITS}; do
  raw_path="$(raw_path_for_split "${split}")"
  qd_dir="${QUESTION_OUTPUT_ROOT}_${split}${SAMPLE_SUFFIX}"
  em_dir="${EVIDENCE_MAP_ROOT}_${split}${SAMPLE_SUFFIX}"
  graph_dir="${GRAPH_ROOT}_${split}${SAMPLE_SUFFIX}"
  chunk_cache="outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/${split}.pkl"

  if [[ ! -s "${chunk_cache}" ]]; then
    echo "[dense-only-traces] missing chunk cache for split=${split}: ${chunk_cache}" >&2
    echo "[dense-only-traces] rerun with RUN_CACHE_BUILD=true or set CHUNK_MMR_FINGERPRINT." >&2
    exit 1
  fi

  question_cache_id="${QUESTION_CACHE_ID:-}"
  if is_true "${MOCK_QUESTIONS}"; then
    question_cache_id="${RAW_DATASET}_mock_${split}${SAMPLE_SUFFIX}"
  fi

  if is_true "${RUN_QD}"; then
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

  if is_true "${RUN_EVIDENCE_MAP}"; then
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
    CONFIG="${CONFIG}" \
    bash scripts/phase5_selectors/run/run_evidence_map_selector_v0_6b.sh
  fi

  if is_true "${RUN_GRAPH_BUILD}"; then
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

  echo "[dense-only-traces] split=${split} graph=${graph_dir}/selection_trace_${split}.jsonl"
done

echo "[dense-only-traces] done"
