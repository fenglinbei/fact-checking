#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

LOAD_ENV="${LOAD_ENV:-true}"
if [[ "${LOAD_ENV}" == "true" || "${LOAD_ENV}" == "1" ]]; then
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
  fi
fi

SPLITS="${SPLITS:-train val test}"
PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"
EXPERIMENT="${EXPERIMENT:-v0_7_liar_raw_atom_facts_abc_chunking}"
CONFIG="${CONFIG:-configs/experiment/v0_7_liar_raw_atom_facts_abc_chunking.yaml}"

TRAIN_RAW="${TRAIN_RAW:-data/raw/LIAR-RAW/train.json}"
VAL_RAW="${VAL_RAW:-data/raw/LIAR-RAW/val.json}"
TEST_RAW="${TEST_RAW:-data/raw/LIAR-RAW/test.json}"
RAW_DATASET="${RAW_DATASET:-liar_raw}"
LABEL_SCHEMA="${LABEL_SCHEMA:-liar6}"

QUESTION_OUTPUT_ROOT="${QUESTION_OUTPUT_ROOT:-outputs/selectors/question_decomp_retrieval/liar_raw_deepseek_v0_abc}"
QUESTION_CACHE_DIR="${QUESTION_CACHE_DIR:-outputs/selectors/question_decomp_retrieval/liar_raw_question_cache_abc}"

RUN_CACHE_BUILD="${RUN_CACHE_BUILD:-false}"
RUN_QUESTION_CACHE="${RUN_QUESTION_CACHE:-true}"
RUN_QD_RETRIEVAL="${RUN_QD_RETRIEVAL:-true}"
RUN_QD_UNION="${RUN_QD_UNION:-true}"

SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
QUESTION_BASE_URL="${QUESTION_BASE_URL:-https://api.deepseek.com}"
QUESTION_MODEL="${QUESTION_MODEL:-deepseek-v4-flash}"
QUESTION_API_KEY_ENV="${QUESTION_API_KEY_ENV:-DEEPSEEK_API_KEY}"
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
MOCK_QUESTIONS="${MOCK_QUESTIONS:-false}"
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

SAMPLE_SUFFIX=""
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  SAMPLE_SUFFIX="_sample${SAMPLE_LIMIT}"
fi

FP_ARGS=(--config "${CONFIG}")
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  FP_ARGS+=(--sample-limit "${SAMPLE_LIMIT}")
fi
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-$(PYTHONPATH=src "${PYTHON_BIN}" scripts/phase5_selectors/build/print_chunk_mmr_fingerprint.py "${FP_ARGS[@]}")}"

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
  elif [[ "${split}" == "val" ]]; then
    printf "%s" "${VAL_RAW}"
  else
    echo "[liar-raw-v0.7-atom-facts-abc-qd] unsupported split: ${split}" >&2
    exit 2
  fi
}

qd_dir_for_split() {
  local split="$1"
  printf "%s_%s%s" "${QUESTION_OUTPUT_ROOT}" "${split}" "${SAMPLE_SUFFIX}"
}

if [[ "${RUN_CACHE_BUILD}" == "true" || "${RUN_CACHE_BUILD}" == "1" ]]; then
  PYTHON_BIN="${PYTHON_BIN}" \
  EXPERIMENT="${EXPERIMENT}" \
  CONFIG="${CONFIG}" \
  SAMPLE_LIMIT="${SAMPLE_LIMIT}" \
  CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
  bash "${SCRIPT_DIR}/run_liar_raw_v0_7_atom_facts_abc_build_cache.sh"
fi

echo "[liar-raw-v0.7-atom-facts-abc-qd] splits      : ${SPLITS}"
echo "[liar-raw-v0.7-atom-facts-abc-qd] dataset     : ${RAW_DATASET}/${LABEL_SCHEMA}"
echo "[liar-raw-v0.7-atom-facts-abc-qd] config      : ${CONFIG}"
echo "[liar-raw-v0.7-atom-facts-abc-qd] output root : ${QUESTION_OUTPUT_ROOT}"
echo "[liar-raw-v0.7-atom-facts-abc-qd] fingerprint : ${CHUNK_MMR_FINGERPRINT}"
echo "[liar-raw-v0.7-atom-facts-abc-qd] stages      : ${RUN_QUESTION_CACHE}/${RUN_QD_RETRIEVAL}/${RUN_QD_UNION}"

for split in ${SPLITS}; do
  raw_path="$(raw_path_for_split "${split}")"
  qd_dir="$(qd_dir_for_split "${split}")"
  chunk_cache="outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/${split}.pkl"

  echo "[liar-raw-v0.7-atom-facts-abc-qd] split       : ${split}"
  echo "[liar-raw-v0.7-atom-facts-abc-qd] raw         : ${raw_path}"
  echo "[liar-raw-v0.7-atom-facts-abc-qd] chunk cache : ${chunk_cache}"
  echo "[liar-raw-v0.7-atom-facts-abc-qd] output      : ${qd_dir}"

  if [[ ! -s "${raw_path}" ]]; then
    echo "[liar-raw-v0.7-atom-facts-abc-qd] missing LIAR-RAW raw split: ${raw_path}" >&2
    exit 1
  fi
  mkdir -p "${qd_dir}"

  if [[ "${RUN_QUESTION_CACHE}" == "true" || "${RUN_QUESTION_CACHE}" == "1" ]]; then
    qcache_args=()
    question_cache_id="${QUESTION_CACHE_ID:-}"
    if [[ -z "${question_cache_id}" && ( "${MOCK_QUESTIONS}" == "true" || "${MOCK_QUESTIONS}" == "1" ) ]]; then
      question_cache_id="liar_raw_abc_mock_${split}${SAMPLE_SUFFIX}"
    fi
    if [[ -n "${question_cache_id}" ]]; then
      qcache_args=(--question-cache-id "${question_cache_id}")
    fi

    PYTHONPATH=src "${PYTHON_BIN}" scripts/phase5_selectors/build/generate_question_decomp_cache.py \
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
  fi

  if [[ "${RUN_QD_RETRIEVAL}" == "true" || "${RUN_QD_RETRIEVAL}" == "1" ]]; then
    if [[ ! -s "${chunk_cache}" ]]; then
      echo "[liar-raw-v0.7-atom-facts-abc-qd] missing ABC chunk cache: ${chunk_cache}" >&2
      echo "[liar-raw-v0.7-atom-facts-abc-qd] run RUN_CACHE_BUILD=true first" >&2
      exit 1
    fi
    if [[ ! -s "${qd_dir}/questions_${split}.jsonl" ]]; then
      echo "[liar-raw-v0.7-atom-facts-abc-qd] missing questions: ${qd_dir}/questions_${split}.jsonl" >&2
      exit 1
    fi

    PYTHONPATH=src "${PYTHON_BIN}" scripts/phase5_selectors/build/build_question_decomp_retrieval.py \
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
  fi

  if [[ "${RUN_QD_UNION}" == "true" || "${RUN_QD_UNION}" == "1" ]]; then
    PYTHONPATH=src "${PYTHON_BIN}" scripts/phase5_selectors/build/build_question_decomp_union.py \
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
done

echo "[liar-raw-v0.7-atom-facts-abc-qd] done"
