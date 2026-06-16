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

TRAIN_RAW="${TRAIN_RAW:-data/raw/RAWFC/train.json}"
VAL_RAW="${VAL_RAW:-data/raw/RAWFC/val.json}"
TEST_RAW="${TEST_RAW:-data/raw/RAWFC/test.json}"

QUESTION_OUTPUT_ROOT="${QUESTION_OUTPUT_ROOT:-outputs/selectors/question_decomp_retrieval/rawfc_deepseek_v0}"
EVIDENCE_MAP_ROOT="${EVIDENCE_MAP_ROOT:-outputs/selectors/evidence_map_selector/rawfc_v0_7_atom_facts}"
GRAPH_ROOT="${GRAPH_ROOT:-outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_budgeted_marginal_adaptive5_10}"

RUN_QD="${RUN_QD:-false}"
RUN_EVIDENCE_MAP="${RUN_EVIDENCE_MAP:-true}"
RUN_GRAPH_BUILD="${RUN_GRAPH_BUILD:-true}"

RUN_TEACHER="${RUN_TEACHER:-true}"
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

CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
TOP_K="${TOP_K:-5}"
MIN_TOP_K="${MIN_TOP_K:-5}"
MAX_TOP_K="${MAX_TOP_K:-10}"
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
TARGET_COVERAGE="${TARGET_COVERAGE:-0.80}"
STOP_GAIN_THRESHOLD="${STOP_GAIN_THRESHOLD:-0.10}"
INSUFFICIENT_GAIN_THRESHOLD="${INSUFFICIENT_GAIN_THRESHOLD:-0.05}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"

raw_path_for_split() {
  local split="$1"
  if [[ "${split}" == "train" ]]; then
    printf "%s" "${TRAIN_RAW}"
  elif [[ "${split}" == "test" ]]; then
    printf "%s" "${TEST_RAW}"
  elif [[ "${split}" == "val" ]]; then
    printf "%s" "${VAL_RAW}"
  else
    echo "[rawfc-v0.7-atom-facts] unsupported split: ${split}" >&2
    exit 2
  fi
}

qd_dir_for_split() {
  local split="$1"
  printf "%s_%s" "${QUESTION_OUTPUT_ROOT}" "${split}"
}

evidence_map_dir_for_split() {
  local split="$1"
  printf "%s_%s" "${EVIDENCE_MAP_ROOT}" "${split}"
}

graph_dir_for_split() {
  local split="$1"
  printf "%s_%s" "${GRAPH_ROOT}" "${split}"
}

run_qd_if_requested() {
  if [[ "${RUN_QD}" != "true" && "${RUN_QD}" != "1" ]]; then
    return
  fi

  echo "[rawfc-v0.7-atom-facts] rebuilding RAWFC QD upstream via existing rawfc v0.6c QD-only path"

  local qd_sample_limit="${SAMPLE_LIMIT:-0}"
  local python_dir
  python_dir="$(dirname -- "${PYTHON_BIN}")"
  if [[ "${PYTHON_BIN}" == */* && -d "${python_dir}" ]]; then
    PATH="${python_dir}:${PATH}" \
    RUN_CACHE_BUILD=false \
    RUN_QD=true \
    RUN_EVIDENCE_MAP=false \
    RUN_GRAPH_BUILD=false \
    RUN_LORA=false \
    RUN_FULLFT=false \
    RUN_TRAIN=false \
    RUN_INFER=false \
    RUN_API_INFER=false \
    RUN_LABEL_TOKEN_INFER=false \
    QUESTION_OUTPUT_ROOT="${QUESTION_OUTPUT_ROOT}" \
    RAW_DATASET=rawfc \
    LABEL_SCHEMA=rawfc3 \
    TRAIN_RAW="${TRAIN_RAW}" \
    VAL_RAW="${VAL_RAW}" \
    TEST_RAW="${TEST_RAW}" \
    SAMPLE_LIMIT="${qd_sample_limit}" \
    bash "${SCRIPT_DIR}/run_rawfc_v0_6c_rule_step_adaptive5_10_all_pipelines.sh"
  else
    RUN_CACHE_BUILD=false \
    RUN_QD=true \
    RUN_EVIDENCE_MAP=false \
    RUN_GRAPH_BUILD=false \
    RUN_LORA=false \
    RUN_FULLFT=false \
    RUN_TRAIN=false \
    RUN_INFER=false \
    RUN_API_INFER=false \
    RUN_LABEL_TOKEN_INFER=false \
    QUESTION_OUTPUT_ROOT="${QUESTION_OUTPUT_ROOT}" \
    RAW_DATASET=rawfc \
    LABEL_SCHEMA=rawfc3 \
    TRAIN_RAW="${TRAIN_RAW}" \
    VAL_RAW="${VAL_RAW}" \
    TEST_RAW="${TEST_RAW}" \
    SAMPLE_LIMIT="${qd_sample_limit}" \
    bash "${SCRIPT_DIR}/run_rawfc_v0_6c_rule_step_adaptive5_10_all_pipelines.sh"
  fi
}

build_split() {
  local split="$1"
  local raw_path qd_dir em_dir graph_dir union_pool features
  raw_path="$(raw_path_for_split "${split}")"
  qd_dir="$(qd_dir_for_split "${split}")"
  em_dir="$(evidence_map_dir_for_split "${split}")"
  graph_dir="$(graph_dir_for_split "${split}")"
  union_pool="${qd_dir}/union_candidate_pool_${split}.jsonl"
  features="${em_dir}/candidate_evidence_map_features_${split}.jsonl"

  echo "[rawfc-v0.7-atom-facts] split        : ${split}"
  echo "[rawfc-v0.7-atom-facts] raw          : ${raw_path}"
  echo "[rawfc-v0.7-atom-facts] qd union     : ${union_pool}"
  echo "[rawfc-v0.7-atom-facts] evidence map : ${em_dir}"
  echo "[rawfc-v0.7-atom-facts] graph        : ${graph_dir}"

  if [[ ! -s "${raw_path}" ]]; then
    echo "[rawfc-v0.7-atom-facts] missing RAWFC raw split: ${raw_path}" >&2
    exit 1
  fi

  if [[ "${RUN_EVIDENCE_MAP}" == "true" || "${RUN_EVIDENCE_MAP}" == "1" ]]; then
    if [[ ! -s "${union_pool}" ]]; then
      echo "[rawfc-v0.7-atom-facts] missing QD union pool: ${union_pool}" >&2
      echo "[rawfc-v0.7-atom-facts] rerun with RUN_QD=true to rebuild RAWFC QD upstream first" >&2
      exit 1
    fi

    SPLIT="${split}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    RAW_PATH="${raw_path}" \
    QD_UNION_POOL_FILE="${union_pool}" \
    OUTPUT_DIR="${em_dir}" \
    ORACLE_RESULTS="" \
    CANDIDATE_TOP_N="${CANDIDATE_TOP_N}" \
    TOP_K="${TOP_K}" \
    RUN_TEACHER="${RUN_TEACHER}" \
    MOCK_EVIDENCE_MAPS="${MOCK_EVIDENCE_MAPS}" \
    TEACHER_BASE_URL="${TEACHER_BASE_URL}" \
    TEACHER_MODEL="${TEACHER_MODEL}" \
    TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV}" \
    CONCURRENCY="${CONCURRENCY}" \
    REQUESTS_PER_MINUTE="${REQUESTS_PER_MINUTE}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    THINKING_TYPE="${THINKING_TYPE}" \
    PROMPT_VERSION="${PROMPT_VERSION}" \
    MAX_EVIDENCE_CHARS="${MAX_EVIDENCE_CHARS}" \
    BUILD_VERIFIER_DATA=false \
    SAMPLE_LIMIT="${SAMPLE_LIMIT}" \
    bash "${SCRIPT_DIR}/run_evidence_map_selector_v0_7_atom_facts.sh"
  fi

  if [[ "${RUN_GRAPH_BUILD}" == "true" || "${RUN_GRAPH_BUILD}" == "1" ]]; then
    if [[ ! -s "${features}" ]]; then
      echo "[rawfc-v0.7-atom-facts] missing evidence-map features: ${features}" >&2
      exit 1
    fi

    SPLIT="${split}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    INPUT="${features}" \
    OUTPUT_DIR="${graph_dir}" \
    CANDIDATE_TOP_N="${CANDIDATE_TOP_N}" \
    MIN_TOP_K="${MIN_TOP_K}" \
    MAX_TOP_K="${MAX_TOP_K}" \
    CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
    TARGET_COVERAGE="${TARGET_COVERAGE}" \
    STOP_GAIN_THRESHOLD="${STOP_GAIN_THRESHOLD}" \
    INSUFFICIENT_GAIN_THRESHOLD="${INSUFFICIENT_GAIN_THRESHOLD}" \
    SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}" \
    bash "${SCRIPT_DIR}/run_evidence_chain_graph_v0_7_atom_facts.sh"
  fi
}

echo "[rawfc-v0.7-atom-facts] splits       : ${SPLITS}"
echo "[rawfc-v0.7-atom-facts] run qd       : ${RUN_QD}"
echo "[rawfc-v0.7-atom-facts] run map/graph: ${RUN_EVIDENCE_MAP}/${RUN_GRAPH_BUILD}"
echo "[rawfc-v0.7-atom-facts] prompt       : ${PROMPT_VERSION}"
echo "[rawfc-v0.7-atom-facts] model        : ${TEACHER_MODEL}"
echo "[rawfc-v0.7-atom-facts] concurrency  : ${CONCURRENCY}"
echo "[rawfc-v0.7-atom-facts] rpm          : ${REQUESTS_PER_MINUTE}"
echo "[rawfc-v0.7-atom-facts] thinking     : ${THINKING_TYPE}"

run_qd_if_requested

for split in ${SPLITS}; do
  build_split "${split}"
done

echo "[rawfc-v0.7-atom-facts] done"
