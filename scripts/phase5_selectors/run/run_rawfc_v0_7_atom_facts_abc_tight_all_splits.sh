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
EXPERIMENT="${EXPERIMENT:-v0_6c_rawfc3_rule_step_adaptive5_10_abc_tight_chunking}"
CONFIG="${CONFIG:-configs/experiment/v0_6c_rawfc3_rule_step_adaptive5_10_abc_tight_chunking.yaml}"
DRY_RUN="${DRY_RUN:-false}"

TRAIN_RAW="${TRAIN_RAW:-data/raw/RAWFC/train.json}"
VAL_RAW="${VAL_RAW:-data/raw/RAWFC/val.json}"
TEST_RAW="${TEST_RAW:-data/raw/RAWFC/test.json}"

QUESTION_OUTPUT_ROOT="${QUESTION_OUTPUT_ROOT:-outputs/selectors/question_decomp_retrieval/rawfc_deepseek_v0_abc_tight}"
EVIDENCE_MAP_ROOT="${EVIDENCE_MAP_ROOT:-outputs/selectors/evidence_map_selector/rawfc_v0_7_atom_facts_abc_tight}"
GRAPH_ROOT="${GRAPH_ROOT:-outputs/selectors/evidence_chain_graph/rawfc_v0_7_atom_facts_abc_tight_budgeted_marginal_adaptive5_10}"

RUN_CACHE_BUILD="${RUN_CACHE_BUILD:-true}"
RUN_QD="${RUN_QD:-true}"
RUN_EVIDENCE_MAP="${RUN_EVIDENCE_MAP:-true}"
RUN_GRAPH_BUILD="${RUN_GRAPH_BUILD:-true}"

RUN_TEACHER="${RUN_TEACHER:-true}"
MOCK_EVIDENCE_MAPS="${MOCK_EVIDENCE_MAPS:-false}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://api.deepseek.com}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
QUESTION_API_KEY_ENV="${QUESTION_API_KEY_ENV:-DEEPSEEK_API_KEY}"
CONCURRENCY="${CONCURRENCY:-128}"
REQUESTS_PER_MINUTE="${REQUESTS_PER_MINUTE:-60000}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
THINKING_TYPE="${THINKING_TYPE:-disabled}"
PROMPT_VERSION="${PROMPT_VERSION:-evidence_map_v0_7_atom_facts_abc}"
MAX_EVIDENCE_CHARS="${MAX_EVIDENCE_CHARS:-700}"

CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
TOP_K="${TOP_K:-5}"
MIN_TOP_K="${MIN_TOP_K:-5}"
MAX_TOP_K="${MAX_TOP_K:-10}"
TARGET_COVERAGE="${TARGET_COVERAGE:-0.80}"
STOP_GAIN_THRESHOLD="${STOP_GAIN_THRESHOLD:-0.10}"
INSUFFICIENT_GAIN_THRESHOLD="${INSUFFICIENT_GAIN_THRESHOLD:-0.05}"
SELECTOR_DIRECTNESS_WEIGHT="${SELECTOR_DIRECTNESS_WEIGHT:-0.30}"
SELECTOR_BACKGROUND_PENALTY="${SELECTOR_BACKGROUND_PENALTY:-0.30}"
OBJECTIVE_BACKGROUND_OR_IRRELEVANT="${OBJECTIVE_BACKGROUND_OR_IRRELEVANT:-0.24}"
OBJECTIVE_LENGTH="${OBJECTIVE_LENGTH:-0.08}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

run_cmd() {
  if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

SAMPLE_SUFFIX=""
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  SAMPLE_SUFFIX="_sample${SAMPLE_LIMIT}"
fi
CHILD_SAMPLE_LIMIT=""
if [[ -n "${SAMPLE_LIMIT}" && "${SAMPLE_LIMIT}" != "0" ]]; then
  CHILD_SAMPLE_LIMIT="${SAMPLE_LIMIT}"
fi

FP_ARGS=(--config "${CONFIG}")
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  FP_ARGS+=(--sample-limit "${SAMPLE_LIMIT}")
fi
if [[ -z "${CHUNK_MMR_FINGERPRINT+x}" ]]; then
  if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
    CHUNK_MMR_FINGERPRINT="dry-run-fingerprint"
  else
    CHUNK_MMR_FINGERPRINT="$(PYTHONPATH=src "${PYTHON_BIN}" scripts/phase5_selectors/build/print_chunk_mmr_fingerprint.py "${FP_ARGS[@]}")"
  fi
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
    echo "[rawfc-v0.7-atom-facts-abc-tight] unsupported split: ${split}" >&2
    exit 2
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

echo "[rawfc-v0.7-atom-facts-abc-tight] splits       : ${SPLITS}"
echo "[rawfc-v0.7-atom-facts-abc-tight] config       : ${CONFIG}"
echo "[rawfc-v0.7-atom-facts-abc-tight] fingerprint  : ${CHUNK_MMR_FINGERPRINT}"
echo "[rawfc-v0.7-atom-facts-abc-tight] roots        : ${QUESTION_OUTPUT_ROOT} | ${EVIDENCE_MAP_ROOT} | ${GRAPH_ROOT}"
echo "[rawfc-v0.7-atom-facts-abc-tight] stages       : cache=${RUN_CACHE_BUILD} qd=${RUN_QD} map=${RUN_EVIDENCE_MAP} graph=${RUN_GRAPH_BUILD}"
echo "[rawfc-v0.7-atom-facts-abc-tight] prompt       : ${PROMPT_VERSION}"
echo "[rawfc-v0.7-atom-facts-abc-tight] selector     : directness=${SELECTOR_DIRECTNESS_WEIGHT} background=${SELECTOR_BACKGROUND_PENALTY}"
echo "[rawfc-v0.7-atom-facts-abc-tight] objective    : background=${OBJECTIVE_BACKGROUND_OR_IRRELEVANT} length=${OBJECTIVE_LENGTH}"

if [[ "${RUN_CACHE_BUILD}" == "true" || "${RUN_CACHE_BUILD}" == "1" ]]; then
  run_cmd env \
    PYTHON_BIN="${PYTHON_BIN}" \
    EXPERIMENT="${EXPERIMENT}" \
    CONFIG="${CONFIG}" \
    SAMPLE_LIMIT="${SAMPLE_LIMIT}" \
    CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
    bash "${SCRIPT_DIR}/run_rawfc_v0_7_atom_facts_abc_build_cache.sh"
fi

if [[ "${RUN_QD}" == "true" || "${RUN_QD}" == "1" ]]; then
  run_cmd env \
    PYTHON_BIN="${PYTHON_BIN}" \
    EXPERIMENT="${EXPERIMENT}" \
    CONFIG="${CONFIG}" \
    SPLITS="${SPLITS}" \
    TRAIN_RAW="${TRAIN_RAW}" \
    VAL_RAW="${VAL_RAW}" \
    TEST_RAW="${TEST_RAW}" \
    QUESTION_OUTPUT_ROOT="${QUESTION_OUTPUT_ROOT}" \
    QUESTION_API_KEY_ENV="${QUESTION_API_KEY_ENV}" \
    SAMPLE_LIMIT="${SAMPLE_LIMIT}" \
    RUN_CACHE_BUILD=false \
    CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
    bash "${SCRIPT_DIR}/run_rawfc_v0_7_atom_facts_abc_qd.sh"
fi

for split in ${SPLITS}; do
  raw_path="$(raw_path_for_split "${split}")"
  qd_dir="$(qd_dir_for_split "${split}")"
  em_dir="$(evidence_map_dir_for_split "${split}")"
  graph_dir="$(graph_dir_for_split "${split}")"
  union_pool="${qd_dir}/union_candidate_pool_${split}.jsonl"
  features="${em_dir}/candidate_evidence_map_features_${split}.jsonl"

  echo "[rawfc-v0.7-atom-facts-abc-tight] split        : ${split}"
  echo "[rawfc-v0.7-atom-facts-abc-tight] raw          : ${raw_path}"
  echo "[rawfc-v0.7-atom-facts-abc-tight] qd union     : ${union_pool}"
  echo "[rawfc-v0.7-atom-facts-abc-tight] evidence map : ${em_dir}"
  echo "[rawfc-v0.7-atom-facts-abc-tight] graph        : ${graph_dir}"

  if [[ "${RUN_EVIDENCE_MAP}" == "true" || "${RUN_EVIDENCE_MAP}" == "1" ]]; then
    if [[ "${DRY_RUN}" != "true" && "${DRY_RUN}" != "1" && ! -s "${union_pool}" ]]; then
      echo "[rawfc-v0.7-atom-facts-abc-tight] missing QD union pool: ${union_pool}" >&2
      echo "[rawfc-v0.7-atom-facts-abc-tight] rerun with RUN_QD=true first" >&2
      exit 1
    fi

    run_cmd env \
      SPLIT="${split}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      RAW_PATH="${raw_path}" \
      QD_UNION_POOL_FILE="${union_pool}" \
      OUTPUT_DIR="${em_dir}" \
      ORACLE_RESULTS="" \
      CANDIDATE_TOP_N="${CANDIDATE_TOP_N}" \
      TOP_K="${TOP_K}" \
      SELECTOR_DIRECTNESS_WEIGHT="${SELECTOR_DIRECTNESS_WEIGHT}" \
      SELECTOR_BACKGROUND_PENALTY="${SELECTOR_BACKGROUND_PENALTY}" \
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
      SAMPLE_LIMIT="${CHILD_SAMPLE_LIMIT}" \
      CONFIG="${CONFIG}" \
      bash "${SCRIPT_DIR}/run_evidence_map_selector_v0_7_atom_facts_abc.sh"
  fi

  if [[ "${RUN_GRAPH_BUILD}" == "true" || "${RUN_GRAPH_BUILD}" == "1" ]]; then
    if [[ "${DRY_RUN}" != "true" && "${DRY_RUN}" != "1" && ! -s "${features}" ]]; then
      echo "[rawfc-v0.7-atom-facts-abc-tight] missing evidence-map features: ${features}" >&2
      exit 1
    fi

    run_cmd env \
      SPLIT="${split}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      CONFIG="${CONFIG}" \
      INPUT="${features}" \
      OUTPUT_DIR="${graph_dir}" \
      CANDIDATE_TOP_N="${CANDIDATE_TOP_N}" \
      MIN_TOP_K="${MIN_TOP_K}" \
      MAX_TOP_K="${MAX_TOP_K}" \
      CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
      TARGET_COVERAGE="${TARGET_COVERAGE}" \
      STOP_GAIN_THRESHOLD="${STOP_GAIN_THRESHOLD}" \
      INSUFFICIENT_GAIN_THRESHOLD="${INSUFFICIENT_GAIN_THRESHOLD}" \
      OBJECTIVE_BACKGROUND_OR_IRRELEVANT="${OBJECTIVE_BACKGROUND_OR_IRRELEVANT}" \
      OBJECTIVE_LENGTH="${OBJECTIVE_LENGTH}" \
      SAMPLE_LIMIT="${SAMPLE_LIMIT}" \
      bash "${SCRIPT_DIR}/run_evidence_chain_graph_v0_7_atom_facts_abc.sh"
  fi
done

echo "[rawfc-v0.7-atom-facts-abc-tight] done"
