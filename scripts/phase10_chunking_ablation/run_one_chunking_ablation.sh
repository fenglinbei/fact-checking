#!/usr/bin/env bash
set -euo pipefail

# Run one retrieval-only chunking-granularity ablation.
#
# This script intentionally uses the ordinary build -> train pipeline, followed
# by local label-token evaluation. It does not use the pipeline API/vLLM infer
# phase and does not call the phase5/phase9 QD, evidence-map, or graph-selector
# wrappers.
#
# Usage:
#   CHUNKING=sentence bash scripts/phase10_chunking_ablation/run_one_chunking_ablation.sh
#   CHUNKING=semantic MODE=build DRY_RUN=true bash scripts/phase10_chunking_ablation/run_one_chunking_ablation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=scripts/phase7_backbone_migration/backbone_cases.sh
source scripts/phase7_backbone_migration/backbone_cases.sh

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

normalize_dataset() {
  case "${1:-}" in
    liar|liar_raw|liar-raw|LIAR-RAW) printf "%s" "liar_raw" ;;
    rawfc|raw_fc|RAWFC) printf "%s" "rawfc" ;;
    *)
      echo "[chunking-ablation] DATASET must be liar_raw or rawfc, got: ${1:-}" >&2
      return 2
      ;;
  esac
}

normalize_chunking() {
  case "${1:-}" in
    raw|report|raw_report|raw-report) printf "%s" "raw" ;;
    semantic|semantic_chunk|semantic-chunk) printf "%s" "semantic" ;;
    sentence|sent) printf "%s" "sentence" ;;
    abc_claim_aware|abc-claim-aware|abc) printf "%s" "abc_claim_aware" ;;
    *)
      echo "[chunking-ablation] CHUNKING must be raw, semantic, sentence, or abc_claim_aware, got: ${1:-}" >&2
      return 2
      ;;
  esac
}

is_true() {
  case "${1:-}" in
    true|1|True|TRUE|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

DATASET="$(normalize_dataset "${DATASET:-rawfc}")"
CHUNKING="$(normalize_chunking "${CHUNKING:-sentence}")"
BACKBONE="${BACKBONE:-qwen3_4b_2507}"
require_backbone "${BACKBONE}"

MODEL_PATH="${MODEL_PATH:-$(backbone_path "${BACKBONE}")}"
MODE="${MODE:-train}"
TOP_K="${TOP_K:-5}"
CANDIDATE_POOL_K="${CANDIDATE_POOL_K:-32}"
SELECTION_METHOD="${SELECTION_METHOD:-mmr_prompt_budget}"
PROMPT_BUDGET_MIN_K="${PROMPT_BUDGET_MIN_K:-${TOP_K}}"
PROMPT_BUDGET_MAX_K="${PROMPT_BUDGET_MAX_K:-20}"
PROMPT_BUDGET_OVERSHOOT_TOLERANCE_TOKENS="${PROMPT_BUDGET_OVERSHOOT_TOLERANCE_TOKENS:-32}"
PROMPT_BUDGET_TARGET_FIELD="${PROMPT_BUDGET_TARGET_FIELD:-prompt_token_count}"
PROMPT_BUDGET_MISSING_REFERENCE="${PROMPT_BUDGET_MISSING_REFERENCE:-error}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MMR_LAMBDA="${MMR_LAMBDA:-0.70}"
ALPHA_DENSE="${ALPHA_DENSE:-0.70}"
ALPHA_LEXICAL="${ALPHA_LEXICAL:-0.20}"
ALPHA_BM25="${ALPHA_BM25:-0.10}"
SEMANTIC_THETA="${SEMANTIC_THETA:-0.5}"
ABC_BOUNDARY_MODE="${ABC_BOUNDARY_MODE:-local_peak}"
ABC_LAMBDA_STD="${ABC_LAMBDA_STD:-0.5}"
ABC_W_SEM="${ABC_W_SEM:-0.75}"
ABC_W_REL="${ABC_W_REL:-0.25}"
ABC_MAX_SENT_PER_CHUNK="${ABC_MAX_SENT_PER_CHUNK:-3}"
ABC_MAX_TOKENS_PER_CHUNK="${ABC_MAX_TOKENS_PER_CHUNK:-150}"
ABC_MIN_TOKENS_PER_CHUNK="${ABC_MIN_TOKENS_PER_CHUNK:-20}"
ABC_SINGLE_SENTENCE_RELEVANCE_THRESHOLD="${ABC_SINGLE_SENTENCE_RELEVANCE_THRESHOLD:-0.55}"
ABC_HIGH_REL_THRESHOLD="${ABC_HIGH_REL_THRESHOLD:-0.70}"
ABC_COREF_BOUNDARY_DISCOUNT="${ABC_COREF_BOUNDARY_DISCOUNT:-0.10}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz8_ga1.json}"
RETRIEVAL_NUM_GPUS="${RETRIEVAL_NUM_GPUS:-${NPROC_PER_NODE}}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/phase10_chunking_ablation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_BASE}/runs}"
CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_BASE}/cache}"
TRACKING_ENABLED="${TRACKING_ENABLED:-true}"
DRY_RUN="${DRY_RUN:-false}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-}"

FORCE_BUILD="${FORCE_BUILD:-false}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"
CHECKPOINT="${CHECKPOINT:-best}"
LABEL_TOKEN_EVAL="${LABEL_TOKEN_EVAL:-true}"
LABEL_TOKEN_SPLITS="${LABEL_TOKEN_SPLITS:-test}"
LABEL_TOKEN_LOG_PREDICTIONS="${LABEL_TOKEN_LOG_PREDICTIONS:-0}"
FORCE_LABEL_TOKEN_EVAL="${FORCE_LABEL_TOKEN_EVAL:-false}"

case "${MODE}" in
  build|train) ;;
  full|infer)
    echo "[chunking-ablation] MODE=${MODE} would run the pipeline API/vLLM infer phase." >&2
    echo "[chunking-ablation] Use MODE=train with LABEL_TOKEN_EVAL=true for aligned label-token metrics." >&2
    exit 2
    ;;
  *)
    echo "[chunking-ablation] MODE must be build or train, got: ${MODE}" >&2
    exit 2
    ;;
esac

case "${DATASET}" in
  liar_raw)
    BASE_EXPERIMENT="${BASE_EXPERIMENT:-b3_label_token_ce_1024}"
    LABEL_SCHEMA="liar6"
    TRAIN_PATH="${TRAIN_PATH:-data/raw/LIAR-RAW/train.json}"
    VAL_PATH="${VAL_PATH:-data/raw/LIAR-RAW/val.json}"
    TEST_PATH="${TEST_PATH:-data/raw/LIAR-RAW/test.json}"
    ;;
  rawfc)
    BASE_EXPERIMENT="${BASE_EXPERIMENT:-v0_6c_rawfc3_rule_step_adaptive5_10_eval25}"
    LABEL_SCHEMA="rawfc3"
    TRAIN_PATH="${TRAIN_PATH:-data/raw/RAWFC/train.json}"
    VAL_PATH="${VAL_PATH:-data/raw/RAWFC/val.json}"
    TEST_PATH="${TEST_PATH:-data/raw/RAWFC/test.json}"
    DEFAULT_PROMPT_BUDGET_REFERENCE_BUILD_DIR="outputs/selector_trace_verifier/rawfc_v0_6c_eval25_backbone/v0_6c_rawfc3_rule_step_adaptive5_10_eval25_${BACKBONE}_lora"
    ;;
esac

PROMPT_BUDGET_REFERENCE_BUILD_DIR="${PROMPT_BUDGET_REFERENCE_BUILD_DIR:-${DEFAULT_PROMPT_BUDGET_REFERENCE_BUILD_DIR:-}}"

case "${SELECTION_METHOD}" in
  mmr_prompt_budget|prompt_budget_mmr|adaptive_budget_mmr)
    CASE_BUDGET_SUFFIX="budget_adaptive5_10_pool${CANDIDATE_POOL_K}_min${PROMPT_BUDGET_MIN_K}_max${PROMPT_BUDGET_MAX_K}"
    if [[ -z "${PROMPT_BUDGET_REFERENCE_BUILD_DIR}" ]]; then
      echo "[chunking-ablation] PROMPT_BUDGET_REFERENCE_BUILD_DIR is required for ${SELECTION_METHOD}" >&2
      exit 2
    fi
    ;;
  *)
    CASE_BUDGET_SUFFIX="top${TOP_K}"
    ;;
esac

CASE_NAME="${CASE_NAME:-${DATASET}_${CHUNKING}_${BACKBONE}_lora_${CASE_BUDGET_SUFFIX}_max${MAX_LENGTH}_hybrid_orig}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-chunking_granularity_${DATASET}}"

resolve_case_run_dir() {
  if [[ -n "${RUN_DIR}" ]]; then
    printf "%s\n" "${RUN_DIR}"
    return 0
  fi

  local search_root="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
  local matches=()
  if [[ ! -d "${search_root}" ]]; then
    echo "[chunking-ablation] cannot resolve run dir; missing search root: ${search_root}" >&2
    return 1
  fi
  mapfile -t matches < <(find "${search_root}" -maxdepth 1 -type d -name "${CASE_NAME}__*" | sort)
  if [[ "${#matches[@]}" -eq 0 ]]; then
    echo "[chunking-ablation] cannot resolve run dir under ${search_root} for ${CASE_NAME}" >&2
    return 1
  fi
  printf "%s\n" "${matches[$((${#matches[@]} - 1))]}"
}

print_label_token_eval_commands() {
  local case_run_dir="$1"
  local split_array=()
  local raw_split
  local split
  IFS=',' read -r -a split_array <<< "${LABEL_TOKEN_SPLITS}"
  for raw_split in "${split_array[@]}"; do
    split="${raw_split//[[:space:]]/}"
    if [[ -z "${split}" ]]; then
      continue
    fi
    local eval_dir="${case_run_dir}/eval/${split}/${CHECKPOINT}/label_token"
    local eval_cmd=(
      "${PYTHON_BIN}" -m sft.label_token_infer
      --run-dir "${case_run_dir}/train"
      --checkpoint "${CHECKPOINT}"
      --split "${split}"
      --config "${case_run_dir}/configs/train.resolved.yaml"
      --output-dir "${eval_dir}"
      --log-predictions "${LABEL_TOKEN_LOG_PREDICTIONS}"
    )
    printf '[chunking-ablation] label-token command:'
    printf ' %q' "${eval_cmd[@]}"
    printf '\n'
  done
}

run_label_token_eval() {
  local case_run_dir="$1"
  local split_array=()
  local raw_split
  local split
  IFS=',' read -r -a split_array <<< "${LABEL_TOKEN_SPLITS}"
  for raw_split in "${split_array[@]}"; do
    split="${raw_split//[[:space:]]/}"
    if [[ -z "${split}" ]]; then
      continue
    fi
    local eval_dir="${case_run_dir}/eval/${split}/${CHECKPOINT}/label_token"
    local metrics_path="${eval_dir}/metrics.json"
    if [[ -s "${metrics_path}" ]] && ! is_true "${FORCE_LABEL_TOKEN_EVAL}"; then
      echo "[chunking-ablation] label-token eval: reuse ${metrics_path}"
      continue
    fi
    local eval_cmd=(
      "${PYTHON_BIN}" -m sft.label_token_infer
      --run-dir "${case_run_dir}/train"
      --checkpoint "${CHECKPOINT}"
      --split "${split}"
      --config "${case_run_dir}/configs/train.resolved.yaml"
      --output-dir "${eval_dir}"
      --log-predictions "${LABEL_TOKEN_LOG_PREDICTIONS}"
    )
    echo "[chunking-ablation] label-token eval: split=${split} checkpoint=${CHECKPOINT}"
    printf '[chunking-ablation] label-token command:'
    printf ' %q' "${eval_cmd[@]}"
    printf '\n'
    "${eval_cmd[@]}"
  done
}

if [[ ! -d "${MODEL_PATH}" && "${DRY_RUN}" != "true" ]]; then
  echo "[chunking-ablation] missing model path: ${MODEL_PATH}" >&2
  exit 1
fi

for path in "${TRAIN_PATH}" "${VAL_PATH}" "${TEST_PATH}"; do
  if [[ ! -s "${path}" && "${DRY_RUN}" != "true" ]]; then
    echo "[chunking-ablation] missing data file: ${path}" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES

cmd=(
  "${PYTHON_BIN}" -m fact_checking.pipeline.run
  "experiment=${BASE_EXPERIMENT}"
  "experiment.name=${EXPERIMENT_NAME}"
  "baseline.variant=${CASE_NAME}"
  "baseline.chunking_strategy=${CHUNKING}"
  "++baseline.label_schema=${LABEL_SCHEMA}"
  "++label_schema=${LABEL_SCHEMA}"
  "pipeline.mode=${MODE}"
  "pipeline.output_root=${OUTPUT_ROOT}"
  "pipeline.cache_root=${CACHE_ROOT}"
  "pipeline.output_subdir=${CASE_NAME}"
  "pipeline.force.build=${FORCE_BUILD}"
  "pipeline.force.train=${FORCE_TRAIN}"
  "++build.data.dataset=${DATASET}"
  "++build.data.label_schema=${LABEL_SCHEMA}"
  "++build.cache_root=${CACHE_ROOT}"
  "build.data.train_path=${TRAIN_PATH}"
  "build.data.val_path=${VAL_PATH}"
  "build.data.test_path=${TEST_PATH}"
  "build.retrieval.selection_method=${SELECTION_METHOD}"
  "build.retrieval.top_k=${TOP_K}"
  "build.retrieval.mmr_lambda=${MMR_LAMBDA}"
  "build.retrieval.alpha_dense=${ALPHA_DENSE}"
  "build.retrieval.alpha_lexical=${ALPHA_LEXICAL}"
  "build.retrieval.alpha_bm25=${ALPHA_BM25}"
  "build.retrieval.num_gpus=${RETRIEVAL_NUM_GPUS}"
  "build.retrieval.chunking.strategy=${CHUNKING}"
  "build.prompt.model_name_or_path=${MODEL_PATH}"
  "build.prompt.auto_length=true"
  "build.prompt.max_length=${MAX_LENGTH}"
  "build.prompt.output_mode=label_only"
  "build.prompt.label_format=letter"
  "++build.prompt.label_schema=${LABEL_SCHEMA}"
  "train.model_name_or_path=${MODEL_PATH}"
  "train.backend=accelerate_deepspeed"
  "train.cuda_visible_devices='${CUDA_VISIBLE_DEVICES}'"
  "train.nproc_per_node=${NPROC_PER_NODE}"
  "train.num_machines=${NUM_MACHINES}"
  "train.mixed_precision=${MIXED_PRECISION}"
  "train.deepspeed_config=${DEEPSPEED_CONFIG}"
  "train.kind=label_token_ce"
  "train.checkpoint_for_infer=${CHECKPOINT}"
  "sft_train.max_length=${MAX_LENGTH}"
  "sft_train.max_new_tokens=1"
  "++sft_train.label_schema=${LABEL_SCHEMA}"
  "sft_train.lora.enabled=true"
  "tracking.enabled=${TRACKING_ENABLED}"
  "swanlab.experiment_name=${CASE_NAME}"
)

case "${CHUNKING}" in
  semantic)
    cmd+=("build.retrieval.chunking.theta=${SEMANTIC_THETA}")
    ;;
  abc_claim_aware)
    cmd+=(
      "build.retrieval.chunking.boundary_mode=${ABC_BOUNDARY_MODE}"
      "build.retrieval.chunking.lambda_std=${ABC_LAMBDA_STD}"
      "build.retrieval.chunking.w_sem=${ABC_W_SEM}"
      "build.retrieval.chunking.w_rel=${ABC_W_REL}"
      "build.retrieval.chunking.max_sent_per_chunk=${ABC_MAX_SENT_PER_CHUNK}"
      "build.retrieval.chunking.max_tokens_per_chunk=${ABC_MAX_TOKENS_PER_CHUNK}"
      "build.retrieval.chunking.min_tokens_per_chunk=${ABC_MIN_TOKENS_PER_CHUNK}"
      "build.retrieval.chunking.allow_single_sentence_if_relevant=true"
      "build.retrieval.chunking.single_sentence_relevance_threshold=${ABC_SINGLE_SENTENCE_RELEVANCE_THRESHOLD}"
      "build.retrieval.chunking.high_rel_threshold=${ABC_HIGH_REL_THRESHOLD}"
      "build.retrieval.chunking.coref_boundary_discount=${ABC_COREF_BOUNDARY_DISCOUNT}"
    )
    ;;
esac

case "${SELECTION_METHOD}" in
  mmr_prompt_budget|prompt_budget_mmr|adaptive_budget_mmr)
    cmd+=(
      "++build.retrieval.prompt_budget.reference_build_dir=${PROMPT_BUDGET_REFERENCE_BUILD_DIR}"
      "++build.retrieval.prompt_budget.target_field=${PROMPT_BUDGET_TARGET_FIELD}"
      "++build.retrieval.prompt_budget.candidate_pool_k=${CANDIDATE_POOL_K}"
      "++build.retrieval.prompt_budget.min_k=${PROMPT_BUDGET_MIN_K}"
      "++build.retrieval.prompt_budget.max_k=${PROMPT_BUDGET_MAX_K}"
      "++build.retrieval.prompt_budget.overshoot_tolerance_tokens=${PROMPT_BUDGET_OVERSHOOT_TOLERANCE_TOKENS}"
      "++build.retrieval.prompt_budget.missing_reference=${PROMPT_BUDGET_MISSING_REFERENCE}"
    )
    ;;
esac

if [[ -n "${RUN_DIR}" ]]; then
  cmd+=("pipeline.run_dir=${RUN_DIR}")
fi

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "[chunking-ablation] dataset/chunking : ${DATASET}/${CHUNKING}"
echo "[chunking-ablation] case             : ${CASE_NAME}"
echo "[chunking-ablation] base experiment  : ${BASE_EXPERIMENT}"
echo "[chunking-ablation] model            : ${MODEL_PATH}"
echo "[chunking-ablation] python           : ${PYTHON_BIN}"
echo "[chunking-ablation] selection        : ${SELECTION_METHOD}"
echo "[chunking-ablation] top_k/pool/maxlen: ${TOP_K}/${CANDIDATE_POOL_K}/${MAX_LENGTH}"
case "${SELECTION_METHOD}" in
  mmr_prompt_budget|prompt_budget_mmr|adaptive_budget_mmr)
    echo "[chunking-ablation] budget ref       : ${PROMPT_BUDGET_REFERENCE_BUILD_DIR}"
    echo "[chunking-ablation] budget min/max   : ${PROMPT_BUDGET_MIN_K}/${PROMPT_BUDGET_MAX_K} (+${PROMPT_BUDGET_OVERSHOOT_TOLERANCE_TOKENS} tok)"
    ;;
esac
echo "[chunking-ablation] alpha            : ${ALPHA_DENSE}/${ALPHA_LEXICAL}/${ALPHA_BM25}"
echo "[chunking-ablation] output base      : ${OUTPUT_BASE}"
echo "[chunking-ablation] output root      : ${OUTPUT_ROOT}"
echo "[chunking-ablation] cache root       : ${CACHE_ROOT}"
echo "[chunking-ablation] mode             : ${MODE}"
echo "[chunking-ablation] label eval       : ${LABEL_TOKEN_EVAL} (${LABEL_TOKEN_SPLITS}/${CHECKPOINT})"

if is_true "${DRY_RUN}"; then
  printf '[chunking-ablation] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ "${MODE}" == "train" ]] && is_true "${LABEL_TOKEN_EVAL}"; then
    print_label_token_eval_commands "${RUN_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}/${CASE_NAME}__<fingerprint>}"
  fi
  exit 0
fi

"${cmd[@]}"

if [[ "${MODE}" == "train" ]] && is_true "${LABEL_TOKEN_EVAL}"; then
  CASE_RUN_DIR="$(resolve_case_run_dir)"
  echo "[chunking-ablation] resolved run dir: ${CASE_RUN_DIR}"
  run_label_token_eval "${CASE_RUN_DIR}"
fi
