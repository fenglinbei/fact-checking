#!/usr/bin/env bash
# Build verifier data from selector/control sources, train a LoRA verifier, and
# run inference. This is the full build-data -> train -> infer workflow; it
# does not reuse the oracle-direct verifier checkpoint by default.
#
# Default case:
#   hybrid_score_top5 from Stage2 oracle candidate_pool train/val rows
#
# Examples:
#   DRY_RUN=true bash scripts/phase5_selectors/run_selector_trace_full_pipeline.sh
#
#   CASE_NAME=deberta_pairwise_same_set_random SOURCE_TYPE=trace \
#   TRAIN_SOURCE=outputs/selectors/stage2_sentence_cross_encoder/deberta_pairwise/eval_train/selection_trace.jsonl \
#   VAL_SOURCE=outputs/selectors/stage2_sentence_cross_encoder/deberta_pairwise/eval_val/selection_trace.jsonl \
#   TRACE_SELECTION_MODE=same_set_random_order \
#   EXPECTED_SELECTOR_NAME=cross_encoder_pairwise \
#   RANDOM_SEEDS=0,1,2,3,4 \
#   bash scripts/phase5_selectors/run_selector_trace_full_pipeline.sh
#
#   CASE_NAME=deberta_listwise_shuffle03 SOURCE_TYPE=trace \
#   TRAIN_SOURCE=outputs/selectors/stage2_sentence_listwise/deberta_listwise_shuffle03/eval_train/selection_trace.jsonl \
#   VAL_SOURCE=outputs/selectors/stage2_sentence_listwise/deberta_listwise_shuffle03/eval_val/selection_trace.jsonl \
#   TRACE_SELECTION_MODE=trace \
#   EXPECTED_SELECTOR_NAME=set_aware_listwise \
#   bash scripts/phase5_selectors/run_selector_trace_full_pipeline.sh
#
# CASE_SPECS can run heterogeneous cases in one invocation:
#   label|source_type|train_source|val_source|selection_mode|seed|expected_selector_name
# Multiple specs are separated by semicolons.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG="${CONFIG:-configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml}"
TRAIN_RAW="${TRAIN_RAW:-data/raw/LIAR-RAW/train.json}"
VAL_RAW="${VAL_RAW:-data/raw/LIAR-RAW/val.json}"
TEST_RAW="${TEST_RAW:-data/raw/LIAR-RAW/test.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_trace_verifier/stage2_sentence}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/b3_selector_trace_full_pipeline}"

ORACLE_TRAIN="${ORACLE_TRAIN:-outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl}"
ORACLE_VAL="${ORACLE_VAL:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/data/models/}"
PROMPT_MODEL_NAME_OR_PATH="${PROMPT_MODEL_NAME_OR_PATH:-}"
TRAIN_MODEL_NAME_OR_PATH="${TRAIN_MODEL_NAME_OR_PATH:-}"

CASE_NAME="${CASE_NAME:-hybrid_score_top5}"
SOURCE_TYPE="${SOURCE_TYPE:-oracle_results}"
TRAIN_SOURCE="${TRAIN_SOURCE:-${TRAIN_TRACE_PATH:-${ORACLE_TRAIN}}}"
VAL_SOURCE="${VAL_SOURCE:-${VAL_TRACE_PATH:-${TRACE_PATH:-${ORACLE_VAL}}}}"
TEST_SOURCE="${TEST_SOURCE:-}"
TRACE_SELECTION_MODE="${TRACE_SELECTION_MODE:-hybrid_score_topk}"
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-}"
RANDOM_SEED="${RANDOM_SEED:-0}"
RANDOM_SEEDS="${RANDOM_SEEDS:-${RANDOM_SEED}}"
CASE_SPECS="${CASE_SPECS:-}"

TOP_K="${TOP_K:-5}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
CHECKPOINTS="${CHECKPOINTS:-best}"
SPLIT="${SPLIT:-val}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_BUILD="${FORCE_BUILD:-false}"
RUN_TRAIN="${RUN_TRAIN:-true}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"
RUN_INFER="${RUN_INFER:-true}"
FORCE_INFER="${FORCE_INFER:-true}"
PIPELINE_RESUME="${PIPELINE_RESUME:-true}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz8_ga1.json}"
TRAIN_BACKEND="${TRAIN_BACKEND:-accelerate_deepspeed}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
PORT_BASE="${PORT_BASE:-35300}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-120}"
STOP_AFTER_INFER="${STOP_AFTER_INFER:-true}"

MERGE_LORA_CACHE="${MERGE_LORA_CACHE:-true}"
MERGE_LORA_CACHE_DIR="${MERGE_LORA_CACHE_DIR:-outputs/cache/merged_lora}"
MERGE_LORA_CACHE_FORCE_REBUILD="${MERGE_LORA_CACHE_FORCE_REBUILD:-false}"

export CUDA_VISIBLE_DEVICES

hydra_string_override() {
  local key="$1"
  local value="$2"
  value="${value//\\/\\\\}"
  value="${value//\'/\\\'}"
  printf "%s='%s'" "${key}" "${value}"
}

split_csv() {
  local raw="$1"
  local -n out_array="$2"
  local items=()
  local item=""
  IFS=',' read -r -a items <<< "${raw}"
  out_array=()
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      out_array+=("${item}")
    fi
  done
}

slugify() {
  local raw="$1"
  raw="${raw//\//_}"
  raw="${raw// /_}"
  raw="${raw//,/}"
  raw="${raw//:/_}"
  raw="${raw//|/_}"
  printf "%s" "${raw}"
}

require_path() {
  local path="$1"
  local message="$2"
  if [[ ! -e "${path}" ]]; then
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
      echo "[selector-trace-full] dry-run missing ${message}: ${path}" >&2
      return 0
    fi
    echo "[selector-trace-full] missing ${message}: ${path}" >&2
    exit 1
  fi
}

resolve_train_run_dir() {
  local train_root="$1"
  local checkpoint="$2"
  if [[ -d "${train_root}/${checkpoint}" ]]; then
    printf "%s" "${train_root}"
    return 0
  fi

  local resolved=""
  if [[ -d "${train_root}" ]]; then
    resolved="$(
      find "${train_root}" -mindepth 2 -maxdepth 2 -type d -name "${checkpoint}" -printf '%T@ %h\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )"
  fi
  if [[ -n "${resolved}" ]]; then
    printf "%s" "${resolved}"
    return 0
  fi

  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf "%s" "${train_root}"
    return 0
  fi

  echo "[selector-trace-full] checkpoint ${checkpoint} not found under ${train_root}" >&2
  echo "[selector-trace-full] expected either ${train_root}/${checkpoint} or ${train_root}/*/${checkpoint}" >&2
  exit 1
}

append_case() {
  local label="$1"
  local source_type="$2"
  local train_source="$3"
  local val_source="$4"
  local mode="$5"
  local seed="$6"
  local expected_selector="$7"
  local -n labels_ref="$8"
  local -n source_types_ref="$9"
  local -n train_sources_ref="${10}"
  local -n val_sources_ref="${11}"
  local -n modes_ref="${12}"
  local -n seeds_ref="${13}"
  local -n expected_ref="${14}"

  labels_ref+=("${label}")
  source_types_ref+=("${source_type}")
  train_sources_ref+=("${train_source}")
  val_sources_ref+=("${val_source}")
  modes_ref+=("${mode}")
  seeds_ref+=("${seed}")
  expected_ref+=("${expected_selector}")
}

resolve_cases() {
  local labels_name="$1"
  local source_types_name="$2"
  local train_sources_name="$3"
  local val_sources_name="$4"
  local modes_name="$5"
  local seeds_name="$6"
  local expected_name="$7"
  local -n labels_ref="${labels_name}"
  local -n source_types_ref="${source_types_name}"
  local -n train_sources_ref="${train_sources_name}"
  local -n val_sources_ref="${val_sources_name}"
  local -n modes_ref="${modes_name}"
  local -n seeds_ref="${seeds_name}"
  local -n expected_ref="${expected_name}"
  labels_ref=()
  source_types_ref=()
  train_sources_ref=()
  val_sources_ref=()
  modes_ref=()
  seeds_ref=()
  expected_ref=()

  if [[ -n "${CASE_SPECS}" ]]; then
    local specs=()
    local spec=""
    IFS=';' read -r -a specs <<< "${CASE_SPECS}"
    for spec in "${specs[@]}"; do
      [[ -z "${spec}" ]] && continue
      local fields=()
      IFS='|' read -r -a fields <<< "${spec}"
      if [[ "${#fields[@]}" -lt 5 ]]; then
        echo "[selector-trace-full] bad CASE_SPECS item: ${spec}" >&2
        exit 1
      fi
      append_case \
        "${fields[0]}" \
        "${fields[1]}" \
        "${fields[2]}" \
        "${fields[3]}" \
        "${fields[4]}" \
        "${fields[5]:-0}" \
        "${fields[6]:-}" \
        "${labels_name}" "${source_types_name}" "${train_sources_name}" "${val_sources_name}" \
        "${modes_name}" "${seeds_name}" "${expected_name}"
    done
    return 0
  fi

  local parsed_random_seeds=()
  local seed=""
  split_csv "${RANDOM_SEEDS}" parsed_random_seeds
  if [[ "${TRACE_SELECTION_MODE}" == "same_set_random_order" && "${#parsed_random_seeds[@]}" -gt 1 ]]; then
    for seed in "${parsed_random_seeds[@]}"; do
      append_case \
        "${CASE_NAME}_seed${seed}" \
        "${SOURCE_TYPE}" \
        "${TRAIN_SOURCE}" \
        "${VAL_SOURCE}" \
        "${TRACE_SELECTION_MODE}" \
        "${seed}" \
        "${EXPECTED_SELECTOR_NAME}" \
        "${labels_name}" "${source_types_name}" "${train_sources_name}" "${val_sources_name}" \
        "${modes_name}" "${seeds_name}" "${expected_name}"
    done
  else
    append_case \
      "${CASE_NAME}" \
      "${SOURCE_TYPE}" \
      "${TRAIN_SOURCE}" \
      "${VAL_SOURCE}" \
      "${TRACE_SELECTION_MODE}" \
      "${RANDOM_SEED}" \
      "${EXPECTED_SELECTOR_NAME}" \
      "${labels_name}" "${source_types_name}" "${train_sources_name}" "${val_sources_name}" \
      "${modes_name}" "${seeds_name}" "${expected_name}"
  fi
}

build_case() {
  local label="$1"
  local source_type="$2"
  local train_source="$3"
  local val_source="$4"
  local mode="$5"
  local seed="$6"
  local expected_selector="$7"
  local output_dir="$8"

  if [[ "${FORCE_BUILD}" != "true" && -f "${output_dir}/train.resolved.yaml" && -f "${output_dir}/build_report.json" ]]; then
    echo "[selector-trace-full] reuse build case=${label} dir=${output_dir}"
    return 0
  fi

  local args=(
    --config "${CONFIG}"
    --train-raw "${TRAIN_RAW}"
    --val-raw "${VAL_RAW}"
    --output-dir "${output_dir}"
    --selection-mode "${mode}"
    --top-k "${TOP_K}"
    --random-seed "${seed}"
    --expected-chunk-mmr-fingerprint "${EXPECTED_CHUNK_MMR_FINGERPRINT}"
    --model-base-path "${MODEL_BASE_PATH}"
  )
  if [[ "${source_type}" == "trace" ]]; then
    args+=(--train-trace "${train_source}" --val-trace "${val_source}")
  elif [[ "${source_type}" == "oracle_results" ]]; then
    args+=(--train-oracle-results "${train_source}" --val-oracle-results "${val_source}")
  else
    echo "[selector-trace-full] unsupported source_type=${source_type}; use trace or oracle_results" >&2
    exit 1
  fi
  if [[ -n "${TEST_SOURCE}" ]]; then
    args+=(--test-raw "${TEST_RAW}")
    if [[ "${source_type}" == "trace" ]]; then
      args+=(--test-trace "${TEST_SOURCE}")
    else
      args+=(--test-oracle-results "${TEST_SOURCE}")
    fi
  fi
  if [[ -n "${expected_selector}" ]]; then
    args+=(--expected-selector-name "${expected_selector}")
  fi
  if [[ -n "${PROMPT_MODEL_NAME_OR_PATH}" ]]; then
    args+=(--prompt-model-name-or-path "${PROMPT_MODEL_NAME_OR_PATH}")
  fi
  if [[ -n "${TRAIN_MODEL_NAME_OR_PATH}" ]]; then
    args+=(--train-model-name-or-path "${TRAIN_MODEL_NAME_OR_PATH}")
  fi
  if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
    args+=(--sample-limit "${SAMPLE_LIMIT}")
  fi

  echo "[selector-trace-full] build-data case=${label} source_type=${source_type} mode=${mode} seed=${seed} output=${output_dir}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[selector-trace-full] dry-run build: python scripts/phase5_selectors/build/build_trace_verifier_data.py'
    printf ' %q' "${args[@]}"
    printf '\n'
    return 0
  fi
  python scripts/phase5_selectors/build/build_trace_verifier_data.py "${args[@]}"
}

train_case() {
  local label="$1"
  local build_dir="$2"
  local train_dir="${build_dir}/train"
  local config_path="${build_dir}/train.resolved.yaml"

  if [[ "${RUN_TRAIN}" != "true" ]]; then
    echo "[selector-trace-full] skip train case=${label} RUN_TRAIN=${RUN_TRAIN}"
    return 0
  fi
  if [[ "${FORCE_TRAIN}" != "true" && -d "${train_dir}/best" ]]; then
    echo "[selector-trace-full] reuse train case=${label} checkpoint=${train_dir}/best"
    return 0
  fi

  local cmd=()
  if [[ "${TRAIN_BACKEND}" == "single" ]]; then
    cmd=(python -m sft.label_token_trainer --config "${config_path}")
  elif [[ "${TRAIN_BACKEND}" == "accelerate_deepspeed" ]]; then
    cmd=(
      accelerate launch
      "--num_processes=${NPROC_PER_NODE}"
      "--num_machines=${NUM_MACHINES}"
      "--mixed_precision=${MIXED_PRECISION}"
      --use_deepspeed
      --deepspeed_config_file "${DEEPSPEED_CONFIG}"
      -m sft.label_token_trainer
      --config "${config_path}"
    )
  else
    echo "[selector-trace-full] unsupported TRAIN_BACKEND=${TRAIN_BACKEND}" >&2
    exit 1
  fi

  echo "[selector-trace-full] train case=${label} backend=${TRAIN_BACKEND} config=${config_path}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[selector-trace-full] dry-run train:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  "${cmd[@]}"
}

run_infer_case() {
  local label="$1"
  local checkpoint="$2"
  local build_dir="$3"
  local port="$4"
  local checkpoint_slug=""
  checkpoint_slug="$(slugify "${checkpoint}")"
  local run_dir="${RUN_ROOT}/${label}_${checkpoint_slug}"
  local config_path="${build_dir}/train.resolved.yaml"
  local train_dir=""
  train_dir="$(resolve_train_run_dir "${build_dir}/train" "${checkpoint}")"

  if [[ "${RUN_INFER}" != "true" ]]; then
    echo "[selector-trace-full] skip infer case=${label} RUN_INFER=${RUN_INFER}"
    return 0
  fi

  local cmd=(
    python -m fact_checking.pipeline.run
    "experiment=b3_oracle_sentence_direct_verifier_1024"
    "pipeline.mode=infer"
    "pipeline.resume=${PIPELINE_RESUME}"
    "pipeline.force.infer=${FORCE_INFER}"
    "$(hydra_string_override pipeline.run_dir "${run_dir}")"
    "$(hydra_string_override train.run_dir "${train_dir}")"
    "$(hydra_string_override infer.config_path "${config_path}")"
    "$(hydra_string_override infer.split "${SPLIT}")"
    "$(hydra_string_override infer.checkpoint "${checkpoint}")"
    "infer.port=${port}"
    "$(hydra_string_override infer.cuda_visible_devices "${CUDA_VISIBLE_DEVICES}")"
    "infer.tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
    "infer.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "infer.wait_seconds=${WAIT_SECONDS}"
    "infer.request_timeout_seconds=${REQUEST_TIMEOUT_SECONDS}"
    "infer.server.stop_after_infer=${STOP_AFTER_INFER}"
    "infer.merge_lora_cache.enabled=${MERGE_LORA_CACHE}"
    "$(hydra_string_override infer.merge_lora_cache.dir "${MERGE_LORA_CACHE_DIR}")"
    "infer.merge_lora_cache.force_rebuild=${MERGE_LORA_CACHE_FORCE_REBUILD}"
  )

  echo "[selector-trace-full] infer case=${label} checkpoint=${checkpoint} train_dir=${train_dir} port=${port} run_dir=${run_dir}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[selector-trace-full] dry-run infer:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  "${cmd[@]}"
}

main() {
  local checkpoints=()
  local labels=()
  local source_types=()
  local train_sources=()
  local val_sources=()
  local modes=()
  local seeds=()
  local expected=()
  local checkpoint=""
  local idx=0
  local port_index=0

  split_csv "${CHECKPOINTS}" checkpoints
  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "[selector-trace-full] CHECKPOINTS is empty" >&2
    exit 1
  fi
  resolve_cases labels source_types train_sources val_sources modes seeds expected
  if [[ "${#labels[@]}" -eq 0 ]]; then
    echo "[selector-trace-full] no cases resolved" >&2
    exit 1
  fi

  require_path "${TRAIN_RAW}" "train raw split"
  require_path "${VAL_RAW}" "val raw split"
  for idx in "${!labels[@]}"; do
    require_path "${train_sources[$idx]}" "train source for case ${labels[$idx]}"
    require_path "${val_sources[$idx]}" "val source for case ${labels[$idx]}"
  done

  echo "[selector-trace-full] output_root=${OUTPUT_ROOT}"
  echo "[selector-trace-full] run_root=${RUN_ROOT}"
  echo "[selector-trace-full] train_backend=${TRAIN_BACKEND} nproc=${NPROC_PER_NODE}"
  echo "[selector-trace-full] cases=${labels[*]}"
  echo "[selector-trace-full] checkpoints=${checkpoints[*]}"

  for idx in "${!labels[@]}"; do
    local label="${labels[$idx]}"
    local source_type="${source_types[$idx]}"
    local train_source="${train_sources[$idx]}"
    local val_source="${val_sources[$idx]}"
    local mode="${modes[$idx]}"
    local seed="${seeds[$idx]}"
    local expected_selector="${expected[$idx]}"
    local build_dir="${OUTPUT_ROOT}/${label}"
    build_case "${label}" "${source_type}" "${train_source}" "${val_source}" "${mode}" "${seed}" "${expected_selector}" "${build_dir}"
    train_case "${label}" "${build_dir}"
    for checkpoint in "${checkpoints[@]}"; do
      run_infer_case "${label}" "${checkpoint}" "${build_dir}" "$((PORT_BASE + port_index))"
      port_index=$((port_index + 1))
    done
  done
}

main "$@"
