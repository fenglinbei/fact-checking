#!/usr/bin/env bash
# Inference-only oracle evidence order sensitivity experiment.
#
# The script keeps the oracle-selected evidence set fixed and changes only the
# order in which the selected evidence is rendered into verifier prompts.
# It then reuses the trained oracle-direct verifier checkpoint for val
# inference. No verifier training is launched.
#
# Default cases:
#   oracle
#   hybrid
#   candidate_pool
#   random_seed0..4
#
# Examples:
#   DRY_RUN=true bash scripts/verifier/run_oracle_direct_order_sensitivity.sh
#   CHECKPOINTS=best ORDERS=oracle,hybrid,candidate_pool bash scripts/verifier/run_oracle_direct_order_sensitivity.sh
#   RANDOM_SEEDS=0,1,2,3,4,5,6,7,8,9 bash scripts/verifier/run_oracle_direct_order_sensitivity.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG="${CONFIG:-configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml}"
ORACLE_VAL="${ORACLE_VAL:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
VAL_RAW="${VAL_RAW:-data/raw/LIAR-RAW/val.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/oracle_direct_verifier/stage2_sentence_order_sensitivity}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/b3_oracle_direct_order_sensitivity}"

DIRECT_VERIFIER_RUN_DIR="${DIRECT_VERIFIER_RUN_DIR:-outputs/oracle_direct_verifier/stage2_sentence/train/b3_oracle_sentence_direct_verifier_1024_20260519-200709}"
EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/data/models/}"
PROMPT_MODEL_NAME_OR_PATH="${PROMPT_MODEL_NAME_OR_PATH:-}"
TRAIN_MODEL_NAME_OR_PATH="${TRAIN_MODEL_NAME_OR_PATH:-}"
FILTER="${FILTER:-all}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

ORDERS="${ORDERS:-oracle,hybrid,candidate_pool,random}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0,1,2,3,4}"
CHECKPOINTS="${CHECKPOINTS:-best}"
SPLIT="${SPLIT:-val}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_BUILD="${FORCE_BUILD:-false}"
FORCE_INFER="${FORCE_INFER:-true}"
PIPELINE_RESUME="${PIPELINE_RESUME:-true}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
PORT_BASE="${PORT_BASE:-35200}"
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
  printf "%s" "${raw}"
}

require_path() {
  local path="$1"
  local message="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[order-sensitivity] missing ${message}: ${path}" >&2
    exit 1
  fi
}

build_case() {
  local order="$1"
  local seed="$2"
  local label="$3"
  local output_dir="$4"

  if [[ "${FORCE_BUILD}" != "true" && -f "${output_dir}/train.resolved.yaml" && -f "${output_dir}/build_report.json" ]]; then
    echo "[order-sensitivity] reuse build case=${label} dir=${output_dir}"
    return 0
  fi

  local args=(
    --config "${CONFIG}"
    --val-oracle-results "${ORACLE_VAL}"
    --val-raw "${VAL_RAW}"
    --output-dir "${output_dir}"
    --expected-chunk-mmr-fingerprint "${EXPECTED_CHUNK_MMR_FINGERPRINT}"
    --model-base-path "${MODEL_BASE_PATH}"
    --order "${order}"
    --order-seed "${seed}"
    --filter "${FILTER}"
    --val-only
  )
  if [[ -n "${PROMPT_MODEL_NAME_OR_PATH}" ]]; then
    args+=(--prompt-model-name-or-path "${PROMPT_MODEL_NAME_OR_PATH}")
  fi
  if [[ -n "${TRAIN_MODEL_NAME_OR_PATH}" ]]; then
    args+=(--train-model-name-or-path "${TRAIN_MODEL_NAME_OR_PATH}")
  fi
  if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
    args+=(--sample-limit "${SAMPLE_LIMIT}")
  fi

  echo "[order-sensitivity] build case=${label} order=${order} seed=${seed} output=${output_dir}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[order-sensitivity] dry-run build: python scripts/oracle_evidence/build_oracle_direct_verifier_data.py'
    printf ' %q' "${args[@]}"
    printf '\n'
    return 0
  fi
  python scripts/oracle_evidence/build_oracle_direct_verifier_data.py "${args[@]}"
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

  local cmd=(
    python -m fact_checking.pipeline.run
    "experiment=b3_oracle_sentence_direct_verifier_1024"
    "pipeline.mode=infer"
    "pipeline.resume=${PIPELINE_RESUME}"
    "pipeline.force.infer=${FORCE_INFER}"
    "$(hydra_string_override pipeline.run_dir "${run_dir}")"
    "$(hydra_string_override train.run_dir "${DIRECT_VERIFIER_RUN_DIR}")"
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

  echo "[order-sensitivity] infer case=${label} checkpoint=${checkpoint} port=${port} run_dir=${run_dir}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[order-sensitivity] dry-run infer:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  "${cmd[@]}"
}

case_labels() {
  local -n out_labels="$1"
  local -n out_orders="$2"
  local -n out_seeds="$3"
  local parsed_orders=()
  local parsed_seeds=()
  local order=""
  local seed=""

  split_csv "${ORDERS}" parsed_orders
  split_csv "${RANDOM_SEEDS}" parsed_seeds
  out_labels=()
  out_orders=()
  out_seeds=()
  for order in "${parsed_orders[@]}"; do
    if [[ "${order}" == "random" ]]; then
      for seed in "${parsed_seeds[@]}"; do
        out_labels+=("random_seed${seed}")
        out_orders+=("random")
        out_seeds+=("${seed}")
      done
    else
      out_labels+=("${order}")
      out_orders+=("${order}")
      out_seeds+=("0")
    fi
  done
}

main() {
  local checkpoints=()
  local labels=()
  local orders=()
  local seeds=()
  local checkpoint=""
  local idx=0
  local port_index=0

  split_csv "${CHECKPOINTS}" checkpoints
  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "[order-sensitivity] CHECKPOINTS is empty" >&2
    exit 1
  fi
  case_labels labels orders seeds
  if [[ "${#labels[@]}" -eq 0 ]]; then
    echo "[order-sensitivity] no order cases resolved from ORDERS=${ORDERS}" >&2
    exit 1
  fi

  require_path "${ORACLE_VAL}" "val oracle results"
  require_path "${VAL_RAW}" "val raw split"
  require_path "${DIRECT_VERIFIER_RUN_DIR}" "direct verifier train run dir"
  for checkpoint in "${checkpoints[@]}"; do
    require_path "${DIRECT_VERIFIER_RUN_DIR}/${checkpoint}" "checkpoint ${checkpoint}"
  done

  echo "[order-sensitivity] output_root=${OUTPUT_ROOT}"
  echo "[order-sensitivity] run_root=${RUN_ROOT}"
  echo "[order-sensitivity] direct_verifier_run_dir=${DIRECT_VERIFIER_RUN_DIR}"
  echo "[order-sensitivity] cases=${labels[*]}"
  echo "[order-sensitivity] checkpoints=${checkpoints[*]}"

  for idx in "${!labels[@]}"; do
    local label="${labels[$idx]}"
    local order="${orders[$idx]}"
    local seed="${seeds[$idx]}"
    local build_dir="${OUTPUT_ROOT}/${label}"
    build_case "${order}" "${seed}" "${label}" "${build_dir}"
    for checkpoint in "${checkpoints[@]}"; do
      run_infer_case "${label}" "${checkpoint}" "${build_dir}" "$((PORT_BASE + port_index))"
      port_index=$((port_index + 1))
    done
  done
}

main "$@"
