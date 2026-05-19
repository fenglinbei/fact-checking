#!/usr/bin/env bash
# Evaluate the oracle-direct verifier checkpoint on non-oracle val evidence:
#   1) fixed-MMR sentence evidence
#   2) current Stage2 pointwise-selected sentence evidence
#
# Defaults run both best and checkpoint-600. Use CHECKPOINTS=best to avoid
# the duplicate check when best is already known to be step 600.
#
# Examples:
#   DRY_RUN=true bash scripts/verifier/run_oracle_direct_val_evidence_checks.sh
#   CHECKPOINTS=best bash scripts/verifier/run_oracle_direct_val_evidence_checks.sh
#   RUN_POINTWISE=false CHECKPOINTS=checkpoint-600 bash scripts/verifier/run_oracle_direct_val_evidence_checks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DIRECT_VERIFIER_RUN_DIR="${DIRECT_VERIFIER_RUN_DIR:-outputs/oracle_direct_verifier/stage2_sentence/train/b3_oracle_sentence_direct_verifier_1024_20260519-200709}"
POINTWISE_MODEL_DIR="${POINTWISE_MODEL_DIR:-outputs/oracle_pointwise/stage2_margin_sentence/logreg}"

CHECKPOINTS="${CHECKPOINTS:-best,checkpoint-600}"
SPLIT="${SPLIT:-val}"
RUN_FIXED_MMR="${RUN_FIXED_MMR:-true}"
RUN_POINTWISE="${RUN_POINTWISE:-true}"
DRY_RUN="${DRY_RUN:-false}"

TOP_K="${TOP_K:-5}"
FIXED_MMR_LAMBDA="${FIXED_MMR_LAMBDA:-0.70}"
POINTWISE_CANDIDATE_POOL_SIZE="${POINTWISE_CANDIDATE_POOL_SIZE:-15}"
POINTWISE_CANDIDATE_POOL_MULTIPLIER="${POINTWISE_CANDIDATE_POOL_MULTIPLIER:-3}"
POINTWISE_STRICT_FINGERPRINT="${POINTWISE_STRICT_FINGERPRINT:-true}"

FORCE_BUILD="${FORCE_BUILD:-false}"
FORCE_INFER="${FORCE_INFER:-true}"
PIPELINE_RESUME="${PIPELINE_RESUME:-true}"
OUTPUT_SUBDIR_PREFIX="${OUTPUT_SUBDIR_PREFIX:-oracle_direct_val}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
BUILD_NUM_GPUS="${BUILD_NUM_GPUS:-${TENSOR_PARALLEL_SIZE}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
PORT_BASE="${PORT_BASE:-35100}"
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
    echo "[oracle-direct-val] missing ${message}: ${path}" >&2
    exit 1
  fi
}

run_case() {
  local label="$1"
  local experiment="$2"
  local checkpoint="$3"
  local port="$4"
  shift 4

  local checkpoint_slug=""
  checkpoint_slug="$(slugify "${checkpoint}")"
  local output_subdir="${OUTPUT_SUBDIR_PREFIX}_${label}_${checkpoint_slug}"

  local cmd=(
    python -m fact_checking.pipeline.run
    "experiment=${experiment}"
    "pipeline.steps=[build,infer]"
    "pipeline.resume=${PIPELINE_RESUME}"
    "pipeline.force.build=${FORCE_BUILD}"
    "pipeline.force.infer=${FORCE_INFER}"
    "pipeline.output_subdir=${output_subdir}"
    "$(hydra_string_override train.run_dir "${DIRECT_VERIFIER_RUN_DIR}")"
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
    "build.retrieval.top_k=${TOP_K}"
    "build.retrieval.num_gpus=${BUILD_NUM_GPUS}"
    "$@"
  )

  echo "[oracle-direct-val] case=${label} checkpoint=${checkpoint} port=${port}"
  echo "[oracle-direct-val] output_subdir=${output_subdir}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[oracle-direct-val] dry-run command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  "${cmd[@]}"
}

main() {
  local extra_overrides=("$@")
  local checkpoints=()
  local checkpoint=""
  local run_index=0
  split_csv "${CHECKPOINTS}" checkpoints
  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "[oracle-direct-val] CHECKPOINTS is empty" >&2
    exit 1
  fi

  require_path "${DIRECT_VERIFIER_RUN_DIR}" "direct verifier train run dir"
  for checkpoint in "${checkpoints[@]}"; do
    require_path "${DIRECT_VERIFIER_RUN_DIR}/${checkpoint}" "checkpoint ${checkpoint}"
  done
  if [[ "${RUN_POINTWISE}" == "true" ]]; then
    require_path "${POINTWISE_MODEL_DIR}/model.npz" "pointwise selector model"
  fi

  echo "[oracle-direct-val] direct_verifier_run_dir=${DIRECT_VERIFIER_RUN_DIR}"
  echo "[oracle-direct-val] split=${SPLIT} checkpoints=${checkpoints[*]}"
  echo "[oracle-direct-val] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  for checkpoint in "${checkpoints[@]}"; do
    if [[ "${RUN_FIXED_MMR}" == "true" ]]; then
      run_case \
        "fixed_mmr_sentence" \
        "b3_oracle_direct_eval_fixed_mmr_sentence_1024" \
        "${checkpoint}" \
        "$((PORT_BASE + run_index))" \
        "baseline.variant=b3_oracle_direct_eval_fixed_mmr_sentence_1024" \
        "build.retrieval.selection_method=mmr" \
        "build.retrieval.mmr_lambda=${FIXED_MMR_LAMBDA}" \
        "build.retrieval.chunking.strategy=sentence" \
        "build.retrieval.chunking.context_k=1" \
        "build.retrieval.chunking.theta=0.7" \
        "build.retrieval.chunking.embedder_model=null" \
        "build.retrieval.chunking.device=cpu" \
        "build.retrieval.chunking.max_length=256" \
        "build.retrieval.chunking.batch_size=64" \
        "build.retrieval.chunking.precision=fp32" \
        "${extra_overrides[@]}"
      run_index=$((run_index + 1))
    fi

    if [[ "${RUN_POINTWISE}" == "true" ]]; then
      run_case \
        "pointwise_sentence" \
        "b3_oracle_direct_eval_pointwise_sentence_1024" \
        "${checkpoint}" \
        "$((PORT_BASE + run_index))" \
        "baseline.variant=b3_oracle_direct_eval_pointwise_sentence_1024" \
        "build.retrieval.selection_method=pointwise_oracle" \
        "$(hydra_string_override build.retrieval.pointwise_oracle.model_dir "${POINTWISE_MODEL_DIR}")" \
        "build.retrieval.pointwise_oracle.candidate_pool_size=${POINTWISE_CANDIDATE_POOL_SIZE}" \
        "build.retrieval.pointwise_oracle.candidate_pool_multiplier=${POINTWISE_CANDIDATE_POOL_MULTIPLIER}" \
        "build.retrieval.pointwise_oracle.strict_fingerprint=${POINTWISE_STRICT_FINGERPRINT}" \
        "build.retrieval.chunking.strategy=sentence" \
        "build.retrieval.chunking.context_k=1" \
        "build.retrieval.chunking.theta=0.7" \
        "build.retrieval.chunking.embedder_model=null" \
        "build.retrieval.chunking.device=cpu" \
        "build.retrieval.chunking.max_length=256" \
        "build.retrieval.chunking.batch_size=64" \
        "build.retrieval.chunking.precision=fp32" \
        "${extra_overrides[@]}"
      run_index=$((run_index + 1))
    fi
  done
}

main "$@"
