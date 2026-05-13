#!/usr/bin/env bash
# Rerun val-set inference for selected val/test gap checks.
#
# Default coverage:
#   - b3_mmr_topk_sweep_1024: top_k=3,5,6
#   - mmr_lambda_sweep_1024: lambda=0.3,0.4,1.0
#   - mmr_lambda_sweep: lambda=0.6,0.7
#
# Usage:
#   bash scripts/pipeline/rerun_val_set_gap_check.sh
#   DRY_RUN=true bash scripts/pipeline/rerun_val_set_gap_check.sh
#   B3_TOP_KS="3,6" MMR_1024_LAMBDAS="0.4,1.0" bash scripts/pipeline/rerun_val_set_gap_check.sh
#   RUN_B3=false RUN_MMR_1024=true RUN_MMR_2048=false bash scripts/pipeline/rerun_val_set_gap_check.sh
#   SKIP_MISSING=false bash scripts/pipeline/rerun_val_set_gap_check.sh

set -euo pipefail
shopt -s nullglob

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

SPLIT="${SPLIT:-val}"
CHECKPOINT="${CHECKPOINT:-best}"
FORCE_INFER="${FORCE_INFER:-true}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_MISSING="${SKIP_MISSING:-true}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

RUN_B3="${RUN_B3:-false}"
RUN_MMR_1024="${RUN_MMR_1024:-true}"
RUN_MMR_2048="${RUN_MMR_2048:-true}"

B3_ROOT="${B3_ROOT:-outputs/runs/b3_mmr_topk_sweep_1024}"
B3_EXPERIMENT="${B3_EXPERIMENT:-b3_mmr_topk_sweep_1024}"
B3_TOP_KS="${B3_TOP_KS:-3,5,6}"
B3_MMR_LAMBDA="${B3_MMR_LAMBDA:-0.7}"

MMR_1024_ROOT="${MMR_1024_ROOT:-outputs/runs/mmr_lambda_sweep_1024}"
MMR_1024_EXPERIMENT="${MMR_1024_EXPERIMENT:-mmr_lambda_sweep_1024}"
MMR_1024_LAMBDAS="${MMR_1024_LAMBDAS:-0.3,0.4,1.0}"

MMR_2048_ROOT="${MMR_2048_ROOT:-outputs/runs/mmr_lambda_sweep}"
MMR_2048_EXPERIMENT="${MMR_2048_EXPERIMENT:-mmr_lambda_sweep}"
MMR_2048_LAMBDAS="${MMR_2048_LAMBDAS:-0.6,0.7}"

MERGE_LORA_CACHE="${MERGE_LORA_CACHE:-true}"
MERGE_LORA_CACHE_DIR="${MERGE_LORA_CACHE_DIR:-outputs/cache/merged_lora}"
MERGE_LORA_CACHE_FORCE_REBUILD="${MERGE_LORA_CACHE_FORCE_REBUILD:-false}"
STOP_AFTER_INFER="${STOP_AFTER_INFER:-true}"

WRITE_SUMMARY="${WRITE_SUMMARY:-true}"
SUMMARY_DIR="${SUMMARY_DIR:-outputs/runs/val_gap_check_summary}"

echo "[rerun_val_set_gap_check] split=${SPLIT}"
echo "[rerun_val_set_gap_check] checkpoint=${CHECKPOINT}"
echo "[rerun_val_set_gap_check] force_infer=${FORCE_INFER}"
echo "[rerun_val_set_gap_check] dry_run=${DRY_RUN}"
echo "[rerun_val_set_gap_check] skip_missing=${SKIP_MISSING}"
echo "[rerun_val_set_gap_check] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

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

find_run_dir() {
  local root="$1"
  shift
  local pattern=""
  local candidate=""
  for pattern in "$@"; do
    for candidate in "${root}"/${pattern}; do
      if [[ -d "${candidate}/train/${CHECKPOINT}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    done
  done
  return 1
}

handle_missing() {
  local message="$1"
  if [[ "${SKIP_MISSING}" == "true" ]]; then
    echo "[rerun_val_set_gap_check] skip: ${message}" >&2
    return 0
  fi
  echo "[rerun_val_set_gap_check] ${message}" >&2
  return 1
}

prepare_train_config() {
  local run_dir="$1"
  mkdir -p "${run_dir}/configs"
  if [[ -f "${run_dir}/configs/train.resolved.yaml" ]]; then
    return 0
  fi
  if [[ -f "${run_dir}/train/config.resolved.yaml" ]]; then
    cp "${run_dir}/train/config.resolved.yaml" "${run_dir}/configs/train.resolved.yaml"
    return 0
  fi
  echo "[rerun_val_set_gap_check] missing train config for ${run_dir}" >&2
  echo "[rerun_val_set_gap_check] expected ${run_dir}/configs/train.resolved.yaml or ${run_dir}/train/config.resolved.yaml" >&2
  return 1
}

run_infer() {
  local experiment="$1"
  local run_dir="$2"
  shift 2

  if [[ "${DRY_RUN}" == "true" ]]; then
    if [[ ! -f "${run_dir}/configs/train.resolved.yaml" && ! -f "${run_dir}/train/config.resolved.yaml" ]]; then
      echo "[rerun_val_set_gap_check] missing train config for ${run_dir}" >&2
      return 1
    fi
    echo "[rerun_val_set_gap_check] dry-run infer experiment=${experiment} run_dir=${run_dir}"
    return 0
  fi

  prepare_train_config "${run_dir}"
  echo "[rerun_val_set_gap_check] infer experiment=${experiment} run_dir=${run_dir}"

  python -m fact_checking.pipeline.run \
    "experiment=${experiment}" \
    pipeline.mode=infer \
    "pipeline.force.infer=${FORCE_INFER}" \
    "$(hydra_string_override pipeline.run_dir "${run_dir}")" \
    "$(hydra_string_override infer.split "${SPLIT}")" \
    "$(hydra_string_override infer.checkpoint "${CHECKPOINT}")" \
    "infer.merge_lora_cache.enabled=${MERGE_LORA_CACHE}" \
    "$(hydra_string_override infer.merge_lora_cache.dir "${MERGE_LORA_CACHE_DIR}")" \
    "infer.merge_lora_cache.force_rebuild=${MERGE_LORA_CACHE_FORCE_REBUILD}" \
    "infer.server.stop_after_infer=${STOP_AFTER_INFER}" \
    "$@"
}

run_b3_topk_val() {
  local values=()
  local top_k=""
  local run_dir=""
  split_csv "${B3_TOP_KS}" values
  if [[ "${#values[@]}" -eq 0 ]]; then
    echo "[rerun_val_set_gap_check] B3_TOP_KS is empty, skip b3"
    return 0
  fi

  echo "[rerun_val_set_gap_check] b3 top_k values=${values[*]} mmr_lambda=${B3_MMR_LAMBDA}"
  for top_k in "${values[@]}"; do
    run_dir="$(find_run_dir \
      "${B3_ROOT}" \
      "build.retrieval.mmr_lambda-${B3_MMR_LAMBDA},build.retrieval.top_k-${top_k}__*" \
      "build.retrieval.top_k-${top_k},build.retrieval.mmr_lambda-${B3_MMR_LAMBDA}__*" \
    )" || {
      handle_missing "cannot find b3 run with train/${CHECKPOINT} for top_k=${top_k} under ${B3_ROOT}" || return 1
      continue
    }
    echo "=== b3_mmr_topk_sweep_1024 top_k=${top_k} ==="
    run_infer \
      "${B3_EXPERIMENT}" \
      "${run_dir}" \
      "build.retrieval.mmr_lambda=${B3_MMR_LAMBDA}" \
      "build.retrieval.top_k=${top_k}"
  done
}

run_mmr_lambda_val() {
  local root="$1"
  local experiment="$2"
  local lambdas_csv="$3"
  local label="$4"
  local values=()
  local lam=""
  local run_dir=""
  split_csv "${lambdas_csv}" values
  if [[ "${#values[@]}" -eq 0 ]]; then
    echo "[rerun_val_set_gap_check] ${label} lambda list is empty, skip"
    return 0
  fi

  echo "[rerun_val_set_gap_check] ${label} lambdas=${values[*]}"
  for lam in "${values[@]}"; do
    run_dir="$(find_run_dir "${root}" "build.retrieval.mmr_lambda-${lam}__*")" || {
      handle_missing "cannot find ${label} run with train/${CHECKPOINT} for lambda=${lam} under ${root}" || return 1
      continue
    }
    echo "=== ${label} lambda=${lam} ==="
    run_infer \
      "${experiment}" \
      "${run_dir}" \
      "build.retrieval.mmr_lambda=${lam}"
  done
}

if [[ "${RUN_B3}" == "true" ]]; then
  run_b3_topk_val
fi

if [[ "${RUN_MMR_1024}" == "true" ]]; then
  run_mmr_lambda_val "${MMR_1024_ROOT}" "${MMR_1024_EXPERIMENT}" "${MMR_1024_LAMBDAS}" "mmr_lambda_sweep_1024"
fi

if [[ "${RUN_MMR_2048}" == "true" ]]; then
  run_mmr_lambda_val "${MMR_2048_ROOT}" "${MMR_2048_EXPERIMENT}" "${MMR_2048_LAMBDAS}" "mmr_lambda_sweep"
fi

if [[ "${WRITE_SUMMARY}" == "true" && "${DRY_RUN}" != "true" ]]; then
  echo "[rerun_val_set_gap_check] writing summary to ${SUMMARY_DIR}"
  python scripts/pipeline/summarize_infer_metrics.py \
    "${B3_ROOT}" \
    "${MMR_1024_ROOT}" \
    "${MMR_2048_ROOT}" \
    --output-dir "${SUMMARY_DIR}" || {
      echo "[rerun_val_set_gap_check] summary failed; inference outputs are still available under each run_dir" >&2
    }
fi
