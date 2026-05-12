#!/usr/bin/env bash
# build.retrieval.top_k 参数扫描：固定 mmr_lambda=0.7，复用已有 train/best checkpoint 做 test inference。
#
# 用法:
#   bash scripts/pipeline/run_mmr_topk_sweep_infer.sh
#   TOP_KS="0,2,4" bash scripts/pipeline/run_mmr_topk_sweep_infer.sh
#   BASE_RUN_DIR=outputs/runs/mmr_lambda_sweep/build.retrieval.mmr_lambda-0.7__5385ea65 \
#     bash scripts/pipeline/run_mmr_topk_sweep_infer.sh
#
# 默认先为每个 top_k build 新候选，再用 BASE_RUN_DIR/train/best 推理。
# 传给本脚本的额外 Hydra overrides 会同时用于 build 和 infer 阶段。

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

TOP_KS="${TOP_KS:-0,2,4,6,8,10,12,14,16,18,20,22,24}"
MMR_LAMBDA="${MMR_LAMBDA:-0.7}"
BASE_RUN_DIR="${BASE_RUN_DIR:-outputs/runs/mmr_lambda_sweep/build.retrieval.mmr_lambda-0.7__5385ea65}"
EXPERIMENT="${EXPERIMENT:-mmr_topk_sweep_infer}"
SPLIT="${SPLIT:-test}"
CHECKPOINT="${CHECKPOINT:-best}"
SUMMARY_CSV="${SUMMARY_CSV:-outputs/runs/${EXPERIMENT}/summary.csv}"
PIPELINE_MODE="${PIPELINE_MODE:-full}"  # full | build | infer
FORCE_BUILD="${FORCE_BUILD:-false}"
FORCE_INFER="${FORCE_INFER:-true}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

if [[ ! -d "${BASE_RUN_DIR}" ]]; then
  echo "[run_mmr_topk_sweep_infer] BASE_RUN_DIR not found: ${BASE_RUN_DIR}" >&2
  exit 1
fi

if [[ -d "${BASE_RUN_DIR}/train" ]]; then
  BASE_TRAIN_DIR="${BASE_RUN_DIR}/train"
else
  BASE_TRAIN_DIR="${BASE_RUN_DIR}"
fi

if [[ ! -d "${BASE_TRAIN_DIR}/${CHECKPOINT}" ]]; then
  echo "[run_mmr_topk_sweep_infer] checkpoint not found: ${BASE_TRAIN_DIR}/${CHECKPOINT}" >&2
  exit 1
fi

case "${PIPELINE_MODE}" in
  full|build|infer) ;;
  *)
    echo "[run_mmr_topk_sweep_infer] unsupported PIPELINE_MODE=${PIPELINE_MODE}; use full, build, or infer" >&2
    exit 1
    ;;
esac

echo "[run_mmr_topk_sweep_infer] top_k=${TOP_KS}"
echo "[run_mmr_topk_sweep_infer] mmr_lambda=${MMR_LAMBDA}"
echo "[run_mmr_topk_sweep_infer] base_run_dir=${BASE_RUN_DIR}"
echo "[run_mmr_topk_sweep_infer] base_train_dir=${BASE_TRAIN_DIR}"
echo "[run_mmr_topk_sweep_infer] checkpoint=${CHECKPOINT}"
echo "[run_mmr_topk_sweep_infer] split=${SPLIT}"
echo "[run_mmr_topk_sweep_infer] summary_csv=${SUMMARY_CSV}"
echo "[run_mmr_topk_sweep_infer] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

IFS=',' read -r -a TOP_K_ARRAY <<< "${TOP_KS}"

find_topk_run_dir() {
  local top_k="$1"
  local pattern="build.retrieval.mmr_lambda-${MMR_LAMBDA},build.retrieval.top_k-${top_k}__*"
  local reverse_pattern="build.retrieval.top_k-${top_k},build.retrieval.mmr_lambda-${MMR_LAMBDA}__*"
  local run_dir=""
  for candidate in "outputs/runs/${EXPERIMENT}"/${pattern} "outputs/runs/${EXPERIMENT}"/${reverse_pattern}; do
    if [[ -f "${candidate}/manifest.json" ]]; then
      run_dir="${candidate}"
      break
    fi
  done
  if [[ -z "${run_dir}" ]]; then
    echo "[run_mmr_topk_sweep_infer] cannot find run dir for top_k=${top_k}" >&2
    exit 1
  fi
  printf '%s\n' "${run_dir}"
}

for raw_top_k in "${TOP_K_ARRAY[@]}"; do
  top_k="${raw_top_k//[[:space:]]/}"
  if [[ -z "${top_k}" ]]; then
    continue
  fi

  echo "=== top_k=${top_k}: build ==="
  TOPK_RUN_DIR=""
  if [[ "${PIPELINE_MODE}" == "full" || "${PIPELINE_MODE}" == "build" ]]; then
    BUILD_LOG="$(mktemp "${TMPDIR:-/tmp}/mmr_topk_build.${top_k}.XXXXXX.log")"
    python -m fact_checking.pipeline.run \
      "experiment=${EXPERIMENT}" \
      pipeline.mode=build \
      "pipeline.force.build=${FORCE_BUILD}" \
      "build.retrieval.mmr_lambda=${MMR_LAMBDA}" \
      "build.retrieval.top_k=${top_k}" \
      "$@" | tee "${BUILD_LOG}"
    TOPK_RUN_DIR="$(awk '/Pipeline completed:/ {print $3}' "${BUILD_LOG}" | tail -n 1)"
  fi

  if [[ -z "${TOPK_RUN_DIR}" ]]; then
    TOPK_RUN_DIR="$(find_topk_run_dir "${top_k}")"
  fi
  echo "[run_mmr_topk_sweep_infer] top_k=${top_k} run_dir=${TOPK_RUN_DIR}"

  if [[ "${PIPELINE_MODE}" == "build" ]]; then
    continue
  fi

  echo "=== top_k=${top_k}: prepare reused inference config ==="
  REUSE_CONFIG="${TOPK_RUN_DIR}/configs/train.reuse_${CHECKPOINT}.top_k_${top_k}.yaml"
  python scripts/pipeline/mmr_topk_reuse.py prepare-config \
    --base-run-dir "${BASE_RUN_DIR}" \
    --topk-run-dir "${TOPK_RUN_DIR}" \
    --output-config "${REUSE_CONFIG}"

  echo "=== top_k=${top_k}: infer with reused checkpoint ==="
  python -m fact_checking.pipeline.run \
    "experiment=${EXPERIMENT}" \
    pipeline.mode=infer \
    "pipeline.run_dir=${TOPK_RUN_DIR}" \
    "pipeline.force.infer=${FORCE_INFER}" \
    "train.run_dir=${BASE_TRAIN_DIR}" \
    "infer.config_path=${REUSE_CONFIG}" \
    "infer.split=${SPLIT}" \
    "infer.checkpoint=${CHECKPOINT}" \
    "build.retrieval.mmr_lambda=${MMR_LAMBDA}" \
    "build.retrieval.top_k=${top_k}" \
    "$@"

  python scripts/pipeline/mmr_topk_reuse.py summarize \
    --top-k "${top_k}" \
    --topk-run-dir "${TOPK_RUN_DIR}" \
    --summary-csv "${SUMMARY_CSV}"
done

if [[ "${PIPELINE_MODE}" != "build" ]]; then
  echo "[run_mmr_topk_sweep_infer] summary: ${SUMMARY_CSV}"
fi
