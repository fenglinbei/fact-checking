#!/usr/bin/env bash
# 基于 b4 的 MMR lambda 扫描启动脚本.
# 默认扫 11 个值 (步长 0.1); 通过环境变量可调.
#
# 用法:
#   bash scripts/pipeline/run_b4_mmr_sweep.sh
#   MMR_LAMBDAS="0.0,0.5,1.0" bash scripts/pipeline/run_b4_mmr_sweep.sh
#   PIPELINE_MODE=build bash scripts/pipeline/run_b4_mmr_sweep.sh   # 仅 build 阶段
#
# 必须先 conda activate cppo. 默认覆盖 CUDA_VISIBLE_DEVICES=0 (单卡 RTX 2060).

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

MMR_LAMBDAS="${MMR_LAMBDAS:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
PIPELINE_MODE="${PIPELINE_MODE:-full}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

echo "[run_b4_mmr_sweep] mmr_lambda=${MMR_LAMBDAS}"
echo "[run_b4_mmr_sweep] pipeline.mode=${PIPELINE_MODE}"
echo "[run_b4_mmr_sweep] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python -m fact_checking.pipeline.run -m \
    experiment=b4_mmr_lambda_sweep \
    "build.retrieval.mmr_lambda=${MMR_LAMBDAS}" \
    "$@"
