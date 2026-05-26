#!/usr/bin/env bash
# b4 3分类基线实验启动脚本.
# 将 LIAR 6 类合并为 3 类 (false/mixed/true) 训练判别式分类器.
#
# 用法:
#   bash scripts/phase1_pipeline/run_b4_3class.sh
#   PIPELINE_MODE=build bash scripts/phase1_pipeline/run_b4_3class.sh   # 仅 build 阶段
#   PIPELINE_MODE=train bash scripts/phase1_pipeline/run_b4_3class.sh   # 仅训练
#
# 必须先 conda activate cppo. 默认覆盖 CUDA_VISIBLE_DEVICES=0 (单卡 RTX 2060).

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

PIPELINE_MODE="${PIPELINE_MODE:-full}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

echo "[run_b4_3class] pipeline.mode=${PIPELINE_MODE}"
echo "[run_b4_3class] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python -m fact_checking.pipeline.run \
    experiment=b4_3class \
    "$@"
