#!/usr/bin/env bash
# b4 证据消融实验启动脚本.
# 仅输入 claim，不拼接检索证据，用于判断检索证据是否提供了有效判别信号.
#
# 用法:
#   bash scripts/pipeline/run_b4_no_evidence.sh
#   PIPELINE_MODE=train bash scripts/pipeline/run_b4_no_evidence.sh  # 仅训练
#
# 必须先 conda activate cppo. 默认覆盖 CUDA_VISIBLE_DEVICES=0 (单卡 RTX 2060).

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

PIPELINE_MODE="${PIPELINE_MODE:-full}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

echo "[run_b4_no_evidence] pipeline.mode=${PIPELINE_MODE}"
echo "[run_b4_no_evidence] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python -m fact_checking.pipeline.run \
    experiment=b4_no_evidence \
    "pipeline.mode=${PIPELINE_MODE}" \
    "$@"
