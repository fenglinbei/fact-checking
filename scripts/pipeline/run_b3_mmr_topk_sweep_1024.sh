#!/usr/bin/env bash
# b3 semantic chunking top_k sweep, max_length=1024.
# Runs the full build -> train -> infer pipeline by default.
#
# Usage:
#   bash scripts/pipeline/run_b3_mmr_topk_sweep_1024.sh
#   TOP_KS="14,16,18" bash scripts/pipeline/run_b3_mmr_topk_sweep_1024.sh
#   PIPELINE_MODE=build bash scripts/pipeline/run_b3_mmr_topk_sweep_1024.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

TOP_KS="${TOP_KS:-10,12,14,16,18,20,22}"
MMR_LAMBDA="${MMR_LAMBDA:-0.7}"
PIPELINE_MODE="${PIPELINE_MODE:-full}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

echo "[run_b3_mmr_topk_sweep_1024] top_k=${TOP_KS}"
echo "[run_b3_mmr_topk_sweep_1024] mmr_lambda=${MMR_LAMBDA}"
echo "[run_b3_mmr_topk_sweep_1024] pipeline.mode=${PIPELINE_MODE}"
echo "[run_b3_mmr_topk_sweep_1024] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python -m fact_checking.pipeline.run -m \
  experiment=b3_mmr_topk_sweep_1024 \
  "pipeline.mode=${PIPELINE_MODE}" \
  "build.retrieval.mmr_lambda=${MMR_LAMBDA}" \
  "build.retrieval.top_k=${TOP_KS}" \
  "$@"
