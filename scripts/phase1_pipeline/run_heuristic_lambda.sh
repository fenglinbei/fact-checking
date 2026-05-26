#!/usr/bin/env bash
# ==============================================================================
# Heuristic-λ MMR full pipeline: build → train → infer
#
# 使用候选数量启发式公式为每条 claim 预测最优 λ：
#   λ = max(0, min(1, -0.0732 * ln(n_candidates) + 0.6127))
#
# 配置继承 configs/experiment/b3_mmr_topk_sweep_1024.yaml
# top_k=5, heuristic lambda enabled
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

EXPERIMENT="${1:-heuristic_lambda_mmr}"
MODE="${2:-full}"

echo "============================================"
echo "  Heuristic-λ MMR Pipeline"
echo "  experiment : ${EXPERIMENT}"
echo "  mode       : ${MODE}"
echo "  root       : ${ROOT}"
echo "============================================"
echo ""

# ---- Phase 1: Build ----
if [ "${MODE}" = "full" ] || [ "${MODE}" = "build" ]; then
    echo "[$(date '+%H:%M:%S')] === Build phase ==="
    python -m fact_checking.pipeline.run \
        "experiment=${EXPERIMENT}" \
        pipeline.mode=build

    echo "[$(date '+%H:%M:%S')] Build done."
    echo ""
fi

# ---- Phase 2: Train ----
if [ "${MODE}" = "full" ] || [ "${MODE}" = "train" ]; then
    echo "[$(date '+%H:%M:%S')] === Train phase ==="
    python -m fact_checking.pipeline.run \
        "experiment=${EXPERIMENT}" \
        pipeline.mode=train

    echo "[$(date '+%H:%M:%S')] Train done."
    echo ""
fi

# ---- Phase 3: Infer ----
if [ "${MODE}" = "full" ] || [ "${MODE}" = "infer" ]; then
    echo "[$(date '+%H:%M:%S')] === Infer phase ==="
    python -m fact_checking.pipeline.run \
        "experiment=${EXPERIMENT}" \
        pipeline.mode=infer

    echo "[$(date '+%H:%M:%S')] Infer done."
    echo ""
fi

echo "[$(date '+%H:%M:%S')] Pipeline completed."
