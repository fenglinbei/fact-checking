#!/usr/bin/env bash
# ==============================================================================
# Sensitivity-gated MMR full pipeline: build → train → infer
#
# Per-claim λ chosen by sensitivity + pool-redundancy + relevance-floor gate.
# Config: configs/experiment/mmr_sensitivity_gated.yaml
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

EXPERIMENT="${1:-mmr_sensitivity_gated}"
MODE="${2:-full}"

echo "============================================"
echo "  Sensitivity-Gated MMR Pipeline"
echo "  experiment : ${EXPERIMENT}"
echo "  mode       : ${MODE}"
echo "  root       : ${ROOT}"
echo "============================================"
echo ""

if [ "${MODE}" = "full" ] || [ "${MODE}" = "build" ]; then
    echo "[$(date '+%H:%M:%S')] === Build phase ==="
    python -m fact_checking.pipeline.run \
        "experiment=${EXPERIMENT}" \
        pipeline.mode=build
    echo "[$(date '+%H:%M:%S')] Build done."
    echo ""
fi

if [ "${MODE}" = "full" ] || [ "${MODE}" = "train" ]; then
    echo "[$(date '+%H:%M:%S')] === Train phase ==="
    python -m fact_checking.pipeline.run \
        "experiment=${EXPERIMENT}" \
        pipeline.mode=train
    echo "[$(date '+%H:%M:%S')] Train done."
    echo ""
fi

if [ "${MODE}" = "full" ] || [ "${MODE}" = "infer" ]; then
    echo "[$(date '+%H:%M:%S')] === Infer phase ==="
    python -m fact_checking.pipeline.run \
        "experiment=${EXPERIMENT}" \
        pipeline.mode=infer
    echo "[$(date '+%H:%M:%S')] Infer done."
    echo ""
fi

echo "[$(date '+%H:%M:%S')] Pipeline completed."
