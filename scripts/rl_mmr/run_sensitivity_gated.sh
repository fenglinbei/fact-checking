#!/usr/bin/env bash
# ==============================================================================
# Sensitivity-gated MMR full pipeline: build → train → infer
#
# Per-claim λ chosen by sensitivity + pool-redundancy + relevance-floor gate.
# Reads the best (θ_s, θ_r, λ_low, gating_mode, ε) from Stage A's dev_grid.csv
# and passes them as Hydra overrides.
#
# Usage:
#   bash scripts/rl_mmr/run_sensitivity_gated.sh \
#       [STAGE_A_DIR]                        # default: outputs/rl_mmr/sensitivity_search
#       [EXPERIMENT]                         # default: mmr_sensitivity_gated
#       [MODE]                               # default: full
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

STAGE_A_DIR="${1:-outputs/rl_mmr/sensitivity_search}"
EXPERIMENT="${2:-mmr_sensitivity_gated}"
MODE="${3:-full}"

STAGE_A_CSV="${STAGE_A_DIR}/dev_grid.csv"

if [ ! -f "${STAGE_A_CSV}" ]; then
    echo "[stage-B][fatal] Stage A CSV not found: ${STAGE_A_CSV}" >&2
    echo "[stage-B][fatal] Run scripts/rl_mmr/run_stage_a_search.sh first." >&2
    exit 1
fi

echo "============================================"
echo "  Sensitivity-Gated MMR Pipeline (Stage B)"
echo "  stage A    : ${STAGE_A_DIR}"
echo "  experiment : ${EXPERIMENT}"
echo "  mode       : ${MODE}"
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# Parse best row from Stage A dev_grid.csv
# ---------------------------------------------------------------------------
BEST_PARAMS=$(python - <<PY
import csv
from pathlib import Path

csv_path = Path("${STAGE_A_CSV}")
with csv_path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    raise SystemExit("[stage-B][fatal] dev_grid.csv is empty")

best = rows[0]  # already sorted by score desc

# Map CSV columns → Hydra overrides. All values are wrapped in single quotes
# to protect against Hydra parsing commas etc.
overrides = []
overrides.append(f"build.retrieval.learned_lambda.sensitivity.theta_s={best['theta_s']}")
overrides.append(f"build.retrieval.learned_lambda.sensitivity.theta_r={best['theta_r']}")
overrides.append(f"build.retrieval.learned_lambda.sensitivity.lambda_low={best['lambda_low']}")
overrides.append(f"build.retrieval.learned_lambda.sensitivity.gating_mode='{best['gating_mode']}'")

eps = best.get("epsilon", "").strip()
if eps and eps.lower() != "none":
    overrides.append(f"build.retrieval.learned_lambda.sensitivity.relevance_floor.epsilon={eps}")

print(" ".join(overrides))
print(f"[stage-B] best row: score={best['score']} acc={best['accuracy']} f1={best['macro_f1']} "
      f"theta_s={best['theta_s']} theta_r={best['theta_r']} lambda_low={best['lambda_low']} "
      f"gating={best['gating_mode']} eps={best.get('epsilon','N/A')}",
      file=__import__('sys').stderr)
PY
)

echo "[stage-B] best params: ${BEST_PARAMS}"
echo ""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
if [ "${MODE}" = "full" ] || [ "${MODE}" = "build" ]; then
    echo "[$(date '+%H:%M:%S')] === Build phase ==="
    # shellcheck disable=SC2086
    python -m fact_checking.pipeline.run \
        "experiment=${EXPERIMENT}" \
        pipeline.mode=build \
        ${BEST_PARAMS}
    echo "[$(date '+%H:%M:%S')] Build done."
    echo ""
fi

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
if [ "${MODE}" = "full" ] || [ "${MODE}" = "train" ]; then
    echo "[$(date '+%H:%M:%S')] === Train phase ==="
    # shellcheck disable=SC2086
    python -m fact_checking.pipeline.run \
        "experiment=${EXPERIMENT}" \
        pipeline.mode=train \
        ${BEST_PARAMS}
    echo "[$(date '+%H:%M:%S')] Train done."
    echo ""
fi

# ---------------------------------------------------------------------------
# Infer
# ---------------------------------------------------------------------------
if [ "${MODE}" = "full" ] || [ "${MODE}" = "infer" ]; then
    echo "[$(date '+%H:%M:%S')] === Infer phase ==="
    # shellcheck disable=SC2086
    python -m fact_checking.pipeline.run \
        "experiment=${EXPERIMENT}" \
        pipeline.mode=infer \
        ${BEST_PARAMS}
    echo "[$(date '+%H:%M:%S')] Infer done."
    echo ""
fi

echo "[$(date '+%H:%M:%S')] Pipeline completed."
