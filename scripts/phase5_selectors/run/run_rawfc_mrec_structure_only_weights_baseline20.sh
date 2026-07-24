#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python" ]]; then
    PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

SOURCE_FEATURE_ROOT="${SOURCE_FEATURE_ROOT:-outputs/selectors/atom_anchor/rawfc_abc_v0_1_baseline20/04_evidence_map}"
TRAIN_INPUT="${TRAIN_INPUT:-${SOURCE_FEATURE_ROOT}/candidate_evidence_map_features_train.jsonl}"
VAL_INPUT="${VAL_INPUT:-${SOURCE_FEATURE_ROOT}/candidate_evidence_map_features_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/atom_anchor/rawfc_abc_v0_1_baseline20/05_mrec_v0_2_learned_marginal_structure_only/weights}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-5}"
EPOCHS="${EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
DRY_RUN="${DRY_RUN:-false}"
ENSURE_WEIGHTS="${ENSURE_WEIGHTS:-true}"

export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

weights_complete() {
  [[ -f "${OUTPUT_DIR}/weights.json" && -f "${OUTPUT_DIR}/manifest.json" ]] || return 1
  "$PYTHON_BIN" - "$OUTPUT_DIR" "$TRAIN_INPUT" "$VAL_INPUT" "$CANDIDATE_TOP_N" "$ROLLOUT_STEPS" "$EPOCHS" "$LEARNING_RATE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
train_input = Path(sys.argv[2])
val_input = Path(sys.argv[3])
candidate_top_n = int(sys.argv[4])
rollout_steps = int(sys.argv[5])
epochs = int(sys.argv[6])
learning_rate = float(sys.argv[7])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
contract = manifest.get("supervision_contract") or {}
params = manifest.get("params") or {}
expected_paths = {
    "train_input": train_input,
    "val_input": val_input,
}
for key, expected in expected_paths.items():
    recorded = Path(str(manifest.get(key) or ""))
    if recorded != expected and recorded.resolve() != expected.resolve():
        raise SystemExit(1)
if manifest.get("training_supervision") != "structure_only":
    raise SystemExit(1)
if manifest.get("selector_name") != "mrec_greedy_transition_v0_2_learned_marginal_structure_only":
    raise SystemExit(1)
if manifest.get("compute_device") != "cpu":
    raise SystemExit(1)
if int(params.get("candidate_top_n") or 0) != candidate_top_n:
    raise SystemExit(1)
if int(params.get("rollout_steps") or 0) != rollout_steps:
    raise SystemExit(1)
if int(params.get("epochs") or 0) != epochs:
    raise SystemExit(1)
if abs(float(params.get("learning_rate") or 0.0) - learning_rate) > 1e-12:
    raise SystemExit(1)
if str(params.get("map_ablation_mode") or "") != "full":
    raise SystemExit(1)
for key, path in (("n_train_rows", train_input), ("n_val_rows", val_input)):
    with path.open(encoding="utf-8") as handle:
        row_count = sum(1 for line in handle if line.strip())
    if int(manifest.get(key) or 0) != row_count:
        raise SystemExit(1)
for key in (
    "oracle_read_row_count",
    "gold_label_read_count",
    "teacher_read_count",
    "utility_read_count",
    "reward_read_count",
):
    if int(contract.get(key, -1)) != 0:
        raise SystemExit(1)
if not str(manifest.get("weight_fingerprint") or ""):
    raise SystemExit(1)
print(f"RAWFC baseline20 structure-only weights complete: {root / 'weights.json'}")
PY
}

cmd=(
  "$PYTHON_BIN" scripts/phase5_selectors/train/train_mrec_learned_marginal_structure_only.py
  --train-input "$TRAIN_INPUT"
  --val-input "$VAL_INPUT"
  --output-dir "$OUTPUT_DIR"
  --candidate-top-n "$CANDIDATE_TOP_N"
  --rollout-steps "$ROLLOUT_STEPS"
  --epochs "$EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --map-ablation-mode full
)

printf '[rawfc-structure-only-weights] train=%s val=%s output=%s top_n=%s steps=%s epochs=%s lr=%s device=CPU\n' \
  "$TRAIN_INPUT" "$VAL_INPUT" "$OUTPUT_DIR" "$CANDIDATE_TOP_N" "$ROLLOUT_STEPS" "$EPOCHS" "$LEARNING_RATE"

if [[ "$DRY_RUN" == "true" ]]; then
  printf '+'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -f "$TRAIN_INPUT" ]] || { printf 'Missing train input: %s\n' "$TRAIN_INPUT" >&2; exit 2; }
[[ -f "$VAL_INPUT" ]] || { printf 'Missing val input: %s\n' "$VAL_INPUT" >&2; exit 2; }
if weights_complete; then
  exit 0
fi
if [[ "$ENSURE_WEIGHTS" != "true" ]]; then
  printf 'Missing or incompatible RAWFC baseline20 structure-only weights: %s\n' "$OUTPUT_DIR" >&2
  exit 2
fi
"${cmd[@]}"
weights_complete
