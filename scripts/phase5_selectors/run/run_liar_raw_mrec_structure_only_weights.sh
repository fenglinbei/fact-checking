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

SOURCE_FEATURE_ROOT="${SOURCE_FEATURE_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1/04_evidence_map}"
TRAIN_INPUT="${TRAIN_INPUT:-${SOURCE_FEATURE_ROOT}/candidate_evidence_map_features_train.jsonl}"
VAL_INPUT="${VAL_INPUT:-${SOURCE_FEATURE_ROOT}/candidate_evidence_map_features_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_structure_only/weights}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-5}"
EPOCHS="${EPOCHS:-30}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
MAP_ABLATION_MODE="${MAP_ABLATION_MODE:-full}"
TRAIN_SAMPLE_LIMIT="${TRAIN_SAMPLE_LIMIT:-0}"
VAL_SAMPLE_LIMIT="${VAL_SAMPLE_LIMIT:-0}"
DRY_RUN="${DRY_RUN:-false}"

export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "$DRY_RUN" != "true" ]]; then
  [[ -f "$TRAIN_INPUT" ]] || { printf 'Missing train input: %s\n' "$TRAIN_INPUT" >&2; exit 2; }
  [[ -f "$VAL_INPUT" ]] || { printf 'Missing val input: %s\n' "$VAL_INPUT" >&2; exit 2; }
  if [[ -d "$OUTPUT_DIR" ]]; then
    shopt -s nullglob dotglob
    existing_output=("$OUTPUT_DIR"/*)
    if (( ${#existing_output[@]} > 0 )); then
      printf 'Refusing to overwrite non-empty output directory: %s\n' "$OUTPUT_DIR" >&2
      exit 2
    fi
  fi
fi

cmd=(
  "$PYTHON_BIN" scripts/phase5_selectors/train/train_mrec_learned_marginal_structure_only.py
  --train-input "$TRAIN_INPUT"
  --val-input "$VAL_INPUT"
  --output-dir "$OUTPUT_DIR"
  --candidate-top-n "$CANDIDATE_TOP_N"
  --rollout-steps "$ROLLOUT_STEPS"
  --epochs "$EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --map-ablation-mode "$MAP_ABLATION_MODE"
  --train-sample-limit "$TRAIN_SAMPLE_LIMIT"
  --val-sample-limit "$VAL_SAMPLE_LIMIT"
)

printf '[structure-only] train input : %s\n' "$TRAIN_INPUT"
printf '[structure-only] val input   : %s\n' "$VAL_INPUT"
printf '[structure-only] output      : %s\n' "$OUTPUT_DIR"
printf '[structure-only] top-n/steps : %s/%s\n' "$CANDIDATE_TOP_N" "$ROLLOUT_STEPS"
printf '[structure-only] epochs/lr   : %s/%s\n' "$EPOCHS" "$LEARNING_RATE"
printf '[structure-only] device      : CPU (CUDA hidden)\n'

if [[ "$DRY_RUN" == "true" ]]; then
  printf '+'
  printf ' %q' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
