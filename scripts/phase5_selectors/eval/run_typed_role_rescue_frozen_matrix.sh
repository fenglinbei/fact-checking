#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

TRACE_ROOT="${TRACE_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1/11_typed_role_rescue_v0_1/prompt_feasible}"
BUILD_ROOT="${BUILD_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__typed_role_rescue_v0_1}"
MATRIX_MANIFEST="${MATRIX_MANIFEST:-${TRACE_ROOT}/val/manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/validation_artifacts/typed_role_rescue_v0_1/val/minmax_best_ef363d90_noadjust}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-4}"
FORCE_MATRIX="${FORCE_MATRIX:-false}"
PYTHON_BIN="${MATRIX_PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"

for required in "$MATRIX_MANIFEST" "$BUILD_ROOT/native_gate_anchor/build/build_val.jsonl"; do
  [[ -f "$required" ]] || { printf 'Missing frozen-matrix input: %s\n' "$required" >&2; exit 2; }
done

force_prepare=false
force_infer=false
force_fanout=false
if [[ "$FORCE_MATRIX" == "true" ]]; then
  force_prepare=true
  force_infer=true
  force_fanout=true
fi

REFERENCE_CONTRACT=configs/validation/baces_native_label_token_reference_v0_1.json \
MATRIX_MANIFEST="$MATRIX_MANIFEST" \
BUILD_ROOT="$BUILD_ROOT" \
OUTPUT_DIR="$OUTPUT_DIR" \
GATE_CELL=native_gate_anchor \
PHASES=prepare,infer,fanout \
EVAL_NPROC_PER_NODE="$EVAL_NPROC_PER_NODE" \
PER_DEVICE_EVAL_BATCH_SIZE=1 \
DATALOADER_NUM_WORKERS=4 \
FORCE_PREPARE="$force_prepare" \
FORCE_INFER="$force_infer" \
FORCE_FANOUT="$force_fanout" \
bash scripts/phase5_selectors/eval/run_baces_deduplicated_raw_logits_matrix.sh

PRED_ROOT="${OUTPUT_DIR}/materialized/cells"
for cell in r_only cor opp ctx retr random full learned_fixed5 native_gate_anchor; do
  path="${PRED_ROOT}/${cell}/label_token/val_predictions.jsonl"
  [[ -f "$path" ]] || { printf 'Missing frozen prediction cell %s: %s\n' "$cell" "$path" >&2; exit 3; }
done

"$PYTHON_BIN" scripts/sentence_trace_method/paired_significance.py \
  --comparison cor_vs_random "$PRED_ROOT/random/label_token/val_predictions.jsonl" "$PRED_ROOT/cor/label_token/val_predictions.jsonl" \
  --comparison opp_vs_random "$PRED_ROOT/random/label_token/val_predictions.jsonl" "$PRED_ROOT/opp/label_token/val_predictions.jsonl" \
  --comparison ctx_vs_random "$PRED_ROOT/random/label_token/val_predictions.jsonl" "$PRED_ROOT/ctx/label_token/val_predictions.jsonl" \
  --comparison full_vs_random "$PRED_ROOT/random/label_token/val_predictions.jsonl" "$PRED_ROOT/full/label_token/val_predictions.jsonl" \
  --comparison full_vs_retr "$PRED_ROOT/retr/label_token/val_predictions.jsonl" "$PRED_ROOT/full/label_token/val_predictions.jsonl" \
  --comparison full_vs_learned_fixed5 "$PRED_ROOT/learned_fixed5/label_token/val_predictions.jsonl" "$PRED_ROOT/full/label_token/val_predictions.jsonl" \
  --comparison r_only_vs_random "$PRED_ROOT/random/label_token/val_predictions.jsonl" "$PRED_ROOT/r_only/label_token/val_predictions.jsonl" \
  --early-stopping-metric macro_f1 \
  --bootstrap-samples 20000 \
  --randomization-samples 20000 \
  --seed 20260715 \
  --output-json "${OUTPUT_DIR}/paired_significance.json"

"$PYTHON_BIN" scripts/phase5_selectors/analyze/annotate_paired_significance_holm.py \
  --input-json "${OUTPUT_DIR}/paired_significance.json" \
  --primary-comparison full_vs_random \
  --secondary-comparison cor_vs_random \
  --secondary-comparison opp_vs_random \
  --secondary-comparison ctx_vs_random \
  --secondary-comparison full_vs_retr \
  --secondary-comparison full_vs_learned_fixed5 \
  --diagnostic-comparison r_only_vs_random \
  --metric macro_f1 \
  --alpha 0.05

printf '[typed-role-rescue-frozen] complete output=%s\n' "$OUTPUT_DIR"
