#!/usr/bin/env bash
# Re-run Evidence-Map ablation under minmax5_10 (K* is variable) on LIAR-RAW.
#
# Why this script exists:
#   The previous map-ablation run used fixed_topk k=5 (minmax5_5), which pins
#   K*=5 and nullifies the selector's resolution-driven stopping. Under minmax5_10
#   the truncation point K* becomes a function of the resolved_atom_rate, so
#   removing map signals shows up as a shift in K* and verifier F1. Bugs B1
#   (missing "medium" in _DIRECTNESS_FACTOR) and B2 (_operation_for_transition
#   ignoring directness for OPEN) have been fixed so no_directness now genuinely
#   degrades atom resolution.
#
# What it runs per variant (no_map / no_directness / no_confidence / no_relation):
#   1. train selector weights  (map_ablation_mode applied to feature mask)
#   2. build mrec traces        (map_ablation_mode applied to transition inference)
#   3. build verifier data      (minmax5_10 prompt evidence policy -> K*)
#   4. train verifier (LoRA)    (RETRAIN_VERIFIER=true)  OR  reuse main checkpoint
#   5. eval on test             (label_token + logit_adjust_tau0p75)
#
# Usage:
#   bash run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_minmax5_10_map_ablation_lora_ebs16_lr2e5_ep12_eval100.sh
#
# Env knobs (all optional, shown with defaults):
#   VARIANTS="no_map no_directness no_confidence no_relation"
#   MODE=full                         # build|train|eval|full  (phase selector for the base wrapper)
#   RETRAIN_VERIFIER=true             # true= retrain verifier per variant; false= reuse main checkpoint (build+infer only)
#   FORCE_WEIGHT_TRAIN=false          # force selector weight retrain even if weights.json exists
#   FORCE_MREC_BUILD=false            # force trace rebuild even if selection_trace_*.jsonl exists
#   DRY_RUN=false                     # true= print commands without executing
#   SAMPLE_LIMIT=0                    # >0 subsamples each split (for smoke tests)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

VARIANTS="${VARIANTS:-no_map no_directness no_confidence no_relation}"
MODE="${MODE:-full}"
RETRAIN_VERIFIER="${RETRAIN_VERIFIER:-true}"
FORCE_WEIGHT_TRAIN="${FORCE_WEIGHT_TRAIN:-false}"
FORCE_MREC_BUILD="${FORCE_MREC_BUILD:-false}"
DRY_RUN="${DRY_RUN:-false}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

BASE_WRAPPER="${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh"
if [[ ! -f "$BASE_WRAPPER" ]]; then
  printf 'Missing base wrapper: %s\n' "$BASE_WRAPPER" >&2
  exit 2
fi

printf '[map-ablation-minmax5-10] VARIANTS=[%s] MODE=%s RETRAIN_VERIFIER=%s DRY_RUN=%s SAMPLE_LIMIT=%s\n' \
  "$VARIANTS" "$MODE" "$RETRAIN_VERIFIER" "$DRY_RUN" "$SAMPLE_LIMIT"

for variant in $VARIANTS; do
  printf '\n========== map_ablation variant: %s (minmax5_10) ==========\n' "$variant"

  export MREC_POLICY_CONFIG="configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_minmax5_10_map_ablation_${variant}.yaml"
  export MAP_ABLATION_MODE="$variant"
  export MODE
  export FORCE_WEIGHT_TRAIN
  export FORCE_MREC_BUILD
  export DRY_RUN
  export SAMPLE_LIMIT

  # When reusing the main-method verifier (RETRAIN_VERIFIER=false), run only the
  # build + infer phases: build trains selector weights and produces traces, but
  # the verifier LoRA is not retrained; eval points eval_lora at the main checkpoint.
  # Implementation note: the base wrapper maps MODE=full to build->train->eval.
  # To reuse the checkpoint, set MODE=build here and run eval separately against
  # the main checkpoint path.
  if [[ "$RETRAIN_VERIFIER" != "true" ]]; then
    export MODE="build"
    printf '[map-ablation-minmax5-10] RETRAIN_VERIFIER=false -> building traces only; eval against main checkpoint separately.\n'
  fi

  bash "$BASE_WRAPPER"

  printf '[map-ablation-minmax5-10] variant %s done.\n' "$variant"
done

printf '\n[map-ablation-minmax5-10] all variants finished.\n'
printf 'Expected metrics per variant:\n'
for variant in $VARIANTS; do
  base="outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10_map_ablation_${variant}_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw"
  printf '  %s:\n    %s/eval/test/best/label_token/metrics.json\n    %s/eval/test/best/label_token_logit_adjust_tau0p75/metrics.json\n' \
    "$variant" "$base" "$base"
done
printf '\nCompare against the full_map baseline (minmax5_10 main method):\n'
printf '  outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/eval/test/best/label_token/metrics.json\n'
