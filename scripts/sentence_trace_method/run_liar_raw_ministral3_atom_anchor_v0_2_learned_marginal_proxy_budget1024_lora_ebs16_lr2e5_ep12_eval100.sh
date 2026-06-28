#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

ATOM_ANCHOR_ROOT="${ATOM_ANCHOR_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1}"
SOURCE_FEATURE_ROOT="${SOURCE_FEATURE_ROOT:-${ATOM_ANCHOR_ROOT}/04_evidence_map}"
TRACE_ROOT="${TRACE_ROOT:-${ATOM_ANCHOR_ROOT}/05_mrec_v0_2_learned_marginal_proxy_budget1024}"
WEIGHT_FILE="${WEIGHT_FILE:-${ATOM_ANCHOR_ROOT}/05_mrec_v0_2_learned_marginal_proxy/weights/weights.json}"
EXPECTED_WEIGHT_FINGERPRINT="${EXPECTED_WEIGHT_FINGERPRINT:-}"
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-mrec_greedy_transition_v0_2_learned_marginal_proxy_budget1024}"
EXPECTED_ADAPTIVE_POLICY="${EXPECTED_ADAPTIVE_POLICY:-learned_marginal_proxy_v0_2}"
EXPECTED_SELECTION_POLICY="${EXPECTED_SELECTION_POLICY:-learned_marginal_proxy}"
SOURCE_SELECTOR_NAME="${SOURCE_SELECTOR_NAME:-v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10}"

MODE="${MODE:-full}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_MREC_BUILD="${FORCE_MREC_BUILD:-false}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1024}"
BUDGET_CANDIDATE_TOP_N="${BUDGET_CANDIDATE_TOP_N:-20}"
BUDGET_MAX_STEPS="${BUDGET_MAX_STEPS:-100}"
BUDGET_MIN_STEPS="${BUDGET_MIN_STEPS:-0}"
BUDGET_TARGET_RESOLVED_RATE="${BUDGET_TARGET_RESOLVED_RATE:-1.0}"
BUDGET_STOP_THRESHOLD="${BUDGET_STOP_THRESHOLD:--1000000000}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  if [[ ! -e "$path" ]]; then
    printf 'Missing %s: %s\n' "$label" "$path" >&2
    exit 2
  fi
}

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

should_build_budget_traces() {
  case "$MODE" in
    build|full) return 0 ;;
    check|train|eval) return 1 ;;
    *) printf 'Unsupported MODE=%s. Use check, build, train, eval, or full.\n' "$MODE" >&2; exit 2 ;;
  esac
}

build_budget_traces() {
  local split input_path output_trace
  IFS=',' read -r -a split_array <<< "${MREC_SPLITS:-train,val,test}"
  for split in "${split_array[@]}"; do
    split="${split// /}"
    [[ -z "$split" ]] && continue
    input_path="${SOURCE_FEATURE_ROOT}/candidate_evidence_map_features_${split}.jsonl"
    output_trace="${TRACE_ROOT}/selection_trace_${split}.jsonl"
    if [[ -f "$output_trace" && "$FORCE_MREC_BUILD" != "true" ]]; then
      printf '[atom-anchor-v0.2-budget1024] reuse budget trace: %s\n' "$output_trace"
      continue
    fi
    require_path "$input_path" "${split} atom-anchor evidence-map features"
    local sample_args=()
    if [[ "$SAMPLE_LIMIT" != "0" ]]; then
      sample_args=(--sample-limit "$SAMPLE_LIMIT")
    fi
    run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/build_mrec_traces.py \
      --input "$input_path" \
      --output-dir "$TRACE_ROOT" \
      --split "$split" \
      --candidate-top-n "$BUDGET_CANDIDATE_TOP_N" \
      --max-steps "$BUDGET_MAX_STEPS" \
      --min-steps "$BUDGET_MIN_STEPS" \
      --token-budget "$TOKEN_BUDGET" \
      --target-resolved-rate "$BUDGET_TARGET_RESOLVED_RATE" \
      --selector-name "$EXPECTED_SELECTOR_NAME" \
      --selection-policy "$EXPECTED_SELECTION_POLICY" \
      --weight-file "$WEIGHT_FILE" \
      --stop-threshold "$BUDGET_STOP_THRESHOLD" \
      --source-selector-name "$SOURCE_SELECTOR_NAME" \
      "${sample_args[@]}"
  done
}

check_budget_manifest() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '[atom-anchor-v0.2-budget1024] DRY_RUN skips budget manifest audit: %s\n' "$TRACE_ROOT"
    return 0
  fi
  "$PYTHON_BIN" - "$TRACE_ROOT" "$WEIGHT_FILE" "$EXPECTED_SELECTOR_NAME" "$EXPECTED_ADAPTIVE_POLICY" "$EXPECTED_SELECTION_POLICY" "$TOKEN_BUDGET" "$BUDGET_MAX_STEPS" "$BUDGET_STOP_THRESHOLD" "$EXPECTED_WEIGHT_FINGERPRINT" <<'PY'
import json
import sys
from pathlib import Path

trace_root = Path(sys.argv[1])
weight_file = Path(sys.argv[2])
expected_selector = sys.argv[3]
expected_adaptive = sys.argv[4]
expected_selection = sys.argv[5]
expected_token_budget = int(sys.argv[6])
expected_max_steps = int(sys.argv[7])
expected_stop_threshold = float(sys.argv[8])
expected_weight_fp = sys.argv[9]

for split in ("train", "val", "test"):
    trace_path = trace_root / f"selection_trace_{split}.jsonl"
    if not trace_path.exists():
        raise SystemExit(f"missing trace: {trace_path}")
    manifest_path = trace_root / f"manifest_{split}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selector_name") != expected_selector:
        raise SystemExit(f"{manifest_path}: selector_name={manifest.get('selector_name')!r}")
    if manifest.get("adaptive_policy") != expected_adaptive:
        raise SystemExit(f"{manifest_path}: adaptive_policy={manifest.get('adaptive_policy')!r}")
    params = manifest.get("params") or {}
    if params.get("selection_policy") != expected_selection:
        raise SystemExit(f"{manifest_path}: selection_policy={params.get('selection_policy')!r}")
    if int(params.get("token_budget") or 0) != expected_token_budget:
        raise SystemExit(f"{manifest_path}: token_budget={params.get('token_budget')!r}")
    if int(params.get("max_steps") or 0) != expected_max_steps:
        raise SystemExit(f"{manifest_path}: max_steps={params.get('max_steps')!r}")
    if abs(float(params.get("stop_threshold") or 0.0) - expected_stop_threshold) > 1e-9:
        raise SystemExit(f"{manifest_path}: stop_threshold={params.get('stop_threshold')!r}")
    recorded_weight = Path(str(params.get("weight_file") or ""))
    if recorded_weight != weight_file and recorded_weight.resolve() != weight_file.resolve():
        raise SystemExit(f"{manifest_path}: weight_file={params.get('weight_file')!r}")
    if expected_weight_fp and manifest.get("weight_fingerprint") != expected_weight_fp:
        raise SystemExit(f"{manifest_path}: weight_fingerprint={manifest.get('weight_fingerprint')!r}")
print(f"v0.2 budget1024 manifest audit ok: {trace_root}")
PY
}

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc-gemma4/bin/python" ]]; then
    export PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-gemma4/bin/python"
  else
    export PYTHON_BIN="python"
  fi
fi

export ATOM_ANCHOR_ROOT
export TRACE_ROOT
export WEIGHT_FILE
export QUALITY_AUDIT="${QUALITY_AUDIT:-${ATOM_ANCHOR_ROOT}/quality_audit_after_fix.json}"
export CASE_SUFFIX="${CASE_SUFFIX:-__atom_anchor_v0_2_learned_marginal_proxy_budget1024}"
export TRACE_TOP_K="${TRACE_TOP_K:-100}"
export TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-mrec_min}"
export EVIDENCE_TEXT_MODE="${EVIDENCE_TEXT_MODE:-full}"
export EXPECTED_SELECTOR_NAME
export EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-}"
export LORA_SUFFIX="${LORA_SUFFIX:-_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
export EVAL_SPLITS="${EVAL_SPLITS:-val,test}"
export RUN_TAU_EVAL="${RUN_TAU_EVAL:-auto}"
export RUN_LABEL="${RUN_LABEL:-atom-anchor-v0.2-learned-proxy-budget1024}"
export RUN_HEADER_LABEL="${RUN_HEADER_LABEL:-atom-anchor-v0.2-learned-proxy-budget1024-full}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-atom-anchor-v0-2}"

printf '[atom-anchor-v0.2-learned-proxy-budget1024] TRACE_ROOT=%s WEIGHT_FILE=%s TOKEN_BUDGET=%s CANDIDATE_TOP_N=%s MAX_STEPS=%s STOP_THRESHOLD=%s TRACE_TOP_K=%s EXPECTED_SELECTOR_NAME=%s EVAL_SPLITS=%s\n' \
  "$TRACE_ROOT" "$WEIGHT_FILE" "$TOKEN_BUDGET" "$BUDGET_CANDIDATE_TOP_N" "$BUDGET_MAX_STEPS" "$BUDGET_STOP_THRESHOLD" "$TRACE_TOP_K" "$EXPECTED_SELECTOR_NAME" "$EVAL_SPLITS"

require_path "$WEIGHT_FILE" "v0.2 learned marginal weight file"
if should_build_budget_traces; then
  build_budget_traces
fi
check_budget_manifest

bash "${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_1_mrec_min_lora_ebs16_lr2e5_ep12_eval100.sh"
