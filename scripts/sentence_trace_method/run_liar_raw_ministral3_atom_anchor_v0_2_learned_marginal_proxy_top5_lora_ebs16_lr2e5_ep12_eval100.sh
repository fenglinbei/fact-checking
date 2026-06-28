#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

ATOM_ANCHOR_ROOT="${ATOM_ANCHOR_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1}"
TRACE_ROOT="${TRACE_ROOT:-${ATOM_ANCHOR_ROOT}/05_mrec_v0_2_learned_marginal_proxy}"
WEIGHT_FILE="${WEIGHT_FILE:-${TRACE_ROOT}/weights/weights.json}"
EXPECTED_WEIGHT_FINGERPRINT="${EXPECTED_WEIGHT_FINGERPRINT:-}"
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-mrec_greedy_transition_v0_2_learned_marginal_proxy}"
EXPECTED_ADAPTIVE_POLICY="${EXPECTED_ADAPTIVE_POLICY:-learned_marginal_proxy_v0_2}"
EXPECTED_SELECTION_POLICY="${EXPECTED_SELECTION_POLICY:-learned_marginal_proxy}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    return 0
  fi
  if [[ ! -e "$path" ]]; then
    printf 'Missing %s: %s\n' "$label" "$path" >&2
    exit 2
  fi
}

check_v0_2_manifest() {
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    printf '[atom-anchor-v0.2-learned-proxy-top5] DRY_RUN skips v0.2 manifest audit: %s\n' "$TRACE_ROOT"
    return 0
  fi
  "$PYTHON_BIN" - "$TRACE_ROOT" "$WEIGHT_FILE" "$EXPECTED_SELECTOR_NAME" "$EXPECTED_ADAPTIVE_POLICY" "$EXPECTED_SELECTION_POLICY" "$EXPECTED_WEIGHT_FINGERPRINT" <<'PY'
import json
import sys
from pathlib import Path

trace_root = Path(sys.argv[1])
weight_file = Path(sys.argv[2])
expected_selector = sys.argv[3]
expected_adaptive = sys.argv[4]
expected_selection = sys.argv[5]
expected_weight_fp = sys.argv[6]

for split in ("train", "val", "test"):
    manifest_path = trace_root / f"manifest_{split}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selector_name") != expected_selector:
        raise SystemExit(f"{manifest_path}: selector_name={manifest.get('selector_name')!r}")
    if manifest.get("adaptive_policy") != expected_adaptive:
        raise SystemExit(f"{manifest_path}: adaptive_policy={manifest.get('adaptive_policy')!r}")
    params = manifest.get("params") or {}
    if params.get("selection_policy") != expected_selection:
        raise SystemExit(f"{manifest_path}: selection_policy={params.get('selection_policy')!r}")
    if int(params.get("min_steps", -1)) != 5 or int(params.get("max_steps", -1)) != 10:
        raise SystemExit(f"{manifest_path}: unexpected min/max steps {params.get('min_steps')}/{params.get('max_steps')}")
    recorded_weight = Path(str(params.get("weight_file") or ""))
    if recorded_weight != weight_file and recorded_weight.resolve() != weight_file.resolve():
        raise SystemExit(f"{manifest_path}: weight_file={params.get('weight_file')!r}")
    if expected_weight_fp and manifest.get("weight_fingerprint") != expected_weight_fp:
        raise SystemExit(f"{manifest_path}: weight_fingerprint={manifest.get('weight_fingerprint')!r}")
print(f"v0.2 learned marginal manifest audit ok: {trace_root}")
PY
}

export ATOM_ANCHOR_ROOT
export TRACE_ROOT
export WEIGHT_FILE
export QUALITY_AUDIT="${QUALITY_AUDIT:-${ATOM_ANCHOR_ROOT}/quality_audit_after_fix.json}"
export CASE_SUFFIX="${CASE_SUFFIX:-__atom_anchor_v0_2_learned_marginal_proxy_top5}"
export TRACE_TOP_K="${TRACE_TOP_K:-5}"
export TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-mrec_min}"
export EVIDENCE_TEXT_MODE="${EVIDENCE_TEXT_MODE:-full}"
export EXPECTED_SELECTOR_NAME
export EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-}"
export LORA_SUFFIX="${LORA_SUFFIX:-_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
export EVAL_SPLITS="${EVAL_SPLITS:-val,test}"
export RUN_TAU_EVAL="${RUN_TAU_EVAL:-auto}"
export RUN_LABEL="${RUN_LABEL:-atom-anchor-v0.2-learned-proxy-top5}"
export RUN_HEADER_LABEL="${RUN_HEADER_LABEL:-atom-anchor-v0.2-learned-proxy-top5-full}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-atom-anchor-v0-2}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc-gemma4/bin/python" ]]; then
    export PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-gemma4/bin/python"
  else
    export PYTHON_BIN="python"
  fi
fi

printf '[atom-anchor-v0.2-learned-proxy-top5] TRACE_ROOT=%s WEIGHT_FILE=%s SELECTION_POLICY=%s ADAPTIVE_POLICY=%s EXPECTED_SELECTOR_NAME=%s TRACE_TOP_K=%s EVAL_SPLITS=%s\n' \
  "$TRACE_ROOT" "$WEIGHT_FILE" "$EXPECTED_SELECTION_POLICY" "$EXPECTED_ADAPTIVE_POLICY" "$EXPECTED_SELECTOR_NAME" "$TRACE_TOP_K" "$EVAL_SPLITS"

require_path "$WEIGHT_FILE" "v0.2 learned marginal weight file"
require_path "${TRACE_ROOT}/selection_trace_train.jsonl" "train v0.2 MREC trace"
require_path "${TRACE_ROOT}/selection_trace_val.jsonl" "val v0.2 MREC trace"
require_path "${TRACE_ROOT}/selection_trace_test.jsonl" "test v0.2 MREC trace"
require_path "${TRACE_ROOT}/manifest_train.json" "train v0.2 MREC manifest"
require_path "${TRACE_ROOT}/manifest_val.json" "val v0.2 MREC manifest"
require_path "${TRACE_ROOT}/manifest_test.json" "test v0.2 MREC manifest"
check_v0_2_manifest

bash "${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_1_mrec_min_lora_ebs16_lr2e5_ep12_eval100.sh"
