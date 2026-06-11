#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
DATASETS="${DATASETS:-liar_raw,rawfc}"
SPLITS="${SPLITS:-train,val,test}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
FORCE_GRAPH_BUILD="${FORCE_GRAPH_BUILD:-false}"
FORCE_STAGE="${FORCE_STAGE:-true}"

SELECTOR_NAME="${SELECTOR_NAME:-v0_7_budgeted_marginal_chain_adaptive3_10}"
SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-evidence_chain_graph_v0_7}"
SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-budgeted_marginal_v0_7}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
MIN_TOP_K="${MIN_TOP_K:-3}"
MAX_TOP_K="${MAX_TOP_K:-10}"
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
TARGET_COVERAGE="${TARGET_COVERAGE:-0.80}"
STOP_GAIN_THRESHOLD="${STOP_GAIN_THRESHOLD:-0.10}"
INSUFFICIENT_GAIN_THRESHOLD="${INSUFFICIENT_GAIN_THRESHOLD:-0.05}"

normalize_dataset() {
  case "${1//-/_}" in
    liar|liarraw|liar_raw) printf '%s\n' "liar_raw" ;;
    rawfc|raw_fc) printf '%s\n' "rawfc" ;;
    *) printf 'Unsupported dataset=%s\n' "$1" >&2; exit 2 ;;
  esac
}

input_path_for() {
  local dataset="$1"
  local split="$2"
  case "$dataset" in
    liar_raw) printf 'outputs/selectors/evidence_map_selector/v0_6b_%s/candidate_evidence_map_features_%s.jsonl\n' "$split" "$split" ;;
    rawfc) printf 'outputs/sentence_trace_method/_raw_sources/rawfc/sentence_rule_step_adaptive5_10/evidence_map_%s/candidate_evidence_map_features_%s.jsonl\n' "$split" "$split" ;;
  esac
}

graph_root_for() {
  local dataset="$1"
  case "$dataset" in
    liar_raw) printf '%s\n' "outputs/selectors/evidence_chain_graph/v0_7_budgeted_marginal_adaptive3_10" ;;
    rawfc) printf '%s\n' "outputs/sentence_trace_method/_raw_sources/rawfc/sentence_rule_step_adaptive5_10/v0_7_budgeted_marginal_adaptive3_10" ;;
  esac
}

stage_force_args=()
if [[ "$FORCE_STAGE" == "true" ]]; then
  stage_force_args=(--force)
fi

IFS=',' read -r -a dataset_array <<< "$DATASETS"
IFS=',' read -r -a split_array <<< "$SPLITS"
for raw_dataset in "${dataset_array[@]}"; do
  raw_dataset="${raw_dataset// /}"
  [[ -z "$raw_dataset" ]] && continue
  dataset="$(normalize_dataset "$raw_dataset")"
  graph_root="$(graph_root_for "$dataset")"
  for split in "${split_array[@]}"; do
    split="${split// /}"
    [[ -z "$split" ]] && continue
    input_path="$(input_path_for "$dataset" "$split")"
    output_dir="${graph_root}_${split}"
    output_trace="${output_dir}/selection_trace_${split}.jsonl"
    if [[ ! -f "$output_trace" || "$FORCE_GRAPH_BUILD" == "true" ]]; then
      SPLIT="$split" \
        INPUT="$input_path" \
        OUTPUT_DIR="$output_dir" \
        CANDIDATE_TOP_N="$CANDIDATE_TOP_N" \
        MIN_TOP_K="$MIN_TOP_K" \
        MAX_TOP_K="$MAX_TOP_K" \
        CHUNK_MMR_FINGERPRINT="$CHUNK_MMR_FINGERPRINT" \
        TARGET_COVERAGE="$TARGET_COVERAGE" \
        STOP_GAIN_THRESHOLD="$STOP_GAIN_THRESHOLD" \
        INSUFFICIENT_GAIN_THRESHOLD="$INSUFFICIENT_GAIN_THRESHOLD" \
        SAMPLE_LIMIT="$SAMPLE_LIMIT" \
        bash scripts/phase5_selectors/run/run_evidence_chain_graph_v0_7.sh
    else
      printf '[v0.7-sources] reuse graph trace: %s\n' "$output_trace"
    fi
  done

  "$PYTHON_BIN" scripts/sentence_trace_method/stage_sources.py \
    --dataset "$dataset" \
    --output-root "$OUTPUT_ROOT" \
    --source-root "$graph_root" \
    --selector-name "$SELECTOR_NAME" \
    --graph-version "$SELECTOR_GRAPH_VERSION" \
    --adaptive-policy "$SELECTOR_ADAPTIVE_POLICY" \
    --sample-limit "$SAMPLE_LIMIT" \
    --splits "$SPLITS" \
    "${stage_force_args[@]}"
done
