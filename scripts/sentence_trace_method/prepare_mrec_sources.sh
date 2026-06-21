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
DATASETS="${DATASETS:-liar_raw}"
SPLITS="${SPLITS:-train,val,test}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_MREC_BUILD="${FORCE_MREC_BUILD:-false}"
FORCE_STAGE="${FORCE_STAGE:-true}"

SELECTOR_NAME="${SELECTOR_NAME:-mrec_greedy_transition_v0_1}"
SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-mrec_trace_v0_1}"
SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-minimal_resolving_chain_v0_1}"
SOURCE_SELECTOR_NAME="${SOURCE_SELECTOR_NAME:-v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10}"
SOURCE_GRAPH_VERSION="${SOURCE_GRAPH_VERSION:-evidence_chain_graph_v0_7}"
SOURCE_ADAPTIVE_POLICY="${SOURCE_ADAPTIVE_POLICY:-budgeted_marginal_v0_7}"

CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
MAX_STEPS="${MAX_STEPS:-10}"
MIN_STEPS="${MIN_STEPS:-0}"
TOKEN_BUDGET="${TOKEN_BUDGET:-0}"
TARGET_RESOLVED_RATE="${TARGET_RESOLVED_RATE:-0.80}"
CONTINUE_AFTER_TARGET_FOR_CONTRAST="${CONTINUE_AFTER_TARGET_FOR_CONTRAST:-false}"
POST_TARGET_FILL_POLICY="${POST_TARGET_FILL_POLICY:-contrast_only}"
DISABLE_FALLBACK="${DISABLE_FALLBACK:-false}"
EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-}"
ALLOW_MULTI_SENTENCE_CANDIDATES="${ALLOW_MULTI_SENTENCE_CANDIDATES:-false}"

normalize_dataset() {
  case "${1//-/_}" in
    liar|liarraw|liar_raw) printf '%s\n' "liar_raw" ;;
    rawfc|raw_fc) printf '%s\n' "rawfc" ;;
    *) printf 'Unsupported dataset=%s\n' "$1" >&2; exit 2 ;;
  esac
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

source_trace_for() {
  local dataset="$1"
  local split="$2"
  printf '%s/_sources/%s/%s/%s/selection_trace_%s.jsonl\n' \
    "$OUTPUT_ROOT" "$dataset" "$SOURCE_SELECTOR_NAME" "$split" "$split"
}

graph_root_for() {
  local dataset="$1"
  printf 'outputs/selectors/mrec/%s/%s\n' "$dataset" "$SELECTOR_NAME"
}

stage_force_args=()
if [[ "$FORCE_STAGE" == "true" ]]; then
  stage_force_args=(--force)
fi

stage_source_args=()
if [[ -n "$EXPECTED_CHUNK_MMR_FINGERPRINT" ]]; then
  stage_source_args+=(--expected-fingerprint "$EXPECTED_CHUNK_MMR_FINGERPRINT")
fi
if [[ "$ALLOW_MULTI_SENTENCE_CANDIDATES" == "true" || "$ALLOW_MULTI_SENTENCE_CANDIDATES" == "1" ]]; then
  stage_source_args+=(--allow-multi-sentence-candidates)
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
    input_path="$(source_trace_for "$dataset" "$split")"
    output_dir="${graph_root}_${split}"
    output_trace="${output_dir}/selection_trace_${split}.jsonl"
    if [[ ! -f "$output_trace" || "$FORCE_MREC_BUILD" == "true" ]]; then
      run_cmd env \
        SPLIT="$split" \
        PYTHON_BIN="$PYTHON_BIN" \
        INPUT="$input_path" \
        OUTPUT_DIR="$output_dir" \
        SAMPLE_LIMIT="$SAMPLE_LIMIT" \
        CANDIDATE_TOP_N="$CANDIDATE_TOP_N" \
        MAX_STEPS="$MAX_STEPS" \
        MIN_STEPS="$MIN_STEPS" \
        TOKEN_BUDGET="$TOKEN_BUDGET" \
        TARGET_RESOLVED_RATE="$TARGET_RESOLVED_RATE" \
        CONTINUE_AFTER_TARGET_FOR_CONTRAST="$CONTINUE_AFTER_TARGET_FOR_CONTRAST" \
        POST_TARGET_FILL_POLICY="$POST_TARGET_FILL_POLICY" \
        DISABLE_FALLBACK="$DISABLE_FALLBACK" \
        SELECTOR_NAME="$SELECTOR_NAME" \
        SOURCE_SELECTOR_NAME="$SOURCE_SELECTOR_NAME" \
        bash scripts/phase5_selectors/run/run_mrec_traces.sh
    else
      printf '[mrec-sources] reuse trace: %s\n' "$output_trace"
    fi
  done

  run_cmd "$PYTHON_BIN" scripts/sentence_trace_method/stage_sources.py \
    --dataset "$dataset" \
    --output-root "$OUTPUT_ROOT" \
    --source-root "$graph_root" \
    --selector-name "$SELECTOR_NAME" \
    --graph-version "$SELECTOR_GRAPH_VERSION" \
    --adaptive-policy "$SELECTOR_ADAPTIVE_POLICY" \
    --sample-limit "$SAMPLE_LIMIT" \
    --splits "$SPLITS" \
    "${stage_source_args[@]}" \
    "${stage_force_args[@]}"
done
