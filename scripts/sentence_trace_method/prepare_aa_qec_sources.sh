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
DATASETS="${DATASETS:-rawfc}"
SPLITS="${SPLITS:-train,val,test}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_AA_QEC_BUILD="${FORCE_AA_QEC_BUILD:-false}"
FORCE_STAGE="${FORCE_STAGE:-true}"

CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
MIN_CHAIN_STEPS="${MIN_CHAIN_STEPS:-5}"
MAX_CHAIN_STEPS="${MAX_CHAIN_STEPS:-10}"
CUE_POLICY="${CUE_POLICY:-qd_prefer}"
CANDIDATE_SCOPE="${CANDIDATE_SCOPE:-selected}"
SELECTION_POLICY="${SELECTION_POLICY:-keep_all_reorder}"
SOURCE_SELECTOR_NAME="${SOURCE_SELECTOR_NAME:-v0_7_budgeted_marginal_chain_adaptive5_10}"
SOURCE_GRAPH_VERSION="${SOURCE_GRAPH_VERSION:-evidence_chain_graph_v0_7}"
SOURCE_ADAPTIVE_POLICY="${SOURCE_ADAPTIVE_POLICY:-budgeted_marginal_v0_7}"
RANDOM_SEED="${RANDOM_SEED:-0}"
EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-}"
ALLOW_MULTI_SENTENCE_CANDIDATES="${ALLOW_MULTI_SENTENCE_CANDIDATES:-false}"

default_selector_name() {
  case "$SELECTION_POLICY" in
    keep_all_reorder) printf 'aa_qec_view_keep_all_%s_%s_min%s_%s\n' "$CUE_POLICY" "$CANDIDATE_SCOPE" "$MIN_CHAIN_STEPS" "$MAX_CHAIN_STEPS" ;;
    primary_secondary_order) printf 'aa_qec_view_primary_secondary_order_%s_%s_min%s_%s\n' "$CUE_POLICY" "$CANDIDATE_SCOPE" "$MIN_CHAIN_STEPS" "$MAX_CHAIN_STEPS" ;;
    shuffled) printf 'aa_qec_view_shuffled_%s_%s_min%s_%s\n' "$CUE_POLICY" "$CANDIDATE_SCOPE" "$MIN_CHAIN_STEPS" "$MAX_CHAIN_STEPS" ;;
    primary_only)
      if [[ "$CANDIDATE_SCOPE" == "top20" ]]; then
        printf 'aa_qec_full_atom_facts_abc_primary_only_%s_%s_max%s\n' "$CUE_POLICY" "$CANDIDATE_SCOPE" "$MAX_CHAIN_STEPS"
      else
        printf 'aa_qec_constrained_atom_facts_abc_primary_only_%s_%s_max%s\n' "$CUE_POLICY" "$CANDIDATE_SCOPE" "$MAX_CHAIN_STEPS"
      fi
      ;;
    primary_secondary)
      if [[ "$CANDIDATE_SCOPE" == "top20" && "$MIN_CHAIN_STEPS" == "0" && "$MAX_CHAIN_STEPS" == "0" ]]; then
        printf 'aa_qec_full_atom_facts_abc_primary_secondary_dynamic_%s_%s\n' "$CUE_POLICY" "$CANDIDATE_SCOPE"
      elif [[ "$CANDIDATE_SCOPE" == "top20" ]]; then
        printf 'aa_qec_full_atom_facts_abc_primary_secondary_%s_%s_max%s\n' "$CUE_POLICY" "$CANDIDATE_SCOPE" "$MAX_CHAIN_STEPS"
      else
        printf 'aa_qec_constrained_atom_facts_abc_primary_secondary_%s_%s_max%s\n' "$CUE_POLICY" "$CANDIDATE_SCOPE" "$MAX_CHAIN_STEPS"
      fi
      ;;
    primary_secondary_fallback_min5)
      if [[ "$CANDIDATE_SCOPE" == "top20" ]]; then
        printf 'aa_qec_full_atom_facts_abc_primary_secondary_fallback_%s_%s_min%s_%s\n' "$CUE_POLICY" "$CANDIDATE_SCOPE" "$MIN_CHAIN_STEPS" "$MAX_CHAIN_STEPS"
      else
        printf 'aa_qec_constrained_atom_facts_abc_primary_secondary_fallback_%s_%s_min%s_%s\n' "$CUE_POLICY" "$CANDIDATE_SCOPE" "$MIN_CHAIN_STEPS" "$MAX_CHAIN_STEPS"
      fi
      ;;
    primary_fallback_min5_no_secondary)
      if [[ "$CANDIDATE_SCOPE" == "top20" ]]; then
        printf 'aa_qec_full_atom_facts_abc_primary_fallback_no_secondary_%s_%s_min%s_%s\n' "$CUE_POLICY" "$CANDIDATE_SCOPE" "$MIN_CHAIN_STEPS" "$MAX_CHAIN_STEPS"
      else
        printf 'aa_qec_constrained_atom_facts_abc_primary_fallback_no_secondary_%s_%s_min%s_%s\n' "$CUE_POLICY" "$CANDIDATE_SCOPE" "$MIN_CHAIN_STEPS" "$MAX_CHAIN_STEPS"
      fi
      ;;
    *) printf 'Unsupported SELECTION_POLICY=%s\n' "$SELECTION_POLICY" >&2; exit 2 ;;
  esac
}

default_selector_adaptive_policy() {
  case "$SELECTION_POLICY" in
    keep_all_reorder|primary_secondary_order|shuffled) printf '%s\n' "aa_qec_view" ;;
    primary_only|primary_secondary|primary_secondary_fallback_min5|primary_fallback_min5_no_secondary)
      if [[ "$CANDIDATE_SCOPE" == "top20" ]]; then
        printf '%s\n' "aa_qec_full_atom_facts_abc"
      else
        printf '%s\n' "aa_qec_constrained_atom_facts_abc"
      fi
      ;;
    *) printf 'Unsupported SELECTION_POLICY=%s\n' "$SELECTION_POLICY" >&2; exit 2 ;;
  esac
}

SELECTOR_NAME="${SELECTOR_NAME:-$(default_selector_name)}"
SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-atom_anchored_qec_v1}"
SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-$(default_selector_adaptive_policy)}"

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
  printf 'outputs/selectors/atom_anchored_qec/%s/%s\n' "$dataset" "$SELECTOR_NAME"
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
    if [[ ! -f "$output_trace" || "$FORCE_AA_QEC_BUILD" == "true" ]]; then
      run_cmd env \
        SPLIT="$split" \
        PYTHON_BIN="$PYTHON_BIN" \
        INPUT="$input_path" \
        OUTPUT_DIR="$output_dir" \
        SAMPLE_LIMIT="$SAMPLE_LIMIT" \
        CANDIDATE_TOP_N="$CANDIDATE_TOP_N" \
        MIN_CHAIN_STEPS="$MIN_CHAIN_STEPS" \
        MAX_CHAIN_STEPS="$MAX_CHAIN_STEPS" \
        CUE_POLICY="$CUE_POLICY" \
        CANDIDATE_SCOPE="$CANDIDATE_SCOPE" \
        SELECTION_POLICY="$SELECTION_POLICY" \
        SOURCE_SELECTOR_NAME="$SOURCE_SELECTOR_NAME" \
        RANDOM_SEED="$RANDOM_SEED" \
        bash scripts/phase5_selectors/run/run_atom_anchored_qec.sh
    else
      printf '[aa-qec-sources] reuse trace: %s\n' "$output_trace"
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
