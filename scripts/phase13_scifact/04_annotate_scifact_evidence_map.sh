#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python" ]]; then
    export PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python"
  else
    export PYTHON_BIN="python"
  fi
fi
DRY_RUN="${DRY_RUN:-false}"
ATOM_ANCHOR_ROOT="${ATOM_ANCHOR_ROOT:-outputs/selectors/scifact_atom_anchor}"
ATOM_UNION_ROOT="${ATOM_UNION_ROOT:-${ATOM_ANCHOR_ROOT}/03_atom_union}"
EVIDENCE_MAP_ROOT="${EVIDENCE_MAP_ROOT:-${ATOM_ANCHOR_ROOT}/04_evidence_map}"
QUALITY_AUDIT="${QUALITY_AUDIT:-${ATOM_ANCHOR_ROOT}/quality_audit_after_fix.json}"
SPLITS="${SPLITS:-train val test}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"

FORCE_EVIDENCE_MAP_PREPARE="${FORCE_EVIDENCE_MAP_PREPARE:-false}"
FORCE_TEACHER="${FORCE_TEACHER:-false}"
FORCE_POSTPROCESS="${FORCE_POSTPROCESS:-false}"
RUN_AUDIT="${RUN_AUDIT:-true}"
MOCK_EVIDENCE_MAPS="${MOCK_EVIDENCE_MAPS:-false}"
NO_PROGRESS="${NO_PROGRESS:-false}"

PROMPT_VERSION="${PROMPT_VERSION:-evidence_map_v0_7_atom_facts_abc}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://api.deepseek.com}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
CONCURRENCY="${CONCURRENCY:-128}"
REQUESTS_PER_MINUTE="${REQUESTS_PER_MINUTE:-0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TOP_P="${TOP_P:-1.0}"
THINKING_TYPE="${THINKING_TYPE:-disabled}"

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

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

sample_args=()
if [[ "$SAMPLE_LIMIT" != "0" ]]; then
  sample_args=(--sample-limit "$SAMPLE_LIMIT")
fi

progress_args=()
if [[ "$NO_PROGRESS" == "true" || "$NO_PROGRESS" == "1" || "$NO_PROGRESS" == "True" ]]; then
  progress_args=(--no-progress)
fi

printf '[scifact-04] ATOM_ANCHOR_ROOT=%s EVIDENCE_MAP_ROOT=%s SPLITS=%s CANDIDATE_TOP_N=%s MODEL=%s MOCK=%s NO_PROGRESS=%s\n' \
  "$ATOM_ANCHOR_ROOT" "$EVIDENCE_MAP_ROOT" "$SPLITS" "$CANDIDATE_TOP_N" "$TEACHER_MODEL" "$MOCK_EVIDENCE_MAPS" "$NO_PROGRESS"

for split in $SPLITS; do
  atom_union_pool="${ATOM_UNION_ROOT}/atom_union_candidate_pool_${split}.jsonl"
  candidate_pool="${EVIDENCE_MAP_ROOT}/evidence_map_candidate_pool_${split}.jsonl"
  annotations="${EVIDENCE_MAP_ROOT}/deepseek_evidence_map_annotations_${split}.jsonl"
  features="${EVIDENCE_MAP_ROOT}/candidate_evidence_map_features_${split}.jsonl"
  require_path "$atom_union_pool" "${split} atom union candidate pool"

  if [[ -s "$candidate_pool" && "$FORCE_EVIDENCE_MAP_PREPARE" != "true" ]]; then
    printf '[scifact-04] reuse evidence-map candidate pool: %s\n' "$candidate_pool"
  else
    run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/prepare_evidence_map_candidate_pool.py \
      --input-candidate-file "$atom_union_pool" \
      --output-dir "$EVIDENCE_MAP_ROOT" \
      --split "$split" \
      --candidate-source atom_union \
      --candidate-top-n "$CANDIDATE_TOP_N" \
      "${sample_args[@]}"
  fi

  if [[ -s "$annotations" && "$FORCE_TEACHER" != "true" ]]; then
    printf '[scifact-04] reuse evidence-map annotations: %s\n' "$annotations"
  else
    require_path "$candidate_pool" "${split} evidence-map candidate pool"
    mock_args=()
    if [[ "$MOCK_EVIDENCE_MAPS" == "true" ]]; then
      mock_args=(--mock-maps)
    fi
    run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/annotate_evidence_maps_deepseek.py \
      --candidate-pool "$candidate_pool" \
      --output-dir "$EVIDENCE_MAP_ROOT" \
      --split "$split" \
      --prompt-version "$PROMPT_VERSION" \
      --base-url "$TEACHER_BASE_URL" \
      --model "$TEACHER_MODEL" \
      --api-key-env "$TEACHER_API_KEY_ENV" \
      --concurrency "$CONCURRENCY" \
      --requests-per-minute "$REQUESTS_PER_MINUTE" \
      --max-tokens "$MAX_TOKENS" \
      --top-p "$TOP_P" \
      --thinking-type "$THINKING_TYPE" \
      --resume \
      "${progress_args[@]}" \
      "${mock_args[@]}" \
      "${sample_args[@]}"
  fi

  if [[ -s "$features" && "$FORCE_POSTPROCESS" != "true" ]]; then
    printf '[scifact-04] reuse evidence-map features: %s\n' "$features"
  else
    require_path "$candidate_pool" "${split} evidence-map candidate pool"
    require_path "$annotations" "${split} evidence-map annotations"
    run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/postprocess_evidence_maps.py \
      --candidate-pool "$candidate_pool" \
      --annotations "$annotations" \
      --output-dir "$EVIDENCE_MAP_ROOT" \
      --split "$split" \
      "${sample_args[@]}"
  fi
done

if [[ "$RUN_AUDIT" == "true" ]]; then
  read -r -a audit_splits <<< "$SPLITS"
  run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/audit_atom_anchor_outputs.py \
    --root "$ATOM_ANCHOR_ROOT" \
    --output "$QUALITY_AUDIT" \
    --splits "${audit_splits[@]}"
fi
