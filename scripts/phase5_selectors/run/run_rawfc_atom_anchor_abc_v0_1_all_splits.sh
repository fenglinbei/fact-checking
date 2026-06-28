#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

LOAD_ENV="${LOAD_ENV:-true}"
if [[ "${LOAD_ENV}" == "true" || "${LOAD_ENV}" == "1" ]]; then
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
  fi
fi

SPLITS="${SPLITS:-train val test}"
PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"
EXPERIMENT="${EXPERIMENT:-v0.6/v0_6c_rawfc3_rule_step_adaptive5_10_abc_chunking}"
CONFIG="${CONFIG:-configs/experiment/v0.6/v0_6c_rawfc3_rule_step_adaptive5_10_abc_chunking.yaml}"
RAW_DATASET="${RAW_DATASET:-rawfc}"
LABEL_SCHEMA="${LABEL_SCHEMA:-rawfc3}"

TRAIN_RAW="${TRAIN_RAW:-data/raw/RAWFC/train.json}"
VAL_RAW="${VAL_RAW:-data/raw/RAWFC/val.json}"
TEST_RAW="${TEST_RAW:-data/raw/RAWFC/test.json}"

ATOM_ANCHOR_ROOT="${ATOM_ANCHOR_ROOT:-outputs/selectors/atom_anchor/rawfc_abc_v0_1}"
CLAIM_ATOM_ROOT="${CLAIM_ATOM_ROOT:-${ATOM_ANCHOR_ROOT}/01_claim_atoms}"
ATOM_RETRIEVAL_ROOT="${ATOM_RETRIEVAL_ROOT:-${ATOM_ANCHOR_ROOT}/02_atom_retrieval}"
ATOM_UNION_ROOT="${ATOM_UNION_ROOT:-${ATOM_ANCHOR_ROOT}/03_atom_union}"
EVIDENCE_MAP_ROOT="${EVIDENCE_MAP_ROOT:-${ATOM_ANCHOR_ROOT}/04_evidence_map}"
QUALITY_AUDIT="${QUALITY_AUDIT:-${ATOM_ANCHOR_ROOT}/quality_audit_after_fix.json}"

RUN_CACHE_BUILD="${RUN_CACHE_BUILD:-true}"
RUN_CLAIM_ATOMS="${RUN_CLAIM_ATOMS:-true}"
RUN_ATOM_RETRIEVAL="${RUN_ATOM_RETRIEVAL:-true}"
RUN_ATOM_UNION="${RUN_ATOM_UNION:-true}"
RUN_EVIDENCE_MAP_PREPARE="${RUN_EVIDENCE_MAP_PREPARE:-true}"
RUN_TEACHER="${RUN_TEACHER:-true}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-true}"
RUN_AUDIT="${RUN_AUDIT:-true}"

FORCE_CACHE_BUILD="${FORCE_CACHE_BUILD:-false}"
FORCE_CLAIM_ATOMS="${FORCE_CLAIM_ATOMS:-false}"
FORCE_ATOM_RETRIEVAL="${FORCE_ATOM_RETRIEVAL:-false}"
FORCE_ATOM_UNION="${FORCE_ATOM_UNION:-false}"
FORCE_EVIDENCE_MAP_PREPARE="${FORCE_EVIDENCE_MAP_PREPARE:-false}"
FORCE_TEACHER="${FORCE_TEACHER:-false}"
FORCE_POSTPROCESS="${FORCE_POSTPROCESS:-false}"

DRY_RUN="${DRY_RUN:-false}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
CHILD_SAMPLE_LIMIT=""
if [[ -n "${SAMPLE_LIMIT}" && "${SAMPLE_LIMIT}" != "0" ]]; then
  CHILD_SAMPLE_LIMIT="${SAMPLE_LIMIT}"
fi

EMBEDDER_MODEL="${EMBEDDER_MODEL:-/data/models/bge-base-en-v1.5}"
EMBEDDER_DEVICE="${EMBEDDER_DEVICE:-cuda}"
EMBEDDER_MAX_LENGTH="${EMBEDDER_MAX_LENGTH:-256}"
EMBEDDER_BATCH_SIZE="${EMBEDDER_BATCH_SIZE:-64}"
EMBEDDER_PRECISION="${EMBEDDER_PRECISION:-fp32}"
PER_ATOM_KEEP="${PER_ATOM_KEEP:-20}"
MERGED_POOL_SIZE="${MERGED_POOL_SIZE:-20}"
SELECTOR_TOP_K="${SELECTOR_TOP_K:-5}"
BASELINE_TOP_K="${BASELINE_TOP_K:-$SELECTOR_TOP_K}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"

MOCK_ATOMS="${MOCK_ATOMS:-false}"
MOCK_EVIDENCE_MAPS="${MOCK_EVIDENCE_MAPS:-false}"
ATOM_BASE_URL="${ATOM_BASE_URL:-https://api.deepseek.com}"
ATOM_MODEL="${ATOM_MODEL:-deepseek-v4-flash}"
ATOM_API_KEY_ENV="${ATOM_API_KEY_ENV:-DEEPSEEK_API_KEY}"
ATOM_API_CONCURRENCY="${ATOM_API_CONCURRENCY:-128}"
ATOM_MAX_TOKENS="${ATOM_MAX_TOKENS:-2048}"
ATOM_THINKING_TYPE="${ATOM_THINKING_TYPE:-disabled}"

TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://api.deepseek.com}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
CONCURRENCY="${CONCURRENCY:-128}"
REQUESTS_PER_MINUTE="${REQUESTS_PER_MINUTE:-2048}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TOP_P="${TOP_P:-1.0}"
THINKING_TYPE="${THINKING_TYPE:-disabled}"
PROMPT_VERSION="${PROMPT_VERSION:-atom_evidence_map_v0_1}"

enabled() {
  case "${1:-false}" in
    true|1|yes|YES|on) return 0 ;;
    *) return 1 ;;
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

raw_path_for_split() {
  local split="$1"
  case "$split" in
    train) printf '%s' "$TRAIN_RAW" ;;
    val) printf '%s' "$VAL_RAW" ;;
    test) printf '%s' "$TEST_RAW" ;;
    *) printf '[rawfc-atom-anchor-abc-v0.1] unsupported split: %s\n' "$split" >&2; exit 2 ;;
  esac
}

all_cache_files_exist() {
  local split
  for split in ${SPLITS}; do
    [[ -f "outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/${split}.pkl" ]] || return 1
  done
  return 0
}

FP_ARGS=(--config "${CONFIG}")
if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
  FP_ARGS+=(--sample-limit "${SAMPLE_LIMIT}")
fi
if [[ -z "${CHUNK_MMR_FINGERPRINT:-}" ]]; then
  if [[ "$DRY_RUN" == "true" ]]; then
    CHUNK_MMR_FINGERPRINT="dry-run-fingerprint"
  else
    CHUNK_MMR_FINGERPRINT="$(PYTHONPATH=src "${PYTHON_BIN}" scripts/phase5_selectors/build/print_chunk_mmr_fingerprint.py "${FP_ARGS[@]}")"
  fi
fi

printf '[rawfc-atom-anchor-abc-v0.1] splits      : %s\n' "$SPLITS"
printf '[rawfc-atom-anchor-abc-v0.1] config      : %s\n' "$CONFIG"
printf '[rawfc-atom-anchor-abc-v0.1] fingerprint : %s\n' "$CHUNK_MMR_FINGERPRINT"
printf '[rawfc-atom-anchor-abc-v0.1] root        : %s\n' "$ATOM_ANCHOR_ROOT"
printf '[rawfc-atom-anchor-abc-v0.1] top-k       : atom_per=%s atom_pool=%s baseline=%s selector=%s evidence_map=%s\n' \
  "$PER_ATOM_KEEP" "$MERGED_POOL_SIZE" "$BASELINE_TOP_K" "$SELECTOR_TOP_K" "$CANDIDATE_TOP_N"
printf '[rawfc-atom-anchor-abc-v0.1] stages      : cache=%s atoms=%s retrieval=%s union=%s prepare=%s teacher=%s postprocess=%s audit=%s\n' \
  "$RUN_CACHE_BUILD" "$RUN_CLAIM_ATOMS" "$RUN_ATOM_RETRIEVAL" "$RUN_ATOM_UNION" "$RUN_EVIDENCE_MAP_PREPARE" "$RUN_TEACHER" "$RUN_POSTPROCESS" "$RUN_AUDIT"

if enabled "$RUN_CACHE_BUILD"; then
  if all_cache_files_exist && [[ "$FORCE_CACHE_BUILD" != "true" ]]; then
    printf '[rawfc-atom-anchor-abc-v0.1] reuse chunk cache: outputs/cache/chunk_mmr/%s\n' "$CHUNK_MMR_FINGERPRINT"
  else
    PIPELINE_ARGS=(
      "experiment=${EXPERIMENT}"
      "pipeline.mode=build"
      "pipeline.force.build=${FORCE_CACHE_BUILD}"
    )
    if [[ "${SAMPLE_LIMIT}" != "0" ]]; then
      PIPELINE_ARGS+=("+build.data.sample_limit=${SAMPLE_LIMIT}")
    fi
    run_cmd env PYTHONPATH=src "$PYTHON_BIN" -m fact_checking.pipeline.run "${PIPELINE_ARGS[@]}"
  fi
fi

for split in ${SPLITS}; do
  raw_path="$(raw_path_for_split "$split")"
  chunk_cache="outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/${split}.pkl"
  claim_atoms="${CLAIM_ATOM_ROOT}/claim_atoms_${split}.jsonl"
  retrieval_trace="${ATOM_RETRIEVAL_ROOT}/retrieval_trace_${split}.jsonl"
  atom_union_pool="${ATOM_UNION_ROOT}/atom_union_candidate_pool_${split}.jsonl"
  candidate_pool="${EVIDENCE_MAP_ROOT}/evidence_map_candidate_pool_${split}.jsonl"
  annotations="${EVIDENCE_MAP_ROOT}/deepseek_evidence_map_annotations_${split}.jsonl"
  features="${EVIDENCE_MAP_ROOT}/candidate_evidence_map_features_${split}.jsonl"

  printf '[rawfc-atom-anchor-abc-v0.1] split=%s raw=%s chunk_cache=%s\n' "$split" "$raw_path" "$chunk_cache"

  sample_args=()
  if [[ -n "$CHILD_SAMPLE_LIMIT" ]]; then
    sample_args=(--sample-limit "$CHILD_SAMPLE_LIMIT")
  fi

  if enabled "$RUN_CLAIM_ATOMS"; then
    if [[ -s "$claim_atoms" && "$FORCE_CLAIM_ATOMS" != "true" ]]; then
      printf '[rawfc-atom-anchor-abc-v0.1] reuse claim atoms: %s\n' "$claim_atoms"
    else
      mock_atom_args=()
      if enabled "$MOCK_ATOMS"; then
        mock_atom_args=(--mock-atoms)
      fi
      run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/generate_claim_atom_cache.py \
        --input-mode raw_split \
        --dataset "$RAW_DATASET" \
        --label-schema "$LABEL_SCHEMA" \
        --raw-path "$raw_path" \
        --output-dir "$CLAIM_ATOM_ROOT" \
        --split "$split" \
        --atom-cache-dir "${CLAIM_ATOM_ROOT}/cache" \
        --atom-base-url "$ATOM_BASE_URL" \
        --atom-model "$ATOM_MODEL" \
        --atom-api-key-env "$ATOM_API_KEY_ENV" \
        --api-concurrency "$ATOM_API_CONCURRENCY" \
        --max-tokens "$ATOM_MAX_TOKENS" \
        --thinking-type "$ATOM_THINKING_TYPE" \
        --no-progress \
        "${mock_atom_args[@]}" \
        "${sample_args[@]}"
    fi
  fi

  if enabled "$RUN_ATOM_RETRIEVAL"; then
    if [[ -s "$retrieval_trace" && "$FORCE_ATOM_RETRIEVAL" != "true" ]]; then
      printf '[rawfc-atom-anchor-abc-v0.1] reuse atom retrieval: %s\n' "$retrieval_trace"
    else
      require_path "$claim_atoms" "${split} claim atoms"
      require_path "$chunk_cache" "${split} RAWFC ABC chunk cache"
      run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/build_atom_conditioned_retrieval.py \
        --claim-atoms-jsonl "$claim_atoms" \
        --chunk-cache-path "$chunk_cache" \
        --split "$split" \
        --output-dir "$ATOM_RETRIEVAL_ROOT" \
        --embedder-model "$EMBEDDER_MODEL" \
        --device "$EMBEDDER_DEVICE" \
        --embedder-max-length "$EMBEDDER_MAX_LENGTH" \
        --embedder-batch-size "$EMBEDDER_BATCH_SIZE" \
        --precision "$EMBEDDER_PRECISION" \
        --per-atom-keep "$PER_ATOM_KEEP" \
        --merged-pool-size "$MERGED_POOL_SIZE" \
        --selector-top-k "$SELECTOR_TOP_K" \
        --baseline-top-k "$BASELINE_TOP_K" \
        --oracle-results "" \
        --no-progress \
        "${sample_args[@]}"
    fi
  fi

  if enabled "$RUN_ATOM_UNION"; then
    if [[ -s "$atom_union_pool" && "$FORCE_ATOM_UNION" != "true" ]]; then
      printf '[rawfc-atom-anchor-abc-v0.1] reuse atom union: %s\n' "$atom_union_pool"
    else
      require_path "${ATOM_RETRIEVAL_ROOT}/baseline_claim_mmr_selected_${split}.jsonl" "${split} baseline claim-MMR selected rows"
      require_path "${ATOM_RETRIEVAL_ROOT}/merged_candidate_pool_${split}.jsonl" "${split} atom merged candidate pool"
      run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/build_atom_retrieval_union.py \
        --baseline-jsonl "${ATOM_RETRIEVAL_ROOT}/baseline_claim_mmr_selected_${split}.jsonl" \
        --atom-pool-jsonl "${ATOM_RETRIEVAL_ROOT}/merged_candidate_pool_${split}.jsonl" \
        --split "$split" \
        --output-dir "$ATOM_UNION_ROOT" \
        --selector-top-k "$SELECTOR_TOP_K" \
        --oracle-results "" \
        "${sample_args[@]}"
    fi
  fi

  if enabled "$RUN_EVIDENCE_MAP_PREPARE"; then
    if [[ -s "$candidate_pool" && "$FORCE_EVIDENCE_MAP_PREPARE" != "true" ]]; then
      printf '[rawfc-atom-anchor-abc-v0.1] reuse evidence-map pool: %s\n' "$candidate_pool"
    else
      require_path "$atom_union_pool" "${split} atom union candidate pool"
      run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/prepare_evidence_map_candidate_pool.py \
        --input-candidate-file "$atom_union_pool" \
        --output-dir "$EVIDENCE_MAP_ROOT" \
        --split "$split" \
        --candidate-source atom_union \
        --candidate-top-n "$CANDIDATE_TOP_N" \
        "${sample_args[@]}"
    fi
  fi

  if enabled "$RUN_TEACHER"; then
    if [[ -s "$annotations" && "$FORCE_TEACHER" != "true" ]]; then
      printf '[rawfc-atom-anchor-abc-v0.1] reuse evidence-map annotations: %s\n' "$annotations"
    else
      require_path "$candidate_pool" "${split} evidence-map candidate pool"
      mock_map_args=()
      if enabled "$MOCK_EVIDENCE_MAPS"; then
        mock_map_args=(--mock-maps)
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
        --no-progress \
        "${mock_map_args[@]}" \
        "${sample_args[@]}"
    fi
  fi

  if enabled "$RUN_POSTPROCESS"; then
    if [[ -s "$features" && "$FORCE_POSTPROCESS" != "true" ]]; then
      printf '[rawfc-atom-anchor-abc-v0.1] reuse evidence-map features: %s\n' "$features"
    else
      require_path "$candidate_pool" "${split} evidence-map candidate pool"
      require_path "$annotations" "${split} evidence-map annotations"
      run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/postprocess_evidence_maps.py \
        --candidate-pool "$candidate_pool" \
        --annotations "$annotations" \
        --output-dir "$EVIDENCE_MAP_ROOT" \
        --split "$split"
    fi
  fi
done

if enabled "$RUN_AUDIT"; then
  read -r -a audit_splits <<< "$SPLITS"
  run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/audit_atom_anchor_outputs.py \
    --root "$ATOM_ANCHOR_ROOT" \
    --output "$QUALITY_AUDIT" \
    --splits "${audit_splits[@]}"
fi

printf '[rawfc-atom-anchor-abc-v0.1] done: %s\n' "$ATOM_ANCHOR_ROOT"
