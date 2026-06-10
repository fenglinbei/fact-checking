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
CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
TOP_K="${TOP_K:-10}"
MIN_TOP_K="${MIN_TOP_K:-5}"
MAX_TOP_K="${MAX_TOP_K:-10}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
MAX_STEPS="${MAX_STEPS:-5}"
RUN_TEACHER="${RUN_TEACHER:-true}"
MOCK_QUESTIONS="${MOCK_QUESTIONS:-false}"
MOCK_EVIDENCE_MAPS="${MOCK_EVIDENCE_MAPS:-false}"
REUSE_QD_CACHE_ROOT="${REUSE_QD_CACHE_ROOT:-outputs/selectors/question_decomp_retrieval/rawfc_question_cache}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://api.deepseek.com}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-flash}"
TEACHER_API_KEY_ENV="${TEACHER_API_KEY_ENV:-DEEPSEEK_API_KEY}"
CONCURRENCY="${CONCURRENCY:-64}"
REQUESTS_PER_MINUTE="${REQUESTS_PER_MINUTE:-60000}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
THINKING_TYPE="${THINKING_TYPE:-disabled}"
PROMPT_VERSION="${PROMPT_VERSION:-evidence_map_v0_6b}"
MAX_EVIDENCE_CHARS="${MAX_EVIDENCE_CHARS:-700}"

if [[ -z "$CHUNK_MMR_FINGERPRINT" ]]; then
  cat >&2 <<'EOF'
CHUNK_MMR_FINGERPRINT is required for RAWFC sentence-source rebuild.
It must point to an existing sentence chunk cache under outputs/cache/chunk_mmr/<fingerprint>/.
The known RAWFC semantic fingerprint 3b94476fd08e must not be used.
EOF
  exit 2
fi

if [[ "$CHUNK_MMR_FINGERPRINT" == "3b94476fd08e" ]]; then
  printf 'Refusing RAWFC semantic chunk fingerprint: %s\n' "$CHUNK_MMR_FINGERPRINT" >&2
  exit 2
fi

for split in train val test; do
  cache_path="outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/${split}.pkl"
  if [[ ! -f "$cache_path" ]]; then
    printf 'Missing sentence chunk cache: %s\n' "$cache_path" >&2
    exit 2
  fi
done

SOURCE_BASE="${OUTPUT_ROOT}/_raw_sources/rawfc/sentence_rule_step_adaptive5_10"
QD_DIR="${SOURCE_BASE}/qd_questions"
QD_CACHE_DIR="${SOURCE_BASE}/qd_question_cache"
RETRIEVAL_ROOT="${SOURCE_BASE}/qd_retrieval"
UNION_ROOT="${SOURCE_BASE}/qd_union"
MAP_ROOT="${SOURCE_BASE}/evidence_map"
GRAPH_ROOT="${SOURCE_BASE}/graph"
mkdir -p "$SOURCE_BASE" "$QD_DIR" "$QD_CACHE_DIR" "$RETRIEVAL_ROOT" "$UNION_ROOT" "$MAP_ROOT" "$GRAPH_ROOT"

if [[ -d "$REUSE_QD_CACHE_ROOT" ]]; then
  cp -a "${REUSE_QD_CACHE_ROOT}/." "$QD_CACHE_DIR/"
fi

sample_args=()
if [[ "$SAMPLE_LIMIT" != "0" ]]; then
  sample_args=(--sample-limit "$SAMPLE_LIMIT")
fi
question_args=()
if [[ "$MOCK_QUESTIONS" == "true" ]]; then
  question_args=(--mock-questions)
fi

for split in train val test; do
  raw_path="data/raw/RAWFC/${split}.json"
  if [[ "$split" == "val" ]]; then
    raw_path="data/raw/RAWFC/val.json"
  fi
  qd_dir="${QD_DIR}_${split}"
  "$PYTHON_BIN" scripts/phase5_selectors/build/generate_question_decomp_cache.py \
    --input-mode raw_split \
    --raw-path "$raw_path" \
    --dataset rawfc \
    --label-schema rawfc3 \
    --output-dir "$qd_dir" \
    --split "$split" \
    --expected-chunk-mmr-fingerprint "$CHUNK_MMR_FINGERPRINT" \
    --question-cache-dir "$QD_CACHE_DIR" \
    --question-model "$TEACHER_MODEL" \
    --question-api-key-env "$TEACHER_API_KEY_ENV" \
    --api-concurrency 1 \
    "${question_args[@]}" \
    "${sample_args[@]}"

  retrieval_dir="${RETRIEVAL_ROOT}_${split}"
  "$PYTHON_BIN" scripts/phase5_selectors/build/build_question_decomp_retrieval.py \
    --questions-jsonl "$qd_dir/questions_${split}.jsonl" \
    --chunk-cache-path "outputs/cache/chunk_mmr/${CHUNK_MMR_FINGERPRINT}/${split}.pkl" \
    --output-dir "$retrieval_dir" \
    --split "$split" \
    --selector-top-k "$TOP_K" \
    "${sample_args[@]}"

  union_dir="${UNION_ROOT}_${split}"
  "$PYTHON_BIN" scripts/phase5_selectors/build/build_question_decomp_union.py \
    --baseline-jsonl "$retrieval_dir/baseline_claim_mmr_selected_${split}.jsonl" \
    --qd-pool-jsonl "$retrieval_dir/merged_candidate_pool_${split}.jsonl" \
    --output-dir "$union_dir" \
    --split "$split" \
    --selector-top-k "$TOP_K" \
    "${sample_args[@]}"

  map_dir="${MAP_ROOT}_${split}"
  "$PYTHON_BIN" scripts/phase5_selectors/build/prepare_evidence_map_candidate_pool.py \
    --input-candidate-file "$union_dir/union_candidate_pool_${split}.jsonl" \
    --output-dir "$map_dir" \
    --split "$split" \
    --candidate-source qd_union \
    --candidate-top-n "$CANDIDATE_TOP_N" \
    "${sample_args[@]}"

  if [[ "$MOCK_EVIDENCE_MAPS" == "true" ]]; then
    "$PYTHON_BIN" scripts/phase5_selectors/build/annotate_evidence_maps_deepseek.py \
      --candidate-pool "$map_dir/evidence_map_candidate_pool_${split}.jsonl" \
      --output-dir "$map_dir" \
      --split "$split" \
      --prompt-version "$PROMPT_VERSION" \
      --max-evidence-chars "$MAX_EVIDENCE_CHARS" \
      --model "mock-evidence-map" \
      --thinking-type "$THINKING_TYPE" \
      --mock-maps \
      "${sample_args[@]}"
  elif [[ "$RUN_TEACHER" != "true" ]]; then
    printf 'Set RUN_TEACHER=true or MOCK_EVIDENCE_MAPS=true for evidence-map construction.\n' >&2
    exit 2
  elif [[ ! -s "$map_dir/deepseek_evidence_map_annotations_${split}.jsonl" ]]; then
    "$PYTHON_BIN" scripts/phase5_selectors/build/annotate_evidence_maps_deepseek.py \
      --candidate-pool "$map_dir/evidence_map_candidate_pool_${split}.jsonl" \
      --output-dir "$map_dir" \
      --split "$split" \
      --prompt-version "$PROMPT_VERSION" \
      --max-evidence-chars "$MAX_EVIDENCE_CHARS" \
      --base-url "$TEACHER_BASE_URL" \
      --model "$TEACHER_MODEL" \
      --api-key-env "$TEACHER_API_KEY_ENV" \
      --concurrency "$CONCURRENCY" \
      --requests-per-minute "$REQUESTS_PER_MINUTE" \
      --max-tokens "$MAX_TOKENS" \
      --thinking-type "$THINKING_TYPE" \
      "${sample_args[@]}"
  fi

  "$PYTHON_BIN" scripts/phase5_selectors/build/postprocess_evidence_maps.py \
    --candidate-pool "$map_dir/evidence_map_candidate_pool_${split}.jsonl" \
    --annotations "$map_dir/deepseek_evidence_map_annotations_${split}.jsonl" \
    --output-dir "$map_dir" \
    --split "$split" \
    "${sample_args[@]}"

  "$PYTHON_BIN" scripts/phase5_selectors/eval/eval_evidence_map_selector_v0_5a.py \
    --candidate-features "$map_dir/candidate_evidence_map_features_${split}.jsonl" \
    --output-dir "$map_dir" \
    --split "$split" \
    --top-k "$TOP_K" \
    "${sample_args[@]}"

  graph_dir="${GRAPH_ROOT}_${split}"
  "$PYTHON_BIN" scripts/phase5_selectors/build/build_evidence_chain_graph_v0_6c.py \
    --input "$map_dir/candidate_evidence_map_features_${split}.jsonl" \
    --output-dir "$graph_dir" \
    --split "$split" \
    --candidate-top-n "$CANDIDATE_TOP_N" \
    --min-top-k "$MIN_TOP_K" \
    --max-top-k "$MAX_TOP_K" \
    --chunk-mmr-fingerprint "$CHUNK_MMR_FINGERPRINT" \
    "${sample_args[@]}"
done

printf 'RAWFC sentence source root: %s\n' "$GRAPH_ROOT"
