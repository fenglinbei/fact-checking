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
DATA_ROOT="${DATA_ROOT:-data/raw/SciFact}"
PROCESSED_ROOT="${PROCESSED_ROOT:-data/processed/SciFact}"
ATOM_ANCHOR_ROOT="${ATOM_ANCHOR_ROOT:-outputs/selectors/scifact_atom_anchor}"
CLAIM_ATOM_ROOT="${CLAIM_ATOM_ROOT:-${ATOM_ANCHOR_ROOT}/01_claim_atoms}"
ATOM_RETRIEVAL_ROOT="${ATOM_RETRIEVAL_ROOT:-${ATOM_ANCHOR_ROOT}/02_atom_retrieval}"
ATOM_UNION_ROOT="${ATOM_UNION_ROOT:-${ATOM_ANCHOR_ROOT}/03_atom_union}"
ABC_OUTPUT_ROOT="${ABC_OUTPUT_ROOT:-outputs/cache/scifact_abc}"
SPLITS="${SPLITS:-train val test}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
ABC_NUM_SHARDS="${ABC_NUM_SHARDS:-$NUM_SHARDS}"
ATOM_RETRIEVAL_NUM_SHARDS="${ATOM_RETRIEVAL_NUM_SHARDS:-$NUM_SHARDS}"
CUDA_DEVICES="${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
NO_PROGRESS="${NO_PROGRESS:-false}"

FORCE_ABC="${FORCE_ABC:-false}"
FORCE_ATOM_RETRIEVAL="${FORCE_ATOM_RETRIEVAL:-false}"
FORCE_ATOM_UNION="${FORCE_ATOM_UNION:-false}"

EMBEDDER_MODEL="${EMBEDDER_MODEL:-/data/models/bge-base-en-v1.5}"
EMBEDDER_DEVICE="${EMBEDDER_DEVICE:-cuda}"
EMBEDDER_MAX_LENGTH="${EMBEDDER_MAX_LENGTH:-256}"
EMBEDDER_BATCH_SIZE="${EMBEDDER_BATCH_SIZE:-64}"
EMBEDDER_PRECISION="${EMBEDDER_PRECISION:-fp32}"

CLAIM_DOC_TOP_K="${CLAIM_DOC_TOP_K:-80}"
ATOM_DOC_TOP_K="${ATOM_DOC_TOP_K:-40}"
UNIVERSE_DOC_TOP_K="${UNIVERSE_DOC_TOP_K:-120}"
BASELINE_TOP_K="${BASELINE_TOP_K:-20}"
PER_ATOM_KEEP="${PER_ATOM_KEEP:-20}"
MERGED_POOL_SIZE="${MERGED_POOL_SIZE:-20}"
SELECTOR_TOP_K="${SELECTOR_TOP_K:-20}"
CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}"
RRF_K="${RRF_K:-60}"
MERGE_MMR_LAMBDA="${MERGE_MMR_LAMBDA:-0.70}"
UNION_MMR_LAMBDA="${UNION_MMR_LAMBDA:-0.70}"
RUN_AUDIT="${RUN_AUDIT:-true}"

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
  case "$1" in
    train) printf '%s\n' "${DATA_ROOT}/claims_train.jsonl" ;;
    val) printf '%s\n' "${DATA_ROOT}/claims_dev.jsonl" ;;
    test) printf '%s\n' "${DATA_ROOT}/claims_test.jsonl" ;;
    *) printf 'Unsupported split=%s\n' "$1" >&2; exit 2 ;;
  esac
}

sample_args=()
if [[ "$SAMPLE_LIMIT" != "0" ]]; then
  sample_args=(--sample-limit "$SAMPLE_LIMIT")
fi

progress_args=()
if [[ "$NO_PROGRESS" == "true" || "$NO_PROGRESS" == "1" || "$NO_PROGRESS" == "True" ]]; then
  progress_args=(--no-progress)
fi

device_for_shard() {
  local shard="$1"
  if [[ "$EMBEDDER_DEVICE" == cuda* ]]; then
    printf 'cuda:%s' "$shard"
  else
    printf '%s' "$EMBEDDER_DEVICE"
  fi
}

validate_shard_devices() {
  local stage="$1"
  local num_shards="$2"
  if [[ "$num_shards" -le 1 || "$EMBEDDER_DEVICE" != cuda* ]]; then
    return 0
  fi
  IFS=',' read -r -a device_list <<< "$CUDA_DEVICES"
  if [[ "${#device_list[@]}" -lt "$num_shards" ]]; then
    printf '[scifact-03] %s NUM_SHARDS=%s but CUDA_VISIBLE_DEVICES exposes only %s device(s): %s\n' \
      "$stage" "$num_shards" "${#device_list[@]}" "$CUDA_DEVICES" >&2
    exit 2
  fi
}

wait_for_shards() {
  local stage="$1"
  shift
  local status=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    printf '[scifact-03] %s shard failed\n' "$stage" >&2
    exit "$status"
  fi
}

printf '[scifact-03] ATOM_ANCHOR_ROOT=%s ABC_OUTPUT_ROOT=%s SPLITS=%s EMBEDDER_MODEL=%s DEVICE=%s CUDA_DEVICES=%s ABC_SHARDS=%s RETRIEVAL_SHARDS=%s FINAL_POOL_SIZE=%s UNION_MMR_LAMBDA=%s\n' \
  "$ATOM_ANCHOR_ROOT" "$ABC_OUTPUT_ROOT" "$SPLITS" "$EMBEDDER_MODEL" "$EMBEDDER_DEVICE" "$CUDA_DEVICES" "$ABC_NUM_SHARDS" "$ATOM_RETRIEVAL_NUM_SHARDS" "$CANDIDATE_TOP_N" "$UNION_MMR_LAMBDA"

for split in $SPLITS; do
  raw_path="$(raw_path_for_split "$split")"
  claim_atoms="${CLAIM_ATOM_ROOT}/claim_atoms_${split}.jsonl"
  latest_cache_marker="${ABC_OUTPUT_ROOT}/latest_${split}_cache_path.txt"
  retrieval_trace="${ATOM_RETRIEVAL_ROOT}/retrieval_trace_${split}.jsonl"
  atom_union_pool="${ATOM_UNION_ROOT}/atom_union_candidate_pool_${split}.jsonl"

  require_path "$raw_path" "${split} SciFact raw split"
  require_path "$claim_atoms" "${split} SciFact claim atoms"
	  require_path "${PROCESSED_ROOT}/scifact_corpus.sqlite" "SciFact corpus SQLite"

	  if [[ "$FORCE_ABC" == "true" || ! -s "$latest_cache_marker" ]]; then
	    if [[ "$ABC_NUM_SHARDS" -gt 1 ]]; then
	      validate_shard_devices "ABC" "$ABC_NUM_SHARDS"
	      pids=()
	      for shard in $(seq 0 $((ABC_NUM_SHARDS - 1))); do
	        shard_device="$(device_for_shard "$shard")"
	        printf '[scifact-03] launch ABC shard=%s/%s device=%s split=%s\n' "$shard" "$ABC_NUM_SHARDS" "$shard_device" "$split"
	        if [[ "$DRY_RUN" == "true" ]]; then
	          run_cmd env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" scripts/phase13_scifact/build_scifact_abc_cache.py \
	            --raw-path "$raw_path" \
	            --claim-atoms-jsonl "$claim_atoms" \
	            --corpus-sqlite "${PROCESSED_ROOT}/scifact_corpus.sqlite" \
	            --output-root "$ABC_OUTPUT_ROOT" \
	            --split "$split" \
	            --embedder-model "$EMBEDDER_MODEL" \
	            --device "$shard_device" \
	            --embedder-max-length "$EMBEDDER_MAX_LENGTH" \
	            --embedder-batch-size "$EMBEDDER_BATCH_SIZE" \
	            --precision "$EMBEDDER_PRECISION" \
	            --claim-doc-top-k "$CLAIM_DOC_TOP_K" \
	            --atom-doc-top-k "$ATOM_DOC_TOP_K" \
	            --universe-doc-top-k "$UNIVERSE_DOC_TOP_K" \
	            --num-shards "$ABC_NUM_SHARDS" \
	            --shard-index "$shard" \
	            "${progress_args[@]}" \
	            "${sample_args[@]}"
	        else
	          env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" scripts/phase13_scifact/build_scifact_abc_cache.py \
	            --raw-path "$raw_path" \
	            --claim-atoms-jsonl "$claim_atoms" \
	            --corpus-sqlite "${PROCESSED_ROOT}/scifact_corpus.sqlite" \
	            --output-root "$ABC_OUTPUT_ROOT" \
	            --split "$split" \
	            --embedder-model "$EMBEDDER_MODEL" \
	            --device "$shard_device" \
	            --embedder-max-length "$EMBEDDER_MAX_LENGTH" \
	            --embedder-batch-size "$EMBEDDER_BATCH_SIZE" \
	            --precision "$EMBEDDER_PRECISION" \
	            --claim-doc-top-k "$CLAIM_DOC_TOP_K" \
	            --atom-doc-top-k "$ATOM_DOC_TOP_K" \
	            --universe-doc-top-k "$UNIVERSE_DOC_TOP_K" \
	            --num-shards "$ABC_NUM_SHARDS" \
	            --shard-index "$shard" \
	            "${progress_args[@]}" \
	            "${sample_args[@]}" &
	          pids+=("$!")
	        fi
	      done
	      if [[ "$DRY_RUN" != "true" ]]; then
	        wait_for_shards "ABC" "${pids[@]}"
	      fi
	      run_cmd "$PYTHON_BIN" scripts/phase13_scifact/build_scifact_abc_cache.py \
	        --raw-path "$raw_path" \
	        --claim-atoms-jsonl "$claim_atoms" \
	        --corpus-sqlite "${PROCESSED_ROOT}/scifact_corpus.sqlite" \
	        --output-root "$ABC_OUTPUT_ROOT" \
	        --split "$split" \
	        --embedder-model "$EMBEDDER_MODEL" \
	        --device "$EMBEDDER_DEVICE" \
	        --embedder-max-length "$EMBEDDER_MAX_LENGTH" \
	        --embedder-batch-size "$EMBEDDER_BATCH_SIZE" \
	        --precision "$EMBEDDER_PRECISION" \
	        --claim-doc-top-k "$CLAIM_DOC_TOP_K" \
	        --atom-doc-top-k "$ATOM_DOC_TOP_K" \
	        --universe-doc-top-k "$UNIVERSE_DOC_TOP_K" \
	        --num-shards "$ABC_NUM_SHARDS" \
	        --merge-shards \
	        "${sample_args[@]}"
	    else
	      run_cmd "$PYTHON_BIN" scripts/phase13_scifact/build_scifact_abc_cache.py \
	        --raw-path "$raw_path" \
	        --claim-atoms-jsonl "$claim_atoms" \
	        --corpus-sqlite "${PROCESSED_ROOT}/scifact_corpus.sqlite" \
	        --output-root "$ABC_OUTPUT_ROOT" \
	        --split "$split" \
	        --embedder-model "$EMBEDDER_MODEL" \
	        --device "$EMBEDDER_DEVICE" \
	        --embedder-max-length "$EMBEDDER_MAX_LENGTH" \
	        --embedder-batch-size "$EMBEDDER_BATCH_SIZE" \
	        --precision "$EMBEDDER_PRECISION" \
	        --claim-doc-top-k "$CLAIM_DOC_TOP_K" \
	        --atom-doc-top-k "$ATOM_DOC_TOP_K" \
	        --universe-doc-top-k "$UNIVERSE_DOC_TOP_K" \
	        "${progress_args[@]}" \
	        "${sample_args[@]}"
	    fi
	  else
	    printf '[scifact-03] reuse ABC cache marker: %s\n' "$latest_cache_marker"
	  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    chunk_cache="${ABC_OUTPUT_ROOT}/DRY_RUN/${split}.pkl"
  else
    chunk_cache="$(< "$latest_cache_marker")"
  fi
  require_path "$chunk_cache" "${split} SciFact ABC chunk cache"

	  if [[ -s "$retrieval_trace" && "$FORCE_ATOM_RETRIEVAL" != "true" ]]; then
	    printf '[scifact-03] reuse atom retrieval: %s\n' "$retrieval_trace"
	  else
	    if [[ "$ATOM_RETRIEVAL_NUM_SHARDS" -gt 1 ]]; then
	      validate_shard_devices "atom retrieval" "$ATOM_RETRIEVAL_NUM_SHARDS"
	      pids=()
	      for shard in $(seq 0 $((ATOM_RETRIEVAL_NUM_SHARDS - 1))); do
	        shard_device="$(device_for_shard "$shard")"
	        printf '[scifact-03] launch atom retrieval shard=%s/%s device=%s split=%s\n' "$shard" "$ATOM_RETRIEVAL_NUM_SHARDS" "$shard_device" "$split"
	        if [[ "$DRY_RUN" == "true" ]]; then
	          run_cmd env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" scripts/phase5_selectors/build/build_atom_conditioned_retrieval.py \
	            --claim-atoms-jsonl "$claim_atoms" \
	            --chunk-cache-path "$chunk_cache" \
	            --split "$split" \
	            --output-dir "$ATOM_RETRIEVAL_ROOT" \
	            --embedder-model "$EMBEDDER_MODEL" \
	            --device "$shard_device" \
	            --embedder-max-length "$EMBEDDER_MAX_LENGTH" \
	            --embedder-batch-size "$EMBEDDER_BATCH_SIZE" \
	            --precision "$EMBEDDER_PRECISION" \
	            --per-atom-keep "$PER_ATOM_KEEP" \
	            --merged-pool-size "$MERGED_POOL_SIZE" \
	            --selector-top-k "$SELECTOR_TOP_K" \
	            --baseline-top-k "$BASELINE_TOP_K" \
	            --rrf-k "$RRF_K" \
	            --merge-mmr-lambda "$MERGE_MMR_LAMBDA" \
	            --oracle-results "" \
	            --num-shards "$ATOM_RETRIEVAL_NUM_SHARDS" \
	            --shard-index "$shard" \
	            "${progress_args[@]}" \
	            "${sample_args[@]}"
	        else
	          env CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" scripts/phase5_selectors/build/build_atom_conditioned_retrieval.py \
	            --claim-atoms-jsonl "$claim_atoms" \
	            --chunk-cache-path "$chunk_cache" \
	            --split "$split" \
	            --output-dir "$ATOM_RETRIEVAL_ROOT" \
	            --embedder-model "$EMBEDDER_MODEL" \
	            --device "$shard_device" \
	            --embedder-max-length "$EMBEDDER_MAX_LENGTH" \
	            --embedder-batch-size "$EMBEDDER_BATCH_SIZE" \
	            --precision "$EMBEDDER_PRECISION" \
	            --per-atom-keep "$PER_ATOM_KEEP" \
	            --merged-pool-size "$MERGED_POOL_SIZE" \
	            --selector-top-k "$SELECTOR_TOP_K" \
	            --baseline-top-k "$BASELINE_TOP_K" \
	            --rrf-k "$RRF_K" \
	            --merge-mmr-lambda "$MERGE_MMR_LAMBDA" \
	            --oracle-results "" \
	            --num-shards "$ATOM_RETRIEVAL_NUM_SHARDS" \
	            --shard-index "$shard" \
	            "${progress_args[@]}" \
	            "${sample_args[@]}" &
	          pids+=("$!")
	        fi
	      done
	      if [[ "$DRY_RUN" != "true" ]]; then
	        wait_for_shards "atom retrieval" "${pids[@]}"
	      fi
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
	        --rrf-k "$RRF_K" \
	        --merge-mmr-lambda "$MERGE_MMR_LAMBDA" \
	        --oracle-results "" \
	        --num-shards "$ATOM_RETRIEVAL_NUM_SHARDS" \
	        --merge-shards \
	        "${sample_args[@]}"
	    else
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
	        --rrf-k "$RRF_K" \
	        --merge-mmr-lambda "$MERGE_MMR_LAMBDA" \
	        --oracle-results "" \
	        "${progress_args[@]}" \
	        "${sample_args[@]}"
	    fi
	  fi

  if [[ -s "$atom_union_pool" && "$FORCE_ATOM_UNION" != "true" ]]; then
    printf '[scifact-03] reuse atom union: %s\n' "$atom_union_pool"
  else
    require_path "${ATOM_RETRIEVAL_ROOT}/baseline_claim_mmr_selected_${split}.jsonl" "${split} baseline claim route"
    require_path "${ATOM_RETRIEVAL_ROOT}/merged_candidate_pool_${split}.jsonl" "${split} atom route merged pool"
    run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/build_atom_retrieval_union.py \
      --baseline-jsonl "${ATOM_RETRIEVAL_ROOT}/baseline_claim_mmr_selected_${split}.jsonl" \
      --atom-pool-jsonl "${ATOM_RETRIEVAL_ROOT}/merged_candidate_pool_${split}.jsonl" \
      --split "$split" \
      --output-dir "$ATOM_UNION_ROOT" \
      --selector-top-k "$SELECTOR_TOP_K" \
      --chunk-cache-path "$chunk_cache" \
      --final-pool-size "$CANDIDATE_TOP_N" \
      --union-mmr-lambda "$UNION_MMR_LAMBDA" \
      --oracle-results "" \
      "${sample_args[@]}"
  fi

  if [[ "$RUN_AUDIT" == "true" ]]; then
    abc_coverage="${chunk_cache%.pkl}_coverage.jsonl"
    require_path "$abc_coverage" "${split} ABC coverage sidecar"
    run_cmd "$PYTHON_BIN" scripts/phase13_scifact/audit_scifact_retrieval.py \
      --raw-path "$raw_path" \
      --baseline-jsonl "${ATOM_RETRIEVAL_ROOT}/baseline_claim_mmr_selected_${split}.jsonl" \
      --atom-pool-jsonl "${ATOM_RETRIEVAL_ROOT}/merged_candidate_pool_${split}.jsonl" \
      --atom-union-jsonl "$atom_union_pool" \
      --abc-coverage-jsonl "$abc_coverage" \
      --output "${ATOM_UNION_ROOT}/retrieval_audit_${split}.json" \
      --split "$split" \
      --top-k "$CANDIDATE_TOP_N"
  fi
done
