#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

EXTRA_ARGS=()
for arg in "$@"; do
  if [[ "${arg}" == *=* && "${arg}" != --*=* ]]; then
    export "${arg}"
  else
    EXTRA_ARGS+=("${arg}")
  fi
done

SPLIT="${SPLIT:-val}"
if [[ -z "${ORACLE_RESULTS:-}" ]]; then
  if [[ "${SPLIT}" == "train" ]]; then
    ORACLE_RESULTS="outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl"
  else
    ORACLE_RESULTS="outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl"
  fi
fi

if [[ -z "${QD_UNION_POOL_JSONL:-}" ]]; then
  if [[ "${SPLIT}" == "train" ]]; then
    QD_UNION_POOL_JSONL="outputs/selectors/question_decomp_retrieval/qwen_v0_train/union_candidate_pool_train.jsonl"
  else
    QD_UNION_POOL_JSONL="outputs/selectors/question_decomp_retrieval/qwen_v0_val/union_candidate_pool_val.jsonl"
  fi
fi

CHUNK_CACHE_PATH="${CHUNK_CACHE_PATH:-outputs/cache/chunk_mmr/432dfc970e75/${SPLIT}.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/roundtable_evidence_map/qwen_qd_union_vs_original_v0_${SPLIT}}"
TOP_K="${TOP_K:-5}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.72}"
MIN_FACTIONS="${MIN_FACTIONS:-2}"
MAX_FACTIONS="${MAX_FACTIONS:-6}"
NO_PROGRESS="${NO_PROGRESS:-false}"

SAMPLE_LIMIT_ARGS=()
if [[ -n "${SAMPLE_LIMIT:-}" ]]; then
  SAMPLE_LIMIT_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

NO_PROGRESS_ARGS=()
if [[ "${NO_PROGRESS}" == "1" || "${NO_PROGRESS}" == "true" || "${NO_PROGRESS}" == "True" ]]; then
  NO_PROGRESS_ARGS=(--no-progress)
fi

STANCE_ARGS=()
if [[ -n "${STANCE_SCORES:-}" ]]; then
  read -r -a STANCE_PATHS <<< "${STANCE_SCORES}"
  STANCE_ARGS=(--stance-scores "${STANCE_PATHS[@]}")
fi

ASPECT_ARGS=()
if [[ -n "${ASPECT_ALIGNMENT:-}" ]]; then
  read -r -a ASPECT_PATHS <<< "${ASPECT_ALIGNMENT}"
  ASPECT_ARGS=(--aspect-alignment "${ASPECT_PATHS[@]}")
fi

echo "[roundtable] split          : ${SPLIT}"
echo "[roundtable] oracle results : ${ORACLE_RESULTS}"
echo "[roundtable] qd union pool  : ${QD_UNION_POOL_JSONL}"
echo "[roundtable] chunk cache    : ${CHUNK_CACHE_PATH}"
echo "[roundtable] output dir     : ${OUTPUT_DIR}"
echo "[roundtable] top k          : ${TOP_K}"
echo "[roundtable] sample limit   : ${SAMPLE_LIMIT:-none}"

PYTHONPATH=src python scripts/phase5_selectors/eval/analyze_roundtable_evidence_map.py \
  --split "${SPLIT}" \
  --oracle-results "${ORACLE_RESULTS}" \
  --qd-union-pool-jsonl "${QD_UNION_POOL_JSONL}" \
  --chunk-cache-path "${CHUNK_CACHE_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --top-k "${TOP_K}" \
  --similarity-threshold "${SIMILARITY_THRESHOLD}" \
  --min-factions "${MIN_FACTIONS}" \
  --max-factions "${MAX_FACTIONS}" \
  "${SAMPLE_LIMIT_ARGS[@]}" \
  "${NO_PROGRESS_ARGS[@]}" \
  "${STANCE_ARGS[@]}" \
  "${ASPECT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
