#!/usr/bin/env bash
# Generate per-lambda oracle prompt JSONL files for learned-lambda training.
# Defaults reuse the b3_mmr_topk_sweep_1024 build strategy.
#
# Usage:
#   bash scripts/learned_lambda/run_generate_oracle_prompts.sh
#   SPLIT_NAME=val bash scripts/learned_lambda/run_generate_oracle_prompts.sh
#   REBUILD_PREMMR_CACHE=false bash scripts/learned_lambda/run_generate_oracle_prompts.sh
#   LAMBDA_GRID="0.00,0.25,0.50,0.75,1.00" bash scripts/learned_lambda/run_generate_oracle_prompts.sh
#   CONFIG_OVERRIDES="build.retrieval.chunking.theta=0.6" bash scripts/learned_lambda/run_generate_oracle_prompts.sh
#   TOP_K=16 bash scripts/learned_lambda/run_generate_oracle_prompts.sh
#   PREMMR_CACHE=outputs/cache/pre_mmr/<fingerprint>/train.pkl REBUILD_PREMMR_CACHE=false bash scripts/learned_lambda/run_generate_oracle_prompts.sh
#   CHUNK_MMR_CACHE=outputs/cache/chunk_mmr/<fingerprint>/train.pkl REBUILD_CHUNK_MMR_CACHE=false bash scripts/learned_lambda/run_generate_oracle_prompts.sh
#   PROGRESS=false bash scripts/learned_lambda/run_generate_oracle_prompts.sh
#
# Extra CLI args are forwarded to generate_oracle_prompts.py, for example:
#   bash scripts/learned_lambda/run_generate_oracle_prompts.sh --top-k 14

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

EXPERIMENT="${EXPERIMENT:-b3_mmr_topk_sweep_1024}"
SPLIT_NAME="${SPLIT_NAME:-train}"
TOP_K="${TOP_K:-}"
PREMMR_CACHE="${PREMMR_CACHE:-}"
PREMMR_CACHE_ROOT="${PREMMR_CACHE_ROOT:-outputs/cache/pre_mmr}"
CHUNK_MMR_CACHE="${CHUNK_MMR_CACHE:-}"
CHUNK_MMR_CACHE_ROOT="${CHUNK_MMR_CACHE_ROOT:-outputs/cache/chunk_mmr}"
if [[ -n "${PREMMR_CACHE}" && -z "${REBUILD_PREMMR_CACHE+x}" ]]; then
  REBUILD_PREMMR_CACHE="false"
else
  REBUILD_PREMMR_CACHE="${REBUILD_PREMMR_CACHE:-true}"
fi
if [[ -n "${CHUNK_MMR_CACHE}" && -z "${REBUILD_CHUNK_MMR_CACHE+x}" ]]; then
  REBUILD_CHUNK_MMR_CACHE="false"
else
  REBUILD_CHUNK_MMR_CACHE="${REBUILD_CHUNK_MMR_CACHE:-${REBUILD_PREMMR_CACHE}}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/learned_lambda/prompts}"
LAMBDA_GRID="${LAMBDA_GRID:-0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00}"
CONFIG_OVERRIDES="${CONFIG_OVERRIDES:-}"
PROGRESS="${PROGRESS:-true}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

if [[ -n "${PREMMR_CACHE}" && ! -f "${PREMMR_CACHE}" ]]; then
  echo "[run_generate_oracle_prompts] PreMMR cache not found: ${PREMMR_CACHE}" >&2
  echo "[run_generate_oracle_prompts] Set PREMMR_CACHE=/path/to/{train,val,test}.pkl and rerun." >&2
  exit 1
fi

if [[ -n "${CHUNK_MMR_CACHE}" && ! -f "${CHUNK_MMR_CACHE}" ]]; then
  echo "[run_generate_oracle_prompts] chunk-MMR cache not found: ${CHUNK_MMR_CACHE}" >&2
  echo "[run_generate_oracle_prompts] Set CHUNK_MMR_CACHE=/path/to/{train,val,test}.pkl and rerun." >&2
  exit 1
fi

echo "[run_generate_oracle_prompts] experiment=${EXPERIMENT}"
echo "[run_generate_oracle_prompts] split_name=${SPLIT_NAME}"
echo "[run_generate_oracle_prompts] top_k=${TOP_K:-from_experiment}"
echo "[run_generate_oracle_prompts] premmr_cache=${PREMMR_CACHE:-auto_by_fingerprint}"
echo "[run_generate_oracle_prompts] premmr_cache_root=${PREMMR_CACHE_ROOT}"
echo "[run_generate_oracle_prompts] rebuild_premmr_cache=${REBUILD_PREMMR_CACHE}"
echo "[run_generate_oracle_prompts] chunk_mmr_cache=${CHUNK_MMR_CACHE:-auto_by_fingerprint}"
echo "[run_generate_oracle_prompts] chunk_mmr_cache_root=${CHUNK_MMR_CACHE_ROOT}"
echo "[run_generate_oracle_prompts] rebuild_chunk_mmr_cache=${REBUILD_CHUNK_MMR_CACHE}"
echo "[run_generate_oracle_prompts] output_dir=${OUTPUT_DIR}"
echo "[run_generate_oracle_prompts] lambda_grid=${LAMBDA_GRID}"
echo "[run_generate_oracle_prompts] progress=${PROGRESS}"
echo "[run_generate_oracle_prompts] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if [[ -n "${CONFIG_OVERRIDES}" ]]; then
  echo "[run_generate_oracle_prompts] config_overrides=${CONFIG_OVERRIDES}"
fi

cmd=(
  python scripts/learned_lambda/generate_oracle_prompts.py
  --experiment "${EXPERIMENT}"
  --output-dir "${OUTPUT_DIR}"
  --premmr-cache-root "${PREMMR_CACHE_ROOT}"
  --chunk-mmr-cache-root "${CHUNK_MMR_CACHE_ROOT}"
  --lambda-grid "${LAMBDA_GRID}"
  --split-name "${SPLIT_NAME}"
)

if [[ -n "${TOP_K}" ]]; then
  cmd+=(--top-k "${TOP_K}")
fi

if [[ -n "${PREMMR_CACHE}" ]]; then
  cmd+=(--premmr-cache "${PREMMR_CACHE}")
fi

if [[ -n "${CHUNK_MMR_CACHE}" ]]; then
  cmd+=(--chunk-mmr-cache "${CHUNK_MMR_CACHE}")
fi

if [[ "${REBUILD_PREMMR_CACHE}" == "true" ]]; then
  cmd+=(--rebuild-premmr-cache)
fi

if [[ "${REBUILD_CHUNK_MMR_CACHE}" == "true" ]]; then
  cmd+=(--rebuild-chunk-mmr-cache)
fi

if [[ "${PROGRESS}" == "false" ]]; then
  cmd+=(--no-progress)
fi

if [[ -n "${CONFIG_OVERRIDES}" ]]; then
  # Split CONFIG_OVERRIDES on shell whitespace so multiple Hydra overrides can be passed.
  # Example: CONFIG_OVERRIDES="build.retrieval.chunking.theta=0.6 build.retrieval.top_k=12"
  # shellcheck disable=SC2206
  overrides=(${CONFIG_OVERRIDES})
  cmd+=(--config-overrides "${overrides[@]}")
fi

cmd+=("$@")

"${cmd[@]}"
