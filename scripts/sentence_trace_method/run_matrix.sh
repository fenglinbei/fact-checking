#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATASETS="${DATASETS:-liar_raw,rawfc}"
MODELS="${MODELS:-llama31_8b,qwen3_4b_2507}"

IFS=',' read -r -a dataset_array <<< "$DATASETS"
IFS=',' read -r -a model_array <<< "$MODELS"

for dataset in "${dataset_array[@]}"; do
  dataset="${dataset// /}"
  [[ -z "$dataset" ]] && continue
  for model in "${model_array[@]}"; do
    model="${model// /}"
    [[ -z "$model" ]] && continue
    printf '\n== sentence_trace_method DATASET=%s MODEL=%s MODE=%s ==\n' "$dataset" "$model" "${MODE:-full}"
    DATASET="$dataset" MODEL="$model" bash scripts/sentence_trace_method/run_one.sh
  done
done
