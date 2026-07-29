#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_NAME="${1:?usage: $0 MODEL_NAME ASSIGNMENT GPU_ID}"
ASSIGNMENT="${2:?usage: $0 MODEL_NAME ASSIGNMENT GPU_ID}"
GPU_ID="${3:?usage: $0 MODEL_NAME ASSIGNMENT GPU_ID}"

case "$MODEL_NAME" in
  qwen3|llama31) ;;
  *)
    echo "MODEL_NAME must be qwen3 or llama31" >&2
    exit 2
    ;;
esac
case "$ASSIGNMENT" in
  a|b) ;;
  *)
    echo "ASSIGNMENT must be a or b" >&2
    exit 2
    ;;
esac
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"
ENTRYPOINT="${ENTRYPOINT:-scripts/sentence_trace_method/cross_verifier_finetune.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/evitrace_cross_verifier_finetune_v1}"
PREPARED_MANIFEST="${PREPARED_MANIFEST:-${OUTPUT_ROOT}/prepared/artifact_manifest.json}"
export PYTHONPATH="${PROJECT_ROOT}/src"

for seed in 20260724 20260725 20260726; do
  "$PYTHON_BIN" "$ENTRYPOINT" infer \
    --prepared-manifest "$PREPARED_MANIFEST" \
    --model-name "$MODEL_NAME" \
    --assignment "$ASSIGNMENT" \
    --seed "$seed" \
    --experiment-root "$OUTPUT_ROOT" \
    --gpu-id "$GPU_ID"
done
