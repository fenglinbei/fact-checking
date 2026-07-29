#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/evitrace_cross_verifier_finetune_v1}"
PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WORKERS=(
  evitrace_infer_qwen_a
  evitrace_infer_qwen_b
  evitrace_infer_llama_a
  evitrace_infer_llama_b
)

while true; do
  alive=0
  for session in "${WORKERS[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      alive=1
    fi
  done
  if (( alive == 0 )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

runtime_count=0
for runtime in "$OUTPUT_ROOT"/inference/*__assignment_*__seed_*/runtime_manifest.json; do
  if [[ ! -f "$runtime" ]]; then
    continue
  fi
  jq -e '
    .complete == true
    and .counts.logical_results == 23892
    and .counts.unique_logits == 19028
  ' "$runtime" >/dev/null
  runtime_count=$((runtime_count + 1))
done
if (( runtime_count != 12 )); then
  echo "Expected 12 complete formal runtime manifests, found ${runtime_count}." >&2
  exit 1
fi

MODE=analyze \
OUTPUT_ROOT="$OUTPUT_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
bash scripts/sentence_trace_method/run_cross_verifier_finetune.sh
