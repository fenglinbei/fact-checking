#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${MANIFEST:-configs/validation/structure_only_clean_results_audit_v0_1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/clean_results_audit}"

exec "$PYTHON_BIN" scripts/phase5_selectors/analyze/summarize_structure_only_clean_results.py \
  --repo-root "$ROOT_DIR" \
  --manifest "$MANIFEST" \
  --output-json "$OUTPUT_ROOT/summary.json" \
  --output-md "$OUTPUT_ROOT/summary.md"
