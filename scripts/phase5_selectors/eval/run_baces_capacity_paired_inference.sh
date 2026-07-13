#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

REFERENCE_CONTRACT="${REFERENCE_CONTRACT:-configs/validation/baces_native_label_token_reference_v0_1.json}"
CONTRAST_REGISTRY="${CONTRAST_REGISTRY:-configs/validation/baces_capacity_contrast_registry_v0_1.json}"
if [[ ! -f "$REFERENCE_CONTRACT" ]]; then
  printf 'Reference contract does not exist: %s\n' "$REFERENCE_CONTRACT" >&2
  exit 2
fi
if [[ ! -f "$CONTRAST_REGISTRY" ]]; then
  printf 'Contrast registry does not exist: %s\n' "$CONTRAST_REGISTRY" >&2
  exit 2
fi

contract_python_bin="$(jq -er '.native_command[0]' "$REFERENCE_CONTRACT")"
contract_adapter_sha256="$(jq -er '.checkpoint.adapter_sha256' "$REFERENCE_CONTRACT")"
PYTHON_BIN="${PYTHON_BIN:-$contract_python_bin}"
SPLIT="${SPLIT:-val}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/validation_artifacts/baces_capacity_prefix_v0_1/${SPLIT}/atom_anchor_v0_2_fullpool_minmax5_10_best_${contract_adapter_sha256:0:8}_noadjust}"
MATRIX_MANIFEST="${MATRIX_MANIFEST:-$OUTPUT_DIR/materialized/matrix_manifest.json}"
CAPACITY_ANALYSIS_MANIFEST="${CAPACITY_ANALYSIS_MANIFEST:-$OUTPUT_DIR/capacity_analysis/manifest.json}"
PAIRED_OUTPUT_DIR="${PAIRED_OUTPUT_DIR:-$OUTPUT_DIR/capacity_paired_inference}"

BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-20000}"
PERMUTATION_SAMPLES="${PERMUTATION_SAMPLES:-20000}"
STATS_SEED="${STATS_SEED:-20260713}"
STATS_ALPHA="${STATS_ALPHA:-0.05}"
FORCE_STATS="${FORCE_STATS:-false}"
DRY_RUN="${DRY_RUN:-false}"

if ! [[ "$BOOTSTRAP_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  printf 'BOOTSTRAP_SAMPLES must be a positive integer: %s\n' "$BOOTSTRAP_SAMPLES" >&2
  exit 2
fi
if ! [[ "$PERMUTATION_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  printf 'PERMUTATION_SAMPLES must be a positive integer: %s\n' "$PERMUTATION_SAMPLES" >&2
  exit 2
fi
if [[ "$FORCE_STATS" != "true" && "$FORCE_STATS" != "false" ]]; then
  printf 'FORCE_STATS must be true or false: %s\n' "$FORCE_STATS" >&2
  exit 2
fi
if [[ "$DRY_RUN" != "true" && "$DRY_RUN" != "false" ]]; then
  printf 'DRY_RUN must be true or false: %s\n' "$DRY_RUN" >&2
  exit 2
fi
for required in "$MATRIX_MANIFEST" "$CAPACITY_ANALYSIS_MANIFEST"; do
  if [[ ! -f "$required" ]]; then
    printf 'Required capacity artifact does not exist: %s\n' "$required" >&2
    exit 2
  fi
done

cmd=("$PYTHON_BIN" -m sft.capacity_prefix_paired_inference
  --matrix-manifest "$MATRIX_MANIFEST"
  --capacity-analysis-manifest "$CAPACITY_ANALYSIS_MANIFEST"
  --contrast-registry "$CONTRAST_REGISTRY"
  --output-dir "$PAIRED_OUTPUT_DIR"
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"
  --permutation-samples "$PERMUTATION_SAMPLES"
  --seed "$STATS_SEED"
  --alpha "$STATS_ALPHA")
if [[ "$FORCE_STATS" == "true" ]]; then
  cmd+=(--force)
fi

printf '[capacity-paired] matrix=%s\n' "$MATRIX_MANIFEST"
printf '[capacity-paired] analysis=%s registry=%s\n' \
  "$CAPACITY_ANALYSIS_MANIFEST" "$CONTRAST_REGISTRY"
printf '[capacity-paired] output=%s bootstrap=%s permutation=%s seed=%s alpha=%s\n' \
  "$PAIRED_OUTPUT_DIR" "$BOOTSTRAP_SAMPLES" "$PERMUTATION_SAMPLES" \
  "$STATS_SEED" "$STATS_ALPHA"
if [[ "$DRY_RUN" == "true" ]]; then
  printf '+'
  printf ' %q' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
