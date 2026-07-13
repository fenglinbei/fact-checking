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
if [[ ! -f "$REFERENCE_CONTRACT" ]]; then
  printf 'Reference contract does not exist: %s\n' "$REFERENCE_CONTRACT" >&2
  exit 2
fi

# Keep the checkpoint and native artifacts under one source of truth. The
# equivalence reference was produced by the v0.2 fullpool/minmax verifier.
contract_python_bin="$(jq -er '.native_command[0]' "$REFERENCE_CONTRACT")"
contract_run_dir="$(jq -er '.checkpoint.run_dir' "$REFERENCE_CONTRACT")"
contract_checkpoint="$(jq -er '.checkpoint.checkpoint' "$REFERENCE_CONTRACT")"
contract_adapter_sha256="$(jq -er '.checkpoint.adapter_sha256' "$REFERENCE_CONTRACT")"
contract_config="$(jq -er '.artifacts.inference_config.path' "$REFERENCE_CONTRACT")"
contract_gate_predictions="$(jq -er '.artifacts.predictions.path' "$REFERENCE_CONTRACT")"
contract_gate_metrics="$(jq -er '.artifacts.metrics.path' "$REFERENCE_CONTRACT")"
contract_gate_build="$(jq -er '.artifacts.build.path' "$REFERENCE_CONTRACT")"
PYTHON_BIN="${PYTHON_BIN:-$contract_python_bin}"

MATRIX_MANIFEST="${MATRIX_MANIFEST:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1/08_baces_factorial_prompt_feasible_v0_2/val/manifest.json}"
BUILD_ROOT="${BUILD_ROOT:-outputs/sentence_trace_method/liar_raw__ministral3_8b__baces_factorial_prompt_feasible_v0_2__val}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/validation_artifacts/baces_factorial_prompt_feasible_v0_2/val/atom_anchor_v0_2_fullpool_minmax5_10_best_${contract_adapter_sha256:0:8}_noadjust}"

RUN_DIR="${RUN_DIR:-$contract_run_dir}"
CHECKPOINT="${CHECKPOINT:-$contract_checkpoint}"
EXPECTED_ADAPTER_SHA256="${EXPECTED_ADAPTER_SHA256:-$contract_adapter_sha256}"

GATE_CELL="${GATE_CELL:-baces_exact__ordinal_replay_minmax5_10}"
CONFIG="${CONFIG:-$contract_config}"
GATE_PREDICTIONS="${GATE_PREDICTIONS:-$contract_gate_predictions}"
GATE_METRICS="${GATE_METRICS:-$contract_gate_metrics}"
GATE_BUILD="${GATE_BUILD:-$contract_gate_build}"

PHASES="${PHASES:-prepare,infer,fanout,stats}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"
FORCE_INFER="${FORCE_INFER:-false}"
FORCE_FANOUT="${FORCE_FANOUT:-false}"
FORCE_STATS="${FORCE_STATS:-false}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-20000}"
PERMUTATION_SAMPLES="${PERMUTATION_SAMPLES:-20000}"
STATS_SEED="${STATS_SEED:-20260713}"
STATS_ALPHA="${STATS_ALPHA:-0.05}"
DRY_RUN="${DRY_RUN:-false}"

if [[ -z "${ACCELERATE_BIN:-}" ]]; then
  python_dir="$(dirname "$PYTHON_BIN")"
  if [[ -x "${python_dir}/accelerate" ]]; then
    ACCELERATE_BIN="${python_dir}/accelerate"
  else
    ACCELERATE_BIN="accelerate"
  fi
fi

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

phase_enabled() {
  local needle="$1"
  [[ ",${PHASES}," == *",${needle},"* ]]
}

if ! [[ "$EVAL_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  printf 'EVAL_NPROC_PER_NODE must be a positive integer: %s\n' "$EVAL_NPROC_PER_NODE" >&2
  exit 2
fi
if ! [[ "$BOOTSTRAP_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  printf 'BOOTSTRAP_SAMPLES must be a positive integer: %s\n' "$BOOTSTRAP_SAMPLES" >&2
  exit 2
fi
if ! [[ "$PERMUTATION_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  printf 'PERMUTATION_SAMPLES must be a positive integer: %s\n' "$PERMUTATION_SAMPLES" >&2
  exit 2
fi

printf '[dedup-matrix] verifier_run=%s checkpoint=%s expected_adapter_sha256=%s\n' \
  "$RUN_DIR" "$CHECKPOINT" "$EXPECTED_ADAPTER_SHA256"
printf '[dedup-matrix] reference_contract=%s\n' "$REFERENCE_CONTRACT"
printf '[dedup-matrix] matrix=%s output=%s phases=%s nproc=%s batch_per_device=%s\n' \
  "$MATRIX_MANIFEST" "$OUTPUT_DIR" "$PHASES" "$EVAL_NPROC_PER_NODE" "$PER_DEVICE_EVAL_BATCH_SIZE"

if phase_enabled prepare; then
  prepare_cmd=("$PYTHON_BIN" -m sft.label_token_matrix_infer prepare
    --matrix-manifest "$MATRIX_MANIFEST"
    --build-root "$BUILD_ROOT"
    --output-dir "$OUTPUT_DIR"
    --split val
    --label-prefix 'Label:')
  if [[ "$FORCE_PREPARE" == "true" ]]; then
    prepare_cmd+=(--force-prepare)
  fi
  run_cmd "${prepare_cmd[@]}"
fi

if phase_enabled infer; then
  infer_args=(-m sft.label_token_matrix_infer infer
    --output-dir "$OUTPUT_DIR"
    --run-dir "$RUN_DIR"
    --checkpoint "$CHECKPOINT"
    --config "$CONFIG"
    --split val
    --expected-world-size "$EVAL_NPROC_PER_NODE"
    --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --expected-adapter-sha256 "$EXPECTED_ADAPTER_SHA256")
  if [[ "$FORCE_INFER" == "true" ]]; then
    infer_args+=(--force-infer)
  fi
  if [[ "$EVAL_NPROC_PER_NODE" -gt 1 ]]; then
    infer_cmd=("$ACCELERATE_BIN" launch
      --multi_gpu
      --num_processes "$EVAL_NPROC_PER_NODE"
      --num_machines "$NUM_MACHINES"
      --mixed_precision "$MIXED_PRECISION"
      "${infer_args[@]}")
  else
    infer_cmd=("$PYTHON_BIN" "${infer_args[@]}")
  fi
  run_cmd "${infer_cmd[@]}"
fi

if phase_enabled fanout; then
  fanout_cmd=("$PYTHON_BIN" -m sft.label_token_matrix_infer fanout
    --output-dir "$OUTPUT_DIR"
    --equivalence-gate-cell "$GATE_CELL"
    --equivalence-gate-predictions "$GATE_PREDICTIONS"
    --equivalence-gate-metrics "$GATE_METRICS"
    --equivalence-gate-build "$GATE_BUILD"
    --equivalence-gate-expected-adapter-sha256 "$EXPECTED_ADAPTER_SHA256"
    --equivalence-gate-reference-contract "$REFERENCE_CONTRACT")
  if [[ "$FORCE_FANOUT" == "true" ]]; then
    fanout_cmd+=(--force-fanout)
  fi
  run_cmd "${fanout_cmd[@]}"
fi

if phase_enabled stats; then
  stats_cmd=("$PYTHON_BIN" -m sft.paired_factorial_inference
    --matrix-manifest "$OUTPUT_DIR/materialized/matrix_manifest.json"
    --output-dir "$OUTPUT_DIR/paired_inference"
    --bootstrap-samples "$BOOTSTRAP_SAMPLES"
    --permutation-samples "$PERMUTATION_SAMPLES"
    --seed "$STATS_SEED"
    --alpha "$STATS_ALPHA")
  if [[ "$FORCE_STATS" == "true" ]]; then
    stats_cmd+=(--force)
  fi
  run_cmd "${stats_cmd[@]}"
fi
