#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

MATRIX_ROOT="${MATRIX_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/no_map_structure_fixed5_matrix_val}"
MATRIX_MANIFEST="${MATRIX_MANIFEST:-${MATRIX_ROOT}/manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/no_map_structure_fixed5_matched_verifier_crossover_step800_val}"
N_RUN_DIR="${N_RUN_DIR:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_no_map_fullpool_minmax5_10_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/train}"
S_RUN_DIR="${S_RUN_DIR:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_fullpool_minmax5_10_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/train}"
S_CONFIG="${S_CONFIG:-${S_RUN_DIR}/config.resolved.yaml}"
N_CAP_MANIFEST="${N_CAP_MANIFEST:-outputs/sentence_trace_method/queues/no_map_fixed800_tail_20260717/no_map_checkpoint800_capped.json}"
CONTRACT_HELPER="${CONTRACT_HELPER:-scripts/phase5_selectors/eval/validate_no_map_fixed5_resume_vs_only.py}"
SUMMARY_HELPER="${SUMMARY_HELPER:-scripts/phase5_selectors/analyze/summarize_no_map_structure_fixed5_crossover.py}"

# Frozen production contracts from the interrupted 2026-07-17 diagnostic.
EXPECTED_MATRIX_SHA256="${EXPECTED_MATRIX_SHA256:-f76148d6fe03ca304c7f9c870ae4dae4c862ef16f2536947c6868f94e0278193}"
EXPECTED_N_ADAPTER_SHA256="${EXPECTED_N_ADAPTER_SHA256:-86847100d511613a7929ff6f520b745e2a152472d4bf6d8062aac15e0ecf4c91}"
EXPECTED_S_ADAPTER_SHA256="${EXPECTED_S_ADAPTER_SHA256:-7b7512cd8f5a37d7087be935c3d768db04a29dd3bd479131bd1c5c7681b9374a}"
EXPECTED_S_COMPLETION_SHA256="${EXPECTED_S_COMPLETION_SHA256:-63321066528372a7d6ed585c83c5c9d69cb68b7db0d6e0426f93f0e8363e957f}"
EXPECTED_N_CAP_MANIFEST_SHA256="${EXPECTED_N_CAP_MANIFEST_SHA256:-cfedc72e9a0c072bcae569c4416fe8cd35f8a96fc9a730b940559cc36032528b}"
EXPECTED_N_INPUT_SHA256="${EXPECTED_N_INPUT_SHA256:-fbd237e7627d96c4208c9a06d29787fad677df6df42f3ef19587b909e6babaab}"
EXPECTED_S_INPUT_SHA256="${EXPECTED_S_INPUT_SHA256:-ad01dc04c2dbb0c0f6ef649e4efb52a69d82852c7bdd6fb5e4a3e3f1c5a3ca4c}"
EXPECTED_N_RAW_MANIFEST_SHA256="${EXPECTED_N_RAW_MANIFEST_SHA256:-2a7806a46679960dc3af656c09891eae3f370e8cef85c9d6e43b61ede8eba6c7}"
EXPECTED_N_MATERIALIZED_MANIFEST_SHA256="${EXPECTED_N_MATERIALIZED_MANIFEST_SHA256:-2b24d8497f89f5656ae6ddcc5fbfe1e4dbc1849b3eb462ab582e1cadf95041bb}"

PYTHON_BIN="${MATRIX_PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/accelerate}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
MAIN_PROCESS_PORT_S="${MAIN_PROCESS_PORT_S:-29688}"
HAMI_SHARED_CACHE_ROOT="${HAMI_SHARED_CACHE_ROOT:-/tmp/lzj_no_map_fixed5_vs_only_hami_$$}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"
FORCE_INFER="${FORCE_INFER:-false}"
FORCE_FANOUT="${FORCE_FANOUT:-false}"

die() {
  printf '[no-map-fixed5-vs-only] ERROR: %s\n' "$*" >&2
  exit 2
}

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

[[ "$DRY_RUN" == "true" || "$DRY_RUN" == "false" ]] || die "DRY_RUN must be true or false"
[[ "$EVAL_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "EVAL_NPROC_PER_NODE must be positive"
[[ "$FORCE_PREPARE" == "false" && "$FORCE_INFER" == "false" && "$FORCE_FANOUT" == "false" ]] || \
  die "force replacement is forbidden in the strict resume-only path"

N_OUTPUT_DIR="${OUTPUT_ROOT}/verifier_n"
S_OUTPUT_DIR="${OUTPUT_ROOT}/verifier_s"

contract_args=(
  --matrix-root "$MATRIX_ROOT"
  --output-root "$OUTPUT_ROOT"
  --n-run-dir "$N_RUN_DIR"
  --s-run-dir "$S_RUN_DIR"
  --n-cap-manifest "$N_CAP_MANIFEST"
  --expected-matrix-sha256 "$EXPECTED_MATRIX_SHA256"
  --expected-n-adapter-sha256 "$EXPECTED_N_ADAPTER_SHA256"
  --expected-s-adapter-sha256 "$EXPECTED_S_ADAPTER_SHA256"
  --expected-s-completion-sha256 "$EXPECTED_S_COMPLETION_SHA256"
  --expected-n-cap-manifest-sha256 "$EXPECTED_N_CAP_MANIFEST_SHA256"
  --expected-n-input-sha256 "$EXPECTED_N_INPUT_SHA256"
  --expected-s-input-sha256 "$EXPECTED_S_INPUT_SHA256"
  --expected-n-raw-manifest-sha256 "$EXPECTED_N_RAW_MANIFEST_SHA256"
  --expected-n-materialized-manifest-sha256 "$EXPECTED_N_MATERIALIZED_MANIFEST_SHA256"
)

printf '[no-map-fixed5-vs-only] scope=val checkpoint=checkpoint-800 events=1234 K=5 cells=N_fixed5,S_fixed5\n'
printf '[no-map-fixed5-vs-only] reuse=verifier_n gpu_infer=verifier_s_only dry_run=%s\n' "$DRY_RUN"

# CPU-only and idempotent: this validates/reuses the already prepared V_S input.
run_cmd "$PYTHON_BIN" -m sft.label_token_matrix_infer prepare \
  --matrix-manifest "$MATRIX_MANIFEST" --build-root "$MATRIX_ROOT" --output-dir "$S_OUTPUT_DIR" \
  --split val --label-prefix 'Label:'

# This executes even in DRY_RUN mode and gates all later GPU work.
"$PYTHON_BIN" "$CONTRACT_HELPER" "${contract_args[@]}"

infer_args=(-m sft.hami_cuda_bootstrap infer --output-dir "$S_OUTPUT_DIR" --run-dir "$S_RUN_DIR" \
  --checkpoint checkpoint-800 --config "$S_CONFIG" --split val --expected-world-size "$EVAL_NPROC_PER_NODE" \
  --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE" --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
  --expected-adapter-sha256 "$EXPECTED_S_ADAPTER_SHA256")

if (( EVAL_NPROC_PER_NODE > 1 )); then
  run_cmd env "CUDA_DEVICE_MEMORY_SHARED_CACHE=${HAMI_SHARED_CACHE_ROOT}/verifier_s.cache" \
    SFT_HAMI_BOOTSTRAP_TARGET_MODULE=sft.label_token_matrix_infer \
    "$ACCELERATE_BIN" launch --multi_gpu --num_processes "$EVAL_NPROC_PER_NODE" --num_machines 1 \
    --mixed_precision bf16 --main_process_port "$MAIN_PROCESS_PORT_S" "${infer_args[@]}"
else
  run_cmd env LOCAL_RANK=0 "CUDA_DEVICE_MEMORY_SHARED_CACHE=${HAMI_SHARED_CACHE_ROOT}/verifier_s.cache" \
    SFT_HAMI_BOOTSTRAP_TARGET_MODULE=sft.label_token_matrix_infer "$PYTHON_BIN" "${infer_args[@]}"
fi

run_cmd "$PYTHON_BIN" -m sft.label_token_matrix_infer fanout \
  --output-dir "$S_OUTPUT_DIR" --unsafe-skip-equivalence-gate

if [[ "$DRY_RUN" == "false" ]]; then
  "$PYTHON_BIN" "$CONTRACT_HELPER" "${contract_args[@]}" --require-s-complete
fi

run_cmd "$PYTHON_BIN" "$SUMMARY_HELPER" \
  --verifier-n-dir "$N_OUTPUT_DIR" --verifier-s-dir "$S_OUTPUT_DIR" --matrix-dir "$MATRIX_ROOT" \
  --output-json "${OUTPUT_ROOT}/summary.json" --output-md "${OUTPUT_ROOT}/summary.md"

printf '[no-map-fixed5-vs-only] complete output=%s\n' "$OUTPUT_ROOT"
