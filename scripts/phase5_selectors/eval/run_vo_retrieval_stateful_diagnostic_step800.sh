#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

MATRIX_MANIFEST="${MATRIX_MANIFEST:-configs/validation/retrieval_stateful_matched_verifier_crossover_step800_v0_1.json}"
MATRIX_MANIFEST_EXPECTED_SHA256="${MATRIX_MANIFEST_EXPECTED_SHA256:-1a6bb748f0006873ba9599de629d71baa19026413350b9f5f36767088868c13e}"
BUILD_ROOT="${BUILD_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/frozen_matrix_val}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/vo_retrieval_stateful_diagnostic_step800_val}"

O_RUN_DIR="${O_RUN_DIR:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_one_shot_fullpool_minmax5_10_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/train}"
O_CONFIG="${O_CONFIG:-${O_RUN_DIR}/config.resolved.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoint-800}"
FROZEN_O_ADAPTER_SHA256="24e661e8efec049f19e4427a4488de57ce4dc7aec97e412315029412f8779aa3"
O_EXPECTED_ADAPTER_SHA256="${O_EXPECTED_ADAPTER_SHA256:-${FROZEN_O_ADAPTER_SHA256}}"

PHASES="${PHASES:-prepare,infer,fanout,summarize}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"
FORCE_INFER="${FORCE_INFER:-false}"
FORCE_FANOUT="${FORCE_FANOUT:-false}"
DRY_RUN="${DRY_RUN:-false}"
PYTHON_BIN="${MATRIX_PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/accelerate}"
HAMI_SHARED_CACHE_ROOT="${HAMI_SHARED_CACHE_ROOT:-/tmp/lzj_vo_rs_diagnostic_hami_$$}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29673}"

O_OUTPUT_DIR="${OUTPUT_ROOT}/verifier_o"

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

die() {
  printf '[vo-retrieval-stateful-diagnostic] ERROR: %s\n' "$*" >&2
  exit 2
}

validate_phases() {
  local phase
  local -a requested=()
  IFS=',' read -r -a requested <<<"$PHASES"
  [[ "${#requested[@]}" -gt 0 ]] || die "PHASES must not be empty"
  for phase in "${requested[@]}"; do
    case "$phase" in
      preflight|prepare|infer|fanout|summarize) ;;
      *) die "unsupported phase ${phase@Q}; expected preflight,prepare,infer,fanout,summarize" ;;
    esac
  done
}

checkpoint_progress_step() {
  local marker
  for marker in \
    "${O_RUN_DIR}/training_complete.json" \
    "${O_RUN_DIR}/latest_state/trainer_state.json"; do
    if [[ -f "$marker" ]]; then
      jq -er '.global_step | numbers' "$marker" 2>/dev/null && return 0
    fi
  done
  return 1
}

stable_sha256() {
  local path="$1"
  local before after actual
  before="$(stat -c '%s:%y' "$path")"
  [[ "${before%%:*}" -gt 0 ]] || die "checkpoint artifact is empty: ${path}"
  actual="$(sha256sum "$path" | awk '{print $1}')"
  after="$(stat -c '%s:%y' "$path")"
  [[ "$before" == "$after" ]] || \
    die "checkpoint artifact changed while hashing; wait for checkpoint save to finish: ${path}"
  printf '%s' "$actual"
}

active_vo_training_pids() {
  local train_config="${O_RUN_DIR%/train}/train.resolved.yaml"
  local cmdline cmd pid
  for cmdline in /proc/[0-9]*/cmdline; do
    [[ -r "$cmdline" ]] || continue
    cmd="$(tr '\0' ' ' <"$cmdline" 2>/dev/null || true)"
    [[ "$cmd" == *"-m sft.hami_cuda_bootstrap"* ]] || continue
    [[ "$cmd" == *"--config ${train_config}"* ]] || continue
    pid="${cmdline#/proc/}"
    pid="${pid%/cmdline}"
    printf '%s ' "$pid"
  done
}

[[ "$CHECKPOINT" == "checkpoint-800" ]] || \
  die "CHECKPOINT is frozen to checkpoint-800; got ${CHECKPOINT}"
[[ "$O_EXPECTED_ADAPTER_SHA256" == "$FROZEN_O_ADAPTER_SHA256" ]] || \
  die "O_EXPECTED_ADAPTER_SHA256 is frozen to ${FROZEN_O_ADAPTER_SHA256}"
[[ "$MATRIX_MANIFEST_EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ ]] || \
  die "MATRIX_MANIFEST_EXPECTED_SHA256 must be a lowercase SHA-256"
[[ "$EVAL_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || \
  die "EVAL_NPROC_PER_NODE must be a positive integer"
[[ "$NUM_MACHINES" == "1" ]] || \
  die "the frozen diagnostic supports NUM_MACHINES=1 only; got ${NUM_MACHINES}"
[[ "$MIXED_PRECISION" == "bf16" ]] || \
  die "the frozen diagnostic requires MIXED_PRECISION=bf16; got ${MIXED_PRECISION}"
[[ "$PER_DEVICE_EVAL_BATCH_SIZE" == "1" ]] || \
  die "the frozen diagnostic requires PER_DEVICE_EVAL_BATCH_SIZE=1; got ${PER_DEVICE_EVAL_BATCH_SIZE}"
validate_phases

if phase_enabled infer && [[ "$DRY_RUN" != "true" ]]; then
  ACTIVE_O_PIDS="$(active_vo_training_pids)"
  if [[ -n "$ACTIVE_O_PIDS" ]]; then
    die "V_O training is still active (pids=${ACTIVE_O_PIDS}); this diagnostic is forbidden from running concurrently with V_O training"
  fi
fi

[[ -f "$MATRIX_MANIFEST" ]] || die "missing R/S matrix manifest: ${MATRIX_MANIFEST}"
MATRIX_MANIFEST_ACTUAL_SHA256="$(sha256sum "$MATRIX_MANIFEST" | awk '{print $1}')"
[[ "$MATRIX_MANIFEST_ACTUAL_SHA256" == "$MATRIX_MANIFEST_EXPECTED_SHA256" ]] || \
  die "R/S matrix manifest SHA mismatch: expected=${MATRIX_MANIFEST_EXPECTED_SHA256} actual=${MATRIX_MANIFEST_ACTUAL_SHA256}"

jq -e '
  .split == "val"
  and .label_schema == "liar6"
  and .expected_k == 5
  and .event_count == 1234
  and .event_id_sequence_sha256 == "65038f1f222b7d990642970ebf7281434abdb17fe61ec1e14ed0c937e8ee6549"
  and .checkpoint_contract.checkpoint == "checkpoint-800"
  and .checkpoint_contract.split == "val"
  and .checkpoint_contract.test_allowed == false
  and .checkpoint_contract.best_alias_allowed == false
  and .cell_count == 2
  and ([.cells[].cell_id] == ["retrieval__fixed5", "stateful__fixed5"])
' "$MATRIX_MANIFEST" >/dev/null || \
  die "matrix manifest violates the frozen val/K5/1234/R/S/step800 contract"

MATRIX_MANIFEST_DIR="$(cd "$(dirname "$MATRIX_MANIFEST")" && pwd)"
SOURCE_MATRIX_RELATIVE="$(jq -er '.source_matrix.path' "$MATRIX_MANIFEST")"
SOURCE_MATRIX_MANIFEST="$(realpath -m "${MATRIX_MANIFEST_DIR}/${SOURCE_MATRIX_RELATIVE}")"
SOURCE_MATRIX_EXPECTED_SHA256="$(jq -er '.source_matrix.sha256' "$MATRIX_MANIFEST")"
[[ -f "$SOURCE_MATRIX_MANIFEST" ]] || \
  die "missing frozen source matrix: ${SOURCE_MATRIX_MANIFEST}"
SOURCE_MATRIX_ACTUAL_SHA256="$(sha256sum "$SOURCE_MATRIX_MANIFEST" | awk '{print $1}')"
[[ "$SOURCE_MATRIX_ACTUAL_SHA256" == "$SOURCE_MATRIX_EXPECTED_SHA256" ]] || \
  die "frozen source matrix SHA mismatch: expected=${SOURCE_MATRIX_EXPECTED_SHA256} actual=${SOURCE_MATRIX_ACTUAL_SHA256}"

resolve_o_adapter_sha() {
  local adapter_file="${O_RUN_DIR}/${CHECKPOINT}/adapter_model.safetensors"
  local adapter_config="${O_RUN_DIR}/${CHECKPOINT}/adapter_config.json"
  local checkpoint_step="${CHECKPOINT#checkpoint-}"
  local progress_step actual_sha
  [[ -f "$O_CONFIG" ]] || die "missing V_O resolved config: ${O_CONFIG}"
  [[ -f "$adapter_file" ]] || die "V_O checkpoint adapter is missing: ${adapter_file}"
  [[ -f "$adapter_config" ]] || \
    die "V_O checkpoint adapter config is missing: ${adapter_config}"
  jq -e '
    (.base_model_name_or_path | type == "string" and length > 0)
    and (.peft_type | type == "string" and length > 0)
  ' "$adapter_config" >/dev/null || \
    die "V_O checkpoint adapter config is incomplete: ${adapter_config}"
  progress_step="$(checkpoint_progress_step)" || \
    die "V_O has no post-save progress marker; latest_state or training_complete must record step ${checkpoint_step}"
  [[ "$progress_step" =~ ^[0-9]+$ && "$progress_step" -ge "$checkpoint_step" ]] || \
    die "V_O checkpoint is not post-save ready: progress_step=${progress_step} required>=${checkpoint_step}"
  actual_sha="$(stable_sha256 "$adapter_file")"
  [[ "$actual_sha" == "$O_EXPECTED_ADAPTER_SHA256" ]] || \
    die "V_O checkpoint-800 adapter SHA mismatch: expected=${O_EXPECTED_ADAPTER_SHA256} actual=${actual_sha}"
  printf '[vo-retrieval-stateful-diagnostic] preflight V_O checkpoint=%s progress_step=%s adapter_sha256=%s\n' \
    "$CHECKPOINT" "$progress_step" "$actual_sha" >&2
  printf '%s' "$actual_sha"
}

if phase_enabled infer || phase_enabled preflight; then
  if [[ "$DRY_RUN" == "true" ]]; then
    O_ADAPTER_SHA256="$O_EXPECTED_ADAPTER_SHA256"
  else
    O_ADAPTER_SHA256="$(resolve_o_adapter_sha)"
    mkdir -p "$HAMI_SHARED_CACHE_ROOT"
  fi
else
  O_ADAPTER_SHA256="$O_EXPECTED_ADAPTER_SHA256"
fi

printf '[vo-retrieval-stateful-diagnostic] split=val checkpoint=checkpoint-800 k=5 events=1234 cells=retrieval__fixed5,stateful__fixed5 phases=%s\n' "$PHASES"
printf '[vo-retrieval-stateful-diagnostic] matrix=%s matrix_sha256=%s output_root=%s\n' \
  "$MATRIX_MANIFEST" "$MATRIX_MANIFEST_ACTUAL_SHA256" "$OUTPUT_ROOT"
printf '[vo-retrieval-stateful-diagnostic] V_O run=%s adapter_sha256=%s output=%s\n' \
  "$O_RUN_DIR" "$O_ADAPTER_SHA256" "$O_OUTPUT_DIR"
printf '[vo-retrieval-stateful-diagnostic] eval_nproc=%s num_machines=1 mixed_precision=bf16 per_device_eval_batch_size=1 cuda_visible_devices=%s single_verifier=true diagnostic_only=true\n' \
  "$EVAL_NPROC_PER_NODE" "${CUDA_VISIBLE_DEVICES:-<all-visible-devices>}"

prepare_matrix() {
  local args=(-m sft.label_token_matrix_infer prepare
    --matrix-manifest "$MATRIX_MANIFEST"
    --build-root "$BUILD_ROOT"
    --output-dir "$O_OUTPUT_DIR"
    --split val
    --label-prefix 'Label:')
  if [[ "$FORCE_PREPARE" == "true" ]]; then
    args+=(--force-prepare)
  fi
  run_cmd "$PYTHON_BIN" "${args[@]}"
}

infer_matrix() {
  local hami_cache="${HAMI_SHARED_CACHE_ROOT}/verifier_o.cache"
  local infer_args=(-m sft.hami_cuda_bootstrap infer
    --output-dir "$O_OUTPUT_DIR"
    --run-dir "$O_RUN_DIR"
    --checkpoint checkpoint-800
    --config "$O_CONFIG"
    --split val
    --expected-world-size "$EVAL_NPROC_PER_NODE"
    --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --expected-adapter-sha256 "$O_ADAPTER_SHA256")
  if [[ "$FORCE_INFER" == "true" ]]; then
    infer_args+=(--force-infer)
  fi
  if [[ "$EVAL_NPROC_PER_NODE" -gt 1 ]]; then
    run_cmd env \
      "CUDA_DEVICE_MEMORY_SHARED_CACHE=${hami_cache}" \
      "SFT_HAMI_BOOTSTRAP_TARGET_MODULE=sft.label_token_matrix_infer" \
      "$ACCELERATE_BIN" launch \
      --multi_gpu \
      --num_processes "$EVAL_NPROC_PER_NODE" \
      --num_machines "$NUM_MACHINES" \
      --mixed_precision "$MIXED_PRECISION" \
      --main_process_port "$MAIN_PROCESS_PORT" \
      "${infer_args[@]}"
  else
    run_cmd env \
      "LOCAL_RANK=0" \
      "CUDA_DEVICE_MEMORY_SHARED_CACHE=${hami_cache}" \
      "SFT_HAMI_BOOTSTRAP_TARGET_MODULE=sft.label_token_matrix_infer" \
      "$PYTHON_BIN" "${infer_args[@]}"
  fi
}

fanout_matrix() {
  local args=(-m sft.label_token_matrix_infer fanout
    --output-dir "$O_OUTPUT_DIR"
    --unsafe-skip-equivalence-gate)
  if [[ "$FORCE_FANOUT" == "true" ]]; then
    args+=(--force-fanout)
  fi
  run_cmd "$PYTHON_BIN" "${args[@]}"
}

if phase_enabled prepare; then
  prepare_matrix
fi

if phase_enabled infer; then
  infer_matrix
fi

if phase_enabled fanout; then
  fanout_matrix
fi

if phase_enabled summarize; then
  run_cmd "$PYTHON_BIN" scripts/phase5_selectors/analyze/summarize_vo_retrieval_stateful_diagnostic.py \
    --verifier-o-dir "$O_OUTPUT_DIR" \
    --output-json "${OUTPUT_ROOT}/summary.json" \
    --output-md "${OUTPUT_ROOT}/summary.md"
fi

printf '[vo-retrieval-stateful-diagnostic] complete phases=%s output=%s\n' "$PHASES" "$OUTPUT_ROOT"
