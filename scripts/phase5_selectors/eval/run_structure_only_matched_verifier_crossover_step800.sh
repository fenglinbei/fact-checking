#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

MATRIX_MANIFEST="${MATRIX_MANIFEST:-configs/validation/structure_only_matched_verifier_crossover_step800_v0_1.json}"
BUILD_ROOT="${BUILD_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/frozen_matrix_val}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/matched_verifier_crossover_step800_val}"

S_RUN_DIR="${S_RUN_DIR:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_fullpool_minmax5_10_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/train}"
O_RUN_DIR="${O_RUN_DIR:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_structure_only_one_shot_fullpool_minmax5_10_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/train}"
S_CONFIG="${S_CONFIG:-${S_RUN_DIR}/config.resolved.yaml}"
O_CONFIG="${O_CONFIG:-${O_RUN_DIR}/config.resolved.yaml}"

CHECKPOINT="${CHECKPOINT:-checkpoint-800}"
S_EXPECTED_ADAPTER_SHA256="${S_EXPECTED_ADAPTER_SHA256:-7b7512cd8f5a37d7087be935c3d768db04a29dd3bd479131bd1c5c7681b9374a}"
O_EXPECTED_ADAPTER_SHA256="${O_EXPECTED_ADAPTER_SHA256:-24e661e8efec049f19e4427a4488de57ce4dc7aec97e412315029412f8779aa3}"
PHASES="${PHASES:-prepare,infer,fanout,summarize}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"
FORCE_INFER="${FORCE_INFER:-false}"
FORCE_FANOUT="${FORCE_FANOUT:-false}"
ALLOW_CONCURRENT_TRAINING="${ALLOW_CONCURRENT_TRAINING:-false}"
DRY_RUN="${DRY_RUN:-false}"
PYTHON_BIN="${MATRIX_PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/accelerate}"
HAMI_SHARED_CACHE_ROOT="${HAMI_SHARED_CACHE_ROOT:-/tmp/lzj_structure_only_crossover_hami_$$}"
MAIN_PROCESS_PORT_S="${MAIN_PROCESS_PORT_S:-29651}"
MAIN_PROCESS_PORT_O="${MAIN_PROCESS_PORT_O:-29652}"

SEED43_TAIL_EXACT_OUTPUT_ROOT="outputs/selector_mechanism_gate/liar_raw_structure_only_core_gate_v0_1/matched_verifier_crossover_seed43_step800_val"
SEED43_TAIL_BASE_DIR="${SEED43_TAIL_BASE_DIR:-outputs/sentence_trace_method/queues/mrec_vo_crossover_20260717_0311_ctxfix}"
SEED43_TAIL_SENTINEL="${SEED43_TAIL_SENTINEL:-${SEED43_TAIL_BASE_DIR}/enable_seed43_pre_crossover_tail}"
SEED43_TAIL_LOCK_FILE="${SEED43_TAIL_LOCK_FILE:-${SEED43_TAIL_BASE_DIR}/seed43_pre_crossover_tail.lock}"
SEED43_PRE_CROSSOVER_HOOK="${SEED43_PRE_CROSSOVER_HOOK:-scripts/sentence_trace_method/run_paired_seed43_pre_crossover_tail.sh}"
SEED43_TAIL_HOOK_TRIGGERED=false
if [[ "$OUTPUT_ROOT" == "$SEED43_TAIL_EXACT_OUTPUT_ROOT" && \
      -f "$SEED43_TAIL_SENTINEL" && \
      "${SEED43_TAIL_HOOK_ACTIVE:-false}" != "true" ]]; then
  exec 8>"$SEED43_TAIL_LOCK_FILE"
  if ! flock -n 8; then
    printf '[matched-verifier-crossover] ERROR: seed43 pre-crossover tail lock is held: %s\n' \
      "$SEED43_TAIL_LOCK_FILE" >&2
    exit 73
  fi
  export SEED43_TAIL_HOOK_ACTIVE=true
  set +e
  env \
    HOOK_PHASE=pre \
    TAIL_HOOK_LOCK_HELD=true \
    TAIL_BASE_DIR="$SEED43_TAIL_BASE_DIR" \
    TAIL_SENTINEL="$SEED43_TAIL_SENTINEL" \
    TAIL_LOCK_FILE="$SEED43_TAIL_LOCK_FILE" \
    SEED43_S_TRAIN_DIR="$S_RUN_DIR" \
    SEED43_O_TRAIN_DIR="$O_RUN_DIR" \
    CROSSOVER_OUTPUT_ROOT="$OUTPUT_ROOT" \
    DRY_RUN="$DRY_RUN" \
    bash "$SEED43_PRE_CROSSOVER_HOOK"
  seed43_tail_rc=$?
  set -e
  if (( seed43_tail_rc == 76 )); then
    printf '[matched-verifier-crossover] seed43 crossover already complete; tail audit finalized\n'
    printf '[matched-verifier-crossover] complete phases=skipped_existing output=%s\n' "$OUTPUT_ROOT"
    exit 0
  fi
  if (( seed43_tail_rc != 0 )); then
    printf '[matched-verifier-crossover] seed43 pre-crossover tail stopped wrapper rc=%s\n' \
      "$seed43_tail_rc" >&2
    exit "$seed43_tail_rc"
  fi
  SEED43_TAIL_HOOK_TRIGGERED=true
fi

S_OUTPUT_DIR="${OUTPUT_ROOT}/verifier_s"
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
  printf '[matched-verifier-crossover] ERROR: %s\n' "$*" >&2
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
  local run_dir="$1"
  local marker
  for marker in \
    "${run_dir}/training_complete.json" \
    "${run_dir}/latest_state/trainer_state.json"; do
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

active_training_pids() {
  local run_dir="$1"
  local train_config="${run_dir%/train}/train.resolved.yaml"
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
[[ "$EVAL_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || \
  die "EVAL_NPROC_PER_NODE must be a positive integer"
[[ "$NUM_MACHINES" == "1" ]] || \
  die "the frozen crossover supports NUM_MACHINES=1 only; got ${NUM_MACHINES}"
[[ "$MIXED_PRECISION" == "bf16" ]] || \
  die "the frozen crossover requires MIXED_PRECISION=bf16; got ${MIXED_PRECISION}"
[[ "$PER_DEVICE_EVAL_BATCH_SIZE" == "1" ]] || \
  die "the frozen crossover requires PER_DEVICE_EVAL_BATCH_SIZE=1; got ${PER_DEVICE_EVAL_BATCH_SIZE}"
validate_phases
if phase_enabled infer && [[ "$DRY_RUN" != "true" && "$ALLOW_CONCURRENT_TRAINING" != "true" ]]; then
  ACTIVE_S_PIDS="$(active_training_pids "$S_RUN_DIR")"
  ACTIVE_O_PIDS="$(active_training_pids "$O_RUN_DIR")"
  if [[ -n "$ACTIVE_S_PIDS" || -n "$ACTIVE_O_PIDS" ]]; then
    die "training is still active on the verifier runs (V_S=${ACTIVE_S_PIDS:-none} V_O=${ACTIVE_O_PIDS:-none}); wait for GPU release or set ALLOW_CONCURRENT_TRAINING=true only with a disjoint GPU allocation"
  fi
fi
[[ -f "$MATRIX_MANIFEST" ]] || die "missing matrix manifest: ${MATRIX_MANIFEST}"

jq -e '
  .split == "val"
  and .checkpoint_contract.checkpoint == "checkpoint-800"
  and .checkpoint_contract.split == "val"
  and .checkpoint_contract.test_allowed == false
  and .checkpoint_contract.best_alias_allowed == false
  and .cell_count == 2
  and ([.cells[].cell_id] == ["one_shot__fixed5", "stateful__fixed5"])
' "$MATRIX_MANIFEST" >/dev/null || die "matrix manifest violates the frozen val/O/S/step800 contract"

MATRIX_MANIFEST_DIR="$(cd "$(dirname "$MATRIX_MANIFEST")" && pwd)"
SOURCE_MATRIX_RELATIVE="$(jq -er '.source_matrix.path' "$MATRIX_MANIFEST")"
SOURCE_MATRIX_MANIFEST="$(realpath -m "${MATRIX_MANIFEST_DIR}/${SOURCE_MATRIX_RELATIVE}")"
SOURCE_MATRIX_EXPECTED_SHA256="$(jq -er '.source_matrix.sha256' "$MATRIX_MANIFEST")"
[[ -f "$SOURCE_MATRIX_MANIFEST" ]] || \
  die "missing frozen source matrix: ${SOURCE_MATRIX_MANIFEST}"
SOURCE_MATRIX_ACTUAL_SHA256="$(sha256sum "$SOURCE_MATRIX_MANIFEST" | awk '{print $1}')"
[[ "$SOURCE_MATRIX_ACTUAL_SHA256" == "$SOURCE_MATRIX_EXPECTED_SHA256" ]] || \
  die "frozen source matrix SHA mismatch: expected=${SOURCE_MATRIX_EXPECTED_SHA256} actual=${SOURCE_MATRIX_ACTUAL_SHA256}"

resolve_adapter_sha() {
  local verifier_id="$1"
  local run_dir="$2"
  local declared_sha="$3"
  local adapter_file="${run_dir}/${CHECKPOINT}/adapter_model.safetensors"
  local adapter_config="${run_dir}/${CHECKPOINT}/adapter_config.json"
  local checkpoint_step="${CHECKPOINT#checkpoint-}"
  local progress_step actual_sha
  [[ -f "${run_dir}/config.resolved.yaml" ]] || \
    die "${verifier_id} resolved config is missing: ${run_dir}/config.resolved.yaml"
  [[ -f "$adapter_file" ]] || die "${verifier_id} checkpoint adapter is missing: ${adapter_file}"
  [[ -f "$adapter_config" ]] || \
    die "${verifier_id} checkpoint adapter config is missing: ${adapter_config}"
  jq -e '
    (.base_model_name_or_path | type == "string" and length > 0)
    and (.peft_type | type == "string" and length > 0)
  ' "$adapter_config" >/dev/null || \
    die "${verifier_id} checkpoint adapter config is incomplete: ${adapter_config}"
  progress_step="$(checkpoint_progress_step "$run_dir")" || \
    die "${verifier_id} has no post-save progress marker; wait until latest_state or training_complete records step ${checkpoint_step}"
  [[ "$progress_step" =~ ^[0-9]+$ && "$progress_step" -ge "$checkpoint_step" ]] || \
    die "${verifier_id} checkpoint is not post-save ready: progress_step=${progress_step} required>=${checkpoint_step}"
  actual_sha="$(stable_sha256 "$adapter_file")"
  if [[ -n "$declared_sha" && "$declared_sha" != "$actual_sha" ]]; then
    die "${verifier_id} checkpoint-800 adapter SHA mismatch: expected=${declared_sha} actual=${actual_sha}"
  fi
  printf '[matched-verifier-crossover] preflight %s checkpoint=%s progress_step=%s adapter_sha256=%s\n' \
    "$verifier_id" "$CHECKPOINT" "$progress_step" "$actual_sha" >&2
  printf '%s' "$actual_sha"
}

if phase_enabled infer || phase_enabled preflight; then
  if [[ "$DRY_RUN" == "true" ]]; then
    S_ADAPTER_SHA256="${S_EXPECTED_ADAPTER_SHA256}"
    O_ADAPTER_SHA256="${O_EXPECTED_ADAPTER_SHA256:-DRY_RUN_O_CHECKPOINT800_SHA256}"
  else
    [[ -f "$S_CONFIG" ]] || die "missing V_S config: ${S_CONFIG}"
    [[ -f "$O_CONFIG" ]] || die "missing V_O config: ${O_CONFIG}"
    S_ADAPTER_SHA256="$(resolve_adapter_sha V_S "$S_RUN_DIR" "$S_EXPECTED_ADAPTER_SHA256")"
    O_ADAPTER_SHA256="$(resolve_adapter_sha V_O "$O_RUN_DIR" "$O_EXPECTED_ADAPTER_SHA256")"
    mkdir -p "$HAMI_SHARED_CACHE_ROOT"
  fi
else
  S_ADAPTER_SHA256="${S_EXPECTED_ADAPTER_SHA256}"
  O_ADAPTER_SHA256="${O_EXPECTED_ADAPTER_SHA256:-not-required-before-infer}"
fi

printf '[matched-verifier-crossover] split=val checkpoint=checkpoint-800 cells=one_shot__fixed5,stateful__fixed5 phases=%s\n' "$PHASES"
printf '[matched-verifier-crossover] matrix=%s build_root=%s output_root=%s\n' "$MATRIX_MANIFEST" "$BUILD_ROOT" "$OUTPUT_ROOT"
printf '[matched-verifier-crossover] V_S run=%s adapter_sha256=%s output=%s\n' "$S_RUN_DIR" "$S_ADAPTER_SHA256" "$S_OUTPUT_DIR"
printf '[matched-verifier-crossover] V_O run=%s adapter_sha256=%s output=%s\n' "$O_RUN_DIR" "$O_ADAPTER_SHA256" "$O_OUTPUT_DIR"
printf '[matched-verifier-crossover] eval_nproc=%s num_machines=1 mixed_precision=bf16 per_device_eval_batch_size=1 cuda_visible_devices=%s hami_bootstrap_target=sft.label_token_matrix_infer diagnostic_only=true\n' \
  "$EVAL_NPROC_PER_NODE" "${CUDA_VISIBLE_DEVICES:-<all-visible-devices>}"

prepare_matrix() {
  local output_dir="$1"
  local args=(-m sft.label_token_matrix_infer prepare
    --matrix-manifest "$MATRIX_MANIFEST"
    --build-root "$BUILD_ROOT"
    --output-dir "$output_dir"
    --split val
    --label-prefix 'Label:')
  if [[ "$FORCE_PREPARE" == "true" ]]; then
    args+=(--force-prepare)
  fi
  run_cmd "$PYTHON_BIN" "${args[@]}"
}

infer_matrix() {
  local verifier_id="$1"
  local output_dir="$2"
  local run_dir="$3"
  local config="$4"
  local adapter_sha256="$5"
  local port="$6"
  local hami_cache="${HAMI_SHARED_CACHE_ROOT}/${verifier_id}.cache"
  local infer_args=(-m sft.hami_cuda_bootstrap infer
    --output-dir "$output_dir"
    --run-dir "$run_dir"
    --checkpoint checkpoint-800
    --config "$config"
    --split val
    --expected-world-size "$EVAL_NPROC_PER_NODE"
    --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE"
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
    --expected-adapter-sha256 "$adapter_sha256")
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
      --main_process_port "$port" \
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
  local output_dir="$1"
  local args=(-m sft.label_token_matrix_infer fanout
    --output-dir "$output_dir"
    --unsafe-skip-equivalence-gate)
  if [[ "$FORCE_FANOUT" == "true" ]]; then
    args+=(--force-fanout)
  fi
  run_cmd "$PYTHON_BIN" "${args[@]}"
}

if phase_enabled prepare; then
  prepare_matrix "$S_OUTPUT_DIR"
  prepare_matrix "$O_OUTPUT_DIR"
fi

if phase_enabled infer; then
  infer_matrix verifier_s "$S_OUTPUT_DIR" "$S_RUN_DIR" "$S_CONFIG" "$S_ADAPTER_SHA256" "$MAIN_PROCESS_PORT_S"
  infer_matrix verifier_o "$O_OUTPUT_DIR" "$O_RUN_DIR" "$O_CONFIG" "$O_ADAPTER_SHA256" "$MAIN_PROCESS_PORT_O"
fi

if phase_enabled fanout; then
  fanout_matrix "$S_OUTPUT_DIR"
  fanout_matrix "$O_OUTPUT_DIR"
fi

if phase_enabled summarize; then
  run_cmd "$PYTHON_BIN" scripts/phase5_selectors/analyze/summarize_matched_verifier_crossover.py \
    --verifier-s-dir "$S_OUTPUT_DIR" \
    --verifier-o-dir "$O_OUTPUT_DIR" \
    --output-json "${OUTPUT_ROOT}/summary.json" \
    --output-md "${OUTPUT_ROOT}/summary.md"
fi

if [[ "$SEED43_TAIL_HOOK_TRIGGERED" == "true" ]]; then
  env \
    HOOK_PHASE=finalize \
    TAIL_HOOK_LOCK_HELD=true \
    TAIL_BASE_DIR="$SEED43_TAIL_BASE_DIR" \
    TAIL_SENTINEL="$SEED43_TAIL_SENTINEL" \
    TAIL_LOCK_FILE="$SEED43_TAIL_LOCK_FILE" \
    SEED43_S_TRAIN_DIR="$S_RUN_DIR" \
    SEED43_O_TRAIN_DIR="$O_RUN_DIR" \
    CROSSOVER_OUTPUT_ROOT="$OUTPUT_ROOT" \
    DRY_RUN="$DRY_RUN" \
    bash "$SEED43_PRE_CROSSOVER_HOOK"
fi

printf '[matched-verifier-crossover] complete phases=%s output=%s\n' "$PHASES" "$OUTPUT_ROOT"
