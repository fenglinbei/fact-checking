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
N_CONFIG="${N_CONFIG:-${N_RUN_DIR}/config.resolved.yaml}"
S_CONFIG="${S_CONFIG:-${S_RUN_DIR}/config.resolved.yaml}"
N_EXPECTED_ADAPTER_SHA256="${N_EXPECTED_ADAPTER_SHA256:-}"
S_EXPECTED_ADAPTER_SHA256="${S_EXPECTED_ADAPTER_SHA256:-7b7512cd8f5a37d7087be935c3d768db04a29dd3bd479131bd1c5c7681b9374a}"
PHASES="${PHASES:-prepare,infer,fanout,summarize}"
EVAL_NPROC_PER_NODE="${EVAL_NPROC_PER_NODE:-4}"
PYTHON_BIN="${MATRIX_PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/accelerate}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
HAMI_SHARED_CACHE_ROOT="${HAMI_SHARED_CACHE_ROOT:-/tmp/lzj_no_map_fixed5_crossover_hami_$$}"
MAIN_PROCESS_PORT_N="${MAIN_PROCESS_PORT_N:-29687}"
MAIN_PROCESS_PORT_S="${MAIN_PROCESS_PORT_S:-29688}"
FORCE_PREPARE="${FORCE_PREPARE:-false}"
FORCE_INFER="${FORCE_INFER:-false}"
FORCE_FANOUT="${FORCE_FANOUT:-false}"
DRY_RUN="${DRY_RUN:-false}"

die() {
  printf '[no-map-fixed5-crossover] ERROR: %s\n' "$*" >&2
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

phase_enabled() {
  [[ ",${PHASES}," == *",$1,"* ]]
}

for phase in ${PHASES//,/ }; do
  case "$phase" in prepare|infer|fanout|summarize|preflight) ;; *) die "unsupported phase=${phase}" ;; esac
done
[[ "$EVAL_NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]] || die "EVAL_NPROC_PER_NODE must be positive"
[[ "$DRY_RUN" == "true" || "$DRY_RUN" == "false" ]] || die "DRY_RUN must be true or false"
[[ -s "$MATRIX_MANIFEST" ]] || die "missing matrix manifest=${MATRIX_MANIFEST}"
jq -e '
  .schema_version == "no-map-structure-fixed5-matrix-v0.1"
  and .matrix_kind == "no_map_structure_matched_verifier_crossover"
  and .split == "val"
  and .expected_k == 5
  and .event_count == 1234
  and .cell_count == 2
  and .all_ready == true
  and ([.cells[].cell_id] == ["N_fixed5", "S_fixed5"])
  and .checkpoint_contract.checkpoint == "checkpoint-800"
  and .checkpoint_contract.test_allowed == false
  and .checkpoint_contract.best_alias_allowed == false
' "$MATRIX_MANIFEST" >/dev/null || die "matrix violates N_fixed5/S_fixed5 val/K5/n=1234 contract"

stable_sha() {
  local path="$1" before after sha
  before="$(stat -c '%s:%Y:%i' "$path")" || return 1
  [[ "${before%%:*}" -gt 0 ]] || return 1
  sha="$(sha256sum "$path" | awk '{print $1}')"
  after="$(stat -c '%s:%Y:%i' "$path")" || return 1
  [[ "$before" == "$after" ]] || return 1
  printf '%s\n' "$sha"
}

resolve_adapter() {
  local role="$1" run_dir="$2" expected="$3" adapter progress actual completion
  adapter="${run_dir}/checkpoint-800/adapter_model.safetensors"
  [[ -s "$adapter" && -s "${run_dir}/checkpoint-800/adapter_config.json" && -s "${run_dir}/config.resolved.yaml" ]] || \
    die "${role} checkpoint/config incomplete"
  if [[ "$role" == "V_N" ]]; then
    [[ ! -e "${run_dir}/training_complete.json" ]] || \
      die "${role} no-map verifier must remain a capped, incomplete fixed-step run"
    progress="$(jq -er '.global_step | numbers' "${run_dir}/latest_state/trainer_state.json" 2>/dev/null || true)"
  else
    completion="${run_dir}/training_complete.json"
    jq -e '.completed == true' "$completion" >/dev/null 2>&1 || \
      die "${role} completed verifier is missing a valid training_complete marker"
    progress="$(jq -er '.global_step | numbers' "$completion" 2>/dev/null || true)"
  fi
  [[ "$progress" =~ ^[0-9]+$ && "$progress" -ge 800 ]] || die "${role} progress marker is below step 800"
  actual="$(stable_sha "$adapter")" || die "${role} adapter is unstable"
  [[ -n "$expected" && "$actual" == "$expected" ]] || die "${role} checkpoint-800 SHA mismatch expected=${expected:-missing} actual=${actual}"
  printf '[no-map-fixed5-crossover] preflight %s progress=%s adapter_sha256=%s\n' "$role" "$progress" "$actual" >&2
  printf '%s\n' "$actual"
}

active_training_for() {
  local config="$1" cmdline cmd
  for cmdline in /proc/[0-9]*/cmdline; do
    [[ -r "$cmdline" ]] || continue
    cmd="$(tr '\0' ' ' < "$cmdline" 2>/dev/null || true)"
    if [[ "$cmd" == *"-m sft.hami_cuda_bootstrap"* && "$cmd" == *"--config ${config}"* ]]; then
      return 0
    fi
  done
  return 1
}

if phase_enabled infer || phase_enabled preflight; then
  if [[ "$DRY_RUN" == "true" ]]; then
    N_ADAPTER_SHA256="${N_EXPECTED_ADAPTER_SHA256:-DRY_RUN_N_SHA256}"
    S_ADAPTER_SHA256="$S_EXPECTED_ADAPTER_SHA256"
  else
    active_training_for "${N_RUN_DIR%/train}/train.resolved.yaml" && die "V_N training is still active"
    active_training_for "${S_RUN_DIR%/train}/train.resolved.yaml" && die "V_S training is still active"
    N_ADAPTER_SHA256="$(resolve_adapter V_N "$N_RUN_DIR" "$N_EXPECTED_ADAPTER_SHA256")"
    S_ADAPTER_SHA256="$(resolve_adapter V_S "$S_RUN_DIR" "$S_EXPECTED_ADAPTER_SHA256")"
    mkdir -p "$HAMI_SHARED_CACHE_ROOT"
  fi
else
  N_ADAPTER_SHA256="${N_EXPECTED_ADAPTER_SHA256:-not-required}"
  S_ADAPTER_SHA256="$S_EXPECTED_ADAPTER_SHA256"
fi

N_OUTPUT_DIR="${OUTPUT_ROOT}/verifier_n"
S_OUTPUT_DIR="${OUTPUT_ROOT}/verifier_s"
printf '[no-map-fixed5-crossover] split=val checkpoint=checkpoint-800 events=1234 cells=N_fixed5,S_fixed5 phases=%s\n' "$PHASES"
printf '[no-map-fixed5-crossover] V_N=%s sha=%s V_S=%s sha=%s\n' "$N_RUN_DIR" "$N_ADAPTER_SHA256" "$S_RUN_DIR" "$S_ADAPTER_SHA256"

prepare_one() {
  local output="$1"
  local args=(-m sft.label_token_matrix_infer prepare --matrix-manifest "$MATRIX_MANIFEST" --build-root "$MATRIX_ROOT" --output-dir "$output" --split val --label-prefix 'Label:')
  [[ "$FORCE_PREPARE" == "true" ]] && args+=(--force-prepare)
  run_cmd "$PYTHON_BIN" "${args[@]}"
}

infer_one() {
  local role="$1" output="$2" run_dir="$3" config="$4" sha="$5" port="$6"
  local args=(-m sft.hami_cuda_bootstrap infer --output-dir "$output" --run-dir "$run_dir" --checkpoint checkpoint-800 --config "$config" --split val --expected-world-size "$EVAL_NPROC_PER_NODE" --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE" --dataloader-num-workers "$DATALOADER_NUM_WORKERS" --expected-adapter-sha256 "$sha")
  [[ "$FORCE_INFER" == "true" ]] && args+=(--force-infer)
  if (( EVAL_NPROC_PER_NODE > 1 )); then
    run_cmd env "CUDA_DEVICE_MEMORY_SHARED_CACHE=${HAMI_SHARED_CACHE_ROOT}/${role}.cache" SFT_HAMI_BOOTSTRAP_TARGET_MODULE=sft.label_token_matrix_infer \
      "$ACCELERATE_BIN" launch --multi_gpu --num_processes "$EVAL_NPROC_PER_NODE" --num_machines 1 --mixed_precision bf16 --main_process_port "$port" "${args[@]}"
  else
    run_cmd env LOCAL_RANK=0 "CUDA_DEVICE_MEMORY_SHARED_CACHE=${HAMI_SHARED_CACHE_ROOT}/${role}.cache" SFT_HAMI_BOOTSTRAP_TARGET_MODULE=sft.label_token_matrix_infer "$PYTHON_BIN" "${args[@]}"
  fi
}

fanout_one() {
  local output="$1"
  local args=(-m sft.label_token_matrix_infer fanout --output-dir "$output" --unsafe-skip-equivalence-gate)
  [[ "$FORCE_FANOUT" == "true" ]] && args+=(--force-fanout)
  run_cmd "$PYTHON_BIN" "${args[@]}"
}

if phase_enabled prepare; then prepare_one "$N_OUTPUT_DIR"; prepare_one "$S_OUTPUT_DIR"; fi
if phase_enabled infer; then
  infer_one verifier_n "$N_OUTPUT_DIR" "$N_RUN_DIR" "$N_CONFIG" "$N_ADAPTER_SHA256" "$MAIN_PROCESS_PORT_N"
  infer_one verifier_s "$S_OUTPUT_DIR" "$S_RUN_DIR" "$S_CONFIG" "$S_ADAPTER_SHA256" "$MAIN_PROCESS_PORT_S"
fi
if phase_enabled fanout; then fanout_one "$N_OUTPUT_DIR"; fanout_one "$S_OUTPUT_DIR"; fi
if phase_enabled summarize; then
  run_cmd "$PYTHON_BIN" scripts/phase5_selectors/analyze/summarize_no_map_structure_fixed5_crossover.py \
    --verifier-n-dir "$N_OUTPUT_DIR" --verifier-s-dir "$S_OUTPUT_DIR" --matrix-dir "$MATRIX_ROOT" \
    --output-json "${OUTPUT_ROOT}/summary.json" --output-md "${OUTPUT_ROOT}/summary.md"
fi
printf '[no-map-fixed5-crossover] complete output=%s\n' "$OUTPUT_ROOT"
