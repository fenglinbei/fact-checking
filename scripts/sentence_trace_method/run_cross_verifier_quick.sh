#!/usr/bin/env bash
set -euo pipefail

# Run the frozen EviTrace cross-verifier quick evaluation:
#   prepare -> Qwen3 inference -> Llama-3.1 inference -> paired analysis
#
# The inference command is resumable; rerunning this wrapper revalidates the
# prepared data and lets cross_verifier_quick.py reuse hash-matched scores.
#
# Common overrides:
#   OUTPUT_ROOT, SEED, BOOTSTRAP, RANDOMIZATION
#   BUILD_PATH, EVITRACE_PATH, S4_PATH
#   BUILD_VAL_PATH, EVITRACE_VAL_PATH, S4_VAL_PATH
#   QWEN_MODEL_PATH, LLAMA_MODEL_PATH
#   GPU_CANDIDATES (comma-separated physical GPU indices)
#   GPU_MIN_FREE_MIB, GPU_STABLE_CHECKS, GPU_POLL_SECONDS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"
DRIVER="${DRIVER:-scripts/sentence_trace_method/cross_verifier_quick.py}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/evitrace_cross_verifier_quick_v1}"
SEED="${SEED:-20260724}"
BOOTSTRAP="${BOOTSTRAP:-10000}"
RANDOMIZATION="${RANDOMIZATION:-100000}"

BUILD_PATH="${BUILD_PATH:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10/build/build_test.jsonl}"
EVITRACE_PATH="${EVITRACE_PATH:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy_fullpool/selection_trace_test.jsonl}"
S4_PATH="${S4_PATH:-outputs/selectors/selector_mechanism_ablation_chunking/liar_raw_abc_selector_mech_s4_atom_union_source_score_ordered_test/selection_trace_test.jsonl}"

BUILD_VAL_PATH="${BUILD_VAL_PATH:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10/build/build_val.jsonl}"
EVITRACE_VAL_PATH="${EVITRACE_VAL_PATH:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy_fullpool/selection_trace_val.jsonl}"
S4_VAL_PATH="${S4_VAL_PATH:-outputs/selectors/selector_mechanism_ablation_chunking/liar_raw_abc_selector_mech_s4_atom_union_source_score_ordered_val/selection_trace_val.jsonl}"

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/data/models/Qwen3-4B-Instruct-2507}"
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-/data/models/Meta-Llama-3.1-8B-Instruct}"

NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"
GPU_CANDIDATES="${GPU_CANDIDATES:-${CUDA_VISIBLE_DEVICES:-}}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-40000}"
GPU_STABLE_CHECKS="${GPU_STABLE_CHECKS:-3}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
MODEL_COOLDOWN_SECONDS="${MODEL_COOLDOWN_SECONDS:-20}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-16}"

# --gpu-id is a physical index and the Python entry point owns device
# isolation. Do not leave a caller-provided CUDA remapping in its environment.
unset CUDA_VISIBLE_DEVICES

PREPARED_DIR="${OUTPUT_ROOT}/prepared"
PREPARED_MANIFEST="${PREPARED_DIR}/artifact_manifest.json"
QWEN_RESULT_DIR="${OUTPUT_ROOT}/inference/qwen3"
LLAMA_RESULT_DIR="${OUTPUT_ROOT}/inference/llama31"
QWEN_RESULT="${QWEN_RESULT_DIR}/logical_results.jsonl"
LLAMA_RESULT="${LLAMA_RESULT_DIR}/logical_results.jsonl"
ANALYSIS_DIR="${OUTPUT_ROOT}/analysis"
COMPLETE_MANIFEST="${ANALYSIS_DIR}/complete_manifest.json"
RUN_LOG="${RUN_LOG:-${OUTPUT_ROOT}/run.log}"
LOCK_FILE="${LOCK_FILE:-${OUTPUT_ROOT}/.run.lock}"

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf 'Another cross-verifier quick run holds lock: %s\n' "${LOCK_FILE}" >&2
  exit 2
fi
exec > >(tee -a "${RUN_LOG}") 2>&1

timestamp() {
  date '+%F %T %Z %z'
}

log() {
  printf '[cross-verifier %s] %s\n' "$(timestamp)" "$*"
}

die() {
  log "ERROR: $*"
  exit 2
}

require_uint() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer, got: ${value}"
}

require_positive_uint() {
  local name="$1"
  local value="$2"
  require_uint "${name}" "${value}"
  (( value > 0 )) || die "${name} must be greater than zero, got: ${value}"
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "missing file: ${path}"
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || die "missing directory: ${path}"
}

log_python_command() {
  printf '[cross-verifier %s] +' "$(timestamp)"
  printf ' PYTHONPATH=src'
  printf ' %q' "${PYTHON_BIN}" "${DRIVER}" "$@"
  printf '\n'
}

run_python() {
  log_python_command "$@"
  PYTHONPATH=src "${PYTHON_BIN}" "${DRIVER}" "$@"
}

declare -a REQUESTED_GPUS=()
if [[ -n "${GPU_CANDIDATES}" ]]; then
  IFS=',' read -r -a REQUESTED_GPUS <<< "${GPU_CANDIDATES//[[:space:]]/}"
  ((${#REQUESTED_GPUS[@]} > 0)) || die "GPU_CANDIDATES did not contain a GPU index"
  for gpu_id in "${REQUESTED_GPUS[@]}"; do
    [[ "${gpu_id}" =~ ^[0-9]+$ ]] || die "GPU_CANDIDATES must contain physical numeric indices: ${GPU_CANDIDATES}"
  done
fi

gpu_is_allowed() {
  local candidate="$1"
  local configured=""
  if ((${#REQUESTED_GPUS[@]} == 0)); then
    return 0
  fi
  for configured in "${REQUESTED_GPUS[@]}"; do
    [[ "${candidate}" == "${configured}" ]] && return 0
  done
  return 1
}

SELECTED_GPU=""

wait_for_stable_gpu() {
  local phase="$1"
  local snapshot=""
  local line=""
  local gpu_id=""
  local free_mib=""
  local best_gpu=""
  local best_free=-1
  local status=""
  local -A streak=()
  local -A current_free=()

  SELECTED_GPU=""
  log "${phase}: waiting for one GPU with >=${GPU_MIN_FREE_MIB} MiB free for ${GPU_STABLE_CHECKS} consecutive checks"
  if ((${#REQUESTED_GPUS[@]} > 0)); then
    log "${phase}: eligible physical GPUs: ${GPU_CANDIDATES}"
  else
    log "${phase}: eligible physical GPUs: all"
  fi

  while true; do
    if ! snapshot="$("${NVIDIA_SMI}" --query-gpu=index,memory.free --format=csv,noheader,nounits 2>&1)"; then
      log "${phase}: nvidia-smi query failed: ${snapshot}"
      sleep "${GPU_POLL_SECONDS}"
      continue
    fi

    current_free=()
    status=""
    while IFS= read -r line; do
      [[ -n "${line}" ]] || continue
      IFS=',' read -r gpu_id free_mib <<< "${line}"
      gpu_id="${gpu_id//[[:space:]]/}"
      free_mib="${free_mib//[[:space:]]/}"
      [[ "${gpu_id}" =~ ^[0-9]+$ && "${free_mib}" =~ ^[0-9]+$ ]] || \
        die "could not parse nvidia-smi row: ${line}"

      if ! gpu_is_allowed "${gpu_id}"; then
        continue
      fi

      current_free["${gpu_id}"]="${free_mib}"
      if (( free_mib >= GPU_MIN_FREE_MIB )); then
        streak["${gpu_id}"]=$(( ${streak["${gpu_id}"]:-0} + 1 ))
      else
        streak["${gpu_id}"]=0
      fi
      status+="${status:+; }gpu${gpu_id}=${free_mib}MiB(streak=${streak["${gpu_id}"]})"
    done <<< "${snapshot}"

    [[ -n "${status}" ]] || die "none of GPU_CANDIDATES were reported by nvidia-smi"
    log "${phase}: ${status}"

    best_gpu=""
    best_free=-1
    for gpu_id in "${!current_free[@]}"; do
      free_mib="${current_free["${gpu_id}"]}"
      if (( ${streak["${gpu_id}"]:-0} >= GPU_STABLE_CHECKS )); then
        if (( free_mib > best_free )) || \
          { (( free_mib == best_free )) && [[ -n "${best_gpu}" ]] && (( gpu_id < best_gpu )); }; then
          best_gpu="${gpu_id}"
          best_free="${free_mib}"
        fi
      fi
    done

    if [[ -n "${best_gpu}" ]]; then
      SELECTED_GPU="${best_gpu}"
      log "${phase}: selected physical GPU ${SELECTED_GPU} (${best_free} MiB free)"
      return 0
    fi
    sleep "${GPU_POLL_SECONDS}"
  done
}

infer_model() {
  local model_name="$1"
  local model_path="$2"
  local result_dir="$3"
  local expected_result="$4"
  local gpu_id=""

  wait_for_stable_gpu "infer/${model_name}"
  gpu_id="${SELECTED_GPU}"
  mkdir -p "${result_dir}"
  log "infer/${model_name}: starting resumable inference"
  run_python infer \
    --prepared-manifest "${PREPARED_MANIFEST}" \
    --model-name "${model_name}" \
    --model-path "${model_path}" \
    --output-dir "${result_dir}" \
    --gpu-id "${gpu_id}" \
    --batch-size "${INFER_BATCH_SIZE}" \
    --seed "${SEED}"
  require_file "${expected_result}"
  log "infer/${model_name}: complete (${expected_result})"
}

main() {
  require_positive_uint "SEED" "${SEED}"
  require_positive_uint "BOOTSTRAP" "${BOOTSTRAP}"
  require_positive_uint "RANDOMIZATION" "${RANDOMIZATION}"
  require_positive_uint "GPU_MIN_FREE_MIB" "${GPU_MIN_FREE_MIB}"
  require_positive_uint "GPU_STABLE_CHECKS" "${GPU_STABLE_CHECKS}"
  require_positive_uint "GPU_POLL_SECONDS" "${GPU_POLL_SECONDS}"
  require_uint "MODEL_COOLDOWN_SECONDS" "${MODEL_COOLDOWN_SECONDS}"
  require_positive_uint "INFER_BATCH_SIZE" "${INFER_BATCH_SIZE}"

  [[ -x "${PYTHON_BIN}" ]] || die "Python is not executable: ${PYTHON_BIN}"
  require_file "${DRIVER}"
  require_file "${BUILD_PATH}"
  require_file "${EVITRACE_PATH}"
  require_file "${S4_PATH}"
  require_file "${BUILD_VAL_PATH}"
  require_file "${EVITRACE_VAL_PATH}"
  require_file "${S4_VAL_PATH}"
  require_dir "${QWEN_MODEL_PATH}"
  require_dir "${LLAMA_MODEL_PATH}"
  command -v flock >/dev/null 2>&1 || die "flock is required"
  command -v "${NVIDIA_SMI}" >/dev/null 2>&1 || die "nvidia-smi is required: ${NVIDIA_SMI}"

  log "============================================================"
  log "EviTrace cross-verifier quick evaluation"
  log "project root:    ${PROJECT_ROOT}"
  log "output root:     ${OUTPUT_ROOT}"
  log "seed:            ${SEED}"
  log "bootstrap:       ${BOOTSTRAP}"
  log "randomization:   ${RANDOMIZATION}"
  log "infer batch:     ${INFER_BATCH_SIZE} unique prompts"
  log "Qwen model:      ${QWEN_MODEL_PATH}"
  log "Llama model:     ${LLAMA_MODEL_PATH}"
  log "run log:         ${RUN_LOG}"
  log "============================================================"

  mkdir -p "${PREPARED_DIR}" "${QWEN_RESULT_DIR}" "${LLAMA_RESULT_DIR}" "${ANALYSIS_DIR}"

  log "prepare: validating frozen artifacts and constructing main + order-only comparisons"
  run_python prepare \
    --build "${BUILD_PATH}" \
    --evitrace "${EVITRACE_PATH}" \
    --s4 "${S4_PATH}" \
    --build-val "${BUILD_VAL_PATH}" \
    --evitrace-val "${EVITRACE_VAL_PATH}" \
    --s4-val "${S4_VAL_PATH}" \
    --output-dir "${PREPARED_DIR}" \
    --seed "${SEED}"
  require_file "${PREPARED_MANIFEST}"
  log "prepare: complete (${PREPARED_MANIFEST})"

  infer_model "qwen3" "${QWEN_MODEL_PATH}" "${QWEN_RESULT_DIR}" "${QWEN_RESULT}"

  if (( MODEL_COOLDOWN_SECONDS > 0 )); then
    log "cooldown: waiting ${MODEL_COOLDOWN_SECONDS}s before the next model"
    sleep "${MODEL_COOLDOWN_SECONDS}"
  fi

  infer_model "llama31" "${LLAMA_MODEL_PATH}" "${LLAMA_RESULT_DIR}" "${LLAMA_RESULT}"

  log "analyze: combining paired results from both external verifiers"
  run_python analyze \
    --prepared-manifest "${PREPARED_MANIFEST}" \
    --result "${QWEN_RESULT}" \
    --result "${LLAMA_RESULT}" \
    --output-dir "${ANALYSIS_DIR}" \
    --bootstrap "${BOOTSTRAP}" \
    --randomization "${RANDOMIZATION}" \
    --seed "${SEED}"
  require_file "${COMPLETE_MANIFEST}"
  "${PYTHON_BIN}" -c \
    'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if data.get("complete") is True else 2)' \
    "${COMPLETE_MANIFEST}" || die "analysis manifest exists but complete is not true"

  log "complete: ${COMPLETE_MANIFEST}"
  log "report:   ${ANALYSIS_DIR}/report.md"
}

main "$@"
