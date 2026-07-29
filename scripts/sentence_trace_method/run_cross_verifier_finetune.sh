#!/usr/bin/env bash
set -euo pipefail

# Frozen serial launcher for the LIAR-RAW fair mixed-arm verifier experiment.
# Completion is decided only by training_complete.json plus a hashable best
# adapter; the presence of best/ alone never suppresses a resumable run.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/data/liaozijie/conda/accelerate-fc/bin/accelerate}"
ENTRYPOINT="${ENTRYPOINT:-scripts/sentence_trace_method/cross_verifier_finetune.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/evitrace_cross_verifier_finetune_v1}"
PREPARED_MANIFEST="${PREPARED_MANIFEST:-${OUTPUT_ROOT}/prepared/artifact_manifest.json}"
MODE="${MODE:-full}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
INFER_GPU_ID="${INFER_GPU_ID:-auto}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-true}"
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-20}"
GPU_READY_SAMPLES="${GPU_READY_SAMPLES:-3}"
BOOTSTRAP="${BOOTSTRAP:-10000}"
RANDOMIZATION="${RANDOMIZATION:-100000}"
DRY_RUN="${DRY_RUN:-false}"
FORCE="${FORCE:-false}"

mkdir -p "${OUTPUT_ROOT}/logs"
export PYTHONPATH="${PROJECT_ROOT}/src"

case "$MODE" in
  prepare|smoke|train|infer|analyze|full) ;;
  *)
    echo "Unsupported MODE=${MODE}; use prepare, smoke, train, infer, analyze, or full." >&2
    exit 2
    ;;
esac
if [[ "$FORCE" == "true" && "$MODE" =~ ^(smoke|train|full)$ ]]; then
  echo "FORCE=true is not supported for training; use latest_state resume or a new OUTPUT_ROOT." >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing PYTHON_BIN: $PYTHON_BIN" >&2
  exit 2
fi
GPU_ARRAY=()
if [[ "$MODE" =~ ^(smoke|train|full)$ ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
  if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
    echo "GPU_IDS must contain exactly four comma-separated GPU indices." >&2
    exit 2
  fi
  declare -A SEEN_GPU_IDS=()
  for gpu in "${GPU_ARRAY[@]}"; do
    gpu="${gpu// /}"
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
      echo "GPU_IDS must contain four non-negative integer indices." >&2
      exit 2
    fi
    if [[ -n "${SEEN_GPU_IDS[$gpu]:-}" ]]; then
      echo "GPU_IDS must contain four distinct indices; duplicate ${gpu}." >&2
      exit 2
    fi
    SEEN_GPU_IDS["$gpu"]=1
  done
fi

wait_for_gpu_group() {
  if [[ "$WAIT_FOR_GPUS" != "true" || "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  local consecutive=0
  local free_csv
  while (( consecutive < GPU_READY_SAMPLES )); do
    if ! free_csv="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)"; then
      consecutive=0
      echo "nvidia-smi unavailable; retrying in ${GPU_POLL_SECONDS}s." >&2
      sleep "$GPU_POLL_SECONDS"
      continue
    fi
    local ready=1
    local gpu
    for gpu in "${GPU_ARRAY[@]}"; do
      gpu="${gpu// /}"
      local free_mib
      free_mib="$(awk -F, -v wanted="$gpu" '
        {gsub(/ /, "", $1); gsub(/ /, "", $2)}
        $1 == wanted {print $2}
      ' <<< "$free_csv")"
      if [[ -z "$free_mib" || ! "$free_mib" =~ ^[0-9]+$ || "$free_mib" -lt "$MIN_FREE_MIB" ]]; then
        ready=0
        break
      fi
    done
    if (( ready == 1 )); then
      consecutive=$((consecutive + 1))
      echo "GPU group ready sample ${consecutive}/${GPU_READY_SAMPLES}."
    else
      consecutive=0
      echo "Waiting for GPUs ${GPU_IDS} to each have >=${MIN_FREE_MIB} MiB free."
    fi
    if (( consecutive < GPU_READY_SAMPLES )); then
      sleep "$GPU_POLL_SECONDS"
    fi
  done
}

prepare() {
  "$PYTHON_BIN" "$ENTRYPOINT" prepare \
    --output-dir "${OUTPUT_ROOT}/prepared" \
    --seed 20260724
}

train_one() {
  local model="$1"
  local assignment="$2"
  local seed="$3"
  shift 3
  local extra=("$@")
  local args=(
    "$PYTHON_BIN" "$ENTRYPOINT" train
    --prepared-manifest "$PREPARED_MANIFEST"
    --model-name "$model"
    --assignment "$assignment"
    --seed "$seed"
    --experiment-root "$OUTPUT_ROOT"
    --gpu-ids "$GPU_IDS"
    --python-bin "$PYTHON_BIN"
    --accelerate-bin "$ACCELERATE_BIN"
  )
  if [[ "$DRY_RUN" == "true" ]]; then
    args+=(--dry-run)
  fi
  args+=("${extra[@]}")
  "${args[@]}"
}

infer_one() {
  local model="$1"
  local assignment="$2"
  local seed="$3"
  shift 3
  local extra=("$@")
  local args=(
    "$PYTHON_BIN" "$ENTRYPOINT" infer
    --prepared-manifest "$PREPARED_MANIFEST"
    --model-name "$model"
    --assignment "$assignment"
    --seed "$seed"
    --experiment-root "$OUTPUT_ROOT"
    --gpu-id "$INFER_GPU_ID"
  )
  if [[ "$DRY_RUN" == "true" ]]; then
    args+=(--dry-run)
  fi
  if [[ "$FORCE" == "true" ]]; then
    args+=(--force)
  fi
  args+=("${extra[@]}")
  "${args[@]}"
}

run_smoke() {
  local model
  for model in qwen3 llama31; do
    wait_for_gpu_group
    train_one "$model" a 20260724 --smoke
    if [[ "$DRY_RUN" != "true" ]]; then
      infer_one "$model" a 20260724 --smoke --max-logical-rows 24
    fi
  done
}

run_formal_train() {
  local model assignment seed
  for model in qwen3 llama31; do
    for assignment in a b; do
      for seed in 20260724 20260725 20260726; do
        wait_for_gpu_group
        train_one "$model" "$assignment" "$seed"
      done
    done
  done
}

run_formal_infer() {
  local model assignment seed
  for model in qwen3 llama31; do
    for assignment in a b; do
      for seed in 20260724 20260725 20260726; do
        infer_one "$model" "$assignment" "$seed"
      done
    done
  done
}

run_analysis() {
  local args=(
    "$PYTHON_BIN" "$ENTRYPOINT" analyze
    --prepared-manifest "$PREPARED_MANIFEST"
    --output-dir "${OUTPUT_ROOT}/analysis"
    --bootstrap "$BOOTSTRAP"
    --randomization "$RANDOMIZATION"
    --seed 20260724
  )
  local model assignment seed run_id
  for model in qwen3 llama31; do
    for assignment in a b; do
      for seed in 20260724 20260725 20260726; do
        run_id="${model}__assignment_${assignment}__seed_${seed}"
        args+=(--result "${OUTPUT_ROOT}/inference/${run_id}/logical_results.jsonl")
      done
    done
  done
  "${args[@]}"
}

case "$MODE" in
  prepare)
    prepare
    ;;
  smoke)
    run_smoke
    ;;
  train)
    run_formal_train
    ;;
  infer)
    run_formal_infer
    ;;
  analyze)
    run_analysis
    ;;
  full)
    prepare
    run_smoke
    run_formal_train
    if [[ "$DRY_RUN" != "true" ]]; then
      run_formal_infer
      run_analysis
    fi
    ;;
esac
