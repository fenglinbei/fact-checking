#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc-gemma4/bin/python" ]]; then
    export PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-gemma4/bin/python"
  elif [[ -x "/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python" ]]; then
    export PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python"
  else
    export PYTHON_BIN="python"
  fi
fi

export MREC_POLICY_CONFIG="${MREC_POLICY_CONFIG:-configs/experiment/mrec_v0.2/scifact_atom_union_structure_only_fullpool_minmax9_9.yaml}"
export SFT_TRAIN_MODULE="${SFT_TRAIN_MODULE:-sft.hami_cuda_bootstrap}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29651}"
export MREC_RUNTIME_CACHE_ROOT="${MREC_RUNTIME_CACHE_ROOT:-${ROOT_DIR}/outputs/cache/runtime/scifact_structure_only}"
export CUDA_DEVICE_MEMORY_SHARED_CACHE="${CUDA_DEVICE_MEMORY_SHARED_CACHE:-/tmp/lzj_scifact_structure_only_hami_$$/cudevshr.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/lzj_scifact_structure_only_hami_$$/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/lzj_scifact_structure_only_hami_$$/torchinductor}"
mkdir -p "$(dirname "$CUDA_DEVICE_MEMORY_SHARED_CACHE")" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
MODE="${MODE:-full}"
DRY_RUN="${DRY_RUN:-false}"

eval "$("$PYTHON_BIN" scripts/sentence_trace_method/mrec_policy_config.py --config "$MREC_POLICY_CONFIG")"

require_path() {
  local path="$1"
  local label="$2"
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  if [[ ! -e "$path" ]]; then
    printf '[scifact-structure-only] missing %s: %s\n' "$label" "$path" >&2
    exit 2
  fi
}

validate_weight_contract() {
  local manifest="${MREC_WEIGHT_OUTPUT_DIR}/manifest.json"
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  require_path "$WEIGHT_FILE" "structure-only weight file"
  require_path "$manifest" "structure-only weight manifest"
  "$PYTHON_BIN" -c '
import json, pathlib, sys
manifest_path = pathlib.Path(sys.argv[1])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if payload.get("training_supervision") != "structure_only":
    raise SystemExit(f"invalid training_supervision in {manifest_path}")
contract = payload.get("supervision_contract") or {}
for key in ("oracle_read_row_count", "gold_label_read_count", "teacher_read_count", "utility_read_count", "reward_read_count"):
    if int(contract.get(key, -1)) != 0:
        raise SystemExit(f"invalid {key} in {manifest_path}: {contract.get(key)!r}")
params = payload.get("params") or {}
if params.get("map_ablation_mode") != "full":
    raise SystemExit(f"invalid map_ablation_mode in {manifest_path}")
' "$manifest"
}

train_structure_only_weights_if_needed() {
  case "$MODE" in
    build|full)
      if [[ -f "$WEIGHT_FILE" ]]; then
        printf '[scifact-structure-only] reuse strict structure-only weights: %s\n' "$WEIGHT_FILE"
      else
        env \
          PYTHON_BIN="$PYTHON_BIN" \
          SOURCE_FEATURE_ROOT="$SOURCE_FEATURE_ROOT" \
          OUTPUT_DIR="$MREC_WEIGHT_OUTPUT_DIR" \
          CANDIDATE_TOP_N="$MREC_WEIGHT_CANDIDATE_TOP_N" \
          ROLLOUT_STEPS="$MREC_WEIGHT_ROLLOUT_STEPS" \
          EPOCHS="$MREC_WEIGHT_EPOCHS" \
          LEARNING_RATE="$MREC_WEIGHT_LEARNING_RATE" \
          MAP_ABLATION_MODE=full \
          DRY_RUN="$DRY_RUN" \
          bash scripts/phase5_selectors/run/run_liar_raw_mrec_structure_only_weights.sh
      fi
      ;;
    check|train|eval|export)
      ;;
    *)
      printf 'Unsupported MODE=%s. Use check, build, train, eval, full, or export.\n' "$MODE" >&2
      exit 2
      ;;
  esac
  validate_weight_contract
}

printf '[scifact-structure-only] config=%s mode=%s weights=%s traces=%s splits=%s device(weights)=CPU train_module=%s port=%s hami_cache=%s\n' \
  "$MREC_POLICY_CONFIG" "$MODE" "$WEIGHT_FILE" "$TRACE_ROOT" "$MREC_SPLITS" \
  "$SFT_TRAIN_MODULE" "$MAIN_PROCESS_PORT" "$CUDA_DEVICE_MEMORY_SHARED_CACHE"

train_structure_only_weights_if_needed

# The generic wrapper must never fall back to the legacy proxy trainer for this run.
exec env \
  MREC_POLICY_CONFIG="$MREC_POLICY_CONFIG" \
  MODE="$MODE" \
  DRY_RUN="$DRY_RUN" \
  FORCE_WEIGHT_TRAIN=false \
  MREC_AUTO_TRAIN_WEIGHTS=false \
  bash scripts/phase13_scifact/05_train_eval_scifact_atom_union_fullpool_lora.sh
