#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=scripts/phase7_backbone_migration/backbone_cases.sh
source "${SCRIPT_DIR}/backbone_cases.sh"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

BACKBONE="${BACKBONE:-}"
MODE="${MODE:-dry_run}"
FINETUNE="${FINETUNE:-lora}"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_trace_verifier/rawfc_v0_6c_eval25_backbone}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/rawfc_v0_6c_eval25_backbone}"
CONFIG_CACHE_ROOT="${CONFIG_CACHE_ROOT:-outputs/cache/backbone_migration/configs}"

LORA_BASE_EXPERIMENT="${LORA_BASE_EXPERIMENT:-v0_6c_rawfc3_rule_step_adaptive5_10_eval25}"
FULLFT_BASE_EXPERIMENT="${FULLFT_BASE_EXPERIMENT:-v0_6c_rawfc3_rule_step_adaptive5_10_eval25_fullft}"

TRACE_ROOT="${TRACE_ROOT:-outputs/selectors/evidence_chain_graph/rawfc_v0_6c_adaptive5_10}"
TRAIN_TRACE="${TRAIN_TRACE:-${TRACE_ROOT}_train/selection_trace_train.jsonl}"
VAL_TRACE="${VAL_TRACE:-${TRACE_ROOT}_val/selection_trace_val.jsonl}"
TEST_TRACE="${TEST_TRACE:-${TRACE_ROOT}_test/selection_trace_test.jsonl}"

usage() {
  cat <<'EOF'
Usage:
  BACKBONE=qwen25_3b MODE=dry_run FINETUNE=lora \
    bash scripts/phase7_backbone_migration/run_one_backbone.sh

Environment:
  BACKBONE   Required. One of: qwen25_15b,qwen3_17b,qwen25_3b,qwen3_4b_2507,qwen3_8b,dsr1_qwen7b
  MODE       dry_run, smoke, full, or infer_only. Default: dry_run
  FINETUNE   lora, fullft, or both. Default: lora
EOF
}

split_csv() {
  local raw="$1"
  local -n out_array="$2"
  local items=()
  local item=""
  IFS=',' read -r -a items <<< "${raw}"
  out_array=()
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      out_array+=("${item}")
    fi
  done
}

check_trace_path() {
  local path="$1"
  local label="$2"
  if [[ -f "${path}" ]]; then
    return 0
  fi
  if [[ "${DRY_RUN_EFFECTIVE}" == "true" ]]; then
    echo "[backbone-migration] dry-run missing ${label}: ${path}" >&2
    return 0
  fi
  echo "[backbone-migration] missing ${label}: ${path}" >&2
  return 1
}

read_trace_fingerprint() {
  local path="$1"
  python -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit(0)
with path.open(encoding="utf-8") as handle:
    line = handle.readline()
row = json.loads(line)
trace = row.get("selection_trace") if isinstance(row, dict) else None
if not isinstance(trace, dict):
    trace = row if isinstance(row, dict) else {}
metadata = trace.get("candidate_pool_metadata")
if not isinstance(metadata, dict):
    metadata = {}
print(trace.get("fingerprint") or metadata.get("chunk_mmr_fingerprint") or "")
' "${path}"
}

prepare_config() {
  local backbone="$1"
  local finetune="$2"
  local case_name="$3"
  local model_path="$4"
  local size_b="$5"
  local base_experiment=""

  if [[ "${finetune}" == "lora" ]]; then
    base_experiment="${LORA_BASE_EXPERIMENT}"
  else
    base_experiment="${FULLFT_BASE_EXPERIMENT}"
  fi

  python "${SCRIPT_DIR}/prepare_backbone_config.py" \
    --backbone "${backbone}" \
    --model-path "${model_path}" \
    --size-b "${size_b}" \
    --finetune "${finetune}" \
    --case-name "${case_name}" \
    --base-experiment "${base_experiment}" \
    --output-root "${CONFIG_CACHE_ROOT}" \
    --print-path
}

run_finetune() {
  local finetune="$1"
  local model_path="$2"
  local size_b="$3"
  local case_name=""
  local config_path=""
  local run_lora=false
  local run_fullft=false
  local suffix=""

  case_name="$(backbone_case_name "${BACKBONE}" "${finetune}")"
  if [[ "${MODE}" == "smoke" ]]; then
    suffix="_smoke${SAMPLE_LIMIT_EFFECTIVE}"
    case_name="${case_name}${suffix}"
  fi
  config_path="$(prepare_config "${BACKBONE}" "${finetune}" "${case_name}" "${model_path}" "${size_b}")"

  if [[ "${finetune}" == "lora" ]]; then
    run_lora=true
  elif [[ "${finetune}" == "fullft" ]]; then
    run_fullft=true
  else
    echo "[backbone-migration] unsupported finetune=${finetune}" >&2
    return 1
  fi

  echo "[backbone-migration] backbone=${BACKBONE} finetune=${finetune} mode=${MODE}"
  echo "[backbone-migration] model=${model_path}"
  echo "[backbone-migration] case=${case_name}"
  echo "[backbone-migration] config=${config_path}"
  if [[ "${finetune}" == "fullft" ]] && backbone_is_large_fullft "${BACKBONE}"; then
    echo "[backbone-migration] fullft large-model policy: per_device_train_batch_size=1 gradient_accumulation_steps=8"
  fi

  env \
    RUN_CACHE_BUILD=false \
    RUN_QD=false \
    RUN_EVIDENCE_MAP=false \
    RUN_GRAPH_BUILD=false \
    RUN_LORA="${run_lora}" \
    RUN_FULLFT="${run_fullft}" \
    RUN_TRAIN="${RUN_TRAIN_EFFECTIVE}" \
    RUN_INFER="${RUN_INFER_EFFECTIVE}" \
    RUN_API_INFER="${RUN_API_INFER:-false}" \
    RUN_LABEL_TOKEN_INFER="${RUN_LABEL_TOKEN_INFER:-true}" \
    DRY_RUN="${DRY_RUN_EFFECTIVE}" \
    SAMPLE_LIMIT="${SAMPLE_LIMIT_EFFECTIVE}" \
    LORA_CONFIG="${config_path}" \
    FULLFT_CONFIG="${config_path}" \
    LORA_EXPERIMENT="${LORA_BASE_EXPERIMENT}" \
    FULLFT_EXPERIMENT="${FULLFT_BASE_EXPERIMENT}" \
    LORA_CASE_NAME="${case_name}" \
    FULLFT_CASE_NAME="${case_name}" \
    PROMPT_MODEL_NAME_OR_PATH="${model_path}" \
    TRAIN_MODEL_NAME_OR_PATH="${model_path}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    RUN_ROOT="${RUN_ROOT}" \
    TRAIN_TRACE="${TRAIN_TRACE}" \
    VAL_TRACE="${VAL_TRACE}" \
    TEST_TRACE="${TEST_TRACE}" \
    CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT_EFFECTIVE}" \
    bash scripts/phase5_selectors/run/run_rawfc_v0_6c_rule_step_adaptive5_10_eval25_all_pipelines.sh
}

if [[ -z "${BACKBONE}" || "${BACKBONE}" == "-h" || "${BACKBONE}" == "--help" ]]; then
  usage
  exit 2
fi
require_backbone "${BACKBONE}"

case "${MODE}" in
  dry_run)
    DRY_RUN_EFFECTIVE=true
    SAMPLE_LIMIT_EFFECTIVE="${SAMPLE_LIMIT:-0}"
    RUN_TRAIN_EFFECTIVE="${RUN_TRAIN:-true}"
    RUN_INFER_EFFECTIVE="${RUN_INFER:-true}"
    ;;
  smoke)
    DRY_RUN_EFFECTIVE=false
    SAMPLE_LIMIT_EFFECTIVE="${SAMPLE_LIMIT:-32}"
    RUN_TRAIN_EFFECTIVE="${RUN_TRAIN:-true}"
    RUN_INFER_EFFECTIVE="${RUN_INFER:-true}"
    ;;
  full)
    DRY_RUN_EFFECTIVE=false
    SAMPLE_LIMIT_EFFECTIVE="${SAMPLE_LIMIT:-0}"
    RUN_TRAIN_EFFECTIVE="${RUN_TRAIN:-true}"
    RUN_INFER_EFFECTIVE="${RUN_INFER:-true}"
    ;;
  infer_only)
    DRY_RUN_EFFECTIVE=false
    SAMPLE_LIMIT_EFFECTIVE="${SAMPLE_LIMIT:-0}"
    RUN_TRAIN_EFFECTIVE=false
    RUN_INFER_EFFECTIVE=true
    ;;
  *)
    echo "[backbone-migration] unsupported MODE=${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac

check_trace_path "${TRAIN_TRACE}" "train trace"
check_trace_path "${VAL_TRACE}" "val trace"
check_trace_path "${TEST_TRACE}" "test trace"

if [[ -n "${CHUNK_MMR_FINGERPRINT:-}" ]]; then
  CHUNK_MMR_FINGERPRINT_EFFECTIVE="${CHUNK_MMR_FINGERPRINT}"
else
  CHUNK_MMR_FINGERPRINT_EFFECTIVE="$(read_trace_fingerprint "${TRAIN_TRACE}")"
  if [[ -z "${CHUNK_MMR_FINGERPRINT_EFFECTIVE}" ]]; then
    if [[ "${DRY_RUN_EFFECTIVE}" == "true" ]]; then
      CHUNK_MMR_FINGERPRINT_EFFECTIVE="dry_run_missing_trace"
    else
      echo "[backbone-migration] failed to read chunk MMR fingerprint from ${TRAIN_TRACE}" >&2
      exit 1
    fi
  fi
fi

model_path="$(backbone_path "${BACKBONE}")"
size_b="$(backbone_size_b "${BACKBONE}")"

python "${SCRIPT_DIR}/check_compat.py" --backbone "${BACKBONE}"

finetunes=()
case "${FINETUNE}" in
  lora|fullft)
    finetunes=("${FINETUNE}")
    ;;
  both)
    finetunes=(lora fullft)
    ;;
  *)
    echo "[backbone-migration] unsupported FINETUNE=${FINETUNE}" >&2
    usage >&2
    exit 2
    ;;
esac

for finetune in "${finetunes[@]}"; do
  run_finetune "${finetune}" "${model_path}" "${size_b}"
done
