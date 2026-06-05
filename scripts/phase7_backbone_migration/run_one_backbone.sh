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
TEXT_BACKBONE_ROOT="${TEXT_BACKBONE_ROOT:-outputs/cache/backbone_migration/text_backbones}"

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
  BACKBONE   Required. One of: qwen25_15b,qwen3_17b,qwen25_3b,qwen3_4b_2507,qwen3_8b,dsr1_qwen7b,llama31_8b,phi4_mini,gemma4_e4b,ministral3_8b
  MODE       dry_run, smoke, full, or infer_only. Default: dry_run
  FINETUNE   lora, fullft, or both. Default: lora
  ORDINAL_LOSS_ALPHA
             Optional. Enable label-token ordinal-aware loss with this alpha and add a case suffix.
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

fullft_deepspeed_config_for_backbone() {
  local backbone="$1"
  local model_variant="${2:-default}"
  local size_tenths="0"
  if [[ -n "${FULLFT_DEEPSPEED_CONFIG:-}" ]]; then
    printf "%s" "${FULLFT_DEEPSPEED_CONFIG}"
    return 0
  fi
  if [[ "${backbone}" == "gemma4_e4b" ]]; then
    printf "%s" "configs/deepspeed_zero3_bsz1_ga8_ultralowpeak.json"
    return 0
  fi
  if [[ "${backbone}" == "ministral3_8b" && "${model_variant}" != "text_only" ]]; then
    printf "%s" "configs/deepspeed_zero3_bsz1_ga8_lowpeak.json"
    return 0
  fi
  if [[ "${backbone}" == "gemma4_e4b" || "${backbone}" == "ministral3_8b" ]]; then
    printf "%s" "configs/deepspeed_zero3.json"
  else
    size_tenths="$(backbone_size_tenths "${backbone}")"
    if [[ "${size_tenths}" -ge 70 ]]; then
      printf "%s" "configs/deepspeed_zero3.json"
    else
      printf "%s" "configs/deepspeed_zero3_bsz2_ga4.json"
    fi
  fi
}

lora_deepspeed_config_for_backbone() {
  local backbone="$1"
  if [[ -n "${LORA_DEEPSPEED_CONFIG:-}" ]]; then
    printf "%s" "${LORA_DEEPSPEED_CONFIG}"
    return 0
  fi
  if [[ "${backbone}" == "ministral3_8b" ]]; then
    printf "%s" "configs/deepspeed_zero2_bsz1_ga8.json"
  fi
}

backbone_supports_text_only() {
  case "$1" in
    gemma4_e4b|ministral3_8b) return 0 ;;
    *) return 1 ;;
  esac
}

text_only_required_for_run() {
  local backbone="$1"
  local finetune="$2"
  local mode="${BACKBONE_TEXT_ONLY:-auto}"
  case "${mode}" in
    true|1|yes) return 0 ;;
    false|0|no) return 1 ;;
    auto)
      [[ "${finetune}" == "fullft" ]] && backbone_supports_text_only "${backbone}"
      return
      ;;
    *)
      echo "[backbone-migration] unsupported BACKBONE_TEXT_ONLY=${mode}" >&2
      return 2
      ;;
  esac
}

text_only_path_for_backbone() {
  local backbone="$1"
  local root="${TEXT_BACKBONE_ROOT}"
  if [[ ! "${root}" = /* ]]; then
    root="${PROJECT_ROOT}/${root}"
  fi
  printf "%s/%s" "${root%/}" "${backbone}"
}

text_only_export_family() {
  case "$1" in
    gemma4_e4b) printf "%s" "gemma4" ;;
    ministral3_8b) printf "%s" "mistral3" ;;
    *) return 1 ;;
  esac
}

resolve_model_path_for_run() {
  local backbone="$1"
  local finetune="$2"
  local source_path="$3"
  local text_path=""
  local family=""
  local text_only_status=0
  text_only_required_for_run "${backbone}" "${finetune}" || text_only_status=$?
  if [[ "${text_only_status}" -eq 2 ]]; then
    return 1
  fi
  if [[ "${text_only_status}" -ne 0 ]]; then
    printf "%s" "${source_path}"
    return 0
  fi
  text_path="$(text_only_path_for_backbone "${backbone}")"
  if [[ -f "${text_path}/config.json" ]]; then
    printf "%s" "${text_path}"
    return 0
  fi
  family="$(text_only_export_family "${backbone}")"
  echo "[backbone-migration] missing text-only checkpoint for ${backbone}: ${text_path}" >&2
  echo "[backbone-migration] build it once with:" >&2
  echo "  PYTHONPATH=src /data/liaozijie/conda/accelerate-fc-gemma4/bin/python scripts/phase7_backbone_migration/export_text_only_backbone.py \\" >&2
  echo "    --source ${source_path} \\" >&2
  echo "    --output ${text_path} \\" >&2
  echo "    --family ${family}" >&2
  echo "[backbone-migration] set BACKBONE_TEXT_ONLY=false to force the original multimodal checkpoint." >&2
  return 1
}

max_length_for_backbone() {
  local backbone="$1"
  local finetune="$2"
  if [[ -n "${BACKBONE_MAX_LENGTH:-}" ]]; then
    printf "%s" "${BACKBONE_MAX_LENGTH}"
    return 0
  fi
  if [[ "${backbone}" == "gemma4_e4b" && "${finetune}" == "fullft" && -n "${GEMMA4_FULLFT_MAX_LENGTH:-}" ]]; then
    printf "%s" "${GEMMA4_FULLFT_MAX_LENGTH}"
  fi
}

ordinal_loss_case_suffix() {
  local alpha="${ORDINAL_LOSS_ALPHA:-}"
  if [[ -z "${alpha}" ]]; then
    return 0
  fi
  python -c '
import sys

alpha = float(sys.argv[1])
if alpha < 0:
    raise SystemExit("ORDINAL_LOSS_ALPHA must be non-negative")
compact = f"{alpha:g}".replace(".", "")
print(f"_ord_abs_a{compact}")
' "${alpha}"
}

prepare_config() {
  local backbone="$1"
  local finetune="$2"
  local case_name="$3"
  local model_path="$4"
  local size_b="$5"
  local deepspeed_config="${6:-}"
  local max_length="${7:-}"
  local model_variant="${8:-default}"
  local base_experiment=""
  local extra_args=()

  if [[ "${finetune}" == "lora" ]]; then
    base_experiment="${LORA_BASE_EXPERIMENT}"
  else
    base_experiment="${FULLFT_BASE_EXPERIMENT}"
  fi
  if [[ -n "${deepspeed_config}" ]]; then
    extra_args+=(--deepspeed-config "${deepspeed_config}")
  fi
  if [[ -n "${max_length}" ]]; then
    extra_args+=(--max-length "${max_length}")
  fi
  if [[ -n "${ORDINAL_LOSS_ALPHA:-}" ]]; then
    extra_args+=(--ordinal-loss-alpha "${ORDINAL_LOSS_ALPHA}")
  fi

  python "${SCRIPT_DIR}/prepare_backbone_config.py" \
    --backbone "${backbone}" \
    --model-path "${model_path}" \
    --size-b "${size_b}" \
    --finetune "${finetune}" \
    --case-name "${case_name}" \
    --base-experiment "${base_experiment}" \
    --model-variant "${model_variant}" \
    --output-root "${CONFIG_CACHE_ROOT}" \
    "${extra_args[@]}" \
    --print-path
}

run_finetune() {
  local finetune="$1"
  local source_model_path="$2"
  local size_b="$3"
  local model_path=""
  local case_name=""
  local config_path=""
  local run_lora=false
  local run_fullft=false
  local lora_deepspeed_config=""
  local fullft_deepspeed_config=""
  local train_deepspeed_config=""
  local max_length_config=""
  local model_variant="default"
  local ordinal_suffix=""
  local suffix=""

  model_path="$(resolve_model_path_for_run "${BACKBONE}" "${finetune}" "${source_model_path}")"
  if [[ "${model_path}" != "${source_model_path}" ]]; then
    model_variant="text_only"
  fi
  case_name="$(backbone_case_name "${BACKBONE}" "${finetune}")"
  if [[ "${finetune}" == "fullft" && "${BACKBONE}" == "ministral3_8b" && "${model_variant}" != "text_only" ]]; then
    case_name="${case_name}_mm_text_effective"
  fi
  ordinal_suffix="$(ordinal_loss_case_suffix)"
  if [[ -n "${ordinal_suffix}" ]]; then
    case_name="${case_name}${ordinal_suffix}"
  fi
  if [[ "${MODE}" == "smoke" ]]; then
    suffix="_smoke${SAMPLE_LIMIT_EFFECTIVE}"
    case_name="${case_name}${suffix}"
  fi
  if [[ "${finetune}" == "fullft" ]]; then
    fullft_deepspeed_config="$(fullft_deepspeed_config_for_backbone "${BACKBONE}" "${model_variant}")"
    train_deepspeed_config="${fullft_deepspeed_config}"
  else
    lora_deepspeed_config="$(lora_deepspeed_config_for_backbone "${BACKBONE}")"
    train_deepspeed_config="${lora_deepspeed_config}"
  fi
  max_length_config="$(max_length_for_backbone "${BACKBONE}" "${finetune}")"
  config_path="$(prepare_config "${BACKBONE}" "${finetune}" "${case_name}" "${model_path}" "${size_b}" "${train_deepspeed_config}" "${max_length_config}" "${model_variant}")"

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
  if [[ "${model_path}" != "${source_model_path}" ]]; then
    echo "[backbone-migration] source_model=${source_model_path}"
    echo "[backbone-migration] model_variant=text_only"
  elif [[ "${finetune}" == "fullft" && "${BACKBONE}" == "ministral3_8b" ]]; then
    echo "[backbone-migration] model_variant=multimodal_text_effective"
    echo "[backbone-migration] freeze_prefixes=model.vision_tower,model.multi_modal_projector"
  fi
  echo "[backbone-migration] case=${case_name}"
  echo "[backbone-migration] config=${config_path}"
  if [[ "${finetune}" == "fullft" ]] && backbone_is_large_fullft "${BACKBONE}"; then
    echo "[backbone-migration] fullft large-model policy: per_device_train_batch_size=1 gradient_accumulation_steps=8"
  fi
  if [[ "${finetune}" == "fullft" ]]; then
    echo "[backbone-migration] fullft deepspeed_config=${fullft_deepspeed_config}"
  fi
  if [[ "${finetune}" == "lora" && -n "${lora_deepspeed_config}" ]]; then
    echo "[backbone-migration] lora deepspeed_config=${lora_deepspeed_config}"
  fi
  if [[ -n "${max_length_config}" ]]; then
    echo "[backbone-migration] max_length=${max_length_config}"
  fi
  if [[ -n "${ORDINAL_LOSS_ALPHA:-}" ]]; then
    echo "[backbone-migration] ordinal_loss_alpha=${ORDINAL_LOSS_ALPHA}"
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
    DEEPSPEED_CONFIG="${train_deepspeed_config}" \
    FULLFT_DEEPSPEED_CONFIG="${fullft_deepspeed_config}" \
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

if ! python "${SCRIPT_DIR}/check_compat.py" --backbone "${BACKBONE}"; then
  if [[ "${DRY_RUN_EFFECTIVE}" == "true" ]]; then
    echo "[backbone-migration] dry-run continues despite compatibility check failure" >&2
  else
    exit 1
  fi
fi

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
