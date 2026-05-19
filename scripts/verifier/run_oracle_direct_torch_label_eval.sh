#!/usr/bin/env bash
# Run verifier evaluation through the Hugging Face/PEFT torch-forward
# label-token path. This is intended to compare against vLLM prompt_logprobs
# scoring without changing prompts or evidence.
#
# Examples:
#   CHECKPOINTS=best bash scripts/verifier/run_oracle_direct_torch_label_eval.sh
#   CHECKPOINTS=best,checkpoint-600 PER_DEVICE_EVAL_BATCH_SIZE=2 bash scripts/verifier/run_oracle_direct_torch_label_eval.sh
#   CONFIG_PATH=outputs/oracle_direct_verifier/stage2_sentence_order_sensitivity/oracle/train.resolved.yaml \
#     OUTPUT_ROOT=outputs/runs/b3_oracle_direct_torch_forward_eval_order \
#     bash scripts/verifier/run_oracle_direct_torch_label_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DIRECT_VERIFIER_RUN_DIR="${DIRECT_VERIFIER_RUN_DIR:-outputs/oracle_direct_verifier/stage2_sentence/train/b3_oracle_sentence_direct_verifier_1024_20260519-200709}"
CONFIG_PATH="${CONFIG_PATH:-outputs/oracle_direct_verifier/stage2_sentence/train.resolved.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/runs/b3_oracle_direct_torch_forward_eval}"
CHECKPOINTS="${CHECKPOINTS:-best}"
SPLIT="${SPLIT:-val}"

PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
STRICT_LABEL_TOKEN_META="${STRICT_LABEL_TOKEN_META:-true}"
DEDUPE_SAMPLE_IDX="${DEDUPE_SAMPLE_IDX:-true}"
MERGE_LORA_FOR_FORWARD="${MERGE_LORA_FOR_FORWARD:-false}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-}"
DRY_RUN="${DRY_RUN:-false}"

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

slugify() {
  local raw="$1"
  raw="${raw//\//_}"
  raw="${raw// /_}"
  raw="${raw//,/}"
  raw="${raw//:/_}"
  printf "%s" "${raw}"
}

require_path() {
  local path="$1"
  local message="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[torch-label-eval] missing ${message}: ${path}" >&2
    exit 1
  fi
}

bool_flag() {
  local name="$1"
  local value="$2"
  if [[ "${value}" == "true" ]]; then
    printf -- "--%s" "${name}"
  else
    printf -- "--no-%s" "${name}"
  fi
}

run_checkpoint() {
  local checkpoint="$1"
  local checkpoint_slug=""
  checkpoint_slug="$(slugify "${checkpoint}")"
  local output_dir="${OUTPUT_ROOT}/${SPLIT}/${checkpoint_slug}"
  local cmd=(
    python scripts/verifier/eval_label_token_torch_forward.py
    --run-dir "${DIRECT_VERIFIER_RUN_DIR}"
    --checkpoint "${checkpoint}"
    --split "${SPLIT}"
    --config "${CONFIG_PATH}"
    --output-dir "${output_dir}"
    --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
    --torch-dtype "${TORCH_DTYPE}"
    --attn-implementation "${ATTN_IMPLEMENTATION}"
    "$(bool_flag strict-label-token-meta "${STRICT_LABEL_TOKEN_META}")"
    "$(bool_flag dedupe-sample-idx "${DEDUPE_SAMPLE_IDX}")"
  )

  if [[ "${MERGE_LORA_FOR_FORWARD}" == "true" ]]; then
    cmd+=(--merge-lora-for-forward)
  fi
  if [[ -n "${BASE_MODEL_NAME_OR_PATH}" ]]; then
    cmd+=(--base-model-name-or-path "${BASE_MODEL_NAME_OR_PATH}")
  fi

  echo "[torch-label-eval] checkpoint=${checkpoint} split=${SPLIT} output=${output_dir}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf '[torch-label-eval] dry-run command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  "${cmd[@]}"
}

main() {
  local checkpoints=()
  local checkpoint=""
  split_csv "${CHECKPOINTS}" checkpoints
  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "[torch-label-eval] CHECKPOINTS is empty" >&2
    exit 1
  fi

  require_path "${DIRECT_VERIFIER_RUN_DIR}" "direct verifier train run dir"
  require_path "${CONFIG_PATH}" "resolved config"
  for checkpoint in "${checkpoints[@]}"; do
    require_path "${DIRECT_VERIFIER_RUN_DIR}/${checkpoint}" "checkpoint ${checkpoint}"
  done

  echo "[torch-label-eval] direct_verifier_run_dir=${DIRECT_VERIFIER_RUN_DIR}"
  echo "[torch-label-eval] config_path=${CONFIG_PATH}"
  echo "[torch-label-eval] checkpoints=${checkpoints[*]}"
  echo "[torch-label-eval] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  for checkpoint in "${checkpoints[@]}"; do
    run_checkpoint "${checkpoint}"
  done
}

main "$@"
