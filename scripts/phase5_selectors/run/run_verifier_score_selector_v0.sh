#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

DIRECT_VERIFIER_RUN_DIR="${DIRECT_VERIFIER_RUN_DIR:-outputs/oracle_direct_verifier/stage2_sentence/train/b3_oracle_sentence_direct_verifier_1024_20260519-200709}"
VERIFIER_CHECKPOINT="${VERIFIER_CHECKPOINT:-best}"
LABEL_PREFIX="${LABEL_PREFIX:-Label:}"
ORACLE_RESULTS="${ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/verifier_score_selector/b3_oracle_direct_v0/val_greedy_stepwise_top5}"
SPLIT="${SPLIT:-val}"

DIAG_SAMPLE_LIMIT="${DIAG_SAMPLE_LIMIT:-512}"
SELECTION_MODE="${SELECTION_MODE:-both}"
SCORE_MODES="${SCORE_MODES:-pred_margin,entropy_neg,base_pred_margin,gold_margin}"
CLAIM_BATCH_SIZE="${CLAIM_BATCH_SIZE:-8}"
RESUME="${RESUME:-true}"
FSYNC_CACHE="${FSYNC_CACHE:-false}"
FINALIZE_ONLY="${FINALIZE_ONLY:-false}"
NO_PROGRESS="${NO_PROGRESS:-false}"

PROMPT_MODEL_NAME_OR_PATH="${PROMPT_MODEL_NAME_OR_PATH:-}"
PROMPT_MAX_LENGTH="${PROMPT_MAX_LENGTH:-1024}"
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-}"
VLLM_TOKENIZER_PATH="${VLLM_TOKENIZER_PATH:-}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_PROMPT_BATCH_SIZE="${VLLM_PROMPT_BATCH_SIZE:-6000}"
GPU_DEVICES="${GPU_DEVICES:-}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[verifier-score-selector] missing ${label}: ${path}" >&2
    exit 1
  fi
}

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "True" ]]
}

cuda_visible() {
  python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PY
}

if [[ -n "${GPU_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
fi

if ! truthy "${FINALIZE_ONLY}"; then
  if ! cuda_visible; then
    if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "${HOME}/miniconda3/etc/profile.d/conda.sh"
      conda activate cppo || true
    fi
  fi
  if ! cuda_visible; then
    echo "[verifier-score-selector] CUDA is not visible, even after trying conda activate cppo; refusing to fall back to API." >&2
    exit 1
  fi
fi

if [[ "${VERIFIER_CHECKPOINT}" == "final" ]]; then
  echo "[verifier-score-selector] VERIFIER_CHECKPOINT=final is not allowed; use best or a synced checkpoint." >&2
  exit 1
fi

VERIFIER_CHECKPOINT_DIR="${DIRECT_VERIFIER_RUN_DIR}/${VERIFIER_CHECKPOINT}"
require_file "${VERIFIER_CHECKPOINT_DIR}/adapter_config.json" "verifier adapter config"
require_file "${VERIFIER_CHECKPOINT_DIR}/adapter_model.safetensors" "verifier adapter weights"
require_file "${VERIFIER_CHECKPOINT_DIR}/tokenizer_config.json" "verifier tokenizer config"
require_file "${DIRECT_VERIFIER_RUN_DIR}/label_token_ce_meta.json" "label-token metadata"
require_file "${ORACLE_RESULTS}" "oracle results"

RESUME_ARGS=(--resume)
if ! truthy "${RESUME}"; then
  RESUME_ARGS=(--no-resume)
fi

FSYNC_ARGS=()
if truthy "${FSYNC_CACHE}"; then
  FSYNC_ARGS=(--fsync-cache)
fi

FINALIZE_ARGS=()
if truthy "${FINALIZE_ONLY}"; then
  FINALIZE_ARGS=(--finalize-only)
fi

PROGRESS_ARGS=()
if truthy "${NO_PROGRESS}"; then
  PROGRESS_ARGS=(--no-progress)
fi

SAMPLE_ARGS=()
if [[ -n "${DIAG_SAMPLE_LIMIT}" ]]; then
  SAMPLE_ARGS=(--sample-limit "${DIAG_SAMPLE_LIMIT}")
fi

echo "[verifier-score-selector] verifier run      : ${DIRECT_VERIFIER_RUN_DIR}"
echo "[verifier-score-selector] checkpoint        : ${VERIFIER_CHECKPOINT}"
echo "[verifier-score-selector] oracle results    : ${ORACLE_RESULTS}"
echo "[verifier-score-selector] output dir        : ${OUTPUT_DIR}"
echo "[verifier-score-selector] selection mode    : ${SELECTION_MODE}"
echo "[verifier-score-selector] score modes       : ${SCORE_MODES}"
echo "[verifier-score-selector] claim batch size  : ${CLAIM_BATCH_SIZE}"
echo "[verifier-score-selector] resume/finalize   : ${RESUME} / ${FINALIZE_ONLY}"

PYTHONPATH=src python scripts/phase5_selectors/eval/eval_verifier_score_selector.py \
  --oracle-results "${ORACLE_RESULTS}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --selection-mode "${SELECTION_MODE}" \
  --score-modes "${SCORE_MODES}" \
  --claim-batch-size "${CLAIM_BATCH_SIZE}" \
  --direct-verifier-run-dir "${DIRECT_VERIFIER_RUN_DIR}" \
  --verifier-checkpoint "${VERIFIER_CHECKPOINT}" \
  --label-prefix "${LABEL_PREFIX}" \
  ${PROMPT_MODEL_NAME_OR_PATH:+--prompt-model-name-or-path "${PROMPT_MODEL_NAME_OR_PATH}"} \
  --prompt-max-length "${PROMPT_MAX_LENGTH}" \
  ${VLLM_MODEL_PATH:+--vllm-model-path "${VLLM_MODEL_PATH}"} \
  ${VLLM_TOKENIZER_PATH:+--vllm-tokenizer-path "${VLLM_TOKENIZER_PATH}"} \
  --vllm-tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm-dtype "${VLLM_DTYPE}" \
  ${VLLM_MAX_MODEL_LEN:+--vllm-max-model-len "${VLLM_MAX_MODEL_LEN}"} \
  --vllm-prompt-batch-size "${VLLM_PROMPT_BATCH_SIZE}" \
  "${RESUME_ARGS[@]}" \
  "${FSYNC_ARGS[@]}" \
  "${FINALIZE_ARGS[@]}" \
  "${PROGRESS_ARGS[@]}" \
  "${SAMPLE_ARGS[@]}"
