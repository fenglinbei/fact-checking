#!/usr/bin/env bash
# Build data, train, and selection-only evaluate the Qwen LLM action selector.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"

MODEL_NAME="${MODEL_NAME:-/data/models/Qwen2.5-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/llm_action_selector/qwen25_3b_vig_soft}"
DATA_DIR="${DATA_DIR:-${OUTPUT_DIR}/data}"
TRAIN_ORACLE_RESULTS="${TRAIN_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_train_stepscores/oracle_results_train.jsonl}"
VAL_ORACLE_RESULTS="${VAL_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
TRAIN_VIG_CACHE="${TRAIN_VIG_CACHE:-outputs/selectors/vig_utility/saved_step_train/vig_records_train.jsonl}"
VAL_VIG_CACHE="${VAL_VIG_CACHE:-outputs/selectors/vig_utility/saved_step_val/vig_records_val.jsonl}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/action_samples_train.jsonl}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/action_samples_val.jsonl}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/evals/val}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
EVAL_LOG_FILE="${EVAL_LOG_FILE:-${OUTPUT_DIR}/logs/eval_llm_action_selector.log}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_CANDIDATE_CHARS="${MAX_CANDIDATE_CHARS:-180}"
TRAIN_SAMPLE_LIMIT="${TRAIN_SAMPLE_LIMIT:-}"
VAL_SAMPLE_LIMIT="${VAL_SAMPLE_LIMIT:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz8_ga1.json}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
CHOICE_BATCH_SIZE="${CHOICE_BATCH_SIZE:-64}"
SCORE_MODE="${SCORE_MODE:-action_token}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1.0e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SOFT_LOSS_WEIGHT="${SOFT_LOSS_WEIGHT:-0.3}"
SOFT_TAU="${SOFT_TAU:-0.2}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT:-}"
NO_PROGRESS="${NO_PROGRESS:-false}"
RUN_BUILD_DATA="${RUN_BUILD_DATA:-true}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
TRAIN_SELECTION_EVAL_MODE="${TRAIN_SELECTION_EVAL_MODE:-best}"
SELECTION_EVAL_SAMPLE_LIMIT="${SELECTION_EVAL_SAMPLE_LIMIT:-128}"
SELECTION_EVAL_TOP_K="${SELECTION_EVAL_TOP_K:-5}"
SELECTION_EVAL_OUTPUT_DIR="${SELECTION_EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/evals/during_train}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-llm-action-selector}"
SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-$(basename "${OUTPUT_DIR}")}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"
SWANLAB_MODE="${SWANLAB_MODE:-}"
SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-}"
SWANLAB_TAGS="${SWANLAB_TAGS:-selector,llm_action,vig_soft}"
SWANLAB_DESCRIPTION="${SWANLAB_DESCRIPTION:-}"
SWANLAB_DISABLED="${SWANLAB_DISABLED:-0}"
REFERENCE_RANKER_METRICS="${REFERENCE_RANKER_METRICS:-outputs/selectors/vig_utility/saved_step_train_to_val/ranker_eval/selection_metrics.json}"
REFERENCE_SEQUENTIAL_METRICS="${REFERENCE_SEQUENTIAL_METRICS:-outputs/selectors/stage2_sentence_sequential/deberta_sequential_deep/eval_val/selection_metrics.json}"

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}" "$(dirname "${EVAL_LOG_FILE}")"

progress_arg=()
if [[ "${NO_PROGRESS}" == "true" || "${NO_PROGRESS}" == "1" ]]; then
  progress_arg+=(--no-progress)
fi

build_extra=()
if [[ -n "${TRAIN_SAMPLE_LIMIT}" ]]; then
  build_extra+=(--sample-limit "${TRAIN_SAMPLE_LIMIT}")
fi

if [[ "${RUN_BUILD_DATA}" == "true" || "${RUN_BUILD_DATA}" == "1" ]]; then
  python scripts/selectors/build_llm_action_selector_data.py \
    --oracle-results "${TRAIN_ORACLE_RESULTS}" \
    --vig-cache "${TRAIN_VIG_CACHE}" \
    --output-jsonl "${TRAIN_DATA}" \
    --split train \
    --max-candidate-chars "${MAX_CANDIDATE_CHARS}" \
    --tokenizer "${MODEL_NAME}" \
    --max-length "${MAX_LENGTH}" \
    "${progress_arg[@]}" \
    "${build_extra[@]}"

  val_extra=()
  if [[ -n "${VAL_SAMPLE_LIMIT}" ]]; then
    val_extra+=(--sample-limit "${VAL_SAMPLE_LIMIT}")
  fi
  python scripts/selectors/build_llm_action_selector_data.py \
    --oracle-results "${VAL_ORACLE_RESULTS}" \
    --vig-cache "${VAL_VIG_CACHE}" \
    --output-jsonl "${VAL_DATA}" \
    --split val \
    --max-candidate-chars "${MAX_CANDIDATE_CHARS}" \
    --tokenizer "${MODEL_NAME}" \
    --max-length "${MAX_LENGTH}" \
    "${progress_arg[@]}" \
    "${val_extra[@]}"
fi

train_cmd=(
  scripts/selectors/train_llm_action_selector.py
  --train-data "${TRAIN_DATA}"
  --val-data "${VAL_DATA}"
  --output-dir "${OUTPUT_DIR}"
  --model-name "${MODEL_NAME}"
  --max-length "${MAX_LENGTH}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --choice-batch-size "${CHOICE_BATCH_SIZE}"
  --score-mode "${SCORE_MODE}"
  --num-train-epochs "${EPOCHS}"
  --learning-rate "${LR}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --soft-loss-weight "${SOFT_LOSS_WEIGHT}"
  --soft-tau "${SOFT_TAU}"
  --eval-every "${EVAL_EVERY}"
  --selection-eval-oracle-results "${VAL_ORACLE_RESULTS}"
  --selection-eval-mode "${TRAIN_SELECTION_EVAL_MODE}"
  --selection-eval-sample-limit "${SELECTION_EVAL_SAMPLE_LIMIT}"
  --selection-eval-top-k "${SELECTION_EVAL_TOP_K}"
  --selection-eval-max-candidate-chars "${MAX_CANDIDATE_CHARS}"
  --selection-eval-output-dir "${SELECTION_EVAL_OUTPUT_DIR}"
  --swanlab-project "${SWANLAB_PROJECT}"
  --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME}"
  --swanlab-tags "${SWANLAB_TAGS}"
  "${progress_arg[@]}"
)
if [[ -n "${EVAL_SAMPLE_LIMIT}" ]]; then
  train_cmd+=(--eval-sample-limit "${EVAL_SAMPLE_LIMIT}")
fi
if [[ -n "${SWANLAB_WORKSPACE}" ]]; then
  train_cmd+=(--swanlab-workspace "${SWANLAB_WORKSPACE}")
fi
if [[ -n "${SWANLAB_MODE}" ]]; then
  train_cmd+=(--swanlab-mode "${SWANLAB_MODE}")
fi
if [[ -n "${SWANLAB_LOGDIR}" ]]; then
  train_cmd+=(--swanlab-logdir "${SWANLAB_LOGDIR}")
fi
if [[ -n "${SWANLAB_DESCRIPTION}" ]]; then
  train_cmd+=(--swanlab-description "${SWANLAB_DESCRIPTION}")
fi
if [[ "${SWANLAB_DISABLED}" == "1" || "${SWANLAB_DISABLED}" == "true" || "${SWANLAB_DISABLED}" == "TRUE" ]]; then
  train_cmd+=(--no-swanlab)
fi

if [[ "${RUN_TRAIN}" == "true" || "${RUN_TRAIN}" == "1" ]]; then
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    accelerate launch \
      --num_processes="${NPROC_PER_NODE}" \
      --num_machines="${NUM_MACHINES}" \
      --mixed_precision="${MIXED_PRECISION}" \
      --use_deepspeed \
      --deepspeed_config_file "${DEEPSPEED_CONFIG}" \
      "${train_cmd[@]}"
  else
    python "${train_cmd[@]}"
  fi
fi

if [[ "${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "1" ]]; then
  references=()
  if [[ -n "${REFERENCE_RANKER_METRICS}" && -f "${REFERENCE_RANKER_METRICS}" ]]; then
    references+=("${REFERENCE_RANKER_METRICS}")
  fi
  if [[ -n "${REFERENCE_SEQUENTIAL_METRICS}" && -f "${REFERENCE_SEQUENTIAL_METRICS}" ]]; then
    references+=("${REFERENCE_SEQUENTIAL_METRICS}")
  fi
  eval_extra=()
  if [[ -n "${EVAL_SAMPLE_LIMIT}" ]]; then
    eval_extra+=(--sample-limit "${EVAL_SAMPLE_LIMIT}")
  fi

  python scripts/selectors/eval_llm_action_selector.py \
    --model-dir "${OUTPUT_DIR}" \
    --model-name "${MODEL_NAME}" \
    --oracle-results "${VAL_ORACLE_RESULTS}" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --split "${EVAL_SPLIT}" \
    --max-length "${MAX_LENGTH}" \
    --score-mode "${SCORE_MODE}" \
    --choice-batch-size "${CHOICE_BATCH_SIZE}" \
    --max-candidate-chars "${MAX_CANDIDATE_CHARS}" \
    --log-file "${EVAL_LOG_FILE}" \
    --reference-metrics "${references[@]}" \
    "${eval_extra[@]}" \
    "${progress_arg[@]}"
fi
