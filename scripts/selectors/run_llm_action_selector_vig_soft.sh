#!/usr/bin/env bash
# Build data, train, and selection-only evaluate the Qwen LLM action selector.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"

MODEL_NAME="${MODEL_NAME:-/data/models/Qwen2.5-3B-Instruct}"
SANITY_MODE="${SANITY_MODE:-none}"
if [[ "${SANITY_MODE}" == "overfit_train_sample" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/llm_action_selector_sanity/overfit_128}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/llm_action_selector/qwen25_3b_vig_soft}"
fi
DATA_DIR="${DATA_DIR:-${OUTPUT_DIR}/data}"
TRAIN_ORACLE_RESULTS="${TRAIN_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_train_stepscores/oracle_results_train.jsonl}"
VAL_ORACLE_RESULTS="${VAL_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
TRAIN_VIG_CACHE="${TRAIN_VIG_CACHE:-outputs/selectors/vig_utility/saved_step_train/vig_records_train.jsonl}"
VAL_VIG_CACHE="${VAL_VIG_CACHE:-outputs/selectors/vig_utility/saved_step_val/vig_records_val.jsonl}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/action_samples_train.jsonl}"
BAD_PREFIX_TRAIN_DATA="${BAD_PREFIX_TRAIN_DATA:-${DATA_DIR}/action_samples_train_bad_prefix.jsonl}"
if [[ "${SANITY_MODE}" == "overfit_train_sample" ]]; then
  VAL_DATA="${VAL_DATA:-${TRAIN_DATA}}"
  BAD_PREFIX_VAL_DATA="${BAD_PREFIX_VAL_DATA:-${BAD_PREFIX_TRAIN_DATA}}"
  EVAL_ORACLE_RESULTS="${EVAL_ORACLE_RESULTS:-${TRAIN_ORACLE_RESULTS}}"
  SELECTION_EVAL_ORACLE_RESULTS="${SELECTION_EVAL_ORACLE_RESULTS:-${TRAIN_ORACLE_RESULTS}}"
else
  VAL_DATA="${VAL_DATA:-${DATA_DIR}/action_samples_val.jsonl}"
  BAD_PREFIX_VAL_DATA="${BAD_PREFIX_VAL_DATA:-${DATA_DIR}/action_samples_val_bad_prefix.jsonl}"
  EVAL_ORACLE_RESULTS="${EVAL_ORACLE_RESULTS:-${VAL_ORACLE_RESULTS}}"
  SELECTION_EVAL_ORACLE_RESULTS="${SELECTION_EVAL_ORACLE_RESULTS:-${VAL_ORACLE_RESULTS}}"
fi
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/evals/val}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
EVAL_LOG_FILE="${EVAL_LOG_FILE:-${OUTPUT_DIR}/logs/eval_llm_action_selector.log}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_CANDIDATE_CHARS="${MAX_CANDIDATE_CHARS:-180}"
if [[ "${SANITY_MODE}" == "overfit_train_sample" ]]; then
  TRAIN_SAMPLE_LIMIT="${TRAIN_SAMPLE_LIMIT:-128}"
  VAL_SAMPLE_LIMIT="${VAL_SAMPLE_LIMIT:-128}"
else
  TRAIN_SAMPLE_LIMIT="${TRAIN_SAMPLE_LIMIT:-}"
  VAL_SAMPLE_LIMIT="${VAL_SAMPLE_LIMIT:-}"
fi
if [[ "${SANITY_MODE}" == "overfit_train_sample" ]]; then
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
fi
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz8_ga1.json}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-16}"
CHOICE_BATCH_SIZE="${CHOICE_BATCH_SIZE:-512}"
SCORE_MODE="${SCORE_MODE:-action_token}"
if [[ "${SANITY_MODE}" == "overfit_train_sample" ]]; then
  EPOCHS="${EPOCHS:-10}"
else
  EPOCHS="${EPOCHS:-2}"
fi
LR="${LR:-1.0e-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
SOFT_LOSS_WEIGHT="${SOFT_LOSS_WEIGHT:-0.01}"
SOFT_TAU="${SOFT_TAU:-0.2}"
SET_LOSS_WEIGHT="${SET_LOSS_WEIGHT:-0.02}"
SET_LOSS_TYPE="${SET_LOSS_TYPE:-multi_positive_ce}"
HARD_LOSS_WEIGHT="${HARD_LOSS_WEIGHT:-1.0}"
PAIRWISE_LOSS_WEIGHT="${PAIRWISE_LOSS_WEIGHT:-0.05}"
BAD_PREFIX_HARD_LOSS_WEIGHT="${BAD_PREFIX_HARD_LOSS_WEIGHT:-0}"
TRAIN_ORDER_AUGMENTATION="${TRAIN_ORDER_AUGMENTATION:-dynamic_random}"
BUILD_BAD_PREFIX_DATA="${BUILD_BAD_PREFIX_DATA:-true}"
BAD_PREFIX_SOURCES="${BAD_PREFIX_SOURCES:-hybrid,random_corrupt}"
BAD_PREFIX_MAX_REPLACEMENTS="${BAD_PREFIX_MAX_REPLACEMENTS:-2}"
BAD_PREFIX_SAMPLE_RATIO="${BAD_PREFIX_SAMPLE_RATIO:-1.0}"
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}"
EVAL_SAMPLE_SEED="${EVAL_SAMPLE_SEED:-20260524}"
if [[ "${SANITY_MODE}" == "overfit_train_sample" ]]; then
  EVAL_EVERY="${EVAL_EVERY:-20}"
  EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT:-128}"
else
  EVAL_EVERY="${EVAL_EVERY:-100}"
  EVAL_SAMPLE_LIMIT="${EVAL_SAMPLE_LIMIT:-}"
fi
NO_PROGRESS="${NO_PROGRESS:-false}"
RUN_BUILD_DATA="${RUN_BUILD_DATA:-true}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
if [[ "${SANITY_MODE}" == "overfit_train_sample" ]]; then
  TRAIN_SELECTION_EVAL_MODE="${TRAIN_SELECTION_EVAL_MODE:-every_eval}"
else
  TRAIN_SELECTION_EVAL_MODE="${TRAIN_SELECTION_EVAL_MODE:-every_eval}"
fi
SELECTION_EVAL_SAMPLE_LIMIT="${SELECTION_EVAL_SAMPLE_LIMIT:-128}"
SELECTION_EVAL_TOP_K="${SELECTION_EVAL_TOP_K:-5}"
SELECTION_EVAL_OUTPUT_DIR="${SELECTION_EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/evals/during_train}"
BEST_SELECTION_METRIC="${BEST_SELECTION_METRIC:-jaccard@5}"
PRIMARY_CHECKPOINT="${PRIMARY_CHECKPOINT:-selection}"
ACTION_LABEL_MODE="${ACTION_LABEL_MODE:-local_choice}"
TRAIN_CANDIDATE_ORDER="${TRAIN_CANDIDATE_ORDER:-random}"
EVAL_CANDIDATE_ORDER="${EVAL_CANDIDATE_ORDER:-candidate_pool}"
CANDIDATE_ORDER_SEED="${CANDIDATE_ORDER_SEED:-20260524}"
RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL:-true}"
BASELINE_EVAL_SAMPLE_LIMIT="${BASELINE_EVAL_SAMPLE_LIMIT:-128}"
BASELINE_EVAL_OUTPUT_DIR="${BASELINE_EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/evals/base_val}"
RUN_INITIAL_EVAL="${RUN_INITIAL_EVAL:-true}"
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

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}" "${OUTPUT_DIR}/metrics" "$(dirname "${EVAL_LOG_FILE}")"

progress_arg=()
if [[ "${NO_PROGRESS}" == "true" || "${NO_PROGRESS}" == "1" ]]; then
  progress_arg+=(--no-progress)
fi

build_extra=()
if [[ -n "${TRAIN_SAMPLE_LIMIT}" ]]; then
  build_extra+=(--sample-limit "${TRAIN_SAMPLE_LIMIT}")
fi

if [[ "${RUN_BUILD_DATA}" == "true" || "${RUN_BUILD_DATA}" == "1" ]]; then
  train_bad_prefix_args=()
  if [[ "${BUILD_BAD_PREFIX_DATA}" == "true" || "${BUILD_BAD_PREFIX_DATA}" == "1" ]]; then
    train_bad_prefix_args+=(
      --bad-prefix-output-jsonl "${BAD_PREFIX_TRAIN_DATA}"
      --bad-prefix-sources "${BAD_PREFIX_SOURCES}"
      --bad-prefix-max-replacements "${BAD_PREFIX_MAX_REPLACEMENTS}"
      --bad-prefix-sample-ratio "${BAD_PREFIX_SAMPLE_RATIO}"
    )
  fi
  python scripts/selectors/build_llm_action_selector_data.py \
    --oracle-results "${TRAIN_ORACLE_RESULTS}" \
    --vig-cache "${TRAIN_VIG_CACHE}" \
    --output-jsonl "${TRAIN_DATA}" \
    --split train \
    --max-candidate-chars "${MAX_CANDIDATE_CHARS}" \
    --tokenizer "${MODEL_NAME}" \
    --max-length "${MAX_LENGTH}" \
    --action-label-mode "${ACTION_LABEL_MODE}" \
    --candidate-order-mode "${TRAIN_CANDIDATE_ORDER}" \
    --candidate-order-seed "${CANDIDATE_ORDER_SEED}" \
    "${progress_arg[@]}" \
    "${train_bad_prefix_args[@]}" \
    "${build_extra[@]}"

  if [[ "${SANITY_MODE}" == "overfit_train_sample" ]]; then
    echo "SANITY_MODE=overfit_train_sample: using train action samples as validation data (${VAL_DATA})."
  else
    val_extra=()
    if [[ -n "${VAL_SAMPLE_LIMIT}" ]]; then
      val_extra+=(--sample-limit "${VAL_SAMPLE_LIMIT}")
    fi
    val_bad_prefix_args=()
    if [[ "${BUILD_BAD_PREFIX_DATA}" == "true" || "${BUILD_BAD_PREFIX_DATA}" == "1" ]]; then
      val_bad_prefix_args+=(
        --bad-prefix-output-jsonl "${BAD_PREFIX_VAL_DATA}"
        --bad-prefix-sources "${BAD_PREFIX_SOURCES}"
        --bad-prefix-max-replacements "${BAD_PREFIX_MAX_REPLACEMENTS}"
        --bad-prefix-sample-ratio "${BAD_PREFIX_SAMPLE_RATIO}"
      )
    fi
    python scripts/selectors/build_llm_action_selector_data.py \
      --oracle-results "${VAL_ORACLE_RESULTS}" \
      --vig-cache "${VAL_VIG_CACHE}" \
      --output-jsonl "${VAL_DATA}" \
      --split val \
      --max-candidate-chars "${MAX_CANDIDATE_CHARS}" \
      --tokenizer "${MODEL_NAME}" \
      --max-length "${MAX_LENGTH}" \
      --action-label-mode "${ACTION_LABEL_MODE}" \
      --candidate-order-mode "${EVAL_CANDIDATE_ORDER}" \
      --candidate-order-seed "${CANDIDATE_ORDER_SEED}" \
      "${progress_arg[@]}" \
      "${val_bad_prefix_args[@]}" \
      "${val_extra[@]}"
  fi
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
  --set-loss-weight "${SET_LOSS_WEIGHT}"
  --set-loss-type "${SET_LOSS_TYPE}"
  --hard-loss-weight "${HARD_LOSS_WEIGHT}"
  --pairwise-loss-weight "${PAIRWISE_LOSS_WEIGHT}"
  --bad-prefix-hard-loss-weight "${BAD_PREFIX_HARD_LOSS_WEIGHT}"
  --train-order-augmentation "${TRAIN_ORDER_AUGMENTATION}"
  --eval-every "${EVAL_EVERY}"
  --eval-sample-mode "${EVAL_SAMPLE_MODE}"
  --eval-sample-seed "${EVAL_SAMPLE_SEED}"
  --selection-eval-oracle-results "${SELECTION_EVAL_ORACLE_RESULTS}"
  --selection-eval-mode "${TRAIN_SELECTION_EVAL_MODE}"
  --selection-eval-sample-limit "${SELECTION_EVAL_SAMPLE_LIMIT}"
  --selection-eval-top-k "${SELECTION_EVAL_TOP_K}"
  --selection-eval-max-candidate-chars "${MAX_CANDIDATE_CHARS}"
  --selection-eval-output-dir "${SELECTION_EVAL_OUTPUT_DIR}"
  --best-selection-metric "${BEST_SELECTION_METRIC}"
  --primary-checkpoint "${PRIMARY_CHECKPOINT}"
  --action-label-mode "${ACTION_LABEL_MODE}"
  --candidate-order-mode "${EVAL_CANDIDATE_ORDER}"
  --candidate-order-seed "${CANDIDATE_ORDER_SEED}"
  --swanlab-project "${SWANLAB_PROJECT}"
  --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME}"
  --swanlab-tags "${SWANLAB_TAGS}"
  "${progress_arg[@]}"
)
if [[ "${BUILD_BAD_PREFIX_DATA}" == "true" || "${BUILD_BAD_PREFIX_DATA}" == "1" ]]; then
  train_cmd+=(--bad-prefix-train-data "${BAD_PREFIX_TRAIN_DATA}")
  train_cmd+=(--bad-prefix-val-data "${BAD_PREFIX_VAL_DATA}")
fi
if [[ -n "${EVAL_SAMPLE_LIMIT}" ]]; then
  train_cmd+=(--eval-sample-limit "${EVAL_SAMPLE_LIMIT}")
fi
if [[ "${RUN_INITIAL_EVAL}" == "true" || "${RUN_INITIAL_EVAL}" == "1" ]]; then
  train_cmd+=(--initial-eval)
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

if [[ "${RUN_BASELINE_EVAL}" == "true" || "${RUN_BASELINE_EVAL}" == "1" ]]; then
  if [[ "${RUN_TRAIN}" == "true" || "${RUN_TRAIN}" == "1" ]]; then
    baseline_extra=()
    if [[ -n "${BASELINE_EVAL_SAMPLE_LIMIT}" ]]; then
      baseline_extra+=(--sample-limit "${BASELINE_EVAL_SAMPLE_LIMIT}")
    fi
    python scripts/selectors/eval_llm_action_selector.py \
      --model-dir "${MODEL_NAME}" \
      --model-name "${MODEL_NAME}" \
      --oracle-results "${EVAL_ORACLE_RESULTS}" \
      --output-dir "${BASELINE_EVAL_OUTPUT_DIR}" \
      --split "${EVAL_SPLIT}" \
      --max-length "${MAX_LENGTH}" \
      --score-mode "${SCORE_MODE}" \
      --choice-batch-size "${CHOICE_BATCH_SIZE}" \
      --max-candidate-chars "${MAX_CANDIDATE_CHARS}" \
      --log-file "${OUTPUT_DIR}/logs/baseline_eval_llm_action_selector.log" \
      --summary-output "${OUTPUT_DIR}/metrics/baseline_selection_eval.json" \
      --action-label-mode "${ACTION_LABEL_MODE}" \
      --candidate-order-mode "${EVAL_CANDIDATE_ORDER}" \
      --candidate-order-seed "${CANDIDATE_ORDER_SEED}" \
      "${baseline_extra[@]}" \
      "${progress_arg[@]}"
  fi
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
    --oracle-results "${EVAL_ORACLE_RESULTS}" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --split "${EVAL_SPLIT}" \
    --max-length "${MAX_LENGTH}" \
    --score-mode "${SCORE_MODE}" \
    --choice-batch-size "${CHOICE_BATCH_SIZE}" \
    --max-candidate-chars "${MAX_CANDIDATE_CHARS}" \
    --log-file "${EVAL_LOG_FILE}" \
    --action-label-mode "${ACTION_LABEL_MODE}" \
    --candidate-order-mode "${EVAL_CANDIDATE_ORDER}" \
    --candidate-order-seed "${CANDIDATE_ORDER_SEED}" \
    --reference-metrics "${references[@]}" \
    "${eval_extra[@]}" \
    "${progress_arg[@]}"
fi
