#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
DATASET="${DATASET:-liar_raw}"
MODEL="${MODEL:-llama31_8b}"
MODE="${MODE:-full}"
EVAL_SPLITS="${EVAL_SPLITS:-val,test}"
CHECKPOINTS="${CHECKPOINTS:-best}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_STAGE="${FORCE_STAGE:-false}"
FORCE_BUILD="${FORCE_BUILD:-false}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"
FORCE_EVAL="${FORCE_EVAL:-false}"
SOURCE_ROOT="${SOURCE_ROOT:-}"
REBUILD_RAWFC_SOURCES="${REBUILD_RAWFC_SOURCES:-false}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero3_bsz1_ga8_lowpeak.json}"
SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE:-true}"
RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE:-$SAVE_LATEST_TRAIN_STATE}"

if [[ -z "${ACCELERATE_BIN:-}" ]]; then
  py_dir="$(dirname "$PYTHON_BIN")"
  if [[ -x "${py_dir}/accelerate" ]]; then
    ACCELERATE_BIN="${py_dir}/accelerate"
  else
    ACCELERATE_BIN="accelerate"
  fi
fi

normalize_dataset() {
  case "${1//-/_}" in
    liar|liarraw|liar_raw) printf '%s\n' "liar_raw" ;;
    rawfc|raw_fc) printf '%s\n' "rawfc" ;;
    *) printf 'Unsupported DATASET=%s\n' "$1" >&2; exit 2 ;;
  esac
}

normalize_model() {
  case "${1//-/_}" in
    llama31_8b|llama3_8b|llama3_1_8b|llama31) printf '%s\n' "llama31_8b" ;;
    qwen3_4b_2507|qwen3_4b|qwen3) printf '%s\n' "qwen3_4b_2507" ;;
    *) printf 'Unsupported MODEL=%s\n' "$1" >&2; exit 2 ;;
  esac
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

training_complete() {
  local train_dir="$1"
  local marker="${train_dir}/training_complete.json"
  [[ -f "$marker" ]] && grep -Eq '"completed"[[:space:]]*:[[:space:]]*true' "$marker"
}

DATASET="$(normalize_dataset "$DATASET")"
MODEL="$(normalize_model "$MODEL")"
CASE_NAME="${DATASET}__${MODEL}"
CONFIG_PATH="scripts/sentence_trace_method/configs/${CASE_NAME}.yaml"
RUN_DIR="${OUTPUT_ROOT}/${CASE_NAME}"
SOURCE_ENV="${RUN_DIR}/source.env"
STAGE_SAMPLE_LIMIT="$SAMPLE_LIMIT"

if [[ ! -f "$CONFIG_PATH" ]]; then
  printf 'Missing config: %s\n' "$CONFIG_PATH" >&2
  exit 2
fi

case "$DATASET" in
  liar_raw)
    TRAIN_RAW="data/raw/LIAR-RAW/train.json"
    VAL_RAW="data/raw/LIAR-RAW/val.json"
    TEST_RAW="data/raw/LIAR-RAW/test.json"
    LABEL_SCHEMA="liar6"
    ;;
  rawfc)
    TRAIN_RAW="data/raw/RAWFC/train.json"
    VAL_RAW="data/raw/RAWFC/val.json"
    TEST_RAW="data/raw/RAWFC/test.json"
    LABEL_SCHEMA="rawfc3"
    ;;
esac

case "$MODEL" in
  llama31_8b) MODEL_PATH="/data/models/Meta-Llama-3.1-8B-Instruct" ;;
  qwen3_4b_2507) MODEL_PATH="/data/models/Qwen3-4B-Instruct-2507" ;;
esac

mkdir -p "$RUN_DIR"

FULL_STAGED_ROOT="${OUTPUT_ROOT}/_sources/${DATASET}/sentence_rule_step_adaptive5_10"
if [[ -z "$SOURCE_ROOT" && "$FORCE_STAGE" != "true" \
  && -f "${FULL_STAGED_ROOT}/train/selection_trace_train.jsonl" \
  && -f "${FULL_STAGED_ROOT}/val/selection_trace_val.jsonl" \
  && -f "${FULL_STAGED_ROOT}/test/selection_trace_test.jsonl" ]]; then
  STAGE_SAMPLE_LIMIT="0"
fi

if [[ "$DATASET" == "rawfc" && "$REBUILD_RAWFC_SOURCES" == "true" ]]; then
  run_cmd bash scripts/sentence_trace_method/build_rawfc_sentence_sources.sh
  SOURCE_ROOT="${OUTPUT_ROOT}/_raw_sources/rawfc/sentence_rule_step_adaptive5_10/graph"
  STAGE_SAMPLE_LIMIT="$SAMPLE_LIMIT"
fi

stage_sources() {
  local cmd=("$PYTHON_BIN" scripts/sentence_trace_method/stage_sources.py
    --dataset "$DATASET"
    --output-root "$OUTPUT_ROOT"
    --sample-limit "$STAGE_SAMPLE_LIMIT"
    --splits train,val,test
    --env-file "$SOURCE_ENV")
  if [[ "$FORCE_STAGE" == "true" ]]; then
    cmd+=(--force)
  fi
  if [[ -n "$SOURCE_ROOT" ]]; then
    cmd+=(--source-root "$SOURCE_ROOT")
  fi
  run_cmd "${cmd[@]}"
}

do_build() {
  if [[ -f "${RUN_DIR}/train.resolved.yaml" && -f "${RUN_DIR}/build/build_report.json" && "$FORCE_BUILD" != "true" ]]; then
    printf 'Build artifacts already exist for %s; set FORCE_BUILD=true to rebuild.\n' "$CASE_NAME"
    return 0
  fi
  if [[ "$DRY_RUN" == "true" && ! -f "$SOURCE_ENV" ]]; then
    local source_suffix=""
    if [[ "$STAGE_SAMPLE_LIMIT" != "0" ]]; then
      source_suffix="_sample${STAGE_SAMPLE_LIMIT}"
    fi
    EXPECTED_CHUNK_MMR_FINGERPRINT="<staged-fingerprint>"
    TRAIN_TRACE="${OUTPUT_ROOT}/_sources/${DATASET}/sentence_rule_step_adaptive5_10${source_suffix}/train/selection_trace_train.jsonl"
    VAL_TRACE="${OUTPUT_ROOT}/_sources/${DATASET}/sentence_rule_step_adaptive5_10${source_suffix}/val/selection_trace_val.jsonl"
    TEST_TRACE="${OUTPUT_ROOT}/_sources/${DATASET}/sentence_rule_step_adaptive5_10${source_suffix}/test/selection_trace_test.jsonl"
  else
    # shellcheck disable=SC1090
    source "$SOURCE_ENV"
  fi
  local cmd=("$PYTHON_BIN" scripts/phase5_selectors/build/build_trace_verifier_data.py
    --config "$CONFIG_PATH"
    --train-raw "$TRAIN_RAW"
    --val-raw "$VAL_RAW"
    --test-raw "$TEST_RAW"
    --dataset "$DATASET"
    --label-schema "$LABEL_SCHEMA"
    --output-dir "$RUN_DIR"
    --selection-mode trace
    --trace-prompt-style plain
    --expected-selector-name sentence_rule_step_adaptive5_10
    --expected-chunk-mmr-fingerprint "$EXPECTED_CHUNK_MMR_FINGERPRINT"
    --top-k 10
    --prompt-model-name-or-path "$MODEL_PATH"
    --train-model-name-or-path "$MODEL_PATH"
    --train-trace "$TRAIN_TRACE"
    --val-trace "$VAL_TRACE"
    --test-trace "$TEST_TRACE")
  if [[ "$SAMPLE_LIMIT" != "0" ]]; then
    cmd+=(--sample-limit "$SAMPLE_LIMIT")
  fi
  run_cmd "${cmd[@]}"
}

do_train() {
  local train_dir="${RUN_DIR}/train"
  if training_complete "$train_dir" && [[ "$FORCE_TRAIN" != "true" ]]; then
    printf 'Training is already complete for %s; set FORCE_TRAIN=true to launch training again.\n' "$CASE_NAME"
    return 0
  fi
  if [[ -d "${train_dir}/best" && "$FORCE_TRAIN" != "true" ]]; then
    if [[ -f "${train_dir}/latest_state/trainer_state.json" ]]; then
      printf 'Best checkpoint exists but training is not marked complete for %s; resuming from latest_state.\n' "$CASE_NAME"
    else
      printf 'Best checkpoint exists but training is not marked complete for %s; launching trainer instead of skipping. No latest_state was found, so this may restart from the beginning.\n' "$CASE_NAME"
    fi
  fi
  run_cmd env \
    SAVE_LATEST_TRAIN_STATE="$SAVE_LATEST_TRAIN_STATE" \
    RESUME_LATEST_TRAIN_STATE="$RESUME_LATEST_TRAIN_STATE" \
    "$ACCELERATE_BIN" launch \
    --num_processes "$NPROC_PER_NODE" \
    --num_machines "$NUM_MACHINES" \
    --mixed_precision "$MIXED_PRECISION" \
    --use_deepspeed \
    --deepspeed_config_file "$DEEPSPEED_CONFIG" \
    -m sft.label_token_trainer \
    --config "${RUN_DIR}/train.resolved.yaml"
}

do_eval() {
  IFS=',' read -r -a split_array <<< "$EVAL_SPLITS"
  IFS=',' read -r -a checkpoint_array <<< "$CHECKPOINTS"
  for split in "${split_array[@]}"; do
    split="${split// /}"
    [[ -z "$split" ]] && continue
    for checkpoint in "${checkpoint_array[@]}"; do
      checkpoint="${checkpoint// /}"
      [[ -z "$checkpoint" ]] && continue
      local metrics_path="${RUN_DIR}/eval/${split}/${checkpoint}/metrics.json"
      if [[ -f "$metrics_path" && "$FORCE_EVAL" != "true" ]]; then
        printf 'Eval artifact already exists: %s; set FORCE_EVAL=true to rerun.\n' "$metrics_path"
        continue
      fi
      run_cmd "$PYTHON_BIN" -m sft.label_token_infer \
        --run-dir "${RUN_DIR}/train" \
        --checkpoint "$checkpoint" \
        --split "$split" \
        --config "${RUN_DIR}/train.resolved.yaml"
    done
  done
}

case "$MODE" in
  stage)
    stage_sources
    ;;
  build)
    stage_sources
    do_build
    ;;
  train)
    do_train
    ;;
  eval)
    do_eval
    ;;
  full)
    stage_sources
    do_build
    do_train
    do_eval
    ;;
  *)
    printf 'Unsupported MODE=%s. Use stage, build, train, eval, or full.\n' "$MODE" >&2
    exit 2
    ;;
esac
