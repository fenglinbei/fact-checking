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
CASE_SUFFIX="${CASE_SUFFIX:-}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_STAGE="${FORCE_STAGE:-false}"
ALLOW_MULTI_SENTENCE_CANDIDATES="${ALLOW_MULTI_SENTENCE_CANDIDATES:-false}"
ALLOW_EMPTY_CANDIDATE_POOL="${ALLOW_EMPTY_CANDIDATE_POOL:-false}"
ALLOW_EMPTY_EVIDENCE="${ALLOW_EMPTY_EVIDENCE:-false}"
FORCE_BUILD="${FORCE_BUILD:-false}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"
FORCE_EVAL="${FORCE_EVAL:-false}"
SOURCE_ROOT="${SOURCE_ROOT:-}"
EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-}"
SELECTOR_NAME="${SELECTOR_NAME:-sentence_rule_step_adaptive5_10}"
SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-sentence_evidence_chain_graph}"
SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-sentence_rule_step}"
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-$SELECTOR_NAME}"
PROMPT_OUTPUT_MODE="${PROMPT_OUTPUT_MODE:-}"
TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-plain}"
EVIDENCE_TEXT_MODE="${EVIDENCE_TEXT_MODE:-full}"
RAW_ROOT="${RAW_ROOT:-}"
COVERAGE_DATA_ROOT="${COVERAGE_DATA_ROOT:-}"
COVERAGE_POLICY="${COVERAGE_POLICY:-all}"
REBUILD_RAWFC_SOURCES="${REBUILD_RAWFC_SOURCES:-false}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero3_bsz1_ga8_lowpeak.json}"
SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE:-true}"
RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE:-$SAVE_LATEST_TRAIN_STATE}"
REQUIRE_PROMPT_INPUT_IDS="${REQUIRE_PROMPT_INPUT_IDS:-false}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-}"
SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-}"
SFT_WEIGHT_DECAY="${SFT_WEIGHT_DECAY:-}"
SFT_WARMUP_RATIO="${SFT_WARMUP_RATIO:-}"
SFT_MAX_GRAD_NORM="${SFT_MAX_GRAD_NORM:-}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-}"
LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-}"

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
    ministral3_8b|ministral3|mistral3_8b) printf '%s\n' "ministral3_8b" ;;
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
CONFIG_CASE_NAME="${DATASET}__${MODEL}"
CASE_NAME="${CONFIG_CASE_NAME}${CASE_SUFFIX}"
CONFIG_PATH="scripts/sentence_trace_method/configs/${CONFIG_CASE_NAME}.yaml"
RUN_DIR="${OUTPUT_ROOT}/${CASE_NAME}"
SOURCE_ENV="${RUN_DIR}/source.env"
STAGE_SAMPLE_LIMIT="$SAMPLE_LIMIT"

if [[ ! -f "$CONFIG_PATH" ]]; then
  printf 'Missing config: %s\n' "$CONFIG_PATH" >&2
  exit 2
fi

if [[ -z "$RAW_ROOT" && -n "$COVERAGE_DATA_ROOT" ]]; then
  RAW_ROOT="${COVERAGE_DATA_ROOT%/}/${DATASET}/${COVERAGE_POLICY}"
fi

case "$DATASET" in
  liar_raw)
    DEFAULT_RAW_ROOT="${RAW_ROOT:-data/raw/LIAR-RAW}"
    TRAIN_RAW="${TRAIN_RAW:-${DEFAULT_RAW_ROOT}/train.json}"
    VAL_RAW="${VAL_RAW:-${DEFAULT_RAW_ROOT}/val.json}"
    TEST_RAW="${TEST_RAW:-${DEFAULT_RAW_ROOT}/test.json}"
    LABEL_SCHEMA="liar6"
    ;;
  rawfc)
    DEFAULT_RAW_ROOT="${RAW_ROOT:-data/raw/RAWFC}"
    TRAIN_RAW="${TRAIN_RAW:-${DEFAULT_RAW_ROOT}/train.json}"
    VAL_RAW="${VAL_RAW:-${DEFAULT_RAW_ROOT}/val.json}"
    TEST_RAW="${TEST_RAW:-${DEFAULT_RAW_ROOT}/test.json}"
    LABEL_SCHEMA="rawfc3"
    ;;
esac

case "$MODEL" in
  llama31_8b) MODEL_PATH="/data/models/Meta-Llama-3.1-8B-Instruct" ;;
  qwen3_4b_2507) MODEL_PATH="/data/models/Qwen3-4B-Instruct-2507" ;;
  ministral3_8b) MODEL_PATH="/data/models/Ministral-3-8B-Instruct-2512" ;;
esac

mkdir -p "$RUN_DIR"

FULL_STAGED_ROOT="${OUTPUT_ROOT}/_sources/${DATASET}/${SELECTOR_NAME}"
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
    --selector-name "$SELECTOR_NAME"
    --graph-version "$SELECTOR_GRAPH_VERSION"
    --adaptive-policy "$SELECTOR_ADAPTIVE_POLICY"
    --env-file "$SOURCE_ENV")
  if [[ "$FORCE_STAGE" == "true" ]]; then
    cmd+=(--force)
  fi
  if [[ "$ALLOW_MULTI_SENTENCE_CANDIDATES" == "true" || "$ALLOW_MULTI_SENTENCE_CANDIDATES" == "1" ]]; then
    cmd+=(--allow-multi-sentence-candidates)
  fi
  if [[ "$ALLOW_EMPTY_CANDIDATE_POOL" == "true" || "$ALLOW_EMPTY_CANDIDATE_POOL" == "1" ]]; then
    cmd+=(--allow-empty-candidate-pool)
  fi
  if [[ -n "$SOURCE_ROOT" ]]; then
    cmd+=(--source-root "$SOURCE_ROOT")
  fi
  if [[ -n "$EXPECTED_CHUNK_MMR_FINGERPRINT" ]]; then
    cmd+=(--expected-fingerprint "$EXPECTED_CHUNK_MMR_FINGERPRINT")
  fi
  run_cmd "${cmd[@]}"
}

do_build() {
  if [[ -f "${RUN_DIR}/train.resolved.yaml" && -f "${RUN_DIR}/build/build_report.json" && "$FORCE_BUILD" != "true" ]]; then
    printf 'Build artifacts already exist for %s; set FORCE_BUILD=true to rebuild.\n' "$CASE_NAME"
    apply_train_config_overrides
    validate_prompt_input_ids_if_required
    return 0
  fi
  if [[ "$DRY_RUN" == "true" && ! -f "$SOURCE_ENV" ]]; then
    local source_suffix=""
    if [[ "$STAGE_SAMPLE_LIMIT" != "0" ]]; then
      source_suffix="_sample${STAGE_SAMPLE_LIMIT}"
    fi
    EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-<staged-fingerprint>}"
    TRAIN_TRACE="${OUTPUT_ROOT}/_sources/${DATASET}/${SELECTOR_NAME}${source_suffix}/train/selection_trace_train.jsonl"
    VAL_TRACE="${OUTPUT_ROOT}/_sources/${DATASET}/${SELECTOR_NAME}${source_suffix}/val/selection_trace_val.jsonl"
    TEST_TRACE="${OUTPUT_ROOT}/_sources/${DATASET}/${SELECTOR_NAME}${source_suffix}/test/selection_trace_test.jsonl"
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
    --trace-prompt-style "$TRACE_PROMPT_STYLE"
    --evidence-text-mode "$EVIDENCE_TEXT_MODE"
    --expected-selector-name "$EXPECTED_SELECTOR_NAME"
    --expected-chunk-mmr-fingerprint "$EXPECTED_CHUNK_MMR_FINGERPRINT"
    --top-k 10
    --prompt-model-name-or-path "$MODEL_PATH"
    --train-model-name-or-path "$MODEL_PATH"
    --train-trace "$TRAIN_TRACE"
    --val-trace "$VAL_TRACE"
    --test-trace "$TEST_TRACE")
  if [[ -n "$PROMPT_OUTPUT_MODE" ]]; then
    cmd+=(--prompt-output-mode "$PROMPT_OUTPUT_MODE")
  fi
  if [[ "$SAMPLE_LIMIT" != "0" ]]; then
    cmd+=(--sample-limit "$SAMPLE_LIMIT")
  fi
  if [[ "$ALLOW_EMPTY_EVIDENCE" == "true" || "$ALLOW_EMPTY_EVIDENCE" == "1" ]]; then
    cmd+=(--allow-empty-evidence)
  fi
  run_cmd "${cmd[@]}"
  apply_train_config_overrides
  validate_prompt_input_ids_if_required
}

apply_train_config_overrides() {
  local config_path="${RUN_DIR}/train.resolved.yaml"
  if [[ "$DRY_RUN" != "true" && ! -f "$config_path" ]]; then
    return 0
  fi
  local cmd=("$PYTHON_BIN" scripts/sentence_trace_method/apply_train_config_overrides.py
    --config "$config_path"
    --deepspeed-config "$DEEPSPEED_CONFIG")
  if [[ -n "$SFT_GRADIENT_ACCUMULATION_STEPS" ]]; then
    cmd+=(--gradient-accumulation-steps "$SFT_GRADIENT_ACCUMULATION_STEPS")
  fi
  if [[ -n "$SFT_LEARNING_RATE" ]]; then
    cmd+=(--learning-rate "$SFT_LEARNING_RATE")
  fi
  if [[ -n "$SFT_NUM_TRAIN_EPOCHS" ]]; then
    cmd+=(--num-train-epochs "$SFT_NUM_TRAIN_EPOCHS")
  fi
  if [[ -n "$SFT_EVAL_STEPS" ]]; then
    cmd+=(--eval-steps "$SFT_EVAL_STEPS")
  fi
  if [[ -n "$SFT_SAVE_STEPS" ]]; then
    cmd+=(--save-steps "$SFT_SAVE_STEPS")
  fi
  if [[ -n "$SFT_EARLY_STOPPING_PATIENCE" ]]; then
    cmd+=(--early-stopping-patience "$SFT_EARLY_STOPPING_PATIENCE")
  fi
  if [[ -n "$SFT_WEIGHT_DECAY" ]]; then
    cmd+=(--weight-decay "$SFT_WEIGHT_DECAY")
  fi
  if [[ -n "$SFT_WARMUP_RATIO" ]]; then
    cmd+=(--warmup-ratio "$SFT_WARMUP_RATIO")
  fi
  if [[ -n "$SFT_MAX_GRAD_NORM" ]]; then
    cmd+=(--max-grad-norm "$SFT_MAX_GRAD_NORM")
  fi
  if [[ -n "$SWANLAB_PROJECT" ]]; then
    cmd+=(--swanlab-project "$SWANLAB_PROJECT")
  fi
  if [[ "$CASE_NAME" == liar_raw__* && -n "$LIAR_CLASS_WEIGHTS" ]]; then
    local raw_weight class_weight
    IFS=',' read -r -a class_weight_array <<< "$LIAR_CLASS_WEIGHTS"
    for raw_weight in "${class_weight_array[@]}"; do
      class_weight="${raw_weight// /}"
      [[ -z "$class_weight" ]] && continue
      cmd+=(--class-weight "$class_weight")
    done
  fi
  cmd+=(--swanlab-experiment-name "$CASE_NAME")
  run_cmd "${cmd[@]}"
}

validate_prompt_input_ids_if_required() {
  if [[ "$REQUIRE_PROMPT_INPUT_IDS" != "true" || "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  "$PYTHON_BIN" -c '
import json
import sys
from pathlib import Path

case_root = Path(sys.argv[1])
for split in ("train", "val", "test"):
    path = case_root / "build" / f"build_{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            row = json.loads(line)
            ids = row.get("prompt_input_ids")
            if not isinstance(ids, list) or not ids:
                raise SystemExit(f"{path}:{line_no} missing prompt_input_ids")
' "$RUN_DIR"
}

do_train() {
  local train_dir="${RUN_DIR}/train"
  apply_train_config_overrides
  validate_prompt_input_ids_if_required
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
  apply_train_config_overrides
  validate_prompt_input_ids_if_required
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
