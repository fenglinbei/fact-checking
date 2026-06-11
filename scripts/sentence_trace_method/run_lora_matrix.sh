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
DATASETS="${DATASETS:-liar_raw,rawfc}"
MODELS="${MODELS:-llama31_8b,qwen3_4b_2507}"
MODE="${MODE:-full}" # build|train|eval|full
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
EVAL_SPLITS="${EVAL_SPLITS:-val,test}"
CHECKPOINTS="${CHECKPOINTS:-best}"
CASE_SUFFIX="${CASE_SUFFIX:-}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_STAGE="${FORCE_STAGE:-false}"
FORCE_BUILD="${FORCE_BUILD:-auto}"
FORCE_LORA_CONFIG="${FORCE_LORA_CONFIG:-false}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"
FORCE_EVAL="${FORCE_EVAL:-false}"
SELECTOR_NAME="${SELECTOR_NAME:-sentence_rule_step_adaptive5_10}"
SELECTOR_GRAPH_VERSION="${SELECTOR_GRAPH_VERSION:-sentence_evidence_chain_graph}"
SELECTOR_ADAPTIVE_POLICY="${SELECTOR_ADAPTIVE_POLICY:-sentence_rule_step}"
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-$SELECTOR_NAME}"
PROMPT_OUTPUT_MODE="${PROMPT_OUTPUT_MODE:-}"
RAW_ROOT="${RAW_ROOT:-}"
COVERAGE_DATA_ROOT="${COVERAGE_DATA_ROOT:-}"
COVERAGE_POLICY="${COVERAGE_POLICY:-all}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz1_ga8.json}"
LORA_SUFFIX="${LORA_SUFFIX:-_lora}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_BIAS="${LORA_BIAS:-none}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-}"
SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-}"
SFT_LOGIT_ADJUST_ENABLED="${SFT_LOGIT_ADJUST_ENABLED:-}"
SFT_LOGIT_ADJUST_TAU="${SFT_LOGIT_ADJUST_TAU:-}"
SFT_COVERAGE_LABEL_TOKEN_ENABLED="${SFT_COVERAGE_LABEL_TOKEN_ENABLED:-}"
SFT_COVERAGE_LABEL_TOKEN_LOSS_WEIGHT="${SFT_COVERAGE_LABEL_TOKEN_LOSS_WEIGHT:-}"
SFT_COVERAGE_LABEL_TOKEN_PREFIX="${SFT_COVERAGE_LABEL_TOKEN_PREFIX:-}"
LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-lora}"
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
    *) printf 'Unsupported dataset=%s\n' "$1" >&2; exit 2 ;;
  esac
}

normalize_model() {
  case "${1//-/_}" in
    llama31_8b|llama3_8b|llama3_1_8b|llama31) printf '%s\n' "llama31_8b" ;;
    qwen3_4b_2507|qwen3_4b|qwen3) printf '%s\n' "qwen3_4b_2507" ;;
    *) printf 'Unsupported model=%s\n' "$1" >&2; exit 2 ;;
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

line_count() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf '%s\n' "-1"
    return 0
  fi
  wc -l < "$path" | tr -d ' '
}

deepspeed_gradient_accumulation_steps() {
  local config_path="$1"
  "$PYTHON_BIN" -c 'import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)
value = cfg.get("gradient_accumulation_steps", "")
print("" if value is None else value)' "$config_path"
}

validate_config_overrides() {
  if [[ -z "$SFT_GRADIENT_ACCUMULATION_STEPS" ]]; then
    return 0
  fi
  if [[ ! -f "$DEEPSPEED_CONFIG" ]]; then
    printf 'DeepSpeed config not found: %s\n' "$DEEPSPEED_CONFIG" >&2
    exit 2
  fi

  local ds_gradient_accumulation_steps
  ds_gradient_accumulation_steps="$(deepspeed_gradient_accumulation_steps "$DEEPSPEED_CONFIG")"
  if [[ -n "$ds_gradient_accumulation_steps" && "$ds_gradient_accumulation_steps" != "$SFT_GRADIENT_ACCUMULATION_STEPS" ]]; then
    printf 'Gradient accumulation mismatch: SFT_GRADIENT_ACCUMULATION_STEPS=%s but %s has gradient_accumulation_steps=%s\n' \
      "$SFT_GRADIENT_ACCUMULATION_STEPS" "$DEEPSPEED_CONFIG" "$ds_gradient_accumulation_steps" >&2
    printf 'Use a matching DeepSpeed config, e.g. configs/deepspeed_zero2_bsz1_ga4.json for ga=4.\n' >&2
    exit 2
  fi
}

training_complete() {
  local train_dir="$1"
  local marker="${train_dir}/training_complete.json"
  [[ -f "$marker" ]] && grep -Eq '"completed"[[:space:]]*:[[:space:]]*true' "$marker"
}

expected_rows_for_split() {
  local dataset="$1"
  local split="$2"
  local source_path="${OUTPUT_ROOT}/_sources/${dataset}/${SELECTOR_NAME}/${split}/selection_trace_${split}.jsonl"
  local source_rows
  source_rows="$(line_count "$source_path")"
  if [[ "$source_rows" == "-1" ]]; then
    printf '%s\n' "-1"
    return 0
  fi
  if [[ "$SAMPLE_LIMIT" != "0" && "$source_rows" -gt "$SAMPLE_LIMIT" ]]; then
    printf '%s\n' "$SAMPLE_LIMIT"
  else
    printf '%s\n' "$source_rows"
  fi
}

build_ready() {
  local dataset="$1"
  local case_name="$2"
  local split expected actual
  for split in train val test; do
    expected="$(expected_rows_for_split "$dataset" "$split")"
    actual="$(line_count "${OUTPUT_ROOT}/${case_name}/build/build_${split}.jsonl")"
    if [[ "$expected" == "-1" || "$actual" != "$expected" ]]; then
      return 1
    fi
  done
  return 0
}

ensure_build() {
  local dataset="$1"
  local model="$2"
  local case_name="${dataset}__${model}${CASE_SUFFIX}"
  printf '\n== prepare build: %s ==\n' "$case_name"
  run_cmd env \
    PYTHON_BIN="$PYTHON_BIN" \
    DATASET="$dataset" \
    MODEL="$model" \
    CASE_SUFFIX="$CASE_SUFFIX" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    SELECTOR_NAME="$SELECTOR_NAME" \
    SELECTOR_GRAPH_VERSION="$SELECTOR_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$SELECTOR_ADAPTIVE_POLICY" \
    EXPECTED_SELECTOR_NAME="$EXPECTED_SELECTOR_NAME" \
    PROMPT_OUTPUT_MODE="$PROMPT_OUTPUT_MODE" \
    RAW_ROOT="$RAW_ROOT" \
    COVERAGE_DATA_ROOT="$COVERAGE_DATA_ROOT" \
    COVERAGE_POLICY="$COVERAGE_POLICY" \
    MODE=stage \
    SAMPLE_LIMIT=0 \
    FORCE_STAGE="$FORCE_STAGE" \
    bash scripts/sentence_trace_method/run_one.sh

  if [[ "$FORCE_BUILD" != "true" && "$DRY_RUN" != "true" ]] && build_ready "$dataset" "$case_name"; then
    printf 'Build is already ready for %s; set FORCE_BUILD=true to rebuild.\n' "$case_name"
    return 0
  fi

  local force_build_flag="true"
  if [[ "$FORCE_BUILD" == "false" ]]; then
    force_build_flag="false"
  fi
  run_cmd env \
    PYTHON_BIN="$PYTHON_BIN" \
    DATASET="$dataset" \
    MODEL="$model" \
    CASE_SUFFIX="$CASE_SUFFIX" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    SELECTOR_NAME="$SELECTOR_NAME" \
    SELECTOR_GRAPH_VERSION="$SELECTOR_GRAPH_VERSION" \
    SELECTOR_ADAPTIVE_POLICY="$SELECTOR_ADAPTIVE_POLICY" \
    EXPECTED_SELECTOR_NAME="$EXPECTED_SELECTOR_NAME" \
    PROMPT_OUTPUT_MODE="$PROMPT_OUTPUT_MODE" \
    RAW_ROOT="$RAW_ROOT" \
    COVERAGE_DATA_ROOT="$COVERAGE_DATA_ROOT" \
    COVERAGE_POLICY="$COVERAGE_POLICY" \
    MODE=build \
    SAMPLE_LIMIT="$SAMPLE_LIMIT" \
    FORCE_BUILD="$force_build_flag" \
    bash scripts/sentence_trace_method/run_one.sh
}

prepare_lora_config() {
  local case_name="$1"
  local source_config="${OUTPUT_ROOT}/${case_name}/train.resolved.yaml"
  local lora_root="${OUTPUT_ROOT}/${case_name}${LORA_SUFFIX}"
  local cmd=("$PYTHON_BIN" scripts/sentence_trace_method/prepare_lora_config.py
    --source-config "$source_config"
    --output-root "$lora_root"
    --experiment-name "${case_name}${LORA_SUFFIX}"
    --swanlab-project "$SWANLAB_PROJECT"
    --r "$LORA_R"
    --alpha "$LORA_ALPHA"
    --dropout "$LORA_DROPOUT"
    --bias "$LORA_BIAS"
    --deepspeed-config "$DEEPSPEED_CONFIG")
  if [[ -n "$SFT_GRADIENT_ACCUMULATION_STEPS" ]]; then
    cmd+=(--gradient-accumulation-steps "$SFT_GRADIENT_ACCUMULATION_STEPS")
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
  if [[ -n "$SFT_LOGIT_ADJUST_ENABLED" ]]; then
    cmd+=(--logit-adjust-enabled "$SFT_LOGIT_ADJUST_ENABLED")
  fi
  if [[ -n "$SFT_LOGIT_ADJUST_TAU" ]]; then
    cmd+=(--logit-adjust-tau "$SFT_LOGIT_ADJUST_TAU")
  fi
  if [[ -n "$SFT_COVERAGE_LABEL_TOKEN_ENABLED" ]]; then
    cmd+=(--coverage-label-token-enabled "$SFT_COVERAGE_LABEL_TOKEN_ENABLED")
  fi
  if [[ -n "$SFT_COVERAGE_LABEL_TOKEN_LOSS_WEIGHT" ]]; then
    cmd+=(--coverage-label-token-loss-weight "$SFT_COVERAGE_LABEL_TOKEN_LOSS_WEIGHT")
  fi
  if [[ -n "$SFT_COVERAGE_LABEL_TOKEN_PREFIX" ]]; then
    cmd+=(--coverage-label-token-prefix "$SFT_COVERAGE_LABEL_TOKEN_PREFIX")
  fi
  if [[ "$case_name" == liar_raw__* && -n "$LIAR_CLASS_WEIGHTS" ]]; then
    local raw_weight class_weight
    IFS=',' read -r -a class_weight_array <<< "$LIAR_CLASS_WEIGHTS"
    for raw_weight in "${class_weight_array[@]}"; do
      class_weight="${raw_weight// /}"
      [[ -z "$class_weight" ]] && continue
      cmd+=(--class-weight "$class_weight")
    done
  fi
  if [[ "$FORCE_LORA_CONFIG" == "true" ]]; then
    cmd+=(--force)
  fi
  run_cmd "${cmd[@]}"
}

train_lora() {
  local case_name="$1"
  local lora_root="${OUTPUT_ROOT}/${case_name}${LORA_SUFFIX}"
  local train_dir="${lora_root}/train"
  if training_complete "$train_dir" && [[ "$FORCE_TRAIN" != "true" ]]; then
    printf 'LoRA training is already complete for %s; set FORCE_TRAIN=true to launch training again.\n' "$case_name"
    return 0
  fi
  if [[ -d "${train_dir}/best" && "$FORCE_TRAIN" != "true" ]]; then
    if [[ -f "${train_dir}/latest_state/trainer_state.json" ]]; then
      printf 'LoRA best checkpoint exists but training is not marked complete for %s; resuming from latest_state.\n' "$case_name"
    else
      printf 'LoRA best checkpoint exists but training is not marked complete for %s; launching trainer instead of skipping. No latest_state was found, so this may restart from the beginning.\n' "$case_name"
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
    --config "${lora_root}/train.resolved.yaml"
}

eval_lora() {
  local case_name="$1"
  local lora_root="${OUTPUT_ROOT}/${case_name}${LORA_SUFFIX}"
  IFS=',' read -r -a split_array <<< "$EVAL_SPLITS"
  IFS=',' read -r -a checkpoint_array <<< "$CHECKPOINTS"
  local split checkpoint metrics_path
  for split in "${split_array[@]}"; do
    split="${split// /}"
    [[ -z "$split" ]] && continue
    for checkpoint in "${checkpoint_array[@]}"; do
      checkpoint="${checkpoint// /}"
      [[ -z "$checkpoint" ]] && continue
      metrics_path="${lora_root}/eval/${split}/${checkpoint}/metrics.json"
      if [[ -f "$metrics_path" && "$FORCE_EVAL" != "true" ]]; then
        printf 'LoRA eval already exists: %s; set FORCE_EVAL=true to rerun.\n' "$metrics_path"
        continue
      fi
      run_cmd "$PYTHON_BIN" -m sft.label_token_infer \
        --run-dir "${lora_root}/train" \
        --checkpoint "$checkpoint" \
        --split "$split" \
        --config "${lora_root}/train.resolved.yaml"
    done
  done
}

case "$MODE" in
  build|train|eval|full) ;;
  *) printf 'Unsupported MODE=%s. Use build, train, eval, or full.\n' "$MODE" >&2; exit 2 ;;
esac

validate_config_overrides

IFS=',' read -r -a dataset_array <<< "$DATASETS"
IFS=',' read -r -a model_array <<< "$MODELS"

for raw_dataset in "${dataset_array[@]}"; do
  raw_dataset="${raw_dataset// /}"
  [[ -z "$raw_dataset" ]] && continue
  dataset="$(normalize_dataset "$raw_dataset")"
  for raw_model in "${model_array[@]}"; do
    raw_model="${raw_model// /}"
    [[ -z "$raw_model" ]] && continue
    model="$(normalize_model "$raw_model")"
    case_name="${dataset}__${model}${CASE_SUFFIX}"
    printf '\n== sentence_trace_lora case=%s mode=%s ==\n' "$case_name" "$MODE"

    if [[ "$MODE" == "build" || "$MODE" == "full" ]]; then
      ensure_build "$dataset" "$model"
      prepare_lora_config "$case_name"
    elif [[ "$MODE" == "train" || "$MODE" == "eval" ]]; then
      if [[ ! -f "${OUTPUT_ROOT}/${case_name}${LORA_SUFFIX}/train.resolved.yaml" ]]; then
        prepare_lora_config "$case_name"
      fi
    fi

    if [[ "$MODE" == "train" || "$MODE" == "full" ]]; then
      train_lora "$case_name"
    fi
    if [[ "$MODE" == "eval" || "$MODE" == "full" ]]; then
      eval_lora "$case_name"
    fi
  done
done
