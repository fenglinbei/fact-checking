#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

SCRIPT_DIR="${ROOT_DIR}/scripts/sentence_trace_method"

export PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
export EVAL_SPLITS="${EVAL_SPLITS:-val}"
export CHECKPOINTS="${CHECKPOINTS:-best}"
export CASE_SUFFIX="${CASE_SUFFIX:-__v0_7_bm_adaptive5_10__qec_min}"
export TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-qec_min}"
export LORA_SUFFIX="${LORA_SUFFIX:-_lora_r32a64_d005_ebs16_lr1em5_ep12_eval50_pat8_rawfc}"
export LORA_R="${LORA_R:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-5}"
export SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-12}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-50}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-50}"
export SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-8}"
export RUN_TAU_EVAL="${RUN_TAU_EVAL:-false}"

RUN_TEST_EVAL="${RUN_TEST_EVAL:-auto}"
TEST_SPLIT="${TEST_SPLIT:-test}"
TEST_CHECKPOINT="${TEST_CHECKPOINT:-best}"
FORCE_EVAL="${FORCE_EVAL:-true}"
DRY_RUN="${DRY_RUN:-false}"
LOG_PREDICTIONS="${LOG_PREDICTIONS:-0}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-}"

should_run_test_eval() {
  case "$RUN_TEST_EVAL" in
    true) return 0 ;;
    false) return 1 ;;
    auto)
      case "${MODE:-full}" in
        full|eval) return 0 ;;
        *) return 1 ;;
      esac
      ;;
    *) printf 'Unsupported RUN_TEST_EVAL=%s. Use true, false, or auto.\n' "$RUN_TEST_EVAL" >&2; exit 2 ;;
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

append_eval_options() {
  if [[ -n "$PER_DEVICE_EVAL_BATCH_SIZE" ]]; then
    cmd+=(--per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE")
  fi
  if [[ -n "$DATALOADER_NUM_WORKERS" ]]; then
    cmd+=(--dataloader-num-workers "$DATALOADER_NUM_WORKERS")
  fi
}

run_test_eval() {
  local case_root="${OUTPUT_ROOT}/rawfc__ministral3_8b${CASE_SUFFIX}${LORA_SUFFIX}"
  local run_dir="${case_root}/train"
  local config="${case_root}/train.resolved.yaml"
  local output_dir="${case_root}/eval/${TEST_SPLIT}/${TEST_CHECKPOINT}/label_token"
  local metrics_path="${output_dir}/metrics.json"

  if [[ "$DRY_RUN" != "true" ]]; then
    [[ -d "$run_dir" ]] || { printf 'Run directory not found: %s\n' "$run_dir" >&2; exit 2; }
    [[ -f "$config" ]] || { printf 'Config not found: %s\n' "$config" >&2; exit 2; }
  fi
  if [[ -f "$metrics_path" && "$FORCE_EVAL" != "true" ]]; then
    printf 'Test eval already exists: %s; set FORCE_EVAL=true to rerun.\n' "$metrics_path"
    return 0
  fi

  cmd=("$PYTHON_BIN" -m sft.label_token_infer
    --run-dir "$run_dir"
    --checkpoint "$TEST_CHECKPOINT"
    --split "$TEST_SPLIT"
    --config "$config"
    --output-dir "$output_dir"
    --logit-adjust off
    --log-predictions "$LOG_PREDICTIONS")
  append_eval_options
  run_cmd "${cmd[@]}"
}

printf '\n[rawfc-ministral3-qec-min-lora] DATASETS=rawfc MODELS=ministral3_8b TRACE_PROMPT_STYLE=%s CASE_SUFFIX=%s LORA_SUFFIX=%s EBS=16 DEEPSPEED_CONFIG=configs/deepspeed_zero2_bsz1_ga4.json SFT_GRADIENT_ACCUMULATION_STEPS=4 SFT_LEARNING_RATE=%s SFT_NUM_TRAIN_EPOCHS=%s SFT_EVAL_STEPS=%s SFT_SAVE_STEPS=%s SFT_EARLY_STOPPING_PATIENCE=%s REQUIRE_PROMPT_INPUT_IDS=true TEST_EVAL=%s/%s RUN_TAU_EVAL=%s\n' \
  "$TRACE_PROMPT_STYLE" "$CASE_SUFFIX" "$LORA_SUFFIX" "$SFT_LEARNING_RATE" "$SFT_NUM_TRAIN_EPOCHS" "$SFT_EVAL_STEPS" "$SFT_SAVE_STEPS" "$SFT_EARLY_STOPPING_PATIENCE" "$TEST_SPLIT" "$TEST_CHECKPOINT" "$RUN_TAU_EVAL"

bash "${SCRIPT_DIR}/run_rawfc_ministral3_v0_7_adaptive5_10_lr1e5_ep12.sh"

if should_run_test_eval; then
  run_test_eval
fi
