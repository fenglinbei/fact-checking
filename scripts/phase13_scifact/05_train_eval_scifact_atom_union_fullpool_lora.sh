#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python" ]]; then
    export PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-mrec-clean/bin/python"
  else
    export PYTHON_BIN="python"
  fi
fi

MREC_POLICY_CONFIG="${MREC_POLICY_CONFIG:-configs/experiment/mrec_v0.2/scifact_atom_union_fullpool_minmax9_9.yaml}"
MODE="${MODE:-full}" # check|build|train|eval|full|export
FINETUNE_MODE="${FINETUNE_MODE:-lora}"
DRY_RUN="${DRY_RUN:-false}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
FORCE_EVAL="${FORCE_EVAL:-false}"

eval "$("$PYTHON_BIN" scripts/sentence_trace_method/mrec_policy_config.py --config "$MREC_POLICY_CONFIG")"

CASE_NAME="${CASE_NAME:-${BASE_CASE_NAME}${CASE_SUFFIX}}"
CASE_ROOT="${CASE_ROOT:-${OUTPUT_ROOT}/${CASE_NAME}}"
case "$FINETUNE_MODE" in
  lora)
    LORA_ROOT="${LORA_ROOT:-${CASE_ROOT}${LORA_SUFFIX}}"
    TRAIN_CASE_ROOT="$LORA_ROOT"
    ;;
  fullft)
    TRAIN_CASE_ROOT="$CASE_ROOT"
    ;;
  *) printf 'Unsupported FINETUNE_MODE=%s. Use lora or fullft.\n' "$FINETUNE_MODE" >&2; exit 2 ;;
esac

DATA_ROOT="${DATA_ROOT:-data/raw/SciFact}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-${TRAIN_CASE_ROOT}/submission}"
SCIFACT_MAX_SENTENCES_PER_DOC="${SCIFACT_MAX_SENTENCES_PER_DOC:-3}"

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

raw_path_for_split() {
  case "$1" in
    train) printf '%s\n' "${DATA_ROOT}/claims_train.jsonl" ;;
    val) printf '%s\n' "${DATA_ROOT}/claims_dev.jsonl" ;;
    test) printf '%s\n' "${DATA_ROOT}/claims_test.jsonl" ;;
    *) printf 'Unsupported split=%s\n' "$1" >&2; exit 2 ;;
  esac
}

prediction_path_for_split() {
  local split="$1"
  printf '%s\n' "${TRAIN_CASE_ROOT}/eval/${split}/best/label_token/${split}_predictions.jsonl"
}

build_path_for_split() {
  local split="$1"
  printf '%s\n' "${TRAIN_CASE_ROOT}/build/build_${split}.jsonl"
}

export_scifact_split() {
  local split="$1"
  local trace_path="${TRACE_ROOT}/selection_trace_${split}.jsonl"
  local raw_path
  raw_path="$(raw_path_for_split "$split")"
  local prediction_path
  prediction_path="$(prediction_path_for_split "$split")"
  local build_path
  build_path="$(build_path_for_split "$split")"
  local output_path="${SUBMISSION_ROOT}/scifact_submission_${split}.jsonl"
  local metrics_path="${SUBMISSION_ROOT}/scifact_official_style_metrics_${split}.json"

  local pred_args=()
  if [[ -f "$prediction_path" ]]; then
    pred_args+=(--predictions "$prediction_path")
  fi
  if [[ -f "$build_path" ]]; then
    pred_args+=(--build-jsonl "$build_path")
  fi
  local metrics_args=()
  if [[ "$split" == "val" ]]; then
    metrics_args=(--metrics-output "$metrics_path")
  fi
  run_cmd "$PYTHON_BIN" scripts/phase13_scifact/export_scifact_submission.py \
    --trace "$trace_path" \
    --raw "$raw_path" \
    --output "$output_path" \
    --max-sentences-per-doc "$SCIFACT_MAX_SENTENCES_PER_DOC" \
    "${pred_args[@]}" \
    "${metrics_args[@]}"
}

predict_scifact_test() {
  local prediction_path
  prediction_path="$(prediction_path_for_split test)"
  local manifest_path="${TRAIN_CASE_ROOT}/eval/test/best/label_token/prediction_manifest.json"
  if [[ "$FORCE_EVAL" != "true" && -f "$prediction_path" && -f "$manifest_path" ]]; then
    printf '[scifact-05] reuse test verifier predictions: %s\n' "$prediction_path"
    return 0
  fi
  run_cmd "$PYTHON_BIN" -m sft.label_token_infer \
    --run-dir "${TRAIN_CASE_ROOT}/train" \
    --checkpoint best \
    --split test \
    --config "${TRAIN_CASE_ROOT}/train.resolved.yaml" \
    --prediction-only
}

printf '[scifact-05] MREC_POLICY_CONFIG=%s MODE=%s FINETUNE_MODE=%s TRAIN_CASE_ROOT=%s TRACE_ROOT=%s WEIGHT_FILE=%s\n' \
  "$MREC_POLICY_CONFIG" "$MODE" "$FINETUNE_MODE" "$TRAIN_CASE_ROOT" "$TRACE_ROOT" "$WEIGHT_FILE"

case "$MODE" in
  check|build|train|eval|full)
    run_cmd env \
      MREC_POLICY_CONFIG="$MREC_POLICY_CONFIG" \
      MODE="$MODE" \
      FINETUNE_MODE="$FINETUNE_MODE" \
      DRY_RUN="$DRY_RUN" \
      SAMPLE_LIMIT="$SAMPLE_LIMIT" \
      bash scripts/sentence_trace_method/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh
    if [[ "$DRY_RUN" != "true" && -f "$WEIGHT_FILE" ]]; then
      mkdir -p "$(dirname "$WEIGHT_FILE")"
      cp "$WEIGHT_FILE" "$(dirname "$WEIGHT_FILE")/learned_marginal_weights.json"
    fi
    ;;
  export)
    ;;
  *)
    printf 'Unsupported MODE=%s. Use check, build, train, eval, full, or export.\n' "$MODE" >&2
    exit 2
    ;;
esac

case "$MODE" in
  full|eval)
    predict_scifact_test
    ;;
esac

case "$MODE" in
  full|eval|export)
    export_scifact_split val
    export_scifact_split test
    ;;
esac
