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

MREC_POLICY_CONFIG="${MREC_POLICY_CONFIG:-configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_policy.yaml}"
eval "$("$PYTHON_BIN" scripts/sentence_trace_method/mrec_policy_config.py --config "$MREC_POLICY_CONFIG")"

MODE="${MODE:-full}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_MREC_BUILD="${FORCE_MREC_BUILD:-false}"
FORCE_WEIGHT_TRAIN="${FORCE_WEIGHT_TRAIN:-false}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

require_path() {
  local path="$1"
  local label="$2"
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  if [[ ! -e "$path" ]]; then
    printf 'Missing %s: %s\n' "$label" "$path" >&2
    exit 2
  fi
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

should_build_fullpool_traces() {
  case "$MODE" in
    build|full) return 0 ;;
    check|train|eval) return 1 ;;
    *) printf 'Unsupported MODE=%s. Use check, build, train, eval, or full.\n' "$MODE" >&2; exit 2 ;;
  esac
}

selection_policy_requires_weight() {
  case "${EXPECTED_SELECTION_POLICY:-}" in
    learned_marginal_proxy|learned_marginal_reward) return 0 ;;
    *) return 1 ;;
  esac
}

train_weights_if_needed() {
  if ! selection_policy_requires_weight; then
    return 0
  fi
  if [[ "${MREC_AUTO_TRAIN_WEIGHTS:-false}" != "true" && "$FORCE_WEIGHT_TRAIN" != "true" ]]; then
    return 0
  fi
  if [[ -f "$WEIGHT_FILE" && "$FORCE_WEIGHT_TRAIN" != "true" ]]; then
    printf '[atom-anchor-v0.2-fullpool-policy] reuse learned marginal weights: %s\n' "$WEIGHT_FILE"
    return 0
  fi
  local train_input val_input
  train_input="${SOURCE_FEATURE_ROOT}/candidate_evidence_map_features_train.jsonl"
  val_input="${SOURCE_FEATURE_ROOT}/candidate_evidence_map_features_val.jsonl"
  require_path "$train_input" "train atom-anchor evidence-map features"
  require_path "$val_input" "val atom-anchor evidence-map features"
  local sample_args=()
  if [[ "${MREC_WEIGHT_SAMPLE_LIMIT:-0}" != "0" ]]; then
    sample_args+=(--sample-limit "$MREC_WEIGHT_SAMPLE_LIMIT")
  fi
  if [[ "${MREC_WEIGHT_TRAIN_SAMPLE_LIMIT:-0}" != "0" ]]; then
    sample_args+=(--train-sample-limit "$MREC_WEIGHT_TRAIN_SAMPLE_LIMIT")
  fi
  if [[ "${MREC_WEIGHT_VAL_SAMPLE_LIMIT:-0}" != "0" ]]; then
    sample_args+=(--val-sample-limit "$MREC_WEIGHT_VAL_SAMPLE_LIMIT")
  fi
  run_cmd "$PYTHON_BIN" scripts/phase5_selectors/train/train_mrec_learned_marginal_proxy.py \
    --train-input "$train_input" \
    --val-input "$val_input" \
    --output-dir "$MREC_WEIGHT_OUTPUT_DIR" \
    --candidate-top-n "$MREC_WEIGHT_CANDIDATE_TOP_N" \
    --rollout-steps "$MREC_WEIGHT_ROLLOUT_STEPS" \
    --epochs "$MREC_WEIGHT_EPOCHS" \
    --learning-rate "$MREC_WEIGHT_LEARNING_RATE" \
    "${sample_args[@]}"
}

build_fullpool_traces() {
  local split input_path output_trace
  IFS=',' read -r -a split_array <<< "$MREC_SPLITS"
  for split in "${split_array[@]}"; do
    split="${split// /}"
    [[ -z "$split" ]] && continue
    if [[ "${TRACE_BUILD_MODE:-mrec}" == "shuffle_existing" ]]; then
      input_path="${TRACE_SHUFFLE_SOURCE_ROOT}/selection_trace_${split}.jsonl"
    else
      input_path="${SOURCE_FEATURE_ROOT}/candidate_evidence_map_features_${split}.jsonl"
    fi
    output_trace="${TRACE_ROOT}/selection_trace_${split}.jsonl"
    if [[ -f "$output_trace" && "$FORCE_MREC_BUILD" != "true" ]]; then
      printf '[atom-anchor-v0.2-fullpool-policy] reuse fullpool trace: %s\n' "$output_trace"
      continue
    fi
    require_path "$input_path" "${split} atom-anchor evidence-map features"
    local sample_args=()
    if [[ "$SAMPLE_LIMIT" != "0" ]]; then
      sample_args=(--sample-limit "$SAMPLE_LIMIT")
    fi
    local token_budget_args=()
    if [[ -n "${TRACE_TOKEN_BUDGET:-}" ]]; then
      token_budget_args=(--token-budget "$TRACE_TOKEN_BUDGET")
    fi
    local contrast_args=()
    if [[ "${TRACE_CONTINUE_AFTER_TARGET_FOR_CONTRAST:-false}" == "true" ]]; then
      contrast_args=(--continue-after-target-for-contrast)
    fi
    case "${TRACE_BUILD_MODE:-mrec}" in
      mrec)
        run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/build_mrec_traces.py \
          --input "$input_path" \
          --output-dir "$TRACE_ROOT" \
          --split "$split" \
          --candidate-top-n "$TRACE_CANDIDATE_TOP_N" \
          --max-steps "$TRACE_MAX_STEPS" \
          --min-steps "$TRACE_MIN_STEPS" \
          "${token_budget_args[@]}" \
          --target-resolved-rate "$TRACE_TARGET_RESOLVED_RATE" \
          "${contrast_args[@]}" \
          --post-target-fill-policy "$TRACE_POST_TARGET_FILL_POLICY" \
          --selector-name "$EXPECTED_SELECTOR_NAME" \
          --selection-policy "$EXPECTED_SELECTION_POLICY" \
          --weight-file "$WEIGHT_FILE" \
          --stop-threshold "$TRACE_STOP_THRESHOLD" \
          --source-selector-name "$SOURCE_SELECTOR_NAME" \
          "${sample_args[@]}"
        ;;
      shuffle_existing)
        run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/shuffle_mrec_trace_order.py \
          --input "$input_path" \
          --output-dir "$TRACE_ROOT" \
          --split "$split" \
          --selector-name "$EXPECTED_SELECTOR_NAME" \
          --adaptive-policy "$EXPECTED_ADAPTIVE_POLICY" \
          --source-selector-name "$SOURCE_SELECTOR_NAME" \
          --seed "$TRACE_SHUFFLE_SEED" \
          "${sample_args[@]}"
        ;;
      *)
        printf 'Unsupported TRACE_BUILD_MODE=%s. Use mrec or shuffle_existing.\n' "${TRACE_BUILD_MODE:-}" >&2
        exit 2
        ;;
    esac
  done
}

printf '[atom-anchor-v0.2-fullpool-policy] MREC_POLICY_CONFIG=%s TRACE_BUILD_MODE=%s SOURCE_FEATURE_ROOT=%s TRACE_ROOT=%s TRACE_SHUFFLE_SOURCE_ROOT=%s TRACE_SHUFFLE_SEED=%s WEIGHT_FILE=%s TRACE_CANDIDATE_TOP_N=%s TRACE_MAX_STEPS=%s TRACE_MIN_STEPS=%s PROMPT_EVIDENCE_POLICY=%s PROMPT_EVIDENCE_MIN_COUNT=%s PROMPT_EVIDENCE_MAX_COUNT=%s PROMPT_EVIDENCE_TOKEN_BUDGET=%s EXPECTED_SELECTOR_NAME=%s QUALITY_AUDIT_MODE=%s EVAL_SPLITS=%s\n' \
  "$MREC_POLICY_CONFIG" \
  "${TRACE_BUILD_MODE:-mrec}" \
  "$SOURCE_FEATURE_ROOT" \
  "$TRACE_ROOT" \
  "${TRACE_SHUFFLE_SOURCE_ROOT:-}" \
  "${TRACE_SHUFFLE_SEED:-0}" \
  "$WEIGHT_FILE" \
  "$TRACE_CANDIDATE_TOP_N" \
  "$TRACE_MAX_STEPS" \
  "$TRACE_MIN_STEPS" \
  "$PROMPT_EVIDENCE_POLICY" \
  "$PROMPT_EVIDENCE_MIN_COUNT" \
  "$PROMPT_EVIDENCE_MAX_COUNT" \
  "$PROMPT_EVIDENCE_TOKEN_BUDGET" \
  "$EXPECTED_SELECTOR_NAME" \
  "$QUALITY_AUDIT_MODE" \
  "$EVAL_SPLITS"

if should_build_fullpool_traces; then
  train_weights_if_needed
fi
if selection_policy_requires_weight; then
  require_path "$WEIGHT_FILE" "v0.2 learned marginal weight file"
fi
if should_build_fullpool_traces; then
  build_fullpool_traces
fi

export ATOM_ANCHOR_ROOT
export TRACE_ROOT
export WEIGHT_FILE
export QUALITY_AUDIT
export QUALITY_AUDIT_MODE
export DATASET
export LABEL_SCHEMA
export TRAIN_RAW
export VAL_RAW
export TEST_RAW
export BASE_CASE_NAME
export CASE_SUFFIX
export OUTPUT_ROOT
export LORA_SUFFIX
export CONFIG_PATH
export TRACE_TOP_K
export TRACE_PROMPT_STYLE
export EVIDENCE_TEXT_MODE
export EXPECTED_SELECTOR_NAME
export EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-}"
export PROMPT_EVIDENCE_POLICY
export PROMPT_EVIDENCE_MIN_COUNT
export PROMPT_EVIDENCE_MAX_COUNT
export PROMPT_EVIDENCE_TOKEN_BUDGET
export PROMPT_EVIDENCE_MAX_LENGTH_GUARD
export EVAL_SPLITS
export RUN_TAU_EVAL
export TAUS
export RUN_LABEL
export RUN_HEADER_LABEL
export WRAPPER_SWANLAB_PROJECT
export MODEL_PATH
export DEEPSPEED_CONFIG
export NPROC_PER_NODE
export SFT_GRADIENT_ACCUMULATION_STEPS
export SFT_LEARNING_RATE
export SFT_NUM_TRAIN_EPOCHS
export SFT_EVAL_STEPS
export SFT_SAVE_STEPS
export SFT_EARLY_STOPPING_PATIENCE
export SFT_EARLY_STOPPING_METRIC
export LORA_R
export LORA_ALPHA
export LORA_DROPOUT
export LORA_BIAS
export CLASS_WEIGHTS
export LIAR_CLASS_WEIGHTS

bash "${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_1_mrec_min_lora_ebs16_lr2e5_ep12_eval100.sh"
