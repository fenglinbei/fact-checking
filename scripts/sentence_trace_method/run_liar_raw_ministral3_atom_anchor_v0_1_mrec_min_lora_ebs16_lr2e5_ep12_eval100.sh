#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="src:${PYTHONPATH}"
else
  export PYTHONPATH="src"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc-gemma4/bin/python" ]]; then
    PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-gemma4/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
if [[ -z "${ACCELERATE_BIN:-}" ]]; then
  py_dir="$(dirname "$PYTHON_BIN")"
  if [[ -x "${py_dir}/accelerate" ]]; then
    ACCELERATE_BIN="${py_dir}/accelerate"
  else
    ACCELERATE_BIN="accelerate"
  fi
fi

ATOM_ANCHOR_ROOT="${ATOM_ANCHOR_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1}"
TRACE_ROOT="${TRACE_ROOT:-${ATOM_ANCHOR_ROOT}/05_mrec}"
QUALITY_AUDIT="${QUALITY_AUDIT:-${ATOM_ANCHOR_ROOT}/quality_audit_after_fix.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
CASE_SUFFIX="${CASE_SUFFIX:-__atom_anchor_v0_1_mrec_min}"
CASE_NAME="${CASE_NAME:-liar_raw__ministral3_8b${CASE_SUFFIX}}"
CASE_ROOT="${CASE_ROOT:-${OUTPUT_ROOT}/${CASE_NAME}}"
LORA_SUFFIX="${LORA_SUFFIX:-_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
LORA_ROOT="${LORA_ROOT:-${CASE_ROOT}${LORA_SUFFIX}}"

MODE="${MODE:-full}" # check|build|train|eval|full
DRY_RUN="${DRY_RUN:-false}"
FORCE_BUILD="${FORCE_BUILD:-auto}"
FORCE_LORA_CONFIG="${FORCE_LORA_CONFIG:-false}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"
FORCE_EVAL="${FORCE_EVAL:-false}"
EVAL_SPLITS="${EVAL_SPLITS:-val,test}"
CHECKPOINTS="${CHECKPOINTS:-best}"
TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-mrec_min}"
EVIDENCE_TEXT_MODE="${EVIDENCE_TEXT_MODE:-full}"
TRACE_TOP_K="${TRACE_TOP_K:-10}"
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-mrec_greedy_transition_v0_1}"
EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-}"

CONFIG_PATH="${CONFIG_PATH:-scripts/sentence_trace_method/configs/liar_raw__ministral3_8b.yaml}"
MODEL_PATH="${MODEL_PATH:-/data/models/Ministral-3-8B-Instruct-2512}"
TRAIN_RAW="${TRAIN_RAW:-data/raw/LIAR-RAW/train.json}"
VAL_RAW="${VAL_RAW:-data/raw/LIAR-RAW/val.json}"
TEST_RAW="${TEST_RAW:-data/raw/LIAR-RAW/test.json}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz1_ga4.json}"
SAVE_LATEST_TRAIN_STATE="${SAVE_LATEST_TRAIN_STATE:-true}"
RESUME_LATEST_TRAIN_STATE="${RESUME_LATEST_TRAIN_STATE:-$SAVE_LATEST_TRAIN_STATE}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_BIAS="${LORA_BIAS:-none}"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-4}"
SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-2e-5}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-12}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-$SFT_EVAL_STEPS}"
SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-12}"
SFT_EARLY_STOPPING_METRIC="${SFT_EARLY_STOPPING_METRIC:-macro_f1}"
LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-atom-anchor}"

RUN_TAU_EVAL="${RUN_TAU_EVAL:-auto}"
TAU_SPLITS="${TAU_SPLITS:-$EVAL_SPLITS}"
TAUS="${TAUS:-0.75}"

run_cmd() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

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

training_complete() {
  local marker="$1/train/training_complete.json"
  [[ -f "$marker" ]] && grep -Eq '"completed"[[:space:]]*:[[:space:]]*true' "$marker"
}

should_run_tau_eval() {
  case "$RUN_TAU_EVAL" in
    true) return 0 ;;
    false) return 1 ;;
    auto)
      case "$MODE" in
        full|eval) return 0 ;;
        *) return 1 ;;
      esac
      ;;
    *) printf 'Unsupported RUN_TAU_EVAL=%s. Use true, false, or auto.\n' "$RUN_TAU_EVAL" >&2; exit 2 ;;
  esac
}

check_quality() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '[atom-anchor-v0.1] DRY_RUN skips artifact quality audit: %s\n' "$QUALITY_AUDIT"
    return 0
  fi
  "$PYTHON_BIN" - "$QUALITY_AUDIT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"missing quality audit: {path}")
report = json.loads(path.read_text())
for split in ("train", "val", "test"):
    item = (report.get("splits") or {}).get(split) or {}
    counts = item.get("counts") or {}
    expected = counts.get("claim_atoms")
    for field in ("retrieval_trace", "atom_union_pool", "candidate_pool", "annotations", "features", "mrec", "verifier"):
        if counts.get(field) != expected:
            raise SystemExit(f"{split}: count mismatch {field}={counts.get(field)} expected={expected}")
    if item.get("missing_annotations"):
        raise SystemExit(f"{split}: missing annotations remain")
    if item.get("feature_fallback_missing_annotation") != 0:
        raise SystemExit(f"{split}: fallback_missing_annotation={item.get('feature_fallback_missing_annotation')}")
    parse_counts = item.get("feature_parse_status_counts") or {}
    if parse_counts != {"ok": expected}:
        raise SystemExit(f"{split}: unexpected parse counts {parse_counts}")
    mrec = item.get("mrec") or {}
    if mrec.get("cue_mismatch_count_sampled_cap5") != 0:
        raise SystemExit(f"{split}: MREC cue mismatch sample is not empty")
    prompt_quality = item.get("verifier_prompt_quality") or {}
    if prompt_quality.get("metadata_field_leaks"):
        raise SystemExit(f"{split}: prompt metadata leaks {prompt_quality.get('metadata_field_leaks')}")
    if prompt_quality.get("check_count_mismatch") != 0:
        raise SystemExit(f"{split}: Check count mismatch")
build_report = ((report.get("verifier") or {}).get("build_report") or {})
if build_report.get("val_only"):
    raise SystemExit("quality audit build_report is val_only=true")
print(f"quality audit ok: {path}")
PY
}

validate_prompt_input_ids() {
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  "$PYTHON_BIN" - "$CASE_ROOT" <<'PY'
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
print(f"prompt_input_ids ok: {case_root / 'build'}")
PY
}

build_verifier_data() {
  if [[ "$FORCE_BUILD" != "true" && -f "${CASE_ROOT}/train.resolved.yaml" && -f "${CASE_ROOT}/build/build_report.json" ]]; then
    printf '[atom-anchor-v0.1] verifier build exists: %s; set FORCE_BUILD=true to rebuild.\n' "$CASE_ROOT"
    validate_prompt_input_ids
    return 0
  fi
  require_path "${TRACE_ROOT}/selection_trace_train.jsonl" "train MREC trace"
  require_path "${TRACE_ROOT}/selection_trace_val.jsonl" "val MREC trace"
  require_path "${TRACE_ROOT}/selection_trace_test.jsonl" "test MREC trace"
  run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/build_trace_verifier_data.py \
    --config "$CONFIG_PATH" \
    --train-trace "${TRACE_ROOT}/selection_trace_train.jsonl" \
    --val-trace "${TRACE_ROOT}/selection_trace_val.jsonl" \
    --test-trace "${TRACE_ROOT}/selection_trace_test.jsonl" \
    --train-raw "$TRAIN_RAW" \
    --val-raw "$VAL_RAW" \
    --test-raw "$TEST_RAW" \
    --dataset liar_raw \
    --label-schema liar6 \
    --output-dir "$CASE_ROOT" \
    --selection-mode trace \
    --trace-prompt-style "$TRACE_PROMPT_STYLE" \
    --evidence-text-mode "$EVIDENCE_TEXT_MODE" \
    --expected-selector-name "$EXPECTED_SELECTOR_NAME" \
    --top-k "$TRACE_TOP_K" \
    --expected-chunk-mmr-fingerprint "$EXPECTED_CHUNK_MMR_FINGERPRINT" \
    --prompt-model-name-or-path "$MODEL_PATH" \
    --train-model-name-or-path "$MODEL_PATH" \
    --no-progress
  validate_prompt_input_ids
}

prepare_lora_config() {
  require_path "${CASE_ROOT}/train.resolved.yaml" "verifier train config"
  local cmd=("$PYTHON_BIN" scripts/sentence_trace_method/prepare_lora_config.py
    --source-config "${CASE_ROOT}/train.resolved.yaml"
    --output-root "$LORA_ROOT"
    --experiment-name "$(basename "$LORA_ROOT")"
    --swanlab-project "$SWANLAB_PROJECT"
    --r "$LORA_R"
    --alpha "$LORA_ALPHA"
    --dropout "$LORA_DROPOUT"
    --bias "$LORA_BIAS"
    --deepspeed-config "$DEEPSPEED_CONFIG"
    --gradient-accumulation-steps "$SFT_GRADIENT_ACCUMULATION_STEPS"
    --learning-rate "$SFT_LEARNING_RATE"
    --num-train-epochs "$SFT_NUM_TRAIN_EPOCHS"
    --eval-steps "$SFT_EVAL_STEPS"
    --save-steps "$SFT_SAVE_STEPS"
    --early-stopping-patience "$SFT_EARLY_STOPPING_PATIENCE"
    --early-stopping-metric "$SFT_EARLY_STOPPING_METRIC")
  IFS=',' read -r -a class_weight_array <<< "$LIAR_CLASS_WEIGHTS"
  local raw_weight class_weight
  for raw_weight in "${class_weight_array[@]}"; do
    class_weight="${raw_weight// /}"
    [[ -z "$class_weight" ]] && continue
    cmd+=(--class-weight "$class_weight")
  done
  if [[ "$FORCE_LORA_CONFIG" == "true" ]]; then
    cmd+=(--force)
  fi
  run_cmd "${cmd[@]}"
}

train_lora() {
  require_path "${LORA_ROOT}/train.resolved.yaml" "LoRA train config"
  if training_complete "$LORA_ROOT" && [[ "$FORCE_TRAIN" != "true" ]]; then
    printf '[atom-anchor-v0.1] LoRA training already complete: %s; set FORCE_TRAIN=true to rerun.\n' "$LORA_ROOT"
    return 0
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
    --config "${LORA_ROOT}/train.resolved.yaml"
}

eval_lora() {
  require_path "${LORA_ROOT}/train.resolved.yaml" "LoRA train config"
  local split checkpoint metrics_path
  IFS=',' read -r -a split_array <<< "$EVAL_SPLITS"
  IFS=',' read -r -a checkpoint_array <<< "$CHECKPOINTS"
  for split in "${split_array[@]}"; do
    split="${split// /}"
    [[ -z "$split" ]] && continue
    for checkpoint in "${checkpoint_array[@]}"; do
      checkpoint="${checkpoint// /}"
      [[ -z "$checkpoint" ]] && continue
      metrics_path="${LORA_ROOT}/eval/${split}/${checkpoint}/metrics.json"
      if [[ -f "$metrics_path" && "$FORCE_EVAL" != "true" ]]; then
        printf '[atom-anchor-v0.1] eval exists: %s; set FORCE_EVAL=true to rerun.\n' "$metrics_path"
        continue
      fi
      run_cmd "$PYTHON_BIN" -m sft.label_token_infer \
        --run-dir "${LORA_ROOT}/train" \
        --checkpoint "$checkpoint" \
        --split "$split" \
        --config "${LORA_ROOT}/train.resolved.yaml"
    done
  done
}

run_tau_eval() {
  if ! should_run_tau_eval; then
    return 0
  fi
  run_cmd env \
    PYTHON_BIN="$PYTHON_BIN" \
    CASE_ROOT="$LORA_ROOT" \
    SPLITS="$TAU_SPLITS" \
    CHECKPOINTS="$CHECKPOINTS" \
    TAUS="$TAUS" \
    LOGIT_ADJUST_MODE=on \
    FORCE_EVAL="$FORCE_EVAL" \
    DRY_RUN="$DRY_RUN" \
    bash scripts/sentence_trace_method/run_lora_label_token_logit_adjust_eval_only.sh
}

printf '[atom-anchor-v0.1-full] CASE_NAME=%s ATOM_ANCHOR_ROOT=%s TRACE_ROOT=%s MODE=%s TRACE_PROMPT_STYLE=%s EVIDENCE_TEXT_MODE=%s TRACE_TOP_K=%s LORA_ROOT=%s EVAL_SPLITS=%s RUN_TAU_EVAL=%s SFT_LEARNING_RATE=%s SFT_NUM_TRAIN_EPOCHS=%s SFT_EVAL_STEPS=%s DEEPSPEED_CONFIG=%s\n' \
  "$CASE_NAME" "$ATOM_ANCHOR_ROOT" "$TRACE_ROOT" "$MODE" "$TRACE_PROMPT_STYLE" "$EVIDENCE_TEXT_MODE" "$TRACE_TOP_K" "$LORA_ROOT" "$EVAL_SPLITS" "$RUN_TAU_EVAL" "$SFT_LEARNING_RATE" "$SFT_NUM_TRAIN_EPOCHS" "$SFT_EVAL_STEPS" "$DEEPSPEED_CONFIG"

case "$MODE" in
  check)
    check_quality
    ;;
  build)
    check_quality
    build_verifier_data
    prepare_lora_config
    ;;
  train)
    check_quality
    if [[ ! -f "${LORA_ROOT}/train.resolved.yaml" ]]; then
      prepare_lora_config
    fi
    train_lora
    ;;
  eval)
    check_quality
    if [[ ! -f "${LORA_ROOT}/train.resolved.yaml" ]]; then
      prepare_lora_config
    fi
    eval_lora
    run_tau_eval
    ;;
  full)
    check_quality
    build_verifier_data
    prepare_lora_config
    train_lora
    eval_lora
    run_tau_eval
    ;;
  *)
    printf 'Unsupported MODE=%s. Use check, build, train, eval, or full.\n' "$MODE" >&2
    exit 2
    ;;
esac
