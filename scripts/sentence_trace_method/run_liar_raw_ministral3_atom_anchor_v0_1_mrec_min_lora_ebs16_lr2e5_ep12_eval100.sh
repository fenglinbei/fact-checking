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
QUALITY_AUDIT_MODE="${QUALITY_AUDIT_MODE:-full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sentence_trace_method}"
CASE_SUFFIX="${CASE_SUFFIX:-__atom_anchor_v0_1_mrec_min}"
DATASET="${DATASET:-liar_raw}"
LABEL_SCHEMA="${LABEL_SCHEMA:-liar6}"
BASE_CASE_NAME="${BASE_CASE_NAME:-${DATASET}__ministral3_8b}"
CASE_NAME="${CASE_NAME:-${BASE_CASE_NAME}${CASE_SUFFIX}}"
CASE_ROOT="${CASE_ROOT:-${OUTPUT_ROOT}/${CASE_NAME}}"
FINETUNE_MODE="${FINETUNE_MODE:-lora}"
case "$FINETUNE_MODE" in
  lora)
    LORA_SUFFIX="${LORA_SUFFIX:-_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
    LORA_ROOT="${LORA_ROOT:-${CASE_ROOT}${LORA_SUFFIX}}"
    TRAIN_CASE_ROOT="$LORA_ROOT"
    ;;
  fullft)
    LORA_SUFFIX="${LORA_SUFFIX-}"
    LORA_ROOT="${LORA_ROOT:-}"
    TRAIN_CASE_ROOT="$CASE_ROOT"
    ;;
  *) printf 'Unsupported FINETUNE_MODE=%s. Use lora or fullft.\n' "$FINETUNE_MODE" >&2; exit 2 ;;
esac

MODE="${MODE:-full}" # check|build|train|eval|full
DRY_RUN="${DRY_RUN:-false}"
REQUIRE_PROMPT_INPUT_IDS="${REQUIRE_PROMPT_INPUT_IDS:-true}"
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
RUN_LABEL="${RUN_LABEL:-atom-anchor-v0.1}"
RUN_HEADER_LABEL="${RUN_HEADER_LABEL:-${RUN_LABEL}-full}"

CONFIG_PATH="${CONFIG_PATH:-scripts/sentence_trace_method/configs/${BASE_CASE_NAME}.yaml}"
MODEL_PATH="${MODEL_PATH:-/data/models/Ministral-3-8B-Instruct-2512}"
if [[ "$DATASET" == "rawfc" ]]; then
  DEFAULT_TRAIN_RAW="data/raw/RAWFC/train.json"
  DEFAULT_VAL_RAW="data/raw/RAWFC/val.json"
  DEFAULT_TEST_RAW="data/raw/RAWFC/test.json"
else
  DEFAULT_TRAIN_RAW="data/raw/LIAR-RAW/train.json"
  DEFAULT_VAL_RAW="data/raw/LIAR-RAW/val.json"
  DEFAULT_TEST_RAW="data/raw/LIAR-RAW/test.json"
fi
TRAIN_RAW="${TRAIN_RAW:-$DEFAULT_TRAIN_RAW}"
VAL_RAW="${VAL_RAW:-$DEFAULT_VAL_RAW}"
TEST_RAW="${TEST_RAW:-$DEFAULT_TEST_RAW}"

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
DEFAULT_CLASS_WEIGHTS="pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8"
CLASS_WEIGHTS="${CLASS_WEIGHTS:-${LIAR_CLASS_WEIGHTS:-$DEFAULT_CLASS_WEIGHTS}}"
LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-$CLASS_WEIGHTS}"
SWANLAB_PROJECT="${WRAPPER_SWANLAB_PROJECT:-${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-atom-anchor}}"
export -n SWANLAB_PROJECT 2>/dev/null || true

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

visible_cuda_device_count() {
  local devices="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -z "$devices" || "$devices" == "NoDevFiles" ]]; then
    printf '\n'
    return 0
  fi
  local raw_device device count=0
  IFS=',' read -r -a cuda_device_array <<< "$devices"
  for raw_device in "${cuda_device_array[@]}"; do
    device="${raw_device// /}"
    [[ -z "$device" ]] && continue
    count=$((count + 1))
  done
  printf '%s\n' "$count"
}

check_distributed_device_request() {
  if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || [[ "$NPROC_PER_NODE" -lt 1 ]]; then
    printf 'NPROC_PER_NODE must be a positive integer: %s\n' "$NPROC_PER_NODE" >&2
    exit 2
  fi

  local visible_count
  visible_count="$(visible_cuda_device_count)"
  if [[ -n "$visible_count" && "$visible_count" -gt 0 && "$NPROC_PER_NODE" -gt "$visible_count" ]]; then
    printf 'Requested NPROC_PER_NODE=%s but CUDA_VISIBLE_DEVICES exposes only %s device(s): %s\n' \
      "$NPROC_PER_NODE" "$visible_count" "${CUDA_VISIBLE_DEVICES:-}" >&2
    printf 'Set NPROC_PER_NODE<=%s, expand CUDA_VISIBLE_DEVICES, or request a matching GPU allocation before launching training.\n' \
      "$visible_count" >&2
    exit 2
  fi
}

check_quality() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '[%s] DRY_RUN skips artifact quality audit: %s mode=%s\n' "$RUN_LABEL" "$QUALITY_AUDIT" "$QUALITY_AUDIT_MODE"
    return 0
  fi
  "$PYTHON_BIN" - "$QUALITY_AUDIT" "$QUALITY_AUDIT_MODE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
mode = sys.argv[2]
if not path.exists():
    raise SystemExit(f"missing quality audit: {path}")
if mode not in {"full", "source_only"}:
    raise SystemExit(f"unsupported QUALITY_AUDIT_MODE={mode}")
report = json.loads(path.read_text())
required_fields = ["retrieval_trace", "atom_union_pool", "candidate_pool", "annotations", "features"]
if mode == "full":
    required_fields.extend(["mrec", "verifier"])
for split in ("train", "val", "test"):
    item = (report.get("splits") or {}).get(split) or {}
    counts = item.get("counts") or {}
    expected = counts.get("claim_atoms")
    for field in required_fields:
        if counts.get(field) != expected:
            raise SystemExit(f"{split}: count mismatch {field}={counts.get(field)} expected={expected}")
    if item.get("missing_annotations"):
        raise SystemExit(f"{split}: missing annotations remain")
    if item.get("feature_fallback_missing_annotation") != 0:
        raise SystemExit(f"{split}: fallback_missing_annotation={item.get('feature_fallback_missing_annotation')}")
    parse_counts = item.get("feature_parse_status_counts") or {}
    if parse_counts != {"ok": expected}:
        raise SystemExit(f"{split}: unexpected parse counts {parse_counts}")
    if mode == "full":
        mrec = item.get("mrec") or {}
        if mrec.get("cue_mismatch_count_sampled_cap5") != 0:
            raise SystemExit(f"{split}: MREC cue mismatch sample is not empty")
        prompt_quality = item.get("verifier_prompt_quality") or {}
        if prompt_quality.get("metadata_field_leaks"):
            raise SystemExit(f"{split}: prompt metadata leaks {prompt_quality.get('metadata_field_leaks')}")
        if prompt_quality.get("check_count_mismatch") != 0:
            raise SystemExit(f"{split}: Check count mismatch")
build_report = ((report.get("verifier") or {}).get("build_report") or {})
if build_report and build_report.get("val_only"):
    raise SystemExit("quality audit build_report is val_only=true")
print(f"quality audit ok: {path} mode={mode}")
PY
}

validate_prompt_input_ids() {
  if [[ "$REQUIRE_PROMPT_INPUT_IDS" != "true" || "$DRY_RUN" == "true" ]]; then
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
    printf '[%s] verifier build exists: %s; set FORCE_BUILD=true to rebuild.\n' "$RUN_LABEL" "$CASE_ROOT"
    validate_prompt_input_ids
    return 0
  fi
  require_path "${TRACE_ROOT}/selection_trace_train.jsonl" "train MREC trace"
  require_path "${TRACE_ROOT}/selection_trace_val.jsonl" "val MREC trace"
  require_path "${TRACE_ROOT}/selection_trace_test.jsonl" "test MREC trace"
  local prompt_evidence_args=()
  if [[ -n "${PROMPT_EVIDENCE_POLICY:-}" ]]; then
    prompt_evidence_args+=(--prompt-evidence-policy "$PROMPT_EVIDENCE_POLICY")
  fi
  if [[ -n "${PROMPT_EVIDENCE_MIN_COUNT:-}" ]]; then
    prompt_evidence_args+=(--prompt-evidence-min-count "$PROMPT_EVIDENCE_MIN_COUNT")
  fi
  if [[ -n "${PROMPT_EVIDENCE_MAX_COUNT:-}" ]]; then
    prompt_evidence_args+=(--prompt-evidence-max-count "$PROMPT_EVIDENCE_MAX_COUNT")
  fi
  if [[ -n "${PROMPT_EVIDENCE_TOKEN_BUDGET:-}" ]]; then
    prompt_evidence_args+=(--prompt-evidence-token-budget "$PROMPT_EVIDENCE_TOKEN_BUDGET")
  fi
  if [[ -n "${PROMPT_EVIDENCE_MAX_LENGTH_GUARD:-}" ]]; then
    prompt_evidence_args+=(--prompt-evidence-max-length-guard "$PROMPT_EVIDENCE_MAX_LENGTH_GUARD")
  fi
  run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/build_trace_verifier_data.py \
    --config "$CONFIG_PATH" \
    --train-trace "${TRACE_ROOT}/selection_trace_train.jsonl" \
    --val-trace "${TRACE_ROOT}/selection_trace_val.jsonl" \
    --test-trace "${TRACE_ROOT}/selection_trace_test.jsonl" \
    --train-raw "$TRAIN_RAW" \
    --val-raw "$VAL_RAW" \
    --test-raw "$TEST_RAW" \
    --dataset "$DATASET" \
    --label-schema "$LABEL_SCHEMA" \
    --output-dir "$CASE_ROOT" \
    --selection-mode trace \
    --trace-prompt-style "$TRACE_PROMPT_STYLE" \
    --evidence-text-mode "$EVIDENCE_TEXT_MODE" \
    --expected-selector-name "$EXPECTED_SELECTOR_NAME" \
    --top-k "$TRACE_TOP_K" \
    "${prompt_evidence_args[@]}" \
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
  IFS=',' read -r -a class_weight_array <<< "$CLASS_WEIGHTS"
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
  require_path "${TRAIN_CASE_ROOT}/train.resolved.yaml" "${FINETUNE_MODE} train config"
  if training_complete "$TRAIN_CASE_ROOT" && [[ "$FORCE_TRAIN" != "true" ]]; then
    printf '[%s] %s training already complete: %s; set FORCE_TRAIN=true to rerun.\n' "$RUN_LABEL" "$FINETUNE_MODE" "$TRAIN_CASE_ROOT"
    return 0
  fi
  check_distributed_device_request
  run_cmd env \
    -u SWANLAB_PROJECT \
    SAVE_LATEST_TRAIN_STATE="$SAVE_LATEST_TRAIN_STATE" \
    RESUME_LATEST_TRAIN_STATE="$RESUME_LATEST_TRAIN_STATE" \
    "$ACCELERATE_BIN" launch \
    --num_processes "$NPROC_PER_NODE" \
    --num_machines "$NUM_MACHINES" \
    --mixed_precision "$MIXED_PRECISION" \
    --use_deepspeed \
    --deepspeed_config_file "$DEEPSPEED_CONFIG" \
    -m sft.label_token_trainer \
    --config "${TRAIN_CASE_ROOT}/train.resolved.yaml"
}

eval_lora() {
  require_path "${TRAIN_CASE_ROOT}/train.resolved.yaml" "${FINETUNE_MODE} train config"
  local split checkpoint metrics_path
  IFS=',' read -r -a split_array <<< "$EVAL_SPLITS"
  IFS=',' read -r -a checkpoint_array <<< "$CHECKPOINTS"
  for split in "${split_array[@]}"; do
    split="${split// /}"
    [[ -z "$split" ]] && continue
    for checkpoint in "${checkpoint_array[@]}"; do
      checkpoint="${checkpoint// /}"
      [[ -z "$checkpoint" ]] && continue
      metrics_path="${TRAIN_CASE_ROOT}/eval/${split}/${checkpoint}/metrics.json"
      if [[ -f "$metrics_path" && "$FORCE_EVAL" != "true" ]]; then
        printf '[%s] eval exists: %s; set FORCE_EVAL=true to rerun.\n' "$RUN_LABEL" "$metrics_path"
        continue
      fi
      run_cmd "$PYTHON_BIN" -m sft.label_token_infer \
        --run-dir "${TRAIN_CASE_ROOT}/train" \
        --checkpoint "$checkpoint" \
        --split "$split" \
        --config "${TRAIN_CASE_ROOT}/train.resolved.yaml"
    done
  done
}

run_tau_eval() {
  if ! should_run_tau_eval; then
    return 0
  fi
  run_cmd env \
    PYTHON_BIN="$PYTHON_BIN" \
    CASE_ROOT="$TRAIN_CASE_ROOT" \
    SPLITS="$TAU_SPLITS" \
    CHECKPOINTS="$CHECKPOINTS" \
    TAUS="$TAUS" \
    LOGIT_ADJUST_MODE=on \
    FORCE_EVAL="$FORCE_EVAL" \
    DRY_RUN="$DRY_RUN" \
    bash scripts/sentence_trace_method/run_lora_label_token_logit_adjust_eval_only.sh
}

printf '[%s] CASE_NAME=%s CASE_SUFFIX=%s DATASET=%s LABEL_SCHEMA=%s ATOM_ANCHOR_ROOT=%s TRACE_ROOT=%s QUALITY_AUDIT_MODE=%s MODE=%s FINETUNE_MODE=%s TRACE_PROMPT_STYLE=%s EVIDENCE_TEXT_MODE=%s TRACE_TOP_K=%s EXPECTED_SELECTOR_NAME=%s TRAIN_CASE_ROOT=%s LORA_ROOT=%s EVAL_SPLITS=%s RUN_TAU_EVAL=%s REQUIRE_PROMPT_INPUT_IDS=%s SFT_LEARNING_RATE=%s SFT_NUM_TRAIN_EPOCHS=%s SFT_EVAL_STEPS=%s SFT_SAVE_STEPS=%s SFT_EARLY_STOPPING_PATIENCE=%s DEEPSPEED_CONFIG=%s NPROC_PER_NODE=%s CLASS_WEIGHTS=%s CUDA_VISIBLE_DEVICES=%s\n' \
  "$RUN_HEADER_LABEL" \
  "$CASE_NAME" "$CASE_SUFFIX" "$DATASET" "$LABEL_SCHEMA" "$ATOM_ANCHOR_ROOT" "$TRACE_ROOT" "$QUALITY_AUDIT_MODE" "$MODE" "$FINETUNE_MODE" "$TRACE_PROMPT_STYLE" "$EVIDENCE_TEXT_MODE" "$TRACE_TOP_K" "$EXPECTED_SELECTOR_NAME" "$TRAIN_CASE_ROOT" "$LORA_ROOT" "$EVAL_SPLITS" "$RUN_TAU_EVAL" "$REQUIRE_PROMPT_INPUT_IDS" "$SFT_LEARNING_RATE" "$SFT_NUM_TRAIN_EPOCHS" "$SFT_EVAL_STEPS" "$SFT_SAVE_STEPS" "$SFT_EARLY_STOPPING_PATIENCE" "$DEEPSPEED_CONFIG" "$NPROC_PER_NODE" "$CLASS_WEIGHTS" "${CUDA_VISIBLE_DEVICES:-<unset>}"

case "$MODE" in
  check)
    check_quality
    ;;
  build)
    check_quality
    build_verifier_data
    if [[ "$FINETUNE_MODE" == "lora" ]]; then
      prepare_lora_config
    fi
    ;;
  train)
    check_quality
    if [[ "$FINETUNE_MODE" == "lora" && ! -f "${LORA_ROOT}/train.resolved.yaml" ]]; then
      prepare_lora_config
    fi
    train_lora
    ;;
  eval)
    check_quality
    if [[ "$FINETUNE_MODE" == "lora" && ! -f "${LORA_ROOT}/train.resolved.yaml" ]]; then
      prepare_lora_config
    fi
    eval_lora
    run_tau_eval
    ;;
  full)
    check_quality
    build_verifier_data
    if [[ "$FINETUNE_MODE" == "lora" ]]; then
      prepare_lora_config
    fi
    train_lora
    eval_lora
    run_tau_eval
    ;;
  *)
    printf 'Unsupported MODE=%s. Use check, build, train, eval, or full.\n' "$MODE" >&2
    exit 2
    ;;
esac
