#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

ATOM_ANCHOR_ROOT="${ATOM_ANCHOR_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1}"
SOURCE_FEATURE_ROOT="${SOURCE_FEATURE_ROOT:-${ATOM_ANCHOR_ROOT}/04_evidence_map}"
TRACE_ROOT="${TRACE_ROOT:-${ATOM_ANCHOR_ROOT}/05_mrec_v0_2_learned_marginal_reward}"
REWARD_CACHE_DIR="${REWARD_CACHE_DIR:-${TRACE_ROOT}/reward_cache}"
WEIGHT_DIR="${WEIGHT_DIR:-${TRACE_ROOT}/weights}"
WEIGHT_FILE="${WEIGHT_FILE:-${WEIGHT_DIR}/weights.json}"
PRIOR_WEIGHT_FILE="${PRIOR_WEIGHT_FILE:-${ATOM_ANCHOR_ROOT}/05_mrec_v0_2_learned_marginal_proxy/weights/weights.json}"
TEACHER_RUN_DIR="${TEACHER_RUN_DIR:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_top5_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/train}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-best}"
BASE_MODEL="${BASE_MODEL:-/data/models/Ministral-3-8B-Instruct-2512}"
SCORING_BACKEND="${SCORING_BACKEND:-vllm}"
TRANSFORMERS_PROMPT_BATCH_SIZE="${TRANSFORMERS_PROMPT_BATCH_SIZE:-24}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-1032}"
VLLM_PROMPT_BATCH_SIZE="${VLLM_PROMPT_BATCH_SIZE:-6000}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-false}"

if [[ "$SCORING_BACKEND" == "vllm" || "$SCORING_BACKEND" == "auto" ]]; then
  export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
fi

EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-mrec_greedy_transition_v0_2_learned_marginal_reward}"
EXPECTED_ADAPTIVE_POLICY="${EXPECTED_ADAPTIVE_POLICY:-learned_marginal_reward_v0_2}"
EXPECTED_SELECTION_POLICY="${EXPECTED_SELECTION_POLICY:-learned_marginal_reward}"
SOURCE_SELECTOR_NAME="${SOURCE_SELECTOR_NAME:-v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10}"

MODE="${MODE:-full}" # check|cache|weights|traces|build|train|eval|full
DRY_RUN="${DRY_RUN:-false}"
FORCE_REWARD_CACHE="${FORCE_REWARD_CACHE:-false}"
FORCE_REWARD_WEIGHTS="${FORCE_REWARD_WEIGHTS:-false}"
FORCE_MREC_BUILD="${FORCE_MREC_BUILD:-false}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
REWARD_CACHE_SPLITS="${REWARD_CACHE_SPLITS:-train,val}"
MREC_SPLITS="${MREC_SPLITS:-train,val,test}"
REWARD_CANDIDATE_TOP_N="${REWARD_CANDIDATE_TOP_N:-20}"
REWARD_ROLLOUT_STEPS="${REWARD_ROLLOUT_STEPS:-10}"
REWARD_ROLLIN_POLICY="${REWARD_ROLLIN_POLICY:-learned_marginal_proxy}"
REWARD_ROLLIN_STOP_THRESHOLD="${REWARD_ROLLIN_STOP_THRESHOLD:-0.0}"
REWARD_TARGET_RESOLVED_RATE="${REWARD_TARGET_RESOLVED_RATE:-1.0}"
REWARD_EPOCHS="${REWARD_EPOCHS:-30}"
REWARD_LEARNING_RATE="${REWARD_LEARNING_RATE:-0.03}"
REWARD_PRIOR_WEIGHT="${REWARD_PRIOR_WEIGHT:-0.02}"
REWARD_MAX_PAIRS_PER_GROUP="${REWARD_MAX_PAIRS_PER_GROUP:-64}"

TRACE_CANDIDATE_TOP_N="${TRACE_CANDIDATE_TOP_N:-20}"
TRACE_MIN_STEPS="${TRACE_MIN_STEPS:-5}"
TRACE_MAX_STEPS="${TRACE_MAX_STEPS:-10}"
TRACE_TARGET_RESOLVED_RATE="${TRACE_TARGET_RESOLVED_RATE:-1.0}"
TRACE_STOP_THRESHOLD="${TRACE_STOP_THRESHOLD:-0.0}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/data/liaozijie/conda/accelerate-fc-gemma4/bin/python" ]]; then
    export PYTHON_BIN="/data/liaozijie/conda/accelerate-fc-gemma4/bin/python"
  else
    export PYTHON_BIN="python"
  fi
fi

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

sample_args() {
  if [[ "$SAMPLE_LIMIT" != "0" ]]; then
    printf '%s\n' --sample-limit "$SAMPLE_LIMIT"
  fi
}

build_reward_cache() {
  local split input_path raw_path output_path
  IFS=',' read -r -a split_array <<< "$REWARD_CACHE_SPLITS"
  for split in "${split_array[@]}"; do
    split="${split// /}"
    [[ -z "$split" ]] && continue
    input_path="${SOURCE_FEATURE_ROOT}/candidate_evidence_map_features_${split}.jsonl"
    raw_path="data/raw/LIAR-RAW/${split}.json"
    output_path="${REWARD_CACHE_DIR}/reward_records_${split}.jsonl"
    if [[ -f "$output_path" && "$FORCE_REWARD_CACHE" != "true" ]]; then
      printf '[mrec-reward] reuse reward cache: %s\n' "$output_path"
      continue
    fi
    require_path "$input_path" "${split} atom-anchor evidence-map features"
    require_path "$raw_path" "${split} raw data"
    require_path "${TEACHER_RUN_DIR}/${TEACHER_CHECKPOINT}/adapter_model.safetensors" "teacher LoRA adapter"
    require_path "${TEACHER_RUN_DIR}/label_token_ce_meta.json" "teacher label token metadata"
    local extra_args=()
    if [[ "$SAMPLE_LIMIT" != "0" ]]; then
      extra_args=(--sample-limit "$SAMPLE_LIMIT")
    fi
    local vllm_extra_args=()
    if [[ "$VLLM_ENFORCE_EAGER" == "true" ]]; then
      vllm_extra_args=(--vllm-enforce-eager)
    fi
    run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/build_mrec_reward_cache.py \
      --input "$input_path" \
      --raw "$raw_path" \
      --output-dir "$REWARD_CACHE_DIR" \
      --split "$split" \
      --candidate-top-n "$REWARD_CANDIDATE_TOP_N" \
      --rollout-steps "$REWARD_ROLLOUT_STEPS" \
      --rollin-policy "$REWARD_ROLLIN_POLICY" \
      --rollin-weight-file "$PRIOR_WEIGHT_FILE" \
      --rollin-stop-threshold "$REWARD_ROLLIN_STOP_THRESHOLD" \
      --target-resolved-rate "$REWARD_TARGET_RESOLVED_RATE" \
      --teacher-run-dir "$TEACHER_RUN_DIR" \
      --teacher-checkpoint "$TEACHER_CHECKPOINT" \
      --base-model "$BASE_MODEL" \
      --scoring-backend "$SCORING_BACKEND" \
      --vllm-tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
      --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --vllm-dtype "$VLLM_DTYPE" \
      --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
      --vllm-prompt-batch-size "$VLLM_PROMPT_BATCH_SIZE" \
      "${vllm_extra_args[@]}" \
      --transformers-prompt-batch-size "$TRANSFORMERS_PROMPT_BATCH_SIZE" \
      --prompt-model-name-or-path "$BASE_MODEL" \
      --trace-prompt-style mrec_min \
      "${extra_args[@]}"
  done
}

train_reward_weights() {
  if [[ -f "$WEIGHT_FILE" && "$FORCE_REWARD_WEIGHTS" != "true" ]]; then
    printf '[mrec-reward] reuse reward weights: %s\n' "$WEIGHT_FILE"
    return 0
  fi
  require_path "${REWARD_CACHE_DIR}/reward_records_train.jsonl" "train reward cache"
  require_path "${REWARD_CACHE_DIR}/reward_records_val.jsonl" "val reward cache"
  require_path "$PRIOR_WEIGHT_FILE" "prior proxy weight file"
  run_cmd "$PYTHON_BIN" scripts/phase5_selectors/train/train_mrec_learned_marginal_reward.py \
    --train-reward-input "${REWARD_CACHE_DIR}/reward_records_train.jsonl" \
    --val-reward-input "${REWARD_CACHE_DIR}/reward_records_val.jsonl" \
    --output-dir "$WEIGHT_DIR" \
    --prior-weight-file "$PRIOR_WEIGHT_FILE" \
    --epochs "$REWARD_EPOCHS" \
    --learning-rate "$REWARD_LEARNING_RATE" \
    --prior-weight "$REWARD_PRIOR_WEIGHT" \
    --max-pairs-per-group "$REWARD_MAX_PAIRS_PER_GROUP"
}

build_reward_traces() {
  local split input_path output_trace
  IFS=',' read -r -a split_array <<< "$MREC_SPLITS"
  for split in "${split_array[@]}"; do
    split="${split// /}"
    [[ -z "$split" ]] && continue
    input_path="${SOURCE_FEATURE_ROOT}/candidate_evidence_map_features_${split}.jsonl"
    output_trace="${TRACE_ROOT}/selection_trace_${split}.jsonl"
    if [[ -f "$output_trace" && "$FORCE_MREC_BUILD" != "true" ]]; then
      printf '[mrec-reward] reuse reward trace: %s\n' "$output_trace"
      continue
    fi
    require_path "$input_path" "${split} atom-anchor evidence-map features"
    require_path "$WEIGHT_FILE" "reward weight file"
    local extra_args=()
    if [[ "$SAMPLE_LIMIT" != "0" ]]; then
      extra_args=(--sample-limit "$SAMPLE_LIMIT")
    fi
    run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/build_mrec_traces.py \
      --input "$input_path" \
      --output-dir "$TRACE_ROOT" \
      --split "$split" \
      --candidate-top-n "$TRACE_CANDIDATE_TOP_N" \
      --max-steps "$TRACE_MAX_STEPS" \
      --min-steps "$TRACE_MIN_STEPS" \
      --target-resolved-rate "$TRACE_TARGET_RESOLVED_RATE" \
      --selector-name "$EXPECTED_SELECTOR_NAME" \
      --selection-policy "$EXPECTED_SELECTION_POLICY" \
      --weight-file "$WEIGHT_FILE" \
      --stop-threshold "$TRACE_STOP_THRESHOLD" \
      --source-selector-name "$SOURCE_SELECTOR_NAME" \
      "${extra_args[@]}"
  done
}

check_reward_manifest() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '[mrec-reward] DRY_RUN skips reward manifest audit: %s\n' "$TRACE_ROOT"
    return 0
  fi
  "$PYTHON_BIN" - "$TRACE_ROOT" "$WEIGHT_FILE" "$EXPECTED_SELECTOR_NAME" "$EXPECTED_ADAPTIVE_POLICY" "$EXPECTED_SELECTION_POLICY" "$TRACE_MIN_STEPS" "$TRACE_MAX_STEPS" "$TRACE_STOP_THRESHOLD" <<'PY'
import json
import sys
from pathlib import Path

trace_root = Path(sys.argv[1])
weight_file = Path(sys.argv[2])
expected_selector = sys.argv[3]
expected_adaptive = sys.argv[4]
expected_selection = sys.argv[5]
expected_min_steps = int(sys.argv[6])
expected_max_steps = int(sys.argv[7])
expected_stop_threshold = float(sys.argv[8])

for split in ("train", "val", "test"):
    manifest_path = trace_root / f"manifest_{split}.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selector_name") != expected_selector:
        raise SystemExit(f"{manifest_path}: selector_name={manifest.get('selector_name')!r}")
    if manifest.get("adaptive_policy") != expected_adaptive:
        raise SystemExit(f"{manifest_path}: adaptive_policy={manifest.get('adaptive_policy')!r}")
    params = manifest.get("params") or {}
    if params.get("selection_policy") != expected_selection:
        raise SystemExit(f"{manifest_path}: selection_policy={params.get('selection_policy')!r}")
    if int(params.get("min_steps") or 0) != expected_min_steps:
        raise SystemExit(f"{manifest_path}: min_steps={params.get('min_steps')!r}")
    if int(params.get("max_steps") or 0) != expected_max_steps:
        raise SystemExit(f"{manifest_path}: max_steps={params.get('max_steps')!r}")
    if abs(float(params.get("stop_threshold") or 0.0) - expected_stop_threshold) > 1e-9:
        raise SystemExit(f"{manifest_path}: stop_threshold={params.get('stop_threshold')!r}")
    recorded_weight = Path(str(params.get("weight_file") or ""))
    if recorded_weight != weight_file and recorded_weight.resolve() != weight_file.resolve():
        raise SystemExit(f"{manifest_path}: weight_file={params.get('weight_file')!r}")
    if not manifest.get("weight_fingerprint"):
        raise SystemExit(f"{manifest_path}: empty weight_fingerprint")
print(f"v0.2 learned marginal reward manifest audit ok: {trace_root}")
PY
}

export ATOM_ANCHOR_ROOT
export TRACE_ROOT
export WEIGHT_FILE
export QUALITY_AUDIT="${QUALITY_AUDIT:-${ATOM_ANCHOR_ROOT}/quality_audit_after_fix.json}"
export CASE_SUFFIX="${CASE_SUFFIX:-__atom_anchor_v0_2_learned_marginal_reward_top10}"
export TRACE_TOP_K="${TRACE_TOP_K:-10}"
export TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-mrec_min}"
export EVIDENCE_TEXT_MODE="${EVIDENCE_TEXT_MODE:-full}"
export EXPECTED_SELECTOR_NAME
export EXPECTED_CHUNK_MMR_FINGERPRINT="${EXPECTED_CHUNK_MMR_FINGERPRINT:-}"
export LORA_SUFFIX="${LORA_SUFFIX:-_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw}"
export EVAL_SPLITS="${EVAL_SPLITS:-val,test}"
export RUN_TAU_EVAL="${RUN_TAU_EVAL:-auto}"
export RUN_LABEL="${RUN_LABEL:-atom-anchor-v0.2-learned-reward-top10}"
export RUN_HEADER_LABEL="${RUN_HEADER_LABEL:-atom-anchor-v0.2-learned-reward-top10-full}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-atom-anchor-v0-2}"

printf '[atom-anchor-v0.2-learned-reward-top10] MODE=%s TRACE_ROOT=%s REWARD_CACHE_DIR=%s WEIGHT_FILE=%s PRIOR_WEIGHT_FILE=%s TEACHER_RUN_DIR=%s SCORING_BACKEND=%s VLLM_TP=%s VLLM_ENFORCE_EAGER=%s NCCL_CUMEM_HOST_ENABLE=%s VLLM_USE_DEEP_GEMM=%s EXPECTED_SELECTOR_NAME=%s TRACE_TOP_K=%s EVAL_SPLITS=%s\n' \
  "$MODE" "$TRACE_ROOT" "$REWARD_CACHE_DIR" "$WEIGHT_FILE" "$PRIOR_WEIGHT_FILE" "$TEACHER_RUN_DIR" "$SCORING_BACKEND" "$VLLM_TENSOR_PARALLEL_SIZE" "$VLLM_ENFORCE_EAGER" "${NCCL_CUMEM_HOST_ENABLE:-}" "${VLLM_USE_DEEP_GEMM:-}" "$EXPECTED_SELECTOR_NAME" "$TRACE_TOP_K" "$EVAL_SPLITS"

case "$MODE" in
  cache)
    build_reward_cache
    ;;
  weights)
    train_reward_weights
    ;;
  traces)
    build_reward_traces
    check_reward_manifest
    ;;
  check)
    check_reward_manifest
    MODE=check bash "${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_1_mrec_min_lora_ebs16_lr2e5_ep12_eval100.sh"
    ;;
  build)
    build_reward_traces
    check_reward_manifest
    MODE=build bash "${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_1_mrec_min_lora_ebs16_lr2e5_ep12_eval100.sh"
    ;;
  train)
    require_path "$WEIGHT_FILE" "reward weight file"
    check_reward_manifest
    MODE=train bash "${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_1_mrec_min_lora_ebs16_lr2e5_ep12_eval100.sh"
    ;;
  eval)
    require_path "$WEIGHT_FILE" "reward weight file"
    check_reward_manifest
    MODE=eval bash "${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_1_mrec_min_lora_ebs16_lr2e5_ep12_eval100.sh"
    ;;
  full)
    build_reward_cache
    train_reward_weights
    build_reward_traces
    check_reward_manifest
    MODE=full bash "${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_1_mrec_min_lora_ebs16_lr2e5_ep12_eval100.sh"
    ;;
  *)
    printf 'Unsupported MODE=%s. Use check, cache, weights, traces, build, train, eval, or full.\n' "$MODE" >&2
    exit 2
    ;;
esac
