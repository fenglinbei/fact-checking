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

PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"

ATOM_ANCHOR_ROOT="${ATOM_ANCHOR_ROOT:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1}"
SOURCE_FEATURE_ROOT="${SOURCE_FEATURE_ROOT:-${ATOM_ANCHOR_ROOT}/04_evidence_map}"
PROBE_ROOT="${PROBE_ROOT:-${ATOM_ANCHOR_ROOT}/05_mrec_v0_2_reward_probe_n20_l10_t256_v64}"
REWARD_CACHE_DIR="${REWARD_CACHE_DIR:-${PROBE_ROOT}/reward_cache}"
WEIGHT_DIR="${WEIGHT_DIR:-${PROBE_ROOT}/weights}"
WEIGHT_FILE="${WEIGHT_FILE:-${WEIGHT_DIR}/weights.json}"
PRIOR_WEIGHT_FILE="${PRIOR_WEIGHT_FILE:-${ATOM_ANCHOR_ROOT}/05_mrec_v0_2_learned_marginal_proxy/weights/weights.json}"
TEACHER_RUN_DIR="${TEACHER_RUN_DIR:-outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_top5_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/train}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-best}"
BASE_MODEL="${BASE_MODEL:-/data/models/Ministral-3-8B-Instruct-2512}"

MODE="${MODE:-full}" # cache_train|cache_val|cache|weights|audit|full
DRY_RUN="${DRY_RUN:-false}"
FORCE_REWARD_CACHE="${FORCE_REWARD_CACHE:-false}"
FORCE_REWARD_WEIGHTS="${FORCE_REWARD_WEIGHTS:-false}"

TRAIN_SAMPLE_LIMIT="${TRAIN_SAMPLE_LIMIT:-256}"
VAL_SAMPLE_LIMIT="${VAL_SAMPLE_LIMIT:-64}"
REWARD_CANDIDATE_TOP_N="${REWARD_CANDIDATE_TOP_N:-20}"
REWARD_ROLLOUT_STEPS="${REWARD_ROLLOUT_STEPS:-10}"
REWARD_ROLLIN_POLICY="${REWARD_ROLLIN_POLICY:-learned_marginal_proxy}"
REWARD_ROLLIN_STOP_THRESHOLD="${REWARD_ROLLIN_STOP_THRESHOLD:-0.0}"
REWARD_TARGET_RESOLVED_RATE="${REWARD_TARGET_RESOLVED_RATE:-1.0}"
REWARD_EPOCHS="${REWARD_EPOCHS:-30}"
REWARD_LEARNING_RATE="${REWARD_LEARNING_RATE:-0.03}"
REWARD_PRIOR_WEIGHT="${REWARD_PRIOR_WEIGHT:-0.02}"
REWARD_MAX_PAIRS_PER_GROUP="${REWARD_MAX_PAIRS_PER_GROUP:-64}"

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

split_sample_limit() {
  case "$1" in
    train) printf '%s\n' "$TRAIN_SAMPLE_LIMIT" ;;
    val) printf '%s\n' "$VAL_SAMPLE_LIMIT" ;;
    *) printf 'Unsupported split=%s\n' "$1" >&2; exit 2 ;;
  esac
}

cache_complete() {
  local split="$1"
  local expected_events="$2"
  if [[ "$DRY_RUN" == "true" ]]; then
    return 1
  fi
  "$PYTHON_BIN" - "$REWARD_CACHE_DIR" "$split" "$expected_events" "$REWARD_CANDIDATE_TOP_N" "$REWARD_ROLLOUT_STEPS" "$SCORING_BACKEND" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
split = sys.argv[2]
expected = int(sys.argv[3])
expected_top_n = int(sys.argv[4])
expected_rollout_steps = int(sys.argv[5])
expected_backend = sys.argv[6]
manifest_path = root / f"manifest_{split}.json"
events_path = root / f"reward_event_summaries_{split}.jsonl"
records_path = root / f"reward_records_{split}.jsonl"
scores_path = root / f"raw_teacher_scores_{split}.jsonl"

if not manifest_path.exists() or not events_path.exists() or not records_path.exists() or not scores_path.exists():
    raise SystemExit(1)
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if manifest.get("status") != "completed":
    raise SystemExit(1)
if int((manifest.get("params") or {}).get("candidate_top_n") or 0) != expected_top_n:
    raise SystemExit(1)
if int((manifest.get("params") or {}).get("rollout_steps") or 0) != expected_rollout_steps:
    raise SystemExit(1)
if str((manifest.get("teacher") or {}).get("scoring_backend_requested") or "") != expected_backend:
    raise SystemExit(1)

def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())

event_count = count_lines(events_path)
record_count = count_lines(records_path)
score_count = count_lines(scores_path)
if event_count < expected or record_count <= 0 or score_count <= 0:
    raise SystemExit(1)
print(f"{split} reward cache complete: events={event_count} records={record_count} scores={score_count}")
PY
}

build_reward_cache_split() {
  local split="$1"
  local sample_limit
  sample_limit="$(split_sample_limit "$split")"
  local input_path="${SOURCE_FEATURE_ROOT}/candidate_evidence_map_features_${split}.jsonl"
  local raw_path="data/raw/LIAR-RAW/${split}.json"
  local resume_arg="--resume"
  if [[ "$FORCE_REWARD_CACHE" == "true" ]]; then
    resume_arg="--no-resume"
  elif cache_complete "$split" "$sample_limit"; then
    printf '[mrec-reward-probe] reuse complete %s reward cache: %s\n' "$split" "$REWARD_CACHE_DIR"
    return 0
  fi

  require_path "$input_path" "${split} atom-anchor evidence-map features"
  require_path "$raw_path" "${split} raw data"
  require_path "$PRIOR_WEIGHT_FILE" "prior learned-marginal proxy weights"
  require_path "${TEACHER_RUN_DIR}/${TEACHER_CHECKPOINT}/adapter_model.safetensors" "teacher LoRA adapter"
  require_path "${TEACHER_RUN_DIR}/label_token_ce_meta.json" "teacher label token metadata"

  local vllm_extra_args=()
  if [[ "$VLLM_ENFORCE_EAGER" == "true" ]]; then
    vllm_extra_args=(--vllm-enforce-eager)
  fi

  run_cmd "$PYTHON_BIN" scripts/phase5_selectors/build/build_mrec_reward_cache.py \
    --input "$input_path" \
    --raw "$raw_path" \
    --output-dir "$REWARD_CACHE_DIR" \
    --split "$split" \
    --sample-limit "$sample_limit" \
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
    --fsync-cache \
    "$resume_arg"
}

audit_reward_cache() {
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '[mrec-reward-probe] DRY_RUN skips reward cache audit: %s\n' "$REWARD_CACHE_DIR"
    return 0
  fi
  cache_complete train "$TRAIN_SAMPLE_LIMIT"
  cache_complete val "$VAL_SAMPLE_LIMIT"
}

train_reward_weights() {
  if [[ -f "$WEIGHT_FILE" && "$FORCE_REWARD_WEIGHTS" != "true" ]]; then
    printf '[mrec-reward-probe] reuse reward weights: %s\n' "$WEIGHT_FILE"
    return 0
  fi
  audit_reward_cache
  require_path "${REWARD_CACHE_DIR}/reward_records_train.jsonl" "train reward cache"
  require_path "${REWARD_CACHE_DIR}/reward_records_val.jsonl" "val reward cache"
  require_path "$PRIOR_WEIGHT_FILE" "prior learned-marginal proxy weights"
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

printf '[mrec-reward-probe] MODE=%s PROBE_ROOT=%s REWARD_CACHE_DIR=%s WEIGHT_FILE=%s TRAIN_SAMPLE_LIMIT=%s VAL_SAMPLE_LIMIT=%s TOP_N=%s ROLLOUT_STEPS=%s SCORING_BACKEND=%s VLLM_TP=%s VLLM_ENFORCE_EAGER=%s NCCL_CUMEM_HOST_ENABLE=%s VLLM_USE_DEEP_GEMM=%s OMP_NUM_THREADS=%s\n' \
  "$MODE" "$PROBE_ROOT" "$REWARD_CACHE_DIR" "$WEIGHT_FILE" "$TRAIN_SAMPLE_LIMIT" "$VAL_SAMPLE_LIMIT" \
  "$REWARD_CANDIDATE_TOP_N" "$REWARD_ROLLOUT_STEPS" "$SCORING_BACKEND" "$VLLM_TENSOR_PARALLEL_SIZE" \
  "$VLLM_ENFORCE_EAGER" "${NCCL_CUMEM_HOST_ENABLE:-}" "${VLLM_USE_DEEP_GEMM:-}" "${OMP_NUM_THREADS:-}"

case "$MODE" in
  cache_train)
    build_reward_cache_split train
    ;;
  cache_val)
    build_reward_cache_split val
    ;;
  cache)
    build_reward_cache_split train
    build_reward_cache_split val
    audit_reward_cache
    ;;
  weights)
    train_reward_weights
    ;;
  audit)
    audit_reward_cache
    ;;
  full)
    build_reward_cache_split train
    build_reward_cache_split val
    train_reward_weights
    ;;
  *)
    printf 'Unsupported MODE=%s. Use cache_train, cache_val, cache, weights, audit, or full.\n' "$MODE" >&2
    exit 2
    ;;
esac
