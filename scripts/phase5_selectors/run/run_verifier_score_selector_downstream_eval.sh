#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

SPLIT="${SPLIT:-val}"
BASE_SELECTOR_DIR="${BASE_SELECTOR_DIR:-outputs/selectors/verifier_score_selector/b3_oracle_direct_v0}"
GREEDY_DIR="${GREEDY_DIR:-${BASE_SELECTOR_DIR}/val_greedy_stepwise_top5}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_SELECTOR_DIR}/downstream_verifier_eval}"
CONFIG="${CONFIG:-configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml}"
DIRECT_VERIFIER_RUN_DIR="${DIRECT_VERIFIER_RUN_DIR:-outputs/oracle_direct_verifier/stage2_sentence/train/b3_oracle_sentence_direct_verifier_1024_20260519-200709}"
CHECKPOINTS="${CHECKPOINTS:-best,checkpoint-600}"
PRIMARY_METRIC="${PRIMARY_METRIC:-macro_f1}"

SELECTOR_SPECS="${SELECTOR_SPECS:-greedy_entropy_neg|${GREEDY_DIR}/verifier_score_greedy_stepwise_top5_entropy_neg/selection_trace.jsonl|verifier_score_greedy_stepwise_top5_entropy_neg;greedy_base_pred_margin|${GREEDY_DIR}/verifier_score_greedy_stepwise_top5_base_pred_margin/selection_trace.jsonl|verifier_score_greedy_stepwise_top5_base_pred_margin}"

BUILD_DATA="${BUILD_DATA:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
RUN_SUMMARY="${RUN_SUMMARY:-true}"
DRY_RUN="${DRY_RUN:-false}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"
TOP_K="${TOP_K:-5}"

PROMPT_MODEL_NAME_OR_PATH="${PROMPT_MODEL_NAME_OR_PATH:-}"
TRAIN_MODEL_NAME_OR_PATH="${TRAIN_MODEL_NAME_OR_PATH:-}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-}"
BASE_MODEL_NAME_OR_PATH="${BASE_MODEL_NAME_OR_PATH:-}"

PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"
STRICT_LABEL_TOKEN_META="${STRICT_LABEL_TOKEN_META:-true}"
DEDUPE_SAMPLE_IDX="${DEDUPE_SAMPLE_IDX:-true}"
MERGE_LORA_FOR_FORWARD="${MERGE_LORA_FOR_FORWARD:-false}"
GPU_DEVICES="${GPU_DEVICES:-}"

if [[ "${SPLIT}" == "train" ]]; then
  RAW_PATH="${RAW_PATH:-data/raw/LIAR-RAW/train.json}"
elif [[ "${SPLIT}" == "test" ]]; then
  RAW_PATH="${RAW_PATH:-data/raw/LIAR-RAW/test.json}"
else
  RAW_PATH="${RAW_PATH:-data/raw/LIAR-RAW/val.json}"
fi

truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "True" ]]
}

cuda_visible() {
  python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PY
}

slugify() {
  local raw="$1"
  raw="${raw//\//_}"
  raw="${raw// /_}"
  raw="${raw//,/}"
  raw="${raw//:/_}"
  raw="${raw//./_}"
  printf "%s" "${raw}"
}

split_csv() {
  local raw="$1"
  local -n out_array="$2"
  local items=()
  local item=""
  IFS=',' read -r -a items <<< "${raw}"
  out_array=()
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      out_array+=("${item}")
    fi
  done
}

bool_arg() {
  local positive="$1"
  local negative="$2"
  local value="$3"
  if truthy "${value}"; then
    printf "%s" "${positive}"
  else
    printf "%s" "${negative}"
  fi
}

require_path() {
  local path="$1"
  local what="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[verifier-score-downstream] missing ${what}: ${path}" >&2
    exit 1
  fi
}

run_cmd() {
  echo "[verifier-score-downstream] $*"
  if truthy "${DRY_RUN}"; then
    return 0
  fi
  "$@"
}

ensure_gpu_for_eval() {
  if ! truthy "${RUN_EVAL}" || truthy "${DRY_RUN}"; then
    return 0
  fi
  if [[ -n "${GPU_DEVICES}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
  fi
  if ! cuda_visible; then
    if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
      # shellcheck source=/dev/null
      source "${HOME}/miniconda3/etc/profile.d/conda.sh"
      conda activate cppo || true
    fi
  fi
  if ! cuda_visible; then
    echo "[verifier-score-downstream] CUDA is not visible, even after trying conda activate cppo; refusing to run huge verifier eval on CPU." >&2
    exit 1
  fi
}

build_selector_data() {
  local selector_slug="$1"
  local trace_path="$2"
  local expected_selector="$3"
  local data_dir="${OUTPUT_DIR}/verifier_data/${selector_slug}"
  local cmd=(
    python scripts/phase5_selectors/build/build_trace_verifier_data.py
    --config "${CONFIG}"
    --val-trace "${trace_path}"
    --val-raw "${RAW_PATH}"
    --output-dir "${data_dir}"
    --selection-mode trace
    --expected-selector-name "${expected_selector}"
    --top-k "${TOP_K}"
    --val-only
  )
  if [[ -n "${PROMPT_MODEL_NAME_OR_PATH}" ]]; then
    cmd+=(--prompt-model-name-or-path "${PROMPT_MODEL_NAME_OR_PATH}")
  fi
  if [[ -n "${TRAIN_MODEL_NAME_OR_PATH}" ]]; then
    cmd+=(--train-model-name-or-path "${TRAIN_MODEL_NAME_OR_PATH}")
  fi
  if [[ -n "${MODEL_BASE_PATH}" ]]; then
    cmd+=(--model-base-path "${MODEL_BASE_PATH}")
  fi
  if [[ -n "${SAMPLE_LIMIT}" ]]; then
    cmd+=(--sample-limit "${SAMPLE_LIMIT}")
  fi
  run_cmd "${cmd[@]}"
}

eval_checkpoint() {
  local selector_slug="$1"
  local checkpoint="$2"
  local checkpoint_slug
  checkpoint_slug="$(slugify "${checkpoint}")"
  local data_dir="${OUTPUT_DIR}/verifier_data/${selector_slug}"
  local eval_dir="${OUTPUT_DIR}/eval/${selector_slug}/${checkpoint_slug}"
  local strict_arg
  local dedupe_arg
  strict_arg="$(bool_arg --strict-label-token-meta --no-strict-label-token-meta "${STRICT_LABEL_TOKEN_META}")"
  dedupe_arg="$(bool_arg --dedupe-sample-idx --no-dedupe-sample-idx "${DEDUPE_SAMPLE_IDX}")"
  local cmd=(
    python scripts/phase4_verifier/eval_label_token_torch_forward.py
    --run-dir "${DIRECT_VERIFIER_RUN_DIR}"
    --checkpoint "${checkpoint}"
    --split "${SPLIT}"
    --config "${data_dir}/train.resolved.yaml"
    --output-dir "${eval_dir}"
    --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
    --torch-dtype "${TORCH_DTYPE}"
    --attn-implementation "${ATTN_IMPLEMENTATION}"
    "${strict_arg}"
    "${dedupe_arg}"
  )
  if truthy "${MERGE_LORA_FOR_FORWARD}"; then
    cmd+=(--merge-lora-for-forward)
  fi
  if [[ -n "${BASE_MODEL_NAME_OR_PATH}" ]]; then
    cmd+=(--base-model-name-or-path "${BASE_MODEL_NAME_OR_PATH}")
  fi
  run_cmd "${cmd[@]}"
}

main() {
  local checkpoints=()
  local specs=()
  local spec=""
  local checkpoint=""
  split_csv "${CHECKPOINTS}" checkpoints
  IFS=';' read -r -a specs <<< "${SELECTOR_SPECS}"

  require_path "${CONFIG}" "experiment config"
  require_path "${RAW_PATH}" "raw split"
  require_path "${DIRECT_VERIFIER_RUN_DIR}" "direct verifier run dir"
  for checkpoint in "${checkpoints[@]}"; do
    require_path "${DIRECT_VERIFIER_RUN_DIR}/${checkpoint}" "checkpoint ${checkpoint}"
  done
  ensure_gpu_for_eval

  mkdir -p "${OUTPUT_DIR}"
  cat > "${OUTPUT_DIR}/run_manifest.json" <<EOF
{
  "split": "${SPLIT}",
  "base_selector_dir": "${BASE_SELECTOR_DIR}",
  "greedy_dir": "${GREEDY_DIR}",
  "output_dir": "${OUTPUT_DIR}",
  "config": "${CONFIG}",
  "raw_path": "${RAW_PATH}",
  "direct_verifier_run_dir": "${DIRECT_VERIFIER_RUN_DIR}",
  "checkpoints": "${CHECKPOINTS}",
  "selector_specs": "${SELECTOR_SPECS}",
  "primary_metric": "${PRIMARY_METRIC}",
  "sample_limit": "${SAMPLE_LIMIT}",
  "dry_run": "${DRY_RUN}"
}
EOF

  echo "[verifier-score-downstream] output=${OUTPUT_DIR}"
  echo "[verifier-score-downstream] checkpoints=${checkpoints[*]}"
  echo "[verifier-score-downstream] selector_specs=${SELECTOR_SPECS}"

  for spec in "${specs[@]}"; do
    [[ -z "${spec}" ]] && continue
    local fields=()
    IFS='|' read -r -a fields <<< "${spec}"
    if [[ "${#fields[@]}" -lt 3 ]]; then
      echo "[verifier-score-downstream] bad SELECTOR_SPECS item: ${spec}" >&2
      exit 1
    fi
    local selector_slug
    local trace_path
    local expected_selector
    selector_slug="$(slugify "${fields[0]}")"
    trace_path="${fields[1]}"
    expected_selector="${fields[2]}"
    require_path "${trace_path}" "selection trace ${selector_slug}"

    if truthy "${BUILD_DATA}"; then
      build_selector_data "${selector_slug}" "${trace_path}" "${expected_selector}"
    fi
    if truthy "${RUN_EVAL}"; then
      if ! truthy "${DRY_RUN}"; then
        require_path "${OUTPUT_DIR}/verifier_data/${selector_slug}/train.resolved.yaml" "verifier data config for ${selector_slug}"
      fi
      for checkpoint in "${checkpoints[@]}"; do
        eval_checkpoint "${selector_slug}" "${checkpoint}"
      done
    fi
  done

  if truthy "${RUN_SUMMARY}"; then
    run_cmd python scripts/phase5_selectors/eval/summarize_verifier_score_selector_downstream_eval.py \
      --output-dir "${OUTPUT_DIR}" \
      --split "${SPLIT}" \
      --primary-metric "${PRIMARY_METRIC}" \
      --direct-verifier-run-dir "${DIRECT_VERIFIER_RUN_DIR}"
  fi

  echo "[verifier-score-downstream] done: ${OUTPUT_DIR}"
}

main "$@"
