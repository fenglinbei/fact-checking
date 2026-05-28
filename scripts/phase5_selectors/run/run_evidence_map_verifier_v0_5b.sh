#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SPLIT="${SPLIT:-val}"
V05A_DIR="${V05A_DIR:-outputs/selectors/evidence_map_selector/v0_5a_${SPLIT}}"
SELECTION_TRACE="${SELECTION_TRACE:-${V05A_DIR}/selection_trace_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/evidence_map_selector/v0_5b_${SPLIT}_map_verifier}"
CONFIG="${CONFIG:-configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml}"

DIRECT_VERIFIER_RUN_DIR="${DIRECT_VERIFIER_RUN_DIR:-outputs/oracle_direct_verifier/stage2_sentence/train/b3_oracle_sentence_direct_verifier_1024_20260519-200709}"
CHECKPOINTS="${CHECKPOINTS:-best,checkpoint-600,checkpoint-550,checkpoint-500,checkpoint-450}"
SELECTORS="${SELECTORS:-v0_5a_evidence_map_top5,v0_5a_base_only_top5,fusion_refit_all_features_plus_direct_ce_top5}"

BUILD_DATA="${BUILD_DATA:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
RUN_SUMMARY="${RUN_SUMMARY:-true}"
DRY_RUN="${DRY_RUN:-false}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"

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
PRIMARY_METRIC="${PRIMARY_METRIC:-macro_f1}"

if [[ "${SPLIT}" == "train" ]]; then
  RAW_PATH="${RAW_PATH:-data/raw/LIAR-RAW/train.json}"
elif [[ "${SPLIT}" == "test" ]]; then
  RAW_PATH="${RAW_PATH:-data/raw/LIAR-RAW/test.json}"
else
  RAW_PATH="${RAW_PATH:-data/raw/LIAR-RAW/val.json}"
fi

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

slugify() {
  local raw="$1"
  raw="${raw//\//_}"
  raw="${raw// /_}"
  raw="${raw//,/}"
  raw="${raw//:/_}"
  raw="${raw//./_}"
  printf "%s" "${raw}"
}

bool_flag() {
  local name="$1"
  local value="$2"
  if [[ "${value}" == "true" || "${value}" == "1" ]]; then
    printf -- "--%s" "${name}"
  else
    printf -- "--no-%s" "${name}"
  fi
}

require_path() {
  local path="$1"
  local what="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[v0.5b-map-verifier] missing ${what}: ${path}" >&2
    exit 1
  fi
}

run_cmd() {
  echo "[v0.5b-map-verifier] $*"
  if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

build_selector_data() {
  local selector="$1"
  local selector_slug="$2"
  local data_dir="${OUTPUT_DIR}/verifier_data/${selector_slug}"
  local cmd=(
    python scripts/phase5_selectors/build/build_evidence_map_verifier_data.py
    --selection-trace "${SELECTION_TRACE}"
    --output-dir "${data_dir}"
    --split "${SPLIT}"
    --raw-path "${RAW_PATH}"
    --expected-selector-name "${selector}"
    --config "${CONFIG}"
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
  local checkpoint_slug=""
  checkpoint_slug="$(slugify "${checkpoint}")"
  local data_dir="${OUTPUT_DIR}/verifier_data/${selector_slug}"
  local eval_dir="${OUTPUT_DIR}/eval/${selector_slug}/${checkpoint_slug}"
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
    "$(bool_flag strict-label-token-meta "${STRICT_LABEL_TOKEN_META}")"
    "$(bool_flag dedupe-sample-idx "${DEDUPE_SAMPLE_IDX}")"
  )
  if [[ "${MERGE_LORA_FOR_FORWARD}" == "true" || "${MERGE_LORA_FOR_FORWARD}" == "1" ]]; then
    cmd+=(--merge-lora-for-forward)
  fi
  if [[ -n "${BASE_MODEL_NAME_OR_PATH}" ]]; then
    cmd+=(--base-model-name-or-path "${BASE_MODEL_NAME_OR_PATH}")
  fi
  run_cmd "${cmd[@]}"
}

main() {
  local selectors=()
  local checkpoints=()
  local selector=""
  local selector_slug=""
  local checkpoint=""

  split_csv "${SELECTORS}" selectors
  split_csv "${CHECKPOINTS}" checkpoints
  if [[ "${#selectors[@]}" -eq 0 ]]; then
    echo "[v0.5b-map-verifier] SELECTORS is empty" >&2
    exit 1
  fi
  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "[v0.5b-map-verifier] CHECKPOINTS is empty" >&2
    exit 1
  fi

  require_path "${SELECTION_TRACE}" "v0.5a selection trace"
  require_path "${RAW_PATH}" "raw split"
  require_path "${CONFIG}" "experiment config"
  require_path "${DIRECT_VERIFIER_RUN_DIR}" "oracle-direct verifier run dir"
  if [[ "${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "1" ]]; then
    for checkpoint in "${checkpoints[@]}"; do
      require_path "${DIRECT_VERIFIER_RUN_DIR}/${checkpoint}" "checkpoint ${checkpoint}"
    done
  fi

  mkdir -p "${OUTPUT_DIR}"
  cat > "${OUTPUT_DIR}/run_manifest.json" <<EOF
{
  "split": "${SPLIT}",
  "v05a_dir": "${V05A_DIR}",
  "selection_trace": "${SELECTION_TRACE}",
  "output_dir": "${OUTPUT_DIR}",
  "config": "${CONFIG}",
  "direct_verifier_run_dir": "${DIRECT_VERIFIER_RUN_DIR}",
  "selectors": "${SELECTORS}",
  "checkpoints": "${CHECKPOINTS}",
  "primary_metric": "${PRIMARY_METRIC}",
  "sample_limit": "${SAMPLE_LIMIT}",
  "dry_run": "${DRY_RUN}"
}
EOF

  echo "[v0.5b-map-verifier] split=${SPLIT}"
  echo "[v0.5b-map-verifier] output=${OUTPUT_DIR}"
  echo "[v0.5b-map-verifier] selectors=${selectors[*]}"
  echo "[v0.5b-map-verifier] checkpoints=${checkpoints[*]}"

  for selector in "${selectors[@]}"; do
    selector_slug="$(slugify "${selector}")"
    if [[ "${BUILD_DATA}" == "true" || "${BUILD_DATA}" == "1" ]]; then
      build_selector_data "${selector}" "${selector_slug}"
    fi
    if [[ "${RUN_EVAL}" == "true" || "${RUN_EVAL}" == "1" ]]; then
      if [[ "${DRY_RUN}" != "true" && "${DRY_RUN}" != "1" ]]; then
        require_path "${OUTPUT_DIR}/verifier_data/${selector_slug}/train.resolved.yaml" "verifier data config for ${selector}"
      fi
      for checkpoint in "${checkpoints[@]}"; do
        eval_checkpoint "${selector_slug}" "${checkpoint}"
      done
    fi
  done

  if [[ "${RUN_SUMMARY}" == "true" || "${RUN_SUMMARY}" == "1" ]]; then
    run_cmd python scripts/phase5_selectors/eval/summarize_map_verifier_v0_5b.py \
      --output-dir "${OUTPUT_DIR}" \
      --split "${SPLIT}" \
      --primary-metric "${PRIMARY_METRIC}" \
      --direct-verifier-run-dir "${DIRECT_VERIFIER_RUN_DIR}"
  fi

  echo "[v0.5b-map-verifier] done: ${OUTPUT_DIR}"
}

main "$@"
