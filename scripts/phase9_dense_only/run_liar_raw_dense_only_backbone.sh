#!/usr/bin/env bash
set -euo pipefail

# Run one LIAR-RAW dense-only trace verifier for a selected backbone.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=scripts/phase7_backbone_migration/backbone_cases.sh
source scripts/phase7_backbone_migration/backbone_cases.sh

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

is_true() {
  case "${1:-}" in
    true|1|True|TRUE|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

ordinal_alpha_suffix() {
  python - "$1" <<'PY'
import sys

alpha = float(sys.argv[1])
if alpha < 0:
    raise SystemExit("ORDINAL_LOSS_ALPHA must be non-negative")
compact = f"{alpha:g}".replace(".", "")
print(f"_ord_abs_a{compact}")
PY
}

read_trace_fingerprint() {
  local path="$1"
  python - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("")
    raise SystemExit(0)
with path.open(encoding="utf-8") as handle:
    line = handle.readline()
row = json.loads(line)
trace = row.get("selection_trace") if isinstance(row, dict) else None
if not isinstance(trace, dict):
    trace = row if isinstance(row, dict) else {}
metadata = trace.get("candidate_pool_metadata")
if not isinstance(metadata, dict):
    metadata = {}
print(trace.get("fingerprint") or metadata.get("chunk_mmr_fingerprint") or "")
PY
}

BACKBONE="${BACKBONE:-qwen3_4b_2507}"
FINETUNE="${FINETUNE:-fullft}"
if [[ "${FINETUNE}" != "fullft" ]]; then
  echo "[liar-dense-backbone] only FINETUNE=fullft is supported in this wrapper." >&2
  exit 2
fi
require_backbone "${BACKBONE}"

MODEL_PATH="${MODEL_PATH:-$(backbone_path "${BACKBONE}")}"
MODEL_SIZE_B="${MODEL_SIZE_B:-$(backbone_size_b "${BACKBONE}")}"
BASE_EXPERIMENT="${BASE_EXPERIMENT:-b3_oracle_sentence_direct_verifier_1024_fullft}"
CONFIG_CACHE_ROOT="${CONFIG_CACHE_ROOT:-outputs/cache/dense_only/liar_raw_backbone_configs}"

ENABLE_ORDINAL_LOSS="${ENABLE_ORDINAL_LOSS:-true}"
ORDINAL_LOSS_ALPHA="${ORDINAL_LOSS_ALPHA:-0.2}"
METHOD_SUFFIX="${METHOD_SUFFIX:-}"
if [[ -z "${METHOD_SUFFIX}" ]] && is_true "${ENABLE_ORDINAL_LOSS}"; then
  METHOD_SUFFIX="$(ordinal_alpha_suffix "${ORDINAL_LOSS_ALPHA}")"
fi

MIN_TOP_K="${MIN_TOP_K:-5}"
MAX_TOP_K="${MAX_TOP_K:-10}"
GRAPH_BUDGET_SLUG="adaptive${MIN_TOP_K}_${MAX_TOP_K}"
TRACE_ROOT="${TRACE_ROOT:-outputs/selectors/evidence_chain_graph/liar_raw_dense_v0_6c_${GRAPH_BUDGET_SLUG}}"
TRAIN_TRACE="${TRAIN_TRACE:-${TRACE_ROOT}_train/selection_trace_train.jsonl}"
VAL_TRACE="${VAL_TRACE:-${TRACE_ROOT}_val/selection_trace_val.jsonl}"
TEST_TRACE="${TEST_TRACE:-${TRACE_ROOT}_test/selection_trace_test.jsonl}"

CASE_NAME="${CASE_NAME:-v0_6c_liar6_dense_rule_step_${GRAPH_BUDGET_SLUG}_${BACKBONE}_fullft${METHOD_SUFFIX}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_trace_verifier/liar_raw_dense_v0_6c_backbone}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/liar_raw_dense_v0_6c_backbone}"

TRAIN_RAW="${TRAIN_RAW:-data/raw/LIAR-RAW/train.json}"
VAL_RAW="${VAL_RAW:-data/raw/LIAR-RAW/val.json}"
TEST_RAW="${TEST_RAW:-data/raw/LIAR-RAW/test.json}"
RAW_DATASET="${RAW_DATASET:-liar_raw}"
LABEL_SCHEMA="${LABEL_SCHEMA:-liar6}"

CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-}"
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-v0_6c_rule_step_${GRAPH_BUDGET_SLUG}}"
TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-plain}"
TOP_K="${TOP_K:-${MAX_TOP_K}}"

FULLFT_DEEPSPEED_CONFIG="${FULLFT_DEEPSPEED_CONFIG:-configs/deepspeed_zero3_bsz1_ga8_lowpeak.json}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_INFER="${RUN_INFER:-true}"
RUN_API_INFER="${RUN_API_INFER:-false}"
RUN_LABEL_TOKEN_INFER="${RUN_LABEL_TOKEN_INFER:-true}"
FORCE_BUILD="${FORCE_BUILD:-false}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"
FORCE_LABEL_TOKEN_INFER="${FORCE_LABEL_TOKEN_INFER:-false}"
FORCE_INFER="${FORCE_INFER:-true}"
SPLIT="${SPLIT:-test}"
CHECKPOINTS="${CHECKPOINTS:-best}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"
DRY_RUN="${DRY_RUN:-false}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[liar-dense-backbone] missing model path: ${MODEL_PATH}" >&2
  exit 1
fi
for trace in "${TRAIN_TRACE}" "${VAL_TRACE}" "${TEST_TRACE}"; do
  if [[ ! -s "${trace}" && "${DRY_RUN}" != "true" ]]; then
    echo "[liar-dense-backbone] missing trace: ${trace}" >&2
    exit 1
  fi
done
if [[ -z "${CHUNK_MMR_FINGERPRINT}" ]]; then
  CHUNK_MMR_FINGERPRINT="$(read_trace_fingerprint "${TRAIN_TRACE}")"
  if [[ -z "${CHUNK_MMR_FINGERPRINT}" ]]; then
    if [[ "${DRY_RUN}" == "true" ]]; then
      CHUNK_MMR_FINGERPRINT="dry_run_missing_trace"
    else
      echo "[liar-dense-backbone] failed to read chunk fingerprint from ${TRAIN_TRACE}" >&2
      exit 1
    fi
  fi
fi

prepare_args=(
  --backbone "${BACKBONE}"
  --model-path "${MODEL_PATH}"
  --size-b "${MODEL_SIZE_B}"
  --finetune fullft
  --base-experiment "${BASE_EXPERIMENT}"
  --case-name "${CASE_NAME}"
  --deepspeed-config "${FULLFT_DEEPSPEED_CONFIG}"
  --output-root "${CONFIG_CACHE_ROOT}"
  --print-path
)
if is_true "${ENABLE_ORDINAL_LOSS}"; then
  prepare_args+=(--ordinal-loss-alpha "${ORDINAL_LOSS_ALPHA}")
fi
CONFIG_PATH="$(python scripts/phase7_backbone_migration/prepare_backbone_config.py "${prepare_args[@]}")"

echo "[liar-dense-backbone] backbone      : ${BACKBONE}"
echo "[liar-dense-backbone] case          : ${CASE_NAME}"
echo "[liar-dense-backbone] model         : ${MODEL_PATH}"
echo "[liar-dense-backbone] config        : ${CONFIG_PATH}"
echo "[liar-dense-backbone] traces        : ${TRAIN_TRACE} / ${VAL_TRACE} / ${TEST_TRACE}"
echo "[liar-dense-backbone] output root   : ${OUTPUT_ROOT}"
echo "[liar-dense-backbone] split         : ${SPLIT}"

CONFIG="${CONFIG_PATH}" \
INFER_EXPERIMENT="${BASE_EXPERIMENT}" \
CASE_NAME="${CASE_NAME}" \
SOURCE_TYPE=trace \
TRAIN_SOURCE="${TRAIN_TRACE}" \
VAL_SOURCE="${VAL_TRACE}" \
TEST_SOURCE="${TEST_TRACE}" \
TRAIN_RAW="${TRAIN_RAW}" \
VAL_RAW="${VAL_RAW}" \
TEST_RAW="${TEST_RAW}" \
RAW_DATASET="${RAW_DATASET}" \
LABEL_SCHEMA="${LABEL_SCHEMA}" \
TRACE_SELECTION_MODE=trace \
TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE}" \
TOP_K="${TOP_K}" \
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME}" \
EXPECTED_CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
SAMPLE_LIMIT="${SAMPLE_LIMIT}" \
SPLIT="${SPLIT}" \
CHECKPOINTS="${CHECKPOINTS}" \
RUN_TRAIN="${RUN_TRAIN}" \
RUN_INFER="${RUN_INFER}" \
RUN_API_INFER="${RUN_API_INFER}" \
RUN_LABEL_TOKEN_INFER="${RUN_LABEL_TOKEN_INFER}" \
FORCE_BUILD="${FORCE_BUILD}" \
FORCE_TRAIN="${FORCE_TRAIN}" \
FORCE_LABEL_TOKEN_INFER="${FORCE_LABEL_TOKEN_INFER}" \
FORCE_INFER="${FORCE_INFER}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
RUN_ROOT="${RUN_ROOT}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
NUM_MACHINES="${NUM_MACHINES}" \
MIXED_PRECISION="${MIXED_PRECISION}" \
DEEPSPEED_CONFIG="${FULLFT_DEEPSPEED_CONFIG}" \
MERGE_LORA_CACHE=false \
FINETUNE_MODE=full-parameter \
PROMPT_MODEL_NAME_OR_PATH="${MODEL_PATH}" \
TRAIN_MODEL_NAME_OR_PATH="${MODEL_PATH}" \
DRY_RUN="${DRY_RUN}" \
bash scripts/phase5_selectors/run/run_selector_trace_full_pipeline.sh
