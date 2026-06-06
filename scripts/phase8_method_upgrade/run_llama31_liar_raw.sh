#!/usr/bin/env bash
set -euo pipefail

# Llama-3.1 FullFT method-upgrade run on LIAR-RAW v0.6c selector traces.
#
# Defaults run an ordinal-aware label-token loss with alpha=0.2, using the
# existing train/val LIAR-RAW v0.6c evidence-chain traces. The script does not
# build a missing LIAR-RAW test trace unless RUN_TEST_GRAPH_BUILD=true.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

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
print(f"a{compact}")
PY
}

MODEL_PATH="${MODEL_PATH:-/data/models/Meta-Llama-3.1-8B-Instruct}"
MODEL_SIZE_B="${MODEL_SIZE_B:-8}"
BASE_EXPERIMENT="${BASE_EXPERIMENT:-b3_oracle_sentence_direct_verifier_1024_fullft}"

ENABLE_ORDINAL_LOSS="${ENABLE_ORDINAL_LOSS:-true}"
ORDINAL_LOSS_ALPHA="${ORDINAL_LOSS_ALPHA:-0.2}"
METHOD_SUFFIX="${METHOD_SUFFIX:-}"
if [[ -z "${METHOD_SUFFIX}" ]]; then
  if is_true "${ENABLE_ORDINAL_LOSS}"; then
    METHOD_SUFFIX="_ord_abs_$(ordinal_alpha_suffix "${ORDINAL_LOSS_ALPHA}")"
  else
    METHOD_SUFFIX="_noord_ctrl"
  fi
fi

MIN_TOP_K="${MIN_TOP_K:-5}"
MAX_TOP_K="${MAX_TOP_K:-10}"
GRAPH_BUDGET_SLUG="adaptive${MIN_TOP_K}_${MAX_TOP_K}"
TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-plain}"

CASE_NAME="${CASE_NAME:-v0_6c_liar6_rule_step_${GRAPH_BUDGET_SLUG}_llama31_8b_fullft${METHOD_SUFFIX}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selector_trace_verifier/liar_raw_v0_6c_method_upgrade}"
RUN_ROOT="${RUN_ROOT:-outputs/runs/liar_raw_v0_6c_method_upgrade_selector_trace_full_pipeline}"
CONFIG_CACHE_ROOT="${CONFIG_CACHE_ROOT:-outputs/cache/method_upgrade/liar_raw_llama31/configs}"

TRAIN_RAW="${TRAIN_RAW:-data/raw/LIAR-RAW/train.json}"
VAL_RAW="${VAL_RAW:-data/raw/LIAR-RAW/val.json}"
TEST_RAW="${TEST_RAW:-data/raw/LIAR-RAW/test.json}"
RAW_DATASET="${RAW_DATASET:-liar_raw}"
LABEL_SCHEMA="${LABEL_SCHEMA:-liar6}"

CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT:-432dfc970e75}"
EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME:-v0_6c_rule_step_${GRAPH_BUDGET_SLUG}}"
TRAIN_TRACE="${TRAIN_TRACE:-outputs/selectors/evidence_chain_graph/v0_6c_${GRAPH_BUDGET_SLUG}_train/selection_trace_train.jsonl}"
VAL_TRACE="${VAL_TRACE:-outputs/selectors/evidence_chain_graph/v0_6c_${GRAPH_BUDGET_SLUG}_val/selection_trace_val.jsonl}"
TEST_GRAPH_DIR="${TEST_GRAPH_DIR:-outputs/selectors/evidence_chain_graph/v0_6c_${GRAPH_BUDGET_SLUG}_test}"
TEST_TRACE="${TEST_TRACE:-${TEST_GRAPH_DIR}/selection_trace_test.jsonl}"
TEST_EVIDENCE_MAP_FEATURES="${TEST_EVIDENCE_MAP_FEATURES:-outputs/selectors/evidence_map_selector/v0_6b_test/candidate_evidence_map_features_test.jsonl}"
RUN_TEST_GRAPH_BUILD="${RUN_TEST_GRAPH_BUILD:-false}"

FULLFT_DEEPSPEED_CONFIG="${FULLFT_DEEPSPEED_CONFIG:-configs/deepspeed_zero3_bsz1_ga8_lowpeak.json}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_INFER="${RUN_INFER:-true}"
RUN_API_INFER="${RUN_API_INFER:-false}"
RUN_LABEL_TOKEN_INFER="${RUN_LABEL_TOKEN_INFER:-true}"
INFER_SPLIT="${INFER_SPLIT:-test}"
CHECKPOINTS="${CHECKPOINTS:-best}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_BUILD="${FORCE_BUILD:-false}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"
FORCE_LABEL_TOKEN_INFER="${FORCE_LABEL_TOKEN_INFER:-false}"
FORCE_INFER="${FORCE_INFER:-true}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-0}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[llama31-liar-raw] missing model path: ${MODEL_PATH}" >&2
  exit 1
fi

if is_true "${RUN_TEST_GRAPH_BUILD}"; then
  SPLIT=test \
  INPUT="${TEST_EVIDENCE_MAP_FEATURES}" \
  OUTPUT_DIR="${TEST_GRAPH_DIR}" \
  CANDIDATE_TOP_N="${CANDIDATE_TOP_N:-20}" \
  MIN_TOP_K="${MIN_TOP_K}" \
  MAX_TOP_K="${MAX_TOP_K}" \
  CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}" \
  SAMPLE_LIMIT="${SAMPLE_LIMIT}" \
  bash scripts/phase5_selectors/run/run_evidence_chain_graph_v0_6c.sh
fi

if [[ "${INFER_SPLIT}" == "test" ]] && is_true "${RUN_LABEL_TOKEN_INFER}" && [[ ! -s "${TEST_TRACE}" ]]; then
  if is_true "${DRY_RUN}"; then
    echo "[llama31-liar-raw] dry-run missing test trace: ${TEST_TRACE}" >&2
  else
    echo "[llama31-liar-raw] missing test trace for INFER_SPLIT=test: ${TEST_TRACE}" >&2
    echo "[llama31-liar-raw] provide TEST_TRACE=... or build it first; RUN_TEST_GRAPH_BUILD=true only works after test evidence-map features exist." >&2
    exit 1
  fi
fi

prepare_args=(
  --backbone llama31_8b
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

echo "[llama31-liar-raw] case              : ${CASE_NAME}"
echo "[llama31-liar-raw] config            : ${CONFIG_PATH}"
echo "[llama31-liar-raw] output_root       : ${OUTPUT_ROOT}"
echo "[llama31-liar-raw] ordinal/alpha     : ${ENABLE_ORDINAL_LOSS}/${ORDINAL_LOSS_ALPHA}"
echo "[llama31-liar-raw] deepspeed         : ${FULLFT_DEEPSPEED_CONFIG}"
echo "[llama31-liar-raw] infer split       : ${INFER_SPLIT}"
echo "[llama31-liar-raw] train/val trace   : ${TRAIN_TRACE} / ${VAL_TRACE}"
echo "[llama31-liar-raw] test trace        : ${TEST_TRACE}"
echo "[llama31-liar-raw] dry_run           : ${DRY_RUN}"

trace_env=(
  CONFIG="${CONFIG_PATH}"
  INFER_EXPERIMENT="${BASE_EXPERIMENT}"
  CASE_NAME="${CASE_NAME}"
  SOURCE_TYPE=trace
  TRAIN_SOURCE="${TRAIN_TRACE}"
  VAL_SOURCE="${VAL_TRACE}"
  TRAIN_RAW="${TRAIN_RAW}"
  VAL_RAW="${VAL_RAW}"
  TEST_RAW="${TEST_RAW}"
  RAW_DATASET="${RAW_DATASET}"
  LABEL_SCHEMA="${LABEL_SCHEMA}"
  TRACE_SELECTION_MODE=trace
  TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE}"
  TOP_K="${MAX_TOP_K}"
  EXPECTED_SELECTOR_NAME="${EXPECTED_SELECTOR_NAME}"
  EXPECTED_CHUNK_MMR_FINGERPRINT="${CHUNK_MMR_FINGERPRINT}"
  SAMPLE_LIMIT="${SAMPLE_LIMIT}"
  SPLIT="${INFER_SPLIT}"
  CHECKPOINTS="${CHECKPOINTS}"
  RUN_TRAIN="${RUN_TRAIN}"
  RUN_INFER="${RUN_INFER}"
  RUN_API_INFER="${RUN_API_INFER}"
  RUN_LABEL_TOKEN_INFER="${RUN_LABEL_TOKEN_INFER}"
  FORCE_BUILD="${FORCE_BUILD}"
  FORCE_TRAIN="${FORCE_TRAIN}"
  FORCE_LABEL_TOKEN_INFER="${FORCE_LABEL_TOKEN_INFER}"
  FORCE_INFER="${FORCE_INFER}"
  OUTPUT_ROOT="${OUTPUT_ROOT}"
  RUN_ROOT="${RUN_ROOT}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${NPROC_PER_NODE}"
  NUM_MACHINES="${NUM_MACHINES}"
  MIXED_PRECISION="${MIXED_PRECISION}"
  DEEPSPEED_CONFIG="${FULLFT_DEEPSPEED_CONFIG}"
  MERGE_LORA_CACHE=false
  FINETUNE_MODE=full-parameter
  PROMPT_MODEL_NAME_OR_PATH="${MODEL_PATH}"
  TRAIN_MODEL_NAME_OR_PATH="${MODEL_PATH}"
  DRY_RUN="${DRY_RUN}"
)

if [[ -s "${TEST_TRACE}" || "${INFER_SPLIT}" == "test" ]]; then
  trace_env+=(TEST_SOURCE="${TEST_TRACE}")
fi

env "${trace_env[@]}" bash scripts/phase5_selectors/run/run_selector_trace_full_pipeline.sh
