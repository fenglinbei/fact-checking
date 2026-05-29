#!/usr/bin/env bash
# Decode-time permutation/calibration diagnostics for an existing LLM action selector.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

MODEL_DIR="${MODEL_DIR:-outputs/selectors/llm_action_selector/qwen25_3b_robust_prefix_v1_smoke}"
MODEL_NAME="${MODEL_NAME:-}"
ORACLE_RESULTS="${ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_DIR}/evals/decode_diagnostic_val}"
SPLIT="${SPLIT:-val}"
DEVICE="${DEVICE:-cuda}"
MAX_LENGTH="${MAX_LENGTH:-}"
SCORE_MODE="${SCORE_MODE:-}"
CHOICE_BATCH_SIZE="${CHOICE_BATCH_SIZE:-512}"
MAX_CANDIDATE_CHARS="${MAX_CANDIDATE_CHARS:-180}"
ACTION_LABEL_MODE="${ACTION_LABEL_MODE:-}"
CANDIDATE_ORDER_MODE="${CANDIDATE_ORDER_MODE:-candidate_pool}"
CANDIDATE_ORDER_SEED="${CANDIDATE_ORDER_SEED:-20260524}"
DIAG_SAMPLE_LIMIT="${DIAG_SAMPLE_LIMIT:-2000}"
NUM_PERMUTATIONS="${NUM_PERMUTATIONS:-8}"
PERMUTATION_SEED="${PERMUTATION_SEED:-20260524}"
PERMUTATION_INCLUDE_BASE_ORDER="${PERMUTATION_INCLUDE_BASE_ORDER:-true}"
AGGREGATION="${AGGREGATION:-mean_zscore}"
CALIBRATION_ALPHA="${CALIBRATION_ALPHA:-0.5}"
CALIBRATION_MODE="${CALIBRATION_MODE:-content_free_width}"
NO_PROGRESS="${NO_PROGRESS:-false}"

REFERENCE_RANKER_METRICS="${REFERENCE_RANKER_METRICS:-outputs/selectors/vig_utility/saved_step_train_to_val/ranker_eval/selection_metrics.json}"
REFERENCE_SEQUENTIAL_METRICS="${REFERENCE_SEQUENTIAL_METRICS:-outputs/selectors/stage2_sentence_sequential/deberta_sequential_deep/eval_val/selection_metrics.json}"

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs"

common_args=(
  --model-dir "${MODEL_DIR}"
  --oracle-results "${ORACLE_RESULTS}"
  --split "${SPLIT}"
  --device "${DEVICE}"
  --choice-batch-size "${CHOICE_BATCH_SIZE}"
  --max-candidate-chars "${MAX_CANDIDATE_CHARS}"
  --candidate-order-mode "${CANDIDATE_ORDER_MODE}"
  --candidate-order-seed "${CANDIDATE_ORDER_SEED}"
)
if [[ -n "${MODEL_NAME}" ]]; then
  common_args+=(--model-name "${MODEL_NAME}")
fi
if [[ -n "${MAX_LENGTH}" ]]; then
  common_args+=(--max-length "${MAX_LENGTH}")
fi
if [[ -n "${SCORE_MODE}" ]]; then
  common_args+=(--score-mode "${SCORE_MODE}")
fi
if [[ -n "${ACTION_LABEL_MODE}" ]]; then
  common_args+=(--action-label-mode "${ACTION_LABEL_MODE}")
fi
if [[ -n "${DIAG_SAMPLE_LIMIT}" ]]; then
  common_args+=(--sample-limit "${DIAG_SAMPLE_LIMIT}")
fi
references=()
if [[ -n "${REFERENCE_RANKER_METRICS}" && -f "${REFERENCE_RANKER_METRICS}" ]]; then
  references+=("${REFERENCE_RANKER_METRICS}")
fi
if [[ -n "${REFERENCE_SEQUENTIAL_METRICS}" && -f "${REFERENCE_SEQUENTIAL_METRICS}" ]]; then
  references+=("${REFERENCE_SEQUENTIAL_METRICS}")
fi
if [[ "${#references[@]}" -gt 0 ]]; then
  common_args+=(--reference-metrics "${references[@]}")
fi
if [[ "${NO_PROGRESS}" == "true" || "${NO_PROGRESS}" == "1" ]]; then
  common_args+=(--no-progress)
fi

permutation_base_arg=(--permutation-include-base-order)
if [[ "${PERMUTATION_INCLUDE_BASE_ORDER}" == "false" || "${PERMUTATION_INCLUDE_BASE_ORDER}" == "0" ]]; then
  permutation_base_arg=(--no-permutation-include-base-order)
fi

run_eval() {
  local name="$1"
  shift
  local out_dir="${OUTPUT_DIR}/${name}"
  python scripts/phase5_selectors/eval/eval_llm_action_selector.py \
    "${common_args[@]}" \
    --output-dir "${out_dir}" \
    --log-file "${OUTPUT_DIR}/logs/${name}.log" \
    "$@"
}

run_eval "raw" \
  --decode-strategy raw \
  --num-permutations 1 \
  --aggregation "${AGGREGATION}" \
  --calibration-mode none \
  --calibration-alpha 0.0

run_eval "calibrated_alpha${CALIBRATION_ALPHA}" \
  --decode-strategy calibrated \
  --num-permutations 1 \
  --aggregation "${AGGREGATION}" \
  --calibration-mode "${CALIBRATION_MODE}" \
  --calibration-alpha "${CALIBRATION_ALPHA}"

run_eval "perm${NUM_PERMUTATIONS}" \
  --decode-strategy permutation \
  --num-permutations "${NUM_PERMUTATIONS}" \
  --permutation-seed "${PERMUTATION_SEED}" \
  "${permutation_base_arg[@]}" \
  --aggregation "${AGGREGATION}" \
  --calibration-mode none \
  --calibration-alpha 0.0

run_eval "perm${NUM_PERMUTATIONS}_calibrated_alpha${CALIBRATION_ALPHA}" \
  --decode-strategy permutation_calibrated \
  --num-permutations "${NUM_PERMUTATIONS}" \
  --permutation-seed "${PERMUTATION_SEED}" \
  "${permutation_base_arg[@]}" \
  --aggregation "${AGGREGATION}" \
  --calibration-mode "${CALIBRATION_MODE}" \
  --calibration-alpha "${CALIBRATION_ALPHA}"

python scripts/phase5_selectors/eval/summarize_llm_action_decode_diagnostic.py \
  --input-dir "${OUTPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}"

echo "Wrote decode diagnostic summary: ${OUTPUT_DIR}/analysis.md"
