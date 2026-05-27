#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
N_STANCE_BUCKETS="${N_STANCE_BUCKETS:-7}"
if [[ -z "${INPUT_BUCKET_FILE:-}" ]]; then
  if [[ "${N_STANCE_BUCKETS}" == "3" ]]; then
    INPUT_BUCKET_FILE="outputs/selectors/count_amplified_stance_bucket_selector/v0_2_${SPLIT}/candidate_stance_buckets_v02_${SPLIT}.jsonl"
  else
    INPUT_BUCKET_FILE="outputs/selectors/count_amplified_stance_bucket_selector/v0_2_${SPLIT}/candidate_stance_buckets_v02_n${N_STANCE_BUCKETS}_${SPLIT}.jsonl"
  fi
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/selectors/oracle_likelihood_constrained_selector/v0_3_1_${SPLIT}}"
FEATURE_SETS="${FEATURE_SETS:-all_features provenance_rank_only all_minus_provenance teacher_directness_stance_only retrieval_quality_only}"
OBJECTIVES="${OBJECTIVES:-pointwise pairwise}"
FOLDS="${FOLDS:-5}"
CROSS_FIT_FOLDS="${CROSS_FIT_FOLDS:-${FOLDS}}"
SEED="${SEED:-20260527}"
EPOCHS="${EPOCHS:-800}"
LR="${LR:-0.05}"
L2="${L2:-0.0001}"
PATIENCE="${PATIENCE:-80}"
EVAL_EVERY="${EVAL_EVERY:-10}"
DEV_FRACTION="${DEV_FRACTION:-0.1}"
TOP_K="${TOP_K:-5}"
ANCHOR_K="${ANCHOR_K:-0}"
SOURCE_PENALTY="${SOURCE_PENALTY:-0}"
STANCE_REGION_PENALTY="${STANCE_REGION_PENALTY:-0}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"

SAMPLE_ARGS=()
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  SAMPLE_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

echo "[oracle-likelihood-v0.3.1] split       : ${SPLIT}"
echo "[oracle-likelihood-v0.3.1] buckets     : ${INPUT_BUCKET_FILE}"
echo "[oracle-likelihood-v0.3.1] output root : ${OUTPUT_ROOT}"
echo "[oracle-likelihood-v0.3.1] objectives  : ${OBJECTIVES}"
echo "[oracle-likelihood-v0.3.1] feature sets: ${FEATURE_SETS}"
echo "[oracle-likelihood-v0.3.1] folds       : ${CROSS_FIT_FOLDS}"
echo "[oracle-likelihood-v0.3.1] eval anchor : ${ANCHOR_K}"
echo "[oracle-likelihood-v0.3.1] source pen  : ${SOURCE_PENALTY}"
echo "[oracle-likelihood-v0.3.1] stance pen  : ${STANCE_REGION_PENALTY}"

mkdir -p "${OUTPUT_ROOT}"

for objective in ${OBJECTIVES}; do
  for feature_set in ${FEATURE_SETS}; do
    run_dir="${OUTPUT_ROOT}/${objective}_${feature_set}"
    echo "[oracle-likelihood-v0.3.1] train objective=${objective} feature_set=${feature_set}"
    PYTHONPATH=src python scripts/phase5_selectors/train/train_oracle_likelihood_constrained_selector.py \
      --candidate-stance-buckets "${INPUT_BUCKET_FILE}" \
      --output-dir "${run_dir}" \
      --split "${SPLIT}" \
      --cross-fit-folds "${CROSS_FIT_FOLDS}" \
      --feature-set "${feature_set}" \
      --objective "${objective}" \
      --seed "${SEED}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --l2 "${L2}" \
      --patience "${PATIENCE}" \
      --eval-every "${EVAL_EVERY}" \
      --dev-fraction "${DEV_FRACTION}" \
      "${SAMPLE_ARGS[@]}"

    echo "[oracle-likelihood-v0.3.1] eval objective=${objective} feature_set=${feature_set}"
    PYTHONPATH=src python scripts/phase5_selectors/eval/eval_oracle_likelihood_constrained_selector.py \
      --scored-candidates "${run_dir}/candidate_oracle_likelihood_scores_${SPLIT}.jsonl" \
      --output-dir "${run_dir}/eval" \
      --split "${SPLIT}" \
      --top-k "${TOP_K}" \
      --anchor-k "${ANCHOR_K}" \
      --source-penalty "${SOURCE_PENALTY}" \
      --stance-region-penalty "${STANCE_REGION_PENALTY}" \
      "${SAMPLE_ARGS[@]}"
  done
done

echo "[oracle-likelihood-v0.3.1] done: ${OUTPUT_ROOT}"
