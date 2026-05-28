#!/usr/bin/env bash
set -euo pipefail

SPLIT="${SPLIT:-val}"
ORACLE_LIKELIHOOD_SCORED="${ORACLE_LIKELIHOOD_SCORED:-outputs/selectors/oracle_likelihood_constrained_selector/v0_3_1_${SPLIT}/pointwise_all_features/candidate_oracle_likelihood_scores_${SPLIT}.jsonl}"
DIRECT_CE_SCORED="${DIRECT_CE_SCORED:-outputs/selectors/direct_evidence_cross_encoder/v0_4a_1_${SPLIT}_default_query/direct_ce_scored_candidates_${SPLIT}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/direct_evidence_cross_encoder/v0_4d_${SPLIT}_default_query_fusion}"
TOP_K="${TOP_K:-5}"
LAMBDAS="${LAMBDAS:-0,0.05,0.10,0.20,0.30,0.50}"
CROSS_FIT_FOLDS="${CROSS_FIT_FOLDS:-5}"
SEED="${SEED:-20260527}"
EPOCHS="${EPOCHS:-800}"
LR="${LR:-0.05}"
L2="${L2:-0.0001}"
PATIENCE="${PATIENCE:-80}"
EVAL_EVERY="${EVAL_EVERY:-10}"
DEV_FRACTION="${DEV_FRACTION:-0.1}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"

SAMPLE_ARGS=()
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  SAMPLE_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

echo "[direct-ce-fusion-v0.4d] split       : ${SPLIT}"
echo "[direct-ce-fusion-v0.4d] oracle file : ${ORACLE_LIKELIHOOD_SCORED}"
echo "[direct-ce-fusion-v0.4d] direct file : ${DIRECT_CE_SCORED}"
echo "[direct-ce-fusion-v0.4d] output dir  : ${OUTPUT_DIR}"
echo "[direct-ce-fusion-v0.4d] lambdas     : ${LAMBDAS}"
echo "[direct-ce-fusion-v0.4d] folds       : ${CROSS_FIT_FOLDS}"

PYTHONPATH=src python scripts/phase5_selectors/eval/eval_direct_evidence_fusion_v0_4d.py \
  --oracle-likelihood-scored "${ORACLE_LIKELIHOOD_SCORED}" \
  --direct-ce-scored "${DIRECT_CE_SCORED}" \
  --output-dir "${OUTPUT_DIR}" \
  --split "${SPLIT}" \
  --top-k "${TOP_K}" \
  --lambdas "${LAMBDAS}" \
  --cross-fit-folds "${CROSS_FIT_FOLDS}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --l2 "${L2}" \
  --patience "${PATIENCE}" \
  --eval-every "${EVAL_EVERY}" \
  --dev-fraction "${DEV_FRACTION}" \
  "${SAMPLE_ARGS[@]}"

echo "[direct-ce-fusion-v0.4d] done: ${OUTPUT_DIR}"
