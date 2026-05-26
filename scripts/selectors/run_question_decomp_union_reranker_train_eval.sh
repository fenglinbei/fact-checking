#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

TRAIN_UNION_POOL_JSONL="${TRAIN_UNION_POOL_JSONL:-outputs/selectors/question_decomp_retrieval/qwen_v0_train/union_candidate_pool_train.jsonl}"
VAL_UNION_POOL_JSONL="${VAL_UNION_POOL_JSONL:-outputs/selectors/question_decomp_retrieval/qwen_v0_val/union_candidate_pool_val.jsonl}"
TRAIN_ORACLE_RESULTS="${TRAIN_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl}"
VAL_ORACLE_RESULTS="${VAL_ORACLE_RESULTS:-outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/question_decomp_retrieval/qwen_v0_train_val/union_feature_reranker_v0_4}"

SELECTOR_TOP_K="${SELECTOR_TOP_K:-5}"
RERANKER_SEED="${RERANKER_SEED:-20260526}"
RERANKER_EPOCHS="${RERANKER_EPOCHS:-500}"
RERANKER_LR="${RERANKER_LR:-0.05}"
RERANKER_L2="${RERANKER_L2:-0.0001}"
RERANKER_PATIENCE="${RERANKER_PATIENCE:-50}"
RERANKER_EVAL_EVERY="${RERANKER_EVAL_EVERY:-10}"
USE_VAL_FOR_EARLY_STOPPING="${USE_VAL_FOR_EARLY_STOPPING:-false}"

TRAIN_SAMPLE_LIMIT_ARGS=()
if [[ -n "${TRAIN_SAMPLE_LIMIT:-}" ]]; then
  TRAIN_SAMPLE_LIMIT_ARGS=(--train-sample-limit "${TRAIN_SAMPLE_LIMIT}")
fi

VAL_SAMPLE_LIMIT_ARGS=()
if [[ -n "${VAL_SAMPLE_LIMIT:-}" ]]; then
  VAL_SAMPLE_LIMIT_ARGS=(--val-sample-limit "${VAL_SAMPLE_LIMIT}")
fi

EARLY_STOP_ARGS=()
if [[ "${USE_VAL_FOR_EARLY_STOPPING}" == "1" || "${USE_VAL_FOR_EARLY_STOPPING}" == "true" || "${USE_VAL_FOR_EARLY_STOPPING}" == "True" ]]; then
  EARLY_STOP_ARGS=(--use-val-for-early-stopping)
fi

echo "[qd-reranker-train-eval] train union : ${TRAIN_UNION_POOL_JSONL}"
echo "[qd-reranker-train-eval] val union   : ${VAL_UNION_POOL_JSONL}"
echo "[qd-reranker-train-eval] output dir  : ${OUTPUT_DIR}"
echo "[qd-reranker-train-eval] top k       : ${SELECTOR_TOP_K}"
echo "[qd-reranker-train-eval] val early stop: ${USE_VAL_FOR_EARLY_STOPPING}"

PYTHONPATH=src python scripts/selectors/train_eval_question_decomp_union_reranker.py \
  --train-union-pool-jsonl "${TRAIN_UNION_POOL_JSONL}" \
  --val-union-pool-jsonl "${VAL_UNION_POOL_JSONL}" \
  --train-oracle-results "${TRAIN_ORACLE_RESULTS}" \
  --val-oracle-results "${VAL_ORACLE_RESULTS}" \
  --output-dir "${OUTPUT_DIR}" \
  --top-k "${SELECTOR_TOP_K}" \
  --seed "${RERANKER_SEED}" \
  --epochs "${RERANKER_EPOCHS}" \
  --lr "${RERANKER_LR}" \
  --l2 "${RERANKER_L2}" \
  --patience "${RERANKER_PATIENCE}" \
  --eval-every "${RERANKER_EVAL_EVERY}" \
  "${EARLY_STOP_ARGS[@]}" \
  "${TRAIN_SAMPLE_LIMIT_ARGS[@]}" \
  "${VAL_SAMPLE_LIMIT_ARGS[@]}" \
  "$@"
