#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
export CASE_SUFFIX="${CASE_SUFFIX:-__v0_7_bm_adaptive5_10_fullft_ebs16_lr1em6_ep3_eval50_pat3_rawfc}"
export SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-6}"
export SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-3}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-50}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-$SFT_EVAL_STEPS}"
export SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-3}"

exec bash scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_lr1e6_ep12_fullft_aligned.sh
