#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHON_BIN="${PYTHON_BIN:-/data/liaozijie/conda/accelerate-fc-gemma4/bin/python}"
export LORA_SUFFIX="${LORA_SUFFIX:-_lora_r32a64_d005_ebs16_lr1em5_ep12_eval50_pat8_rawfc}"
export LORA_R="${LORA_R:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-1e-5}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-50}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-50}"

exec bash scripts/sentence_trace_method/run_rawfc_ministral3_v0_7_adaptive5_10_lr1e5_ep12.sh
