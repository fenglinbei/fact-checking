#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults for the second LoRA tuning pass:
# - halve the effective batch by changing grad accumulation from 8 to 4
# - extend training from 5 to 8 epochs
# - evaluate and checkpoint every 100 optimizer updates
# - stop after 8 evals without improvement
# - apply LIAR-only label-token CE class weights
export LORA_SUFFIX="${LORA_SUFFIX:-_lora_halfbatch_ep8_eval100_pat8_liarw}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2_bsz1_ga4.json}"
export SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-4}"
export SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-8}"
export SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-100}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-$SFT_EVAL_STEPS}"
export SFT_EARLY_STOPPING_PATIENCE="${SFT_EARLY_STOPPING_PATIENCE:-8}"
export LIAR_CLASS_WEIGHTS="${LIAR_CLASS_WEIGHTS:-pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-fact-checking-sentence-trace-method-lora-tuned}"

printf '[lora-tuned] LORA_SUFFIX=%s\n' "$LORA_SUFFIX"
printf '[lora-tuned] DEEPSPEED_CONFIG=%s\n' "$DEEPSPEED_CONFIG"
printf '[lora-tuned] gradient_accumulation_steps=%s num_train_epochs=%s eval_steps=%s save_steps=%s patience=%s\n' \
  "$SFT_GRADIENT_ACCUMULATION_STEPS" "$SFT_NUM_TRAIN_EPOCHS" "$SFT_EVAL_STEPS" "$SFT_SAVE_STEPS" "$SFT_EARLY_STOPPING_PATIENCE"
printf '[lora-tuned] LIAR_CLASS_WEIGHTS=%s\n' "$LIAR_CLASS_WEIGHTS"

exec bash "${SCRIPT_DIR}/run_lora_matrix.sh"
