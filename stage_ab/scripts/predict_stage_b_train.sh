#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
python -m liar_raw.training.predict_stage_b \
  --config configs/stage_b.yaml \
  --checkpoint outputs/liar-raw/stage_b/best_model.pt \
  --split train
