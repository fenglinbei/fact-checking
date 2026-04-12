#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
python -m liar_raw.training.train_stage_b --config configs/stage_b.yaml
