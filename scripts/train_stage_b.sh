#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
python -m fact_checking.training.train_stage_b --config configs/stage_b.yaml
