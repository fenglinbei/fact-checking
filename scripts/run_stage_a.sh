#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src
python -m fact_checking.retrieval.build_stage_a --config configs/stage_a.yaml
