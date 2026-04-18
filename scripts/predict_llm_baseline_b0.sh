#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/run_llm_baseline.py --config configs/baseline_b0.yaml --split test
