#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH=src
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

python -m sft.test_eval "$@"
