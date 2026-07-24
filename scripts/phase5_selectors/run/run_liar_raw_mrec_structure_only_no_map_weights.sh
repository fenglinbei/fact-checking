#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

export MAP_ABLATION_MODE="no_map"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_structure_only_no_map/weights}"

exec bash scripts/phase5_selectors/run/run_liar_raw_mrec_structure_only_weights.sh

