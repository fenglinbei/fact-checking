#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_LIAR_RAW="${RUN_LIAR_RAW:-false}"
export RUN_RAWFC="${RUN_RAWFC:-true}"
export RAWFC_CASE_SUFFIX="${RAWFC_CASE_SUFFIX:-__mrec_min_anchor_only}"
export TRACE_PROMPT_STYLE="${TRACE_PROMPT_STYLE:-mrec_min}"

bash "${SCRIPT_DIR}/run_mrec_ministral3_main.sh"
