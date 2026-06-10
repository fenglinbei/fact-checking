#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

printf '[lora-tuned] Deprecated wrapper name; forwarding to eval100/pat8 defaults.\n'
exec bash "${SCRIPT_DIR}/run_lora_matrix_halfbatch_ep8_eval100_pat8_liar_weights.sh" "$@"
