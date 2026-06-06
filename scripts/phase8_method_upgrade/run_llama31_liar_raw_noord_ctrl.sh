#!/usr/bin/env bash
set -euo pipefail

# No-ordinal control for the Llama-3.1 FullFT LIAR-RAW method-upgrade run.
# This wrapper intentionally reuses run_llama31_liar_raw.sh so the data,
# DeepSpeed, and inference settings stay aligned with the ordinal variant.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ENABLE_ORDINAL_LOSS=false
export ORDINAL_LOSS_ALPHA="${ORDINAL_LOSS_ALPHA:-0}"
export METHOD_SUFFIX="${METHOD_SUFFIX:-_noord_ctrl}"

exec bash "${SCRIPT_DIR}/run_llama31_liar_raw.sh" "$@"
