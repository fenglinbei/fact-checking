#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=scripts/phase7_backbone_migration/backbone_cases.sh
source "${SCRIPT_DIR}/backbone_cases.sh"

BACKBONES="${BACKBONES:-$(backbone_transfer_csv)}"
MODE="${MODE:-dry_run}"
FINETUNE="${FINETUNE:-lora}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

split_csv() {
  local raw="$1"
  local -n out_array="$2"
  local items=()
  local item=""
  IFS=',' read -r -a items <<< "${raw}"
  out_array=()
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "${item}" ]]; then
      out_array+=("${item}")
    fi
  done
}

backbones=()
split_csv "${BACKBONES}" backbones
if [[ "${#backbones[@]}" -eq 0 ]]; then
  echo "[backbone-transfer] BACKBONES is empty" >&2
  exit 2
fi

echo "[backbone-transfer] backbones=${backbones[*]}"
echo "[backbone-transfer] mode=${MODE} finetune=${FINETUNE} continue_on_error=${CONTINUE_ON_ERROR}"

failures=()
for backbone in "${backbones[@]}"; do
  require_backbone "${backbone}"
  echo "[backbone-transfer] running ${backbone}"
  if BACKBONE="${backbone}" MODE="${MODE}" FINETUNE="${FINETUNE}" bash "${SCRIPT_DIR}/run_one_backbone.sh"; then
    echo "[backbone-transfer] done ${backbone}"
  else
    echo "[backbone-transfer] failed ${backbone}" >&2
    failures+=("${backbone}")
    if [[ "${CONTINUE_ON_ERROR}" != "true" && "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi
done

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "[backbone-transfer] failures: ${failures[*]}" >&2
  exit 1
fi

echo "[backbone-transfer] all requested transfer backbones completed"
