#!/usr/bin/env bash
# Shared registry for RAWFC v0.6c A-group backbone migration runs.

BACKBONE_CASE_ORDER=(
  qwen25_15b
  qwen3_17b
  qwen25_3b
  qwen3_4b_2507
  qwen3_8b
  dsr1_qwen7b
)

backbone_default_csv() {
  local IFS=,
  printf "%s" "${BACKBONE_CASE_ORDER[*]}"
}

backbone_path() {
  case "$1" in
    qwen25_15b) printf "%s" "/data/models/Qwen2.5-1.5B-Instruct" ;;
    qwen3_17b) printf "%s" "/data/models/Qwen3-1.7B" ;;
    qwen25_3b) printf "%s" "/data/models/Qwen2.5-3B-Instruct" ;;
    qwen3_4b_2507) printf "%s" "/data/models/Qwen3-4B-Instruct-2507" ;;
    qwen3_8b) printf "%s" "/data/models/Qwen3-8B" ;;
    dsr1_qwen7b) printf "%s" "/data/models/DeepSeek-R1-Distill-Qwen-7B" ;;
    *) return 1 ;;
  esac
}

backbone_size_b() {
  case "$1" in
    qwen25_15b) printf "%s" "1.5" ;;
    qwen3_17b) printf "%s" "1.7" ;;
    qwen25_3b) printf "%s" "3.0" ;;
    qwen3_4b_2507) printf "%s" "4.0" ;;
    qwen3_8b) printf "%s" "8.0" ;;
    dsr1_qwen7b) printf "%s" "7.0" ;;
    *) return 1 ;;
  esac
}

backbone_size_tenths() {
  case "$1" in
    qwen25_15b) printf "%s" "15" ;;
    qwen3_17b) printf "%s" "17" ;;
    qwen25_3b) printf "%s" "30" ;;
    qwen3_4b_2507) printf "%s" "40" ;;
    qwen3_8b) printf "%s" "80" ;;
    dsr1_qwen7b) printf "%s" "70" ;;
    *) return 1 ;;
  esac
}

backbone_label() {
  case "$1" in
    qwen25_15b) printf "%s" "Qwen2.5-1.5B-Instruct" ;;
    qwen3_17b) printf "%s" "Qwen3-1.7B" ;;
    qwen25_3b) printf "%s" "Qwen2.5-3B-Instruct" ;;
    qwen3_4b_2507) printf "%s" "Qwen3-4B-Instruct-2507" ;;
    qwen3_8b) printf "%s" "Qwen3-8B" ;;
    dsr1_qwen7b) printf "%s" "DeepSeek-R1-Distill-Qwen-7B" ;;
    *) return 1 ;;
  esac
}

backbone_is_known() {
  backbone_path "$1" >/dev/null
}

backbone_is_large_fullft() {
  local size_tenths
  size_tenths="$(backbone_size_tenths "$1")" || return 1
  [[ "${size_tenths}" -ge 70 ]]
}

backbone_case_name() {
  local backbone="$1"
  local finetune="$2"
  printf "v0_6c_rawfc3_rule_step_adaptive5_10_eval25_%s_%s" "${backbone}" "${finetune}"
}

require_backbone() {
  local backbone="$1"
  if ! backbone_is_known "${backbone}"; then
    echo "[backbone-migration] unknown BACKBONE=${backbone}" >&2
    echo "[backbone-migration] known: $(backbone_default_csv)" >&2
    return 1
  fi
}
