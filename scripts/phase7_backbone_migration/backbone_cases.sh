#!/usr/bin/env bash
# Shared registry for RAWFC v0.6c backbone migration runs.

BACKBONE_CASE_ORDER=(
  qwen25_15b
  qwen3_17b
  qwen25_3b
  qwen3_4b_2507
  qwen3_8b
  dsr1_qwen7b
)

BACKBONE_TRANSFER_CASE_ORDER=(
  llama31_8b
  phi4_mini
  gemma4_e4b
  ministral3_8b
)

backbone_default_csv() {
  local IFS=,
  printf "%s" "${BACKBONE_CASE_ORDER[*]}"
}

backbone_transfer_csv() {
  local IFS=,
  printf "%s" "${BACKBONE_TRANSFER_CASE_ORDER[*]}"
}

backbone_path() {
  case "$1" in
    qwen25_15b) printf "%s" "/data/models/Qwen2.5-1.5B-Instruct" ;;
    qwen3_17b) printf "%s" "/data/models/Qwen3-1.7B" ;;
    qwen25_3b) printf "%s" "/data/models/Qwen2.5-3B-Instruct" ;;
    qwen3_4b_2507) printf "%s" "/data/models/Qwen3-4B-Instruct-2507" ;;
    qwen3_8b) printf "%s" "/data/models/Qwen3-8B" ;;
    dsr1_qwen7b) printf "%s" "/data/models/DeepSeek-R1-Distill-Qwen-7B" ;;
    llama31_8b) printf "%s" "/data/models/Meta-Llama-3.1-8B-Instruct" ;;
    phi4_mini) printf "%s" "/data/models/Phi-4-mini-instruct" ;;
    gemma4_e4b) printf "%s" "/data/models/gemma-4-E4B-it" ;;
    ministral3_8b) printf "%s" "/data/models/Ministral-3-8B-Instruct-2512" ;;
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
    llama31_8b) printf "%s" "8.0" ;;
    phi4_mini) printf "%s" "3.8" ;;
    gemma4_e4b) printf "%s" "8.0" ;;
    ministral3_8b) printf "%s" "8.4" ;;
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
    llama31_8b) printf "%s" "80" ;;
    phi4_mini) printf "%s" "38" ;;
    gemma4_e4b) printf "%s" "80" ;;
    ministral3_8b) printf "%s" "84" ;;
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
    llama31_8b) printf "%s" "Meta-Llama-3.1-8B-Instruct" ;;
    phi4_mini) printf "%s" "Phi-4-mini-instruct" ;;
    gemma4_e4b) printf "%s" "Gemma-4-E4B-it" ;;
    ministral3_8b) printf "%s" "Ministral-3-8B-Instruct-2512" ;;
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
