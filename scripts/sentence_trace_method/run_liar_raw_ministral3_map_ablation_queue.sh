#!/usr/bin/env bash
# Evidence-Map 消融实验队列（LIAR-RAW, Ministral-3-8B, fixed_topk k=5）
#
# 顺序执行 4 个 map 消融变体：no_map / no_directness / no_confidence / no_relation
# 每个变体走完整流程：训练 selector 权重 → 构建 trace → verifier 推理（复用主方法 checkpoint）
#
# 证据容量对齐：所有变体统一 fixed_topk k=5（minmax5_5），D(full_map) 基准复用已有 minmax5_5 产物。
#
# 用法：
#   bash scripts/sentence_trace_method/run_liar_raw_ministral3_map_ablation_queue.sh           # 等待 GPU 空闲后跑全部
#   WAIT_GPU=false bash scripts/sentence_trace_method/run_liar_raw_ministral3_map_ablation_queue.sh  # 不等待直接跑
#   SMOKE_TEST=true bash scripts/sentence_trace_method/run_liar_raw_ministral3_map_ablation_queue.sh # 单样本冒烟测试
#   START_FROM=no_directness bash scripts/sentence_trace_method/run_liar_raw_ministral3_map_ablation_queue.sh  # 从指定变体开始
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"

# ─── 配置 ───────────────────────────────────────────────────────────────────
# 4 个消融变体（顺序执行）；full_map 基准已有产物不在此队列内
VARIANTS=("no_map" "no_directness" "no_confidence" "no_relation")

# 是否等待 GPU 空闲（默认等待）
WAIT_GPU="${WAIT_GPU:-true}"
# GPU 空闲判定：占用低于此阈值（MiB）的 GPU 数 >= MIN_FREE_GPUS
GPU_FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-2000}"
MIN_FREE_GPUS="${MIN_FREE_GPUS:-4}"
# 轮询间隔（秒）
GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-120}"

# 冒烟测试模式：只跑单样本验证流程通畅
SMOKE_TEST="${SMOKE_TEST:-false}"
# 从指定变体开始（跳过之前的）
START_FROM="${START_FROM:-}"

# 核心运行脚本
CORE_SCRIPT="${SCRIPT_DIR}/run_liar_raw_ministral3_atom_anchor_v0_2_fullpool_policy_lora_ebs16_lr2e5_ep12_eval100.sh"

# 日志目录
LOG_DIR="${ROOT_DIR}/outputs/logs/map_ablation"
mkdir -p "$LOG_DIR"

# ─── 工具函数 ─────────────────────────────────────────────────────────────────
log() {
  printf '[map-ablation-queue %s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

count_free_gpus() {
  # 统计显存占用低于阈值的 GPU 数
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -v th="$GPU_FREE_THRESHOLD_MIB" '$1 < th {c++} END {print c+0}'
}

wait_for_gpu() {
  if [[ "$WAIT_GPU" != "true" ]]; then
    log "WAIT_GPU=false，跳过 GPU 等待"
    return 0
  fi
  log "等待 GPU 空闲（需要 >= ${MIN_FREE_GPUS} 张 GPU 显存 < ${GPU_FREE_THRESHOLD_MIB} MiB）..."
  while true; do
    free=$(count_free_gpus)
    if [[ "$free" -ge "$MIN_FREE_GPUS" ]]; then
      log "检测到 ${free} 张 GPU 空闲，开始执行"
      return 0
    fi
    log "当前 ${free}/${MIN_FREE_GPUS} 张 GPU 空闲，${GPU_POLL_INTERVAL}s 后重试..."
    sleep "$GPU_POLL_INTERVAL"
  done
}

run_variant() {
  local mode="$1"
  local config="configs/experiment/mrec_v0.2/learned_marginal_proxy_fullpool_minmax5_5_map_ablation_${mode}.yaml"
  local log_file="${LOG_DIR}/map_ablation_${mode}_$(date '+%Y%m%d_%H%M%S').log"

  log "===== 变体 ${mode} 开始 ====="
  log "配置: ${config}"
  log "日志: ${log_file}"

  # 环境变量传递给核心脚本
  export MREC_POLICY_CONFIG="$config"
  export MAP_ABLATION_MODE="$mode"
  export FINETUNE_MODE="lora"
  export REQUIRE_PROMPT_INPUT_IDS="false"
  # 强制重新训练权重和构建 trace（每个变体独立）
  export FORCE_WEIGHT_TRAIN="true"
  export FORCE_MREC_BUILD="true"
  export MODE="full"

  if [[ "$SMOKE_TEST" == "true" ]]; then
    export SAMPLE_LIMIT="2"
    log "SMOKE_TEST=true，限制 2 样本"
  else
    export SAMPLE_LIMIT="0"
  fi

  # 执行核心脚本，输出同时到终端和日志文件
  if bash "$CORE_SCRIPT" 2>&1 | tee "$log_file"; then
    log "===== 变体 ${mode} 完成 ====="
    return 0
  else
    local rc=$?
    log "===== 变体 ${mode} 失败 (exit=${rc}) ====="
    log "日志见: ${log_file}"
    return $rc
  fi
}

# ─── 主流程 ───────────────────────────────────────────────────────────────────
log "Evidence-Map 消融实验队列启动"
log "变体列表: ${VARIANTS[*]}"
log "WAIT_GPU=${WAIT_GPU} SMOKE_TEST=${SMOKE_TEST} START_FROM='${START_FROM}'"

wait_for_gpu

# 处理 START_FROM：跳过之前的变体
skip=true
if [[ -z "$START_FROM" ]]; then
  skip=false
fi

failed_variants=()
for variant in "${VARIANTS[@]}"; do
  if [[ "$skip" == "true" ]]; then
    if [[ "$variant" == "$START_FROM" ]]; then
      skip=false
      log "从变体 ${variant} 恢复执行"
    else
      log "跳过变体 ${variant}（START_FROM=${START_FROM}）"
      continue
    fi
  fi

  if ! run_variant "$variant"; then
    failed_variants+=("$variant")
    log "变体 ${variant} 失败，继续下一个（不中断队列）"
  fi
done

# ─── 汇总 ─────────────────────────────────────────────────────────────────────
log "===== 队列执行完毕 ====="
if [[ ${#failed_variants[@]} -eq 0 ]]; then
  log "所有 4 个变体均成功"
  log "full_map 基准（minmax5_5）已有产物，无需重跑"
  log "结果目录:"
  for v in "${VARIANTS[@]}"; do
    log "  ${v}: outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_5_map_ablation_${v}_*/eval/test/"
  done
  exit 0
else
  log "失败的变体: ${failed_variants[*]}"
  log "成功的变体的结果仍可用，可手动重跑失败的"
  exit 1
fi
