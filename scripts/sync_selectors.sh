#!/bin/bash
set -euo pipefail

# ============================================================
# 远程目标: <user@host>:/absolute/path/to/outputs/selectors/
# ============================================================
REMOTE="fenglin@yd.frp-ski.com:/home/fenglin/project/fact-checking/outputs/selectors/"

# ============================================================
# 排除规则: 模型权重 + 大文件
# ============================================================
EXCLUDES=(
    --exclude='*.safetensors'
    --exclude='*.pt'
    --exclude='*.pth'
    --exclude='*.bin'
    --exclude='*.ckpt'
    --exclude='tokenizer.json'
    --exclude='spm.model'
    --exclude='sentencepiece.bpe.model'
    # --exclude='*.jsonl'
)

SSH_PORT=16880
SRC="outputs/selectors/"

if [ -z "$REMOTE" ]; then
    echo "请先编辑脚本设置 REMOTE 变量为目标地址。"
    echo "  格式: user@host:/path/to/outputs/selectors/"
    exit 1
fi

if [ "${1:-}" == "--dry-run" ]; then
    rsync -avzP -e "ssh -p $SSH_PORT" --dry-run "${EXCLUDES[@]}" "$SRC" "$REMOTE"
else
    echo "同步中..."
    rsync -avzP -e "ssh -p $SSH_PORT" "${EXCLUDES[@]}" "$SRC" "$REMOTE"
    echo "同步完成。"
fi
