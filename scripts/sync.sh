#!/bin/bash
set -euo pipefail

REMOTE="fenglin@yd.frp-ski.com:/home/fenglin/project/fact-checking/outputs"
SSH_PORT=16880
ALL=false
DRY_RUN=""

for arg in "${@}"; do
    case "$arg" in
        --all) ALL=true ;;
        --dry-run) DRY_RUN="--dry-run" ;;
        *) SUB="$arg" ;;
    esac
done

SRC="outputs/oracle_evidence"

EXCLUDES=()
if [ "$ALL" != true ]; then
    EXCLUDES=(
        --exclude='*.safetensors'
        --exclude='*.pt'
        --exclude='*.pth'
        --exclude='*.bin'
        --exclude='*.ckpt'
        --exclude='tokenizer.json'
        --exclude='spm.model'
        --exclude='sentencepiece.bpe.model'
    )
fi

if [ -z "$REMOTE" ]; then
    echo "请先编辑脚本设置 REMOTE 变量为目标地址。"
    exit 1
fi

echo "同步中..."
# shellcheck disable=SC2086
rsync -avzP -e "ssh -p $SSH_PORT" $DRY_RUN "${EXCLUDES[@]}" "$SRC" "$REMOTE"
echo "同步完成。"
