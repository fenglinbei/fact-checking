#!/bin/bash
set -euo pipefail

REMOTE="fenglin@yd.frp-ski.com:/home/fenglin/project/fact-checking/outputs/selectors/evidence_chain_graph"
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

DST="outputs/selectors/"

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

echo "反向同步中（远程 → 本地）..."
# shellcheck disable=SC2086
rsync -avzP -e "ssh -p $SSH_PORT" $DRY_RUN "${EXCLUDES[@]}" "$REMOTE" "$DST"
echo "反向同步完成。"
