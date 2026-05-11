#!/usr/bin/env bash
# 用 B2a merge 修复重跑 test 推理（复用已有 best checkpoint）
set -euo pipefail

cd /data/liaozijie/fact-checking
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

LAMBDAS=(0.0 0.1 0.2 0.3 0.4 0.5)

for lam in "${LAMBDAS[@]}"; do
  # 找到含有 train/best/ 的训练目录（排除 infer-only 产生的空目录）
  run_dir=""
  for d in outputs/runs/mmr_lambda_sweep_1024/build.retrieval.mmr_lambda-${lam}__*; do
    if [ -d "$d/train/best" ]; then
      run_dir="$d"
      break
    fi
  done

  if [ -z "$run_dir" ]; then
    echo "=== lambda=${lam}: no train dir with train/best/, skip ==="
    continue
  fi

  echo "=== lambda=${lam} run_dir=${run_dir} ==="

  # pipeline 的 _run_infer 需要 run_dir/configs/train.resolved.yaml
  mkdir -p "${run_dir}/configs"
  if [ ! -f "${run_dir}/configs/train.resolved.yaml" ]; then
    cp "${run_dir}/train/config.resolved.yaml" "${run_dir}/configs/train.resolved.yaml"
  fi

  python -m fact_checking.pipeline.run \
    experiment=mmr_lambda_sweep_1024 \
    pipeline.mode=infer \
    pipeline.force.infer=true \
    "build.retrieval.mmr_lambda=${lam}" \
    infer.split=test \
    "pipeline.run_dir=${run_dir}"
done
