# Pointwise V1b True-side Anchor Extension

生成日期: 2026-05-18

## 目的

V1a pointwise selector 排除了 `mostly-true / true`，val verifier evaluation 显示 true-side 被进一步拉低。V1b 的目标不是强行提升 true 类，而是在 retained oracle supervision 之外加入低权重 true-side anchor，避免 selector 继续向 false-side 偏移。

## 修改文件

```text
src/fact_checking/oracle_pointwise.py
scripts/selectors/build_pointwise_oracle_dataset.py
scripts/selectors/train_pointwise_oracle_selector.py
scripts/selectors/eval_pointwise_oracle_selector.py
configs/experiment/b3_pointwise_oracle_selector_v1b_1024.yaml
scripts/selectors/run_pointwise_v1b_vllm_infer.sh
```

## 数据策略

新增 `--filter-preset v1b`:

```text
V1a retained main pool:
  oracle_correct == true
  gold_label in {pants-fire,false,barely-true,half-true}
  final_logprob >= -0.5
  n_candidates > 5
  supervision_weight = 1.0

V1b true-side anchors:
  gold_label in {mostly-true,true}
  n_candidates > 5
  oracle_correct == true
  mostly-true supervision_weight = 0.25
  true supervision_weight = 0.10
```

脚本也支持 conservative fixed-MMR anchor:

```text
--fixed-mmr-predictions <predictions.jsonl>
--fixed-mmr-candidates <aligned build_<split>.jsonl>
```

当 fixed-MMR 在 true-side 样本上预测正确时，对齐的 fixed-MMR evidence set 可作为 anchor positives。若不提供这两个参数，V1b 默认只使用 oracle-correct true-side anchor。

## 训练权重

`compute_row_weights()` 现在读取每个 claim 的 `supervision_weight`。

实现上只对 retained labels 做 label balance；true-side anchor 不进入 retained label balance，避免少量 true anchor 因类别稀缺被反向放大。anchor claim 的最终权重由:

```text
supervision_weight * claim_positive_negative_balance
```

控制。

## Pipeline 配置

新增:

```text
configs/experiment/b3_pointwise_oracle_selector_v1b_1024.yaml
```

默认使用:

```text
outputs/oracle_pointwise/v1b/logreg/model.npz
```

作为 build 阶段 `pointwise_oracle` selector。

## 一键运行脚本

```bash
bash scripts/selectors/run_pointwise_v1b_vllm_infer.sh
```

默认流程:

1. 从 train oracle 构造 V1b pointwise rows。
2. 训练 NumPy logistic regression selector。
3. 在 val oracle 上做 selection-only V1b check。
4. 使用 `pipeline.steps=[build,infer]` 跑 vLLM verifier evaluation。

默认关键路径:

```text
outputs/oracle_pointwise/v1b/data/train_pointwise.jsonl
outputs/oracle_pointwise/v1b/logreg/model.npz
outputs/oracle_pointwise/v1b/logreg/eval_val_v1b/selection_metrics.json
outputs/runs/b3_pointwise_oracle_selector_v1b_1024/pointwise_oracle_v1b_eval_val__<run_id>/infer/val/best/<infer_id>/api/metrics.json
```

常用覆盖:

```bash
INFER_SPLIT=test \
INFER_PORT=35022 \
PIPELINE_OUTPUT_SUBDIR=pointwise_oracle_v1b_eval_test \
bash scripts/selectors/run_pointwise_v1b_vllm_infer.sh
```

只重跑 vLLM infer，不重建数据/不重训:

```bash
SKIP_DATASET=true \
SKIP_TRAIN=true \
SKIP_SELECTION_EVAL=true \
bash scripts/selectors/run_pointwise_v1b_vllm_infer.sh
```

调整 anchor 权重:

```bash
MOSTLY_TRUE_ANCHOR_WEIGHT=0.15 \
TRUE_ANCHOR_WEIGHT=0.05 \
bash scripts/selectors/run_pointwise_v1b_vllm_infer.sh
```

## 已验证

本轮验证范围:

```text
compileall
bash -n run script
Hydra config parse
small-sample V1b data build
small-sample V1b train
```

未在本轮启动完整 vLLM infer。
