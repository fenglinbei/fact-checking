# Pointwise Oracle Selector Pipeline 集成

生成日期: 2026-05-17

## 目标

把第一版 `Pointwise evidence utility model` 从离线 selection-only probe 接入正式 `build -> train -> infer` pipeline，使 verifier evaluation 可以直接消费 pointwise selector 产出的 prompt。

## 修改文件

```text
src/fact_checking/oracle_pointwise.py
src/fact_checking/build/candidates.py
src/fact_checking/pipeline/runner.py
configs/experiment/b3_pointwise_oracle_selector_1024.yaml
```

## 代码逻辑

### 1. Pointwise selector 推理 API

`src/fact_checking/oracle_pointwise.py` 新增:

```text
load_pointwise_selector_model()
build_pointwise_inference_pool()
score_pointwise_features()
select_candidates_pointwise_oracle()
```

推理流程:

1. 从 `ChunkMMRSample` 重新计算 `dense_score / lexical_score / bm25_score / hybrid_score`。
2. 按 canonical text 去重。
3. 默认取 hybrid top `top_k * candidate_pool_multiplier` 作为候选池。
4. 构造与训练时一致的 pointwise feature。
5. 加载 `model.npz` 中的 logistic regression 权重、均值、方差和 feature schema。
6. 对候选逐条打 `pointwise_score`。
7. 按 `pointwise_score` 选最终 top-k evidence。

注意: 当前模型仍使用旧 oracle set 训练，旧 oracle set 没有保存严格原始 candidate pool，因此它是可接 pipeline 的 V1 selector，但不是最终权威 selector。

### 2. Build pipeline selection method

`src/fact_checking/build/candidates.py` 新增 selection method:

```yaml
build:
  retrieval:
    selection_method: pointwise_oracle
    pointwise_oracle:
      model_dir: outputs/oracle_pointwise/v1/logreg
      candidate_pool_size: null
      candidate_pool_multiplier: 3
      dump_trace: true
```

输出仍是标准 build JSONL，每行包含:

```text
event_id
claim
label
explain
candidates
prompt
target
gold_label
gold_id
prompt_token_count
evidence_count
```

因此 infer 端不需要改。`_build_training_row()` 会把 pointwise 选出的 candidates 转成 verifier prompt。

每个 split 额外保存 trace:

```text
pointwise_oracle_trace_<split>.jsonl
```

trace 中包含:

```text
event_id
top_k
candidate_pool_size
n_source_candidates
n_pool_candidates
score_mean
score_max
selected[*].pointwise_score
selected[*].hybrid_score
selected[*].text
```

### 3. Build + infer evaluation-only 支持

`src/fact_checking/pipeline/runner.py` 增加了 `pipeline.steps=[build,infer]` 的桥接逻辑。

当 steps 包含 `build` 和 `infer`、但不包含 `train` 时，runner 会根据刚生成的 build outputs 写出本次 run 的:

```text
<run_dir>/configs/train.resolved.yaml
```

然后 infer 自动使用这份 config。这样可以用新的 pointwise evidence 评估已有 verifier checkpoint，而不必重训 verifier。

## 新增实验配置

```text
configs/experiment/b3_pointwise_oracle_selector_1024.yaml
```

该配置继承 `b3_mmr_topk_sweep_1024`，并覆盖:

```yaml
experiment:
  name: b3_pointwise_oracle_selector_1024

build:
  retrieval:
    top_k: 5
    selection_method: pointwise_oracle
    pointwise_oracle:
      model_dir: outputs/oracle_pointwise/v1/logreg
      candidate_pool_size: null
      candidate_pool_multiplier: 3
      dump_trace: true
```

`candidate_pool_size: null` 表示使用:

```text
effective_candidate_pool_size = top_k * candidate_pool_multiplier
```

当前默认即 `5 * 3 = 15`，与 oracle search two-stage pool 尺寸对齐。

## 运行方式

### 1. 配置检查

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  --cfg job \
  experiment=b3_pointwise_oracle_selector_1024 \
  pipeline.mode=build

PYTHONPATH=src python -m fact_checking.pipeline.run \
  --cfg job \
  experiment=b3_pointwise_oracle_selector_1024 \
  'pipeline.steps=[build,infer]' \
  'train.run_dir="outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train"' \
  infer.split=val \
  infer.port=35011
```

确认输出中包含:

```text
build.retrieval.selection_method: pointwise_oracle
build.retrieval.top_k: 5
build.retrieval.pointwise_oracle.model_dir: outputs/oracle_pointwise/v1/logreg
```

### 2. 只构建 pointwise evidence

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b3_pointwise_oracle_selector_1024 \
  pipeline.mode=build \
  pipeline.output_subdir=pointwise_oracle_v1
```

主要输出:

```text
outputs/cache/build/<build_id>/build_train.jsonl
outputs/cache/build/<build_id>/build_val.jsonl
outputs/cache/build/<build_id>/build_test.jsonl
outputs/cache/build/<build_id>/pointwise_oracle_trace_train.jsonl
outputs/cache/build/<build_id>/pointwise_oracle_trace_val.jsonl
outputs/cache/build/<build_id>/pointwise_oracle_trace_test.jsonl
```

检查 candidates 是否带 pointwise 分数:

```bash
jq '{event_id, evidence_count, first_candidate: .candidates[0]}' \
  outputs/cache/build/<build_id>/build_val.jsonl | head -n 80
```

### 3. 用已有 verifier checkpoint 做 evaluation-only

用于只比较 evidence selector，不重训 verifier:

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b3_pointwise_oracle_selector_1024 \
  'pipeline.steps=[build,infer]' \
  pipeline.output_subdir=pointwise_oracle_eval_val \
  'train.run_dir="outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train"' \
  infer.split=val \
  infer.checkpoint=best \
  infer.port=35011
```

输出位置:

```text
outputs/runs/b3_pointwise_oracle_selector_1024/pointwise_oracle_eval_val__<run_id>/
```

其中:

```text
configs/train.resolved.yaml
infer/val/best/<infer_id>/api/metrics.json
infer/val/best/<infer_id>/api/predictions.jsonl
```

这一路径会评估“同一个 verifier checkpoint + pointwise-selected evidence”的下游 accuracy / macro-F1。

### 4. 完整 build -> train -> infer

如果要训练一个使用 pointwise evidence 的 verifier:

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b3_pointwise_oracle_selector_1024 \
  pipeline.mode=full \
  pipeline.output_subdir=pointwise_oracle_full
```

该模式会重新训练 verifier，因此不能单独归因于 evidence selector。

### 5. 覆盖 selector 模型或候选池大小

使用新版 oracle set 重训 pointwise selector 后，可直接覆盖模型目录:

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b3_pointwise_oracle_selector_1024 \
  pipeline.mode=build \
  build.retrieval.pointwise_oracle.model_dir=outputs/oracle_pointwise/v2/logreg
```

若想显式指定候选池大小:

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b3_pointwise_oracle_selector_1024 \
  pipeline.mode=build \
  build.retrieval.pointwise_oracle.candidate_pool_size=20
```

## 建议评估顺序

1. `pipeline.mode=build` 检查 build JSONL 和 trace。
2. `pipeline.steps=[build,infer] infer.split=val` 用已有 verifier checkpoint 做 val evaluation-only。
3. 与原 fixed-MMR `top_k=5, mmr_lambda=0.7` 的 val metrics 对比。
4. 如果 val 有收益，再跑 `infer.split=test`。
5. 若 full split 无收益，额外按 label 分析 `true / mostly-true` 是否被 false-side bias 拉低。

## 已验证

已执行:

```bash
PYTHONPATH=src python -m compileall \
  src/fact_checking/oracle_pointwise.py \
  src/fact_checking/build/candidates.py \
  src/fact_checking/pipeline/runner.py

PYTHONPATH=src python -m fact_checking.pipeline.run \
  --cfg job \
  experiment=b3_pointwise_oracle_selector_1024 \
  pipeline.mode=build

PYTHONPATH=src python -m fact_checking.pipeline.run \
  --cfg job \
  experiment=b3_pointwise_oracle_selector_1024 \
  'pipeline.steps=[build,infer]' \
  'train.run_dir="outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train"' \
  infer.split=val \
  infer.port=35011
```

并用本地 `outputs/cache/chunk_mmr/57e1c87dcd33/val.pkl` + `outputs/oracle_pointwise/v1/logreg/model.npz` 做了单样本 selector sanity check:

```text
n_selected = 5
n_pool = 15
```

本轮未启动 vLLM verifier evaluation。
