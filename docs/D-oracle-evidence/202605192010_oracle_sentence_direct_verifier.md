# Oracle Sentence Evidence Direct Verifier 实现说明

生成日期：2026-05-19

## 目标

验证 sentence-level Stage2 oracle evidence supervision 能否直接转化为 verifier 下游泛化。

这个实验不训练 selector，也不使用 fixed-MMR / pointwise / reranker 重新选证据，而是直接把 oracle search 已选出的 evidence set 渲染成 verifier 训练样本：

```text
oracle_results_<split>.jsonl
-> selected_indices
-> candidate_pool[selected_indices]
-> normal build_<split>.jsonl prompt/target
-> sft.label_token_trainer
```

它回答的问题是：

> 如果 verifier 直接看到 sentence-level oracle-selected evidence，label-token CE verifier 是否能学到比 fixed-MMR 更好的分类行为？

如果 direct verifier 仍不能提升，说明瓶颈不只是 selector 学不到 oracle evidence，而可能是 verifier 无法把这类 oracle evidence supervision 转成可泛化决策。

## 输入产物

当前主线输入：

```text
train oracle:
outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl

val oracle:
outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl
```

这两份产物都是 sentence-level Stage2 margin oracle：

```text
chunk_mmr_fingerprint = 432dfc970e75
candidate pool = dedup -> hybrid top15
oracle set = greedy margin top5
```

raw data 用于补齐 `explain` 和标准化后的 label：

```text
data/raw/LIAR-RAW/train.json
data/raw/LIAR-RAW/val.json
data/raw/LIAR-RAW/test.json
```

## 新增文件

| 文件 | 作用 |
|---|---|
| `scripts/oracle_evidence/build_oracle_direct_verifier_data.py` | 从 oracle results 构造 verifier-ready `build_<split>.jsonl` |
| `configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml` | 复用 Stage1 label-token CE 训练配置，标记为 oracle direct verifier 实验 |
| `scripts/verifier/run_oracle_sentence_direct_verifier.sh` | 构造数据；可选启动 label-token CE 训练 |

## 构造逻辑

`build_oracle_direct_verifier_data.py` 对每个 split 执行：

1. 读取 raw LIAR-RAW split，建立 `event_id -> SampleRecord`。
2. 读取 `oracle_results_<split>.jsonl`。
3. 校验每条 oracle row 的：

```text
candidate_pool_metadata.chunk_mmr_fingerprint == 432dfc970e75
selected_indices 均落在 candidate_pool 范围内
```

4. 用 `selected_indices` 从 `candidate_pool` 中取出 evidence candidates。
5. 默认保持 oracle greedy 选择顺序：

```text
order = oracle
```

也可显式改成：

```text
order = hybrid         # 按 hybrid_score 降序
order = candidate_pool # 按 candidate_pool index 升序
```

6. 调用正式 build pipeline 的 prompt 构造函数：

```text
fact_checking.build.candidates._build_training_row()
```

因此生成的 prompt/target 格式与正常 `build_<split>.jsonl` 一致，包括：

```text
prompt
target
gold_label
gold_id
gold_explain
prompt_token_count
target_token_count
evidence_count
was_truncated
candidates
```

7. 在每条 row 额外写入审计字段：

```json
"oracle_direct": {
  "source_oracle_results": "...",
  "candidate_pool_fingerprint": "...",
  "chunk_mmr_fingerprint": "432dfc970e75",
  "search_objective": "margin",
  "oracle_is_correct": true,
  "oracle_pred_label": "...",
  "oracle_margin": 0.12,
  "oracle_selected_indices": [4, 2, 7, 1, 11],
  "order": "oracle"
}
```

8. 输出 `build_report.json`，记录每个 split 的样本数、label 分布、fingerprint 计数、oracle correct rate、prompt token 分布和 truncation rate。

## 输出目录

默认 wrapper 输出：

```text
outputs/oracle_direct_verifier/stage2_sentence/
```

文件：

```text
build_train.jsonl
build_val.jsonl
build_report.json
train.resolved.yaml
```

如果提供 `ORACLE_TEST`，还会生成：

```text
build_test.jsonl
```

否则 `train.resolved.yaml` 中的 `test_candidates` 会暂时指向 `build_val.jsonl`，因为 label-token trainer 本身只读取 train/val；真正 test infer 需要单独准备 test evidence。

## 运行方式

### 1. 只构造数据

```bash
bash scripts/verifier/run_oracle_sentence_direct_verifier.sh
```

等价于：

```bash
PYTHONPATH=src python scripts/oracle_evidence/build_oracle_direct_verifier_data.py \
  --config configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml \
  --train-oracle-results outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl \
  --val-oracle-results outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl \
  --output-dir outputs/oracle_direct_verifier/stage2_sentence \
  --expected-chunk-mmr-fingerprint 432dfc970e75 \
  --model-base-path /data/models/ \
  --order oracle \
  --filter all
```

### 2. 构造数据并启动训练

目标服务器上运行：

```bash
RUN_TRAIN=true \
bash scripts/verifier/run_oracle_sentence_direct_verifier.sh
```

训练入口是：

```text
python -m sft.label_token_trainer --config outputs/oracle_direct_verifier/stage2_sentence/train.resolved.yaml
```

wrapper 会用 `accelerate launch + deepspeed_zero2_bsz8_ga1` 启动，默认：

```text
NPROC_PER_NODE=4
NUM_MACHINES=1
MIXED_PRECISION=bf16
DEEPSPEED_CONFIG=configs/deepspeed_zero2_bsz8_ga1.json
```

可用环境变量覆盖：

```bash
OUTPUT_DIR=outputs/oracle_direct_verifier/stage2_sentence_order_hybrid \
ORDER=hybrid \
RUN_TRAIN=true \
bash scripts/verifier/run_oracle_sentence_direct_verifier.sh
```

### 3. 小样本 smoke test

```bash
SAMPLE_LIMIT=32 \
RUN_TRAIN=false \
bash scripts/verifier/run_oracle_sentence_direct_verifier.sh
```

本机如果没有 `/data/models/Qwen2.5-7B-Instruct`，可以临时指定本地 tokenizer 只做格式 smoke：

```bash
PROMPT_MODEL_NAME_OR_PATH=/path/to/local/qwen-tokenizer \
TRAIN_MODEL_NAME_OR_PATH=/path/to/local/qwen-model \
SAMPLE_LIMIT=5 \
RUN_TRAIN=false \
bash scripts/verifier/run_oracle_sentence_direct_verifier.sh
```

## 过滤选项

默认使用所有 oracle rows：

```text
FILTER=all
```

也支持诊断性过滤：

```text
FILTER=oracle_correct   # 只保留 Stage1 verifier 在 oracle evidence 上已判对的样本
FILTER=margin_positive  # 只保留 margin > 0 的样本
```

主实验建议先用 `FILTER=all`。因为 direct verifier 目标是检验“gold-conditioned best evidence set”作为训练监督是否可泛化，而不是只训练 Stage1 已经能判对的 easy subset。

## 已验证

本地执行了语法检查：

```bash
PYTHONPATH=src python -m compileall scripts/oracle_evidence/build_oracle_direct_verifier_data.py
```

本地使用可用 tokenizer 做了 5 样本 smoke：

```bash
PYTHONPATH=src python scripts/oracle_evidence/build_oracle_direct_verifier_data.py \
  --config configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml \
  --train-oracle-results outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl \
  --val-oracle-results outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl \
  --output-dir outputs/oracle_direct_verifier/stage2_sentence_smoke \
  --expected-chunk-mmr-fingerprint 432dfc970e75 \
  --prompt-model-name-or-path /home/fenglin/project/hateSpeechDetection/models/Qwen3-1.7B \
  --train-model-name-or-path /home/fenglin/project/hateSpeechDetection/models/Qwen3-1.7B \
  --sample-limit 5 \
  --no-progress
```

结果：

```text
train: rows=5 skipped=0 oracle_acc=0.8000 trunc=0.0000
val: rows=5 skipped=0 oracle_acc=0.4000 trunc=0.0000
```

抽查 `build_train.jsonl`：

```text
event_id = 324.json
gold_label = mostly-true
target = Label: E
evidence_count = 5
chunk_mmr_fingerprint = 432dfc970e75
order = oracle
```

本机缺少 `accelerate`，因此未在本机启动 label-token CE 训练；目标服务器环境已有训练依赖，直接使用 wrapper 的 `RUN_TRAIN=true` 路径。

## 判读标准

Direct verifier 是 selector 之前的必要验证：

| 结果 | 解读 |
|---|---|
| 明显高于 fixed-MMR / pointwise full | oracle evidence supervision 可被 verifier 吸收，后续值得做更强 selector |
| 接近 fixed-MMR | oracle evidence 上界主要来自 Stage1 verifier 的 gold-conditioned search，训练后泛化转化有限 |
| 低于 fixed-MMR | 问题不只是 selector，可能是 oracle evidence set 对训练 verifier 不稳定或引入分布偏移 |

该实验不应用作 deployable test 结果，因为 train/val evidence 是 gold-conditioned oracle search 得到的；它只用于判断后续 selector 学习是否值得继续。
