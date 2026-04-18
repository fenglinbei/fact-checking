# 阶段 A / B

本项目基于你提供的 JSON 格式，实现了一个用于 LIAR-RAW 的**无 oracle（oracle-free）**流水线。

- **阶段 A**：对报道句子做冻结式稠密检索，可选融合简单词法重叠分数与本地 BM25-like 分数，再通过 MMR 做多样化。
- **阶段 B**：使用带有**潜在证据注意力（latent evidence attention）**和**序数分类头（ordinal classification head）**的交叉编码器。训练仅依赖**claim 级 6 分类标签**，**不使用** `is_evidence`。

代码有意拆分为多个模块，而不是单文件实现。

## 目录结构

```text
stage_ab/
├── configs/
│   ├── stage_a.yaml
│   └── stage_b.yaml
├── scripts/
│   ├── run_stage_a.sh
│   ├── train_stage_b.sh
│   └── predict_stage_b.sh
├── src/liar_raw/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── io.py
│   │   └── types.py
│   ├── models/
│   │   ├── latent_evidence.py
│   │   ├── ordinal.py
│   │   └── sparsemax.py
│   ├── retrieval/
│   │   ├── build_stage_a.py
│   │   ├── embedder.py
│   │   ├── mmr.py
│   │   └── text_utils.py
│   └── training/
│       ├── metrics.py
│       ├── predict_stage_b.py
│       ├── stage_b_data.py
│       └── train_stage_b.py
├── pyproject.toml
└── requirements.txt
```

## 原始数据格式要求

每个 split 都是一个 JSON 列表。每条样本格式如下：

```json
{
  "event_id": "11972.json",
  "claim": "Building a wall on the U.S.-Mexico border will take literally years.",
  "label": "true",
  "explain": "...",
  "reports": [
    {
      "report_id": 4815065,
      "link": "https://...",
      "content": "...",
      "domain": "https://...",
      "tokenized": [
        {"sent": "...", "is_evidence": 0},
        {"sent": "...", "is_evidence": 0}
      ]
    }
  ]
}
```

说明：

1. 若存在 `reports[*].tokenized`，加载器会使用 `tokenized[*].sent` 作为句子集合。
2. 若缺失 `tokenized`，加载器会退回到对 `content` 做简单分句。
3. 本流水线会忽略 `is_evidence`。

## 阶段 A 的工作方式

对每条 claim：

1. 收集所有 report 句子
2. 用冻结的 query encoder 对 claim 编码
3. 用冻结的 passage encoder 对每个句子编码
4. 用加权混合分数对句子打分：

```text
hybrid = 0.70 * dense + 0.20 * lexical_overlap + 0.10 * bm25_like
```

5. 运行 MMR，避免返回大量近重复句子
6. 将 top-k 候选以 JSONL 保存

输出文件示例（`stage_a_train.jsonl`）：

```json
{
  "event_id": "11972.json",
  "claim": "Building a wall on the U.S.-Mexico border will take literally years.",
  "label": "true",
  "explain": "...",
  "candidates": [
    {
      "report_id": 123,
      "sent_idx": 0,
      "text": "Engineering experts agree the wall would take years.",
      "dense_score": 0.81,
      "lexical_score": 0.42,
      "bm25_score": 0.68,
      "hybrid_score": 0.76,
      "link": "https://...",
      "domain": "https://..."
    }
  ]
}
```

## 阶段 B 的工作方式

阶段 B 读取阶段 A 的候选结果，并训练 claim 级模型。

对每条 claim，取 top-k 候选句子并构建 `claim + sentence` 配对。

模型会对**每个句子**预测：

- 潜在注意力权重（该句的重要程度）
- 潜在支持概率
- 潜在反驳概率

随后聚合为 claim 级特征：

- support score（支持分数）
- refute score（反驳分数）
- support-refute margin（支持-反驳间隔）
- total evidence strength（总证据强度）
- attention-weighted sentence representation（注意力加权句向量）

这些特征会送入：

- 6 分类头
- CORAL 序数头

损失函数为：

```text
L = cross_entropy
  + λ1 * coral_loss
  + λ2 * margin_regression
  + λ3 * support/refute overlap penalty
  + λ4 * attention entropy penalty
```

该方法仍然是**无 oracle**，因为训练信号仅来自 claim 级标签。

## 环境准备

```bash
cd liar_raw_oracle_free
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## 第 1 步：修改配置

更新 `configs/stage_a.yaml` 中的路径：

```yaml
data:
  train_path: /path/to/liar_raw/train.json
  val_path: /path/to/liar_raw/val.json
  test_path: /path/to/liar_raw/test.json
```

阶段 A 完成后，确认 `configs/stage_b.yaml` 指向新生成的 JSONL 文件。

## 第 2 步：运行阶段 A

```bash
bash scripts/run_stage_a.sh
```

或手动执行：

```bash
PYTHONPATH=src python -m liar_raw.retrieval.build_stage_a --config configs/stage_a.yaml
```

## 第 3 步：训练阶段 B

```bash
bash scripts/train_stage_b.sh
```

或手动执行：

```bash
PYTHONPATH=src python -m liar_raw.training.train_stage_b --config configs/stage_b.yaml
```

最佳 checkpoint 将保存到：

```text
outputs/stage_b/best_model.pt
```

## 第 4 步：预测并导出证据

```bash
bash scripts/predict_stage_b.sh
```

或手动执行：

```bash
PYTHONPATH=src python -m liar_raw.training.predict_stage_b \
  --config configs/stage_b.yaml \
  --checkpoint outputs/stage_b/best_model.pt \
  --split test
```

导出的 JSONL 包含：

- 预测标签
- 各类别概率
- top 支持证据句
- top 反驳证据句

## 实用默认参数

推荐初始设置：

- 阶段 A embedder：`BAAI/bge-base-en-v1.5`
- 阶段 A top-k：`24`
- 阶段 B backbone：`microsoft/deberta-v3-base`
- 阶段 B batch size：`4`
- max length：`256`

若显存/内存紧张：

- 将阶段 A 的 batch size 从 `64` 降到 `16`
- 将阶段 B 的 top-k 从 `24` 降到 `16`
- 将阶段 B 的 max length 从 `256` 降到 `192`
- 仅保留最后 `1` 层 encoder 为可训练状态

## 重要注意事项

1. 本代码刻意保持**无 oracle**，因此不会在任何地方使用 `is_evidence`。
2. 此处阶段 A 是**冻结检索**，不是可训练检索器。
3. 由于 support/refute 是潜在变量，抽取出的证据是模型解释，不是金标监督。
4. 默认阶段 B 实现优化的是 claim 级 macro-F1，而不是句子级 evidence F1。

## 推荐的下一步升级

最值得优先做的升级，是新增一个**小规模、人工审校的句子级开发集**，并仅将其用于模型选择与证据评估。这样既能保持训练阶段无 oracle，又能检查潜在 support/refute 句子是否确实合理。
