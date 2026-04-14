# LIAR-RAW 后半段管线：Stage C / D / E

这套工程接在你已经完成的 **Stage A / B** 后面，补齐：

- **Stage C**：Claim complexity gating + sub-claim decomposition
- **Stage D**：Sub-claim evidence graph verifier（图式再判别）
- **Stage E**：Extractive / Abstractive explanation generation（解释生成）

## 依赖的输入

你需要两类输入：

1. **原始 split JSON**
   - `train.json`
   - `val.json`
   - `test.json`

2. **Stage B prediction JSONL**
   每条至少包含这些字段：
   - `event_id`
   - `claim`
   - `gold_label` 或 `label`
   - `support_evidence` : list[dict]
   - `refute_evidence` : list[dict]

推荐兼容的 evidence schema：
```json
{
  "event_id": "11972.json",
  "claim": "...",
  "gold_label": "true",
  "pred_label": "true",
  "support_evidence": [
    {
      "sentence": "...",
      "report_id": 4815065,
      "position": 6,
      "domain": "https://...",
      "importance": 0.41,
      "latent_support_score": 0.31
    }
  ],
  "refute_evidence": [
    {
      "sentence": "...",
      "report_id": 4815066,
      "position": 3,
      "domain": "https://...",
      "importance": 0.22,
      "latent_refute_score": 0.18
    }
  ]
}
```

如果字段名略有不同，只要能从 evidence dict 里读到 sentence / report_id / position / domain / score，本工程都能兼容。

---

## 目录

```text
liar_raw_stage_cde/
├── configs/
│   ├── build_graph_inputs.yaml
│   ├── graph_verifier.yaml
│   └── explainer.yaml
├── requirements.txt
├── scripts/
│   ├── build_graph_inputs.py
│   ├── train_graph_verifier.py
│   ├── predict_graph_verifier.py
│   ├── train_explainer.py
│   └── generate_explanations.py
└── src/
    └── liar_raw_cde/
        ├── __init__.py
        ├── utils/
        │   ├── __init__.py
        │   ├── io.py
        │   ├── labels.py
        │   ├── seed.py
        │   └── text.py
        ├── stage_c/
        │   ├── __init__.py
        │   ├── gating.py
        │   ├── decompose.py
        │   └── assign.py
        ├── stage_d/
        │   ├── __init__.py
        │   ├── graph_builder.py
        │   ├── dataset.py
        │   ├── collator.py
        │   ├── model.py
        │   ├── losses.py
        │   ├── trainer.py
        │   └── inference.py
        └── stage_e/
            ├── __init__.py
            ├── templater.py
            ├── dataset.py
            ├── trainer.py
            ├── inference.py
            └── faithfulness.py
```

---

## 实现逻辑

### Stage C：为什么要先做 gating
不是所有 claim 都适合拆成多个 sub-claims。  
如果 claim 很短、没有明显并列结构、比较结构、时间条件或数值条件，就直接保留原 claim，避免 over-splitting。

默认 gating 规则：
- token 长度过短 → 不拆
- clause 数过少 → 不拆
- 没有连接词 / 比较词 / 时间或数值线索 → 不拆

### Stage C：怎么拆
默认提供两种方式：
1. `heuristic`：纯规则切分，最稳、最省资源
2. `hf_local`：用本地 Hugging Face causal LM 按 JSON schema 拆解（可选）

默认先用 `heuristic`，因为它不依赖额外大模型。

### Stage C：怎么把 evidence 分给 sub-claims
对每个 sub-claim，分别从 Stage B 的 support / refute evidence 里按 semantic similarity 选 top-k。  
这样每个子 claim 都有自己的支持与反驳证据，方便图模型聚合。

### Stage D：图是怎么建的
节点类型：
- `claim`
- `subclaim`
- `support_evidence`
- `refute_evidence`

边类型：
- `claim_to_subclaim`
- `subclaim_to_support`
- `subclaim_to_refute`
- `same_report`
- `lexical_overlap`

图模型不会依赖 torch-geometric，而是自己实现一个轻量 relation-aware message passing，方便复现。

### Stage D：为什么还要再判一次 6 类
Stage B 更像“latent evidence scorer + 初判”。  
Stage D 的作用是：
- 在 sub-claim 层面重新组织 support/refute 证据
- 显式建模“同一子事实被哪些句子支持或反驳”
- 再输出更稳定的 6 类 verdict

### Stage E：解释怎么生成
提供两条路：

1. **template / extractive baseline**
   - 几乎不 hallucinate
   - 直接拿 top support / refute 句子拼出 explanation

2. **abstractive explainer**
   - 输入是结构化 claim + sub-claims + selected evidence
   - 输出是自然语言 explanation
   - 支持监督微调（目标是原始数据里的 `explain`）

### Stage E：faithfulness filter
生成后会做一轮过滤：
- 将 explanation 切句
- 每句都要和 selected evidence 有足够高的 lexical / semantic 对齐
- 对齐不够的句子会被删掉或用 extractive fallback 替换

---

## 典型运行顺序

### 1) 生成 graph inputs
```bash
python scripts/build_graph_inputs.py --config configs/build_graph_inputs.yaml
```

### 2) 训练 graph verifier
```bash
python scripts/train_graph_verifier.py --config configs/graph_verifier.yaml
```

### 3) 输出 graph-level verdict
```bash
python scripts/predict_graph_verifier.py --config configs/graph_verifier.yaml --split test
```

### 4) 训练 explanation model（可选）
```bash
python scripts/train_explainer.py --config configs/explainer.yaml
```

### 5) 生成 explanations
```bash
python scripts/generate_explanations.py --config configs/explainer.yaml --split test
```

---

## 你最可能先调的地方

### 如果你更在意 verdict 分数
优先调：
- `build_graph_inputs.yaml` 里的 `top_k_support_per_subclaim`
- `top_k_refute_per_subclaim`
- `graph_verifier.yaml` 里的 `hidden_size`
- `num_layers`
- `lambda_ordinal`

### 如果你更在意 explanation 质量
优先调：
- `explainer.yaml` 里的 `model_name`
- `max_input_length`
- `max_output_length`
- `faithfulness.semantic_threshold`

---

## 一个务实建议
先跑：
- Stage C + D
- Stage E 只用 `template` 模式

等 graph verifier 稳定以后，再训练 abstractive explainer。

## Stage B/C Bad Case 可视化

新增脚本：`stage_cde/scripts/visualize_bad_cases.py`，可用于：

- Stage B 预测错误分布可视化（混淆矩阵、ordinal 距离分布）
- Stage C/D 图判别错误分析（按 subclaim 数量的误差趋势）
- 导出 bad case CSV，方便后续人工排查
- （可选）基于 Stage C 构图结果渲染 bad case 图拓扑（需 `networkx`）

示例命令：

```bash
python stage_cde/scripts/visualize_bad_cases.py \
  --stage_b_predictions stage_ab/outputs/liar-raw/stage_b/stage_b_predictions_test.jsonl \
  --graph_predictions stage_cde/outputs/graph_verifier/test.graph_predictions.jsonl \
  --graph_inputs stage_cde/outputs/graph_inputs/test.graph.jsonl \
  --output_dir stage_cde/outputs/badcase_viz \
  --top_n_badcases 80 \
  --max_graph_cases 8
```

输出产物示例：

- `stage_b_confusion_matrix.png`
- `stage_b_error_distance.png`
- `stage_b_badcases.csv`
- `graph_confusion_matrix.png`
- `graph_error_distance.png`
- `graph_subclaim_vs_error.png`
- `graph_badcases.csv`
- `stage_c_gating_summary.png`
- `badcase_graph_*.png`
