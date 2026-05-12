# run_mmr_lambda_sweep.sh 完整训练逻辑

## 整体架构

`scripts/pipeline/run_mmr_lambda_sweep.sh` 是 Hydra multirun 参数扫描入口，对 **mmr_lambda**（MMR 多样性惩罚系数）进行 11 个值的扫描（默认 0.0~1.0，步长 0.1）。

入口模块：`fact_checking.pipeline.run`（通过 `-m` 开启 Hydra multirun）

```bash
python -m fact_checking.pipeline.run -m \
    experiment=mmr_lambda_sweep \
    "build.retrieval.mmr_lambda=0.0,0.1,...,1.0"
```

实验配置基于 `b0` 基线（生成式 SFT 模式），使用 **Qwen2.5-7B-Instruct** 作为基座模型，**LoRA** 微调。

Pipeline 分三个阶段：**Build → Train → Infer**。

| 配置项 | 值 | 来源 |
|--------|-----|------|
| 嵌入模型 | BGE-base-en-v1.5 | `build.retrieval.embedder_model` |
| 基座模型 | Qwen2.5-7B-Instruct | `train.model_name_or_path` |
| top_k | 16 | `build.retrieval.top_k` |
| α_dense / α_lexical / α_bm25 | 0.70 / 0.20 / 0.10 | `build.retrieval.*` |
| max_length | 2048 | `sft_train.max_length` |
| LoRA r/alpha | 16 / 32 | `sft_train.lora.*` |
| 训练轮数 | 2 | `sft_train.num_train_epochs` |

---

## 1. Build 阶段（候选证据检索 + Prompt 构建）

**入口**：`fact_checking.build.candidates.run_build()`  
**文件**：`src/fact_checking/build/candidates.py`

### 1.1 两阶段架构

#### Phase 1 — Pre-MMR（GPU 嵌入计算）

- 使用 **BGE-base-en-v1.5** 作为嵌入模型（768 维向量）
- 将每个样本的所有句子 + claim 批量编码
- 结果以 pickle 缓存到 `outputs/cache/pre_mmr/<fingerprint>/`，排除 `mmr_lambda` 字段的指纹——因此**跨 λ 值可复用**
- 支持多 GPU 并行：`num_gpus=4`，每个 GPU 处理数据的一个分片
- 批量预处理大小：`prefetch_size=200`（每批 200 个样本一起编码）

#### Phase 2 — Chunk-MMR cache（GPU embedding）

- 从 Pre-MMR cache 读取句子与 claim embedding
- 先按 `build.retrieval.chunking` 组织 evidence chunk candidate
- 对每个 chunk 文本重新运行 BGE embedding
- 结果以 pickle 缓存到 `outputs/cache/chunk_mmr/<fingerprint>/`，排除 `top_k` 与 `mmr_lambda` 字段的指纹

#### Phase 3 — MMR + 候选构建（CPU-only）

- 从 chunk-MMR cache 加载 `chunk_emb` 与 `claim_emb`
- 对每个样本独立运行 chunk-level MMR 选择算法
- 最终输出 JSONL 文件，每行包含候选证据 + 预构建的 prompt-target 对

### 1.2 检索流程（逐样本）

对每个样本，执行以下步骤：

#### (a) 句子级与 chunk 级嵌入

```python
sent_emb = embedder.encode(sent_texts, is_query=False)  # [N, 768]
claim_emb = embedder.encode([claim], is_query=True)[0]   # [768]
chunk_emb = embedder.encode(chunk_texts, is_query=False) # [M, 768]
```

Pre-MMR 句子向量用于 chunking 边界与 learned-lambda 特征；MMR 的候选单位是 evidence chunk，dense/MMR 使用重新编码后的 `chunk_emb`。BGE 模型对 query（claim）自动添加指令前缀：`"Represent this sentence for searching relevant passages: "`。对 passage（chunk 文本）不做特殊处理。

#### (b) 三种分数的计算与融合

**稠密语义分（dense）**：内积相似度（向量已 L2 归一化，等价于余弦相似度）

```
dense_scores = chunk_emb @ claim_emb   # [M]
```

**词汇重叠分（lexical）**：基于内容词（去除英文停用词后）的 F1 分数

```
q_tokens = {非停用词 token 的 Counter}  # 来自 claim
s_tokens = {非停用词 token 的 Counter}  # 来自 evidence chunk
overlap = Σ min(q_tokens[t], s_tokens[t])

precision = overlap / len(s_tokens)
recall    = overlap / len(q_tokens)
F1        = 2 * precision * recall / (precision + recall)
```

**BM25 分数**：简化版 BM25（`retrieval/text_utils.py:56-73`）

```
k1 = 1.2, b = 0.75, avgdl = 18.0

对 query 中每个词 t：
  tf = s_ctr[t]
  idf = log(1 + 1/(1+tf)) + 0.5
  score_t = idf * (tf * (k1+1)) / (tf + k1 * (1-b + b * dl/avgdl))
```

**分数融合**：三者分别做 Min-Max 归一化后加权求和

```python
dense_scaled   = minmax_scale(dense_scores)     # → [0, 1]
lexical_scaled = minmax_scale(lexical_scores)   # → [0, 1]
bm25_scaled    = minmax_scale(bm25_scores)      # → [0, 1]

hybrid_scores = 0.70 * dense_scaled + 0.20 * lexical_scaled + 0.10 * bm25_scaled
```

#### (c) MMR 多样性选择

**文件**：`src/fact_checking/retrieval/mmr.py`

```python
keep_indices = maximal_marginal_relevance(
    query_scores=hybrid_scores,
    sentence_vectors=sent_emb,    # 用于计算句子间余弦相似度
    top_k=16,
    lambda_weight=mmr_lambda,     # 扫描变量
)
```

MMR 算法流程：

1. **首轮**：选择 hybrid_score 最高的句子
2. **迭代**（直至选满 top_k 个）：
   ```
   max_sim_to_selected[i] = max(max_sim_to_selected[i],
                                 cosine_similarity(sent[i], last_selected))

   mmr_score[i] = λ * hybrid_score[i] - (1 - λ) * max_sim_to_selected[i]
   ```
3. 每轮选出 mmr_score 最大的未选句子

**λ 的含义**：

| λ 值 | 行为 |
|------|------|
| 0.0 | 纯多样性：与已选项最不相似的句子得分最高 |
| 0.5 | 相关性与多样性等权 |
| 1.0 | 纯相关性：等价于按 hybrid_score 降序取 top_k |
| 0.70 | b0/b4 默认值 |

#### (d) 去重

按 `canonicalize_sentence(text)`（小写 + 合并空白符）去重。相同文本仅保留 hybrid_score 最高的那条。

#### (e) Chunking

**文件**：`src/fact_checking/build/chunking.py`

默认策略为 `sentence`（直接返回该句子），也支持：

| 策略 | 行为 |
|------|------|
| `sentence` | 返回 sent_idx 对应的单句 |
| `ctx_window` | 返回以 sent_idx 为中心的 2k+1 句窗口 |
| `raw` | 返回整个 report 全文 |
| `semantic` | 将相邻余弦相似度 > θ 的句子合并为一段 |
| `ctx_semantic` | 先按窗口 k 分组，再按窗口平均向量相似度合并 |

### 1.3 Prompt 构建

**文件**：`src/fact_checking/build/candidates.py:471-698`

#### (a) 系统提示词

```
You are a careful fact-checking assistant for LIAR-RAW claims.
Classify claims using only the claim and retrieved evidence supplied by the user.
```

#### (b) 用户内容模板（`output_mode=label_only`, `label_format=letter`）

```
Classify the claim into exactly one LIAR-RAW label.

Labels:
- A (pants-fire): completely false and implausible
- B (false): false based on the available evidence
- C (barely-true): mostly false, with only a small element of truth
- D (half-true): partly true and partly false
- E (mostly-true): mostly true, with minor missing context or caveats
- F (true): accurate based on the available evidence

Rules:
- Use the retrieved evidence as the primary source.
- Do not invent facts not supported by the evidence.
- Respond with exactly one line: Label: <a single letter from A-F>

Claim:
<claim text>

Evidence:
[1] <evidence sentence 1>
[2] <evidence sentence 2>
...
```

Evidence 块格式：
```python
def _format_evidence_block(evidence_texts):
    return "\n".join(f"[{i}] {text}" for i, text in enumerate(evidence_texts, 1))
```

#### (c) 目标（Target）构造

```python
# label_format=letter 时：
target = f"Label: {LABEL_LETTERS[gold_label]}"
# 例: target = "Label: A"
```

#### (d) Chat Template 封装

```python
messages = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": user_content},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)
```

使用 Qwen2.5 的 tokenizer 将 system + user 消息按模型标准格式序列化。

#### (e) 自动截断（`auto_length=true`）

如果 prompt + target 总 token 数超过 `max_length=2048`：
1. 计算 target 的 token 数，得到 prompt token 预算
2. 从尾部（得分最低的 evidence）逐个移除
3. 重新构建 prompt 直到长度满足要求

#### (f) JSONL 输出字段

```json
{
  "event_id": "...",
  "claim": "...",
  "label": "false",
  "explain": "...",
  "candidates": [{"report_id": ..., "sent_idx": ..., "text": ..., "hybrid_score": ...}],
  "prompt": "<完整 chat template prompt>",
  "target": "Label: B",
  "gold_label": "false",
  "gold_id": 1,
  "prompt_token_count": 512,
  "target_token_count": 3,
  "evidence_count": 16,
  "evidence_count_before": 28,
  "was_truncated": false
}
```

---

## 2. Train 阶段（生成式 SFT）

**文件**：`src/sft/trainer.py`

### 2.1 启动方式

Pipeline 通过 `accelerate launch` + DeepSpeed ZeRO-2 启动：

```bash
accelerate launch \
  --num_processes=4 --num_machines=1 --mixed_precision=bf16 \
  --use_deepspeed --deepspeed_config_file configs/deepspeed_zero2.json \
  -m sft.trainer --config <path>/train.resolved.yaml
```

### 2.2 数据集

直接读取 Build 阶段产出的 JSONL 文件：
- `prompt` 字段：预构建的完整 prompt（system + user + evidence）
- `target` 字段：`"Label: A"` 等目标文本

数据集不做额外拼接，直接使用预构建的 prompt-target 对进行标准 Seq2Seq 语言模型训练。验证集和测试集同理。

### 2.3 模型与 LoRA 配置

- 基座模型：**Qwen2.5-7B-Instruct**
- LoRA 参数：
  - `r=16`, `alpha=32`, `dropout=0.05`
  - `bias=none`
  - 目标模块：`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

### 2.4 训练超参数

| 参数 | 值 |
|------|-----|
| per_device_train_batch_size | 4 |
| gradient_accumulation_steps | 2 |
| 有效 batch size | 4×2×4GPU = 32 |
| learning_rate | 1e-5 |
| warmup_ratio | 0.03 |
| lr_scheduler_type | cosine |
| num_train_epochs | 2 |
| weight_decay | 0.0 |
| bf16 | true |
| max_length | 2048 |
| padding | longest |
| use_length_bucket | true |
| use_flash_attention_2 | true |
| max_grad_norm | 1.0 |
| gradient_checkpointing | true |
| eval/save steps | 100 |

最佳模型按 **macro_f1** 选择，保存在 `<run_dir>/train/best/`。判别式训练器（`classifier_trainer.py`）同样通过 `metric_for_best_model="macro_f1"` 按 macro_f1 选择。

### 2.5 损失函数

标准 **Cross-Entropy 损失**（language modeling loss）。仅在 target 部分计算（HuggingFace Trainer 自动将 labels 中 padding token 对应位置设为 -100）。

### 2.6 Logit Adjustment（推理端类先验校正）

`b0.yaml` 配置：

```yaml
sft_train:
  logit_adjust:
    enabled: true
    tau: 1.0
```

**公式**：`adjusted_logit = raw_logit - τ * log(class_prior)`

其中 `class_prior` 从训练集标签分布估计。`τ=1.0` 为完整先验校正（Menon 2021 默认设置），`τ=0` 则退化为纯模型 logit。

---

## 3. Infer 阶段

**文件**：`src/fact_checking/infer/api.py`

### 3.1 推理方式

使用 **vLLM OpenAI-compatible API**：

1. 自动启动 vLLM 服务器，加载 LoRA checkpoint
2. 使用 `OpenAICompletionsClient` 调用 `/completions` 接口（`temperature=0.0`，确定性生成）
3. 推理完成后自动关闭 vLLM 服务器

```python
payload = {
    "model": "fact-checking-sft",
    "prompt": prompt,          # 从 Build JSONL 读取的预构建 prompt
    "max_tokens": 8,           # 只需生成长度很短的 "Label: X"
    "temperature": 0.0,
}
```

vLLM 启动参数：

| 参数 | 值 |
|------|-----|
| tensor_parallel_size | 4 |
| gpu_memory_utilization | 0.90 |
| dtype | auto |
| max_model_len | 2048 |
| enable_lora | true |

### 3.2 标签解析

**文件**：`src/sft/parser.py`

对 vLLM 返回的原始文本，按三级优先级解析：

1. **精确字母行匹配**（最高优先级）：
   ```python
   pattern = re.compile(r"(?mi)^\s*label\s*:\s*([A-F])\b")
   # A → pants-fire, B → false, ..., F → true
   ```

2. **完整标签名行匹配**：正则搜索 `Label: <text>` 行中的标签关键词
   ```
   匹配顺序（从具体到一般，避免 "true" 误匹配到 "barely-true"）：
   pants-fire → barely-true → half-true → mostly-true → false → true
   ```

3. **全文关键词扫描**：在全部输出文本中搜索标签关键词（同上顺序）

解析失败返回 `-1`，计入 `parse_error_rate`。

### 3.3 评估指标

计算 6 分类的：

| 指标 | 公式 |
|------|------|
| Accuracy | `mean(pred == gold)` |
| Macro Precision | 各类别 precision 的算术平均 |
| Macro Recall | 各类别 recall 的算术平均 |
| Macro F1 | 各类别 F1 的算术平均 |
| Parse Error Rate | `mean(pred < 0)` |
| Per-class metrics | 每个标签的 P / R / F1 |
| Confusion Matrix | 7 列（6 类 + parse_error）× 6 行 |

---

## 4. 标签体系

**文件**：`src/fact_checking/data/constants.py`

### 4.1 原始 LIAR-RAW 6 类

| ID | 标签名 | 字母 | 定义 |
|----|--------|------|------|
| 0 | pants-fire | A | completely false and implausible |
| 1 | false | B | false based on the available evidence |
| 2 | barely-true | C | mostly false, with only a small element of truth |
| 3 | half-true | D | partly true and partly false |
| 4 | mostly-true | E | mostly true, with minor missing context or caveats |
| 5 | true | F | accurate based on the available evidence |

### 4.2 标签归一化

**文件**：`src/sft/data/labels.py`

```python
def normalize_gold_label(row):
    gold_label = str(row.get("label", "")).strip().lower()
    if gold_label not in LABEL2ID:
        return ""    # 无效标签返回空字符串，对应数据会被过滤
    return gold_label
```

### 4.3 3 分类折叠（b4_3class 实验）

```python
LABELS_3CLASS = ["false", "mixed", "true"]
LABEL_MAP_6TO3 = {0: 0, 1: 0,   # pants-fire, false → false
                   2: 1, 3: 1,   # barely-true, half-true → mixed
                   4: 2, 5: 2}   # mostly-true, true → true
```

---

## 5. 完整执行流程（一张图）

```
┌──────────────────────────────────────────────────────────────┐
│ Phase 1: Pre-MMR (GPU Embedding, 跨 λ 共享缓存)               │
├──────────────────────────────────────────────────────────────┤
│ LIAR-RAW JSON → BGE-base-en-v1.5 编码                         │
│   claim → claim_emb [D]                                       │
│   每个 report 的句子 → sent_emb [N, D]                         │
│   → pickle 缓存 (outputs/cache/pre_mmr/<fp>/)                  │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│ Phase 2: Chunk-MMR cache (GPU Embedding, 跨 top_k/λ 共享缓存) │
├──────────────────────────────────────────────────────────────┤
│ 对每个样本:                                                    │
│   1. 按 chunking strategy 生成 evidence chunk candidate        │
│   2. 重新编码 chunk_texts → chunk_emb [M, D]                   │
│   → pickle 缓存 (outputs/cache/chunk_mmr/<fp>/)                │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│ Phase 3: MMR + Candidate (CPU-only, 每个 λ 独立执行)          │
├──────────────────────────────────────────────────────────────┤
│ 对每个样本:                                                    │
│   1. dense_scores = chunk_emb @ claim_emb                     │
│   2. lexical_scores = F1(content_tokens(claim), chunk)        │
│   3. bm25_scores = simplified BM25(claim, chunk)              │
│   4. hybrid = 0.7*minmax(dense) + 0.2*minmax(lexical)         │
│                              + 0.1*minmax(bm25)               │
│   5. MMR select(hybrid, chunk_emb, top_k=16, λ=扫描值)        │
│   6. 去重 + 排序                                              │
│   7. 构建 Chat Prompt (Qwen2.5 template)                      │
│   8. 自动截断至 max_length=2048                               │
│   9. target = "Label: <A-F>"                                  │
│   → JSONL 输出                                                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│ Train (生成式 SFT)                                             │
├──────────────────────────────────────────────────────────────┤
│ accelerate + DeepSpeed ZeRO-2 × 4 GPU                        │
│ Qwen2.5-7B-Instruct + LoRA (r=16, α=32)                      │
│ CrossEntropyLoss on target tokens ("Label: X")                │
│ 2 epochs, lr=1e-5, cosine scheduler                           │
│ best checkpoint → <run_dir>/train/best/                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│ Infer (vLLM API)                                               │
├──────────────────────────────────────────────────────────────┤
│ vLLM Server (TP=4, LoRA adapter)                              │
│ /completions API: max_tokens=8, temperature=0.0               │
│ Parse output → Label Letter (A-F) → Label ID (0-5)           │
│ Evaluate: Accuracy, Macro F1, Confusion Matrix                │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. MMR Lambda 扫描总结

```
λ=0.0: 纯多样性 — MMR 选择与已选项最不相似的句子    ─┐
     :                                                ├─ 11 个 multirun
λ=1.0: 纯相关性 — 等价于按 hybrid_score 降序取 top_k  ─┘

默认值:  0.70 (b0/b4 基线的文档默认值)
扫描值:  0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
```

关键优化：Build 阶段的 **Pre-MMR 嵌入缓存**排除 `mmr_lambda` 字段的指纹，因此 11 个 λ 值共享同一份 GPU 嵌入结果。只有 Phase 2（MMR 选择 + Prompt 构建）因 λ 不同而独立执行，大幅减少重复计算。

### 相关文件索引

| 文件 | 作用 |
|------|------|
| `scripts/pipeline/run_mmr_lambda_sweep.sh` | 扫描启动脚本 |
| `configs/experiment/mmr_lambda_sweep.yaml` | 基于 b0 的扫描实验配置 |
| `configs/experiment/b0.yaml` | b0 基线配置（生成式 SFT） |
| `configs/experiment/b4_mmr_lambda_sweep.yaml` | 基于 b4（判别式分类器）的扫描配置 |
| `configs/pipeline/default.yaml` | Pipeline 默认配置 |
| `configs/train/default.yaml` | 训练默认配置 |
| `configs/build/default.yaml` | Build 默认配置 |
| `src/fact_checking/pipeline/run.py` | Hydra 入口（`@hydra.main`） |
| `src/fact_checking/pipeline/runner.py` | Pipeline 编排（build→train→infer） |
| `src/fact_checking/build/candidates.py` | Build 主逻辑（检索、MMR、Prompt 构建） |
| `src/fact_checking/retrieval/mmr.py` | MMR 算法实现 |
| `src/fact_checking/retrieval/embedder.py` | BGE 文本嵌入封装 |
| `src/fact_checking/retrieval/text_utils.py` | 词汇分、BM25 分计算 |
| `src/fact_checking/build/chunking.py` | 句子切块策略 |
| `src/fact_checking/data/constants.py` | 标签常量定义 |
| `src/fact_checking/data/io.py` | 数据加载（JSON/JSONL） |
| `src/fact_checking/data/types.py` | 数据类型定义 |
| `src/sft/data/labels.py` | 标签归一化 |
| `src/sft/trainer.py` | 生成式 SFT 训练器 |
| `src/sft/dataset/` | 数据集/数据加载器实现 |
| `src/fact_checking/infer/api.py` | vLLM API 推理 |
| `src/sft/infer_common.py` | 推理上下文构建 |
| `src/sft/parser.py` | 标签解析（输出→label ID） |
| `src/sft/metrics.py` | 分类指标计算 |
