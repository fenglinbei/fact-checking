# Sequential Pointer Selector 架构解读

日期: 2026-05-21

## 问题域

两个架构层面问题的解答：

1. 为什么叫"Stage2 sentence-level top15 candidate pool"——Stage2 的含义是什么？
2. `SequentialPointerSelectorModel` 中 Item Projection 的作用和维度变化。

## Stage1 与 Stage2 的分工

整个证据检索管线是两阶段的，加 Stage2 前缀是因为该候选池是**第二阶段 selector 的输入**，而非第一阶段粗筛产物。

### Stage1 — 粗粒度的 chunk 级召回

在 `src/fact_checking/build/candidates.py` 中实现：

- 对原始报告文本做分块（chunk），每个 chunk 可以是多句话或单句话
- 计算每个 chunk 与 claim 的 hybrid score：dense embedding + lexical overlap + BM25
- 用 MMR（最大边际相关性）做多样性重排，去冗余
- 输出：粗筛后的 chunk 候选集，数量较大（如 top 24），覆盖面广但精度有限

### Stage2 — 细粒度的 sentence 级精选

在 Stage1 的 chunk 候选基础上，将粒度细化到句子级别（`chunking.strategy = sentence`，每个 chunk 就是一句话），并固定为大小 15 的候选池：

- `DEFAULT_CANDIDATE_POOL_SIZE = 15`，定义在 `src/fact_checking/selectors/stage2_oracle.py:10`
- 训练 oracle 标签：每个 candidate 的 margin（加入该 evidence 后 verifier 对正确 label 的预测概率提升量）来自 oracle search（`search_objective: "margin"`）
- 由训练好的 neural selector（pointwise logistic regression → cross-encoder → listwise transformer → sequential pointer selector）从 15 候选中选出最终 top 5

### Pipeline 视角

```
原始报告文本
  → Stage1 chunk 粗筛 (hybrid score + MMR, top K chunks)
  → Stage2 sentence 候选池 (固定 15 个句子, 带 oracle margin 标签)
  → Stage2 neural selector (选 top 5)
  → 最终 prompt evidence
```

数据产物因此天然带着"stage2"标记——文件名如 `stage2_margin_train_sharded`、`stage2_margin_val_*`、类名如 `Stage2OracleExample`。

## Item Projection 的作用与维度变化

### 在 forward 中的位置

```
DeBERTa-v3-base Encoder
  → [N_candidates, 768]  (pool_pair_embeddings, mean pooling)
  → Item Projection
  → [N_candidates, 256]
  → pad_flat_items
  → [batch_size, max_candidates(15), 256]
  → TransformerEncoder (Set Encoder)
  → [batch_size, max_candidates, 256]  (H_i_ctx)
```

### 维度变化

```python
# sequential.py:225-229
encoder_hidden = getattr(self.encoder.config, "hidden_size", 768)  # 768
self.item_projection = nn.Sequential(
    nn.Linear(encoder_hidden, self.hidden_size),  # 768 → 256
    nn.GELU(),
    nn.Dropout(self.dropout),
)
```

即：**768 → 256**。

### 两个作用

**1. 降维**

后端 Set Encoder 是 2 层 `TransformerEncoder`，`d_model=256`。如果直接喂 768 维，计算量和参数量会膨胀。N ≤ 15 虽然不大，但每个 candidate 都要参与 self-attention（pairwise 交互），降维后更高效。

**2. 语义适配（task-specific bottleneck）**

DeBERTa 的 pooler output 编码的是预训练任务的通用语义。这里的下游任务是从 15 个候选中判断 evidence utility——即哪个句子能帮 verifier 更好判断 claim 真伪。projection 是可学习的线性变换，允许模型把 768 维中与"证据选择"相关的方向保留，与"自然语言理解"相关但与此任务无关的方向压缩或丢弃。

可以理解为：encoder 负责"理解 claim-candidate 语义关系"，projection 负责"把理解结果翻译成 selector 后端需要的格式"。冻结 encoder（`--freeze-pair-encoder`）时，projection 是唯一的语义适配通道。

## H_i_ctx 完整变换链路

从一个原始 claim 和 N 条 sentence evidence 开始，追踪到最终的 `H_i_ctx`。代码入口在 `forward_sequential_examples`（`sequential.py:367`），最终在 `SequentialPointerSelectorModel.forward`（`sequential.py:242`）中完成。

以 batch_size=1 为例，设该 claim 下有 15 个 candidate evidence sentence。

---

### Step 0 — 原始数据结构

```python
example.claim         # str:     "Barack Obama was born in Kenya."
example.candidates    # list:    [{"text": "Obama was born on Aug 4, 1961 in Honolulu.", ...}, ...]  × 15
```

每个 candidate 是一个 dict，核心字段为 `text`（evidence 句子文本）、`hybrid_score`、`candidate_idx` 等。

---

### Step 1 — 提取 candidate text

代码：`candidate_text()`（`stage2_oracle.py:150`）

```python
# 对每个 candidate
text = str(candidate.get("text") or "").strip()
```

15 个 candidate → 15 个 str。

---

### Step 2 — 构造 claim-candidate pair 文本并 tokenize

代码：`forward_sequential_groups()`（`sequential.py:392-409`），调用 `tokenize_claim_candidate_pairs()`（`cross_encoder.py:83`）

```python
# 15 个 candidate → 15 对 (claim, text)
claims  = [claim_str] * 15   # 同一个 claim 复制 15 次
texts   = [candidate_text(c) for c in candidates]  # 15 个 evidence 句子

# 每对格式化为：
left   = "Claim: Barack Obama was born in Kenya."
right  = "Evidence: Obama was born on Aug 4, 1961 in Honolulu."

# DeBERTa tokenizer 将 left/right 作为 pair 编码：
#   [CLS] Claim: ... [SEP] Evidence: ... [SEP]
#   token_type_ids: 0=claim_part, 1=evidence_part
tokenizer(left, right, padding=True, truncation=True, max_length=384)

# 输出：
# input_ids:      [15, L]     # L ≤ 384, 同一 batch 内 pad 到最长
# attention_mask: [15, L]
# token_type_ids: [15, L]
```

**形状变化**：`15 个 str pair` → `{"input_ids": [15, L], "attention_mask": [15, L], ...}`

---

### Step 3 — DeBERTa Encoder 前向

代码：`SequentialPointerSelectorModel.forward()`（`sequential.py:248`）

```python
outputs = self.encoder(**encoded_inputs)
# outputs.last_hidden_state: [15, L, 768]
# outputs.pooler_output:     [15, 768]  (如果可用)
```

每个 token 输出 768 维 hidden state。

**形状变化**：`[15, L] token ids` → `[15, L, 768] token embeddings`

---

### Step 4 — Mean Pooling 得到 pair embeddings

代码：`pool_pair_embeddings()`（`sequential.py:774`）

```python
# 优先用 pooler_output（经 tanh 的全句表示）
if outputs.pooler_output is not None:
    return outputs.pooler_output  # [15, 768]

# 否则对 last_hidden_state 做 mean pooling（attention_mask 排除 padding）
hidden = outputs.last_hidden_state            # [15, L, 768]
mask = attention_mask.to(hidden.dtype)        # [15, L] → [15, L, 1]
summed = (hidden * mask).sum(dim=1)           # [15, 768]
denom = mask.sum(dim=1).clamp_min(1.0)        # [15, 1]
return summed / denom                         # [15, 768]
```

DeBERTa-v3-base 通常有 `pooler_output`，但对于某些 DeBERTa 变体没有，所以 fallback 到 mean pooling。

**形状变化**：`[15, L, 768]` → `[15, 768]`

此时每个 candidate 被压缩为一个 768 维向量 `h_i_pair`，编码了"该证据与 claim 的配对语义关系"。

---

### Step 5 — Item Projection 降维

代码：`sequential.py:249`

```python
item_embeddings = self.item_projection(pair_embeddings)
# Linear(768→256) → GELU → Dropout
# [15, 768] → [15, 256]
```

**形状变化**：`[15, 768]` → `[15, 256]`

---

### Step 6 — pad_flat_items 打包为 padded batch

代码：`pad_flat_items()`（`listwise.py:603`）

```python
padded_items, mask = pad_flat_items(item_embeddings, group_sizes=[15])

# pad_flat_items 内部逻辑：
flat:        [15, 256]              # 1 个 claim 的 15 个 candidate 的投影向量
group_sizes: [15]                   # 表示该 batch 只有一个组，组内 15 个
max_group:   15                     # 最大组大小

padded:      [1, 15, 256]           # 行 0 放 15 个向量
mask:        [1, 15]  (全 True)     # 前 15 个位置有效
```

多 claim batch 的情况（如 batch_size=3，group_sizes=[15, 12, 14]）：

```
flat = [item_0_0, ..., item_0_14, item_1_0, ..., item_1_11, item_2_0, ..., item_2_13]
       |---- claim0  ----|  |--- claim1 ---|  |------- claim2 -------|

padded = [
  [item_0_0, ..., item_0_14],           # 15 个全有效
  [item_1_0, ..., item_1_11, 0...0],    # 后 3 个是 zero-padding
  [item_2_0, ..., item_2_13, 0],        # 最后 1 个是 zero-padding
]

mask = [
  [T,T,T,T,T,T,T,T,T,T,T,T,T,T,T],
  [T,T,T,T,T,T,T,T,T,T,T,T,F,F,F],
  [T,T,T,T,T,T,T,T,T,T,T,T,T,T,F],
]
```

**形状变化**：`[total_candidates, 256]` → `[B, max_candidates, 256]` + `[B, max_candidates]` mask

---

### Step 7 — Set Encoder (TransformerEncoder) 上下文编码

代码：`sequential.py:252`

```python
context = self.set_encoder(
    padded_items,                          # [B, max_candidates, 256]
    src_key_padding_mask=~mask,            # ~mask: True 的位置被屏蔽（padding 位置）
)
# context: [B, max_candidates, 256]
```

这是 2 层 `TransformerEncoder`，batch_first=True，`d_model=256`，`nhead=4`，`dim_feedforward=1024`。

此时发生的事情：
- 每个 candidate 通过 self-attention 看到同一 claim 下的**其他所有 candidate**
- `src_key_padding_mask=~mask` 确保 padding 位置不参与 attention（既不当 query 也不当 key）
- 输出 `H_i_ctx` 是 **set-contextualized** 的表示：同一个 candidate 在与其他 candidate 交互后，编码了互补、竞争、冗余关系

---

### Step 8 — LayerNorm 稳定输出

代码：`sequential.py:253`

```python
context = self.output_norm(context)
# [B, max_candidates, 256]
```

形状不变，归一化后作为最终的 `H_i_ctx`。

---

### 结果结构

```python
SequentialForwardOutput(
    context_embeddings=context,   # [B, max_candidates, 256]  ← H_i_ctx
    candidate_mask=mask,          # [B, max_candidates]       ← Bool
)
```

---

### 一步速览

```
原始输入:
  claim: "Barack Obama was born in Kenya."
  candidates: [{"text": "Obama was born... Honolulu.", ...}, ...] × 15

→ Step 1: candidate_text()          → 15 个 str
→ Step 2: tokenize pairs            → {"input_ids": [15,L], "attention_mask": [15,L], ...}
           格式: "Claim: <claim>" + "Evidence: <text>"
→ Step 3: DeBERTa Encoder           → .last_hidden_state [15, L, 768]
→ Step 4: pool_pair_embeddings()    → h_i_pair          [15, 768]
           mean pooling over tokens (attention_mask 排除 padding)
→ Step 5: Item Projection           → [15, 256]
           Linear(768→256) → GELU → Dropout
→ Step 6: pad_flat_items()          → [1, 15, 256] + [1, 15] mask
           将同一 claim 的 15 个向量打包为 padded batch
→ Step 7: TransformerEncoder        → H_i_ctx [1, 15, 256]
           2 层 self-attention，candidate 间交互建模互补/竞争/冗余
→ Step 8: LayerNorm                 → H_i_ctx [1, 15, 256]  (归一化)
```

## Set Encoder 容量是否是瓶颈

### 问题

Set Encoder（2 层 TransformerEncoder，d_model=256，nhead=4，dim_feedforward=1024）因深度/宽度不够，是否构成候选间交互建模的瓶颈？

### 当前配置

```python
# sequential.py:230-238
TransformerEncoderLayer(
    d_model=256,
    nhead=4,                  # 每 head 64 维
    dim_feedforward=1024,     # 4× d_model
    dropout=0.1,
    activation="gelu",
    batch_first=True,
)
self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
```

约 1.6M 参数。输入 [B, 15, 256]，输出同样形状的 H_i_ctx。

### 现有证据不太支持"Set Encoder 是首要瓶颈"

`deberta_sequential_deep` best checkpoint（step 2500）的 step-wise 诊断：

| step | accuracy | entropy |
|-----:|---------:|--------:|
| 0 | 0.1664 | 2.5699 |
| 1 | 0.0921 | 2.5844 |
| 2 | 0.0797 | 2.5020 |
| 3 | 0.0714 | 2.4294 |
| 4 | 0.0794 | 2.3414 |

`first_wrong_step_mean = 0.1954`

**线索 1：step0 不涉及真实 prefix，但已经很低。** Step0 使用 `learned_start_prefix`（无历史 prefix），此时模型判断几乎完全依赖 H_i_ctx 本身。熵高达 2.57（15 类均匀分布熵 = ln(15) ≈ 2.71），说明模型在第一步就没有形成强烈的候选偏好。但如果 Set Encoder 容量是主因，step0 的差应归因于 H_i_ctx 质量——但 step0 的 prefix interaction 特征全退化为和 `start_prefix` 的交互，这涉及的是 `start_prefix` 的学习质量，而非 Set Encoder。

**线索 2：step0→step1 骤降，不是持平。** 如果 Set Encoder 编码不足是瓶颈，step0 差而 step1+ 应持平或略好（有了真实 prefix 信号补充）。实际 step1 accuracy 跌到 0.0921——有了真实 prefix 后更差了。这说明 prefix-conditioned scoring 在当前特征空间里没有有效学到"下一条最该选什么"，而非 set 编码本身。

**线索 3：Step3 listwise 用了类似 set encoder 结构。** 其消融实验表明去掉 rank/index prior 后 recall@5 仅从 0.3732→0.3826。Set Encoder 本身（无论深度）并非区分各变体的关键因素——关键因素一直是"有没有 position/rank shortcut"。

### 更深层的瓶颈：Pair Encoder 的证据效用判别能力

回看信息流：768 维 DeBERTa pair embedding → 256 维 projection → Set Encoder。Set Encoder 做候选间交互，但底层每个 candidate 的语义质量已经被 pair encoder + 256 维 bottleneck 决定了。如果这 256 维向量本身无法充分编码"这条 evidence 对核查这个 claim 有多大用"，那再多层 self-attention 也只是在低质量表示上做排列组合。

这也是 Plan 中建议 Step4.1 优先加 claim aspect coverage、stance utility semantics 等 targeted features 的原因——从源头丰富 candidate 表示，让 Set Encoder 有更丰富的材料来建模候选间关系，而非在现有贫瘠输入上堆更多层。

### 诊断实验建议

低成本验证：加宽加深 Set Encoder：

```bash
--list-hidden-size 512 --list-layers 4 --list-heads 8
```

预期判断：
- 若 recall@5 和 step0 accuracy **明显上升**（step0 从 0.166→0.22+），说明 Set Encoder 容量确实是瓶颈
- 若指标**基本不变**（大概率），说明瓶颈不在 Set Encoder，应继续走 Plan 中的 Step4.1 B/C（first-step weighting、aspect coverage），从源头改善 candidate 表示质量

### 结论

Set Encoder 容量有限理论上可能构成瓶颈，但现有 step-wise 诊断更倾向于指向 pair encoder 的证据效用表示不足和 prefix-conditioned scoring 学习困难，而非 set-level 交互建模不够深。优先从源头（candidate 表示质量）改善，而非在 Set Encoder 上堆参数。

### Tokenization 截断诊断结果 (2026-05-21)

对训练集（`stage2_margin_train_sharded`, 10065 样本, 142309 个 candidate pair）用 DeBERTa-v3-base tokenizer 做完整统计：

| Bucket (max pair len) | 样本数 | 占比 | 截断样本 | 截断率 |
|---|---|---|---|---|
| >640 | 8 | 0.1% | 8 | 100% |
| 513-640 | 13 | 0.1% | 13 | 100% |
| 385-512 | 59 | 0.6% | 59 | 100% |
| 257-384 | 269 | 2.7% | 0 | 0% |
| ≤256 | 9716 | 96.5% | 0 | 0% |

**至少有一个 candidate 被截断的样本: 80 (0.8%)**
**未被截断的样本: 9985 (99.2%)**

所有 pair 的 token 长度分布：
- 均值: 65.7，中位数: 59
- P50: 59, P75: 76, P90: 98, P95: 117, P99: 175
- 最小: 15, 最大: 859

**结论：tokenization 截断不是瓶颈。** 99.2% 的样本完全不被截断，pair 长度中位数仅 59 tokens，远低于 384。evidence 句子普遍很短，"Claim: ... Evidence: ... " 模板开销也小。即使在最大长度的样本中（859 tokens），也只有 1 个 candidate 被截断。

## 架构层面的潜在短板

从整个 Sequential Pointer Selector 的 tokenization → encoding → projection → set context → prefix → interaction → scoring → loss → decode 链路中，每个环节都存在可讨论的弱点。

### 1. Tokenization 侧的截断损失

**现状**：`max_length=384`（`cross_encoder.py:83`），tokenizer 对输入 pair 做 `truncation=True`。

```
"Claim: <claim_text>" + "Evidence: <evidence_text>"
```

DeBERTa tokenizer 将 claim 和 evidence 拼接为 `[CLS] Claim: ... [SEP] Evidence: ... [SEP]`，超长部分从右侧截断。这意味着**长 evidence sentence 的尾部信息被丢弃**。对于包含关键数字、日期、人名的长句，被截掉的可能是最关键的核查信息。384 对于 pair encoding 不算特别短，但 fact-checking evidence 常有 100+ token 的长句，claim 本身也可能有 50+ token，留给 evidence 的空间被压缩。

**诊断方式**：统计训练集中被截断的 sample 比例，按 `(claim_tokens + evidence_tokens > 384)` 分桶对比各桶的 step0 accuracy。

---

### 2. Mean Pooling 的信息坍缩

**现状**：`pool_pair_embeddings()`（`sequential.py:774`）将 `[L, 768]` 的 token 序列压缩为单个 `[768]` 向量。

```python
summed = (hidden * mask).sum(dim=1)    # 所有 token 求和
denom = mask.sum(dim=1).clamp_min(1.0) # token 数量
return summed / denom                   # 简单平均
```

这个过程**丢弃了所有 token 级的细粒度对齐信息**。比如 claim 中的"2016"与 evidence 中的"2015"在 token 级别可能形成高 attention 的"几乎匹配但实际矛盾"信号，但 mean pooling 后这种局部冲突被平均掉了。

对比方案：用 DeBERTa 的 `pooler_output`（基于 [CLS] + tanh 的全句表示）已经是对比任务优化的，但 mean pooling fallback 在 pooler_output 为空时会丢失更多信息。如果当前使用的是 pooler_output，那 bottleneck 在预训练任务的 pooler 是否适配 evidence utility 判断；如果 fallback 到了 mean pooling，则信息损失更严重。

---

### 3. Item Projection 的单一线性压缩

**现状**：`Linear(768 → 256) → GELU → Dropout`，一层线性变换。

这是整个架构中最窄的 bottleneck——**768 维 DeBERTa 表示经由单层线性层压缩到 256 维，参数量仅 768×256+256 ≈ 197K**。没有中间层、没有残差、没有多头拆分。对于需要同时编码"语义相关性 + 事实一致性 + 数字匹配 + 时间对齐 + 实体对应 + 辩护方向"的多维 evidence utility 信号，一层线性投影能否在 256 维里有效解耦这些信号是存疑的。

**已实现修改 (2026-05-21)**：替换为 2 层 MLP（768 → 512 → 256）+ 残差连接，参数量约 660K。

代码变更（`sequential.py`）：

```python
# Core model __init__ — 替换 self.item_projection，新增 self.proj_residual
self.item_projection = nn.Sequential(
    nn.Linear(encoder_hidden, self.hidden_size * 2),   # 768 → 512
    nn.GELU(),
    nn.Dropout(self.dropout),
    nn.Linear(self.hidden_size * 2, self.hidden_size),  # 512 → 256
)
self.proj_residual = nn.Linear(encoder_hidden, self.hidden_size)  # 768 → 256

# forward() — 使用残差连接
projected = self.item_projection(pair_embeddings)
residual = self.proj_residual(pair_embeddings)
item_embeddings = F.gelu(projected + residual)
```

`selector_head_state_dict` 和 `load_selector_head_state_dict` 同步新增 `proj_residual`，后者通过 `"proj_residual" in payload` 做向后兼容。`model_config` 新增 `"proj_num_layers": 2, "proj_residual": True`。

**实验结果 (2026-05-21)**：与 claim_start 修改打包为 `deberta_sequential_deep_proj2_v2`（mask_weight=0.5），训练 6000 步。全指标倒退，recall@5 从 0.3852→0.3717（-1.35pp），top1_match 从 0.1664→0.1523。详见 [实验对比](#proj2-v2-实验对比-2026-05-21)。

---

### 4. Set Encoder 深度与宽度不足

**现状**：2 层 TransformerEncoder，d_model=256，nhead=4（每 head 64 维），dim_feedforward=1024，约 1.6M 参数。输入 [B, 15, 256]，输出同样形状的 H_i_ctx。

具体配置见 [`Set Encoder 容量是否是瓶颈`](#set-encoder-容量是否是瓶颈) 中的分析。潜在不足：

- **每 head 仅 64 维**，用于捕捉 15 个 candidate 间 105 对 pairwise 关系（互补/竞争/冗余/信息增量），表达能力不宽裕
- **仅 2 层**：虽说不像 CNN 需要堆深才能扩大感受野（self-attention 一层就能看到所有位置），但额外层能构造更高阶组合特征，如"candidate A 与 B 都覆盖了同一实体，但 A 的时间信息跟 claim 更接近"这种多跳比较
- **256 维本身**就是从 768 维 DeBERTa 输出经由 Item Projection 压缩而来，此维度同时约束了 set-level 交互的表示空间

但现有 step-wise 诊断不太支持 Set Encoder 是首要瓶颈：step0 accuracy 已经很低（0.1664），且 step0→step1 骤降而非持平，问题更可能出在 prefix-conditioned scoring 和 pair encoder 的证据效用表示上。详见[该节](#set-encoder-容量是否是瓶颈)的完整分析。

诊断实验：`--list-hidden-size 512 --list-layers 4 --list-heads 8`。

---

### 5. `start_prefix` 与 claim 无关

**现状**：`start_prefix` 是一个可学习的 `nn.Parameter`（`sequential.py:72`），训练后对所有 claim 都一样。

Step0 时 prefix 就是这个固定向量，不包含**任何当前 claim 特定信息**。模型在第一步选择的唯一 claim-specific 依据来自 `H_i_ctx` 中的 candidate 表示，而 prefix 侧只提供一个"空白"对照。这意味着 step0 的选择全权依赖于 `H_i_ctx` 的绝对质量——模型需要能从 15 个 candidate 中挑出最好的，但没有任何来自 claim 的"你应该关注什么"的提示。

**已实现修改 (2026-05-21)**：用当前 claim 下所有候选 pair embedding 的均值作为 step0 的 prefix（`claim_start`）。均值的语义逻辑：candidate 间共同编码的 claim 相关信息在均值中得以保留，而各 candidate 特有噪声被抵消。有真实 prefix 后（t>0）仍使用原有均值池化逻辑，`start_prefix` 保留作为 fallback。

代码变更（`sequential.py`）：

```python
# SequentialForwardOutput 新增字段
@dataclass(frozen=True)
class SequentialForwardOutput:
    context_embeddings: torch.Tensor
    candidate_mask: torch.Tensor
    claim_start: torch.Tensor | None = None   # NEW: [B, 256]

# forward() 末尾新增 claim_start 计算
claim_start = _build_claim_start(item_embeddings, group_sizes)
return SequentialForwardOutput(
    context_embeddings=context, candidate_mask=mask, claim_start=claim_start,
)

# DeepInteractionPointerHead.prefix_representation() — 多步逻辑
def prefix_representation(self, context_embeddings, selected_mask, *, claim_start=None):
    has_prefix = selected_mask.any(dim=1)
    if not has_prefix.any():
        if claim_start is not None:
            return claim_start.to(dtype=...)         # step0: claim-grounded
        return self.start_prefix.expand(...)          # fallback: global vector
    # t > 0: mean pool selected H_j_ctx (原有逻辑)
    pooled = (context_embeddings * selected).sum(dim=1) / counts
    fallback = claim_start if claim_start is not None else start.expand_as(pooled)
    return torch.where(has_prefix.unsqueeze(-1), pooled, fallback)
```

辅助函数 `_build_claim_start(item_embeddings, group_sizes)` 按组内均值构建 `[B, 256]`。

`score_step`、`teacher_forcing_logits`、`greedy_decode` 均新增 `claim_start` 参数并透传至 `prefix_representation`。`teacher_forcing_sequential_logits` 和 `predict_sequential_groups` 从 `output.claim_start` 提取并传递。

`model_config` 新增 `"claim_start": "candidate_pool_mean"`。

**实验结果 (2026-05-21)**：与 proj2 修改打包为 `deberta_sequential_deep_proj2_v2`（mask_weight=0.5），训练 6000 步。step0 accuracy 不升反降（0.1664→0.1523），说明 candidate pool mean 作为 claim 参照可能引入噪声（坏候选稀释有用信号）。详见 [实验对比](#proj2-v2-实验对比-2026-05-21)。

#### proj2_v2 实验对比 (2026-05-21)

实验 `OUTPUT_DIR=deberta_sequential_deep_proj2_v2 MASK_LOSS_WEIGHT=0.5`，同时包含 proj2 残差投影 + claim_start + mask=0.5。训练完成 6000 步，24 个 validation 点，未触发早停（oracle_rank_ndcg@5 从 0.2569 缓慢爬升至 0.3297），最佳 checkpoint 在最终步。

对比 deep baseline（mask=0, best step=2500）、mask05（best step=3250）：

| 指标 | deep (mask=0) | mask05 | proj2_v2 | vs baseline |
|---|---|---|---|---|
| recall@5 | 0.3852 | 0.3662 | **0.3717** | -1.35 pp |
| jaccard@5 | 0.2615 | 0.2472 | **0.2509** | -1.06 pp |
| top1_match | 0.1664 | 0.1499 | **0.1523** | -1.41 pp |
| prefix_match@3 | 0.0055 | 0.0039 | **0.0024** | — |
| prefix_match@5 | 0.0000 | 0.0000 | **0.0000** | — |
| oracle_rank_ndcg@5 | 0.3306 | 0.3270 | **0.3297** | -0.09 pp |
| pairwise_order_acc@5 | 0.5871 | 0.5764 | **0.5776** | -0.95 pp |
| ordered_exact_match@5 | 0.0078 | 0.0063 | **0.0047** | — |
| ordered_hit@5 | 0.1020 | 0.0920 | **0.0929** | — |

Step-wise 诊断：

| step | deep acc | deep ent | mask05 acc | mask05 ent | proj2_v2 acc | proj2_v2 ent |
|-----:|---------:|---------:|-----------:|-----------:|-------------:|-------------:|
| 0 | 0.1664 | 2.5699 | 0.1499 | 2.5890 | **0.1523** | 2.5323 |
| 1 | 0.0921 | 2.5844 | 0.0740 | 2.5552 | **0.0748** | 2.3606 |
| 2 | 0.0797 | 2.5020 | 0.0678 | 2.4828 | **0.0710** | 2.2640 |
| 3 | 0.0714 | 2.4294 | 0.0714 | 2.4109 | **0.0674** | 2.1826 |
| 4 | 0.0794 | 2.3414 | 0.0770 | 2.3192 | **0.0826** | 2.0965 |
| fws_mean | 0.1954 | — | 0.1688 | — | **0.1735** | — |

**结论**：proj2_v2 在所有指标上均未超越 deep baseline。

三个关键发现：

1. **全指标倒退**。两个修改打包后无任何改善。尤其 step0 accuracy（claim_start 最应改善的指标）反而从 0.1664 跌到 0.1523。

2. **熵降但方向错**。proj2_v2 在所有 step 上 entropy 都更低（如 step0: 2.53 vs 2.57），说明模型更"自信"，但 accuracy 同时下降——自信在了错误的方向。新增特征可能让某些 spurious pattern 更容易被捕捉。

3. **学习更慢**。oracle_rank_ndcg@5 从 step 500 的 0.2569 缓慢爬到 step 6000 的 0.3297，而 baseline 在 step 2500 就达到了 0.3306。新增 460K 参数没有加速收敛，反而拖慢。

**归因推断**（打包实验无法分离贡献，以下为推测）：

- **claim_start 的 candidate pool mean 可能有噪声**：15 个候选（含坏候选）等权平均，坏候选的噪声直接进入 prefix。更干净的做法是用 claim-only embedding
- **proj2 深层投影可能需要更多训练或调学习率**：参数量翻 3 倍，但 LR 不变，最终步才接近 baseline
- **`start_prefix` 的 learnable bias 设计可能不当**：`claim_start + start` 假设 start_prefix 对所有 claim 施加同一偏置，但各 claim 可能需要不同的方向

**后续建议**：分离消融——proj2 单独、claim_start（改为 claim-only embedding）单独，分别归因。

---

### 6. Prefix 表示的均值池化

**现状**：`prefix_representation()`（`sequential.py:83`）对已选 candidate 的 `H_j_ctx` 做等权平均池化。

```python
pooled = (context_embeddings * selected).sum(dim=1) / counts
```

这意味着模型把"第一条选中的 evidence"和"第四条选中的 evidence"在 prefix 中等同对待。但实际上早期选择的 evidence 通常更重要（margin 更高），模型应该被允许在 prefix 中对其加权。此外，简单平均会随着 prefix 增长而稀释早期 evidence 的信号——选了 4 条后，每条只占 prefix 的 1/4 权重，最早那条可能最重要的证据被淹没。

---

### 7. 交互特征的设计假设

**现状**：6 个手工交互特征（`score_step`，`sequential.py:103-110`）：

```
context_embeddings       (H_i_ctx)        [256]  — candidate 自身
prefix_expanded          (P_t)            [256]  — prefix 自身
product                  (H * P)          [256]  — 元素级匹配
abs_diff                 (|H - P|)        [256]  — 元素级差异
cosine                   (cos(H, P))      [1]    — 方向相似度
bilinear                 (H^T W P)        [256]  — 可学习非对称交互
```

这 6 个特征隐含的假设是：前缀-候选关系可以用**逐元素乘积 + 逐元素差异 + 单一相似度 + 双线性评分**来描述。但 fact-checking 中下一条 ideal evidence 的判断可能涉及更结构化的比较——比如"当前 prefix 已覆盖了 claim 的 entity/location 但没覆盖 time/quantity，这条 candidate 正好覆盖了 time/quantity"。这种**多维条件判断**在 256 维的连续向量交互中可能无法被清晰表达，因为没有显式的维度解耦机制。

---

### 8. Scorer 仅 2 层

**现状**：`MLP(1281 → 256 → 1)`（`sequential.py:76-81`）。

1281 维拼接特征 → 256 隐层 → 1 标量 logit。只有 2 层，隐层后直接输出，中间没有更多非线性变换。对于需要综合 6 种异质交互信号判断一个 scalar utility 的任务，2 层可能不够深。但更关键的是这个 scorer **对所有 step 共享参数**——它必须在 step0、step1、...、step4 使用同一套权重，而不同 step 的交互特征分布可能完全不同（step0 时 prefix=start_prefix，step4 时 prefix 是 4 个 evidence 的均值）。

---

### 9. CE Loss 的单一顺序假设

**现状**：teacher forcing CE（`sequential_teacher_forcing_loss`，`sequential.py:458`）按 oracle greedy order 逐步监督：

```python
target = selected_indices[t]  # 第 t 步"必须"选 oracle 排序中第 t 个
ce_loss = cross_entropy(logits, target)
```

**这是整个架构中可能最根本的 loss 层面的弱点**。Oracle greedy order 是**一种**合理顺序，但不是**唯一**合理顺序。考虑：

- Oracle 按 margin 贪心排序：[c3, c7, c1, c9, c5]
- 模型预测：[c7, c3, c1, c9, c5]

模型第 0 步选了 c7 而非 c3，但 c7 在 oracle 排序中排第 1。这两条 evidence 可能同样是好的第一条——模型只是对"哪个应该先选"有不同的判断。但 CE loss 在第 0 步给了模型一个非 zero 的惩罚（target=c3, pred=c7 ≠ c3），在第 1 步又给了一个惩罚（target=c7 但 c7 已被选走了）。

**CE loss 把顺序错误当成同等严重的错误来惩罚**，而不区分"选了 oracle set 中的另一条"和"选了 oracle set 之外的 noise"。这解释了为什么 L_mask（order-agnostic BCE）被作为 Step4.1-A 的第一优先级修正——它显式告诉模型"这些 candidate 在整个 set 层面都是好的"，而不仅仅"第 t 步必须选这一个"。

---

### 10. L_mask=0 时模型完全没有 set-level 信号

**现状**：`deberta_sequential_deep` 使用 `mask_loss_weight=0.0`。

在这个配置下，模型的 **唯一监督信号** 来自逐步 CE——每一步必须精确命中 oracle greedy order 的下一个 index。模型没有任何机制知道"选了一条在 oracle set 中但不在当前位置的 candidate，比选了一条完全不在 oracle set 中的 candidate 更好"。CE 对这两种错误一视同仁。

这就是为什么 Step4.1-A（`mask_loss_weight=0.2`）已经实现并在跑——它补上了这个监督信号的缺口。

**L_mask 消融实验结果 (2026-05-21)**：三个变体在验证集上的最佳 checkpoint 对比：

| 指标 | deep (mask=0.0) step=2500 | mask02 (0.2) step=2250 | mask05 (0.5) step=2250 |
|---|---|---|---|
| recall@5 | 0.3852 | 0.3758 | 0.3834 |
| jaccard@5 | 0.2615 | 0.2531 | 0.2588 |
| top1_match | 0.1664 | 0.1625 | 0.1719 |
| prefix_match@3 | 0.0055 | 0.0031 | 0.0031 |
| prefix_match@5 | 0.0000 | 0.0000 | 0.0000 |
| oracle_rank_ndcg@5 | 0.3306 | 0.3322 | 0.3352 |
| pairwise_order_acc@5 | 0.5871 | 0.5777 | 0.5995 |
| ordered_exact_match@5 | 0.0078 | 0.0071 | 0.0063 |
| precision@5 | 0.3852 | 0.3758 | 0.3834 |
| ordered_hit@5 | 0.1020 | 0.1013 | 0.1020 |
| first_wrong_step_mean | 0.1954 | 0.1860 | 0.1939 |

Step-wise accuracy：mask05 的 step0=0.1719（vs deep 0.1664）、step1=0.0874（vs 0.0921），变化在波动范围内。

**判断：L_mask 不足以修正"CE Loss 的单一顺序假设"这一短板。**

- Set metrics 不升反降：recall@5 最高 0.3852（即 baseline），jaccard@5 最高 0.2615
- Order metrics 只有微弱增益：oracle_rank_ndcg@5 +0.0046, pairwise_order_acc@5 +0.0124，不足以称"已修正"
- mask02（中间权重）在多项指标上反而不如 mask=0，说明权重调优空间窄
- 均未达到 Step4.1-A go 线（recall@5≥0.40, jaccard@5≥0.275）

**解读**：L_mask 告诉模型"哪些 candidate 在 oracle set 中"，但模型即使知道哪些是好的，仍无法从 deep semantic features 中学会区分。问题回到了 Pair Encoder 的证据效用表示能力——如果 256 维向量本身不编码足够的 evidence utility 信息，知道 label 也无济于事。这进一步支持了 Step4.1-C（claim aspect coverage）和 Step4.2（stance utility semantics）等从源头丰富 candidate 表示的方向。

---

### 11. Exposure Bias（训练-推理不匹配）

**现状**：训练时 prefix 来自 oracle（`teacher_forcing_logits` 用 `selected_indices[:t]`），推理时 prefix 来自模型自己的选择（`greedy_decode` 用 `selected_mask` 逐步累积）。

这是 seq2seq 的经典问题。如果 step0 模型选了 oracle_set 中排第 2 的 candidate（对 set 质量影响不大），那 step1 的 prefix 就偏离了 oracle prefix，后续每一步的输入分布都和训练时不同。step0 accuracy 0.1664 意味着 **83% 的样本从第 1 步开始就在 off-policy prefix 下运行**。

但 Plan 中明确将 OPD/DAgger 推迟到 Step5，因为 step0 在 oracle prefix 下（即 teacher forcing 的理想条件）也只有 0.1664——如果连理想 prefix 下第一步都选不对，那 exposure bias 就不是当前的主要矛盾。

---

### 12. 无显式多样性信号

**现状**：整个训练 pipeline 不包含任何鼓励 top-5 set 多样性的 loss。MMR 在 Stage1 chunk 粗筛阶段做了 diversity，但 Stage2 neural selector 完全没有继承这一信号。

如果 CE + L_mask 让模型学会了"选 oracle set 中的 5 条 evidence"，但不保证这 5 条覆盖了 claim 的不同侧面。模型可能选出 5 条语义高度相似但都排在高 margin 的 evidence，因为 oracle 本身也可能偏好某类 evidence。

这正是 Plan 中 Step4.1-C（claim aspect coverage）和 Step4.2（stance utility semantics）试图解决的方向——给模型提供"覆盖了哪些核查方面"的信号。

#### 多样性在事实核查中的两面性

**多样性不是普适正确的方向。** 对 fact-checking 而言，evidence set 的"决策有用性"才是目标。多样 vs 聚焦只是手段，取决于 claim 性质：

**多样性有益的 case**：复杂、多方面的 claim，如"The economy improved, unemployment fell, and exports grew"——需要覆盖 GDP / 就业 / 贸易等多个核查方面；存在争议的 claim，双方 evidence 都需要呈现。

**多样性有害的 case**：简单事实型 claim，如"Obama was born in Hawaii"——多条同向 confirmatory evidence 比分散的"出生地 + 母亲信息 + 教育背景"更有核查价值。盲目追求多样可能引入 peripheral 但不 verifying 的 noise，稀释真正的关键 evidence。

**核心洞察：需要的是"策略性地决定什么样的 evidence set 最大化 verifier 效用"，而非盲目追求多样性或相似性。**

#### 针对性策略

以下四个策略从不同角度解决"evidence set 构造"问题，按推荐优先级排列：

**策略 1：Stance-aware Selection（最优先）**

不追求语义多样性，追求**观点完备性**。思路来自 L-defense：fact-checking 最需要的不是覆盖更多 topic，而是呈现支持和反对双方的证据，让 verifier 在对比中做出更可靠判断。

具体做法：在每个 candidate 中加入其对该 claim 的立场方向特征（support / refute / qualify / neutral），让 Pointer Head 的 scorer 能看到每条 candidate 的立场信号。前缀中立场分布可被隐式编码（通过 prefix 中已选 evidence 的立场所占比例）。

立场检测不需要 gold label：可以用 NLI 模型或 DeBERTa 本身 fine-tune 一个轻量 stance classification head，给出 `P(support|claim, candidate)`、`P(refute|claim, candidate)` 等分数作为 selector 的额外输入特征。

**策略 2：每步拼接 claim 自身表示（成本最低，可与已实现的 claim_start 互补）**

当前 scorer 每一特征都来自 candidate-contextualized 的 `H_i_ctx` 和 prefix `P_t` 的交互。缺乏一个直接的 claim-only 语义参照。`claim_start` 仅在 step0 提供 claim-grounded 参照，step1+ 消失。

改进：在 scorer 的交互特征中增加一个 claim 自身的表示 `h_claim`（如 DeBERTa pooler on claim-only text，或 candidate pair embedding 的均值）。这让模型在每个 step 都能**直接以 claim 语义为参照**，判断"这个 candidate 补足了 claim 核查的哪个缺失部分"，从而隐式学习到"什么类型的 claim 需要什么类型的 evidence set"。

这个改动零外部依赖，只需在 `encode_inputs` 时额外 tokenize claim-only 文本并在 forward 中产出 `h_claim`，拼接进 scorer 的特征向量中。参数量增加极少（扩展 scorer 输入维度即可）。

**策略 3：Redundancy vs Complementarity 分解**

将"两条 evidence 的关系"分解为两个正交维度：

- **内容重叠度**：说同一件事的程度（候选间语义相似度）
- **立场一致度**：对 claim 的支持/反对方向是否一致

四种组合对应四种信息价值：

| | 立场一致 | 立场不一致 |
|---|---|---|
| 内容高重叠 | 冗余但确认性强（简单 claim 需此） | **直接矛盾**（核查最有价值） |
| 内容低重叠 | 互补同向（复杂 claim 需此） | 侧面冲突（信息量大但易混淆） |

不做全局的"促进多样"，而是在特征层将这两个维度分开表达（如 stance 用策略 1 的 stance head 输出、内容重叠用 `cos(H_i_ctx, H_j_ctx)`），让 Pointer Head 的 scorer 自行学习组合。

**策略 4：Verifier-guided Implicit Strategy Learning**

当前 oracle margin 已经在每条 evidence 上编码了 verifier 效用。Oracle 在某些 claim 上选 5 条同向 evidence，在另一些 claim 上选立场各异、覆盖多方面的 evidence——模型理论上可以通过 teacher-forcing CE 从 oracle 中**隐式学习"不同 claim 需要不同策略"**。

改进方向：不做显式多样性/立场 loss，而是确保训练数据覆盖足够的 claim 类型多样性（事实型 / 观点型 / 因果型），并在 scorer 中充分提供 claim 语义参照（策略 2），让模型有条件学到条件化策略。

#### 方案对比

| 策略 | 训练成本 | 推理成本 | 外部依赖 | 与现有架构兼容性 | 推荐度 |
|---|---|---|---|---|---|
| 策略 1: Stance-aware 特征 | 中 | 低 | stance head/NLI 模型 | 扩展 scorer 输入 | ⭐⭐⭐⭐⭐ |
| 策略 2: 每步 h_claim | 极低 | 零 | 无 | 正向互补 claim_start | ⭐⭐⭐⭐ |
| 策略 3: 冗余/互补分解 | 中 | 低 | 依赖策略 1 的 stance 信号 | 建立在策略 1+2 之上 | ⭐⭐⭐ |
| 策略 4: Verifier-guided 隐式学习 | 实现简单 | 零 | 无 | 数据分布层面 | ⭐⭐ |
| ~~盲目多样性 loss~~ | 低 | 零 | 无 | — | ❌ 不应作为主线 |
| Step4.1-C: aspect coverage (Plan) | 中 | 中 | aspect 标注/抽取 | Plan 中既定方向 | ⭐⭐⭐⭐ |

#### 实验顺序建议

1. **先实现策略 2**（每步 h_claim）——成本最低，与已实现的 `claim_start` 互补，单独验证可归因
2. **并行探索策略 1**（stance-aware）——搭建轻量 stance head，验证 stance 特征对 selector 的帮助
3. **若策略 1+2 有效**，将策略 3（冗余/互补分解）作为特征工程在策略 1 基础上叠加
4. Step4.1-C（aspect coverage）按 Plan 既定顺序进行，不因这些策略被推迟

---

### 13. Greedy Decode 的短视性

**现状**：`greedy_decode`（`sequential.py:164`）每步 argmax，不回溯。选完 5 条后不会有全局检查"这 5 条整体是否最优"。

在 teacher forcing 下模型学习的是给定 oracle prefix 时的 one-step optimal，但 greedy decode 下需要的是给定 self-generated prefix 时的全局 optimal。两者之间的 gap 不只在 prefix 分布（exposure bias），还在优化目标——greedy 每一步局部最优不等于全局最优。

---

### 14. 没有 Step Encoding

**现状**：Pointer Head 的 `scorer` 对所有 step 共享相同权重，prefix state 的唯一差异来自 `selected_mask` 累积。模型无法通过"我现在在第 3 步，只剩两个位置可选"这种 step-level 元信息来调整选择策略。

这可能是小问题——前缀内容本身就隐含了步数信息——但在早期 step（前缀还很少时），模型如果知道"现在是第 0 步，要选最重要的"，可能会调整对 candidate 的评分。

---

### 15. Oracle Label 噪声

**现状**：Oracle greedy order 由 margin 贪心构建。Margin 是 verifier（一个 finetuned Qwen2.5-7B）对每条 evidence 的 label probability 增益。但 verifier 本身有误差——它可能在训练集上 overfit，对某些 evidence type 有系统性偏好。这种偏好通过 margin → oracle order → teacher forcing 传递给了 selector。

具体来说，如果 verifier 偏向数字型 evidence（因为数字在训练数据中关联更明确），那 oracle 会把数字型 candidate 排在前面，selector 学习模仿这种偏好，最终可能忽略同样重要但不含数字的 evidence。

---

### 短板总览与优先级

| 优先级 | 短板 | 严重程度 | 当前状态 |
|---|---|---|---|
| 1 | CE Loss 的单一顺序假设 | 高 — CE 对"选了 oracle set 中但位置不对"和"选了 noise"一视同仁 | Step4.1-A L_mask 已部分修正 |
| 2 | item projection 单层线性压缩 768→256 | 中 — 可能丢多维信号 | 2026-05-21 已实现：2层MLP+残差 |
| 3 | start_prefix 与 claim 无关 | 中 — step0 缺乏 claim-grounded 参照系 | 2026-05-21 已实现：candidate pool mean |
| 4 | 无显式多样性信号 | ~~中~~ — 盲目多样性不适用于fact-checking，已转为针对性策略（stance-aware / h_claim） | 2026-05-21 方案已明确，待实现策略1、2 |
| 5 | Set Encoder 深度与宽度不足 | 中低 — 有理论担忧但现有诊断不支持其为首要瓶颈 | 未处理，有诊断实验方案 |
| 6 | Mean pooling 信息坍缩 | 中低 — token 级对齐信号丢失 | 未处理 |
| 7 | Prefix 均值池化等权 | 中低 — 早期关键 evidence 被稀释 | 未处理 |
| 8 | 交互特征设计假设 | 中低 — 多维条件判断表达受限 | 未处理 |
| 9 | Greedy decode 短视 | 中低 — 在 oracle prefix 下训练，greedy prefix 下推理 | Step5 OPD plan 中 |
| 10 | Scorer 仅 2 层 + step 共享 | 低 — 可能限制交互信号综合 | 未处理 |
| 11 | Tokenization 截断 | ~~低~~ — 已诊断：0.8% 样本受影响，非瓶颈 | 2026-05-21 已排除 |
| 12 | Exposure bias | 低（在当前瓶颈下）— step0 在 oracle prefix 下已低 | Step5 OPD plan 中 |
| 13 | 无 step encoding | 低 — 前缀内容隐含步数 | 未处理 |
| 14 | Oracle label 噪声 | 低 — 间接影响 | 未诊断 |

当前最大的两个损失面问题（1 和 4）已被 Plan 覆盖。第 2、3 项（item projection 加深、start_prefix claim-conditioned）已于 2026-05-21 实现，待跑实验验证。第 5 项（Set Encoder 容量）诊断实验成本低，可在合适时机顺带验证。

## 相关文件

- `src/fact_checking/selectors/sequential.py` — Item Projection 定义（:225-229）、forward 流程（:242-254）、pool_pair_embeddings（:774-784）
- `src/fact_checking/selectors/cross_encoder.py` — `tokenize_claim_candidate_pairs`（:83-99），pair 格式化与 tokenization
- `src/fact_checking/selectors/listwise.py` — `pad_flat_items`（:603-615），展平向量重排为 padded batch
- `src/fact_checking/selectors/stage2_oracle.py` — `candidate_text`（:150-151）、`Stage2OracleExample` 结构、`DEFAULT_CANDIDATE_POOL_SIZE = 15`（:10）
- `src/fact_checking/build/candidates.py` — Stage1/Stage2 分支逻辑
- `src/fact_checking/oracle_pointwise.py` — oracle margin 计算、candidate pool 构建

## 关联文档

- [[202605202008_sequential_pointer_selector_step4_plan]] — Step4 完整实现计划
- [[202605201437_experiment_progress_timeline]] — 实验进展时间线
