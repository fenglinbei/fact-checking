# Adjacent Boundary Chunking：面向 Fact-Checking Evidence Graph 的相邻边界切分方案

## 1. 目标定位

当前任务中，每个样本包含一个 claim 以及对应的 N 条原始 report。原始 report 通常是一大段文本，内部可能混合背景信息、核心事实、引用、反驳、时间线补充、统计数据和评论性内容。直接使用 report-level 作为证据单元会过粗；直接使用 sentence-level 会导致语义碎片化；使用全局两两相似度聚类又容易因为传递性相似导致 chunk 膨胀。

Adjacent Boundary Chunking, 简称 ABC，目标是在 report 内部只依据相邻句子之间的语义连续性判断切分边界，得到长度可控、语义连续、适合构建证据链图的 evidence chunk。

核心原则如下：

```text
只在相邻句子之间判断是否切分，不做任意两句之间的全局聚类。
```

即判断：

```text
s1 | s2 | s3 | ... | sm
```

中每个相邻位置：

```text
boundary_i = 是否在 s_i 和 s_{i+1} 之间切开
```

而不是判断任意句子对是否属于同一个 cluster。

这样可以避免如下问题：

```text
s1 和 s2 相似
s2 和 s3 相似
s3 和 s4 相似
于是 s1-s4 被合并
```

ABC 的基本假设是：report 是线性文本，证据语义连续性主要体现在相邻句子之间；如果相邻句子之间出现语义断裂，则适合作为 chunk 边界。

---

## 2. 方法总览

给定一条 report：

```text
R = [s1, s2, ..., sm]
```

其中 `s_i` 是切分后的第 i 个句子。ABC 的输出是若干连续 sentence span：

```text
C = [c1, c2, ..., ck]
```

每个 chunk 为：

```text
c_j = [s_a, s_{a+1}, ..., s_b]
```

其中 `a <= b`，并且 chunk 内句子必须来自原 report 的连续区间。

整体流程：

```text
原始 report
  ↓
句子切分
  ↓
相邻句子 embedding similarity 计算
  ↓
边界分数 boundary score 计算
  ↓
初始切分
  ↓
长度约束与后处理
  ↓
输出 bounded evidence chunks
  ↓
进入 evidence candidate pool / evidence graph construction
```

---

## 3. 输入与输出定义

### 3.1 输入

每个样本包含：

```json
{
  "sample_id": "xxx",
  "claim": "claim text",
  "reports": [
    {
      "report_id": "R1",
      "text": "raw report text"
    },
    {
      "report_id": "R2",
      "text": "raw report text"
    }
  ]
}
```

### 3.2 输出

ABC 对每条 report 输出若干 evidence chunk：

```json
{
  "sample_id": "xxx",
  "report_id": "R1",
  "chunk_id": "R1_C3",
  "sent_start": 5,
  "sent_end": 7,
  "num_sentences": 3,
  "num_tokens": 92,
  "chunk_text": "...",
  "sentence_ids": [5, 6, 7],
  "boundary_left_score": 0.71,
  "boundary_right_score": 0.66,
  "claim_relevance": 0.58,
  "metadata": {
    "method": "ABC-claim-aware-v1",
    "max_sent_per_chunk": 3,
    "max_tokens_per_chunk": 150
  }
}
```

后续 selector / graph builder 不再直接面对完整 report，而是面对这些长度可控的 chunk。

---

## 4. 核心思想：相邻边界而非全局聚类

### 4.1 原 semantic-level 的问题

原方案是：

```text
若两两句子的 embedding 相似度达到阈值 0.5，则判定为同一 chunk
```

该方案容易产生三类问题：

1. **传递性膨胀**：A-B 相似，B-C 相似，C-D 相似，最终 A-D 被合并，但整体已经不再是单一证据单元。
2. **实体共现误合并**：多个句子共享同一个人物或组织名，但论证功能不同，仍然会被合并。
3. **图节点过大**：chunk 内可能包含多个事实、多个 claim atom 对应信息，证据链图节点失去可解释性。

### 4.2 ABC 的改法

ABC 只考虑相邻句子：

```text
sim_i = cos(emb(s_i), emb(s_{i+1}))
```

如果相邻句子相似度低，或出现实体切换、时间切换、引用主体切换、强转折，则在二者之间切分。

基础边界分数可定义为：

$$B_i = 1 - cos(emb(s_i), emb(s_{i+1}))$$

其中 `B_i` 越大，表示 `s_i` 和 `s_{i+1}` 越可能存在主题或证据功能断裂。

---

## 5. 边界分数设计

### 5.1 基础版本：ABC-basic

最简单版本只使用相邻 embedding similarity：

$$sim_i = cos(emb(s_i), emb(s_{i+1}))$$

$$B_i = 1 - sim_i$$

如果 `B_i` 大于阈值，则切分：

$$boundary_i = 1[B_i > \tau_b]$$

推荐初始设置：

```yaml
boundary_threshold: 0.55
min_sent_per_chunk: 1
max_sent_per_chunk: 3
max_tokens_per_chunk: 150
```

如果 embedding 相似度普遍偏高，也可以改为：

```yaml
sim_threshold: 0.40
boundary_i = 1 if sim_i < sim_threshold
```

### 5.2 局部极值版本：ABC-local-peak

不同 report 的相似度分布可能不同，所以全局阈值不稳定。可以使用 report 内部的相对边界强度。

先计算：

```text
B = [B1, B2, ..., B_{m-1}]
```

然后对 `B_i` 做局部平滑：

$$\tilde{B_i} = mean(B_{i-w}, ..., B_i, ..., B_{i+w})$$

如果 `B_i` 是局部高点，且超过 report 内均值一定幅度，则切分：

$$boundary_i = 1[B_i > mean(B) + \lambda \cdot std(B)]$$

推荐初始设置：

```yaml
smoothing_window: 1
lambda_std: 0.5
min_boundary_gap: 1
```

这里的 `min_boundary_gap` 用于避免连续两个边界过近，导致 chunk 过碎。

### 5.3 Claim-aware 版本：ABC-claim-aware

fact-checking 中，chunk 的目标不是普通主题分割，而是服务于 claim verification。因此边界判定可以加入 claim relevance。

计算每个句子对 claim 的相关性：

$$rel_i = rel(s_i, c)$$

其中可沿用你当前的 hybrid score：

$$rel(s_i,c)=0.70\cdot dense(s_i,c)+0.20\cdot lexical(s_i,c)+0.10\cdot BM25(s_i,c)$$

若相邻句子对 claim 的作用差异很大，则更可能应该切开：

$$D_i^{rel} = |rel_i - rel_{i+1}|$$

综合边界分数：

$$B_i = w_{sem}(1-sim_i) + w_{rel}|rel_i-rel_{i+1}|$$

推荐初始权重：

```yaml
w_sem: 0.75
w_rel: 0.25
```

该版本适合你的任务，因为同一 report 中经常存在“背景句”和“关键证据句”相邻出现。背景句与证据句可能语义相关，但对 claim 的判别价值不同，应该避免被无条件合并。

### 5.4 Feature-rich 版本：ABC-feature-rich

可以进一步加入实体、时间、转折和引用主体特征：

$$B_i = w_{sem}(1-sim_i) + w_{rel}|rel_i-rel_{i+1}| + w_{ent}(1-Jaccard(E_i,E_{i+1})) + w_{disc}Disc_i + w_{quote}Quote_i + w_{time}Time_i$$

其中：

```text
E_i: s_i 中识别出的实体集合
Jaccard(E_i,E_{i+1}): 相邻句子的实体重叠度
Disc_i: 是否出现转折/话题转换标记
Quote_i: 引用主体是否变化
Time_i: 时间表达是否出现明显变化
```

推荐初始权重：

```yaml
w_sem: 0.55
w_rel: 0.20
w_ent: 0.10
w_disc: 0.05
w_quote: 0.05
w_time: 0.05
```

该版本不建议作为第一版实现。第一版优先实现 ABC-basic 和 ABC-claim-aware，确认有效后再扩展 feature-rich。

---

## 6. 后处理规则

ABC 的效果很大程度取决于后处理。建议把后处理作为硬约束，而不是只依赖边界分数。

### 6.1 最大长度约束

每个 chunk 必须满足：

```yaml
max_sent_per_chunk: 3
max_tokens_per_chunk: 150
```

如果初始 chunk 超过上限，则在 chunk 内寻找最弱相邻连接处切开。

最弱连接定义为：

$$i^* = argmax_i B_i$$

即在边界分数最高的位置二次切分。

### 6.2 最小长度约束

如果 chunk 过短，比如只有一个很短的句子，并且该句本身不是高相关证据句，则可以合并到相邻 chunk。

合并优先级：

```text
1. 优先合并到相邻相似度更高的一侧
2. 如果左右相似度接近，优先合并到 claim relevance 更高的一侧
3. 如果合并后超过 max_tokens，则不合并
```

建议规则：

```yaml
min_tokens_per_chunk: 20
allow_single_sentence_if_relevant: true
single_sentence_relevance_threshold: 0.55
```

也就是说，如果单句本身对 claim 高相关，可以保留为单句 chunk。否则才触发合并。

### 6.3 高相关句保护

如果某个句子对 claim 极高相关，不应该因为长度或相似度规则被埋在过长 chunk 内。

规则：

```text
如果 rel_i >= high_rel_threshold，则 s_i 至少应该能作为 anchor 被单独定位。
```

推荐参数：

```yaml
high_rel_threshold: 0.70
```

实现方式有两种：

```text
方式 A：高相关句所在 chunk 仍然保留，但记录 anchor_sent_id。
方式 B：额外生成 anchor-level node，chunk 作为 context metadata。
```

建议采用方式 A 起步，后续如果做 Atom-Anchored Evidence Graph，再升级到方式 B。

### 6.4 引用与归属保护

新闻 report 中经常出现引用：

```text
A said that ... . He added that ... .
```

如果第二句存在代词或省略，单独切分可能导致 verifier 不知道 “He” 指谁。因此遇到引用延续时，允许保留在同一 chunk。

简单规则：

```text
如果 s_{i+1} 以 he/she/they/it/this/that/these/those 开头，且 s_i 中存在实体或发言主体，则降低 boundary score。
```

可以实现为：

$$B_i = B_i - \delta_{coref}$$

推荐：

```yaml
coref_boundary_discount: 0.10
```

---

## 7. 推荐实现版本

建议按三个版本逐步实现。

### 7.1 ABC-basic-v1

只使用相邻 embedding similarity 和长度约束。

适合作为最小可用版本。

```text
sentence split
→ adjacent similarity
→ sim_i < threshold 切分
→ max_sent / max_tokens 后处理
```

优点：简单、稳定、可解释。

缺点：不感知 claim，可能把背景句和证据句合并。

### 7.2 ABC-claim-aware-v1

加入 claim relevance 差异。

```text
sentence split
→ adjacent similarity
→ sentence-claim relevance
→ combined boundary score
→ threshold/local peak 切分
→ max_sent / max_tokens 后处理
```

这是我建议你优先用于主实验的版本。

### 7.3 ABC-anchor-v1

在 chunk 内额外标记 anchor sentence。

```text
chunk = bounded context unit
anchor = chunk 内 claim relevance 最高的句子
```

输出时保留：

```json
{
  "chunk_text": "...",
  "anchor_sentence": "...",
  "anchor_sent_id": 6
}
```

后续 evidence graph 可以使用 anchor 作为节点核心，chunk 作为上下文。

这与后续 Atom-Anchored QA Evidence Chain 更兼容。

---

## 8. 伪代码

### 8.1 主函数

```python
def adjacent_boundary_chunking(report_text, claim, config):
    # 1. Sentence splitting
    sentences = split_into_sentences(report_text)
    if len(sentences) == 0:
        return []
    if len(sentences) == 1:
        return build_single_chunk(sentences, claim, config)

    # 2. Sentence embeddings
    sent_embs = embed_sentences(sentences)
    claim_emb = embed_text(claim)

    # 3. Adjacent semantic similarity
    adj_sims = []
    for i in range(len(sentences) - 1):
        sim = cosine(sent_embs[i], sent_embs[i + 1])
        adj_sims.append(sim)

    # 4. Claim relevance
    rel_scores = []
    for s in sentences:
        rel = compute_claim_relevance(s, claim, claim_emb, config)
        rel_scores.append(rel)

    # 5. Boundary score
    boundary_scores = []
    for i in range(len(sentences) - 1):
        b_sem = 1.0 - adj_sims[i]
        b_rel = abs(rel_scores[i] - rel_scores[i + 1])
        b = config.w_sem * b_sem + config.w_rel * b_rel

        # optional feature adjustment
        b = adjust_boundary_score(
            b=b,
            left_sentence=sentences[i],
            right_sentence=sentences[i + 1],
            config=config,
        )
        boundary_scores.append(b)

    # 6. Initial boundary decision
    boundaries = decide_boundaries(boundary_scores, sentences, config)

    # 7. Build initial chunks
    chunks = build_chunks_from_boundaries(sentences, boundaries)

    # 8. Post-processing
    chunks = enforce_max_length(chunks, boundary_scores, config)
    chunks = merge_too_short_chunks(chunks, sent_embs, rel_scores, config)

    # 9. Add metadata and anchor sentence
    output_chunks = []
    for chunk in chunks:
        output_chunks.append(build_chunk_record(chunk, sentences, rel_scores, boundary_scores, config))

    return output_chunks
```

### 8.2 边界判断函数

```python
def decide_boundaries(boundary_scores, sentences, config):
    boundaries = []

    if config.boundary_mode == "absolute":
        for i, score in enumerate(boundary_scores):
            is_boundary = score >= config.boundary_threshold
            boundaries.append(is_boundary)

    elif config.boundary_mode == "local_peak":
        mean_b = mean(boundary_scores)
        std_b = std(boundary_scores)
        dynamic_threshold = mean_b + config.lambda_std * std_b

        for i, score in enumerate(boundary_scores):
            left = boundary_scores[i - 1] if i > 0 else -1
            right = boundary_scores[i + 1] if i < len(boundary_scores) - 1 else -1
            is_local_peak = score >= left and score >= right
            is_boundary = is_local_peak and score >= dynamic_threshold
            boundaries.append(is_boundary)

    else:
        raise ValueError(f"Unknown boundary_mode: {config.boundary_mode}")

    boundaries = remove_boundaries_that_create_invalid_chunks(boundaries, sentences, config)
    return boundaries
```

### 8.3 最大长度后处理

```python
def enforce_max_length(chunks, boundary_scores, config):
    final_chunks = []

    for chunk in chunks:
        queue = [chunk]
        while queue:
            cur = queue.pop(0)
            if is_valid_length(cur, config):
                final_chunks.append(cur)
            else:
                split_pos = find_weakest_connection(cur, boundary_scores)
                left_chunk, right_chunk = split_chunk(cur, split_pos)
                queue.append(left_chunk)
                queue.append(right_chunk)

    return final_chunks
```

### 8.4 过短 chunk 合并

```python
def merge_too_short_chunks(chunks, sent_embs, rel_scores, config):
    merged = []
    i = 0

    while i < len(chunks):
        cur = chunks[i]

        if not is_too_short(cur, config):
            merged.append(cur)
            i += 1
            continue

        if has_high_relevance_sentence(cur, rel_scores, config):
            merged.append(cur)
            i += 1
            continue

        left_candidate = merged[-1] if len(merged) > 0 else None
        right_candidate = chunks[i + 1] if i + 1 < len(chunks) else None

        target = choose_merge_target(cur, left_candidate, right_candidate, sent_embs, rel_scores, config)

        if target == "left":
            merged[-1] = merge_chunks(merged[-1], cur)
            i += 1
        elif target == "right":
            chunks[i + 1] = merge_chunks(cur, chunks[i + 1])
            i += 1
        else:
            merged.append(cur)
            i += 1

    return merged
```

---

## 9. 参数建议

### 9.1 第一版默认参数

```yaml
method: ABC-claim-aware-v1
sentence_splitter: spacy_or_nltk
embedding_model: same_as_current_dense_retriever
boundary_mode: local_peak
smoothing_window: 1
lambda_std: 0.5
w_sem: 0.75
w_rel: 0.25
min_sent_per_chunk: 1
max_sent_per_chunk: 3
min_tokens_per_chunk: 20
max_tokens_per_chunk: 150
allow_single_sentence_if_relevant: true
single_sentence_relevance_threshold: 0.55
high_rel_threshold: 0.70
coref_boundary_discount: 0.10
```

### 9.2 更保守的参数

如果切得太碎：

```yaml
lambda_std: 0.8
max_sent_per_chunk: 4
max_tokens_per_chunk: 180
coref_boundary_discount: 0.15
```

### 9.3 更激进的参数

如果 chunk 仍然太长：

```yaml
lambda_std: 0.3
max_sent_per_chunk: 2
max_tokens_per_chunk: 120
w_sem: 0.65
w_rel: 0.35
```

---

## 10. 与 Evidence Graph 的衔接

建议将 ABC 输出的 chunk 分成两层使用。

### 10.1 Graph node 层

证据链图中的节点可以使用 chunk，也可以使用 chunk 内 anchor sentence。

推荐结构：

```text
node_id: R1_C3
node_text: anchor_sentence
context_text: chunk_text
source_report: R1
sent_span: 5-7
```

即：

```text
图推理使用 anchor_sentence
verifier 输入展示 chunk_text
```

这样可以兼顾图结构清晰度和语义完整度。

### 10.2 Candidate scoring 层

对 chunk 的 relevance 可以由两部分构成：

$$score(chunk,c)=\alpha \cdot max_i rel(s_i,c)+\beta \cdot mean_i rel(s_i,c)+\gamma \cdot rel(chunk,c)$$

推荐初始权重：

```yaml
alpha: 0.50
beta: 0.25
gamma: 0.25
```

解释：

```text
max_i rel(s_i,c): 保证关键证据句不被平均稀释
mean_i rel(s_i,c): 反映 chunk 整体相关性
rel(chunk,c): 反映完整 chunk 与 claim 的匹配程度
```

### 10.3 Prompt rendering 层

建议在 prompt 中明确区分 anchor 与 context：

```text
[E3 | report=R1 | span=sent_5-7]
Anchor: The unemployment rate fell to 3.8% in March.
Context: The Labor Department released new figures on Friday. The unemployment rate fell to 3.8% in March. Analysts had expected no change.
```

如果 prompt 长度紧张，可以只显示：

```text
[E3 | R1:sent_5-7]
The Labor Department released new figures on Friday. The unemployment rate fell to 3.8% in March. Analysts had expected no change.
```

---

## 11. 实验矩阵

建议至少跑以下对比：

| Method | Evidence unit | 是否 claim-aware | 是否长度上限 | 图节点建议 |
|---|---|---:|---:|---|
| Report-level | full report | 否 | 否 | report node |
| Sentence-level | sentence | 是 | 天然短 | sentence node |
| Old semantic-level | pairwise clustering chunk | 否/弱 | 否 | chunk node |
| ABC-basic | adjacent boundary chunk | 否 | 是 | chunk node |
| ABC-claim-aware | adjacent boundary chunk | 是 | 是 | chunk node |
| ABC-anchor | adjacent boundary chunk + anchor | 是 | 是 | anchor node + chunk context |

重点建议优先比较：

```text
sentence-level vs old semantic-level vs ABC-basic vs ABC-claim-aware vs ABC-anchor
```

---

## 12. 评价指标

### 12.1 下游性能指标

```text
accuracy
macro-F1
weighted-F1
per-class F1
```

### 12.2 Evidence quality 指标

```text
avg_chunk_tokens
avg_chunk_sentences
chunk_length_std
num_chunks_per_report
num_chunks_per_sample
selected_evidence_tokens_per_sample
evidence_redundancy
oracle evidence coverage
```

其中 evidence redundancy 可以定义为被选中证据之间的平均相似度：

$$Redundancy = mean_{i \neq j} cos(emb(e_i), emb(e_j))$$

oracle evidence coverage 可定义为：

$$Coverage@k = \frac{|OracleEvidence \cap SelectedEvidence@k|}{|OracleEvidence|}$$

如果没有人工 oracle evidence，可以用当前 verifier 下穷举得到的 oracle set 作为近似监督信号。

### 12.3 切分质量人工抽检

建议每种方法抽样 50-100 条 report，人工标注以下错误类型：

```text
1. over-split: 一个完整事实被切成多个碎片
2. under-split: 多个事实被合并成一个过长 chunk
3. attribution loss: 引用主体丢失
4. temporal confusion: 时间线混在一起
5. claim-irrelevant expansion: chunk 中加入太多无关背景
6. graph-node ambiguity: 一个节点包含多个不同证据功能
```

---

## 13. 预期效果与判断逻辑

如果 ABC-basic 优于 old semantic-level，说明你的主要问题来自全局聚类膨胀。

如果 ABC-claim-aware 优于 ABC-basic，说明 claim relevance 对 report 内部切分有帮助，任务更偏 evidence-oriented segmentation，而不是普通 topic segmentation。

如果 ABC-anchor 优于 ABC-claim-aware，说明下游 evidence graph 更适合使用细粒度 anchor，而 chunk 应主要作为上下文容器。

如果 sentence-level 仍然最优，说明 verifier 更偏好短证据输入，此时可以保留 sentence-level 作为主节点，但使用 ABC chunk 作为 metadata context。

---

## 14. 常见失败模式与修复

### 14.1 切得太碎

表现：大量 1 句 chunk，verifier 无法理解代词和引用归属。

修复：

```yaml
lambda_std: 增大
max_sent_per_chunk: 增大到 4
min_tokens_per_chunk: 增大到 30
coref_boundary_discount: 增大
```

### 14.2 chunk 仍然太长

表现：一个 chunk 中包含多个事实或多个 claim atom。

修复：

```yaml
max_sent_per_chunk: 降低到 2
max_tokens_per_chunk: 降低到 120
w_rel: 增大
lambda_std: 降低
```

### 14.3 背景句和证据句混在一起

表现：chunk 前半段是背景，后半段是核心证据。

修复：

```yaml
使用 ABC-claim-aware
提高 w_rel
对高 rel 句设置 anchor_sent_id
```

### 14.4 引用主体丢失

表现：chunk 以 “He said...” 或 “The agency added...” 开头，但缺少前一句主体。

修复：

```yaml
增加 coref_boundary_discount
对代词开头句强制向前扩展一句
记录 quote attribution metadata
```

### 14.5 同一实体导致误合并

表现：多句都提到同一个人或组织，但实际是不同事件。

修复：

```yaml
加入 time_shift feature
加入 discourse marker feature
提高 max_sent hard cap 的优先级
```

---

## 15. 实现检查清单

第一阶段：最小可用版本

```text
[ ] report sentence splitter
[ ] sentence embedding
[ ] adjacent similarity calculation
[ ] boundary score = 1 - adjacent similarity
[ ] local peak / threshold boundary decision
[ ] max_sent_per_chunk hard cap
[ ] max_tokens_per_chunk hard cap
[ ] chunk json output
```

第二阶段：claim-aware 版本

```text
[ ] sentence-claim relevance calculation
[ ] hybrid relevance score reuse
[ ] boundary score 加入 |rel_i - rel_{i+1}|
[ ] chunk-level relevance score
[ ] anchor sentence selection
```

第三阶段：graph integration

```text
[ ] chunk_id 与 report_id / sentence span 对齐
[ ] graph node 使用 anchor 或 chunk
[ ] prompt rendering 区分 anchor/context
[ ] selector 输入替换为 ABC chunks
[ ] 记录 selected chunk 的来源和边界分数
```

第四阶段：实验与诊断

```text
[ ] 跑 sentence-level baseline
[ ] 跑 old semantic-level baseline
[ ] 跑 ABC-basic
[ ] 跑 ABC-claim-aware
[ ] 跑 ABC-anchor
[ ] 输出 chunk 长度分布
[ ] 输出 verifier token cost
[ ] 输出 case study
```

---

## 16. 推荐默认落地方案

建议当前项目优先实现：

```text
ABC-claim-aware-v1 + max 3 sentences + max 150 tokens + anchor sentence metadata
```

默认配置：

```yaml
method: ABC-claim-aware-v1
boundary_mode: local_peak
lambda_std: 0.5
w_sem: 0.75
w_rel: 0.25
max_sent_per_chunk: 3
max_tokens_per_chunk: 150
min_tokens_per_chunk: 20
high_rel_threshold: 0.70
anchor_selection: max_claim_relevance_sentence
```

最终输出给 evidence graph 的推荐格式：

```json
{
  "node_id": "R1_C3",
  "node_type": "abc_chunk",
  "anchor_text": "chunk 内 claim relevance 最高的句子",
  "context_text": "完整 chunk 文本",
  "report_id": "R1",
  "sent_span": [5, 7],
  "claim_relevance": 0.58,
  "boundary_info": {
    "left": 0.71,
    "right": 0.66
  }
}
```

一句话总结：

```text
ABC 不把相似句子做全局聚类，而是在相邻句子之间寻找语义边界，并用长度上限与 claim relevance 约束 chunk，最终得到既不碎片化、也不膨胀的 evidence unit。
```

---

## 17. 参考方法

本方案主要借鉴以下方向，但针对 fact-checking evidence graph 做了任务化改造：

```text
1. TextTiling：基于局部语义或词汇连续性检测 topic boundary。
2. C99：利用句间相似度结构进行线性文本分割。
3. Sentence-window retrieval：以句子为小节点，同时保留周围上下文窗口。
4. Parent-document retrieval：检索小单元，但回填较大的 parent context。
5. Fact-checking sentence evidence：以句子或多句证据作为 claim verification 的基本证据来源。
```
