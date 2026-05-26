# Evidence Selector 实验路线与方法调研整理

文档更新时间：2026-05-19

本文整理当前 fact-checking evidence selector 主线的实验顺序、实验约束、评估规范、GRPO+OPD 可行性，以及可参考的相关文献方法。本文的核心立场是：当前任务不是继续预测 MMR scalar lambda，而是学习一个可部署的 **utility-aware ordered evidence selector**，在固定的 sentence-level top15 candidate pool 中输出有序 top5 evidence list，使下游 verifier 输入尽量接近 Stage2 margin oracle 的 `selected_indices` 与 greedy order。

本文中所有公式均写成单行 `$$...$$`，避免在部分 Markdown / LaTeX 渲染器中因换行导致公式解析失败。

---

## 1. 当前任务定义

### 1.1 输入与输出

当前 selector 的部署输入是：

```text
claim c
candidate pool C = {d_1, ..., d_N}, N <= 15
candidate texts + retrieval scores + metadata + embeddings / pairwise similarities
```

输出是：

```text
ordered evidence list L_5 = [d_{i_1}, d_{i_2}, ..., d_{i_5}]
```

其中 `L_5` 是有序 top5，而不是无序集合。排序本身是任务目标的一部分，因为 order-sensitivity 诊断已经显示，同一组 oracle evidence 只改变顺序，verifier 输出会显著变化。

### 1.2 训练监督的语义

Stage2 oracle 的监督不是人工 evidence relevance label，而是 verifier-utility supervision。当前 oracle objective 为 margin：

$$\text{margin}(c,S_5)=\log P(y_{\text{gold}}\mid c,S_5)-\max_{y\ne y_{\text{gold}}}\log P(y\mid c,S_5)$$

因此 `selected_indices` 应理解为：在当前 calibrated verifier / scorer 下，能够最大化 gold-label margin 的 evidence set 与 greedy order。它不等同于“客观上最相关的句子”，也不等同于传统 IR relevance label。

### 1.3 当前瓶颈判断

当前 oracle evidence + oracle-direct verifier 已经在 val 上达到约 0.7111 accuracy / 0.7169 macro-F1；同一个 verifier 换成 fixed-MMR sentence evidence 后回到约 0.2716 / 0.2663，换成当前 pointwise sentence evidence 后约 0.2637 / 0.2596。当前 pointwise selector 的 selection-only 指标也较弱，recall@5≈0.3755、Jaccard@5≈0.2536。因此主要瓶颈不是 verifier、decode、checkpoint 或 prompt budget，而是 selector 没有选到接近 oracle evidence distribution 的证据。

---

## 2. 不可变实验约束

后续 selector 的训练、selection-only eval、build pipeline 必须共享同一候选池口径。

```text
chunk_mmr_fingerprint = 432dfc970e75
chunking.strategy     = sentence
candidate pool        = dedup -> hybrid top15
selector output       = ordered top5
```

硬约束如下：

1. 不使用 semantic chunking 作为主线。Semantic-level oracle 在 paired subset 上显著低于 sentence-level oracle，只保留为 granularity diagnostic。
2. 不使用旧 V1 reconstructed / positive-injected candidate pool。
3. Selection-only eval 中不得额外注入 oracle positives。
4. 不混用其他 chunk cache fingerprint。
5. 训练、评估、pipeline build 都要检查并拒绝 fingerprint mismatch。
6. gold label、oracle margin、oracle selected flag、oracle label logprobs 只能用于训练标签、过滤或分析，不能作为部署时 selector 输入。
7. 所有模型选择和阈值调节只在 val 上做；test 只用于最终确认。

### 2.1 Hybrid candidate pool

候选池由正式 build pipeline 产生，selector 不重新检索。流程为：

```text
raw report sentences
-> sentence chunks
-> BGE embedding + lexical / BM25 scoring
-> candidate dedup
-> hybrid_score descending top15
-> selector chooses ordered top5
```

Hybrid score 由三路分数组成：dense 0.70、lexical 0.20、BM25 0.10。Stage2 oracle result 中保存的 `candidate_pool` 就是 selector 需要面对的候选池；`selected_indices` 是该 `candidate_pool` 内的坐标。

---

## 3. 总体实验顺序

推荐按以下顺序推进。核心原则是：先用监督学习和蒸馏证明 oracle evidence pattern 在正式候选池下可学，再考虑 RL refinement。

```text
Step 0: 数据与评估 harness 审计
Step 1: Cross-encoder pairwise reranker
Step 2: Cross-encoder score + light MMR diversity 后处理
Step 3: Set-aware / listwise 15-candidate reranker
Step 4: Sequential pointer selector
Step 5: OPD / DAgger-style on-policy distillation
Step 6: KL-constrained GRPO refinement
```

第一轮不建议直接从 GRPO 开始。当前已经有强 oracle supervision，action space 也很小，直接 RL 会把一个可诊断的 ranking/listwise learning 问题变成高方差 reward optimization 问题。更合理的路线是先 SFT/listwise imitation，再用 OPD 修正 exposure bias，最后用 GRPO 做小步 end-to-end utility 微调。

---

## 4. Step 0：数据与评估 harness 审计

### 4.1 目标

在训练任何新 selector 前，先建立可复用的数据加载、fingerprint audit、selection-only eval、trace 保存与 full-pipeline trigger。否则后续模型结果容易重复 V1 旧错误：候选池不一致、positive injection、cache mismatch 或评估前重排。

### 4.2 必做检查

每条 oracle row 读取后必须检查：

```text
candidate_pool exists
selected_indices exists
0 <= selected_indices[i] < len(candidate_pool)
candidate_pool_metadata.chunk_mmr_fingerprint == 432dfc970e75
search_objective == margin
chunking == sentence or equivalent metadata confirms sentence-level pool
len(candidate_pool) <= 15
len(selected_indices) <= 5
```

如果 fingerprint mismatch，直接 fail fast，不做 silent fallback。

### 4.3 Trace 文件要求

每个 selector 的 selection-only eval 必须保存 JSONL trace。每条记录至少包含：

```text
event_id
claim
gold_label
candidate_pool
candidate_scores
oracle_ordered_indices
selector_ordered_indices
selector_scores
set_overlap
recall@5
precision@5
jaccard@5
top1_match
prefix_match@1 / @3 / @5
ordered_hit@5
oracle_rank_ndcg@5
pairwise_order_acc@5
overlap_pair_count
ordered_exact_match@5
fingerprint
```

注意：`selector_ordered_indices` 不得在 eval 前按 hybrid score、candidate_pool index 或其他规则重排。selector 输出顺序必须原样进入评估。

---

## 5. Step 1：Cross-encoder pairwise reranker

### 5.1 目标

这是替换当前 logreg pointwise 的最小强模型。它仍然逐候选打分，但通过 claim-candidate token-level cross-attention 学习 utility-aware relevance，应该明显强于 BGE / hybrid score / 手工特征。

### 5.2 模型

推荐优先尝试：

```text
microsoft/deberta-v3-base
answerdotai/ModernBERT-base
BAAI/bge-reranker-base / bge-reranker-large, if compute allows
```

输入格式：

```text
[CLS] Claim: <claim> [SEP] Evidence: <candidate_text>
```

输出：

```text
score_i = f_theta(claim, candidate_i)
```

推理时对每条 claim 的最多 15 个候选分别打分，按 `score_i` 降序取 ordered top5。

### 5.3 训练数据

每个 claim 内构造 positives 和 negatives：

```text
positives = candidate_pool[selected_indices]
negatives = candidate_pool indices not in selected_indices
```

负例采样优先级：

| 负例类型 | 作用 |
|---|---|
| hybrid rank <= 5 但 oracle 未选 | 学会区分“检索相关”与“verifier utility” |
| fixed-MMR top5 但 oracle 未选 | 直接击败 MMR baseline |
| 与 oracle positive BGE 相似度高但未选 | 学会去冗余 |
| label / domain / sent_idx 相近但未选 | 防止靠浅层 metadata shortcut |
| 其余 random negatives | 保持覆盖 |

### 5.4 Loss

基础 pairwise logistic loss：

$$\mathcal{L}_{\text{pair}}=-\log\sigma(s(c,d^+)-s(c,d^-))$$

加入 selected-mask BCE：

$$\mathcal{L}_{\text{bce}}=-\sum_i z_i\log p_i+(1-z_i)\log(1-p_i)$$

其中 `z_i=1` 表示 candidate `i` 在 oracle selected_indices 中。

对 oracle order 加位置权重：

$$w_t=\frac{1}{\log_2(t+1)}$$

总 loss 建议从以下组合开始：

$$\mathcal{L}=\mathcal{L}_{\text{pair}}+0.3\mathcal{L}_{\text{bce}}+0.5\mathcal{L}_{\text{order-pair}}$$

其中 `order-pair` 是 selected evidence 内部的相对顺序约束，例如 oracle rank 更靠前的 evidence 分数应高于 rank 更靠后的 evidence。

### 5.5 预期与 Go/No-Go

该模型必须先在 selection-only 上明显超过当前 pointwise baseline。建议初始 gate：

```text
recall@5 >= 0.50
jaccard@5 >= 0.35
oracle_rank_ndcg@5 > current pointwise
oracle_rank_ndcg@5 > hybrid-order control
top1_match > current pointwise
```

如果 Step 1 连 set metrics 都达不到上述目标，不进入 full pipeline。应先检查 hard negative sampling、loss 权重、fingerprint audit、是否意外重排、是否使用了错误 candidate pool。

---

## 6. Step 2：Cross-encoder score + light MMR diversity

### 6.1 目标

Cross-encoder 仍然逐候选打分，不知道已选 top5 内部是否冗余。Step 2 在 neural score 之上加入轻量 set-aware 后处理，作为从 pointwise scoring 到 sequential selection 的过渡。

### 6.2 选择函数

第 `t` 步对 remaining candidate `i` 计算：

$$\text{score}_t(i)=\alpha\tilde{s}_i+\delta\widetilde{\text{hybrid}}_i-\beta\max_{j\in S_{t-1}}\text{sim}(d_i,d_j)+\gamma\text{coverage}_t(i)$$

其中：

| 项 | 含义 |
|---|---|
| `s_i` | cross-encoder neural utility score |
| `hybrid_i` | 原 retrieval score prior |
| `sim(d_i,d_j)` | BGE cosine similarity，用于 redundancy penalty |
| `coverage_t(i)` | 对 claim 中未覆盖实体、数字、关键内容词的增益 |

### 6.3 参数建议

先做小网格，不要重回 scalar λ 大规模调参路线：

```text
beta ∈ {0.00, 0.05, 0.10, 0.20}
alpha = 1.0
delta ∈ {0.0, 0.1, 0.2}
gamma ∈ {0.0, 0.05}
```

如果 `beta > 0` 导致 recall@5 或 oracle_rank_ndcg@5 下降，应保留纯 neural score。当前 oracle selected set 未必偏好多样性，它偏好 verifier margin，因此 diversity penalty 只能是轻量辅助。

---

## 7. Step 3：Set-aware / listwise 15-candidate reranker

### 7.1 目标

如果 Step 1 / Step 2 set metrics 有提升但 full pipeline 仍然低，说明逐候选 scoring 仍不足。Step 3 应把 top15 candidate pool 作为一个整体输入，显式建模候选间 interaction、相对重要性和局部 ranking context。

### 7.2 架构

推荐架构：

```text
for each candidate i:
    h_i = PairEncoder(claim, candidate_i)
    numeric_i = [hybrid, dense, lexical, bm25, rank, sent_idx, length, entity_overlap, number_overlap]
    x_i = projection(concat(h_i, numeric_i))

H = TransformerEncoder(x_1, ..., x_N, attention_mask)
score_i = MLP(H_i)
```

PairEncoder 可以是 cross-encoder CLS，也可以先用较轻的 claim/candidate encoder 产生 pair representation。N<=15，set Transformer 成本可控。

### 7.3 Loss

selected mask：

$$\mathcal{L}_{\text{mask}}=-\sum_i z_i\log p_i+(1-z_i)\log(1-p_i)$$

Plackett-Luce / ListMLE order loss：

$$\mathcal{L}_{\text{PL}}=-\sum_{t=1}^{K}\log\frac{\exp(s_{\pi_t})}{\sum_{j\in R_t}\exp(s_j)}$$

其中 `π_t` 是 oracle greedy order 的第 `t` 个 candidate，`R_t` 是第 `t` 步尚未放置的候选集合。

位置加权 pairwise loss：

$$\mathcal{L}_{\text{pos-pair}}=\sum_{a<b}w_{a,b}\cdot[-\log\sigma(s_{\pi_a}-s_{\pi_b})]$$

推荐总 loss：

$$\mathcal{L}=\lambda_1\mathcal{L}_{\text{mask}}+\lambda_2\mathcal{L}_{\text{PL}}+\lambda_3\mathcal{L}_{\text{pos-pair}}$$

初始可设：

```text
lambda_1 = 0.3
lambda_2 = 1.0
lambda_3 = 0.5
```

### 7.4 输入顺序与 permutation 问题

Candidate pool 原始顺序包含 hybrid rank prior，不应完全抹掉。但模型也不能只记住初始 rank。因此建议：

1. 保留 candidate rank embedding 与 hybrid score。
2. 训练中做少量 candidate permutation augmentation。
3. 输出始终按 candidate id 对齐。
4. 单独报告 no-shuffle 与 shuffle-aug 两个版本。

### 7.5 何时进入下一步

若 listwise reranker 的 set metrics 已明显优于 cross-encoder，但 order metrics 仍低，进入 Step 4 sequential selector。若 order metrics 高但 full pipeline 低，需要检查 verifier 是否对 learned selector evidence distribution 不稳，而不是立即上 RL。

---

## 8. Step 4：Sequential pointer selector

### 8.1 目标

Sequential selector 直接模拟 oracle greedy search：每一步在 remaining candidates 中选择下一条 evidence，输出天然有序。它比 listwise reranker更适合学习 coverage、redundancy、prefix dependence 和 oracle greedy order。

### 8.2 状态与动作

状态：

```text
s_t = claim + candidate_pool + selected_prefix S_{t-1} + remaining_mask + retrieval features + pairwise similarities
```

动作：

```text
a_t ∈ remaining candidate indices
optional STOP action, if len(selected_indices) < 5 cases need support
```

### 8.3 模型

推荐从以下结构开始：

```text
candidate representation: h_i from PairEncoder or cached cross-encoder layer
prefix representation: pooled selected_prefix representation + max/mean sim-to-prefix
step representation: t embedding + previous action embedding
logit_i,t = MLP([h_i, prefix_repr, step_repr, numeric_i, redundancy_i, coverage_i])
```

### 8.4 Teacher forcing loss

$$\mathcal{L}_{\text{seq}}=-\sum_{t=1}^{K}w_t\log\pi_\theta(a_t^\star\mid s_t)$$

其中 `a_t^*` 是 oracle selected_indices 的第 `t` 个 index。

### 8.5 主要风险

Sequential selector 的主要问题是 exposure bias：训练时 prefix 来自 oracle，推理时 prefix 来自模型自身。一旦第一步错，后续 state distribution 会偏离训练分布。这个问题正是 Step 5 OPD / DAgger-style distillation 要解决的。

---

## 9. Step 5：OPD / DAgger-style on-policy distillation

### 9.1 可行性判断

OPD 在当前任务上可行性高，优先级高于直接 GRPO。原因是：

1. 当前任务是短 horizon sequential decision，K<=5，action space<=15。
2. 已有 Stage2 margin oracle，可作为 teacher 给任意 prefix 下的 next-action target。
3. OPD 能在 student policy 自己访问到的 states 上提供 dense supervision，缓解 teacher forcing 的 exposure bias。
4. 相比 RL final reward，OPD 的 step-level signal 更密集、更可诊断。

### 9.2 基本算法

先用 Step 4 的 teacher forcing policy 作为初始 student。然后迭代：

```text
for each OPD round k:
    for each train claim:
        rollout current selector to produce prefix S_{t-1}
        for each step t:
            query teacher on current on-policy prefix
            score each remaining candidate by margin(c, S_{t-1} ∪ {d_i})
            convert teacher scores to hard argmax or soft target distribution
            add (state, teacher target) to aggregated dataset
    train selector on aggregated dataset with CE + KL
```

Teacher soft target：

$$q_t(i)=\frac{\exp(m_t(i)/\tau)}{\sum_{j\in R_t}\exp(m_t(j)/\tau)}$$

其中 `m_t(i)` 是在当前 prefix 下加入 candidate `i` 后的 oracle margin。

OPD loss：

$$\mathcal{L}_{\text{OPD}}=\sum_t\operatorname{KL}(q_t(\cdot)\,\|\,\pi_\theta(\cdot\mid s_t))+\rho\cdot\mathcal{L}_{\text{CE-hard}}$$

### 9.3 成本估算

每条 claim 每轮最多 teacher scoring 次数为：

$$15+14+13+12+11=65$$

这与 greedy oracle top15 search 同量级。为了控制成本，可以先在 hard subset 上做 OPD：

```text
current selector top1 wrong
or Jaccard@5 < 0.2
or oracle_rank_ndcg@5 low
or full-pipeline wrong but oracle evidence correct
```

### 9.4 关键实现细节

1. Roll-in policy 可用 mixture：前期 `0.5 teacher + 0.5 student`，后期逐渐转为 mostly student。
2. 对 teacher distribution 很平的 state 降权，避免学习 noisy argmax。
3. 每轮 OPD 需要保留 before/after metrics，防止聚合数据导致遗忘初始 oracle prefix。
4. OPD 不使用 gold label 作为 selector inference feature；gold label 只用于 teacher margin scoring 的离线训练过程。
5. OPD 后必须重新跑 selection-only gate，不能只看 training loss。

---

## 10. Step 6：GRPO refinement

### 10.1 可行性判断

GRPO 在当前任务上可行，但应作为 OPD 后的小步 refinement，而不是第一阶段主线。直接从弱 pointwise / weak sequential policy 上 GRPO 风险较高：reward 方差大、credit assignment 弱、容易过拟合固定 verifier 的偏差，并且难以诊断。

推荐判断：

```text
OPD feasibility: high
GRPO feasibility before OPD: low to medium
GRPO feasibility after OPD: medium to medium-high
```

### 10.2 GRPO 任务映射

每个 claim 是一个 group prompt：

```text
group = claim + candidate_pool
sample G ordered top5 lists from policy
score each list by verifier utility reward
normalize rewards within group
update policy with clipped ratio + KL to reference policy
```

Group-relative advantage：

$$A_i=\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}{\operatorname{std}(R_1,\ldots,R_G)+\epsilon}$$

Reward 建议：

$$R(S_5)=\alpha\cdot\text{margin}(c,S_5)+\beta\cdot\mathbb{1}[\hat{y}=y_{\text{gold}}]+\gamma\cdot\text{oracle\_rank\_ndcg@5}(S_5)-\eta\cdot\text{redundancy}(S_5)$$

KL-constrained policy update 的目标可以简化写为：

$$\mathcal{L}_{\text{GRPO}}=-\mathbb{E}\left[\min(r_iA_i,\operatorname{clip}(r_i,1-\epsilon,1+\epsilon)A_i)-\lambda_{\text{KL}}\operatorname{KL}(\pi_\theta\|\pi_{\text{ref}})\right]$$

其中 `π_ref` 应使用 OPD 后的 selector，而不是 uniform policy 或 current weak pointwise model。

### 10.3 进入 GRPO 的前置条件

只有满足以下条件才建议进入 GRPO：

```text
selection-only recall@5 >= 0.50
selection-only jaccard@5 >= 0.35
oracle_rank_ndcg@5 and top1_match clearly improve over pointwise / hybrid-order control
learned selector + oracle-direct verifier exceeds fixed-MMR evidence + oracle-direct verifier
no obvious fingerprint / truncation / duplicate / decode issue
```

若 learned selector + oracle-direct verifier 仍在 0.26-0.28 区间，不应上 GRPO，因为这说明 evidence distribution gap 尚未被监督学习解决。

### 10.4 GRPO 的主要风险与控制

| 风险 | 控制方式 |
|---|---|
| 固定 verifier reward hacking | reward 加入 oracle-order shaping、redundancy penalty，并做 held-out verifier / baseline verifier 对照 |
| reward 方差高 | group size G 不宜太小，先用 OPD policy 降低探索空间 |
| 过早策略坍缩 | 强 KL to OPD reference，小学习率，少轮训练 |
| credit assignment 弱 | GRPO 前必须先做 sequential + OPD，提供合理 initial policy |
| 只优化 margin 不提升 macro-F1 | reward 同时包含 correctness、margin 和 order-aware shaping |

---

## 11. Selection-only Gate 规范

### 11.1 Set metrics

Recall@5：

$$\text{Recall@5}=\frac{|S_{\text{pred}}\cap S_{\text{oracle}}|}{|S_{\text{oracle}}|}$$

Precision@5：

$$\text{Precision@5}=\frac{|S_{\text{pred}}\cap S_{\text{oracle}}|}{|S_{\text{pred}}|}$$

Jaccard@5：

$$\text{Jaccard@5}=\frac{|S_{\text{pred}}\cap S_{\text{oracle}}|}{|S_{\text{pred}}\cup S_{\text{oracle}}|}$$

需要同时报告 micro average、macro by label、per-label breakdown。

### 11.2 Order metrics

| Metric | 定义 | 注意点 |
|---|---|---|
| `top1_match` | `prediction[0] == gold_list[0]` | 捕捉 oracle 第一证据 |
| `prefix_match@k` | 前 k 个位置逐位相同 | 报告 k=1/3/5 |
| `ordered_hit@5` | 逐位置命中比例 | 不等价于 set overlap |
| `oracle_rank_ndcg@5` | oracle rank 作为 graded relevance | `rel(gold[i])=5-i`，不能全设为 1 |
| `pairwise_order_acc@5` | 重合 evidence pair 的相对顺序一致率 | 必须同时报告 `overlap_pair_count` |
| `ordered_exact_match@5` | 完整 ordered list 完全一致 | 严格指标，通常较低 |

oracle_rank_ndcg@5 的 relevance 定义：

$$\text{rel}(d)=\begin{cases}5-t,& d=\text{oracle}[t]\\0,& d\notin\text{oracle}\end{cases}$$

NDCG 计算：

$$\text{NDCG@5}=\frac{\sum_{r=1}^{5}\frac{2^{\text{rel}(\text{pred}_r)}-1}{\log_2(r+1)}}{\text{IDCG@5}}$$

### 11.3 Controls

每个 selector 至少与以下 controls 对照：

```text
current pointwise baseline
hybrid_score top5 baseline
fixed-MMR sentence evidence
same predicted set + hybrid-order reorder
same predicted set + candidate_pool-order reorder
same predicted set + random-order mean over seeds 0-4
```

如果 set metrics 过关但 order metrics 与 hybrid-order / random-order control 接近，不进入 full pipeline；先修正 order loss 或 sequential training。

---

## 12. Full Pipeline Gate 规范

Selection-only 过关后，先跑：

```text
learned selector evidence + oracle-direct verifier
```

对照条件：

| Condition | Val accuracy | Val macro-F1 |
|---|---:|---:|
| oracle evidence + oracle-direct verifier | 0.7111 | 0.7169 |
| fixed-MMR sentence + oracle-direct verifier | 0.2716 | 0.2663 |
| current pointwise sentence + oracle-direct verifier | 0.2637 | 0.2596 |
| fixed-MMR sentence baseline verifier | 0.2951 | 0.2981 |

Go / No-Go：

1. 若 selector + oracle-direct verifier 仍在 0.26-0.28，说明 selector 没有解决 evidence distribution gap。
2. 若超过 fixed-MMR evidence + oracle-direct verifier，但低于 fixed-MMR sentence baseline verifier，需要检查 verifier 是否只适配 oracle evidence distribution。
3. 只有 val 明确超过 fixed-MMR baseline 后，才考虑 test。
4. full pipeline 报告必须检查 parse error、duplicate sample_idx、prompt truncation、fingerprint mismatch。

---

## 13. 数据过滤与 ablation 规范

主实验建议使用 all rows，与 oracle-direct verifier 的构造逻辑保持一致。然后做过滤 ablation。

| Dataset variant | 目的 | 风险 |
|---|---|---|
| all rows | 最大覆盖，主线口径 | 包含 oracle 未使 verifier 正确的样本 |
| `is_correct == true` | 只学 verifier 真实受益的 oracle set | 偏向容易样本 |
| `margin > 0` | 只学 gold margin 为正的 set | 可能丢掉困难但有学习价值样本 |
| high margin | 降低 label noise | 样本少，可能过拟合 |
| label-balanced | 防止 false-side / true-side 偏移 | 可能改变真实分布 |

所有过滤必须写入 report，并且 train/val 不能混用过滤口径。主表用 all rows，过滤结果作为辅助诊断。

---

## 14. 推荐实现里程碑

### Milestone A：selection-only harness

交付：

```text
load_stage2_oracle.py
fingerprint_audit.py
eval_ordered_selector.py
trace schema
hybrid / current pointwise / random-order controls
```

通过标准：能复现 current pointwise recall@5≈0.3755、Jaccard@5≈0.2536，并补齐 order metrics。

### Milestone B：cross-encoder pairwise reranker

交付：

```text
train_cross_encoder_pairwise.py
hard_negative_sampler.py
eval_cross_encoder_selector.py
selection-only report
```

通过标准：recall@5≥0.50、Jaccard@5≥0.35，order metrics 明显强于 pointwise / hybrid controls。

### Milestone C：neural score + diversity后处理

交付：

```text
sequential_mmr_postprocess.py
beta grid report
full trace for each beta
```

通过标准：不牺牲 recall 的情况下提升 oracle_rank_ndcg@5 或 full-pipeline val。

### Milestone D：set-aware listwise reranker

交付：

```text
train_listwise_selector.py
set_transformer_head.py
ListMLE / selected-mask / order-pair losses
shuffle augmentation ablation
```

通过标准：selection-only 与 full-pipeline 都优于 cross-encoder pairwise。

### Milestone E：sequential selector + OPD

交付：

```text
train_sequential_selector.py
opd_rollout_teacher.py
opd_aggregated_dataset.jsonl
OPD round-by-round metrics
```

通过标准：order metrics 明显提升，full-pipeline val 超过 fixed-MMR evidence + oracle-direct verifier。

### Milestone F：GRPO refinement

交付：

```text
group_sampler.py
reward_scorer.py
grpo_update.py
KL-to-OPD-reference monitor
```

通过标准：在不破坏 selection-only order metrics 的前提下，full-pipeline val 有稳定增益。

---

## 15. 与相关文献的对应关系

### 15.1 Cross-encoder reranking

SentenceTransformers 的 retrieve-and-rerank pipeline 采用两阶段思路：先用高效 retriever 找候选，再用 cross-encoder 同时输入 query 和 document 输出相关性分数。BGE reranker 也采用 cross-encoder 架构，对初始检索结果做 top-k rerank。

当前可参考之处：

1. 本项目候选池只有 top15，cross-encoder 推理成本可接受。
2. Cross-encoder 是替代 logreg pointwise 的最低风险强 baseline。
3. 不足是它默认逐候选打分，需要 Step 2 / Step 3 处理 coverage、redundancy 和 order。

参考：

- SentenceTransformers: Retrieve & Re-Rank Pipeline: https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- SentenceTransformers: Cross-Encoders: https://www.sbert.net/examples/cross_encoder/applications/README.html
- BGE Reranker documentation: https://bge-model.com/tutorial/5_Reranking/5.1.html
- BAAI/bge-reranker-large: https://huggingface.co/BAAI/bge-reranker-large

### 15.2 Pairwise / listwise learning-to-rank

ListNet / ListMLE 等 learning-to-rank 方法强调，排序模型不应只把单个 item 或 item pair 当独立训练实例，而应把整个 list 的 predicted ranking 与 ground-truth ranking 作为训练对象。

当前可参考之处：

1. 本项目每个 claim 内的候选相对顺序比跨 claim 绝对分数更可靠。
2. `selected_indices` 的 greedy order 可自然形成 ListMLE / Plackett-Luce loss。
3. 需要避免过度强制 non-selected 之间的任意顺序；loss 应主要约束 selected prefix 与 selected-vs-nonselected 关系。

参考：

- Cao et al., Learning to Rank: From Pairwise Approach to Listwise Approach: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf
- Xia et al., Listwise Approach to Learning to Rank: Theory and Algorithm / ListMLE: https://icml.cc/Conferences/2008/papers/167.pdf

### 15.3 SetRank 与 set-aware ranking

SetRank 的核心主张是 ranking model 应能建模 cross-document interactions，同时对输入 permutation 保持合理不变性。它使用 multi-head self-attention 共同编码 retrieved documents。

当前可参考之处：

1. 本项目 top15 candidates 存在明显候选间交互：冗余、覆盖、证据互补、order effect。
2. Set-aware Transformer 比 pointwise cross-encoder 更适合表达“这条 evidence 是否在当前候选集合中有用”。
3. 由于 candidate rank 是有用 retrieval prior，本项目不应追求完全 permutation-invariant；应保留 rank embedding，同时做 shuffle augmentation 检查模型是否过度依赖初始顺序。

参考：

- Pang et al., SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval: https://arxiv.org/abs/1912.05891

### 15.4 DLCM 与 local ranking context

DLCM 提出在已有初始 ranking 上编码 top results 的 local ranking context，再 refinement 排序。它说明 top results 的局部分布本身是有用信号，逐 item scoring 不足以捕捉这种上下文。

当前可参考之处：

1. 本项目 candidate pool 已固定为 hybrid top15，正是一个 local ranking context。
2. 可将 hybrid rank、dense/lexical/BM25、candidate embedding distribution 作为 listwise model 输入。
3. 可以借鉴“先初始排序，再上下文 refinement”的结构，而不必照搬 RNN。

参考：

- Ai et al., Learning a Deep Listwise Context Model for Ranking Refinement: https://arxiv.org/pdf/1804.05936

### 15.5 Fact verification 中的 utility-aware retrieval

FER 明确提出 fact verification evidence retrieval 不应只优化 relevance，而应利用 claim verifier 的反馈信号，关注 evidence 对最终 verification 的 utility。+VeriRel 同样将 verification success 纳入 ranking，使 retrieval 从普通相关性向 verification-aware ranking 转移。

当前可参考之处：

1. 本项目 Stage2 margin oracle 正是 verifier-utility supervision。
2. 训练目标不应只做 claim-candidate relevance，而要对齐 margin、correctness、oracle order。
3. full-pipeline gate 必须保留，因为 selection-only overlap 不等于 verifier accuracy/F1。

参考：

- Zhang et al., From Relevance to Utility: Evidence Retrieval with Feedback for Fact Verification / FER: https://aclanthology.org/2023.findings-emnlp.422/
- FER OpenReview page: https://openreview.net/forum?id=d0qmGnKfXa
- Deng et al., +VeriRel: Verification Feedback to Enhance Document Retrieval for Scientific Fact Checking: https://arxiv.org/abs/2508.11122

### 15.6 LLM listwise reranking

RankLLM 支持 pointwise、pairwise、listwise reranking，并适配 vLLM / SGLang / TensorRT-LLM 等推理后端。PE-Rank 进一步用 passage embeddings 压缩候选上下文，让 LLM listwise reranking 更高效。

当前可参考之处：

1. 因为 top15 sentence 很短，LLM listwise reranker 可以作为 strong diagnostic baseline。
2. 但 generative index 输出可能有格式错误、index hallucination、decode constraint 成本和推理不稳定问题。
3. 第一轮主线仍建议用 encoder / set-aware scorer；LLM listwise 可作为补充对照或后续 strong reranker。

参考：

- RankLLM: A Python Package for Reranking with LLMs: https://arxiv.org/html/2505.19284v1
- RankLLM GitHub: https://github.com/castorini/rank_llm
- PE-Rank: Leveraging Passage Embeddings for Efficient Listwise Reranking with Large Language Models: https://arxiv.org/abs/2406.14848

### 15.7 DAgger 与 OPD

DAgger 处理 sequential prediction 中的 distribution mismatch：普通 imitation learning 只在 expert trajectories 上训练，推理时一旦 learner 自己犯错，就会进入训练时没见过的 states。DAgger 通过在 learner-induced states 上查询 expert action 并聚合数据来缓解 compounding errors。OPD 与此思想相近：student 采样自己的轨迹，teacher 在这些 on-policy states 上给 dense supervision。

当前可参考之处：

1. Sequential selector 的 exposure bias 与 DAgger/OPD 的问题设定高度一致。
2. Stage2 margin oracle 可以作为 teacher，在 student prefix 下重新计算 next-action target。
3. 相比 GRPO，OPD 的 supervision 更密集、训练更稳定、诊断更直接。

参考：

- Ross, Gordon, Bagnell, A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning: https://proceedings.mlr.press/v15/ross11a.html
- Thinking Machines, On-Policy Distillation: https://thinkingmachines.ai/blog/on-policy-distillation/
- On-Policy Self-Distillation for Large Language Models: https://arxiv.org/html/2601.18734v1

### 15.8 GRPO

GRPO 是 DeepSeekMath 提出的 PPO 变体，使用同一 prompt 下多条 sampled outputs 的 group-relative reward 来估计 advantage，从而减少对 value model / critic 的需求。

当前可参考之处：

1. 本项目自然形成 group：同一 claim + candidate pool 下采样多个 ordered top5 evidence lists。
2. Reward 可以由 verifier margin、argmax correctness、oracle-order shaping 和 redundancy penalty 组成。
3. 但 GRPO 不适合作为第一步；它更适合在 OPD policy 已经接近 oracle distribution 后做小步 refinement。

参考：

- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models: https://arxiv.org/abs/2402.03300

---

## 16. 建议优先级与决策表

| 阶段 | 优先级 | 是否现在做 | 进入条件 | 停止条件 |
|---|---:|---|---|---|
| Harness / fingerprint audit | P0 | 是 | 立即 | 无法复现 current pointwise / controls |
| Cross-encoder pairwise | P0 | 是 | Harness 完成 | recall@5 / jaccard@5 无明显提升 |
| Neural score + diversity | P1 | 是 | Cross-encoder 有基本提升 | diversity 降低 recall 或 order metrics |
| Set-aware listwise | P1 | 是 | Pairwise 过 gate 或接近 gate | listwise 不优于 cross-encoder |
| Sequential pointer selector | P2 | 条件做 | listwise order metrics 不足 | exposure bias 严重且 OPD 无改善 |
| OPD / DAgger | P2 | 条件做 | sequential baseline 可用 | teacher cost 过高或 val order/full metrics 不增 |
| GRPO | P3 | 暂不直接做 | OPD policy 过 selection-only 与 full-pipeline gate | reward hacking、val 不增、selection metrics 下降 |
| LLM listwise reranker | P3 | 可作诊断 | 需要 strong baseline | decode/index 错误或成本不可控 |

---

## 17. 推荐报告模板

每个 selector 实验报告应包含以下字段。

### 17.1 基本信息

```text
experiment_name
selector_type
base_model
train_data_path
val_data_path
chunk_mmr_fingerprint
candidate_pool_policy
filter_policy
losses
negative_sampling
max_length
batch_size
learning_rate
seed
```

### 17.2 Selection-only metrics

```text
recall@5
precision@5
jaccard@5
macro_recall@5
macro_jaccard@5
per-label recall/jaccard
top1_match
prefix_match@1 / @3 / @5
ordered_hit@5
oracle_rank_ndcg@5
pairwise_order_acc@5
overlap_pair_count
ordered_exact_match@5
```

### 17.3 Controls

```text
hybrid top5
fixed-MMR sentence
current pointwise
same set + hybrid-order
same set + candidate-pool-order
same set + random-order seeds 0-4
```

### 17.4 Full pipeline metrics

```text
learned selector evidence + oracle-direct verifier
accuracy
macro-F1
per-label F1
confusion matrix
parse error rate
unique sample_idx count
duplicate count
prompt truncation rate
fingerprint mismatch count
```

### 17.5 Error analysis

至少按以下 buckets 分析：

```text
selector hits oracle but verifier wrong
selector misses oracle and verifier wrong
selector set overlap high but order wrong
top1 wrong but set overlap high
high hybrid rank false positives
redundant false positives
label-specific failures
low-margin oracle rows
```

---

## 18. 最终建议

当前最合理的主线是：

```text
先把 evaluation harness 和 fingerprint audit 固化
-> 训练 cross-encoder pairwise reranker，作为第一个强 baseline
-> 加轻量 diversity 后处理，测试 redundancy 是否是瓶颈
-> 升级到 set-aware listwise reranker，建模候选间 interaction
-> 若 order metrics 仍不足，再做 sequential pointer selector
-> 用 OPD / DAgger 在 student-induced states 上蒸馏 margin oracle
-> 最后才考虑 KL-constrained GRPO refinement
```

其中 OPD 是当前最值得尝试的 RL-like 思路；GRPO 是后期 refinement，不是第一阶段主线。若资源有限，应优先完成 cross-encoder pairwise、set-aware listwise 和 OPD，而不是直接运行 GRPO。

