# 新 Selector 模型调研 Brief

文档更新时间：2026-05-19 21:48 CST

本文面向没有本项目上下文的同学，说明当前需要调研的新 selector 模型到底要解决什么问题、训练数据来自哪里、候选池口径是什么、oracle 信息如何解释，以及什么实验结果才算有效。

## 一句话任务定义

给定一条 fact-checking claim 和固定候选池中的最多 15 条 sentence-level evidence candidates，训练一个可部署的 selector，从候选池中输出最多 5 条有序 evidence list，使下游 verifier 的输入尽量接近 Stage2 oracle evidence set 与 oracle greedy order。

形式化地说：

```text
input:  claim c, candidate pool C = {d_1, ..., d_N}, N <= 15
output: ordered evidence list L_5, |L_5| <= 5
goal:   L_5 should approximate oracle selected_indices and oracle greedy order,
        then improve verifier accuracy/F1
```

当前不要继续调研“预测 MMR scalar lambda”的方法；该路线已多轮失败。新方向是 evidence set selector / reranker / listwise set selection。

## 为什么需要新 Selector

早期 fixed-MMR 和 pointwise selector 都没有充分利用 oracle evidence supervision。

关键现象：

| 条件 | val accuracy | val macro-F1 | 说明 |
|---|---:|---:|---|
| oracle evidence + oracle-direct verifier | 0.7111 | 0.7169 | gold-conditioned upper-bound |
| Stage2 oracle under Stage1 verifier | 0.6593 | 0.6620 | oracle search 原始 verifier 口径 |
| fixed-MMR sentence + oracle-direct verifier | 0.2716 | 0.2663 | 非 oracle evidence 对照 |
| pointwise sentence + oracle-direct verifier | 0.2637 | 0.2596 | 非 oracle evidence 对照 |
| fixed-MMR sentence baseline verifier | 0.2951 | 0.2981 | 先前 val baseline |

解释：

1. verifier 能利用 oracle evidence：direct verifier 在 oracle evidence 上达到 0.7111 / 0.7169。
2. verifier 换成 fixed-MMR 或当前 pointwise evidence 后立即回到 0.26-0.27。
3. 当前 pointwise selector 的 selection-only 指标太弱：

```text
recall@5  = 0.3755
jaccard@5 = 0.2536
```

所以当前瓶颈不是 verifier、decode、checkpoint 或 prompt budget，而是 selector 没有选到接近 oracle evidence distribution 的证据。

补充的 order-sensitivity 诊断显示：即使固定同一组 oracle evidence，只改变 evidence 在 prompt 中的顺序，也会显著改变 verifier 输出。当前 API inference 口径下：

| oracle-selected evidence order | val accuracy | val macro-F1 | 说明 |
|---|---:|---:|---|
| oracle greedy order | 0.6327 | 0.6430 | Stage2 oracle search 保存的选择顺序 |
| hybrid / candidate_pool order | 0.4639 | 0.4721 | 同一组 evidence，按 retrieval / pool 顺序重排 |
| random order mean, seed0-4 | 0.4739 | 0.4829 | 同一组 evidence，随机重排均值 |

这张表只用于同一 API inference 口径内比较顺序影响，不替代 train-time label-token eval 的 `0.7111 / 0.7169` upper-bound。

因此当前主线不能把 oracle evidence 简化成无序集合。新 selector 需要同时学习“选哪些 evidence”和“以什么顺序给 verifier”。

## 必须固定的候选池口径

后续 selector 的训练、selection-only eval、build pipeline 必须共享同一套候选池口径。

```text
chunk_mmr_fingerprint = 432dfc970e75
chunking.strategy     = sentence
candidate pool        = dedup -> hybrid top15
selector output       = top5
```

这几个约束是硬前提：

1. 不要使用 semantic chunking 作为主线。
2. 不要使用旧 V1 reconstructed / positive-injected candidate pool。
3. 不要在 selection-only eval 中额外注入 oracle positives。
4. 不要混用其他 chunk cache fingerprint。
5. 训练、评估、pipeline build 都要能检查并拒绝 fingerprint mismatch。

### Chunking

当前主线使用 sentence-level chunk：

```text
build.retrieval.chunking.strategy = sentence
```

每个 evidence candidate 是原 report 中的单句或句级 chunk。配置里可能仍出现 `context_k=1`、`theta=0.7` 等字段，但在 sentence strategy 下核心语义是单句候选。

semantic-level Stage2 oracle 已做过 paired 对照，明显弱于 sentence-level：

```text
paired_n      = 1720
sentence_acc  = 0.6192
semantic_acc  = 0.5407
sentence_only = 247
semantic_only = 112
```

因此新 selector 调研默认只围绕 sentence-level 候选池展开。

### Hybrid Scoring 与候选池

候选池先由正式 build pipeline 产生，不由 selector 自己重新检索。

Hybrid score 由三部分组成：

| score | 权重 | 含义 |
|---|---:|---|
| dense | 0.70 | BGE embedding 相似度 |
| lexical | 0.20 | claim 与 candidate 的内容词 overlap F1 |
| bm25 | 0.10 | 简化 BM25 lexical score |

流程：

```text
raw report sentences
-> sentence chunks
-> embedding + lexical/BM25 scoring
-> candidate dedup
-> hybrid_score descending top15
-> selector chooses top5
```

Stage2 oracle result 中保存的 `candidate_pool` 就是 selector 应该面对的候选池。`selected_indices` 是进入该 `candidate_pool` 后的坐标。

## Oracle 信息如何解释

当前可用的 oracle supervision 来自 Stage2 margin oracle search。

主要文件：

```text
train:
outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl

val:
outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl
```

每条 oracle row 至少要关注：

| 字段 | 含义 |
|---|---|
| `event_id` | 与 LIAR-RAW 样本对齐的稳定 ID |
| `claim` | 待核查声明 |
| `gold_label` / `label` | 真实 6 类标签 |
| `candidate_pool` | dedup -> hybrid top15 后的候选证据 |
| `selected_indices` | oracle 选中的 candidate_pool 坐标 |
| `search_objective` | 当前主线为 `margin` |
| `gold_logprob` | 最终 evidence set 下 gold label token logprob |
| `best_wrong_logprob` | 最强错误 label token logprob |
| `margin` | `gold_logprob - best_wrong_logprob` |
| `pred_label` | oracle verifier 在该 evidence set 下的 argmax label |
| `is_correct` | `pred_label == gold_label` |
| `candidate_pool_metadata.chunk_mmr_fingerprint` | 必须为 `432dfc970e75` |

oracle search 的目标不是人工标注“哪句话客观上是证据”，而是寻找能让当前 verifier 对 gold label margin 最大化的 evidence set：

```text
margin = log P(y_gold | claim, S_5) - max_{y != y_gold} log P(y | claim, S_5)
```

因此这些标签应被理解为 utility supervision，而不是人工 evidence relevance label。

## 推荐的数据构造方式

### 基础训练样本

对每个 claim，使用同一个 `candidate_pool` 内的 selected vs non-selected 关系构造监督：

```text
positives = candidate_pool[selected_indices]
negatives = candidate_pool indices not in selected_indices
```

建议优先构造 pairwise / listwise 数据，而不是独立 pointwise binary classification。

原因：

1. 每个 claim 内候选的相对排序比跨 claim 绝对分数更可靠。
2. selector 最终要选 top5，不只是判断单条 evidence 是否相关。
3. evidence 之间存在 coverage / redundancy / order 交互，独立 pointwise 很难表达。

### 可选过滤

主实验建议先用 all rows，与当前 oracle-direct verifier 构造逻辑保持一致。

可做 ablation：

| filter | 目的 | 风险 |
|---|---|---|
| all | 最大覆盖，保持主线口径 | 包含 oracle 本身未使 verifier 正确的样本 |
| `is_correct == true` | 只学 verifier 真正受益的 oracle set | 样本减少，可能偏向容易样本 |
| `margin > 0` | 只学 gold margin 为正的 set | 仍可能丢掉困难样本 |
| high margin | 提升标签置信度 | 容易过拟合少数高置信样本 |

任何过滤都必须在 report 里写明，并且 train/val 不得混用。

## Selector 输入边界

部署时 selector 可以使用：

| 信息 | 是否可用 | 说明 |
|---|---|---|
| claim text | 可用 | 必需 |
| candidate text | 可用 | 必需 |
| source metadata | 可用 | 如 report_id/domain/link/sent_idx |
| hybrid/dense/lexical/bm25 score | 可用 | 来自正式 retrieval |
| candidate rank | 可用 | top15 内位置 |
| BGE candidate embedding | 可用 | 来自 chunk cache |
| pairwise candidate similarity | 可用 | 可用于 diversity / redundancy |
| gold label | 不可用 | 只可用于训练分析，不可用于推理 |
| oracle margin / oracle selected flag | 不可用 | 训练标签，不可作为推理特征 |
| oracle verifier label_logprobs | 不可用 | 推理时没有 gold-conditioned oracle scoring |

如果调研 label-aware selector，只能使用可部署预测信号，例如一个 verifier 的 predicted label distribution；不能使用 gold label。

## 候选模型调研方向

### 1. Cross-encoder reranker

输入：

```text
[claim, candidate_text]
```

输出单候选分数，再取 top5。训练可以使用 pairwise ranking loss：

```text
score(claim, positive) > score(claim, negative)
```

优点：

1. 实现简单，替代当前 logreg pointwise 的第一候选。
2. 能建模 claim-candidate token-level interaction。
3. 推理成本为每 claim 15 次 cross-encoder forward，可接受。

不足：

1. 默认仍是逐候选打分，不直接建模 redundancy。
2. 需要额外机制处理 top5 内 diversity。

### 2. Listwise reranker

输入：

```text
claim + all 15 candidate texts
```

输出 15 个候选的排序或有序 top5。训练目标可以是 ListNet / ListMLE / soft target distribution / selected mask。

优点：

1. 直接看到同一 claim 下的候选集合。
2. 更适合 top5 selection。
3. 可以学习候选间的相对重要性。

不足：

1. 实现复杂度高于 cross-encoder。
2. 需要控制 15 条候选拼接后的 token budget。

### 3. Pairwise preference reranker

输入 pair：

```text
(claim, candidate_positive, candidate_negative)
```

训练目标：

```text
positive candidate should outrank negative candidate
```

优点：

1. 与 oracle selected_indices 的相对监督最匹配。
2. 对每个 claim 可产生多组训练 pair。
3. 可以用 margin ranking / contrastive loss。

不足：

1. 推理时仍需转成单候选 score 或排序策略。
2. pair 采样策略会影响训练稳定性。

### 4. Set-aware / sequential selector

按步选择：

```text
S_0 = empty
for t in 1..5:
    score each remaining candidate conditioned on claim and S_{t-1}
    select next candidate
```

优点：

1. 直接建模 coverage 与 redundancy。
2. 更接近 oracle greedy top5 的生成过程。
3. 可以融合 MMR-style features 和 neural score。

不足：

1. 训练与推理代码复杂。
2. 需要处理 exposure bias：训练时条件在 oracle prefix 上，推理时条件在模型已选集合上。

### 5. Hybrid neural score + MMR

先用 neural reranker 得到 relevance score，再用 MMR-like diversity penalty 选 top5：

```text
score = neural_relevance - beta * redundancy_to_selected
```

优点：

1. 比纯 neural top5 更稳。
2. 保留当前 retrieval pipeline 的可解释性。
3. 适合从 cross-encoder reranker 平滑升级。

不足：

1. beta / diversity 仍需调参。
2. 如果 oracle selected set 不强调 diversity，可能引入额外偏差。

## Selection-only Gate

任何新 selector 先跑 selection-only eval，再跑 full pipeline。这里的 selector 输出必须被视为 **有序 top5 list**，不能只当成无序集合。

评估口径必须是：

```text
candidate pool = dedup -> hybrid top15
gold list      = oracle selected_indices in Stage2 greedy order
prediction     = selector ordered top5
set metrics    = recall@5, precision@5, jaccard@5, macro by label
order metrics  = top1_match, prefix_match@k, oracle_rank_ndcg@5,
                 pairwise_order_acc@5, ordered_exact_match@5
```

当前 pointwise baseline：

```text
recall@5        = 0.3755
jaccard@5       = 0.2536
macro_recall@5  = 0.3780
macro_jaccard@5 = 0.2555
```

当前 pointwise baseline 还需要补齐排序指标；以后任何 selection-only baseline 都必须同时记录 set-level 和 order-level 指标。

但这些 set-level 指标无法区分“选中同一批 evidence 但顺序错误”的情况。由于 order-sensitivity 实验已经证明 oracle-direct verifier 对顺序敏感，后续 selection-only 报告必须额外包含以下排序指标：

| metric | 定义 | 捕捉的问题 |
|---|---|---|
| `top1_match` | `prediction[0] == gold_list[0]` 的比例 | oracle 第一条证据是否被放在最前 |
| `prefix_match@k` | 前 k 个位置逐位相同的比例，可报告 `k=1/3/5` | 是否恢复 oracle greedy prefix |
| `ordered_hit@5` | `sum_i 1[prediction[i] == gold_list[i]] / min(5, len(gold_list))` | 位置级恢复，不只看集合重合 |
| `oracle_rank_ndcg@5` | 用 oracle rank 生成 graded relevance，例如对 `i=0..4` 设 `rel(gold_list[i]) = 5 - i`，对 selector 排序算 NDCG@5 | 高价值 oracle 证据是否排在前面 |
| `pairwise_order_acc@5` | 对同时出现在 prediction 和 gold_list 中的 oracle evidence pair，统计相对顺序一致比例，并同时报告有效 pair 数 | 集合选对后，内部相对顺序是否正确 |
| `ordered_exact_match@5` | ordered prediction 与 gold_list 完全一致的比例 | 严格恢复完整 oracle order |

实现时要注意：

1. `pairwise_order_acc@5` 必须和 `overlap_pair_count` 一起报告；如果一个样本只重合 0-1 条 evidence，pairwise order 没有意义。
2. `oracle_rank_ndcg@5` 不能把所有 oracle evidence 都设成同一 relevance，否则它会退化成普通 positive NDCG，仍然捕捉不到 oracle greedy order。
3. hybrid-order / candidate_pool-order / random-order control 应在同一个 predicted set 上重排后计算，用来判断收益是否真的来自 selector 的输出顺序。
4. selection-only trace 中必须保存 `selector_ordered_indices`、`oracle_ordered_indices`、`selector_scores`，不要在 eval 前按 hybrid score 或 candidate_pool index 重排。

建议初始 gate：

```text
must beat current pointwise by a clear margin
target recall@5          >= 0.50
target jaccard@5         >= 0.35
target oracle_rank_ndcg@5 improves over current pointwise and over hybrid-order control
target pairwise_order_acc@5 improves over current pointwise, with enough overlap_pair_count support
target top1_match        improves over current pointwise
```

如果新 selector 连 selection-only set metrics 都不能明显超过当前 pointwise，不应进入完整 verifier pipeline。若 set metrics 过关但 ordering metrics 接近 hybrid / random order control，也不应直接进入主线 full pipeline；应先修正训练目标或后处理，确认 selector 输出顺序能接近 oracle greedy order。

## Full Pipeline Gate

Selection-only 过关后，再跑：

```text
learned selector evidence + oracle-direct verifier
```

对照指标：

| condition | val accuracy | val macro-F1 |
|---|---:|---:|
| oracle evidence + oracle-direct verifier | 0.7111 | 0.7169 |
| fixed-MMR sentence + oracle-direct verifier | 0.2716 | 0.2663 |
| current pointwise sentence + oracle-direct verifier | 0.2637 | 0.2596 |
| fixed-MMR sentence baseline verifier | 0.2951 | 0.2981 |

Go / No-Go：

1. 若 selector + oracle-direct verifier 仍在 0.26-0.28，说明 selector 没有解决 evidence distribution gap。
2. 若超过 fixed-MMR evidence + oracle-direct verifier，但仍低于 baseline verifier，需要判断是否是 verifier 对非 oracle evidence 分布不稳。
3. 只有 val 明确超过 fixed-MMR baseline 后，才考虑 test。

## 不要重复的旧错误

### 不要把 V1 selection-only gate 当强依据

V1 selection-only gate 已降级为无效或弱参考：

1. 候选池不是正式 pipeline pool。
2. 构造时注入了 oracle positives。
3. cache / chunking 与当前主线不一致。

后续只能使用 Stage2 oracle 保存的真实 `candidate_pool`。

### 不要换成 semantic chunking 主线

semantic-level oracle paired subset 明显弱于 sentence-level。除非是专门做 granularity diagnostic，否则新 selector 不应使用 `e0b01520364d` 作为主线 cache。

### 不要用 gold-conditioned 特征做推理

`selected_indices`、`margin`、`gold_logprob`、`is_correct`、`gold_label` 都是训练或分析信息，不能作为部署时 selector 输入。

### 不要先跑 test

当前所有 selector 选择和阈值都应在 val 上完成。test 只用于最终确认，不用于调参。

## 研究交付物建议

调研或实现一个 selector 方案时，建议至少交付：

1. 模型方案说明：pointwise / pairwise / listwise / sequential，输入输出和推理成本。
2. 数据构造说明：正负样本、pair 采样、过滤规则、是否使用 all oracle rows。
3. 训练配置：base model、loss、batch size、max length、负例比例。
4. selection-only 报告：recall@5、jaccard@5、macro by label、top1_match、oracle_rank_ndcg@5、pairwise_order_acc@5、ordered_exact_match@5，并与 current pointwise / hybrid-order / random-order control 对照。
5. full pipeline 报告：至少 val split，先不跑 test。
6. trace 文件：每条 claim 的 candidate pool、selector score、selector ordered indices、oracle ordered indices、set overlap、rank-aware order metrics。
7. fingerprint 审计：明确 `chunk_mmr_fingerprint=432dfc970e75`。

## 推荐的第一轮实验顺序

1. Cross-encoder pairwise reranker：最小替代当前 logreg pointwise。
2. Cross-encoder score + MMR diversity：在 reranker relevance 上加入 set-level diversity。
3. Listwise 15-candidate reranker：如果 pairwise 明显提升但 full pipeline 仍不够。
4. Sequential selector：如果 listwise 仍无法处理 coverage / redundancy / order。

第一轮不建议直接上复杂 RL / GRPO。当前最需要的是先证明 oracle selected evidence pattern 能在正式候选池口径下被一个 stronger selector 学到。
