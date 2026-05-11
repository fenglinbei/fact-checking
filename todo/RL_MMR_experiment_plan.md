# 实验计划：Claim-Adaptive MMR Evidence Retrieval for Fact-Checking

> 目标：围绕事实核查任务中的 evidence retrieval，系统比较 `reranker-only`、`fixed-MMR`、`learned-λ MMR`、`RL/DPO/GRPO learned-λ MMR`、`reranker + learned-λ MMR`、`reranker + RL-MMR` 六条路线。本文档只定义思想、公式、实验结构与模块边界，具体实现由后续 agent 完成。

---

## 1. 研究目标

给定一条 claim `c` 和一个 report/chunk 候选集合 `C = {d_1, ..., d_N}`，目标是选择一个 evidence set：

\[
S_K = \{d_1^*, ..., d_K^*\}
\]

使其对事实核查 verifier 的最终判断最有用，而不仅仅是单文档相关性最高。

核心假设：

1. `reranker-only` 优化的是 pointwise/listwise relevance，但不显式优化 evidence set coverage 与 redundancy。
2. `fixed-MMR` 能减少冗余，但固定 λ 无法适应不同 claim 的 evidence need。
3. `learned-λ MMR` 能根据 claim 与候选池状态预测 query/claim-specific λ。
4. `RL/DPO/GRPO learned-λ MMR` 能进一步直接优化 set-level utility。
5. 最强路线预计是 `reranker + RL-MMR`，即 reranker 负责 relevance，RL-MMR 负责 evidence set selection。

---

## 2. 统一任务定义

输入：

- Claim: `c`
- Corpus: report 或 chunk 集合 `D`
- Candidate pool: 初检索得到的 `C_N(c) ⊂ D`
- 可选 metadata: source、time、author、domain、stance、report type
- 可选 gold evidence: `E*`
- 可选 gold verdict: `y*`

输出：

- Evidence set: `S_K`
- 可选 verdict: `ŷ = Verifier(c, S_K)`
- 可选 evidence utility score

统一 pipeline：

```text
claim c
  -> candidate retrieval: C_N(c)
  -> candidate scoring: Rel(c, d)
  -> evidence selector: S_K
  -> verifier: y_hat
  -> evaluation
```

---

## 3. 统一符号

| 符号 | 含义 |
|---|---|
| `c` | claim |
| `D` | 全部 reports/chunks |
| `C_N(c)` | 初检索候选集合 |
| `d` | 单个候选 report/chunk |
| `S_t` | 第 `t` 步之前已经选择的 evidence set |
| `K` | 最终 evidence 数量预算 |
| `Rel(c,d)` | claim-document relevance score |
| `Sim(d_i,d_j)` | document-document similarity |
| `Red(d,S_t)` | `d` 相对已选集合的冗余度 |
| `Cov(d,S_t,c)` | `d` 带来的新增 claim/aspect coverage |
| `λ` | relevance-diversity trade-off |
| `λ_t` | 第 `t` 步的动态 λ |
| `π_θ` | RL/DPO/GRPO policy |
| `R` | reward |

---

## 4. 统一 MMR 选择公式

基础 MMR：

\[
d_t = \arg\max_{d \in C \setminus S_{t-1}}
\left[
\lambda \cdot Rel(c,d)
-
(1-\lambda) \cdot Red(d,S_{t-1})
\right]
\]

常用冗余项：

\[
Red(d,S_{t-1}) = \max_{s \in S_{t-1}} Sim(d,s)
\]

可扩展为 fact-checking-aware 版本：

\[
Score(d_t) =
\lambda_t Rel(c,d)
-
(1-\lambda_t) Red(d,S_{t-1})
+ \alpha Cov(d,S_{t-1},c)
+ \beta SrcNovelty(d,S_{t-1})
+ \gamma StanceNovelty(d,S_{t-1})
\]

---

## 5. 六组主实验

### 5.1 System 1: reranker-only

目的：建立强相关性基线。

流程：

```text
C_N(c) -> reranker score Rel_rerank(c,d) -> top-K evidence
```

输出：

\[
S_K = TopK_{d \in C_N(c)} Rel_{rerank}(c,d)
\]

说明：该系统不显式建模 diversity、coverage 或 redundancy。

---

### 5.2 System 2: fixed-MMR

目的：测试固定多样性控制是否优于纯相关性选择。

流程：

```text
C_N(c) -> base relevance score -> fixed-λ MMR -> S_K
```

公式：

\[
\lambda = \lambda_0
\]

\[
d_t = \arg\max_d
\left[
\lambda_0 Rel_{base}(c,d)
-
(1-\lambda_0) Red(d,S_{t-1})
\right]
\]

说明：`λ_0` 通过 dev set 选择；所有 claims 共享同一个 λ。

---

### 5.3 System 3: learned-λ MMR, non-RL

目的：测试 claim-adaptive λ 是否优于 fixed λ。

流程：

```text
C_N(c) -> feature extractor -> λ_hat(c,C) -> MMR -> S_K
```

公式：

\[
\hat{\lambda}(c,C) = \sigma(g_\theta(F_c, F_C, F_{c,C}))
\]

\[
d_t = \arg\max_d
\left[
\hat{\lambda}(c,C) Rel_{base}(c,d)
-
(1-\hat{\lambda}(c,C)) Red(d,S_{t-1})
\right]
\]

训练思想：

1. 对每条训练 claim 网格搜索 oracle λ。
2. 以 evidence utility 最大的 λ 作为监督目标。
3. 学习从 claim/candidate features 到 λ 的映射。

Oracle λ：

\[
\lambda^*(c) = \arg\max_{\lambda \in \Lambda} Metric(S_K(c,\lambda), E^*, y^*)
\]

---

### 5.4 System 4: RL/DPO/GRPO learned-λ MMR

目的：让 λ policy 直接优化 set-level utility，而不是只拟合 oracle λ。

流程：

```text
C_N(c) -> policy πθ -> λ or λ_t -> MMR -> S_K -> reward/preference update
```

两种 action 设计：

#### A. Single-step λ policy

\[
\lambda \sim \pi_\theta(\lambda | c,C)
\]

然后运行 MMR 得到 `S_K`。

#### B. Step-wise λ policy

\[
\lambda_t \sim \pi_\theta(\lambda | c,C,S_{t-1},t)
\]

每一步重新决定 relevance-diversity trade-off。

Reward 思想：

\[
R =
w_1 R_{evidence}
+ w_2 R_{verdict}
+ w_3 R_{coverage}
+ w_4 R_{diversity}
- w_5 R_{redundancy}
- w_6 R_{cost}
\]

DPO preference 思想：

\[
S^+ \succ S^-
\]

其中 `S+` 是 reward 更高的 evidence set，`S-` 是 reward 更低的 evidence set。

GRPO 思想：同一 claim 采样多条 trajectories，使用组内相对 reward 作为 advantage。

\[
A_i = \frac{R_i - mean(R_1,...,R_G)}{std(R_1,...,R_G)}
\]

---

### 5.5 System 5: reranker + learned-λ MMR

目的：测试 learned-λ MMR 是否能在 strong reranker relevance score 上提升 evidence set utility。

流程：

```text
C_N(c) -> reranker Rel_rerank(c,d) -> λ_hat(c,C) -> MMR -> S_K
```

公式：

\[
\hat{\lambda}(c,C) = \sigma(g_\theta(F_c, F_C, F_{c,C}))
\]

\[
d_t = \arg\max_d
\left[
\hat{\lambda}(c,C) Rel_{rerank}(c,d)
-
(1-\hat{\lambda}(c,C)) Red(d,S_{t-1})
\right]
\]

说明：该系统用于验证“reranker 提供更强 relevance，learned-λ MMR 提供 set-level selection”的组合价值。

---

### 5.6 System 6: reranker + RL-MMR

目的：主方法。让 RL/DPO/GRPO policy 在 reranker relevance 的基础上优化最终 evidence set。

流程：

```text
C_N(c)
  -> reranker Rel_rerank(c,d)
  -> RL/DPO/GRPO policy πθ
  -> λ_t or evidence action
  -> MMR-like selector
  -> S_K
  -> verifier/reward
```

公式：

\[
\lambda_t \sim \pi_\theta(\lambda | c,C,S_{t-1},Rel_{rerank},M_C)
\]

\[
d_t = \arg\max_d
\left[
\lambda_t Rel_{rerank}(c,d)
-
(1-\lambda_t) Red(d,S_{t-1})
+ \alpha Cov(d,S_{t-1},c)
\right]
\]

说明：该系统是主要目标系统，预期在复杂 claim、多跳 claim、冲突证据 claim、高冗余候选池上优于 `reranker-only`。

---

## 6. 训练策略

### 6.1 Stage 0: candidate retrieval backbone

所有系统共享候选池，以保证公平比较。

```text
BM25 / dense / hybrid retrieval -> C_N(c)
```

### 6.2 Stage 1: supervised λ warm start

为 System 3/5/6 提供初始化。

\[
\lambda^*(c) = \arg\max_{\lambda \in \Lambda} Utility(S_K(c,\lambda))
\]

\[
\min_\theta \sum_c \ell(g_\theta(c,C), \lambda^*(c))
\]

### 6.3 Stage 2: preference construction

为 DPO/GRPO/RL 生成候选 evidence sets。

```text
claim c
  -> sample λ / λ_t / trajectories
  -> generate S_1, ..., S_m
  -> score each S_i
  -> construct S+ / S- pairs
```

### 6.4 Stage 3: DPO tuning

用于稳定的离线 preference optimization。

\[
\mathcal{L}_{DPO}
=
-\log \sigma
\left(
\beta [
\log \pi_\theta(S^+|c) - \log \pi_\theta(S^-|c)
-
\log \pi_{ref}(S^+|c) + \log \pi_{ref}(S^-|c)
]
\right)
\]

### 6.5 Stage 4: GRPO/PPO online refinement

用于直接优化 reward。

```text
for each claim:
  sample G evidence-selection trajectories
  evaluate reward R_i
  compute group-relative advantages
  update policy
```

---

## 7. Reward / Utility 定义

统一 utility 可由以下组件组合：

| Reward component | 含义 |
|---|---|
| `R_evidence` | 与 gold evidence 的 overlap / recall / F1 |
| `R_verdict` | verifier verdict 是否正确 |
| `R_coverage` | claim facets/subclaims 是否被覆盖 |
| `R_diversity` | source、semantic、stance、time 多样性 |
| `R_redundancy` | evidence set 内部重复度惩罚 |
| `R_cost` | evidence 数量、reranker calls、LLM tokens、latency |

建议主 reward：

\[
R =
w_1 EvidenceF1
+ w_2 VerdictCorrect
+ w_3 Coverage
+ w_4 SourceDiversity
- w_5 Redundancy
- w_6 Cost
\]

对于不同数据集，可替换为 FEVER score、FEVEROUS score、AVeriTeC score 或自定义 joint score。

---

## 8. 实验数据设置

候选数据集：

| 数据集 | 用途 |
|---|---|
| FEVER | 基础 evidence retrieval / claim verification |
| FEVEROUS | 文本 + 表格证据，复杂 evidence set |
| HoVer | 多跳事实核查 |
| SciFact | 科学 claim verification |
| AVeriTeC | 真实世界 claim + web evidence |
| 自建 report corpus | 与目标应用最一致 |

优先级：

1. 先在一个可控数据集上完成六系统比较。
2. 再扩展到复杂数据集或真实 report corpus。
3. 最终重点报告 claim complexity / redundancy / multi-hop / conflicting evidence 分桶结果。

---

## 9. 评价指标

### 9.1 Evidence retrieval metrics

- Recall@K
- Precision@K
- Evidence F1
- MRR
- nDCG
- Complete evidence set recall

### 9.2 Fact-checking metrics

- Verdict accuracy
- Macro-F1
- Joint evidence-verdict score
- FEVER score
- FEVEROUS score
- AVeriTeC score

### 9.3 Set-level utility metrics

- Redundancy rate
- Mean pairwise similarity
- Source diversity
- Stance diversity
- Aspect/subclaim coverage
- Evidence sufficiency

### 9.4 Cost metrics

- Candidate pool size `N`
- Final evidence count `K`
- Reranker calls
- LLM calls
- Tokens consumed
- Latency
- GPU memory

---

## 10. 核心对比表

| 编号 | 系统 | Relevance source | Diversity control | Learned? | RL/preference? | 主要用途 |
|---|---|---|---|---|---|---|
| 1 | reranker-only | reranker | none | yes | no | 强相关性基线 |
| 2 | fixed-MMR | base retriever | fixed λ | no | no | 固定多样性基线 |
| 3 | learned-λ MMR | base retriever | predicted λ | yes | no | 非强化 learned λ |
| 4 | RL/DPO/GRPO learned-λ MMR | base retriever | policy λ / λ_t | yes | yes | 强化/偏好版 λ |
| 5 | reranker + learned-λ MMR | reranker | predicted λ | yes | no | reranker 与 learned λ 组合 |
| 6 | reranker + RL-MMR | reranker | policy λ / λ_t | yes | yes | 主方法 |

---

## 11. 消融实验

建议至少保留以下消融：

1. `single λ` vs `step-wise λ_t`
2. `query-only features` vs `query + candidate-set features`
3. `base relevance` vs `reranker relevance`
4. `supervised λ` vs `DPO λ` vs `GRPO λ`
5. `semantic redundancy only` vs `semantic + source + stance redundancy`
6. `fixed K` vs `STOP action`
7. `evidence reward only` vs `evidence + verdict reward`
8. `without cost penalty` vs `with cost penalty`
9. `report-level selection` vs `chunk/sentence-level selection`
10. `no verifier feedback` vs `with verifier feedback`

---

## 12. Claim 分桶分析

为解释方法优势，建议对测试集按以下维度分桶：

| 维度 | 分桶思路 |
|---|---|
| Claim complexity | 简单 / 多实体 / 多条件 / 多跳 |
| Candidate redundancy | 低冗余 / 中冗余 / 高冗余 |
| Evidence type | 单证据 / 多证据 / 跨 source |
| Verdict type | supported / refuted / NEI / conflicting |
| Temporal dependence | 无时间约束 / 有时间约束 |
| Numerical dependence | 无数字 / 有数字或比较 |

预期：RL-MMR 在复杂、多跳、高冗余、跨 source、conflicting evidence 场景中收益最大。

---

## 13. 后续 agent 的模块边界

建议代码实现按以下模块拆分：

```text
Retriever
  input: claim, corpus
  output: candidate ids C_N

Reranker
  input: claim, candidate texts
  output: Rel_rerank(c,d)

SimilarityBuilder
  input: candidate texts / embeddings
  output: Sim(d_i,d_j)

FeatureExtractor
  input: claim, candidates, scores, metadata
  output: F_c, F_C, F_cC

MMRSelector
  input: candidates, relevance scores, similarity matrix, λ or λ_t
  output: selected evidence set

LambdaPredictor
  input: features
  output: λ

RLPolicy
  input: state
  output: λ, λ_t, STOP, or evidence action

RewardModel
  input: claim, selected evidence, gold evidence/verdict, verifier output
  output: reward / preference

Verifier
  input: claim, selected evidence
  output: verdict and confidence

Evaluator
  input: predictions, gold labels/evidence, costs
  output: metrics
```

---

## 14. 最小可行实验顺序

1. 完成 shared candidate retrieval。
2. 实现 `reranker-only`。
3. 实现 `fixed-MMR`。
4. 网格搜索 oracle λ。
5. 训练 `learned-λ MMR`。
6. 构造 preference pairs。
7. 训练 `DPO learned-λ MMR`。
8. 在 DPO 基础上尝试 `GRPO/PPO refinement`。
9. 接入 reranker relevance，完成 System 5 和 6。
10. 做主表、消融表、分桶分析、成本分析。

---

## 15. 预期主结论形式

目标不是证明 MMR 完全替代 reranker，而是证明：

\[
reranker + RL\text{-}MMR > reranker\text{-}only
\]

在以下方面至少部分成立：

1. Evidence Recall@K
2. Evidence set coverage
3. Redundancy reduction
4. Verdict accuracy / joint score
5. Cost-performance trade-off
6. Complex claim subset performance

