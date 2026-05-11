# 研究综述与方法整理：可学习 λ-MMR、强化学习与事实核查 Evidence Retrieval

> 目标：整理“基于 MMR 的 claim-adaptive evidence retrieval”研究路线，包括可学习 λ、reranker 对比、传统强化学习、LLM 领域强化学习与偏好优化（PPO / RLHF / DPO / GRPO 等）。本文档用于后续论文设计、实验设计和代码 agent 构建。

---

## 1. 总体判断

本研究方向是可行的，但应避免把问题表述为“用 MMR 替代 reranker”。更稳健的研究定位是：

> 在事实核查 evidence retrieval 中，学习一个 claim-adaptive / state-adaptive 的 diversity policy，使 evidence set 同时满足 relevance、coverage、low redundancy、source diversity、stance diversity 与 downstream verifier utility。

核心结论：

1. MMR 是经典 relevance-diversity reranking 方法，λ 控制相关性与新颖性之间的 trade-off。
2. 学习 query-specific trade-off 已有相关研究，尤其是 selective diversification。
3. 事实核查场景与普通检索不同，目标不是找到单篇最相关文档，而是找到能支持、反驳或判定证据不足的 evidence set。
4. Reranker 擅长 pointwise/pairwise relevance；MMR/RL-MMR 擅长 set-level selection。
5. RL/DPO/GRPO 的价值在于直接优化 evidence set utility，而不是只拟合 oracle λ 或单文档 relevance。
6. 最有潜力的最终系统是 `reranker + RL-MMR`。

---

## 2. 任务背景

给定 claim：

$$
c
$$

给定 report/chunk 候选集合：

$$
C = \{d_1, d_2, ..., d_N\}
$$

目标是选择 evidence set：

$$
S_K = \{d_1^*, d_2^*, ..., d_K^*\}
$$

使 verifier 输出正确 verdict：

$$
\hat{y} = Verifier(c, S_K)
$$

并且 evidence set 本身具备：

- 与 claim 高相关；
- 覆盖 claim 的不同子事实；
- 减少同质化重复；
- 尽量包含必要的支持/反驳/冲突信息；
- 控制检索、reranking 和 LLM 上下文成本。

事实核查 evidence retrieval 的特殊性在于：

1. 单文档高相关不等于 evidence sufficient。
2. 多个低冗余 evidence 可能比多个近重复高相关 evidence 更有价值。
3. 支持证据和反驳证据可能同时存在。
4. 对 NEI / conflicting / cherry-picking 类型 claim，需要暴露不同来源、不同角度的 evidence。
5. 下游 verifier 的 utility 比传统 IR relevance 更重要。

---

## 3. MMR 与 λ

经典 MMR 选择公式：

$$
d_t = \arg\max_{d \in C \setminus S_{t-1}}
\left[
\lambda \cdot Rel(c,d)
-
(1-\lambda) \cdot \max_{s \in S_{t-1}} Sim(d,s)
\right]
$$

其中：

- `Rel(c,d)` 表示 claim 与候选 evidence 的相关性；
- `Sim(d,s)` 表示候选 evidence 与已选 evidence 的相似度；
- `λ` 越大，越偏向 relevance；
- `λ` 越小，越偏向 diversity/novelty。

原始 MMR 工作将 query relevance 和 information novelty 结合，用于检索 reranking 和摘要选择 [R1]。

在本研究中，λ 不应只是全局超参数，而应与 claim、候选池、已选 evidence 状态相关：

$$
\lambda = \lambda(c,C)
$$

更强形式是 step-wise λ：

$$
\lambda_t = \lambda(c,C,S_{t-1},t)
$$

直觉：第一条 evidence 应更重 relevance，后续 evidence 应逐步重视 coverage、source diversity 与 redundancy reduction。

---

## 4. 从 fixed λ 到 learned λ

### 4.1 Fixed-MMR

Fixed-MMR 使用全局 λ：

$$
\lambda = \lambda_0
$$

优点：

- 简单；
- 可解释；
- 低成本；
- 可直接作为 reranker 后处理。

缺点：

- 所有 claims 共享同一个 trade-off；
- 简单 claim 和复杂 claim 使用相同 diversity strength；
- 不能根据候选池冗余程度动态调整；
- 对 conflicting evidence 或 multi-hop evidence 不够敏感。

### 4.2 Learned-λ MMR

学习一个 λ predictor：

$$
\hat{\lambda}(c,C) = \sigma(g_\theta(F_c, F_C, F_{c,C}))
$$

其中：

- `F_c`: claim features，例如长度、实体数、时间/数字、否定、比较、因果词、子 claim 数；
- `F_C`: candidate pool features，例如 score entropy、top-k score gap、候选聚类数、pairwise similarity 分布；
- `F_cC`: claim-candidate interaction features，例如 top-k relevance profile、stance distribution、source distribution。

监督信号可通过 oracle λ 构造：

$$
\lambda^*(c) = \arg\max_{\lambda \in \Lambda} Utility(S_K(c,\lambda), E^*, y^*)
$$

训练目标：

$$
\min_\theta \sum_c \ell(\hat{\lambda}(c,C), \lambda^*(c))
$$

### 4.3 相关工作：Selective Diversification

Santos 等人的 selective diversification 明确提出：并非所有 query 都同等 ambiguous，因此不同 query 应使用不同 diversification strategy；他们学习 per-query relevance-diversity trade-off，而不是对所有 query 采用同一个权重 [R2]。

这与本研究高度相关，但差异在于：

| 维度 | Selective diversification | 本研究 |
|---|---|---|
| 任务 | Web search diversification | Fact-checking evidence retrieval |
| 目标 | 覆盖 query intents / subtopics | 支撑事实核查 verdict |
| 输出 | diversified ranking | evidence set |
| 优化目标 | α-nDCG 等 diversity IR metrics | evidence recall + verifier utility + cost |
| λ 语义 | query ambiguity trade-off | claim evidence need trade-off |

### 4.4 xQuAD 与 aspect-aware diversification

xQuAD 通过 sub-queries / aspects 显式建模 query 的多个潜在方面，并选择能够覆盖未覆盖 aspects 的文档 [R3]。

对事实核查的启发：claim 可分解为多个 aspects/subclaims。例如：

```text
Claim = entity + event + time + quantity + causal relation
```

因此，可以从普通 MMR 扩展到 aspect-aware MMR：

$$
Score(d) =
\lambda Rel(c,d)
+
(1-\lambda) AspectNovelty(d,S,c)
$$

或：

$$
Score(d) =
\lambda Rel(c,d)
-
(1-\lambda) Red(d,S)
+ \alpha Cov(d,S,c)
$$

### 4.5 Learning MMR Model

Xia 等人的工作将 MMR 模型参数学习化，并直接优化 diversity evaluation measures，例如 α-NDCG [R4]。这说明“学习 MMR 参数 / 直接优化 diversity metric”在 IR 中已有先例。

本研究的差异是把 diversity objective 转换为 fact-checking utility objective。

---

## 5. 事实核查 Evidence Retrieval 相关工作

### 5.1 FEVER

FEVER 是大规模 Fact Extraction and Verification 数据集，claim 被标注为 Supported、Refuted 或 NotEnoughInfo，并要求为 Supported/Refuted claim 标注必要 evidence [R5]。

对本研究的意义：

- 提供 claim + gold evidence + verdict；
- 适合初步验证 evidence recall 和 verifier accuracy；
- 但 claim 多来自 Wikipedia 人工改写，真实世界复杂度有限。

### 5.2 FEVEROUS

FEVEROUS 扩展到非结构化文本和结构化表格/列表 evidence，评价结合 label accuracy 和 evidence retrieval [R6]。

对本研究的意义：

- 更适合测试 evidence set selection；
- 表格 + 文本混合 evidence 对 coverage 和 diversity 要求更高；
- 可用于验证多源、多类型 evidence selection。

### 5.3 HoVer

HoVer 是 many-hop fact extraction and claim verification 数据集，claim 需要从多个 Wikipedia articles 中抽取 evidence，最多涉及四跳推理 [R7]。

对本研究的意义：

- 多跳场景天然需要多样 evidence；
- reranker-only 可能倾向选出同一跳的冗余 evidence；
- RL-MMR 可建模 step-wise evidence acquisition。

### 5.4 SciFact

SciFact 面向科学 claim verification，需要从研究文献中选择 evidence-containing abstracts，并识别支持或反驳 claim 的 rationales [R8]。

对本研究的意义：

- 用于测试 domain transfer；
- 科学证据常需要多篇论文或多个 rationale 共同支撑；
- 对 source reliability 和 evidence sufficiency 要求更高。

### 5.5 AVeriTeC

AVeriTeC 是真实世界 claim verification 数据集，包含来自多个 fact-checking organizations 的真实 claims，并用 question-answer evidence 及 textual justification 支撑 verdict [R9]。AVeriTeC shared task 要求系统检索 evidence 并预测 veracity，AVeriTeC score 只有在 verdict 正确且 retrieved evidence 达到质量阈值时才认为验证正确 [R10]。

对本研究的意义：

- 最接近“从 report/web 中检索 evidence 判断 claim”的目标；
- evidence 质量与 verdict 绑定；
- 适合评估 evidence set utility，而不仅是 retrieval relevance。

---

## 6. Reranker 相关工作与对比

### 6.1 Reranker-only 的优势

BERT passage reranking 将 query 和 passage 拼接输入 Transformer cross-encoder，从而获得强 query-document interaction 表征 [R11]。

MonoT5 将 document ranking 转化为 sequence-to-sequence 任务，用 T5 生成 relevance 判断或相关 token [R12]。

在事实核查中，BERT for Evidence Retrieval and Claim Verification 使用一个 BERT 模型做 evidence sentence retrieval，另一个 BERT 模型做 claim verification，并在 FEVER 上取得强结果 [R13]。

### 6.2 Reranker-only 的限制

Reranker 的典型目标是：

$$
score(c,d)
$$

即评估单个候选 `d` 与 claim `c` 的相关性或蕴含关系。

但 evidence retrieval 的最终目标是：

$$
Utility(c,S_K)
$$

即一组 evidence 是否共同足以验证 claim。

因此，reranker-only 可能出现：

1. top-K evidence 语义近重复；
2. 只覆盖 claim 的一个子方面；
3. 忽略反方或冲突 evidence；
4. 对跨 source corroboration 建模不足；
5. 成本较高，尤其在大候选池上。

### 6.3 Reranker 与 MMR 的互补关系

合理分工：

| 组件 | 作用 |
|---|---|
| Reranker | 给出更准确的 `Rel(c,d)` |
| MMR / RL-MMR | 从候选池中选择更有用的 evidence set |
| Verifier | 根据 evidence set 判断 verdict |

因此主假设不是：

$$
RL\text{-}MMR > reranker
$$

而是：

$$
reranker + RL\text{-}MMR > reranker\text{-}only
$$

尤其在复杂、多跳、高冗余、多 source、conflicting evidence 的 claim 子集上。

---

## 7. 为什么引入强化学习

监督学习 learned-λ 的问题是：

1. λ target 来自网格搜索，是间接目标；
2. evidence set utility 往往是 non-differentiable；
3. evidence selection 是 sequential process；
4. 每一步选择会影响后续 redundancy 和 coverage；
5. 最终 reward 可能来自 verifier accuracy、human preference 或 LLM judge。

强化学习适合将 evidence retrieval 建模为序列决策：

$$
S_0 = \emptyset
$$

$$
a_t \sim \pi_\theta(a|s_t)
$$

$$
S_t = S_{t-1} \cup \{d_t\}
$$

$$
R = Utility(c,S_K,y^*)
$$

动作可以是：

1. 选择 λ；
2. 选择 step-wise λ_t；
3. 直接选择 evidence；
4. 选择 STOP；
5. 选择检索 query / subclaim / source；
6. 选择是否调用 reranker 或 LLM。

---

## 8. MDP 形式化

### 8.1 State

$$
s_t = \{c, C, S_{t-1}, Rel, Sim, Meta, Coverage_{t-1}, Cost_{t-1}, t\}
$$

其中：

- `c`: claim；
- `C`: 候选集合；
- `S_{t-1}`: 已选 evidence；
- `Rel`: relevance scores；
- `Sim`: candidate similarity matrix；
- `Meta`: source、time、domain、stance 等；
- `Coverage`: 当前子事实覆盖情况；
- `Cost`: 已消耗预算。

### 8.2 Action

可选 action space：

#### λ-action

$$
a_t = \lambda_t \in \{0,0.05,...,1.0\}
$$

然后由 MMR 规则选择下一条 evidence。

#### Evidence-action

$$
a_t = d_t \in C \setminus S_{t-1}
$$

policy 直接选择 evidence。

#### Hybrid-action

$$
a_t = (\lambda_t, d_t)
$$

或先选 λ，再从 MMR top-m 中采样 evidence。

#### STOP-action

$$
a_t \in \{select, stop\}
$$

用于 dynamic K / cost-aware retrieval。

### 8.3 Transition

$$
S_t = S_{t-1} \cup \{d_t\}
$$

并更新 coverage、redundancy、source diversity、cost、verifier confidence。

### 8.4 Reward

基础 reward：

$$
R =
w_1 R_{evidence}
+ w_2 R_{verdict}
+ w_3 R_{coverage}
+ w_4 R_{diversity}
- w_5 R_{redundancy}
- w_6 R_{cost}
$$

其中：

$$
R_{evidence} = F1(S_K,E^*)
$$

$$
R_{verdict} = \mathbb{1}[\hat{y}=y^*]
$$

$$
R_{coverage} = \frac{|CoveredSubclaims(S_K)|}{|Subclaims(c)|}
$$

$$
R_{redundancy} = \frac{1}{|S_K|^2}\sum_{i,j} Sim(d_i,d_j)
$$

$$
R_{cost} = \alpha |S_K| + \beta Calls_{reranker} + \gamma Tokens_{LLM}
$$

### 8.5 Process reward

为缓解 sparse terminal reward，可加入 step-level reward：

$$
r_t =
\Delta EvidenceCoverage_t
+
\Delta VerifierConfidence_t
-
\Delta Redundancy_t
-
\Delta Cost_t
$$

---

## 9. 可使用的 RL / Preference Optimization 路线

### 9.1 Contextual Bandit for λ

每条 claim 只做一次 λ 选择：

$$
\lambda \sim \pi_\theta(\lambda|c,C)
$$

运行 MMR 后得到 `S_K`，再计算 reward。

优点：

- 简单；
- 稳定；
- 训练成本低；
- 与 learned-λ baseline 对齐。

缺点：

- 无法逐步调整 λ；
- 无法根据已选 evidence 状态改变策略。

适合作为 RL 最小版本。

### 9.2 Sequential RL for λ_t

每一步选择一个 λ：

$$
\lambda_t \sim \pi_\theta(\lambda|c,C,S_{t-1},t)
$$

再由 MMR 选出 evidence。

优点：

- 适合多跳和多 evidence 场景；
- 可学习“前期重 relevance，后期重 diversity”的策略；
- 可加入 STOP action。

缺点：

- credit assignment 更难；
- 需要 process reward 或更强的 variance reduction。

### 9.3 PPO

PPO 是常用 policy gradient 方法，使用 clipped surrogate objective 控制 policy update 幅度 [R18]。

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

$$
L^{CLIP}(\theta)=
\mathbb{E}_t
\left[
\min
\left(
 r_t(\theta)A_t,
 clip(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

在本研究中的用法：

- policy 输出 λ_t 或 evidence action；
- reward 来自 evidence utility / verifier utility；
- 可用 supervised λ predictor 作为 warm-start policy。

### 9.4 RLHF / RLAIF

RLHF 典型流程：

1. 收集 demonstrations 做 SFT；
2. 收集人类偏好训练 reward model；
3. 使用 PPO 等 RL 方法优化 policy。

InstructGPT 使用了这种思想来对齐语言模型行为 [R19]。

迁移到 evidence retrieval：

1. 为同一 claim 生成多个 evidence sets；
2. 人类 fact-checker 或 LLM judge 比较 evidence set 质量；
3. 训练 reward model：

$$
r_\psi(c,S)
$$

4. 用 RL 优化 evidence selector。

偏好维度应包括：

- evidence 是否足以判断 claim；
- 是否覆盖关键子事实；
- 是否包含支持/反驳/冲突信息；
- 是否避免同源重复；
- source 是否可靠；
- 是否降低 verifier uncertainty。

### 9.5 DPO

DPO 通过 preference pairs 直接优化 policy，不需要显式 reward model，也不需要在线 RL 采样 [R20]。

给定：

$$
S^+ \succ S^-
$$

DPO loss：

$$
\mathcal{L}_{DPO}
=
-\log \sigma
\left(
\beta[
\log \pi_\theta(S^+|c) - \log \pi_\theta(S^-|c)
-
\log \pi_{ref}(S^+|c) + \log \pi_{ref}(S^-|c)
]
\right)
$$

在本研究中的偏好构造方式：

```text
same claim c
  -> sample multiple λ / trajectories
  -> generate evidence sets S_1 ... S_m
  -> score each evidence set
  -> choose S+ and S-
```

DPO 特别适合本研究，因为 evidence set preference 可以离线生成：

- gold evidence score；
- verifier utility；
- LLM judge；
- human preference；
- composite reward。

### 9.6 GRPO

GRPO 是 PPO 的一种变体，在 DeepSeekMath 中被用于提升数学推理能力，同时优化 PPO 的内存使用 [R21]。其基本思想是对同一 prompt 采样一组 outputs，用组内相对 reward 估计 advantage：

$$
A_i = \frac{R_i - mean(R_1,...,R_G)}{std(R_1,...,R_G)}
$$

在本研究中：

```text
same claim c
  -> sample G evidence-selection trajectories
  -> compute reward R_1 ... R_G
  -> normalize within group
  -> update policy
```

GRPO 的优势：

- 与 claim-level 多 trajectory sampling 自然匹配；
- 不一定需要单独 value model；
- 适合 LLM-style policy 或小型 neural policy。

### 9.7 LLM Agent + RL

更激进版本：让 LLM agent 执行以下动作：

```text
decompose claim
search / retrieve
select evidence
adjust λ
verify sufficiency
stop or continue
produce verdict
```

这适合多轮在线 fact-checking，但工程复杂度高。建议作为扩展，而不是第一阶段主系统。

---

## 10. 与 RL 直接相关的已有工作

### 10.1 MDP-DIV / Search Result Diversification as RL

MDP-DIV 将 search result diversification 建模为 Markov Decision Process，并用 RL 优化 diversity measures，例如 α-DCG、S-recall [R22]。

启发：evidence selection 也可以看成 sequential ranking，每一步选择一个能够带来最大长期 utility 的 evidence。

### 10.2 MA4DIV

MA4DIV 将 search result diversification 建模为 cooperative multi-agent RL，每个 document 作为 agent，直接优化 α-NDCG 等 diversity metrics [R23]。

启发：

- 传统 greedy selection 可能陷入局部最优；
- RL 可直接优化 ranking-level / set-level diversity metric；
- evidence retrieval 也可从 greedy MMR 扩展到 RL selector。

### 10.3 RL-MMR for Multi-document Summarization

RL-MMR 将 MMR 与深度强化学习结合，用于多文档摘要；其动机包括减少搜索空间和处理多文档冗余 [R24]。

启发：

- MMR 可以作为 RL 的 inductive bias；
- RL 可学习超越 fixed heuristic 的 selection strategy；
- 多文档摘要中的 redundancy 与 fact-checking evidence redundancy 有结构相似性。

### 10.4 Evidence Retrieval with Feedback for Fact Verification

FER 主张事实核查 retrieval 应从 relevance 转向 verifier utility，使用 claim verifier 的反馈来优化 evidence retrieval [R25]。

启发：

- 本研究 reward 不应只来自 IR relevance；
- verifier feedback 可以作为 evidence selector 的训练信号；
- learned λ / RL-MMR 可直接优化 downstream utility。

### 10.5 FFRR

FFRR 针对 black-box LLM fact-checking，通过 LLM 的 fine-grained feedback 构造 reward，优化 retrieval policy [R26]。

启发：

- 当 verifier/LLM 不可微时，可用反馈作为 RL reward；
- 可以对 retrieved documents 做细粒度评分；
- 适合用 LLM judge 构造 evidence reward。

### 10.6 Query Rewriting for Misinformation Discovery

该工作使用 offline RL / Decision Transformer 学习 claim query 的编辑动作，以优化 misinformation discovery 的 retrieval metrics [R27]。

启发：

- RL 可用于检索前 query rewriting；
- 本研究也可扩展为“先生成 subqueries，再用 RL-MMR 选择 evidence”；
- 可把 claim decomposition 与 MMR selection 合并成 agentic pipeline。

### 10.7 DynamicRAG

DynamicRAG 把 RAG 中的 reranker 建模为 RL agent，动态调整 retrieved documents 的顺序和数量，并使用 LLM output quality 作为 reward [R28]。

启发：

- 与本研究最接近的 RAG/reranking 思想之一；
- 支持 dynamic K；
- 支持将 reranking/evidence selection 与 downstream generation/verifier quality 对齐。

### 10.8 R3-RAG

R3-RAG 使用 RL 训练 LLM 逐步 reason 和 retrieve，reward 包含 answer correctness outcome reward 与 relevance-based document verification process reward [R29]。

启发：

- 可以借鉴 outcome + process reward 设计；
- 适合 multi-hop evidence retrieval；
- 可作为后续 agentic fact-checking 方向。

### 10.9 Veri-R1

Veri-R1 将 claim verification 设为 online setting，使 LLM 与 search engine 交互，并用 GRPO 训练；其 reward 包含 label reward、evidence reward、format reward 等 [R30]。

启发：

- GRPO 可用于 claim verification；
- evidence reward 对 faithful verification 很关键；
- 可作为本研究 RL-MMR 的 LLM-agent 扩展参照。

---

## 11. RAG / Preference Optimization 相关工作

### 11.1 DPA-RAG

DPA-RAG 提出 dual preference alignment，在 RAG 中对 reranker 与 LLM 的 knowledge preference 进行对齐 [R31]。

启发：

- reranker 输出的“相关”未必等于 LLM/verifier 需要的“有用”；
- preference data 可用于训练 reranker/selector；
- 本研究可把 evidence set preference 作为 alignment signal。

### 11.2 RPO

RPO 将 retrieval relevance awareness 纳入 RAG alignment，目标是让模型更稳健地处理外部检索知识 [R32]。

启发：

- RAG 中的 preference optimization 不应只优化生成器；
- retrieval relevance / utility 可以进入偏好目标；
- 可对 verifier 使用类似 retrieval-aware alignment。

### 11.3 BPO-RAG

BPO-RAG 提出 bi-level preference-learning，第一阶段学习选择更优 evidence sets，第二阶段对齐生成器 [R33]。

启发：

- set-level evidence preference 是合理训练目标；
- 直接对应本研究的 `DPO learned-λ MMR`；
- 支持“先优化 evidence selection，再优化 verification/generation”。

---

## 12. 本研究的六种系统路线

### 12.1 reranker-only

$$
S_K = TopK_{d \in C} Rel_{rerank}(c,d)
$$

定位：强相关性 baseline。

预期：在简单 claim 和单证据场景表现强；在高冗余或多跳 evidence 场景可能不够。

### 12.2 fixed-MMR

$$
d_t = \arg\max_d
\left[
\lambda_0 Rel_{base}(c,d)
-
(1-\lambda_0) Red(d,S_{t-1})
\right]
$$

定位：固定 diversity baseline。

预期：减少冗余，但缺乏 claim-adaptive 能力。

### 12.3 learned-λ MMR, non-RL

$$
\hat{\lambda} = \sigma(g_\theta(F_c,F_C,F_{c,C}))
$$

$$
d_t = \arg\max_d
\left[
\hat{\lambda} Rel_{base}(c,d)
-
(1-\hat{\lambda}) Red(d,S_{t-1})
\right]
$$

定位：测试 query/claim-specific λ 是否有效。

预期：优于 fixed-MMR，尤其在 claim complexity 和 candidate redundancy 差异明显时。

### 12.4 RL/DPO/GRPO learned-λ MMR

$$
\lambda_t \sim \pi_\theta(\lambda|s_t)
$$

$$
R = Utility(c,S_K,y^*)
$$

定位：将 λ learning 从 supervised regression 升级为 set-level utility optimization。

预期：优于 supervised learned-λ，尤其当 reward 包含 verifier utility 与 coverage 时。

### 12.5 reranker + learned-λ MMR

$$
d_t = \arg\max_d
\left[
\hat{\lambda} Rel_{rerank}(c,d)
-
(1-\hat{\lambda}) Red(d,S_{t-1})
\right]
$$

定位：reranker 提供强 relevance；learned λ 控制 diversity。

预期：比 reranker-only 更低冗余、更高 evidence coverage。

### 12.6 reranker + RL-MMR

$$
\lambda_t \sim \pi_\theta(\lambda|c,C,S_{t-1},Rel_{rerank},Sim,Meta)
$$

$$
d_t = \arg\max_d
\left[
\lambda_t Rel_{rerank}(c,d)
-
(1-\lambda_t) Red(d,S_{t-1})
+ \alpha Cov(d,S_{t-1},c)
\right]
$$

定位：主方法。

预期：在整体 utility、complex claim subset、redundancy reduction 和 cost-performance trade-off 上最强。

---

## 13. 方法设计建议

### 13.1 主方法名称候选

- RL-MMR for Fact-Checking Evidence Retrieval
- Claim-Adaptive MMR for Evidence Retrieval
- Reinforced Claim-Adaptive Diversity for Fact Verification
- Reranker-Guided RL-MMR Evidence Selection
- Learning Evidence Diversity Policies for Fact-Checking

### 13.2 推荐训练流程

#### Stage 1: Supervised warm start

1. 对训练 claim 做 λ grid search；
2. 用 evidence/verifier utility 得到 oracle λ；
3. 训练 λ predictor；
4. 作为 DPO/GRPO reference policy。

#### Stage 2: DPO preference tuning

1. 对每条 claim 生成多个 evidence sets；
2. 用 utility score 排序；
3. 构造 preference pairs；
4. 用 DPO 训练 λ-policy 或 selector-policy。

#### Stage 3: GRPO refinement

1. 每条 claim 采样 G 条 trajectories；
2. 计算 reward；
3. 组内标准化 advantage；
4. 更新 policy。

#### Stage 4: Reranker integration

1. 将 base relevance 替换为 reranker relevance；
2. 重训或 finetune λ-policy；
3. 比较 reranker-only、reranker + learned-λ、reranker + RL-MMR。

### 13.3 推荐 action space

第一阶段建议使用 λ-action：

$$
a_t = \lambda_t
$$

原因：

- 动作空间小；
- 与原始研究问题一致；
- 容易解释；
- 训练稳定；
- 能保留 MMR inductive bias。

第二阶段可尝试 hybrid action：

$$
a_t = (\lambda_t, d_t)
$$

或：

```text
policy selects λ_t -> MMR returns top-m candidates -> policy samples one evidence
```

### 13.4 推荐 reward priority

首选 reward：

$$
R = EvidenceF1 + VerdictCorrect + Coverage - Redundancy - Cost
$$

若 gold evidence 不完整，可加入：

- verifier confidence improvement；
- LLM judge evidence sufficiency；
- source reliability；
- human preference。

---

## 14. 相比 reranker 的潜在优势

RL-MMR / learned-λ MMR 的优势不在于更好地理解单个 report 是否 relevant，而在于选择更有用的 evidence set。

| 维度 | reranker-only | learned/RL-MMR |
|---|---|---|
| 单文档语义匹配 | 强 | 中等，取决于 relevance source |
| evidence set coverage | 不显式优化 | 显式优化 |
| redundancy control | 不显式优化 | 显式优化 |
| source diversity | 不显式优化 | 可显式优化 |
| stance/conflict exposure | 不显式优化 | 可显式优化 |
| cost-aware retrieval | 通常弱 | 可作为 reward |
| 可解释性 | 中等 | 较强，λ 与 reward 可解释 |
| 与 verifier 对齐 | 间接 | 可直接通过 reward/preference 对齐 |

最合理的主张：

$$
reranker + RL\text{-}MMR
$$

优于：

$$
reranker\text{-}only
$$

尤其在：

- 多跳 claim；
- claim 包含多个实体或多个子事实；
- 候选池高度冗余；
- source 重复严重；
- 支持与反驳 evidence 并存；
- 需要控制 LLM context budget。

---

## 15. 主要风险

### 15.1 Gold evidence 不完整

事实核查数据集通常只标注部分 evidence。未标注但有效 evidence 可能被 reward 错误惩罚。

应对：

- 加入 verifier utility reward；
- 加入 LLM judge evidence sufficiency；
- 人工抽样评估；
- 报告 gold-only 与 judge-assisted 两套指标。

### 15.2 Diversity 引入噪声

过低 λ 可能选择“不同但无用”的 evidence。

应对：

$$
Rel(c,d) > \theta
$$

即设置 relevance floor。

或者将 MMR 改为：

$$
Score(d) = Rel(c,d) + \alpha Novelty(d,S) - \beta Noise(d)
$$

### 15.3 相似 evidence 不一定无用

事实核查中，多个独立来源确认同一事实可能有 corroboration value。普通 MMR 会误惩罚这类证据。

应对：

- 区分 semantic redundancy 与 source redundancy；
- 对 same-source duplicate 强惩罚；
- 对 independent-source corroboration 弱惩罚或加分。

### 15.4 RL 训练不稳定

Sparse reward、候选空间大、credit assignment 难。

应对：

- supervised warm start；
- DPO 先行；
- λ-action 降低动作空间；
- process reward；
- group-relative advantage；
- offline preference optimization。

### 15.5 单一 λ 表达能力有限

对于复杂 claim，仅一个 λ 可能无法表达 aspect、source、stance、time 等多维 trade-off。

应对：

从 scalar λ 扩展为 multi-weight policy：

$$
\mathbf{w}_t = (w_{rel}, w_{red}, w_{cov}, w_{src}, w_{stance}, w_{cost})
$$

$$
Score(d)=
 w_{rel}Rel
-w_{red}Red
+w_{cov}Cov
+w_{src}SrcNovelty
+w_{stance}StanceNovelty
-w_{cost}Cost
$$

---

## 16. 建议论文贡献表述

可以将贡献写成：

1. 提出 claim-adaptive MMR evidence selector，将 MMR λ 从 fixed hyperparameter 改为 claim/candidate-state adaptive policy。
2. 将 fact-checking evidence retrieval 建模为 set-level utility optimization，而非 pointwise relevance ranking。
3. 引入 DPO/GRPO/RLHF-style preference/reward learning，使 evidence selector 与 verifier utility 对齐。
4. 系统比较 reranker-only、fixed-MMR、learned-λ MMR、RL/DPO/GRPO learned-λ MMR、reranker + learned-λ MMR、reranker + RL-MMR。
5. 通过复杂 claim、高冗余候选池、多跳、冲突 evidence 分桶分析，解释 diversity policy 的实际收益。

---

## 17. 推荐实验结论目标

主表期望验证：

$$
learned\text{-}\lambda\ MMR > fixed\text{-}MMR
$$

$$
RL/DPO/GRPO\ learned\text{-}\lambda\ MMR > supervised\ learned\text{-}\lambda\ MMR
$$

$$
reranker + learned\text{-}\lambda\ MMR > reranker\text{-}only
$$

$$
reranker + RL\text{-}MMR > reranker + learned\text{-}\lambda\ MMR
$$

其中最重要的是：

$$
reranker + RL\text{-}MMR > reranker\text{-}only
$$

不能只报告 aggregate score，还要报告：

- evidence recall；
- evidence set redundancy；
- subclaim/aspect coverage；
- verdict accuracy；
- joint evidence-verdict score；
- cost；
- claim complexity buckets。

---

## 18. 相关文献索引

### MMR、多样化检索、可学习 λ

[R1] Carbonell, J. and Goldstein, J. 1998. *The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries*.  
https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf

[R2] Santos, R. L. T., Macdonald, C., and Ounis, I. 2010. *Selectively Diversifying Web Search Results*. CIKM.  
https://eprints.gla.ac.uk/44426/  
https://terrierteam.dcs.gla.ac.uk/publications/santos2010cikm.pdf

[R3] Santos, R. L. T., Peng, J., Macdonald, C., and Ounis, I. 2010. *Explicit Search Result Diversification through Sub-queries*. ECIR.  
https://link.springer.com/chapter/10.1007/978-3-642-12275-0_11

[R4] Xia, L., Xu, J., Lan, Y., Guo, J., and Cheng, X. 2015. *Learning Maximal Marginal Relevance Model via Directly Optimizing Diversity Evaluation Measures*. SIGIR.  
https://dl.acm.org/doi/10.1145/2766462.2767710

[R5a] Khan, S. H. et al. 2026. *DF-RAG: Query-Aware Diversity for Retrieval-Augmented Generation*. Findings of EACL / arXiv.  
https://aclanthology.org/2026.findings-eacl.150/  
https://arxiv.org/abs/2601.17212

### 事实核查数据集与 evidence retrieval

[R5] Thorne, J. et al. 2018. *FEVER: a Large-scale Dataset for Fact Extraction and VERification*. NAACL.  
https://aclanthology.org/N18-1074/

[R6] Aly, R. et al. 2021. *FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured information*.  
https://aclanthology.org/2021.fever-1.1/

[R7] Jiang, Y. et al. 2020. *HoVer: A Dataset for Many-Hop Fact Extraction And Claim Verification*. Findings of EMNLP.  
https://aclanthology.org/2020.findings-emnlp.309/

[R8] Wadden, D. et al. 2020. *Fact or Fiction: Verifying Scientific Claims*. EMNLP.  
https://aclanthology.org/2020.emnlp-main.609/

[R9] Schlichtkrull, M., Guo, Z., and Vlachos, A. 2023. *AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web*.  
https://arxiv.org/abs/2305.13117

[R10] Schlichtkrull, M. et al. 2024. *The Automated Verification of Textual Claims (AVeriTeC) Shared Task*.  
https://aclanthology.org/2024.fever-1.1/  
https://arxiv.org/abs/2410.23850

### Reranker 与事实核查 reranking

[R11] Nogueira, R. and Cho, K. 2019. *Passage Re-ranking with BERT*.  
https://arxiv.org/abs/1901.04085

[R12] Nogueira, R. et al. 2020. *Document Ranking with a Pretrained Sequence-to-Sequence Model*.  
https://arxiv.org/abs/2003.06713

[R13] Soleimani, A., Monz, C., and Worring, M. 2020. *BERT for Evidence Retrieval and Claim Verification*.  
https://arxiv.org/abs/1910.02655  
https://pmc.ncbi.nlm.nih.gov/articles/PMC7148011/

[R14] Malviya, S. et al. 2024. *Evidence Retrieval for Fact Verification using Multi-stage ReRanking*. Findings of EMNLP.  
https://aclanthology.org/2024.findings-emnlp.428/

### RL / feedback / agentic retrieval for fact-checking and RAG

[R18] Schulman, J. et al. 2017. *Proximal Policy Optimization Algorithms*.  
https://arxiv.org/abs/1707.06347

[R19] Ouyang, L. et al. 2022. *Training Language Models to Follow Instructions with Human Feedback*.  
https://arxiv.org/abs/2203.02155

[R20] Rafailov, R. et al. 2023. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*.  
https://arxiv.org/abs/2305.18290

[R21] Shao, Z. et al. 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*.  
https://arxiv.org/abs/2402.03300

[R22] Xia, L. et al. 2017. *Adapting Markov Decision Process for Search Result Diversification*.  
https://jiafengguo.github.io/2017/2017-Adapting%20Markov%20Decision%20Process%20for%20Search%20Result%20Diversification.pdf

[R23] Chen, Y. et al. 2024/2025. *MA4DIV: Multi-Agent Reinforcement Learning for Search Result Diversification*.  
https://arxiv.org/abs/2403.17421  
https://openreview.net/forum?id=qvq7g5S7Hf

[R24] Mao, Y. et al. 2020. *Multi-document Summarization with Maximal Marginal Relevance-guided Reinforcement Learning*. EMNLP.  
https://aclanthology.org/2020.emnlp-main.136/  
https://arxiv.org/abs/2010.00117

[R25] Zhang, H. et al. 2023. *From Relevance to Utility: Evidence Retrieval with Feedback for Fact Verification*. Findings of EMNLP.  
https://aclanthology.org/2023.findings-emnlp.422/  
https://arxiv.org/abs/2310.11675

[R26] Zhang, X. and Gao, W. 2024. *Reinforcement Retrieval Leveraging Fine-grained Feedback for Fact Checking News Claims with Black-Box LLM*. LREC-COLING.  
https://arxiv.org/abs/2404.17283  
https://aclanthology.org/2024.lrec-main.1209.pdf

[R27] Kazemi, A. et al. 2023. *Query Rewriting for Effective Misinformation Discovery*.  
https://arxiv.org/abs/2210.07467

[R28] Sun, J. et al. 2025. *DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation*.  
https://arxiv.org/abs/2505.07233  
https://openreview.net/forum?id=NuCtKoflsV

[R29] Li, Y. et al. 2025. *R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning*.  
https://arxiv.org/abs/2505.23794  
https://aclanthology.org/2025.findings-emnlp.554.pdf

[R30] He, Q. et al. 2025. *Veri-R1: Toward Precise and Faithful Claim Verification via Online Reinforcement Learning*.  
https://arxiv.org/abs/2510.01932

### Preference optimization for RAG

[R31] Dong, G. et al. 2024/2025. *Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation*.  
https://arxiv.org/abs/2406.18676  
https://openreview.net/forum?id=2ZaqnRIUCV

[R32] Yan, S.-Q. and Ling, Z.-H. 2025. *RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation*. ACL.  
https://arxiv.org/abs/2501.13726  
https://aclanthology.org/2025.acl-long.261/

[R33] Cao, S. 2026. *Bi-Level Preference Optimization for Retrieval-Augmented Generation*. AAAI Student Abstract.  
https://ojs.aaai.org/index.php/AAAI/article/view/42194

---

## 19. 最终建议

建议将研究问题正式定义为：

> Learning Claim-Adaptive Evidence Diversity Policies for Fact-Checking

而不是单纯的：

> Learning λ for MMR

核心方法建议：

$$
Candidate Retrieval
\rightarrow
Reranker
\rightarrow
RL/DPO/GRPO\text{-}trained\ Adaptive\ MMR
\rightarrow
Verifier
$$

核心实证目标：

$$
reranker + RL\text{-}MMR > reranker\text{-}only
$$

核心解释目标：

> 当 claim 复杂、候选池冗余、证据跨 source、存在冲突信息或需要多跳推理时，set-level diversity policy 比 pointwise relevance ranking 更符合事实核查 evidence retrieval 的真实需求。

