# 面向 AAAI 投稿的 Fact-Checking 方法整理与拔高方案

## 0. 核心判断

当前方法已经具备较完整的两阶段 fact-checking pipeline：先用结构化 selector 从 claim-associated reports 中选出 evidence chain，再用 label-token verifier 输出六分类标签。方法中的高潜力部分不是最终的 Qwen verifier，也不是 hybrid retrieval，而是如下链条：

```text
claim atoms → evidence map → heterogeneous evidence graph → adaptive evidence-chain selection
```

如果保持当前 v0.6c 形态，方法容易被审稿人理解为“结构化 RAG + 启发式 evidence selector + LLM verifier”的工程 pipeline。其主要风险在于：selector 目前更像规则 cascade，而不是一个具有清晰优化目标、可学习机制或理论性质的算法。

因此，建议将论文主贡献从“设计了一个 rule-step adaptive selector”拔高为：

> 面向细粒度事实核查的最小充分证据链选择问题，并提出 atom-aware、graph-aware、relation-aware、conflict-aware 的 budgeted evidence-chain optimization 方法。

一个更适合作为论文主方法的表述是：

```text
Minimum Sufficient Evidence Chain Selection for Fine-Grained Fact Verification
```

或：

```text
Atom-Graph Evidence Chain Selection for Fine-Grained Fact Verification
```

---

## 1. 当前方法概括

### 1.1 任务定义

给定一个样本：

```text
x = (claim c, report collection R)
```

目标是预测六分类 veracity label：

$$\hat{y} \in \{\text{pants-fire}, \text{false}, \text{barely-true}, \text{half-true}, \text{mostly-true}, \text{true}\}$$

当前 verifier 使用 label-token 形式输出 A-F：

```text
A -> pants-fire
B -> false
C -> barely-true
D -> half-true
E -> mostly-true
F -> true
```

### 1.2 当前 pipeline

从完整端到端主流程看，当前方法可以简写为：

$$C_{\mathrm{claim}} = \mathrm{ClaimMMR}(\mathrm{HybridRetrieve}(c, R))$$

$$Q = \mathrm{QuestionDecompose}(c)$$

$$C_{\mathrm{qd}} = \mathrm{QDRetrieve}(Q, R)$$

$$C = \mathrm{UnionRank}(C_{\mathrm{claim}}, C_{\mathrm{qd}})$$

$$M = \mathrm{EvidenceMap}(c, C)$$

$$G = \mathrm{BuildGraph}(c, M)$$

$$E^* = \mathrm{RuleStepSelect}(G; \mathrm{min}=5, \mathrm{max}=10)$$

$$\hat{y} = \mathrm{Verifier}(c, E^*)$$

整体方法公式为：

$$\hat{y} = \mathrm{Verifier}\left(c, \mathrm{RuleStepSelect}\left(\mathrm{BuildGraph}\left(c, \mathrm{EvidenceMap}\left(c, \mathrm{UnionRank}\left(\mathrm{ClaimMMR}\left(\mathrm{HybridRetrieve}(c,R)\right), \mathrm{QDRetrieve}\left(\mathrm{QuestionDecompose}(c),R\right)\right)\right)\right)\right)\right)$$

### 1.3 Candidate Evidence Retrieval

系统先将 report collection 切分为候选 evidence units，例如 sentence 或 semantic chunk。每个候选 evidence 与 claim 计算三类相关性：

```text
dense similarity
lexical overlap
BM25-like score
```

三者加权得到 hybrid relevance score：

$$s(e, c) = 0.70 \cdot s_{\mathrm{dense}}(e, c) + 0.20 \cdot s_{\mathrm{lexical}}(e, c) + 0.10 \cdot s_{\mathrm{BM25}}(e, c)$$

随后使用 MMR 控制相关性与多样性，得到 claim-level baseline candidate pool。需要注意：在完整 v0.6c 端到端流程中，这个 baseline pool 不会直接进入 evidence-map stage，而是会先与 question decomposition retrieval 产生的候选池做 union，再把 union 后的候选送入 evidence-map。

### 1.4 Question Decomposition Retrieval and Union

在 claim-level hybrid retrieval 之后，系统还会运行 question decomposition (QD) 扩展检索。该步骤将原始 claim 分解为若干面向核查的子问题：

$$Q = \{q_1, q_2, \ldots, q_m\}$$

每个子问题会在同一批 report evidence units 上单独执行检索，并沿用与 claim-level retrieval 相同的三路 hybrid relevance recipe：

$$s(e, q_i) = 0.70 \cdot s_{\mathrm{dense}}(e, q_i) + 0.20 \cdot s_{\mathrm{lexical}}(e, q_i) + 0.10 \cdot s_{\mathrm{BM25}}(e, q_i)$$

每个 question route 默认保留若干高分候选，再通过 RRF-style route merging 合并为 QD candidate pool。合并时会考虑：

```text
question route rank
max question hybrid score
question hit count
question focus
```

随后系统将两类候选做 union：

```text
baseline claim-MMR candidates
QD merged candidate pool
```

union 阶段会对重复 evidence 做 canonical text 去重，并为每条候选记录来源特征，例如：

```text
from_baseline
baseline_rank
baseline_hybrid_score
from_qd
qd_pool_rank
qd_rrf_score
qd_question_hit_count
qd_max_question_hybrid
union_pool_rank
```

最终 evidence-map stage 消费的是 `union_candidate_pool_<split>.jsonl`，并按 `union_pool_rank` 取 `candidate_top_n=20` 作为 map annotation 的候选证据池。因此，完整流程应理解为：

```text
HybridRetrieve + ClaimMMR
    +
QuestionDecompose + QDRetrieve + route merge
    ->
Union candidate pool
    ->
Evidence Map Construction
```

### 1.5 Evidence Map Construction

v0.6b evidence-map stage 消费 QD-union 后的 top-20 candidate evidence pool，并将 claim 分解为 atomic facts：

$$A = \{A_1, A_2, \ldots, A_m\}$$

每个 atom 包含：

```text
text
type
importance
```

每条 evidence 会被标注如下结构化字段：

```text
covered_atom_ids
relation: support / refute / qualify / mixed / background / irrelevant
directness: direct / partial / context / none
evidence_role
key_spans
duplicate_group
confidence
```

这些字段不直接作为最终 label，而是作为 v0.6c selector 的结构化输入。这里的 atom decomposition 与上一节的 question decomposition 不是同一个步骤：QD 是 map 之前的检索扩展；atom decomposition 是 evidence-map stage 内部用于建立 claim-evidence alignment 的结构化标注。

### 1.6 Map Feature Scoring

系统将 evidence-map annotation 后处理为候选 evidence 特征。每条 evidence 得到：

```text
atom_coverage_score
map_relation
map_directness
directness_score
polar_relation_score
background_penalty_score
evidence_map_quality_score
```

其中 `evidence_map_quality_score` 综合 atom coverage、directness、stance polarity、confidence、key span，并惩罚 background / irrelevant evidence。

### 1.7 Evidence-Chain Graph Construction

v0.6c 将 claim、claim atoms 和 evidence candidates 构造成异构图：

```text
claim node: C0
atom nodes: A1 ... Am
evidence nodes: E1 ... En
```

图中的边包括：

```text
claim_has_atom
evidence_covers_atom
duplicate
same_source_context
complements
corroborates
tension
bridge_context
```

该设计的核心思想是：不只看单条 evidence 的 retrieval score，而是建模 evidence 之间是否互补、相互印证、存在张力，或者是否只是背景上下文。

### 1.8 Rule-Step Adaptive Evidence Selection

当前 selector 名称为：

```text
v0_6c_rule_step_adaptive5_10
```

默认参数为：

```text
candidate_top_n = 20
min_top_k = 5
max_top_k = 10
```

当前选择过程如下：

```text
anchor_core:
    首先选择直接或部分直接、覆盖 claim atom、且 relation 属于 support/refute/qualify/mixed 的核心 evidence。

P1_new_atom_core:
    优先选择能覆盖尚未覆盖 claim atom 的核心 evidence。

P2_strong_edge_core:
    若候选 evidence 与已选核心 evidence 存在 complements / corroborates / tension，则作为强关系证据加入。

P3_bridge_context:
    选择与核心 evidence 有 bridge_context 关系的背景或上下文 evidence。

fallback_core_first:
    若未达到 min_top_k 且没有规则候选，则用 core-first fallback 补足。
```

当已选 evidence 数量达到 `min_top_k` 后，如果没有新的 rule candidate，则停止；否则最多选到 `max_top_k=10`。

### 1.9 Trace-to-Verifier Data

selector 输出 `selection_trace`，其中保存：

```text
candidate_pool
selector_ordered_indices
candidate_scores
selector_name
fingerprint
```

随后 `build_trace_verifier_data.py` 根据 trace 坐标恢复被选 evidence。当前默认 prompt style 是：

```text
TRACE_PROMPT_STYLE=plain
```

因此 verifier 默认只看到：

```text
Claim:
<claim>

Evidence:
[1] <selected evidence text>
[2] <selected evidence text>
...
```

也就是说，verifier 默认不会看到 P1/P2/P3 rule、selector score、oracle 信息或 graph 结构标签。`trace_lite` 是后续 v0.6e 消融，不是 v0.6c 默认主方法。

### 1.10 Label-Token Verifier

Verifier 使用 Qwen2.5-7B-Instruct，训练目标不是生成长解释，而是输出一个 label token：

```text
Label: A/B/C/D/E/F
```

训练时使用 label-token cross entropy。当前 pipeline 支持 LoRA 与 FullFT 两种训练模式。

推理时，在 `Label:` 后对 A-F 六个 label token 做 constrained scoring 或 prompt logprob scoring，选择 log-probability 最大的 token：

$$\hat{y} = \arg\max_{y \in \mathcal{Y}} P(y \mid c, E^*)$$

---

## 2. 当前方法的创新性风险

当前方法的潜力很明确，但如果直接作为 AAAI 主方法投稿，可能面临如下审稿质疑。

### 2.1 selector 规则过于启发式

审稿人可能会问：

```text
P1/P2/P3 的顺序为什么合理？
min_top_k=5 和 max_top_k=10 为什么合理？
support/refute/qualify/mixed 为什么都被视为 core relation？
tension edge 何时是有效证据，何时只是噪声？
bridge_context 何时帮助 verifier，何时引入 background pollution？
```

如果这些问题只能通过 empirical ablation 回答，方法容易显得工程化。

### 2.2 graph 只服务于 selector，没有进入 verifier

当前 verifier 看到的是 plain evidence text，而不是 graph structure、relation labels、directness labels 或 selection rationale。因此，最终模型能力可能被理解为：

```text
retrieved evidence text + LLM classifier
```

这会削弱 graph-aware selector 的可见贡献。除非实验能证明 graph-aware selection 在多个 verifier 上都稳定有效，否则 graph 的贡献可能被认为只是上游特征工程。

### 2.3 label-token verifier 本身不是主要创新

Qwen2.5-7B-Instruct + label-token CE 是实用设计，但并不是强方法创新。该 verifier 更适合作为 downstream judge，而不是论文主贡献。

### 2.4 claim-associated reports 可能存在 leakage 风险

如果 report collection 来自 fact-checking reports，里面可能包含 journalist reasoning、verdict wording、rating cue 或 conclusion paragraph。此时任务可能被审稿人理解为：

```text
fact-check report summarization / verdict extraction
```

而不是 realistic fact-checking。因此必须严格说明 evidence source、verdict cue masking、conclusion removal、temporal leakage control 和 source provenance。

---

## 3. 第一条拔高方向：从规则 cascade 到 evidence-chain optimization

建议将 v0.6c selector 从：

```text
RuleStepSelect(G)
```

重写为：

```text
BudgetedMarginalChainSelect(G)
```

核心问题定义为：

> 给定 claim、candidate evidence、claim atoms 和 typed evidence graph，选择一个小规模 evidence subset，使其能够覆盖重要 claim atoms，保留互补和冲突证据，避免重复与背景噪声，并在 verifier budget 内达到最大验证效用。

形式化为：

$$E^* = \arg\max_{S \subseteq C,\ k_{\min} \le |S| \le k_{\max}} F(S; c, G)$$

其中目标函数可以设计为：

$$F(S; c, G) = \lambda_{\mathrm{cov}} \cdot \mathrm{AtomCoverage}(S) + \lambda_{\mathrm{dir}} \cdot \mathrm{Directness}(S) + \lambda_{\mathrm{rel}} \cdot \mathrm{StanceUtility}(S) + \lambda_{\mathrm{edge}} \cdot \mathrm{ChainCoherence}(S, G) + \lambda_{\mathrm{ten}} \cdot \mathrm{ConflictAwareness}(S, G) - \lambda_{\mathrm{dup}} \cdot \mathrm{Redundancy}(S) - \lambda_{\mathrm{bg}} \cdot \mathrm{BackgroundNoise}(S) - \lambda_{\mathrm{len}} \cdot |S|$$

这样，P1/P2/P3 不再是经验规则，而是目标函数的 marginal gain 近似。

### 3.1 Atom coverage term

claim atom coverage 可以写为：

$$\mathrm{AtomCoverage}(S) = \sum_{a \in A} w_a \cdot \mathbb{I}\left[\exists e \in S: e \text{ covers } a\right]$$

其中 `w_a` 是 atom importance。

如果考虑 evidence coverage confidence 与 directness，可以写为 soft coverage：

$$\mathrm{AtomCoverage}(S) = \sum_{a \in A} w_a \cdot \max_{e \in S} q(e, a)$$

其中：

$$q(e, a) = \mathrm{coverage}(e, a) \cdot \mathrm{directness}(e, a) \cdot \mathrm{confidence}(e, a)$$

### 3.2 Directness term

direct evidence 应该高于 background evidence。可以定义：

$$\mathrm{Directness}(S) = \sum_{e \in S} d(e)$$

其中：

$$d(e) \in \{d_{\mathrm{direct}}, d_{\mathrm{partial}}, d_{\mathrm{context}}, d_{\mathrm{none}}\}$$

并满足：

$$d_{\mathrm{direct}} > d_{\mathrm{partial}} > d_{\mathrm{context}} > d_{\mathrm{none}}$$

### 3.3 Stance utility term

relation polarity 的作用不应只是支持或反驳，还应表达 qualify / mixed 对细粒度 veracity 的贡献：

$$\mathrm{StanceUtility}(S) = \sum_{e \in S} r(e)$$

其中：

$$r(e) = f\left(\mathrm{relation}(e), \mathrm{directness}(e), \mathrm{confidence}(e)\right)$$

### 3.4 Chain coherence term

inter-evidence edge 可以进入 pairwise graph utility：

$$\mathrm{ChainCoherence}(S, G) = \sum_{(e_i, e_j) \in G[S]} \psi(e_i, e_j)$$

其中：

$$\psi(e_i, e_j) = \psi_{\mathrm{comp}}(e_i, e_j) + \psi_{\mathrm{corr}}(e_i, e_j) + \psi_{\mathrm{bridge}}(e_i, e_j) - \psi_{\mathrm{dup}}(e_i, e_j)$$

### 3.5 Conflict awareness term

`tension` 不建议只作为 positive edge。更合理的设计是把它作为 conflict-aware utility：

$$\mathrm{ConflictAwareness}(S, G) = \sum_{(e_i, e_j) \in G[S]} \tau(e_i, e_j)$$

其中 `tau` 需要区分有效冲突和噪声冲突：

$$\tau(e_i, e_j) = \tau_{\mathrm{credible}}(e_i, e_j) - \tau_{\mathrm{lowcred}}(e_i, e_j) + \tau_{\mathrm{qualified}}(e_i, e_j)$$

### 3.6 Redundancy and background penalties

重复证据应受到惩罚：

$$\mathrm{Redundancy}(S) = \sum_{(e_i, e_j) \in S \times S} \mathbb{I}\left[\mathrm{duplicate\_group}(e_i) = \mathrm{duplicate\_group}(e_j)\right]$$

背景噪声也应被惩罚：

$$\mathrm{BackgroundNoise}(S) = \sum_{e \in S} \mathbb{I}\left[\mathrm{relation}(e) \in \{\mathrm{background}, \mathrm{irrelevant}\}\right]$$

### 3.7 Greedy marginal selection

用 marginal gain 逐步选择 evidence：

$$e_t = \arg\max_{e \in C \setminus S_{t-1}} \Delta F(e \mid S_{t-1})$$

其中：

$$\Delta F(e \mid S) = F(S \cup \{e\}; c, G) - F(S; c, G)$$

停止条件可以从“没有 rule candidate”改成：

$$\mathrm{stop} \iff |S| \ge k_{\min} \land \max_{e \in C \setminus S} \Delta F(e \mid S) \le \epsilon$$

最终算法为：

```text
Input: claim c, candidate evidence C, evidence graph G, budget B
Output: selected evidence chain E*

S0 = ∅
for t = 1 ... B:
    e_t = argmax_e ΔF(e | S_{t-1})
    if |S_{t-1}| >= k_min and ΔF(e_t | S_{t-1}) <= ε:
        break
    S_t = S_{t-1} ∪ {e_t}
return S_t
```

该表述能将 v0.6c 从 heuristic selector 升级为一个 budgeted evidence-chain optimization algorithm。

---

## 4. 第二条拔高方向：将 verifier 改为 ordinal-aware 或 atom-level truthfulness verifier

当前六分类 label 具有天然序关系：

```text
pants-fire < false < barely-true < half-true < mostly-true < true
```

因此，普通 label-token CE 没有利用 veracity scale 的 ordinal structure。建议引入 ordinal-aware objective。

### 4.1 Ordinal-aware loss

可以保留 label-token CE，同时加入距离敏感损失：

$$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \alpha \cdot \mathcal{L}_{\mathrm{ordinal}}$$

标签距离可以定义为：

$$\mathrm{cost}(y, \hat{y}) = |\mathrm{rank}(y) - \mathrm{rank}(\hat{y})|$$

期望 ordinal penalty 可以写为：

$$\mathcal{L}_{\mathrm{ordinal}} = \sum_{y' \in \mathcal{Y}} P(y' \mid c, E^*) \cdot |\mathrm{rank}(y') - \mathrm{rank}(y)|$$

这样，模型把 `pants-fire` 预测成 `true` 会比预测成 `false` 受到更大惩罚。

### 4.2 Cumulative ordinal regression

也可以将六分类转化为五个 threshold 判断：

$$P(y > t \mid c, E^*) \quad \text{for } t = 1, 2, 3, 4, 5$$

最终标签由 threshold probabilities 组合得到：

$$\hat{y} = 1 + \sum_{t=1}^{5} \mathbb{I}\left[P(y > t \mid c, E^*) > 0.5\right]$$

### 4.3 Atom-level truthfulness aggregation

更进一步，可以将每个 atom 的 truthfulness 作为中间变量：

$$v_i \in [-1, 1]$$

其中：

```text
-1: atom is refuted
 0: atom is qualified / uncertain / partially supported
+1: atom is supported
```

整体 truthfulness score 定义为：

$$T(c, E^*) = \frac{\sum_{i=1}^{m} w_i \cdot v_i}{\sum_{i=1}^{m} w_i}$$

然后通过 thresholds 映射为六分类标签：

$$\hat{y} = \mathrm{Bucket}\left(T(c, E^*)\right)$$

这样 verifier 就不只是“claim + evidence → label token”，而是：

```text
claim decomposition → atom-level veracity estimation → fine-grained truthfulness aggregation
```

这比普通 label-token CE 更符合 fact-checking 的解释需求。

---

## 5. 第三条拔高方向：将 conflict / tension 做成正式贡献

当前图中有 `tension` edge，但如果只是把 tension evidence 加入 selection，它的学术贡献还不够。现实 fact-checking 的难点是：

```text
在冲突证据中判断哪一方可信；
判断冲突是否来自时间差、定义差异、范围差异或来源质量差异；
判断 claim 是否应被判为 qualify / mixed / half-true。
```

建议将方法升级为：

```text
credibility-aware conflict-resolving evidence chain selection
```

### 5.1 新增 source / provenance features

可以为 evidence 增加如下字段：

```text
source_type
source_reliability
publication_time
is_primary_source
is_secondary_source
is_fact_check_site
stance_consistency
leakage_risk
temporal_validity
```

### 5.2 tension edge 分类

`tension` 可以进一步细分：

```text
credible_tension
low_credibility_tension
temporal_tension
definition_tension
scope_tension
measurement_tension
```

### 5.3 Conflict utility

冲突效用不应简单奖励所有 tension，而应区分高质量冲突和低质量噪声：

$$\mathrm{ConflictUtility}(S) = \mathrm{HighCredConflict}(S) + \mathrm{QualifiedResolution}(S) - \mathrm{LowCredConflictNoise}(S)$$

其中：

$$\mathrm{HighCredConflict}(S) = \sum_{(e_i,e_j) \in G[S]} \mathbb{I}\left[\mathrm{tension}(e_i,e_j) \land \mathrm{credible}(e_i) \land \mathrm{credible}(e_j)\right]$$

低可信冲突噪声可以写为：

$$\mathrm{LowCredConflictNoise}(S) = \sum_{(e_i,e_j) \in G[S]} \mathbb{I}\left[\mathrm{tension}(e_i,e_j) \land \neg\mathrm{credible}(e_i,e_j)\right]$$

该方向可以把方法从“graph-aware selection”进一步提升为“conflict-aware fact-checking reasoning”。

---

## 6. 第四条拔高方向：增加 evidence sufficiency / necessity 评估

如果主贡献是 selector，那么只报告最终 Macro-F1 不够。需要证明选出的 evidence chain 具有如下性质：

```text
sufficient: 选出的 evidence 足以支持正确判断；
necessary: 移除关键 evidence 后性能下降；
non-redundant: 证据链不是重复堆叠；
robust: 加入 distractor 后判断不容易被污染；
transferable: 更换 verifier 后 selector 仍有效。
```

### 6.1 Selection quality metrics

建议报告：

```text
atom coverage
important atom coverage
direct evidence ratio
background evidence ratio
duplicate ratio
selected chain length
edge utilization rate
relation distribution
```

Atom coverage 可以定义为：

$$\mathrm{Coverage}(S) = \frac{|\{a \in A: \exists e \in S, e \text{ covers } a\}|}{|A|}$$

Important atom coverage 可以定义为：

$$\mathrm{ImportantCoverage}(S) = \frac{\sum_{a \in A} w_a \cdot \mathbb{I}[\exists e \in S: e \text{ covers } a]}{\sum_{a \in A} w_a}$$

Background evidence ratio 可以定义为：

$$\mathrm{BackgroundRatio}(S) = \frac{|\{e \in S: \mathrm{relation}(e) \in \{\mathrm{background}, \mathrm{irrelevant}\}\}|}{|S|}$$

Duplicate ratio 可以定义为：

$$\mathrm{DuplicateRatio}(S) = 1 - \frac{|\{\mathrm{duplicate\_group}(e): e \in S\}|}{|S|}$$

### 6.2 Sufficiency evaluation

Sufficiency 评估目标是证明 selected evidence 本身足够：

$$\mathrm{Sufficiency} = \mathrm{Perf}\left(\mathrm{Verifier}(c, E^*)\right)$$

需要与以下输入对比：

```text
claim only
claim + top-k retrieval evidence
claim + all top-20 evidence
claim + random k evidence
claim + map_quality top-k evidence
claim + oracle evidence upper bound
```

### 6.3 Necessity / comprehensiveness evaluation

Necessity 评估目标是证明关键证据被移除后性能应下降：

$$\mathrm{NecessityDrop} = \mathrm{Perf}\left(\mathrm{Verifier}(c, E^*)\right) - \mathrm{Perf}\left(\mathrm{Verifier}(c, E^* \setminus E_{\mathrm{key}})\right)$$

其中 `E_key` 可以定义为 selector marginal gain 最高的 evidence，或者覆盖重要 atom 的 evidence。

### 6.4 Distractor robustness evaluation

加入 distractor 后，好的 selector / verifier 应该保持稳定：

$$\mathrm{RobustnessDrop} = \mathrm{Perf}\left(\mathrm{Verifier}(c, E^*)\right) - \mathrm{Perf}\left(\mathrm{Verifier}(c, E^* \cup D)\right)$$

其中 `D` 可以包括：

```text
high lexical overlap distractors
duplicate paraphrases
background-only chunks
low-credibility conflicting evidence
same-topic irrelevant evidence
```

### 6.5 Cross-verifier transfer

如果同一个 selector 在多个 verifier 上都提升，则贡献更可信。建议测试：

```text
Qwen2.5-7B verifier
smaller instruction model verifier
larger instruction model verifier
non-LLM classifier verifier
ordinal verifier
```

---

## 7. 第五条拔高方向：处理 report leakage 与 realistic evidence setting

如果当前 `R` 来自 claim-associated fact-checking reports，必须认真处理 leakage。建议设计两个实验 setting。

### 7.1 Report-associated setting

```text
Evidence comes from cleaned fact-check reports.
Task positioning: evidence-chain verdict inference.
```

该 setting 必须执行：

```text
remove verdict sentence
remove rating cue
remove conclusion paragraph
mask label-like expressions
remove explicit PolitiFact rating words
record whether evidence appears before or after the verdict sentence
```

可以定义 leakage risk：

$$\mathrm{LeakageRisk}(e) = \mathbb{I}[e \text{ contains verdict cue}] + \mathbb{I}[e \text{ appears in conclusion}] + \mathbb{I}[e \text{ contains label-like phrase}]$$

将 leakage risk 纳入 selector penalty：

$$F'(S; c, G) = F(S; c, G) - \lambda_{\mathrm{leak}} \cdot \sum_{e \in S} \mathrm{LeakageRisk}(e)$$

### 7.2 Open-evidence setting

```text
Evidence comes from source links, web documents, official records, or temporally valid external documents.
Task positioning: realistic automated fact-checking.
```

该 setting 更接近真正 fact-checking，但成本更高。可以作为额外实验或迁移评估，而不是第一版主实验。

### 7.3 Temporal validity

对于 claim 发布时间为 `t_c` 的样本，evidence 发布时间 `t_e` 应满足：

$$t_e \le t_c + \Delta$$

其中 `Delta` 需要根据任务设定解释。严格 fact-checking setting 中通常应避免使用 claim verdict 之后才出现的 evidence。

---

## 8. 第六条拔高方向：将 evidence map 变成可评估中间表示

当前 evidence map 是方法中很有价值的部分，但如果只作为 selector features，会显得像 preprocessing。建议将其明确为：

```text
Structured Evidence Map for Fine-Grained Claim Verification
```

并在论文中作为 intermediate representation 进行诊断评估。

### 8.1 Evidence-map evaluation dimensions

建议评估：

```text
atom decomposition quality
atom importance agreement
evidence-atom alignment accuracy
relation label accuracy
directness label accuracy
key_span faithfulness
duplicate_group precision
confidence calibration
```

例如，evidence-atom alignment accuracy 可以写为：

$$\mathrm{AlignAcc} = \frac{1}{|C|} \sum_{e \in C} \mathbb{I}\left[\widehat{A}(e) = A^{\mathrm{gold}}(e)\right]$$

如果一个 evidence 可覆盖多个 atoms，则可用 F1：

$$\mathrm{AlignF1}(e) = \frac{2 \cdot |\widehat{A}(e) \cap A^{\mathrm{gold}}(e)|}{|\widehat{A}(e)| + |A^{\mathrm{gold}}(e)|}$$

### 8.2 Gold / noisy map ablation

如果 evidence map 是 LLM 自动标注，不能把它当 gold rationale。建议报告：

```text
human-checked map upper bound
LLM-induced map
noisy map
randomized relation map
randomized atom coverage map
no directness map
no key_span map
```

这能回答审稿人关于 graph 标签可靠性的质疑。

---

## 9. 建议升级后的 AAAI-strength 方法版本

建议将方法改写为：

```text
C_claim = ClaimMMR(HybridRetrieve(c, R))

Q = QuestionDecompose(c)

C_qd = QDRetrieve(Q, R)
    per-question hybrid retrieval, route rank, RRF-style merge

C = UnionRank(C_claim, C_qd)
    baseline claim evidence + decomposed-question evidence

M = InduceEvidenceMap(c, C)
    atoms, evidence-atom coverage, relation, directness, spans, confidence

G = BuildTypedEvidenceGraph(c, M)
    atom nodes, evidence nodes, typed relational edges

E* = BudgetedMarginalChainSelect(G)
    maximize coverage, directness, stance utility, graph coherence,
    conflict awareness, and credibility;
    penalize redundancy, leakage risk, background noise

ŷ = OrdinalVerifier(c, E*)
    label-token CE + ordinal calibration / atom-level aggregation
```

核心目标函数可以写为：

$$E^* = \arg\max_{S \subseteq C,\ |S| \le B} \left[\sum_{a \in A} w_a \cdot g\left(\max_{e \in S} q(e, a)\right) + \sum_{e \in S} u(e) + \sum_{(e_i,e_j) \in G[S]} \psi(e_i,e_j) - \Omega(S)\right]$$

其中：

```text
q(e, a): evidence 对 atom a 的覆盖、直接性与置信度
u(e): 单条 evidence utility，包括 relation、directness、confidence、source credibility
ψ(e_i, e_j): evidence pair utility，包括 complements、corroborates、tension、bridge_context
Ω(S): redundancy、background noise、leakage risk 和 length penalty
```

Greedy selection 为：

$$e_t = \arg\max_{e \in C \setminus S_{t-1}} \left[F(S_{t-1} \cup \{e\}; c, G) - F(S_{t-1}; c, G)\right]$$

停止条件为：

$$|S| \ge k_{\min} \land \max_{e \in C \setminus S} \Delta F(e \mid S) \le \epsilon$$

最终预测为：

$$\hat{y} = \arg\max_{y \in \mathcal{Y}} P(y \mid c, E^*)$$

如果使用 ordinal verifier，则最终预测也可以写为：

$$\hat{y} = \mathrm{Bucket}\left(\frac{\sum_{i=1}^{m} w_i \cdot v_i}{\sum_{i=1}^{m} w_i}\right)$$

---

## 10. 建议实验设计

### 10.1 Retriever baselines

```text
BM25 top-k
dense top-k
hybrid top-k
MMR top-k
top-20 all evidence
random k from top-20
```

### 10.2 Selector baselines

```text
map_quality_score sort
atom coverage greedy only
directness only
relation only
no graph edges
no bridge context
no tension edges
no duplicate penalty
fixed top-5
fixed top-10
LLM-as-selector baseline
```

### 10.3 Verifier baselines

```text
claim-only
claim + top-k retrieval
claim + selected evidence
claim + oracle / report evidence upper bound
CE verifier
ordinal verifier
atom-level aggregation verifier
```

### 10.4 Full ablation

需要重点验证：

```text
v0.6c full
- atom nodes
- edge types
- evidence-map features
- adaptive stopping
- bridge_context
- tension
- duplicate grouping
- confidence score
- directness score
- relation score
```

### 10.5 Sensitivity analysis

建议测试：

```text
candidate_top_n ∈ {10, 20, 30, 50}
min_top_k ∈ {3, 5, 7}
max_top_k ∈ {5, 10, 15}
hybrid weights varied
λ parameters varied
ε varied
```

如果方法只在 `candidate_top_n=20, min_top_k=5, max_top_k=10` 下有效，会显得像调参结果。如果在较宽参数范围内稳定有效，可信度会高很多。

---

## 11. 建议论文贡献写法

不建议写成：

```text
We propose a rule-step adaptive selector with P1/P2/P3.
```

建议写成：

```text
We formulate evidence selection for fine-grained fact verification as a budgeted evidence-chain optimization problem over an atom-evidence heterogeneous graph.

We introduce a structured evidence map that aligns candidate evidence with atomic claim units, relation polarity, directness, and evidence roles.

We propose a graph-aware marginal selection algorithm that constructs compact, sufficient, non-redundant evidence chains by jointly optimizing atom coverage, directness, relation utility, inter-evidence coherence, conflict awareness, and context bridging.

We show that the selected evidence chains improve fine-grained veracity prediction, evidence sufficiency, robustness to distractors, and cross-verifier transfer.
```

中文表述可以写为：

```text
本文将细粒度事实核查中的证据选择形式化为一个基于 atom-evidence 异构图的 budgeted evidence-chain optimization 问题。

本文提出结构化 evidence map，将候选证据与 claim atoms、relation polarity、directness、evidence role 和 key spans 对齐。

本文提出 graph-aware marginal evidence-chain selector，在有限 verifier budget 内联合优化 atom coverage、directness、stance utility、graph coherence、conflict awareness 与 redundancy penalty。

实验表明，所选证据链在最终六分类 veracity prediction、evidence sufficiency、distractor robustness 和 cross-verifier transfer 上均优于 top-k retrieval 与非图选择方法。
```

---

## 12. 建议方法命名

当前名称：

```text
v0_6c_rule_step_adaptive5_10
```

适合作为内部实验版本名，但不适合作为论文方法名。

建议论文方法名：

```text
AG-Select: Atom-Graph Evidence Chain Selection
```

或：

```text
MSEC: Minimal Sufficient Evidence Chain Selection
```

或：

```text
AGECS: Atom-Graph Evidence Chain Selector
```

或：

```text
BMC-Select: Budgeted Marginal Chain Selection
```

其中我更推荐：

```text
AG-Select
```

因为它简短，能直接传达 atom-aware 和 graph-aware 的核心贡献。

---

## 13. 建议论文标题

可以考虑如下标题：

```text
Atom-Graph Evidence Chain Selection for Fine-Grained Fact Verification
```

```text
Selecting Minimal Sufficient Evidence Chains for Fine-Grained Fact-Checking
```

```text
Budgeted Evidence-Chain Optimization for Fine-Grained Veracity Prediction
```

```text
From Claims to Evidence Chains: Atom-Aware Graph Selection for Fact Verification
```

```text
Conflict-Aware Evidence Chain Selection for Fine-Grained Fact-Checking
```

如果当前版本主要强调 selector，推荐第一个：

```text
Atom-Graph Evidence Chain Selection for Fine-Grained Fact Verification
```

如果后续加入 credibility / tension reasoning，推荐第五个：

```text
Conflict-Aware Evidence Chain Selection for Fine-Grained Fact-Checking
```

---

## 14. 推荐版本路线

最短升级路线：

```text
v0.6c rule-step selector
→ v0.7 objective-based marginal selector
→ v0.8 ordinal verifier + sufficiency evaluation
→ v1.0 credibility/conflict-aware evidence-chain selector
```

### 14.1 v0.7

目标：将当前 P1/P2/P3 规则改写为 marginal gain optimization。

核心变化：

```text
定义 F(S; c, G)
将 rule priority 改为 ΔF(e | S)
将 stop rule 改为 marginal gain threshold
补充参数敏感性实验
```

### 14.2 v0.8

目标：强化 verifier 与 evaluation。

核心变化：

```text
加入 ordinal-aware loss
加入 atom-level truthfulness aggregation
加入 sufficiency / necessity / distractor robustness 评估
加入 cross-verifier transfer
```

### 14.3 v1.0

目标：形成更强的 AAAI 主方法版本。

核心变化：

```text
加入 source credibility
加入 conflict type classification
加入 leakage risk penalty
加入 temporal validity control
在外部数据集上做迁移验证
```

---

## 15. 总结

当前 v0.6c 的真正贡献不应被表述为：

```text
一个 rule-step adaptive evidence selector
```

而应被表述为：

```text
一个面向细粒度事实核查的 atom-aware、graph-aware、relation-aware、conflict-aware evidence-chain selection framework
```

核心升级路径是：

$$\mathrm{RuleStepSelect}(G) \rightarrow \arg\max_{S \subseteq C} F(S; c, G)$$

即把规则选择器提升为 evidence-chain optimization。

如果只保留当前 v0.6c，并在 LIAR-RAW 上报告最终六分类结果，AAAI 主技术轨风险较高，容易被认为是 heuristic-heavy pipeline engineering。若能完成以下四点，论文说服力会显著增强：

```text
1. 将 selector 形式化为 budgeted evidence-chain optimization。
2. 增加 evidence sufficiency / necessity / robustness 评估。
3. 严格处理 report leakage、verdict cue masking 和 temporal/source provenance。
4. 将 verifier 改成 ordinal-aware 或 atom-level truthfulness aggregation。
```

进一步加入 credibility-aware conflict resolution，并在外部数据集上证明迁移性后，该方法会更接近一个具有清晰问题定义、方法抽象和实证深度的 AAAI 级主方法。

---

## 16. 可参考的相关工作方向

以下方向适合作为 related work 组织线索：

```text
fine-grained fact verification
claim decomposition
multi-hop evidence retrieval
heterogeneous evidence graph
rationale selection
evidence sufficiency and comprehensiveness
ordinal veracity prediction
conflicting evidence in RAG-based fact-checking
source credibility and temporal leakage
```

可重点关注：

```text
LIAR: A Benchmark Dataset for Fake News Detection
AVeriTeC: A Dataset for Real-world Claim Verification with Evidence
HoVer: A Dataset for Multi-hop Fact Extraction and Claim Verification
FEVER / FEVEROUS: Fact Extraction and Verification over textual and structured evidence
Complex Claim Verification with Evidence Retrieval and Summarization
CONFACT: Benchmarking LLMs Against Conflicting Evidence in RAG-based Fact-Checking
Heterogeneous Evidence Graph methods for fact-checking
```
