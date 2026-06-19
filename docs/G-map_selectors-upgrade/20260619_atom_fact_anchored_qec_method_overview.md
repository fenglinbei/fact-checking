# Atom-Fact Anchored QEC Method Overview

## 0. 核心判断

当前最强线已经不只是 `retrieval + verifier`，而是一条细粒度证据组织链：

```text
ABC claim-aware evidence units
    -> claim / question retrieval union
    -> atomic evidence map
    -> typed evidence graph
    -> budgeted evidence-chain selection
    -> QEC prompt
    -> label-token verifier
```

论文中应把主贡献表述为 Atom-Fact Anchored QEC (AA-QEC)：

> 面向细粒度事实核查的 atom-fact anchored question-guided evidence-chain selection。

也就是说，方法核心不是让 verifier 看到更多 evidence，而是在有限 verifier budget 内选择一条 compact、coverage-aware、non-redundant、question-guided 的最小充分证据链。`qec_min` 和 `qec_map` 是同一证据链方法的两种 prompt realization，不应被写成两套独立算法。

---

## 1. 任务定义

给定一个 fact-checking 样本：

```text
x = (claim c, report collection R)
```

目标是预测 veracity label：

```text
y in Y
```

其中 LIAR-RAW 使用六分类标签：

```text
Y_liar = {pants-fire, false, barely-true, half-true, mostly-true, true}
```

RAWFC 使用三分类标签：

```text
Y_rawfc = {false, half, true}
```

方法不是直接学习：

```text
ŷ = Verifier(c, top-k evidence)
```

而是显式构造中间证据链：

```text
E* = ChainSelect(c, R)
ŷ = Verifier(c, QEC(E*))
```

其中 `E*` 是面向 claim components 的 evidence chain，`QEC(E*)` 是把每条 evidence 渲染为 question-guided chain step 的 verifier input。

---

## 2. 当前方法总流程

端到端流程可以写为：

$$U = \mathrm{ABCChunk}(R, c)$$

$$C_{\mathrm{claim}} = \mathrm{ClaimMMR}(\mathrm{HybridRetrieve}(c, U))$$

$$Q = \mathrm{QuestionDecompose}(c)$$

$$C_{\mathrm{qd}} = \mathrm{QDRetrieve}(Q, U)$$

$$C = \mathrm{UnionRank}(C_{\mathrm{claim}}, C_{\mathrm{qd}})$$

$$M = \mathrm{EvidenceMap}(c, C)$$

$$G = \mathrm{BuildTypedGraph}(c, M)$$

$$E^* = \mathrm{BudgetedChainSelect}(G; k_{\min}, k_{\max})$$

$$\pi = \mathrm{QECFormat}(c, E^*)$$

$$\hat{y} = \mathrm{LabelTokenVerifier}(\pi)$$

合并为一个公式：

$$\hat{y} = \mathrm{Verifier}\left(\mathrm{QECFormat}\left(c,\ \mathrm{BudgetedChainSelect}\left(\mathrm{BuildTypedGraph}\left(c,\ \mathrm{EvidenceMap}\left(c,\ \mathrm{UnionRank}\left(\mathrm{ClaimMMR}\left(\mathrm{HybridRetrieve}(c,U)\right),\ \mathrm{QDRetrieve}\left(\mathrm{QuestionDecompose}(c),U\right)\right)\right)\right)\right)\right)\right)$$

该公式中的关键变化是：verifier 输入不再是无结构 evidence list，而是由 evidence map 和 typed graph 约束后的 chain steps。

---

## 3. 方法模块

### 3.1 ABC Claim-Aware Evidence Unit Construction

系统先把每个 report 切分为候选 evidence units。当前最强线使用 ABC claim-aware chunking，其目标不是普通 topic segmentation，而是服务于 claim verification。

对相邻句子边界，方法同时考虑：

```text
semantic boundary
claim relevance difference
length budget
```

claim relevance 可以沿用三路 hybrid score：

$$s(u, c) = 0.70 \cdot s_{\mathrm{dense}}(u, c) + 0.20 \cdot s_{\mathrm{lexical}}(u, c) + 0.10 \cdot s_{\mathrm{BM25}}(u, c)$$

边界分数可概括为：

$$B_i = w_{\mathrm{sem}} \cdot (1 - \mathrm{sim}(s_i, s_{i+1})) + w_{\mathrm{rel}} \cdot |rel(s_i,c) - rel(s_{i+1},c)|$$

这样做的作用是：当背景句和关键证据句语义上相邻但对 claim 的判别价值不同，chunker 可以在它们之间切开，避免后续 evidence map 和 verifier 被背景上下文稀释。

RAWFC tight variant 使用更保守的边界参数和更短 evidence unit 上限。它不是一个新的 selector，而是同一 evidence unit construction 的 dataset-specific conservative instantiation。

### 3.2 Claim-Level Retrieval and Question-Decomposition Retrieval

在 evidence units `U` 上，系统先执行 claim-level hybrid retrieval：

```text
dense similarity
lexical overlap
BM25-like score
```

并用 MMR 控制 relevance 与 diversity，得到：

$$C_{\mathrm{claim}} = \mathrm{ClaimMMR}(\mathrm{HybridRetrieve}(c, U))$$

随后执行 question decomposition retrieval。claim 被分解为若干核查子问题：

$$Q = \{q_1, q_2, \ldots, q_m\}$$

每个 question route 在同一批 evidence units 上单独检索：

$$C_i = \mathrm{HybridRetrieve}(q_i, U)$$

然后通过 route rank、question hit count、max question score 和 RRF-style merge 合并为：

$$C_{\mathrm{qd}} = \mathrm{MergeRoutes}(C_1, C_2, \ldots, C_m)$$

最终候选池来自两路 union：

```text
claim-level MMR candidates
    +
question-decomposition candidates
    ->
union candidate pool
```

union 阶段保留候选来源特征，例如 baseline rank、question route rank、QD hit count 和 union pool rank。这些特征既用于 evidence map，也用于后续 chain selection 的 base signal。

### 3.3 Atomic Evidence Map Construction

Evidence map stage 消费 union 后的 candidate pool，并把 claim 分解为 atomic facts：

$$A = \{a_1, a_2, \ldots, a_n\}$$

每个 atom 对应 claim 中一个更小的可验证 component，例如实体、时间、数量、归因或比较关系。

对每条候选 evidence `e`，evidence map 标注：

```text
covered_atom_ids
relation: support / refute / qualify / mixed / background / irrelevant
directness: direct / partial / context / none
evidence_role
key_spans
duplicate_group
confidence
```

这里的 atomic facts 是证据组织单元，不是要求 verifier 输出的中间答案。方法只使用它们来追踪：

```text
哪些 claim components 已被覆盖
哪些 atoms 仍缺直接证据
哪些 evidence 只是背景或重复上下文
```

### 3.4 Map Feature Scoring

Evidence map annotation 会被后处理为候选 evidence utility。核心特征包括：

```text
atom_coverage_score
map_relation
map_directness
directness_score
polar_relation_score
background_penalty_score
evidence_map_quality_score
```

可以把单条 evidence utility 写成：

$$u(e) = f(\mathrm{coverage}(e), \mathrm{directness}(e), \mathrm{relation}(e), \mathrm{confidence}(e), \mathrm{keyspan}(e)) - \lambda_{\mathrm{bg}} \cdot \mathrm{background}(e)$$

该分数不是最终 label，而是 chain selection 的结构化输入。

### 3.5 Typed Evidence Graph Construction

方法把 claim、atoms 和 evidence candidates 构造成 typed evidence graph：

```text
claim node: C0
atom nodes: a1 ... an
evidence nodes: e1 ... em
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

图的作用是把 evidence selection 从独立排序变成 subset selection：候选 evidence 的价值不只取决于自身分数，还取决于它是否补足未覆盖 atom、是否与已选 evidence 互补、是否只是重复、是否提供必要 bridge context。

### 3.6 Budgeted Evidence-Chain Selection

给定 graph `G` 和候选池 `C`，方法选择一个有限长度证据链：

$$E^* = \arg\max_{S \subseteq C,\ k_{\min} \le |S| \le k_{\max}} F(S; c, G)$$

目标函数可以概括为：

$$F(S; c, G) = \lambda_{\mathrm{cov}} \cdot \mathrm{AtomCoverage}(S) + \lambda_{\mathrm{map}} \cdot \mathrm{MapQuality}(S) + \lambda_{\mathrm{base}} \cdot \mathrm{RetrievalUtility}(S) + \lambda_{\mathrm{edge}} \cdot \mathrm{ChainCoherence}(S,G) - \lambda_{\mathrm{dup}} \cdot \mathrm{Redundancy}(S) - \lambda_{\mathrm{bg}} \cdot \mathrm{BackgroundNoise}(S) - \lambda_{\mathrm{len}} \cdot |S|$$

Atom coverage 可以写为：

$$\mathrm{AtomCoverage}(S) = \sum_{a \in A} w_a \cdot \mathbb{I}[\exists e \in S: e \text{ covers } a]$$

图关系效用可以写为：

$$\mathrm{ChainCoherence}(S,G) = \sum_{(e_i,e_j) \in G[S]} \psi(e_i,e_j)$$

其中 `psi` 奖励 complements、corroborates、有效 tension 和必要 bridge context，并惩罚 duplicate 或低价值背景上下文。

实际选择采用 greedy marginal gain：

$$e_t = \arg\max_{e \in C \setminus S_{t-1}} \Delta F(e \mid S_{t-1})$$

其中：

$$\Delta F(e \mid S) = F(S \cup \{e\}; c, G) - F(S; c, G)$$

停止条件是：

```text
达到最小链长后，如果新增 evidence 的 marginal gain 不再足够，则停止；
否则继续选择，直到达到最大链长。
```

因此，方法不是固定 top-k，而是根据样本复杂度构造 adaptive evidence chain。

### 3.7 Atom-Fact Anchored Chain Step Construction

Budgeted selector 给出 evidence subset 后，AA-QEC 进一步把 selected evidence 转换为 chain steps：

```text
z_t = (role_t, check_t, evidence_t, map_t)
```

其中：

```text
role_t: primary / secondary / fallback
check_t: 当前 evidence 正在检查的问题或 claim atom
evidence_t: 原始 evidence text 或 anchor text
map_t: 可选的 compact evidence-map metadata
```

`check_t` 的来源优先级为：

```text
1. QD question route
2. covered claim atom
3. claim-level fallback cue
```

这一步把 evidence selection 的内部结构显式转换为 verifier 可读的 prompt cue。它仍然遵循 evidence-as-answer 原则：`check_t` 只说明要检查什么，答案仍来自 evidence text 本身。

### 3.8 QEC Prompt Formatting

QEC prompt 有两个主要 realization。

`qec_min` 只显示最小 chain cue：

```text
Evidence:
[1] Check: <verification cue>
<evidence text>

[2] Check: <verification cue>
<evidence text>
```

`qec_map` 在 `qec_min` 基础上加入 compact map tags：

```text
Evidence:
[1] Check: <verification cue>
Map: covers=<atom ids>; relation=<relation>; directness=<directness>
<evidence text>
```

两者都不要求模型输出 reasoning trace，也不生成中间 answer。它们的区别只是 verifier 是否看到 compact alignment metadata。

### 3.9 Label-Token Verifier

最终 verifier 接收：

```text
π = QECFormat(c, E*)
```

并预测 label token：

$$\hat{y} = \arg\max_{y \in \mathcal{Y}} P(y \mid \pi)$$

LIAR-RAW 中 label token 对应 A-F，RAWFC 中对应三分类标签。该 verifier 是 downstream judge，不是本文最主要的创新点；方法贡献应落在 evidence unit construction、atomic evidence map、typed graph selection 和 QEC prompting 这条链上。

---

## 4. Dataset-Specific Instantiations

### 4.1 LIAR-RAW

LIAR-RAW 主线采用：

```text
ABC claim-aware evidence units
atomic evidence-map alignment
v0.7 budgeted marginal evidence chain
qec_min / qec_map prompt realizations
label-token verifier
```

该设置适合六分类政治声明，因为 claim 往往包含多个实体、时间、数量、归因或比较子命题。单纯 top-k evidence list 容易覆盖冗余背景而漏掉关键 component；atom-aware evidence chain 则显式优化 component coverage 与 evidence non-redundancy。

论文中建议把 `qec_min` 和 `qec_map` 写成 prompt sensitivity，而不是两个不同方法。当前结果显示二者接近，因此更稳妥的表述是：

```text
The same atom-fact anchored evidence chain can be rendered either with minimal question cues or with compact map metadata.
```

### 4.2 RAWFC

RAWFC 使用更保守的 instantiation：

```text
ABC claim-aware rawfc-tight evidence units
anchor-only evidence rendering
qec_min prompt
label-token verifier
```

该设置保留 atomic alignment 和 evidence-chain selection，但收紧 evidence unit 与 anchor 的关系。其目标是降低小样本闭集证据场景中的 prompt noise，而不是扩大 evidence context。

在论文叙述中，RAWFC 的重点应写成 dataset-specific conservative policy：

```text
For the smaller closed-evidence RAWFC setting, we instantiate the same framework with tighter claim-aware units and anchor-only rendering to reduce background drift.
```

---

## 5. Current Implementation Record

| Dataset | Method instantiation | Prompt form | Current checkpoint note | Role in paper narrative |
|---|---|---|---|---|
| LIAR-RAW | v0.7 atomic-fact ABC evidence chain | `qec_map` / `qec_min` | `qec_map` checkpoint-2400: test macro-F1 0.3587, selection 0.7686; `qec_min` checkpoint-1600: test macro-F1 0.3578, selection 0.7765. | Main LIAR-RAW instantiation; treat prompt difference as realization/sensitivity, not a separate algorithm. |
| RAWFC | v0.7 atomic-fact ABC tight anchor-only evidence chain | `qec_min` | Keep the val-best alias for official selection; direct test diagnostics over inspected checkpoints favor checkpoint-400 with test macro-F1 0.6716 and selection 0.6716. | Dataset-specific conservative instantiation; use for smaller closed-evidence setting and anchor-tightness discussion. |

---

## 6. Recommended Paper Wording

We formulate fine-grained fact-checking evidence selection as an atom-fact anchored evidence-chain construction problem. Given a claim and its associated reports, our method first builds claim-aware evidence units, retrieves candidates through both claim-level and question-decomposition routes, and induces a structured evidence map that aligns candidates with atomic claim components, relation polarity, directness, and key spans.

On top of the evidence map, we construct a typed evidence graph and select a compact evidence chain under a verifier budget. The selection objective jointly rewards atom coverage, map quality, retrieval utility, and inter-evidence coherence, while penalizing redundancy, background drift, and excessive length.

The selected chain is rendered with Question-guided Evidence Chain prompting. Each evidence unit is paired with a concise verification cue, and an optional map-aware realization exposes compact alignment metadata. Both prompt forms keep evidence as the answer and avoid generating intermediate rationales.

For RAWFC, we instantiate the same framework with tighter claim-aware evidence units and anchor-only rendering, which reduces prompt noise in the smaller closed-evidence setting while preserving the core atomic alignment and evidence-chain design.
