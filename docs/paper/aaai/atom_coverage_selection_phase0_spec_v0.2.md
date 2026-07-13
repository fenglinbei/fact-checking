# BACES Phase 0 问题与算法边界规范 v0.2

> **Superseded notice（2026-07-12）**：v0.2 正确删除了 evidence dependency、multi-hop 与 conflict reasoning，但把 partial/direct 压成二值覆盖，并把顺序降为纯渲染选择。当前冻结版以 `atom_coverage_sequencing_phase0_spec_v0.3.md` 为准：使用 \(q_{ij}\in\{0,1,2\}\) 的 ordinal structural coverage、prefix acquisition objective 与 exact ordered-state DP。本文档仅作决策审计记录。

状态：**Superseded — 不再实现或引用为主方法**

取代：`barec_scsg_phase0_spec_v0.1.md`

适用范围：论文问题定义、方法命名、coverage selector 实现、trace schema、预算控制、实验与主张边界。

不追溯改写：历史 `transition_v0_1`、`learned_marginal_proxy`、旧 evidence-chain artifacts 与已完成实验名称。它们只作为历史 baseline，不再代表本文冻结后的主方法。

---

## 0. 一页冻结结论

### 0.1 固定问题名称

- 任务对象：**Atom-Coverage Evidence Set Selection**（原子覆盖证据集选择）。
- 正式问题：**Budgeted Atom-Coverage Evidence Selection（BACES）**。
- 数学归属：**Weighted Budgeted Maximum Coverage**；同时使用 cardinality upper bound 与可选 knapsack/token budget 时，是带基数和背包约束的单调子模最大化实例。
- 主求解器：`baces_exact_mask_dp_v0_2`，论文中称 **exact mask dynamic programming**，不把它包装成新近似算法。
- 可扩展 baseline：`baces_marginal_greedy_v0_2`，论文中称 **marginal coverage greedy**。
- 输出：**selected evidence set**；输入 verifier 前可确定性地排成 **ordered evidence slate**。该顺序仅是渲染顺序，不是推理链。

### 0.2 必须退役的主论文表述

以下术语不再用于描述主方法：

- `Evidence-Chain`、`evidence-chain construction`；
- multi-hop reasoning、reasoning path、reasoning trajectory；
- evidence dependency、bridge endpoint、prerequisite relation；
- conflict reasoning、contradiction resolution；
- atom-resolving state machine；
- `OPEN / CONTRAST / QUALIFY / CORROBORATE / BRIDGE / CONTEXT` 操作；
- `U/S/R/Q/C` atom state；
- “前一步证据改变后一步证据的语义价值”这一未被实现验证的主张。

旧代码或图中的 `chain` 可以作为兼容字段保留，但不得被当作算法语义。

### 0.3 一句话问题定义

> Given a claim decomposed into atomic propositions, a flat candidate evidence pool, and typed atom–evidence alignments, BACES selects a budget-feasible evidence subset that maximizes the total weight of covered claim atoms. It does not model evidence dependencies, multi-hop composition, or conflict resolution.

对应中文：

> 给定声明原子、平铺的候选证据池以及带类型的 atom–evidence 对齐，BACES 在数量与可选 token 预算下选择证据子集，使被有效证据覆盖的声明原子总权重最大；它不建模证据间依赖、多跳组合或冲突消解。

### 0.4 固定系统边界

```text
claim
  -> claim atoms
  -> flat Atom-Union candidate pool
  -> pair-level atom-evidence map
  -> BACES set optimization
  -> deterministic evidence-slate rendering
  -> prompt max-length guard
  -> verifier
```

BACES 只包含上图中的“set optimization”。Atomization、retrieval、LLM evidence-map generation、prompt truncation 与 verifier classification 均是上下游模块，不属于 BACES 求解本身。

---

## 1. 为什么必须取消 Evidence-Chain

“Evidence-Chain”在事实核查与推理文献中通常会让读者合理期待至少一种结构：

1. 后一条证据依赖前一条证据提供的实体或中间结论；
2. 证据之间存在可验证的有向边或先决条件；
3. 多条证据经过组合才能完成单条证据不能完成的多跳证明；
4. 顺序改变会改变推理是否合法或结论是否成立；
5. 支持与反驳证据的冲突被显式检测和消解。

本文当前方法没有定义或验证上述对象。现有 evidence map 只提供 atom–evidence pair，不提供 evidence–evidence edge；selector 的逐步加入只是计算边际覆盖与确定展示顺序。因此继续使用 chain 会形成超出实现的语义承诺。

冻结后的语义是：

- evidence units 在优化模型中是平铺集合元素；
- candidate 的价值只由它新增覆盖了哪些 atoms、代价是多少决定；
- 已选集合影响后续 candidate 的**边际覆盖量**，但不产生逻辑依赖；
- 同时覆盖多个 atoms 是集合覆盖，不等于多跳推理；
- support/refute 等 relation 用于判断 pair 是否能形成有效覆盖，不代表 selector 在执行冲突推理；
- 输出顺序只服务于可复现渲染、审计与 verifier 输入。

---

## 2. BACES 实例

### 2.1 Claim atoms

给定 claim $c$，atomizer 输出：

\[
\mathcal A(c)=\{a_1,\ldots,a_m\},\qquad 1\le m\le 6.
\]

每个 atom 具有非负权重：

\[
w_i\ge 0.
\]

Phase 0 主配置固定为：

\[
w_i=1,\quad \forall i.
\]

理由是当前 `importance` 来自同一个 LLM atomization 输出，尚无独立标注证明它能稳定表达任务重要性。非均匀 `importance` 只作为 sensitivity/ablation，不作为主结果默认项。

### 2.2 Flat candidate pool

Atom-Union retrieval 产生平铺候选池：

\[
\mathcal E=\{e_1,\ldots,e_n\}.
\]

每条 candidate 至少包含：

- `evidence_id` 或等价 stable identity；
- evidence text / anchor text；
- retrieval metadata；
- token cost $c_j\in\mathbb Z_{\ge 0}$；
- pair-level evidence-map rows。

候选池中的原始顺序不是 oracle order，也不是 BACES 的解。

### 2.3 Pair-level typed alignments

对每个 $(a_i,e_j)$，evidence map 可以给出：

\[
M_{ij}=(r_{ij},d_{ij},\gamma_{ij},s_{ij}),
\]

其中：

- $r_{ij}$：relation；
- $d_{ij}$：directness；
- $\gamma_{ij}$：map confidence；
- $s_{ij}$：key span。

**Pair-level map 是 coverage 的唯一事实来源。** candidate-level collapsed summary 只可用于旧 artifact 兼容和诊断，不得覆盖 pair-level 判断。

---

## 3. 有效覆盖谓词

### 3.1 冻结定义

定义二值有效覆盖：

\[
V_{ij}=\mathbf 1\!\left[
r_{ij}\in\mathcal R_{\mathrm{cov}}
\land d_{ij}\in\{\texttt{direct},\texttt{partial}\}
\land \gamma_{ij}>0
\land s_{ij}\ne\varnothing
\right].
\]

其中 Phase 0 固定：

\[
\mathcal R_{\mathrm{cov}}=
\{\texttt{support},\texttt{refute},\texttt{qualify},\texttt{mixed}\}.
\]

若当前 schema 使用近义标签，则在 schema adapter 中一次性映射到该 canonical vocabulary；不得在 solver 内散落数据集特化分支。

### 3.2 relation 的边界

relation 在 BACES 中只回答：“这个 evidence 是否对该 atom 构成可计数的实质性证据？”

它不回答：

- claim 最终真假；
- 两条 evidence 是否互相冲突；
- 多条 evidence 应如何合成结论；
- verifier 应如何处理 support 与 refute 的并存。

因此，同一个 atom 同时有 support 与 refute pair 时，BACES 仍只把该 atom 计为已覆盖一次。冲突检测与裁决不在本文方法边界内。

### 3.3 candidate coverage set

每条证据的覆盖集合为：

\[
Z_j=\{a_i\in\mathcal A: V_{ij}=1\}.
\]

一条 evidence 可同时覆盖多个 atoms；更新时必须一次加入全部 $Z_j$，不能只选择所谓 primary atom。

`background/context/irrelevant/none/insufficient` pair 不产生 coverage。它们可以作为候选元数据存在，但不能通过 `BRIDGE` 或其他操作获得隐式正收益。

---

## 4. 优化目标与预算

### 4.1 Weighted maximum coverage objective

对任意选择集合 $S\subseteq\mathcal E$，定义：

\[
F(S)=\sum_{i=1}^{m}w_i\,
\mathbf 1\!\left[a_i\in\bigcup_{e_j\in S}Z_j\right].
\]

BACES 求解：

\[
\begin{aligned}
\max_{S\subseteq\mathcal E}\quad &F(S)\\
\text{s.t.}\quad &|S|\le K_{\max},\\
&\sum_{e_j\in S}c_j\le B \quad \text{（若启用 token budget）}.
\end{aligned}
\]

未启用 token budget 时，第二个约束省略。只使用 $K_{\max}$ 时是 weighted max-$k$-cover；只使用 $B$ 时是 budgeted maximum coverage；同时使用时具有 cardinality 与 knapsack 两类上界。

### 4.2 `minmax` 的精确定义

`minmax(5,10)` 中：

- $K_{\max}=10$ 是 BACES 的优化约束；
- $K_{\min}=5$ 是 verifier 输入的 rendering floor，不是 maximum-coverage 问题必要的数学约束；
- 覆盖在少于 5 条 evidence 时饱和，不代表必须伪造额外覆盖收益；
- 为满足 rendering floor 加入的 evidence 必须标记为 `ZERO_GAIN_FILL`。

这一区分避免把“模型至少看 5 条文本”的工程策略误写成新的组合优化结构。

### 4.3 不采用连续质量饱和目标作为主定义

以下目标可以作为扩展消融：

\[
F_q(S)=\sum_i w_i\max_{e_j\in S}q_{ij}.
\]

但它应被称为 **quality-saturated coverage** 或 facility-location-style objective，而不是严格的 maximum coverage。Phase 0 主定义固定使用二值 $V_{ij}$，避免把 directness、confidence 和 relation 权重悄然混入目标。

map confidence、directness 与 retrieval score 在主方法中只用于：

1. 有效覆盖 gate；
2. 同一覆盖目标下的确定性 tie-break；
3. 单独的 map-quality ablation。

---

## 5. 计算性质与理论主张

### 5.1 可以主张

当 $w_i\ge0$ 时，$F$ 具有：

- normalized：$F(\varnothing)=0$；
- monotone：若 $S\subseteq T$，则 $F(S)\le F(T)$；
- submodular：对 $S\subseteq T$ 与 $e\notin T$，

\[
F(S\cup\{e\})-F(S)
\ge
F(T\cup\{e\})-F(T).
\]

一般规模的 BACES optimization 是 NP-hard；其 threshold decision version 可表述为 NP-complete。论文应写“NP-hard optimization problem”，不写含混的“NPC submodular problem”。

### 5.2 不能主张

- 不把 BACES 宣称为全新的组合优化问题；它是 fact-checking 场景中的 weighted budgeted maximum coverage formulation。
- 不把标准 maximum-coverage greedy 或 bitmask DP 宣称为新算法。
- 不把一般问题 NP-hard 当作当前实例只能近似求解的理由。
- 不在未证明时给联合 cardinality+knapsack 的当前 greedy 写 (1-1/e) 保证。
- 不把 map-implied coverage 当作真实 factual sufficiency。

### 5.3 NP-hardness 归约骨架

从 weighted maximum coverage 归约：给定 universe
$U=\{u_1,\ldots,u_m\}$、元素权重、子集族
$\mathcal C=\{C_1,\ldots,C_n\}$ 与选择上限 $K$，构造：

- 每个 universe element $u_i$ 对应一个 atom $a_i$，并令 $w_i$ 等于其元素权重；
- 每个 subset $C_j$ 对应一条 evidence $e_j$；
- 当且仅当 $u_i\in C_j$ 时令 $V_{ij}=1$；
- 令 $K_{\max}=K$，并关闭 token budget。

任意 subset selection 与 evidence selection 一一对应，且两者目标值完全相同。因此一般 BACES 至少与 weighted max-$k$-cover 一样困难。

从 budgeted maximum coverage 归约时，进一步令 evidence cost 等于原 subset cost、token budget 等于原预算，并取 $K_{\max}=n$。从 0/1 knapsack 归约时，让每条 evidence 只覆盖一个互不相同的 atom，atom weight 等于 item value，evidence cost 等于 item cost。由此 BACES 同时包含 coverage 与 knapsack 特例。

对 decision version，给定阈值 $L$，证书是一个 evidence subset；可在多项式时间验证预算与 $F(S)\ge L$，结合上述归约得到 NP-completeness。论文正文只需保留 max-coverage 归约，knapsack 特例与 decision-version 说明可放 appendix。

### 5.4 当前实例为何采用精确求解

当前 atomizer 代码固定 $m\le6$。因此只有至多 $2^6=64$ 个 atom-coverage masks。即使一般 BACES 是 NP-hard，当前系统实例也可以用 mask DP 精确求解，无需把 heuristic greedy 作为主求解器。

这也是 Phase 0 的强制 reviewer check：若论文只报告 greedy，审稿人可以合理追问“为何不在 64 个 masks 上给出 exact optimum？”

---

## 6. Canonical preprocessing

### 6.1 Oracle isolation

BACES 的 allowed inputs 只有：

- claim atoms 与 atom weights；
- candidate text、stable identity、token cost；
- retrieval metadata；
- pair-level evidence map；
- duplicate/provenance metadata；
- $K_{\max}$、可选 $B$ 和 rendering 参数。

以下字段必须从 solver 输入视图中物理隔离：

- `oracle_ordered_keys`；
- `oracle_selected_count`；
- gold evidence / gold rationale；
- verdict label；
- verifier logits、loss、correctness、reward；
- teacher/checkpoint 派生 signal；
- 由上述信息生成的 proxy 或融合字段。

`oracle_ordered_keys` 只可用于离线诊断和历史对照，不能进入 coverage、tie-break、dedup representative 选择或 rendering order。

### 6.2 Pre-optimization deduplication

为保持标准 maximum-coverage 语义，canonical dedup 在优化前完成：

1. 根据 `duplicate_group`、stable source/span identity、规范化文本等建立 duplicate class；
2. 每组只保留一个真实存在的 representative；
3. representative 使用冻结的确定性规则选出；
4. 不允许把不同 duplicate rows 的 atom sets 取 union 后生成不存在的“超级证据”。

建议 representative tie-break：

\[
\bigl(
-F(\{e_j\}),
-|Z_j|,
-\overline\gamma_j,
-\mathrm{retrieval}_j,
c_j,
\mathrm{stable\_key}_j
\bigr).
\]

其中所有浮点量先使用固定精度量化，`stable_key` 不依赖 candidate array index。

### 6.3 Stable key

`stable_key` 至少组合：dataset/event、document/report identity、sentence/span identity 与规范化 text hash。禁止以 `candidate_idx` 作为最终 tie-break，因为 pool merge 或序列化变化会改变 index。

### 6.4 Reachable atoms

定义：

\[
\mathcal A_{\mathrm{reach}}=\bigcup_{e_j\in\mathcal E}Z_j.
\]

同时报告：

- all-atom coverage：分母为全部 $\mathcal A$；
- reachable-atom coverage：分母为 $\mathcal A_{\mathrm{reach}}$；
- unreachable atom count。

不得把 retrieval/map 不可达的 atom 误记为 selector 未完成的 step，也不得只报告 reachable coverage 掩盖上游召回失败。

---

## 7. 主求解器：exact mask dynamic programming

### 7.1 State

将每条 candidate 的 $Z_j$ 编码为 $m$-bit mask $z_j$。逐个处理 candidates，维护：

\[
\mathrm{dp}[k,\mu]
=\text{达到 coverage mask }\mu\text{ 且使用 }k\text{ 条 evidence 的最小 token cost}.
\]

初始化：

\[
\mathrm{dp}[0,0]=0,
\]

其他状态为 $+\infty$。

对 candidate $e_j$ 做 0/1 更新：

\[
\mu'=\mu\lor z_j,
\qquad
\mathrm{dp}'[k+1,\mu']
=\min\bigl(
\mathrm{dp}'[k+1,\mu'],
\mathrm{dp}[k,\mu]+c_j
\bigr).
\]

使用 backpointer 重建真实 evidence subset。若 token cost 相同，按第 7.3 节的 canonical subset tie-break 保留一个解。

### 7.2 Feasible frontier

定义 mask 价值：

\[
W(\mu)=\sum_{i:\mu_i=1}w_i.
\]

对每个上界 $k\le K_{\max}$，从所有满足以下条件的状态中选最优集合 $S_k^*$：

\[
|S|\le k,
\qquad
\sum_{e\in S}c_e\le B\quad\text{（若启用）}.
\]

注意这里是“至多 $k$”而非“恰好 $k$”，因此 coverage plateau 不会被零收益 evidence 伪装成求解收益。

全预算最优值为：

\[
F^*=F(S_{K_{\max}}^*).
\]

### 7.3 Optimal-set tie-break

多个 subset 有相同 coverage 时，依次比较：

1. 更少的 selected evidence（rendering fill 尚未发生）；
2. 更低总 token cost；
3. 更高 valid pair 平均 confidence；
4. 更高原始 retrieval score；
5. 排序后的 stable-key tuple 字典序。

第 3–4 项只决定同一最优 coverage mask 的 canonical representative，不改变优化目标。论文必须把它们写为 tie-break，而非额外 objective gain。

### 7.4 Complexity

时间复杂度：

\[
O(nK_{\max}2^m).
\]

滚动数组空间复杂度：

\[
O(K_{\max}2^m),
\]

另加 backpointer 存储。当前 $m\le6$ 时，这一求解是实际可行且精确的。

### 7.5 Solver guarantee

在以下前提下，DP 返回冻结目标的全局最优 coverage subset：

- candidates 已完成 canonical dedup；
- atom weights 与 token costs 固定；
- coverage 是二值 mask union；
- 约束只有 count upper bound 与可选 additive token budget。

该 guarantee 只针对 map-defined BACES objective，不代表所选证据在人工判断下事实充分，也不保证 verifier label 正确。

---

## 8. `minmax` controller 与 rendering fill

### 8.1 Coverage-core size

从 DP frontier 选择达到全预算最优覆盖的最小 evidence 数：

\[
k_{\mathrm{cov}}
=\min\{k\le K_{\max}:F(S_k^*)=F^*\}.
\]

coverage core 为：

\[
S_{\mathrm{cov}}=S_{k_{\mathrm{cov}}}^*.
\]

### 8.2 Rendering floor

若 $|S_{\mathrm{cov}}|\ge K_{\min}$，则：

\[
S_{\mathrm{sel}}=S_{\mathrm{cov}}.
\]

若 $|S_{\mathrm{cov}}|<K_{\min}$，从未选 canonical candidates 中选择 marginal coverage 为 0、且加入后仍满足 token budget 的 candidates，再按确定性 fill key 加入：

\[
K_{\min}-|S_{\mathrm{cov}}|
\]

条 evidence，并将 operation 标为 `ZERO_GAIN_FILL`。fill key 建议为：有效 pair 质量、retrieval relevance、更低 token cost、stable key。

fill evidence 不得提高 `solver_coverage_value`。如果预算内没有足够的 zero-gain candidates，则不违反 budget 来强行补齐，而是记录 `min_count_unreachable=true`。若一个预算可行的 fill item 意外覆盖新 atom，说明 $S_{\mathrm{cov}}$ 或 coverage 计算有 bug。

### 8.3 数量契约

固定使用以下名称：

- $K_{\mathrm{pool}}$：dedup 后可供选择的候选数；
- $K_{\mathrm{cov}}$：DP coverage core 数量；
- $K_{\mathrm{sel}}$：加入 rendering fill 后的选中数量；
- $K_{\mathrm{final}}$：经过 prompt max-length guard 后实际进入 verifier 的数量。

不变量：

\[
K_{\mathrm{cov}}\le K_{\mathrm{sel}}\le
\min(K_{\max},K_{\mathrm{pool}}),
\qquad
K_{\mathrm{final}}\le K_{\mathrm{sel}}.
\]

若 $K_{\mathrm{pool}}<K_{\min}$，或 token budget 无法容纳 $K_{\min}$ 条 evidence，允许 $K_{\mathrm{sel}}<K_{\min}$，并记录 `min_count_unreachable=true`。

### 8.4 与旧 prefix stop 的区别

旧 minmax 逻辑沿一个 heuristic full order 取最短前缀。冻结后的主方法直接优化每个 feasible subset，不要求最终解必须是某个预先生成全排序的前缀。这样才能让“set selection”与“display order”真正分离。

---

## 9. Evidence-slate rendering

### 9.1 顺序不是推理结构

BACES 输出集合。为了序列化到 prompt，使用确定性 display order；本文称其为 **ordered evidence slate**，不得称 chain/path。

### 9.2 主 display order

对 $S_{\mathrm{cov}}$ 进行 restricted marginal-coverage ordering：从空 covered mask 开始，每步在已选集合内部选择新增 atom weight 最大的 evidence；平局按 valid-pair quality、retrieval score、token cost、stable key 处理。

随后追加 `ZERO_GAIN_FILL` items。

这一排序只提供：

- 可复现的 prompt 位置；
- first-cover trace；
- 便于人类审计的 coverage progression。

它不改变 $S_{\mathrm{sel}}$，也不参与 DP 最优性证明。

### 9.3 必须做的 order control

对同一个 selected set 至少比较：

- marginal-coverage order；
- original retrieval order；
- fixed-seed shuffle。

若三者 verifier 结果无显著差异，论文应明确结论是“set selection 有效、顺序作用有限”，不能继续宣称 evidence organization/chain-ordering 是核心贡献。

---

## 10. Prompt max-length guard

max-length guard 是 BACES 后的独立层：

```text
DP coverage set
  -> rendering floor
  -> display order
  -> tokenizer-aware max-length guard
  -> final verifier evidence slate
```

若 guard 移除 evidence，必须同时记录：

- removed stable keys；
- removed operation (`COVER` 或 `ZERO_GAIN_FILL`)；
- `selected_coverage_value`；
- `final_coverage_value`；
- `coverage_lost_by_truncation`；
- $K_{\mathrm{sel}}$ 与 $K_{\mathrm{final}}$。

论文中的 selector-quality 指标以 `selected` 口径报告，同时单列 prompt-realized `final` 口径。禁止把 guard 后丢失的 coverage 归因于 selector。

---

## 11. Trace schema v0.2

### 11.1 Canonical trace

```json
{
  "schema_version": "baces_trace_v0_2",
  "selection_policy": "baces_exact_mask_dp_v0_2",
  "event_id": "...",
  "atom_ids": ["A1", "A2"],
  "atom_weights": {"A1": 1.0, "A2": 1.0},
  "reachable_atom_ids": ["A1", "A2"],
  "unreachable_atom_ids": [],
  "candidate_pool_size_raw": 20,
  "candidate_pool_size_dedup": 18,
  "max_evidence_count": 10,
  "min_rendered_evidence_count": 5,
  "token_budget": null,
  "coverage_core_keys": ["..."],
  "selected_keys": ["..."],
  "display_ordered_keys": ["..."],
  "k_cov": 2,
  "k_sel": 5,
  "solver_coverage_value": 2.0,
  "selected_coverage_value": 2.0,
  "all_atom_coverage": 1.0,
  "reachable_atom_coverage": 1.0,
  "zero_gain_fill_count": 3,
  "steps": []
}
```

### 11.2 Canonical step

```json
{
  "step": 1,
  "operation": "COVER",
  "candidate_key": "...",
  "candidate_idx_legacy": 7,
  "valid_coverage_atom_ids": ["A1", "A2"],
  "newly_covered_atom_ids": ["A1", "A2"],
  "already_covered_atom_ids": [],
  "marginal_coverage_value": 2.0,
  "cumulative_coverage_value": 2.0,
  "cumulative_all_atom_coverage": 1.0,
  "cumulative_token_cost": 143,
  "target_coverage_reached": true
}
```

Phase 0 只允许两类 operation：

- `COVER`：加入时 marginal coverage (>0)；
- `ZERO_GAIN_FILL`：只为 rendering floor 加入，marginal coverage (=0)。

### 11.3 禁止字段

新 trace 不生成：

- `atom_states`；
- `conflicted_atom_ids`；
- `bridge_from / bridge_to`；
- `corroboration_count`；
- `reasoning_transition`；
- 任何虚构的 evidence dependency。

### 11.4 兼容别名

为避免一次性破坏旧 prompt-builder，可在 adapter 层临时提供：

```text
target_resolved       := target_coverage_reached
resolved_atom_rate    := cumulative_all_atom_coverage
unresolved_atom_ids   := reachable_atom_ids - covered_atom_ids
```

别名只能出现在 compatibility view，不能出现在论文方法定义或 canonical v0.2 trace。兼容层不得伪造 `atom_states` 或 `conflicted`。

---

## 12. 可扩展 baseline：marginal coverage greedy

当未来取消 $m\le6$ 或候选池显著扩大时，可使用 greedy baseline。

对当前 covered set $C_t$，定义：

\[
\Delta(e_j\mid C_t)
=\sum_{a_i\in Z_j\setminus C_t}w_i.
\]

cardinality-only greedy 每步选择最大 $\Delta$ 的 candidate；token-budget heuristic 可按 $\Delta/c_j$ 排序，并显式比较 best singleton。

冻结边界：

- 它是 scalability baseline，不是当前主 solver；
- cardinality-only 标准 greedy 的经典近似性质只能在相应标准假设下陈述；
- 当前 density heuristic 在联合 count+token 约束下不默认声称 (1-1/e)；
- 必须用 exact DP 在当前数据上报告 greedy optimality gap。

建议指标：

\[
\mathrm{gap}
=\frac{F(S^*)-F(S_{\mathrm{greedy}})}{\max(F(S^*),\epsilon)}.
\]

---

## 13. Phase 0 实现验收测试

### 13.1 Coverage correctness

1. 单 evidence 多 atom pair 时，一步覆盖全部 valid atoms。
2. `background/context/irrelevant/none/insufficient` 不产生 coverage。
3. support/refute 同时出现时 atom 只计一次，不生成 conflict state。
4. key span 为空或 confidence 非正的 pair 不产生 coverage。
5. candidate summary 与 pair rows 冲突时，以 pair rows 为准。

### 13.2 Exactness

1. 小实例穷举所有 subsets，DP objective 与穷举最优值完全一致。
2. count-only、token-only、count+token 三种约束分别测试。
3. 多个相同 coverage mask 的 subset 按冻结 tie-break 返回一致解。
4. 重排 candidate array 后，selected stable keys 完全相同。

### 13.3 Dedup

1. duplicate class 只保留一个 representative。
2. 不得 union duplicate members 的 coverage。
3. duplicate metadata 缺失时使用 stable text/span identity 回退。

### 13.4 Oracle poison tests

向同一输入写入、删除或随机改写下列字段，输出必须 bitwise-equivalent（除被过滤字段的审计日志外）：

- `oracle_ordered_keys`；
- gold label/evidence；
- verifier scores；
- learned weight/reward fields。

### 13.5 Budget and rendering

1. $K_{\mathrm{cov}}\le K_{\mathrm{sel}}\le K_{\max}$。
2. coverage 在 $K_{\min}$ 前饱和时，只添加 `ZERO_GAIN_FILL`。
3. token budget 启用时 selected total cost 不超过 $B$。
4. prompt guard 后 $K_{\mathrm{final}}\le K_{\mathrm{sel}}$，coverage loss 可回放。
5. pool 或 token budget 不能容纳 $K_{\min}$ 条 evidence 时显式标记而不复制 evidence。

### 13.6 Determinism

相同 canonical input 在不同进程、不同候选序列化顺序与不同 Python hash seed 下返回相同：

- coverage core stable keys；
- selected set；
- display order；
- objective values；
- trace steps。

---

## 14. 实验边界与必补展示

### 14.1 主比较

matched candidate pool、matched (K/B)、matched verifier 下至少包含：

1. retrieval top-$k$；
2. MMR / existing static selector；
3. map-quality greedy；
4. BACES marginal coverage greedy；
5. BACES exact mask DP；
6. historical learned-marginal selector（只作既有 baseline，不作为主方法组成）。

### 14.2 必须拆分的结果层

- upstream reachability：candidate recall、reachable atom rate；
- map quality：人工 pair relation/directness 质量；
- selector conditional quality：在同一 pool 上的 coverage、redundancy、cost；
- prompt-realized quality：truncation 后 coverage；
- downstream utility：verifier Macro-F1 / dataset-specific official metric。

### 14.3 关键表/图

1. Coverage–count Pareto curve：$k=1,\ldots,K_{\max}$ 的 $F(S_k^*)$。
2. Coverage–token Pareto curve：不同 $B$ 下的 exact frontier。
3. Greedy optimality-gap distribution。
4. Selected-set order control：marginal/retrieval/shuffle。
5. Internal-map coverage 与 human/gold coverage 的相关性及偏差。
6. $K_{\mathrm{cov}},K_{\mathrm{sel}},K_{\mathrm{final}}$ 分布和 truncation loss。
7. 上游不可达与 selector 失误的 error decomposition。

### 14.4 外部评价边界

内部 LLM map 同时定义目标又评价目标会构成循环验证。因此主论文不能只报告 `resolved_atom_rate`。至少需要一种独立评价：

- human atom–evidence coverage；
- 数据集 gold evidence 对齐；
- 独立标注协议下的 sufficiency/redundancy；
- 与 map generator 不同来源的审计评价。

外部评价仍然只能证明 evidence-set quality；除非新增证据依赖标注，否则不能改称 multihop/chain quality。

---

## 15. 论文 novelty 与 claim 边界

### 15.1 可以作为贡献的内容

1. 将 claim decomposition 与 typed atom–evidence alignment 接到一个明确的 budgeted coverage interface。
2. 把 fact-checking evidence selection 从 independent relevance top-$k$ 重述为 set-level marginal coverage selection。
3. 在当前小 atom universe 上给出可审计的 exact coverage frontier，并将 selection、rendering 与 truncation 分层。
4. 用外部标注证明 map-defined coverage 与实际证据质量之间的关系。
5. 系统展示 coverage/cost/downstream-performance trade-off 及跨数据集稳定性。

### 15.2 不能作为贡献的内容

1. “提出一个新的 maximum coverage 算法”；
2. “解决 multi-hop fact verification”；
3. “学习 evidence dependencies”；
4. “消解互相冲突的证据”；
5. “生成 faithful reasoning chain”；
6. “exact DP 克服了 NP-hardness”；
7. 仅凭内部 map coverage 声称 factual sufficiency。

### 15.3 对 AAAI novelty 的诚实判断

冻结后的 BACES 比 Evidence-Chain 语义更准确，但也暴露一个事实：**weighted maximum coverage formulation + standard DP/greedy 本身不足以被包装成新的组合优化算法。**

因此 AAAI 方法贡献必须主要由“fact-checking-specific problem interface + noisy typed coverage construction + exact/controlled evaluation + downstream evidence”支撑。若希望主张真正的新算法，需要在 Phase 0 之后另行引入并真实实现至少一种非平凡结构，例如 robust/noisy coverage、group/provenance constraints、uncertainty-aware objective 或其他可证明的新约束；不能借用 chain/conflict 术语代替实际算法结构。

这类扩展在实现、公式和实验完成前不属于 BACES v0.2，也不得预写进主 claim。

---

## 16. 论文替换词典

| 旧表述 | 冻结后表述 |
|---|---|
| evidence chain | selected evidence set / ordered evidence slate |
| chain construction | evidence-set selection |
| reasoning step | selection/display step |
| atom resolution | atom coverage |
| resolved atom | covered atom under the evidence map |
| unresolved atom | uncovered/retrieval-unreachable atom |
| state-conditioned utility | marginal coverage gain |
| bridge evidence | zero-coverage/context evidence（不进入主目标） |
| contrast/corroborate | relation metadata（不作为 operation） |
| sufficiency-aware prefix | coverage-frontier selection + rendering floor |
| chain length | selected evidence count |
| chain order | display order |
| chain faithfulness | external evidence-set quality |

---

## 17. 推荐的 Method 章节骨架

```text
3 Method
  3.1 Task Boundary and System Overview
  3.2 Claim Atoms and Flat Candidate Pool
  3.3 Typed Atom-Evidence Coverage Interface
  3.4 Budgeted Atom-Coverage Evidence Selection
      3.4.1 Objective and Constraints
      3.4.2 Properties and General Complexity
  3.5 Exact Mask-DP Solver for the Bounded Atom Universe
  3.6 Coverage Frontier, Rendering Floor, and Evidence Slate
  3.7 Interface to the Verifier
```

Algorithm 1 只描述：dedup、mask construction、DP、frontier selection、fill 标记与 display ordering。不要在 Algorithm 1 中混入 atomization API、retrieval、map generation、prompt builder 或 verifier。

---

## 18. Phase 0 冻结清单

以下项目自 v0.2 起固定：

- [x] 主对象是 flat evidence set，不是 evidence chain。
- [x] 不建模 multihop、evidence dependency 或 conflict resolution。
- [x] 问题名为 BACES，数学目标为 binary weighted maximum coverage。
- [x] 主 atom weights 使用 uniform 1.0；importance 只做 sensitivity。
- [x] pair-level map 是唯一 coverage source。
- [x] 一条 evidence 同时更新全部 valid atoms。
- [x] 先 dedup，后优化；不合并 duplicate coverage。
- [x] `oracle_ordered_keys` 等字段与 solver 物理隔离。
- [x] $K_{\max}$ 与可选 token $B$ 是优化约束；$K_{\min}$ 是 rendering floor。
- [x] 当前 $m\le6$ 使用 exact mask DP 为主 solver。
- [x] marginal greedy 仅为 scalability/optimality-gap baseline。
- [x] selected set 与 display order 分离。
- [x] operation 只保留 `COVER` 与 `ZERO_GAIN_FILL`。
- [x] prompt max-length guard 是独立后处理层。
- [x] 内部 map coverage 与外部 evidence quality 分开评价。
- [x] 不声称新 maximum-coverage 算法或 reasoning-chain 能力。

任何改变上述条目的方案都需要新版本号，并同步修改问题定义、solver、trace schema、实验矩阵与论文 claim；不能只改名或只改公式。
