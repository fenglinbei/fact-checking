# BAREC / SCSG Phase 0 算法规范 v0.1（候选冻结版）

> **Superseded notice（2026-07-12）**：作者确认本文不解决 multi-hop reasoning、evidence dependency 或 conflict reasoning。v0.2 曾将对象收缩为 flat binary evidence-set selection，随后又因真实方法存在非逻辑的 prefix order 与 partial/direct 质量升级而被取代。本候选版仅保留为决策审计记录；最新实现规范以 `atom_coverage_sequencing_phase0_spec_v0.3.md` 为准。

状态：**Superseded — 不再实现或引用为主方法**

适用范围：论文方法定义、`structural_state_greedy` 实现、trace schema、prompt-prefix 构建与 Phase 0 验收

不修改：现有 `transition_v0_1`、`learned_marginal_proxy` 和历史 artifacts

本文档将 BAREC 问题与 SCSG 算法冻结为一套可实现、可回放、可审计的规范。除第 14 节列出的六项作者决策外，其余接口与不变量视为 v0.1 固定项。

---

## 1. 命名、研究对象与边界

### 1.1 固定命名

- 问题：**Budgeted Atom-Resolving Evidence-Chain Construction（BAREC）**
- 算法：**State-Conditioned Structural Greedy（SCSG）**
- selection policy：`structural_state_greedy`
- selector name：`barec_scsg_v0_1`
- canonical trace schema：`barec_trace_v0_1`
- canonical step schema：`scsg_step_v0_1`

### 1.2 方法对象

给定 claim atoms、Atom-Union candidate pool 与 pair-level typed atom–evidence alignments，SCSG 生成一个确定性的完整 evidence ordering；随后 BAREC prefix controller 在 count budget 下选择该 ordering 的一个前缀，供 verifier rendering 使用。

方法边界固定为：

```text
claim atoms
→ Atom-Union candidate pool
→ pair-level atom–evidence map
→ SCSG full structural ordering
→ BAREC sufficiency-aware prefix
→ prompt max-length guard
→ verifier
```

### 1.3 v0.1 明确不做

- 不训练 selector weights；
- 不生成 preference pairs；
- 不读取 verdict label、gold evidence、oracle order、verifier score 或 checkpoint；
- 不声称全局最优、minimum chain 或 factual sufficiency；
- 不把 candidate-level collapsed map summary 当作状态更新依据；
- 不把 selector metadata 渲染进 verifier prompt；
- 不对完整 SCSG 声称 coverage-only greedy 的近似保证。

---

## 2. 当前实现审计结论与重建理由

现有 `transition_v0_1` 不能直接重命名为 SCSG，原因包括：

1. 一条 evidence 即使有效对齐多个 atoms，也只选择并更新一个 primary atom。
2. OPEN 没有严格的 directness/confidence/span gate，context/none 也可能推进状态。
3. 当前 BRIDGE 大量来自 irrelevant/none 与 insufficient/none，并不具有真正的 bridge endpoints。
4. CORROBORATE 不要求新 provenance，同一 report 的重复表述也可能被当作独立佐证。
5. `U/S/R/Q/C` 被当作可覆盖的单一状态，无法保留多个 relation 的历史。
6. 最终 tie-break 使用 candidate index，且字符串排序会出现 index 10 排在 2 前的问题。
7. 旧 proxy/fusion 相关字段可能包含 oracle 或 verifier-derived signal。
8. selector early stop 与后续 prompt prefix policy 尚未形成严格的三层数量契约。

LIAR-RAW validation artifact 的只读审计显示：

- 1,067 个 candidates 同时具有多个 valid resolving atoms，分布于 286/1,274 个样本；
- 旧 trace 中没有 multi-atom step；
- 旧 4,453 个 steps 中 BRIDGE 为 2,859，其中 1,859 个是 `irrelevant/none`；
- 40 个 context/none alignments 被用于 OPEN、CONTRAST 或 CORROBORATE。

因此 v0.1 必须使用新 policy 与新 trace schema，而不是修改名字后复用旧状态机。

---

## 3. BAREC 输入与输出

### 3.1 输入实例

一个 BAREC 实例定义为：

\[
\mathcal I=
(c,\mathcal A,\mathcal C,M,\ell,k_{\min},k_{\max},B_{\mathrm{ctx}}),
\]

其中：

- \(c\)：待核查 claim；
- \(\mathcal A=\{a_i\}_{i=1}^{m}\)：claim atoms；
- \(\mathcal C=\{u_j\}_{j=1}^{n}\)：candidate evidence units；
- \(M\)：pair-level atom–evidence map；
- \(\ell(u_j)\)：evidence-content token cost；
- \(k_{\min},k_{\max}\)：prefix count controller；
- \(B_{\mathrm{ctx}}\)：verifier prompt 的最终上下文上限。

理论定义不固定 \(m\)；当前实现中的 `m_max=6` 仅作为工程配置。

### 3.2 输出

SCSG 输出完整 ordering：

\[
\mathcal O=[u_1,\ldots,u_{K_{\mathrm{order}}}].
\]

BAREC controller 输出其前缀：

\[
\mathcal T_B=mathcal O_{1:K_{\mathrm{sel}}},
\qquad
K_{\mathrm{sel}}\le k_{\max}.
\]

generic prompt builder 可能继续从尾部截断，得到 verifier 最终可见前缀：

\[
\mathcal T_{\mathrm{final}}
=
\mathcal T_{B,1:K_{\mathrm{final}}}.
\]

必须始终满足：

\[
K_{\mathrm{final}}
\le K_{\mathrm{sel}}
\le K_{\mathrm{order}}
\le K_{\mathrm{pool}}.
\]

---

## 4. Selector allowed-field view

### 4.1 原则

SCSG 不直接接收原始 JSON row。实现必须先构造一个显式 allowlist view；不在 allowlist 中的字段即使存在，也不能进入 validity、state、score、tie-break 或 stop 计算。

### 4.2 允许字段

Claim atom：

- `atom_id`；
- `proposition` / `text`；
- `importance`。

Candidate identity 与 provenance：

- `candidate_uid`、`candidate_key`；
- `text` / `evidence_text` / `canonical_text`；
- `report_id`、`source_group`、`source_report`；
- `sent_start`、`sent_end`、`chunk_id`；
- `duplicate_group`；
- `num_tokens` 或合法 token-cost 字段。

Retrieval tie-break：

- `union_pool_rank`；
- `atom_pool_rank`、`baseline_rank` 仅作缺失回退；
- `hybrid_score`、`baseline_hybrid_score`；
- `dense_score`、`lexical_score`、`bm25_score` 仅作 diagnostics。

Pair-level map：

- `candidate_atom_alignments[*].evidence_id`；
- `atom_id`；
- `relation`；
- `directness`；
- `confidence`；
- `key_spans`；
- `duplicate_group`。

### 4.3 禁止字段

禁止所有：

- `oracle_*`；
- `gold_label`、`label`；
- `verifier_*`；
- `oracle_likelihood_score`；
- `direct_ce_score`；
- `fusion_refit_score`；
- `evidence_map_base_score`；
- 来源不明确的 `base_score`；
- learned selector weights、utility score、reward；
- candidate index / evidence-local `E01` 编号作为 tie-break。

主方法输入 manifest 必须证明 candidate source 为 Atom-Union 或另一条明确不含 oracle/verifier signal 的 approved source。旧 fusion candidate source 不得直接作为 SCSG 主输入。

---

## 5. Canonical schema 与输入门禁

### 5.1 Claim atom

每个 atom 规范化为：

```json
{
  "atom_id": "A1",
  "proposition": "...",
  "importance": 1.0
}
```

要求：

- `atom_id` 非空且在样本内唯一；
- proposition 非空；
- importance 规范化到 \((0,1]\)，缺失时为 1.0；
- 主实验不允许静默构造 `A1=Full claim`。

### 5.2 Candidate stable key

`evidence_id=E01...` 是 annotation-pool 局部编号，不可用于确定性。

canonical stable key 定义为：

```text
sha1(
  canonical_report_or_source_id
  || sent_start
  || sent_end
  || normalized_evidence_text
)
```

其中 normalized text 至少执行 Unicode normalization、lowercase、whitespace collapse。所有最终 tie-break 使用该 stable key 的字典序。

### 5.3 Provenance ID

按以下优先级构造：

1. `report:<report_id>`；
2. `source_group` 中明确的 `report:*` 或 `domain:*`；
3. `source_report.report_id`；
4. 规范化 source domain；
5. 否则为 `UNKNOWN`。

禁止使用 `candidate_uid` 或 `candidate_key` 伪造独立来源。`UNKNOWN` 可以保留 evidence，但不能产生 source novelty 或 CORROBORATE gain。

### 5.4 Duplicate ID

duplicate identity 依次由以下字段构造：

1. 非空 `duplicate_group`；
2. 相同 stable key；
3. 相同规范化全文 hash；
4. 相同 report/span identity。

同一 candidate 的 pair rows 若给出多个不同的非空 duplicate groups，输入门禁失败；不静默择一。

### 5.5 Pair-level source of truth

`candidate_atom_alignments` 是唯一状态来源。以下 candidate-level summary 只允许展示或旧代码兼容：

- `map_relation`；
- `map_directness`；
- `map_confidence`；
- `covered_atom_ids`；
- `evidence_map_quality_score`。

原因是 collapsed summary 分别选择最高 relation、最高 directness 与最大 confidence，可能组合出一条实际不存在的 pair。

### 5.6 一 evidence–atom 一行

每个 `(candidate_stable_key, atom_id)` 最多存在一行 canonical pair。若同一 pair 同时出现两个 relation/directness，输入门禁失败；不得通过 relation priority 静默消解。

### 5.7 已知 artifact migration gate

现有 LIAR validation map 中已发现：

- 1 个 candidate 存在同一 evidence–atom 的多 relation/directness rows；
- 5 个 candidates 的 pair rows 给出不一致的 non-empty duplicate groups。

实现 SCSG 前必须通过确定性的上游修复重新导出这些 rows，或将对应样本列入明确 quarantine report。不得在 selector 内临时挑一个 relation/duplicate ID，以免同一算法在不同 pair 顺序下产生不同结果。

---

## 6. Relation、directness 与 alignment predicates

### 6.1 Canonical relation

上游 relation：

```text
support, refute, qualify, mixed,
insufficient, background, irrelevant
```

SCSG canonicalization：

```text
support      → S
refute       → R
qualify      → Q
mixed        → Q
insufficient → NON_RESOLVING
background   → NON_RESOLVING
irrelevant   → INVALID
```

`mixed` 不解释为同时加入 S 与 R，而解释为非极性的 qualified/mixed observation；它不会单独制造 conflict。

### 6.2 Canonical directness

```text
direct  → DIRECT
partial → PARTIAL
context → CONTEXT
none    → NONE
```

### 6.3 Valid resolving alignment

候选冻结定义为：

\[
V^{\mathrm{resolve}}_{ij}=1
\iff
\begin{cases}
r_{ij}\in\{S,R,Q\},\\
d_{ij}\in\{\mathrm{DIRECT},\mathrm{PARTIAL}\},\\
0<\gamma_{ij}\le1,\\
\mathrm{key\_spans}_{ij}\neq\varnothing,\\
a_i\text{ 与 }u_j\text{ ID 合法。}
\end{cases}
\]

v0.1 推荐不设置 0.5/0.7 的 confidence 硬阈值。Confidence 尚未校准，仅在结构增益完全相同时用于 tie-break；缺失或 0 视为 invalid。Phase 1 必须补充 \(\tau_\gamma\in\{0^+,0.3,0.5,0.7\}\) 敏感性。

### 6.4 Valid context alignment

候选冻结定义为：

\[
V^{\mathrm{context}}_{ij}=1
\iff
\begin{cases}
r_{ij}\in\{\mathrm{background},\mathrm{insufficient},Q\},\\
d_{ij}=\mathrm{CONTEXT},\\
\gamma_{ij}>0,\\
\mathrm{key\_spans}_{ij}\neq\varnothing,\\
\mathrm{provenance}(u_j)\neq\mathrm{UNKNOWN}.
\end{cases}
\]

Context alignment 不更新 relation state，不提高 address level，也不进入 saturation 计算。

`irrelevant`、`none`、无 span、无 atom id 或非法枚举永远没有结构收益。

---

## 7. SCSG atom state

### 7.1 Canonical state

对每个 atom \(a_i\)，状态为：

\[
H_i^{(t)}=
\left(
\{\lambda_i^{(t)}(r)\}_{r\in\{S,R,Q\}},
\{\Pi_i^{(t)}(r)\}_{r\in\{S,R,Q\}},
X_i^{(t)}
\right),
\]

其中：

- \(\lambda_i(r)\in\{\mathrm{NONE},\mathrm{PARTIAL},\mathrm{DIRECT}\}\)：该 relation 已观察到的最高 directness；
- \(\Pi_i(r)\)：该 relation 已选 evidence 的 provenance IDs；
- \(X_i\)：已选 context provenance IDs。

DIRECTNESS 顺序为：

\[
\mathrm{NONE}<\mathrm{PARTIAL}<\mathrm{DIRECT}.
\]

初始化：所有 \(\lambda_i(r)=\mathrm{NONE}\)，所有 provenance sets 为空。

### 7.2 派生量

Stance set：

\[
Y_i^{(t)}=
\{r\in\{S,R,Q\}:\lambda_i^{(t)}(r)>\mathrm{NONE}\}.
\]

Atom address level：

\[
L_i^{(t)}=
\max_{r\in\{S,R,Q\}}\lambda_i^{(t)}(r).
\]

Conflict：

\[
\mathrm{Conflict}_i^{(t)}
=
\mathbf 1[S\in Y_i^{(t)}\land R\in Y_i^{(t)}].
\]

用于旧工具展示的 compatibility label：

| 条件 | label |
|---|---|
| \(Y_i=\varnothing\) | U |
| S 与 R 同时存在 | C |
| Q 存在且无 S/R conflict | Q |
| 仅 S | S |
| 仅 R | R |

该 label 仅为派生展示，不再作为真实状态容器。

### 7.3 为什么必须 per-relation

`direct-support + partial-refute` 与 `partial-support + direct-refute` 都可得到 stance set `{S,R}` 和 atom-level DIRECT，但它们对下一条 evidence 的边际价值不同。只有 \(\lambda_i(S),\lambda_i(R),\lambda_i(Q)\) 分开保存，状态才足以支持下一步决策和完整回放。

---

## 8. Pool-attainable address target

### 8.1 最高可达层级

在 selection 前，仅根据 allowed pair view 计算每个 atom 在 candidate pool 中的最高可达 address level：

\[
b_i^\star=
\begin{cases}
\mathrm{DIRECT},&\exists u_j:V_{ij}^{\mathrm{resolve}}=1\land d_{ij}=\mathrm{DIRECT},\\
\mathrm{PARTIAL},&\nexists u_j:V_{ij}^{\mathrm{resolve}}=1\land d_{ij}=\mathrm{DIRECT},\quad
\exists u_j:V_{ij}^{\mathrm{resolve}}=1\land d_{ij}=\mathrm{PARTIAL},\\
\mathrm{NONE},&\text{otherwise.}
\end{cases}
\]

### 8.2 Saturation

对 reachable atom：

\[
\mathrm{Sat}_i^{(t)}
=
\mathbf 1[L_i^{(t)}\ge b_i^\star].
\]

加权 best-attainable saturation：

\[
\rho_{\mathrm{attain}}^{(t)}
=
\frac{
\sum_{i:b_i^\star>\mathrm{NONE}}\omega_i\mathrm{Sat}_i^{(t)}
}{
\sum_{i:b_i^\star>\mathrm{NONE}}\omega_i
}.
\]

若没有 reachable atom，则 \(\rho_{\mathrm{attain}}=0\)，不得自动视为已满足。

v0.1 推荐 target：

\[
\rho_{\mathrm{attain}}=1.0.
\]

该指标命名为 **map-implied best-attainable address saturation**，不能称 factual resolution 或 external sufficiency。

同时必须独立报告：

- total direct-address coverage；
- total direct-or-partial address coverage；
- unreachable atom rate；
- conflict capture rate。

---

## 9. Pair-level structural effects

所有 effects 都相对于同一个 pre-state \(H^{(t-1)}\) 计算。一个 pair 可以同时产生多个 effect，例如 direct refute 既可将 relation level 从 NONE 提升为 DIRECT，也可与既有 support 形成 polar contrast。

### 9.1 DIRECT-OPEN / DIRECT-UPGRADE

若 valid-direct pair 使 atom-level address level 从 NONE/PARTIAL 提升到 DIRECT：

```text
DIRECT_OPEN     : NONE    → DIRECT
DIRECT_UPGRADE  : PARTIAL → DIRECT
```

二者共同计入 direct-address gain。

### 9.2 PARTIAL-OPEN

若 atom 当前完全未被 resolving evidence addressed，valid-partial pair 使其首次达到 PARTIAL：

```text
PARTIAL_OPEN: NONE → PARTIAL
```

若该 atom 在 pool 中存在 valid-direct evidence，则 partial 仍可进入 ordering，但不能使该 atom saturated。

### 9.3 POLAR-CONTRAST

仅以下关系构成 polar contrast：

\[
S\leftrightarrow R.
\]

`qualify/mixed` 不算 CONTRAST，也不会单独产生 conflict。

### 9.4 RELATION-EXTEND

在 atom 已有 stance 后首次引入新的、非极性对立 relation，例如：

- S/R 后首次出现 Q；
- Q 后首次出现 S 或 R，但尚未形成 S–R polar contrast。

使用单独的 RELATION_EXTEND/QUALIFY 层，避免把 qualification 错写成 contradiction。

### 9.5 CORROBORATE

CORROBORATE 仅在以下条件同时成立时产生一次增益：

1. 当前 pair relation 已在 \(Y_i\) 中；
2. 当前 provenance 已知；
3. 当前 provenance 不在 \(\Pi_i(r)\) 中；
4. 当前 evidence 不是 duplicate；
5. 该 relation 在此前只有一个独立 provenance。

即只有从 1 个来源增加到 2 个来源时产生 corroboration gain；第三个及以后来源仍进入 provenance ledger，但不再产生新的结构增益。

### 9.6 CONTEXT

v0.1 删除 BRIDGE 术语，因为当前 schema 没有明确的 bridge endpoints。

CONTEXT：

- 不更新 \(\lambda_i(r)\)；
- 不更新 stance set；
- 不提高 address coverage 或 saturation；
- 仅对尚无 resolving address 的 atom 产生低层增益；
- 每个 atom 至多产生一次 context gain；
- 排在 CORROBORATE 后、FALLBACK 前。

### 9.7 FALLBACK

没有任何结构 effect 的非 duplicate candidate 进入 FALLBACK。FALLBACK 只使用 approved retrieval fields、token cost 与 stable key 排序，不改变 atom state。

---

## 10. Candidate-level structural key

### 10.1 原子权重

\(\omega_i\in(0,1]\) 来自规范化 atom importance；缺失时为 1.0。v0.1 不学习或搜索 atom weights。

### 10.2 Gain components

对 candidate \(u\)，基于同一 pre-state 计算：

\[
G_D(u)=\sum_i\omega_i\mathbf 1[u\text{ directly opens/upgrades }a_i],
\]

\[
G_P(u)=\sum_i\omega_i\mathbf 1[u\text{ partially opens }a_i],
\]

\[
G_C(u)=\sum_i\omega_i\mathbf 1[u\text{ adds polar contrast to }a_i],
\]

\[
G_E(u)=\sum_i\omega_i\mathbf 1[u\text{ extends a new non-polar relation for }a_i],
\]

\[
G_R(u)=\sum_i\omega_i\mathbf 1[u\text{ creates second-source corroboration for }a_i],
\]

\[
G_X(u)=\sum_i\omega_i\mathbf 1[u\text{ provides the first valid context for }a_i].
\]

Pair confidence tie-break：

\[
Q(u)=
\sum_{i:\,u\text{ has positive structural effect on }a_i}
\omega_i\gamma_{ij}.
\]

### 10.3 Lexicographic key

结构 candidate 的 key 冻结为：

\[
\mathbf K_t(u)=
\Big(
G_D,
G_P,
G_C,
G_E,
G_R,
G_X,
Q,
N_{\mathrm{source}},
-r_{\mathrm{retrieval}},
s_{\mathrm{retrieval}},
-\ell(u)
\Big),
\]

最大化 \(\mathbf K_t(u)\)，数值完全相同时选择 stable key 字典序最小的 candidate。

其中：

- \(N_{\mathrm{source}}=1\) 仅当 candidate 有正结构 effect 且 provenance 为全局新来源；
- \(r_{\mathrm{retrieval}}\) 默认使用 `union_pool_rank`，缺失时依次回退 atom/baseline rank；
- \(s_{\mathrm{retrieval}}\) 默认使用 raw `hybrid_score`，不使用 composite oracle/fusion score；
- token cost 只作为末级 tie-break，不作为软权重线性减分。

该顺序实现：

```text
DIRECT address
> PARTIAL coverage of untouched atoms
> polar contrast
> qualification / relation extension
> second-source corroboration
> aligned context
> retrieval fallback
```

### 10.4 FALLBACK key

若所有 \(G_*\) 均为 0，则按以下 key 排序：

```text
lower approved retrieval rank
→ higher raw hybrid score
→ lower token cost
→ lexicographically smaller stable key
```

### 10.5 Primary atom

Primary atom 仅用于 prompt cue 与 trace 展示，不影响 candidate score 或其他 atom updates。

选择顺序为：

1. pair effect 所属结构层级更高；
2. atom importance 更高；
3. pair confidence 更高；
4. atom ID 字典序更小。

---

## 11. Multi-atom simultaneous update

选定 candidate 后：

1. 所有 pair effects 均基于同一个 pre-state 计算；
2. 对 candidate 的所有 valid resolving pairs 同步执行 union/max update；
3. 对 valid context pairs 更新 \(X_i\)，但不更新 relation level；
4. 最后统一派生 stance set、conflict、atom-level address 与 saturation；
5. 更新顺序不得依赖 `candidate_atom_alignments` 数组顺序。

对每个 relation：

\[
\lambda_i^{(t)}(r)
=
\max\left(
\lambda_i^{(t-1)}(r),
d_{ij}
\right),
\]

\[
\Pi_i^{(t)}(r)
=
\Pi_i^{(t-1)}(r)
\cup\{\mathrm{provenance}(u_j)\},
\]

其中 UNKNOWN 不加入可证明独立性的 provenance set。

---

## 12. SCSG full-order algorithm

```text
Algorithm SCSG-FULL-ORDER
Input: atoms A, candidate pool C, canonical pair map M
Output: full deterministic ordering O and replayable state trace

1  validate input and build allowed-field view
2  canonicalize atoms, candidates, provenance, duplicates and pair rows
3  compute best-attainable level b_i* for every atom
4  initialize per-relation atom states H^(0)
5  O ← []
6  while an unselected, non-duplicate candidate remains:
7      evaluate every remaining candidate against the same current state
8      compute all pair effects and candidate key K_t(u)
9      choose lexicographically maximal key; stable key breaks exact ties
10     append candidate to O
11     synchronously update every valid aligned atom
12     record step, atom_updates[], key components and trace state
13     reject remaining hard duplicates of the selected evidence
14 return O and trace
```

### 12.1 Ordering stop

主配置采用 `ordering_mode=fullpool`：

- selector 不因 saturation 提前停止；
- selector token budget 为 `null`；
- `max_order_steps=0` 表示遍历所有非 duplicate candidates；
- 无结构增益后继续输出 deterministic fallback order。

合法 stop reasons：

- `pool_exhausted_after_dedup`；
- `max_order_steps`，仅非主配置；
- `invalid_input`，build gate 应失败而非进入主实验。

---

## 13. BAREC prefix 与三层预算

### 13.1 主配置

候选冻结主配置：

```text
ordering_mode: fullpool
selector_token_budget: null
prompt_policy: minmax
k_min: 5
k_max: 10
target_attainable_saturation: 1.0
contrast_closure: true
prompt_max_length: 1024
```

### 13.2 Contrast-closed target

推荐 `target_satisfied` 同时要求：

1. \(\rho_{\mathrm{attain}}=1.0\)；
2. 当前剩余候选中不存在尚未捕获的 valid polar contrast。

RELATION_EXTEND、CORROBORATE 与 CONTEXT 不作为停止必需条件。

这可避免恰好在第 5 条完成 atom coverage，却把紧随其后的反向证据截掉。

### 13.3 minmax 定义

令 \(p^*\) 为第一个满足：

\[
p\ge k_{\min}
\land
\mathrm{target\_satisfied}(u_p)=1
\]

的位置，则：

\[
K_{\mathrm{sel}}
=
\begin{cases}
p^*,&p^*\le k_{\max},\\
\min(K_{\mathrm{order}},k_{\max}),&\text{otherwise.}
\end{cases}
\]

若 \(K_{\mathrm{order}}<k_{\min}\)，允许 \(K_{\mathrm{sel}}<k_{\min}\)，stop reason 为 `end_of_trace`。

Prefix stop reasons 固定为：

- `target_satisfied`；
- `max_evidence_count`；
- `end_of_trace`。

### 13.4 Context guard

generic prompt builder 在 prefix 之后执行 max-length protection。它只能从尾部删除 evidence 或截短最后一条 evidence text，不得重排前缀。

必须单独记录：

- `prompt_truncation_reason`；
- `final_prompt_evidence_indices`；
- `evidence_text_truncated`；
- \(K_{\mathrm{final}}\)。

论文不得把 \(K_{\mathrm{sel}}\) 当作 verifier 必然可见的数量。

---

## 14. 待作者拍板的六项决策

以下为本版推荐默认值；作者回复“按推荐冻结”即可全部确认。

这些选择来自当前 artifact，而非纯概念偏好：LIAR validation 的 8,368 个 direct/partial resolving pairs 中只有 59 个 confidence 低于 0.5，且该批 pairs 均带非空 key span；另一方面，只有约 57.8% 的 claims 能让所有 atoms 在 pool 中 direct-reachable，direct-or-partial reachable 约为 76.3%。因此固定 0.5 gate 的实际作用很小，而 direct-only 全原子停止会使大量样本永远无法提前停止。

| ID | 决策 | 推荐默认值 | 主要理由 |
|---|---|---|---|
| D1 | Confidence gate | `gamma > 0 + nonempty key_span`；confidence 仅 tie-break | 避免未校准的 0.5 成为隐藏跨域超参；Phase 1 做阈值敏感性 |
| D2 | BRIDGE | v0.1 删除 BRIDGE，改为不改状态的 CONTEXT | 当前没有 bridge endpoints；旧 BRIDGE 主要是噪声 |
| D3 | Qualification | 新增 RELATION_EXTEND/QUALIFY，绝不并入 CONTRAST | 只有 S↔R 是 polar contrast |
| D4 | Partial vs contrast | coverage-first：PARTIAL_OPEN > CONTRAST | 优先触达尚未覆盖的 atom；另做 contrast-first 敏感性 |
| D5 | Stop semantics | best-attainable saturation=1.0，partial 仅在 pool 无 direct 时满足 | 避免 partial 一律 resolved，也避免 direct-only 永不停止 |
| D6 | Contrast closure | saturation 后仍吸收 pending polar contrast，再允许 minmax stop | 防止 coverage 恰好满足时截掉关键反向证据 |

已经直接冻结、不再保留选项：

- pair-level source of truth；
- 一 evidence 同步更新所有 valid atoms；
- per-relation directness/provenance state；
- provenance-aware second-source corroboration；
- stable-key deterministic tie-break；
- oracle/gold/verifier allowed-field isolation；
- full ordering 与 prefix controller 分层；
- canonical trace 使用新 schema version。

---

## 15. Canonical trace contract

### 15.1 Trace-level

```json
{
  "barec_trace_version": "barec_trace_v0_1",
  "scsg_policy_version": "scsg_v0_1",
  "selector_name": "barec_scsg_v0_1",
  "ordering_mode": "fullpool",
  "oracle_usage": "none",
  "oracle_fields_ignored": true,
  "verifier_usage": "none",
  "weight_file": "",
  "candidate_pool_count": 20,
  "selector_ordered_count": 18,
  "selector_ordered_stable_keys": ["..."],
  "ordering_stop_reason": "pool_exhausted_after_dedup",
  "pool_attainable_levels": {"A1": "DIRECT"},
  "barec_steps": []
}
```

Canonical method trace 不复制 `gold_label`。下游训练数据可以按 `event_id` 在 selector 完成后另行 join label。

### 15.2 Step-level

每个 step 至少记录：

```json
{
  "step": 1,
  "step_schema": "scsg_step_v0_1",
  "candidate_stable_key": "...",
  "candidate_uid": "...",
  "selector_candidate_idx": 7,
  "evidence_id": "E08",
  "provenance_id": "report:123",
  "duplicate_id": "G2",
  "token_cost": 81,
  "primary_atom_id": "A1",
  "operation": "OPEN",
  "operation_subtype": "DIRECT_OPEN",
  "aligned_atom_ids": ["A1", "A2"],
  "valid_resolving_atom_ids": ["A1", "A2"],
  "updated_atom_ids": ["A1", "A2"],
  "newly_direct_addressed_atom_ids": ["A1"],
  "atom_updates": [],
  "structural_key": {
    "direct_gain": 1.0,
    "partial_gain": 0.0,
    "contrast_gain": 0.0,
    "relation_extend_gain": 0.0,
    "corroboration_gain": 0.0,
    "context_gain": 0.0,
    "confidence_tiebreak": 0.95,
    "retrieval_rank": 3,
    "retrieval_score": 0.81,
    "token_cost": 81
  },
  "trace_state": {}
}
```

### 15.3 Atom update

```json
{
  "atom_id": "A1",
  "relation": "S",
  "directness": "DIRECT",
  "confidence": 0.95,
  "key_spans": ["..."],
  "valid": true,
  "pair_effects": ["DIRECT_OPEN"],
  "relation_level_before": "NONE",
  "relation_level_after": "DIRECT",
  "stance_set_before": [],
  "stance_set_after": ["S"],
  "atom_level_before": "NONE",
  "atom_level_after": "DIRECT",
  "provenance_added": true,
  "saturated_after": true
}
```

### 15.4 Trace state

```json
{
  "selected_count": 5,
  "weighted_attainable_saturation": 1.0,
  "total_direct_address_coverage": 0.67,
  "total_address_coverage": 1.0,
  "unreachable_atom_ids": [],
  "conflicted_atom_ids": ["A2"],
  "pending_polar_contrast_count": 0,
  "target_satisfied": true,
  "target_resolved": true,
  "atom_states_after": {}
}
```

`target_resolved` 只作为旧 prompt-policy 的 compatibility alias；论文与新代码均使用 `target_satisfied` / `attainable_saturation`。

### 15.5 Verifier-row 三层字段

必须新增或明确：

- `prompt_evidence_ordered_count` = \(K_{\mathrm{order}}\)；
- `prompt_evidence_selected_count_before_prompt_truncation` = \(K_{\mathrm{sel}}\)；
- `evidence_count` = \(K_{\mathrm{final}}\)；
- `prompt_evidence_selected_indices`；
- `final_prompt_evidence_indices`；
- `selector_ordering_stop_reason`；
- `prompt_evidence_stop_reason`；
- `prompt_truncation_reason`。

---

## 16. Compatibility projection

为了复用现有 `mrec_min` rendering，可从 canonical step 投影旧字段：

- `atom_id` = primary atom；
- `state_before/state_after` = primary atom compatibility label；
- `covered_atom_ids` = valid resolving atoms；
- `relation/directness/confidence` = primary pair；
- `operation` = primary operation；
- `compat_chain_steps` 保持一个 evidence 一个 prompt block。

Canonical state 与 `atom_updates[]` 不进入 verifier prompt。Verifier 仍只看到：

```text
Check: {primary atom proposition}
{original evidence text}
```

---

## 17. Config candidate freeze

```yaml
trace:
  selector_name: barec_scsg_v0_1
  selection_policy: structural_state_greedy
  trace_schema: barec_trace_v0_1
  ordering_mode: fullpool
  candidate_top_n: 0
  max_order_steps: 0
  selector_token_budget: null
  confidence_gate: positive_with_key_span
  relation_extend_enabled: true
  context_enabled: true
  corroboration_max_sources_for_gain: 2
  contrast_closure: true
  weight_file: ""
  oracle_usage: none
  verifier_usage: none

prompt_evidence:
  policy: minmax
  min_evidence_count: 5
  max_evidence_count: 10
  target_metric: attainable_saturation
  target_value: 1.0
  evidence_token_budget: null
  trace_prompt_style: mrec_min
  evidence_text_mode: full

build:
  prompt:
    max_length: 1024
```

---

## 18. Phase 0 验收测试

### 18.1 Schema 与 validity

1. 缺失 atoms、重复 atom ID、非法 pair enum、重复 pair 触发 build gate。
2. `irrelevant/none`、`insufficient/none`、context resolving relation 均不更新 relation state。
3. valid alignment 必须有正 confidence 与 key span。
4. 修改 candidate-level `map_relation/map_directness/covered_atom_ids` 不改变 SCSG 输出。
5. `mixed` 稳定映射到 Q，不同时制造 S/R conflict。

### 18.2 State 与 multi-atom

6. 一条 evidence 同步更新两个或更多 valid atoms。
7. Pair 数组重排不改变 candidate key、updates 或 final state。
8. `direct-S + partial-R` 与 `partial-S + direct-R` 产生不同 per-relation state。
9. S→R 产生 conflict；随后 Q 不覆盖或清除 conflict。
10. partial 后 direct 同 relation 产生 DIRECT_UPGRADE。
11. atom 只有 partial reachable 时，partial 可完成 saturation；pool 有 direct 时不可。

### 18.3 Provenance、duplicate 与 corroboration

12. 同 relation、同 report 不产生 corroboration。
13. 同 relation、第二个独立 report 产生一次 corroboration。
14. 第三个独立 report 不再产生新的 corroboration gain。
15. UNKNOWN provenance 不产生 corroboration/source novelty。
16. duplicate group、stable identity、全文 hash 或同 span duplicate 被硬拒绝。

### 18.4 Determinism 与 oracle isolation

17. 候选输入顺序随机化后，stable-key ordering 完全一致。
18. 同输入重复运行两次，canonical method trace bit-identical。
19. oracle fields absent/empty/random/adversarial 时，ordering、steps、prefix 与 prompt 完全相同。
20. 修改 gold label、verifier score、weight file path 等禁止字段不改变任何方法输出。
21. direct config 没有 weight training、oracle source 或 verifier-derived score。

### 18.5 Ordering、prefix 与 prompt

22. 所有 nonduplicate candidates 都进入 full ordering；零增益 candidate 进入 fallback tail。
23. 始终满足 \(K_{pool}\ge K_{order}\ge K_{sel}\ge K_{final}\)。
24. target 在 min 前、min、min/max 之间、max 的边界行为符合定义。
25. 无 target 时分别覆盖 `max_evidence_count` 与 `end_of_trace`。
26. saturation 已满足但仍有 pending polar contrast 时不停止。
27. selected indices 始终是 full ordering 的前缀。
28. max-length guard 只删尾部，不改变剩余顺序。
29. 最终 prompt 只包含 `final_prompt_evidence_indices` 对应 evidence。
30. state/operation/structural key 不泄漏进 verifier-visible prompt。

### 18.6 Replay 与报告

31. 从初始 state 和 `atom_updates[]` 可完整回放 final state。
32. replay state 与每步 `trace_state`、最终 state 完全一致。
33. report 汇总 pool/order/sel/final 四层数量、三层 stop reasons、operation/effect 分布与 truncation rate。
34. main build 中 oracle usage、invalid pair、out-of-range index、duplicate selected 与 post-build overflow 均为 0。

---

## 19. Phase 0 完成门槛

作者确认 D1–D6 后，Phase 0 按以下顺序结束：

1. 将本文档状态改为 `Frozen v0.1`；
2. 新建独立 `structural_state_greedy` policy，不修改旧 learned baseline 行为；
3. 新建 `barec_trace_v0_1` schema 与 compatibility projector；
4. 实现第 18 节 P0 tests；
5. 32-row LIAR smoke 生成 full ordering、fixed-5 与 minmax(5,10)；
6. 人工构造 prompt overflow 验证 \(K_{\mathrm{final}}\)；
7. 通过 oracle poison、determinism、state replay 与三层 K gate；
8. 冻结 config/hash 后再进入 Phase 1 validation diagnostic。

Phase 0 不以 verifier F1 为完成条件；完成条件是算法定义、输入隔离、状态回放、确定性和 artifact 契约全部闭合。
