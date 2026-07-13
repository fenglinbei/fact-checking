# BACES Phase 0 有序覆盖问题与算法规范 v0.3

状态：**Phase 0 Frozen（问题与算法契约冻结；实现迁移待完成）**

日期：2026-07-12

取代：atom_coverage_selection_phase0_spec_v0.2.md

适用范围：论文问题定义、方法命名、ordered selector、trace schema、budget/materialization、prompt interface、理论与实验主张。

历史 transition、learned_marginal_proxy、MREC trace 与既有实验只作为 baseline 保留，不自动改名为 BACES v0.3。

---

## 0. 一页冻结结论

### 0.1 固定名称

- 问题：**Budgeted Atom-Coverage Evidence Sequencing（BACES）**。
- 中文：**预算约束的原子覆盖证据排序**。
- 输出对象：**ordered evidence slate**，不是 evidence chain。
- 主 policy：baces_lexicographic_early_coverage_v0_3。
- 主 solver：exact ordered-state dynamic programming。
- 可扩展 baseline：prefix-conditioned marginal coverage greedy。
- 主 prompt policy：selected_set，直接消费 solver 冻结的 slate；不再由旧 minmax 二次选择。

### 0.2 冻结的一句话定义

> Given claim atoms, a flat evidence pool, and typed atom–evidence alignments, BACES constructs a budget-feasible ordered evidence slate that first maximizes terminal graded atom coverage and, among terminal-optimal slates, minimizes when each unit of coverage quality is first acquired.

中文：

> 给定声明原子、平铺候选证据池与 typed atom–evidence 对齐，BACES 构造满足预算的有序证据 slate：首先最大化最终的分级原子覆盖；在最终覆盖相同的序列中，使 partial/direct 覆盖质量尽可能早地进入前缀。

### 0.3 有序性边界

BACES 保留：

- prefix 已覆盖 atoms 对下一候选边际收益的影响；
- 同一终态覆盖下的 early-coverage order；
- solver 输出顺序对 verifier-visible prefix 的影响；
- count、token、rendering floor 与最终上下文截断。

BACES 不建模：

- evidence-to-evidence logical dependency；
- multi-hop composition；
- bridge/prerequisite relation；
- support/refute conflict resolution；
- reasoning path 或 faithful proof chain。

关键区别：

> 前一步改变后一步的覆盖边际价值，不等于后一步在逻辑上依赖前一步。

### 0.4 冻结系统边界

~~~text
claim atoms
  -> flat Atom-Union candidate pool
  -> pair-level typed atom-evidence map
  -> canonical deduplication
  -> exact BACES coverage-core sequencing
  -> budget-feasible ZERO_GAIN_FILL
  -> frozen selected slate
  -> BACES rendering
  -> tokenizer-aware auto truncation
  -> verifier
~~~

---

## 1. 从 v0.2 到 v0.3 的纠偏

v0.2 正确删除了 evidence-chain、多跳、冲突和证据依赖语义，但错误地把输出收缩为无序 set，再把顺序降为纯 display choice。

当前方法实际存在两种非逻辑的顺序作用：

1. selector 每一步根据当前 prefix 重算候选边际价值；
2. verifier 只接收 ordered slate 或其前缀，位置会影响可见集合与呈现。

因此 v0.3 不恢复 chain，而把问题从 Evidence Selection 改为 Evidence Sequencing。

若只优化：

\[
\max_S F(S),
\]

则任意 permutation 都有相同目标值，不能把 sequencing 作为贡献。v0.3 必须让聚合的 prefix coverage quality（等价 padded prefix AUC）正式进入目标。

### 1.1 为什么不再把 coverage 做成二值

二值 valid gate 仍然保留，用来排除 irrelevant/context/none 或无 key span 的 pair；但 pair quality 不再压成单一 covered bit。若令 direct 与 partial 都等于 1，则：

- partial 首次出现后，later direct 被错误判为 zero gain；
- selector 无法区分“弱触达 atom”和“直接支撑 atom”；
- 论文声称的 prefix-conditioned quality upgrade 在目标中不存在；
- 当前 \(m\le6\) 时本已很短的 coverage core 会进一步退化。

例如：

\[
\mathbf q_{\mathrm P}=(1,0),\qquad
\mathbf q_{\mathrm D}=(2,0),\qquad
\mathbf q_{\mathrm A2}=(0,2).
\]

二值折叠会把 \(\mathbf q_{\mathrm P}\) 与 \(\mathbf q_{\mathrm D}\) 视为等价；ordinal objective 则在相同预算下明确偏好 direct evidence，并允许 \(1\to2\) 成为可审计的正增益 transition。

不采用连续 confidence-weighted score，是因为它会同时引入置信度校准、任意缩放与浮点 tie 问题。三态 \(0/1/2\) 是当前冻结目标的最小非二值状态；在 \(m\le6\) 时最多只有 \(3^6=729\) 个 ordinal states，仍可精确求解。

---

## 2. 当前实现审计与规范迁移

### 2.1 历史 learned 路径

当前实现的实际数据流是：

~~~text
candidate_pool
  -> learned prefix-dependent full order
  -> stored per-step target_resolved
  -> old minmax/resolve_stop reads stored state
  -> K_sel
  -> mrec_min rendering
  -> auto_truncate_evidence tail removal
  -> K_final
~~~

已确认：

- selector_ordered_indices 与 selected_indices 在历史 fullpool trace 中保存同一完整顺序；
- candidate_top_n=0、max_steps=0、token_budget=null、极低 stop threshold 时，learned selector 会继续排序非重复候选；
- target_resolved 只写入 step trace，不会停止该 fullpool selector；
- 旧 minmax 不根据新 prefix 回放 map/state，只读取旧 step 中预存的 target_resolved；
- resolve_stop 与 minmax 的代码逻辑相同，差别主要是配置的 min/max；
- 旧 minmax 不执行 evidence_token_budget；只有 budget policy 检查该预算；
- 真正删除 prompt evidence 的是 build_training_row 后的 auto_truncate_evidence；max-length guard 主要做事后 warn/error；
- build row 的 candidates 仍可保留 \(K_{\mathrm{sel}}\) 条，即使实际 prompt 已降为 \(K_{\mathrm{final}}\) 条。

### 2.2 当前 artifact 事实

LIAR-RAW validation 当前有 1,274 个样本，atom-count 分布为：

| atom count | claims | proportion |
|---:|---:|---:|
| 1 | 809 | 63.50% |
| 2 | 317 | 24.88% |
| 3 | 96 | 7.54% |
| 4 | 48 | 3.77% |
| 5 | 3 | 0.24% |
| 6 | 1 | 0.08% |

因此即使采用 ordinal coverage，单 atom 样本的 coverage-core 仍很短；多 atom 样本只有 465/1,274。v0.3 必须按 atom count 报告 early-order 效果，不能用总体平均掩盖退化。

当前 learned minmax(5,10) validation build 中：

- 177/1,274 个样本发生 \(K_{\mathrm{sel}}>K_{\mathrm{final}}\)；
- 302 个样本在 prompt policy 层选择了 10 条；
- 存在 \(K_{\mathrm{sel}}=10,K_{\mathrm{final}}=6\) 但 candidates 仍长 10 的样本。

这些事实要求 v0.3 显式记录 final indices/keys，不能用 candidates 长度替代 verifier-visible count。

### 2.3 v0.3 主路径

~~~text
canonical pool
  -> exact positive-gain coverage core
  -> ZERO_GAIN_FILL under remaining budget
  -> freeze S_sel and K_sel
  -> preserve solver order
  -> selected_set prompt policy
  -> auto truncation
  -> explicit S_final and K_final
~~~

历史 learned baseline 继续走旧 trace/minmax 路径；BACES 主方法不得复用陈旧的 target_resolved 二次选择。

---

## 3. 输入实例

### 3.1 Claim atoms

\[
\mathcal A(c)=\{a_1,\ldots,a_m\},
\qquad
1\le m\le6.
\]

每个 atom 的权重满足：

\[
w_i>0.
\]

主配置固定：

\[
w_i=1.
\]

LLM importance-weighted variant只作 sensitivity，不进入主定义。若使用小数权重，solver 前按冻结精度量化为正整数，避免浮点 tie 破坏 exactness。

### 3.2 Candidate pool

\[
\mathcal E=\{e_1,\ldots,e_n\}.
\]

每条 evidence 至少包含：

- stable evidence identity；
- source/span identity；
- evidence text 与可选 anchor text；
- additive token cost \(c_j\in\mathbb Z_{\ge0}\)；
- retrieval metadata；
- pair-level typed alignments。

候选池原始顺序不是 BACES 解。

### 3.3 Pair-level typed map

对每个 \((a_i,e_j)\)：

\[
M_{ij}=(r_{ij},d_{ij},\gamma_{ij},s_{ij}),
\]

其中 \(r\) 是 relation，\(d\) 是 directness，\(\gamma\) 是 confidence，\(s\) 是 key span。

Pair-level map 是 coverage 的唯一来源。Candidate-level collapsed summary 仅作历史兼容。

---

## 4. Ordinal structural-coverage contract

### 4.1 Valid predicate

\[
V_{ij}
=
\mathbf 1
\left[
r_{ij}\in\mathcal R_{\mathrm{cov}}
\land
d_{ij}\in\{\texttt{direct},\texttt{partial}\}
\land
\gamma_{ij}>0
\land
s_{ij}\ne\varnothing
\right],
\]

其中：

\[
\mathcal R_{\mathrm{cov}}
=
\{\texttt{support},\texttt{refute},\texttt{qualify},\texttt{mixed}\}.
\]

若上游 schema 使用 \(\texttt{supports/refutes/qualifies/partially\_supports}\) 等别名，必须在 schema adapter 中一次性映射到上述 canonical vocabulary；solver 内不得散落数据集或模型特化分支。

在 valid gate 之上定义固定的 ordinal pair quality：

\[
q_{ij}
=
\begin{cases}
2,&V_{ij}=1\land d_{ij}=\texttt{direct},\\
1,&V_{ij}=1\land d_{ij}=\texttt{partial},\\
0,&\text{otherwise}.
\end{cases}
\]

若同一 \((a_i,e_j)\) 有多条 canonical pair rows，则对每条 row \(h\) 按同一规则得到 \(q_{ijh}\)，再固定：

\[
q_{ij}
=
\max_{h\in\operatorname{Rows}(i,j)}q_{ijh}.
\]

没有 row 时定义 \(q_{ij}=0\)。原始 rows 继续留在 trace 供冲突审计，但不生成额外 stance 或 multiplicity gain。

因此 \(q_{ij}\in\{0,1,2\}\) 分别表示：

- 0：uncovered / invalid；
- 1：partial coverage；
- 2：direct coverage。

Confidence 在主定义中只作 valid gate，不连续乘入 \(q_{ij}\)，也不参与 dedup、coverage core、ZERO_GAIN_FILL 或 canonical tie-break。这样避免未经校准的 LLM confidence 变成隐蔽权重，并保持有限状态 exactness。若未来要让正 confidence 大小影响选择，必须作为独立 calibrated variant，而不是静默改变 v0.3。

每条 evidence 的结构向量为：

\[
\mathbf q_j=(q_{1j},\ldots,q_{mj})\in\{0,1,2\}^m.
\]

一条 evidence 必须同步更新全部 valid aligned atoms，不再只更新 primary atom。

### 4.2 Relation 的有限作用

Relation 只参与 valid gate，不产生 stance state；directness 决定 coverage level。

同一 atom 同时出现 support 与 refute 时：

- 每个 prefix 对该 atom 只保留最高 directness level；
- 不生成 conflict state；
- selector 不做真假裁决；
- verifier 负责最终 label prediction。

### 4.3 Prefix state update

定义 prefix ordinal state：

\[
\mathbf x^{(0)}=\mathbf 0,
\]

\[
x_i^{(t)}
=
\max\left(x_i^{(t-1)},q_{i,\pi_t}\right).
\]

即 state 使用 componentwise max，而不是加和：

- partial evidence 可把 0 升到 1；
- later direct evidence 可把 1 升到 2；
- direct 后的 partial 无新增结构收益；
- 两条 partial 不会被自动解释为 full/direct；
- 同一 evidence 重复选择不会二次增加 state。

这保留了 non-binary structural proxy，同时不引入 evidence dependency 或未经验证的“partial pieces 可组合为事实充分”假设。

### 4.4 Terminal ordinal coverage

定义未归一化结构覆盖：

\[
U(\mathbf x)
=
\sum_{i=1}^{m}w_i x_i.
\]

归一化 coverage：

\[
F(\mathbf x)
=
\frac{U(\mathbf x)}
{2\sum_{i=1}^{m}w_i}.
\]

\(F\in[0,1]\)。Partial atom贡献其权重的一半，direct atom贡献完整权重。

为区分 selector 失败与 candidate-pool 不可达，另定义 pool-reachable upper bound：

\[
U_{\mathrm{pool}}
=
\sum_{i=1}^{m}w_i\max_{e_j\in\mathcal E}q_{ij}.
\]

当 \(U_{\mathrm{pool}}>0\) 时，reachable-normalized coverage 为：

\[
F_{\mathrm{reach}}(\mathbf x)
=
\frac{U(\mathbf x)}{U_{\mathrm{pool}}}.
\]

当 \(U_{\mathrm{pool}}=0\) 时，\(F_{\mathrm{reach}}\) 记为 NA，而不是伪造为 0 或 1。主 objective 仍优化未归一化 \(U\)；\(F\) 与 \(F_{\mathrm{reach}}\) 只用于跨样本报告。

### 4.5 明确不做 additive multi-cover

以下不属于 v0.3 主定义：

- 两条 partial 自动累积为 direct；
- same-level corroboration gain；
- provenance diversity reward；
- relation balance；
- continuous confidence-weighted coverage；
- verifier-derived utility。

若未来希望多条 same-level evidence 继续获得正收益，必须显式定义 multi-cover/provenance semantics，并重证 state sufficiency 与 exact solver；不能暗中塞入 tie-break。

---

## 5. Ordered evidence slate

### 5.1 Coverage-core sequence

BACES 输出正覆盖增益 core：

\[
\pi
=
(e_{\pi_1},\ldots,e_{\pi_L}),
\qquad
L\le \bar K_{\max}.
\]

Prefix ordinal state 已由第 4.3 节定义：

\[
\mathbf x^{(t)}
=
\max\left(\mathbf x^{(t-1)},\mathbf q_{\pi_t}\right),
\]

其中 max 按分量执行。每一步结构增益：

\[
\Delta_t
=
U(\mathbf x^{(t)})
-
U(\mathbf x^{(t-1)}).
\]

每一步必须满足：

\[
\Delta_t>0.
\]

因此 coverage core 中不包含 duplicate、zero-gain 或 purely contextual evidence。

### 5.2 Weighted coverage-unit acquisition time

对 atom \(a_i\) 的 ordinal unit \(\ell\in\{1,2\}\)，定义：

\[
\tau_{i,\ell}(\pi)
=
\min\{t:x_i^{(t)}\ge\ell\},
\]

仅对 terminal state 中实际取得的 units 计时。定义：

\[
T(\pi)
=
\sum_{i=1}^{m}
\sum_{\ell=1}^{x_i^{(L)}}
w_i\tau_{i,\ell}(\pi).
\]

等价递推形式：

\[
T(\pi)
=
\sum_{t=1}^{L}t\Delta_t.
\]

\(T(\pi)\) 越小，partial/direct coverage units 越早出现在 sequence prefix。

### 5.3 冻结的 lexicographic objective

BACES 按以下 tuple 做严格词典序最小化：

\[
\min_{\mathrm{lex}}
\left(
-U(\mathbf x^{(L)}),
T(\pi),
L,
\sum_{t=1}^{L}c_{\pi_t},
\operatorname{keys}(\pi)
\right).
\]

语义依次为：

1. 最大化 terminal graded atom coverage；
2. terminal coverage 相同时最小化 coverage-unit acquisition time；
3. terminal \(U\) 与 acquisition time \(T\) 都相同时使用更短 core；
4. 再最小化 token cost；
5. 最后选择字典序更小的 stable-key sequence 确定性决胜。

实现必须比较 tuple，不能使用任意 big-\(M\) scalarization。

### 5.4 与 padded prefix AUC 的等价性

取固定 horizon \(H=K_{\max}\)。若 \(t>L\)，令：

\[
\mathbf x^{(t)}=\mathbf x^{(L)}.
\]

定义：

\[
\operatorname{AUC}_H(\pi)
=
\sum_{t=1}^{H}U(\mathbf x^{(\min(t,L))}).
\]

有：

\[
\operatorname{AUC}_H(\pi)
=
(H+1)U(\mathbf x^{(L)})-T(\pi).
\]

所以在 terminal coverage 相同时：

\[
\min T(\pi)
\iff
\max \operatorname{AUC}_H(\pi).
\]

该定义无需额外 discount hyperparameter。

不使用未 padding 的 \(\sum_{t=1}^{L}U(\mathbf x^{(t)})\)，因为它可能奖励故意拉长 sequence。

---

## 6. Budget contract

Coverage core 约束：

\[
0\le K_{\min}\le K_{\max}.
\]

令 \(n=K_{\mathrm{pool}}\) 为 canonical dedup 后候选数，solver 的有效 count cap 为：

\[
\bar K_{\max}=\min(K_{\max},n).
\]

允许配置的 \(K_{\max}>n\)；此时只会因 pool 不足而 underfill，不伪造候选。

\[
L\le \bar K_{\max},
\]

\[
\sum_{t=1}^{L}c_{\pi_t}\le B
\quad
\text{（若启用 token budget）}.
\]

冻结解释：

- \(K_{\max}\)：selected-slate count upper bound，也是 coverage-core upper bound；
- \(B\)：additive evidence token upper bound；
- \(K_{\min}\)：soft rendering floor，不是 primary optimization constraint；
- prompt model max length：solver 后的独立约束。

主论文不得把 \(K_{\min}\) 伪装为新的 maximum-coverage constraint。

---

## 7. Exact ordered-state DP

### 7.1 State

每条 candidate 已编码为 ordinal vector：

\[
\mathbf q_j\in\{0,1,2\}^{m}.
\]

启用整数 token budget 时：

\[
D[k,\mathbf x,b]
\]

表示达到 ordinal state \(\mathbf x\)、恰用 \(k\) 条 core evidence、总 token cost 恰为 \(b\) 时的最小 weighted coverage-unit acquisition time。

初始化：

\[
D[0,\mathbf 0,0]=0.
\]

其他状态为 \(+\infty\)。

### 7.2 Transition

对 candidate \(e_j\)：

\[
\mathbf x'
=
\max(\mathbf x,\mathbf q_j),
\]

\[
\Delta_j(\mathbf x)
=
U(\mathbf x')-U(\mathbf x).
\]

只允许：

\[
\Delta_j(\mathbf x)>0.
\]

递推：

\[
D[k+1,\mathbf x',b+c_j]
=
\min
\left\{
D[k+1,\mathbf x',b+c_j],
D[k,\mathbf x,b]+(k+1)\Delta_j(\mathbf x)
\right\}.
\]

因为新增 partial/direct quality units 首次出现在位置 \(k+1\)，其 acquisition-time 增量正是 \((k+1)\Delta_j(\mathbf x)\)。

在 exact-budget cell \(D[k,\mathbf x,b]\) 内，\(b\) 已固定；因此按 \((T,\operatorname{keys}(\pi))\) 保留 acquisition time 更小、再 stable-key sequence 更小的 canonical backpointer。只有无 \(B\) 的 \(D[k,\mathbf x]\) 才在同一 cell 内继续比较 total token cost。

实现时按 \(k=0,\ldots,r-1\) 展开 layer；对每个 reachable \((\mathbf x,b)\) 枚举全部 canonical candidates。不得写成 candidate-outer 的一次性 0/1 knapsack 更新，否则会把 sequence order 错误限制为 candidate array order。

### 7.3 无 token budget

未启用 \(B\) 时，使用：

\[
D[k,\mathbf x].
\]

状态内按：

\[
(T,\text{token cost},\text{stable sequence})
\]

保存最优 backpointer。

### 7.4 Pareto alternative

当 \(B\) 很大时，可为每个 \((k,\mathbf x)\) 保存 nondominated \((b,T)\) frontier：

\[
(b_1,T_1)\prec(b_2,T_2)
\iff
b_1\le b_2
\land
T_1\le T_2.
\]

至少一项严格更优时删除 dominated state。

当 \((b,T)\) 完全相同时保留 stable-key sequence 更小的 frontier entry。

### 7.5 Terminal selection

在所有：

\[
k\le \bar K_{\max},
\qquad
b\le B
\]

的可行状态中，按第 5.3 节的完整 lex tuple 选择终态并回溯 sequence。

### 7.6 不需要 used-candidate state

若 \(e_j\) 已在较早位置选过，则 componentwise：

\[
\mathbf q_j\le \mathbf x^{(t-1)},
\]

再次选择必有：

\[
\Delta_j(\mathbf x^{(t-1)})=0,
\]

不满足正增益 transition。

删除重复项不会降低 terminal coverage，只会降低 count/cost 并使后续 acquisition time不增。因此 coverage core 存在无重复最优解，ordinal vector \(\mathbf x\) 是未来正增益候选的充分状态。

### 7.7 Complexity

令：

\[
r=\min(\bar K_{\max},2m).
\]

每个 core step 至少新增一个 ordinal coverage unit，而总单位数最多为 \(2m\)，所以：

\[
L\le r.
\]

启用显式整数 \(B\)：

\[
O(nr3^mB)
\]

次 arithmetic state relaxations，滚动空间：

\[
O(3^mB),
\]

回溯另计。若朴素复制并逐 key 比较最长为 \(r\) 的 backpointer sequence，则含 canonical tie-break 的保守时间界为：

\[
O(nr^2 3^mB).
\]

若实现额外维护可 \(O(1)\) 比较的 canonical prefix rank，则恢复前述 transition bound。该复杂度对 \(B\) 是 pseudo-polynomial，不得称 strongly polynomial。

无 token budget：

\[
O(nr3^m)
\]

次 arithmetic relaxations；朴素 sequence tie comparison 下保守为 \(O(nr^2 3^m)\)。

### 7.8 Exactness claim

在 canonical pool、ordinal vectors、positive quantized weights、additive integer costs、count/token upper bounds下，DP：

- 全局最大化 map-defined terminal coverage；
- 在所有 terminal-optimal sequences 中全局最小化 weighted coverage-unit acquisition time；
- 等价地最大化 fixed-horizon padded prefix AUC；
- 返回 deterministic canonical sequence。

该 guarantee 不代表 human factual sufficiency 或 verifier correctness。

---

## 8. Coverage core、rendering floor 与 selected slate

### 8.1 Core

\[
K_{\mathrm{core}}=L^\star.
\]

Core 每一步均为 solver_role=CORE；兼容字段 operation=COVER。

### 8.2 Soft rendering floor

若：

\[
K_{\mathrm{core}}<K_{\min},
\]

令 terminal core state 为 \(\mathbf x^\star\)，只保留满足 \(\mathbf q_j\le\mathbf x^\star\) 的 canonical unselected candidates；它们在 core 后的 marginal coverage 必为 0。定义：

\[
D_j=\sum_iw_i\mathbf 1[q_{ij}=2],
\qquad
P_j=\sum_iw_i\mathbf 1[q_{ij}=1].
\]

冻结的 ascending fill key 为：

\[
\left(
c_j,
-D_j,
-P_j,
-r_j^{\mathrm{retr}},
\operatorname{stablekey}(e_j)
\right),
\]

其中缺失或非有限 retrieval score 统一视为 \(-\infty\)。cost 放在第一位，使 soft floor 在剩余 token budget 下尽可能补足条数；结构质量和 retrieval 只在相同 cost 内决胜。

将按该 key 排序后的 candidates 记为 \(j_1,j_2,\ldots\)，剩余 token budget 为：

\[
B_{\mathrm{rem}}
=
B-\sum_{e_j\in\pi^\star}c_j
\]

（无 token budget 时视为 \(+\infty\)）。追加数量固定为：

\[
f^\star
=
\max\left\{
f:
\begin{array}{l}
0\le f\le \min(N_0,K_{\min}-K_{\mathrm{core}}),\\
K_{\mathrm{core}}+f\le K_{\max},\\
\sum_{\ell=1}^{f}c_{j_\ell}\le B_{\mathrm{rem}}
\end{array}
\right\}.
\]

其中 \(N_0\) 是 eligible zero-gain candidates 数。Fill 即 \((e_{j_1},\ldots,e_{j_{f^\star}})\)。由于 cost 是第一排序项且非负，该规则在 eligible zero-gain pool 内取得最大可行 fill count，并保持确定性。

Fill 每一步均为 solver_role=FILL；兼容字段 operation=ZERO_GAIN_FILL。

Fill 不进入 terminal coverage、coverage-unit acquisition time 或 solver contribution。

若 pool 或 token budget 无法补足 \(K_{\min}\)，允许 underfill，并记录：

min_count_unreachable=true。

\(K_{\min}\) 是 soft floor。若未来改成硬约束，必须把 fill feasibility 纳入 DP，另起版本。

### 8.3 Selected slate

\[
S_{\mathrm{sel}}
=
\pi^\star
\oplus
\operatorname{Fill}.
\]

\[
K_{\mathrm{sel}}
=
|S_{\mathrm{sel}}|.
\]

Solver core order必须原样保留；fill 只能追加，不能插入 core 中间。

### 8.4 不再把旧 minmax 当作 BACES 主 controller

Exact sequence 的最后一个 core step就是首次达到 terminal-optimal coverage 的位置。

因此主 BACES 不需要：

- full candidate ordering；
- stored target_resolved；
- 沿 full order 二次截 prefix。

若只取达到 \(\rho<1\) 的更早 prefix，它牺牲 primary terminal objective，应作为独立 low-budget rendering policy，而非 BACES 主解。

历史 minmax(5,10) 仍可作为 learned-selector baseline 名称。

---

## 9. Ordinal coverage 的强制风险声明

由于：

\[
m\le6,
\qquad
K_{\mathrm{core}}\le2m\le12,
\]

Ordinal v0.3 比 binary 保留更多结构信息，但 same-level repeated evidence 仍为 zero gain。

典型退化：

- \(m=1\)：至多发生 0→1 与 1→2 两次正升级，core 最多 2；
- \(m=2\)：core 最多 4；
- \(m=3\)：core 最多 6；
- 第 7–10 条只有在较多 atoms 与多次 partial→direct upgrade 下才可能具有正收益；
- direct 后的 partial、第二条同级 partial、第二条同级 direct 都是 zero gain；
- same-atom corroboration 不属于当前 objective。

因此 Phase 1 强制报告：

- atom-count 分层；
- \(K_{\mathrm{core}}\) 分布；
- ZERO_GAIN_FILL count/rate；
- 0→1、0→2、1→2 transition count；
- partial/direct terminal-state distribution；
- multi-atom subset 结果；
- 第 7–10 条 evidence 对现有 learned baseline 的真实贡献。

如果现有收益依赖 same-level corroboration、provenance accumulation 或更多 same-atom evidence，ordinal-max BACES 不足以解释，必须显式升级为 multi-cover/provenance-aware objective。

不能一边声称 componentwise-max ordinal coverage，一边暗中给相同 level 的重复 evidence 正收益。

---

## 10. Display order 与 order controls

### 10.1 主 order

主 display order：

\[
\pi^\star
\oplus
\operatorname{Fill}.
\]

它同时是 selected slate order，不再额外按 retrieval 重排。

### 10.2 Same-set order control

至少比较：

- BACES early-coverage order；
- retrieval order within the same frozen \(S_{\mathrm{sel}}\)；
- candidate-pool order within the same frozen set；
- event-specific fixed-seed random order within the same frozen set。

每种 order 必须：

1. 先冻结完全相同的 selected stable-key set；
2. 只重排这 \(K_{\mathrm{sel}}\) 条；
3. 根据新顺序重新计算 cumulative ordinal state 与 coverage-unit acquisition time；
4. 使用 seed+event_id 生成 permutation；
5. 固定 count/token/prompt rendering。

同时必须区分两类 step 属性：

- solver_role：该 candidate 在原始解中是 CORE 还是 FILL，冻结后不随重排改变；
- display_operation：该 candidate 在当前展示前缀中是 ORDINAL_UPGRADE 还是 DISPLAY_ZERO_GAIN，必须随新顺序重算。

一个原本的 FILL 被移到 core 前面时可能产生正 display marginal；一个原本的 CORE 被移到后面时也可能变成 display zero-gain。不得据此篡改 solver_role。相应地分别记录：

- core acquisition time/AUC：原始 solver core objective；
- display acquisition time/AUC：当前 \(K_{\mathrm{sel}}\) 顺序；
- final acquisition time/AUC：实际 verifier-visible \(K_{\mathrm{final}}\) 顺序。

旧 shuffle_existing 只保持 full trace 全集，随后 top-k 会改变 visible set，不能作为严格 order-only 证据。

### 10.3 Order claim gate

若 same-set order 对 external prefix coverage、human reading effort 或 downstream verifier 没有稳定影响，论文只能主张 prefix-conditioned solver trace，不得把 evidence ordering 写成核心实证贡献。

---

## 11. Prompt interface 与 truncation

### 11.1 主 builder policy

新增：

selected_set 或 trace_all。

其语义：

- 读取 display_ordered_indices；
- 消费全部 \(K_{\mathrm{sel}}\) 条；
- 不再做 target stop；
- 不再用 CLI top-k 二次截断；
- 只允许 tokenizer-aware context truncation。

### 11.2 三层数量

- \(K_{\mathrm{core}}\)：正覆盖增益 core；
- \(K_{\mathrm{sel}}\)：core + fill；
- \(K_{\mathrm{final}}\)：实际进入 verifier 的 evidence 数。

不变量：

\[
K_{\mathrm{core}}
\le
K_{\mathrm{sel}}
\le
\min(K_{\max},K_{\mathrm{pool}}),
\]

\[
K_{\mathrm{final}}\le K_{\mathrm{sel}}.
\]

### 11.3 Final visibility

Build row 必须新增：

- prompt_evidence_selected_indices/keys；
- final_prompt_evidence_indices/keys；
- removed_by_prompt_truncation_indices/keys；
- \(K_{\mathrm{core}},K_{\mathrm{sel}},K_{\mathrm{final}}\)；
- selected_coverage_value；
- final_coverage_value；
- final_coverage_visibility_basis（span_visible 或 identity_only）；
- selected_display_acquisition_time/padded_auc；
- final_prompt_acquisition_time/padded_auc；
- coverage_lost_by_truncation；
- text-only truncation candidate key 与最终可见文本信息。

不能再根据 candidates 长度推断 \(K_{\mathrm{final}}\)。

若 truncation 发生在单条 evidence 文本内部，只有在至少一个用于 \(V_{ij}\) 的 key span 仍实际可见时，该 pair 才能计入 span_visible final coverage；若当前实现只能按 evidence identity 回放，必须标记 identity_only，且不得把该值写成严格 verifier-visible coverage。

### 11.4 Guarantee boundary

若 auto truncation 删除或重排 core evidence：

- final slate 不再保持 DP terminal optimum；
- final coverage-unit acquisition time需重新计算；
- coverage loss归于 prompt realization layer，不归于 selector。

---

## 12. Canonical trace schema

### 12.1 Trace-level fields

~~~json
{
  "schema_version": "baces_sequence_trace_v0_3",
  "solver_version": "baces_exact_ordered_state_dp_v0_3",
  "map_schema_version": "...",
  "candidate_pool_projection_schema": "baces_solver_projection_v0_3",
  "dedup_policy_version": "baces_dedup_v0_3",
  "cost_tokenizer_id": "...",
  "cost_tokenizer_revision": "...",
  "selection_policy": "baces_lexicographic_early_coverage_v0_3",
  "event_id": "...",
  "claim_atoms": ["A1", "A2", "A3"],
  "atom_weights": {"A1": 1, "A2": 1, "A3": 1},
  "ordinal_levels": {"invalid": 0, "partial": 1, "direct": 2},
  "partial_utility_lambda": 0.5,
  "k_min": 5,
  "k_max": 5,
  "token_budget": 600,
  "candidate_pool": ["<18 canonical candidate projections>"],
  "candidate_pool_fingerprint": "...",
  "k_pool_raw": 20,
  "k_pool_dedup": 18,
  "k_core": 3,
  "k_sel": 5,
  "coverage_core_indices": [],
  "coverage_core_keys": [],
  "selected_indices": [],
  "selected_keys": [],
  "display_ordered_indices": [],
  "display_ordered_keys": [],
  "selected_set_fingerprint": "...",
  "display_order_policy": "baces_early_coverage",
  "pool_reachable_ordinal_units": 6,
  "terminal_ordinal_state": {"A1": 2, "A2": 2, "A3": 1},
  "terminal_ordinal_coverage_units": 5,
  "terminal_normalized_coverage": 0.833333,
  "terminal_reachable_normalized_coverage": 0.833333,
  "core_token_cost": 273,
  "selected_token_cost": 455,
  "solver_objective_tuple": [-5, 8, 3, 273, ["..."]],
  "core_weighted_coverage_acquisition_time": 8,
  "display_weighted_coverage_acquisition_time": 8,
  "prefix_auc_horizon": 5,
  "core_padded_prefix_auc": 22,
  "display_padded_prefix_auc": 22,
  "zero_gain_fill_count": 2,
  "baces_steps": []
}
~~~

### 12.2 Step-level fields

~~~json
{
  "step": 1,
  "solver_role": "CORE",
  "solver_core_position": 1,
  "operation": "COVER",
  "display_operation": "ORDINAL_UPGRADE",
  "candidate_pool_idx": 7,
  "candidate_stable_key": "...",
  "evidence_id": "E08",
  "token_cost": 91,
  "valid_coverage_atom_ids": ["A1", "A2"],
  "pair_coverage_levels": {"A1": 2, "A2": 1},
  "display_coverage_levels_before": {"A1": 0, "A2": 0},
  "display_coverage_levels_after": {"A1": 2, "A2": 1},
  "display_upgraded_atom_ids": ["A1", "A2"],
  "display_marginal_coverage_units": 3,
  "display_cumulative_coverage_units": 3,
  "display_cumulative_normalized_coverage": 0.5,
  "display_weighted_acquisition_time_so_far": 3,
  "target_coverage_reached": false,
  "cue_text": "...",
  "cue_source": "claim_atom"
}
~~~

solver_role 只允许：

- CORE；
- FILL。

display_operation 只允许：

- ORDINAL_UPGRADE；
- DISPLAY_ZERO_GAIN。

兼容字段 operation 由 solver_role 确定，只允许：

- COVER；
- ZERO_GAIN_FILL。

### 12.3 禁止 canonical fields

- stance atom_states U/S/R/Q/C；
- state_before/state_after；
- conflicted_atom_ids；
- bridge endpoints；
- CONTRAST/CORROBORATE/BRIDGE；
- reasoning_transition；
- oracle-derived rank。

### 12.4 Compatibility view

临时 adapter 可提供：

~~~text
selector_ordered_indices := display_ordered_indices
selected_indices         := display_ordered_indices
target_resolved          := target_coverage_reached
resolved_atom_rate       := display_cumulative_normalized_coverage
unresolved_atom_ids      := atom_ids whose coverage level is 0
not_direct_atom_ids      := atom_ids whose coverage level is below 2
~~~

可生成最小 mrec_steps/compat_chain_steps 供旧 mrec_min 找 cue，但不得伪造 stance state。

其中 target_coverage_reached 当且仅当当前 display prefix state 等于 terminal_ordinal_state。在主 BACES order 中，这与最后一个 CORE step 首次达到 \(U^\star\) 等价；在 same-set control 中必须按展示顺序重算。它不表示 full human coverage、事实充分或 verifier 已能正确判定。

长期应新增 baces_min rendering，并让 builder 优先读取 baces_steps。

---

## 13. Canonical preprocessing

### 13.1 Oracle/verifier isolation

Solver allowed inputs：

- atoms 与冻结 weights；
- evidence stable identity/text/cost；
- pair-level typed map；
- retrieval metadata；
- duplicate/provenance metadata；
- count/token budgets。

禁止输入：

- oracle_ordered_keys；
- oracle_selected_count；
- gold evidence/rationale/label；
- verifier logits/loss/correctness/reward；
- teacher/checkpoint signal；
- learned weight file；
- 由以上信号融合的 proxy。

Canonical trace 中的 candidate_pool 不是原始 row 的无筛选拷贝，而是 solver-visible projection，只允许包含：

- stable evidence/source/span identity 与 normalized-text hash；
- canonical \(\mathbf q_j\) 及其可审计 pair-row references；
- integer token cost；
- canonical retrieval score；
- duplicate-class metadata。

candidate_pool_fingerprint 对以下 canonical JSON 一起 hash：atoms/quantized weights、上述 candidate projection、\(K_{\min}/K_{\max}/B\)、ordinal utility scale、map/projection/dedup policy versions、cost tokenizer id/revision。原始 oracle、gold、verifier 或 learned metadata 如需审计，只能写入 canonical solver trace 之外的 raw-input sidecar。

因此 oracle poison 的 bitwise-equivalence 范围是 canonical solver decision/objective trace；不得把被刻意改写的 raw metadata 原样回抄进该 trace 后再宣称 fingerprint 应不变。

### 13.2 Dedup before optimization

1. 使用 duplicate_group、source/span identity、normalized text 建 class；
2. 每组只保留一个真实 representative；
3. 不按分量 max/union 不同 duplicate rows 的 ordinal vectors；
4. representative 使用 ascending tuple
   \[
   \left(
   -U(\mathbf q_j),
   -D_j,
   c_j,
   -r_j^{\mathrm{retr}},
   \operatorname{stablekey}(e_j)
   \right)
   \]
   的第一个真实 candidate；\(D_j\) 按第 8.2 节定义，缺失 retrieval score 视为 \(-\infty\)；
5. candidate array index 不得作为最终 tie-break。

### 13.3 Stable identity

stable key 至少组合：

- dataset/event；
- document/report；
- sentence/span；
- normalized-text hash。

Pool merge、序列化与 Python hash seed 变化不得改变输出 stable-key sequence。

---

## 14. Theory and claim boundary

### 14.1 General hardness

忽略 secondary order objective，terminal set utility 为：

\[
U(S)
=
\sum_i w_i\max_{e_j\in S}q_{ij}.
\]

这是 quality-saturated / facility-location-style monotone submodular coverage。令 \(q_{ij}\in\{0,2\}\) 时退化为 weighted maximum coverage，因此一般规模下为 NP-hard。

加入 coverage-unit acquisition-time secondary criterion 后，问题也与 ordered cover/min-sum cover 类问题相邻。

论文可写：

- general formulation is NP-hard；
- bounded atom universe admits exact ordinal-state optimization；
- exactness只针对冻结 map-defined objective。

### 14.2 不可写

- first mathematical definition of fact checking；
- first formal evidence-selection problem；
- first atom-aware evidence coverage；
- first Set Cover / maximum coverage formulation；
- first set-level/submodular evidence selector；
- first prefix-conditioned evidence ranking；
- novel maximum-coverage or DP algorithm；
- exact solution to general NP-hard evidence selection；
- evidence dependency、multi-hop 或 conflict reasoning；
- map coverage equals factual sufficiency；
- sequencing guarantees verifier improvement。

### 14.3 Submodular guarantee boundary

Terminal ordinal coverage \(U(S)=\sum_iw_i\max_{e_j\in S}q_{ij}\) 是 monotone submodular set objective。

完整 lex sequence objective不是普通 set function。标准 cardinality greedy 的 \(1-1/e\) 只可在对应 terminal set objective 与标准假设下陈述，不自动适用于：

- coverage-unit acquisition-time secondary objective；
- joint count+token constraints；
- prompt-truncated final slate。

---

## 15. Closest-work boundary

| Work | 已有对象 | 与 v0.3 的关键差异 |
|---|---|---|
| [DGN](https://arxiv.org/abs/1810.12464) | FEVER 上 cardinality-constrained learned submodular subset selection；forward marginal greedy | latent feature set objective；无 explicit typed ordinal atom state、token budget、padded prefix-coverage AUC objective或 bounded-state exact solver |
| [Minimal Evidence Group](https://aclanthology.org/2025.trustnlp-main.8/) | atomic claim units、binary evidence-unit vector、Set Cover reduction、最小充分 evidence group | 输出 unordered binary full-cover group；要求 full support；无 partial/direct prefix trajectory、joint token feasibility或 ordered exact solver |
| [User-Centric Evidence Ranking](https://aclanthology.org/2026.eacl-long.340/) | full evidence ranking、minimal sufficient prefix、incremental previous-evidence conditioning | 用 first sufficient prefix 衡量成功；无 explicit ordinal atom-incidence trajectory、hard count/token feasible slate或 exact state solver |

v0.3 novelty 只能落在以下联合对象：

> typed-incidence-derived ordinal atom coverage + padded prefix-quality AUC + explicit count/token feasibility + bounded-atom exact optimization.

任何单项都不能声称 first。

---

## 16. 推荐贡献表述

### 16.1 默认安全版本

> We formulate Budgeted Atom-Coverage Evidence Sequencing, which lexicographically optimizes terminal partial-to-direct atom coverage and padded prefix-coverage AUC for a verifier-facing evidence slate over explicit typed atom–evidence alignments under count and token constraints.

> For the bounded atom universe induced by claim decomposition, we provide an exact and auditable ordered-state solver that maximizes terminal graded coverage and minimizes weighted coverage-unit acquisition time among terminal-optimal sequences.

> We evaluate terminal coverage, early-coverage efficiency, realized prompt coverage, and downstream verification under matched candidate pools and budgets.

### 16.2 Conditional first claim

只有完成系统文献审计后，才允许：

> To the best of our knowledge, this is the first fact-checking formulation that jointly optimizes terminal typed-incidence-derived ordinal atom coverage and its padded prefix-coverage AUC under explicit evidence-count and token budgets.

jointly 不得删除。

### 16.3 中文

> 本文将 verifier-facing 证据组织形式化为预算约束的原子覆盖证据排序：在显式 typed atom–evidence 对齐上，词典序优化最终分级覆盖与 padded prefix-coverage AUC，并在受限 atom universe 下给出精确、可审计的求解。

不要写“首次精确定义 fact-checking”。

---

## 17. Experiment contract

### 17.1 Matched baselines

同一 pool、同一 budget、同一 verifier：

1. retrieval top-k；
2. MMR/static ranker；
3. map-quality greedy；
4. terminal-utility-only exact state DP；
5. prefix-conditioned marginal coverage greedy；
6. BACES exact ordered-state DP；
7. MEG-style minimum full-cover baseline；
8. historical learned_marginal_proxy；
9. same-set retrieval/random order controls。

### 17.2 Intrinsic metrics

- terminal normalized ordinal coverage；
- reachable-normalized ordinal coverage；
- weighted coverage-unit acquisition time；
- padded prefix AUC；
- coverage@prefix \(t\)；
- \(K_{\mathrm{core}}\)；
- \(K_{\mathrm{sel}}\)；
- \(K_{\mathrm{final}}\)；
- ZERO_GAIN_FILL rate；
- selected/final token cost；
- truncation coverage loss；
- greedy optimality gap。

### 17.3 External metrics

内部 map 同时定义和评价 objective 会循环验证。必须补：

- human/gold atom–evidence coverage；
- external coverage-unit acquisition time或prefix AUC；
- evidence redundancy；
- downstream label metric；
- human reading-effort 或 sufficiency audit（若声称用户收益）。

### 17.4 Mandatory stratification

- \(m=1\)；
- \(m\ge2\)；
- \(m\ge3\)；
- fully reachable / partially reachable；
- truncated / non-truncated；
- core-dominant / fill-dominant。

若 ordered gains 只来自多 atom subset，正文必须明确，不得用总体平均掩盖。

### 17.5 Mandatory ordinal-objective ablation

主配置 \(q=0/1/2\) 等价于令 normalized per-atom utility：

\[
g_{0.5}(0)=0,\qquad
g_{0.5}(1)=0.5,\qquad
g_{0.5}(2)=1.
\]

这是一项冻结、透明的建模刻度，不声称 partial 在事实语义上天然等于半个 direct。至少比较：

- direct-only：\(g_0=(0,0,1)\)；
- main ordinal：\(g_{0.5}=(0,0.5,1)\)；
- collapsed-valid binary：\(g_1=(0,1,1)\)；
- sensitivity：\(\lambda\in\{0.25,0.5,0.75\}\)，其中 \(g_\lambda=(0,\lambda,1)\)。

所有 variants 使用同一三态 prefix state，只改变 state utility；有理数 utility 在 solver 前统一量化，因此 exact ordered-state DP 与 \(3^m\) 状态界不变。若主结论仅在单个 \(\lambda\) 成立，论文必须降级为 sensitivity finding，不能把 ordinal scale 写成稳健方法优势。

对任意 \(\lambda\)，正式替换为：

\[
U_\lambda(\mathbf x)
=
\sum_iw_i g_\lambda(x_i),
\qquad
\Delta_t^\lambda
=
U_\lambda(\mathbf x^{(t)})-U_\lambda(\mathbf x^{(t-1)}),
\]

\[
T_\lambda(\pi)
=
\sum_t t\Delta_t^\lambda,
\qquad
\operatorname{AUC}_{H,\lambda}
=
(H+1)U_\lambda(\mathbf x^{(L)})-T_\lambda(\pi).
\]

所有 \(\lambda\) variants 使用共同整数 scale \(Q=4\)：direct utility 固定为 4，partial utility 分别为 \(0,1,2,3,4\)。令：

\[
U_{\lambda,\mathrm{pool}}
=
\sum_iw_i\max_j g_\lambda(q_{ij}).
\]

跨 \(\lambda\) 不直接比较 raw \(T_\lambda\) 或 raw AUC，而报告 mean acquired-unit rank \(T_\lambda/U_\lambda(\mathbf x^{(L)})\)（terminal utility非零时）以及：

\[
\overline{\operatorname{AUC}}_{H,\lambda}
=
\frac{\operatorname{AUC}_{H,\lambda}}
{H\,U_{\lambda,\mathrm{pool}}}
\]

（pool upper bound非零时）。主 v0.3 的 \(q=0/1/2\) 与 \(Q=4,\lambda=0.5\) 只差常数缩放，arg optimum 不变。

---

## 18. Phase 0 acceptance tests

### 18.1 Coverage

- 一条 evidence 同步覆盖全部 valid atoms；
- invalid relation/directness/span/confidence 不覆盖；
- support/refute 同 atom 取最高有效 directness level，不生成两份 stance coverage；
- pair-level rows 覆盖 candidate summary；
- step marginal 等于本次新增 ordinal coverage units 的加权和；
- 主配置下，单 atom 的 0→1、0→2、1→2 marginal 分别为 \(w_i,2w_i,w_i\)；当 \(w_i=1\) 时才简写为 1、2、1 units；
- 两条 partial 不累加成 direct，same-level repeat 的 marginal 为 0。

### 18.2 Exactness

- 小实例枚举所有 distinct candidate sequences；
- DP terminal coverage与穷举一致；
- terminal-optimal中 coverage-unit acquisition time与穷举一致；
- 每个 sequence 都满足 \(\operatorname{AUC}_H=(H+1)U-T\)；
- 构造 terminal \(U\) 相同但 ordinal states 不同的实例，验证仍按 \(T\) 决胜；
- 构造 \(U,T\) 相同的实例，依次验证更短 \(L\)、更低 cost、stable-key sequence；
- count-only、token-only、joint constraints分别测试；
- Pareto frontier pruning 与不剪枝 DP/穷举结果一致；
- atom weights量化与 tie-break确定。

### 18.3 Orderedness

- 两个 sequence terminal ordinal state相同但 acquisition time不同，DP选择更早者；
- 任意 core step marginal必须大于0；
- core length不超过 \(\min(K_{\max},2m)\)；
- candidate不可重复；
- random/retrieval order重放 prefix coverage state。
- same-set control 保持 solver_role 不变，但重算 display_operation、display marginal、display \(T\)/AUC。

### 18.4 Fill/budget

- fill只在 core后；
- fill marginal=0；
- fill candidate distinct；
- soft floor预算不足时 underfill而不违规；
- \(K_{\mathrm{core}}\le K_{\mathrm{sel}}\le K_{\max}\)。
- 覆盖 \(K_{\min}=0\)、\(K_{\min}=K_{\max}\)、\(K_{\max}>K_{\mathrm{pool}}\) 与 \(B_{\mathrm{rem}}=0\)；
- fill cost-first rule 达到 eligible zero-gain pool 中的最大可行 fill count。

### 18.5 Oracle poison

增删或随机改写以下字段，canonical solver decision/objective trace 必须 bitwise-equivalent：

- oracle_ordered_keys；
- gold label/evidence；
- verifier score；
- learned weights/reward。

同时验证 candidate_pool_fingerprint 只覆盖 solver-visible projection；poisoned raw metadata 只可出现在 trace 外 sidecar。

### 18.6 Builder

- selected_set policy消费全部 \(K_{\mathrm{sel}}\)；
- 不读取 stored target_resolved 二次截断；
- same-set shuffle保持 visible stable-key set；
- shuffle后 coverage progression重算；
- CORE/FILL solver_role 与 realized display operation不混写；
- final indices/keys与实际 prompt一致；
- truncation coverage loss与 final \(T\)/AUC 可回放。

### 18.7 Determinism

候选序列化、进程与 Python hash seed变化后保持：

- core keys；
- core order；
- selected set；
- fill order；
- objective tuple；
- trace。

---

## 19. Go/no-go rules

### 19.1 Sequencing claim 可以保留

需要同时满足：

- external prefix coverage或human reading-effort显示 early order有效；
- multi-atom subset 上 exact sequence优于同 set 的 retrieval/random order；
- selected/final coverage边界可审计；
- fill并未主导全部结果解释。

### 19.2 Sequencing claim 必须降级

若：

- order-only controls无稳定差异；
- gains仅来自 selected set而非 order；
- \(m=1\) 与 fill样本完全主导总体结果；
- internal map AUC无法被外部评价支持，

则论文中心退回 atom-aware budgeted evidence-set selection，不再把 sequencing 写作核心 novelty。

### 19.3 Ordinal-max objective 必须升级

若第 7–10 条的稳定收益来自当前 state 已饱和后的 same-level corroboration、provenance accumulation 或 same-atom multiple evidence，则 ordinal-max v0.3 no-go；下一版必须显式选择：

- per-level capped multiplicity / multi-cover；
- per-atom demand；
- provenance-aware submodular gain；
- 或其他可验证 objective。

升级前不得把该收益归入 ordinal-max atom coverage。第 7–10 条若仍在进行未饱和的 atom/level 升级，则不单独触发 no-go。

---

## 20. 实现迁移清单

建议新增而非修改历史 policy：

- src/fact_checking/selectors/baces_sequence.py；
- baces_sequence_trace_v0_3 schema/validator；
- build_baces_sequences.py；
- prompt_evidence.policy=selected_set；
- trace_prompt_style=baces_min；
- exact exhaustive-reference tests；
- same-visible-set order-control builder；
- final prompt evidence identity audit。

不覆盖：

- transition_v0_1；
- learned_marginal_proxy；
- historical MREC artifacts；
- existing experiment names。

---

## 21. Phase 0 冻结清单

- [x] 使用 Evidence Sequencing，不使用 Evidence Chain。
- [x] 有序性定义为 prefix-conditioned coverage，不是逻辑依赖。
- [x] 主 objective 词典序最大化 terminal graded coverage、最小化 coverage-unit acquisition time。
- [x] 主定义为 \(q_{ij}\in\{0,1,2\}\) 的 typed-incidence-derived ordinal coverage。
- [x] 主刻度为 partial \(=0.5\)、direct \(=1\)，并强制 direct-only、collapsed-binary 与 \(\lambda\) sensitivity。
- [x] current \(m\le6\) 使用 exact ordered-state DP，状态数为 \(3^m\)。
- [x] \(K_{\min}\) 是 soft rendering floor。
- [x] ZERO_GAIN_FILL 与 coverage core严格分离。
- [x] same-set controls 中 solver_role冻结，display marginal与 \(T\)/AUC 重算。
- [x] BACES 主 builder直接消费 frozen selected slate。
- [x] 自动截断后显式记录 final evidence identities。
- [x] same-set order control必须重放 coverage state。
- [x] oracle/verifier/learned signal与 solver隔离。
- [x] candidate fingerprint只覆盖 versioned solver-visible projection，raw poison metadata置于 sidecar。
- [x] closest-work以 DGN、MEG、User-Centric Ranking 为核心。
- [x] 不声称 first mathematical fact-checking formulation。
- [x] 强制披露 \(K_{\mathrm{core}}\le2m\le12\) 与单 atom退化。
- [x] 若稳定收益来自 ordinal state 饱和后的 zero-gain repeats，触发 ordinal-objective no-go。

改变上述任何一项都需要新版本号，并同步修改 objective、solver state、trace、实验矩阵与论文 claim。
