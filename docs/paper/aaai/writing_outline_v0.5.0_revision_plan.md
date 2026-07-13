# AAAI v0.5 重构计划：从 learned-selector pipeline 到预算约束的原子覆盖与证据呈现

> **Phase 0 boundary erratum（2026-07-12）**：本文档中的 BAREC、SCSG、Evidence-Chain、状态转移、OPEN/CONTRAST/CORROBORATE/BRIDGE 与 chain-quality 主张已被正式退役，不再执行。冻结后的问题与算法边界以 [`atom_coverage_sequencing_phase0_spec_v0.3.md`](atom_coverage_sequencing_phase0_spec_v0.3.md) 为准：主对象改为 **Budgeted Atom-Coverage Evidence Sequencing（BACES）**，使用 \(q_{ij}\in\{0,1,2\}\) 表示 invalid/partial/direct structural coverage，优先最大化 terminal graded coverage，再最小化 coverage-unit acquisition time；当前 \(m\le6\) 的实例使用 \(3^m\) exact ordered-state DP。它保留 prefix-conditioned 边际有序性，但不建模 multi-hop、evidence dependency 或 conflict resolution。本文档中与 oracle isolation、pair-level map、dedup、预算对照、外部评价、实验成本和阶段门槛有关的规划仍保留；冲突时以 v0.3 为准。

> **Paper-framing addendum（2026-07-14）**：v0.5 正文采用“**集合决定语义充分性，顺序决定有限判别者下的呈现可用性**”的两层口径。该口径不改变 BACES v0.3 的实现目标，而是明确其词典序目标的层级：terminal graded coverage 是 permutation-invariant 的 primary set objective；coverage-unit acquisition time 是仅在 terminal-optimal slates 之间生效的 secondary presentation objective。ordered slate 不得解释为 reasoning chain、multi-hop dependency 或证据之间的逻辑先后。

本文档是基于 `writing_outline_v0.4.1.md` 与当前项目实现形成的独立重构方案。它不直接改写 v0.4.1，也不把旧结果自动继承为新方法结果。核心目标是同时完成两件事：

1. 将论文中心从“atomization、retrieval、map、learned selector、verifier 组成的 pipeline”转为“一个新的结构化证据构造问题及其求解算法”。
2. 删除 selector 学习层后，仍通过正式问题定义、bounded-state exact solver、外部 evidence-slate 质量评价和跨域可迁移性，形成可支撑 AAAI 方法论文的贡献闭环。

---

## 0. 论文问题口径冻结：集合本体与呈现顺序

### 0.1 核心立场

本文不在以下两个观点之间二选一，而是把它们放在不同层级：

1. **语义层。** 对一个理想、完整读取且 permutation-invariant 的判别者，证据充分性是集合属性。只要底层集合 \(S\) 不变，重排不应改变它对 claim 的事实支持能力。
2. **接口层。** 现实中的 LLM verifier 与人类读者具有上下文上限、前缀消费、位置偏置与有限注意力。同一集合的排列会改变哪些有效信息更早可见以及判别者能否充分利用该集合。

因此，v0.5 的冻结判断是：

> **观点 1 决定证据选择的数学本体；观点 2 描述有限判别者下的工程现实。本文先确定预算内 terminal coverage 的最优集合族，再在不牺牲该终态质量的所有可行 slate 中联合选择 verifier-facing presentation。**

这一区分也固定了三个术语：

- **evidence set**：承载 terminal semantic coverage 的无序对象；
- **ordered evidence slate**：实际提交给 verifier 的有序接口对象；
- **evidence chain / reasoning path**：暗示逻辑依赖、多跳组合或推理先后，本文禁用。

在模块口径上，**set objective** 先确定 terminal-optimal 集合族 \(\mathcal S^*\)，**evidence presenter/orderer** 再在其所有可行序列化中选择 \(\pi^*\)；次级目标也可以在 terminal-equivalent 的不同 sets 之间决胜，而不只是重排某个预先任意固定的唯一 \(S^*\)。BACES exact solver 为了直接求解单一 lexicographic objective 而联合物化二者，但这种联合计算不改变上述语义分层；正文优先称其为 `BACES solver`，不写“selector 认为 evidence 天然有序”。

### 0.2 两层词典序形式化

给定 atom–evidence 分级对齐 \(q_{ij}\in\{0,1,2\}\)，其中 0/1/2 分别表示 invalid、partial 与 direct coverage。对任意可行 evidence set \(S\)，定义：

\[
x_i(S)=\max_{e_j\in S}q_{ij},
\qquad
U(S)=\sum_{i=1}^{m}w_i x_i(S).
\]

令数量与 token 预算诱导的可行集合族为：

\[
\mathcal F=
\left\{
S\subseteq\mathcal C:
|S|\le K_{\max},
\sum_{e_j\in S}c_j\le B
\right\}.
\]

第一层只定义“选哪些证据”：

\[
U^*=\max_{S\in\mathcal F}U(S),
\qquad
\mathcal S^*=\arg\max_{S\in\mathcal F}U(S).
\]

该目标完全 permutation-invariant。它不因为 solver 采用逐步状态更新或动态规划，就把证据之间解释成有逻辑依赖。

第二层定义“如何呈现 terminal-optimal evidence”。对 sequence \(\pi\) 的底层集合记为 \(S_\pi\)，并令：

\[
\Omega^*=\{\pi:S_\pi\in\mathcal S^*\}.
\]

对 terminal state 中实际获得的 coverage unit，定义其首次进入前缀的位置 \(\tau_{i,\ell}(\pi)\)，以及加权 acquisition time：

\[
\tau_{i,\ell}(\pi)
=
\min\left\{
t:
\max_{1\le s\le t}q_{i,\pi_s}\ge\ell
\right\},
\]

\[
T(\pi)
=
\sum_{i=1}^{m}
\sum_{\ell=1}^{x_i(S_\pi)}
w_i\tau_{i,\ell}(\pi).
\]

随后求：

\[
\pi^*
\in
\arg\min_{\pi\in\Omega^*}
\left(
T(\pi),
|\pi|,
\sum_{e_j\in S_\pi}c_j,
\operatorname{keys}(\pi)
\right)_{\mathrm{lex}}.
\]

这与 BACES v0.3 的联合写法

\[
\min_{\pi}^{\mathrm{lex}}
\left(-U(S_\pi),T(\pi),|\pi|,c(S_\pi),\operatorname{keys}(\pi)\right)
\]

完全等价，但两阶段写法更适合论文叙事：**terminal coverage 决定集合质量，acquisition time 只在不损失 terminal coverage 时决定呈现质量。**

此处 \(\pi^*\) 表示 objective 优化得到的 positive-gain coverage core。若 prompt contract 要求达到 soft \(K_{\min}\)，rendering layer 只能在 \(\pi^*\) 后追加 `ZERO_GAIN_FILL`；fill 不进入上述 terminal/presentation objective，也不能插入 core 改写其顺序。

### 0.3 本文实际解决与不解决的问题

BACES 解决的是：在显式 count/token 预算下，依据 typed atom–evidence incidence 构造 terminal graded coverage 最优、且有效 coverage 尽早进入前缀的 verifier-facing evidence slate。它使用 structural coverage objective，不读取 verdict label、gold evidence、verifier logits 或 verifier-specific reward。

这里的 \(U(S)\) 是显式、可审计的 **map-defined structural surrogate**，不是对人工事实充分性的直接观测；因此 terminal set claim 仍需由 gold/human evidence evaluation 外部验证。并且 \(q=2\) 只表示 direct alignment 提供更多 coverage units，不构成“所有 direct evidence 必须先于所有 partial evidence”的硬全序：atom weights、多 atom 覆盖与预算仍可能使一条 partial evidence 更早出现。

BACES 不声称：

- evidence 具有内在或唯一正确顺序；
- direct evidence 在所有认知或推理场景中都必须绝对优先；
- 前一条 evidence 是后一条 evidence 的逻辑 prerequisite；
- early-coverage order 必然提高任意 verifier 的 verdict accuracy；
- selected slate 构成 multi-hop proof、conflict resolution path 或 faithful explanation。

若判别者完整、等权地使用全部 evidence，第二层会退化为无性能含义的 canonical ordering；若判别者具有 prefix、截断或位置敏感性，第二层才可能带来实际收益。该收益必须通过 strict same-set order control 独立验证，不能由内部 prefix-AUC 指标直接推出。

### 0.4 引出本文工作的冻结桥接句

正文从任务背景过渡到本文方法时，统一使用以下逻辑：

> Evidence sufficiency is fundamentally set-valued, whereas evidence usability can be sequence-sensitive for a bounded verifier. This distinction motivates a lexicographic formulation: we first maximize the terminal graded atom coverage of the selected evidence under explicit budgets, and only among terminal-optimal slates do we prioritize earlier acquisition of useful coverage. The resulting order is a verifier-facing presentation policy rather than a claim about multi-hop reasoning or evidence dependency.

对应中文口径：

> 证据充分性本质上是集合属性，而在有限判别者下，证据可利用性可能对顺序敏感。基于这一区分，本文首先在显式预算下最大化所选证据的最终分级原子覆盖；仅在最终覆盖最优的 slate 之间，进一步使有效覆盖尽早进入前缀。由此得到的顺序是一种面向 verifier 的呈现策略，而非对多跳推理或证据依赖的建模。

---

## 1. 总体判断与路线选择

### 1.1 推荐路线

冻结的问题名称为：

> **Budgeted Atom-Coverage Evidence Sequencing（BACES）**

输出对象为：

> **ordered evidence slate**，而不是 evidence chain。

一句话问题定义为：

> Given claim atoms, a flat evidence pool, and typed atom–evidence alignments, BACES first maximizes the terminal graded atom coverage of a budget-feasible evidence set and then, without sacrificing terminal coverage, minimizes when its useful coverage is first acquired in the verifier-facing slate.

新论文的中心对象应当是：

\[
(\mathcal A,\mathcal C,Q,\mathbf c,B,K_{\max})
\longrightarrow
\text{terminal set optimization}
\longrightarrow
\mathcal S^*
\longrightarrow
\text{coverage-aware presentation}
\longrightarrow
\pi^*,
\]

而不是：

\[
\text{六阶段 pipeline}
\longrightarrow
\text{learned selector}
\longrightarrow
\text{verifier score}.
\]

### 1.2 为什么这条路线仍然可以构成方法贡献

AAAI 的方法贡献不要求组件必须可学习；新的研究问题、算法、理论或系统性实证都可以构成 novelty。这里真正需要守住的贡献不是“没有训练”，而是以下三者的组合：

1. **问题形式化。** 将 evidence selection 写成显式 typed incidence、分级 atom coverage、count/token budget 下的 terminal set optimization，并将 verifier-facing order 严格限制为不牺牲 terminal coverage 的次级目标。
2. **精确求解。** 利用 claim decomposition 诱导的 bounded atom universe，在当前 \(m\le6\) 上用 exact ordered-state DP 联合求解 terminal coverage 与 acquisition-time tie-breaking，而不是训练一个 selector 去近似同一结构规则。
3. **归因评价。** 分开评价 terminal set quality、same-set presentation quality 与 downstream verdict performance，并用 matched-budget、factorial 和 strict same-set controls 隔离各层收益。

因此，“去掉学习层”本身既不是贡献，也不会自动降低贡献；真正决定论文强度的是：两层 BACES formulation 是否清晰且有用、bounded-state exact solver 是否相对 static/greedy/learned baselines 产生可验证优势，以及 terminal set quality 是否有独立于内部 map 的外部证据。只有要把 order 写成 headline empirical contribution 时，presentation gain 才必须另外通过 strict same-set external evidence。

### 1.3 必须保持的表述边界

- 可以说 **selector-training-free**，不能说整个系统 training-free；atomization/evidence map 仍依赖冻结 LLM，verifier 仍可能使用标签训练。
- 可以说结构求解器 **verifier-agnostic**，前提是它不读取 verifier score，且跨 verifier 实验不重新调 objective 或 policy。
- 可以说 **provenance-preserving verifier input**，不能直接说 faithful explanation；后者需要证据删除或替换的因果干预。
- 可以说 solver 对冻结的 map-defined lexicographic objective 是 **exact** 或 **globally optimal**；不能把该保证外推为人工事实充分、verifier correctness 或全局最优的事实核查证据。
- 结构规则直接用于推理后，不再称 `proxy`；正文统一称 `terminal coverage objective` 与 `presentation objective`。
- `ordered slate` 只表示接口顺序；正文不得使用 `evidence chain`、`reasoning path`、`prerequisite` 或 `multi-hop` 描述该顺序。
- order gain 是待验证的条件性主张。若 strict same-set control 不支持它，保留 terminal set contribution，并将次级顺序降为 deterministic rendering policy。
- `MREC` 中的 “Minimal” 建议从论文方法名中移除；仓库内部旧 artifact 名称可以保留以维持兼容。

---

## 2. 新问题与算法的完整逻辑

### 2.1 输入、两种输出表示与预算

给定 claim atoms \(\mathcal A=\{a_i\}_{i=1}^{m}\)、canonical evidence pool \(\mathcal C=\{e_j\}_{j=1}^{n}\)、typed atom–evidence map、正且冻结量化的 atom weights \(\mathbf w\)、additive evidence costs \(\mathbf c\)，以及 count/token budgets \((K_{\max},B)\)。map 经固定 valid gate 与 schema adapter 后投影为：

\[
Q=(q_{ij})\in\{0,1,2\}^{m\times n}.
\]

同一个解同时有两种表示：

\[
S_\pi=\{e_{\pi_1},\ldots,e_{\pi_L}\}
\quad\text{and}\quad
\pi=(e_{\pi_1},\ldots,e_{\pi_L}).
\]

其中 \(S_\pi\) 用于定义 terminal semantic coverage，\(\pi\) 用于定义 verifier-facing prefix quality。二者满足：

\[
L\le K_{\max},
\qquad
\sum_{e_j\in S_\pi}c_j\le B.
\]

### 2.2 Graded terminal set objective

Pair quality 固定为：valid direct alignment 取 2，valid partial alignment 取 1，invalid、irrelevant、context-only 或无可见 key span 的 alignment 取 0。Confidence 只参与 valid gate，不连续乘入 objective；relation 只决定 pair 是否可计数，不引入 stance、冲突或推理状态。

对 set \(S\) 的 atom state 使用 componentwise max：

\[
x_i(S)=\max_{e_j\in S}q_{ij}.
\]

于是：

- partial evidence 可使 atom 从 0 升到 1；
- direct evidence 可使 atom从 0/1 升到 2；
- 两条 partial 不自动合成为 direct；
- same-level corroboration、source diversity 与 stance balance 不产生隐藏增益；
- 一条 evidence 同步更新其所有 valid aligned atoms。

Terminal utility 为：

\[
U(S)=\sum_i w_i x_i(S).
\]

这是 primary、permutation-invariant 的 set objective。所谓 prefix-conditioned marginal 只表示候选相对于“当前已覆盖 atom state”的边际覆盖会变化，不表示 evidence-to-evidence dependency。

### 2.3 Secondary presentation objective

在所有 terminal-optimal slates 中，BACES 最小化 weighted coverage-unit acquisition time \(T(\pi)\)，等价于在固定 horizon 上最大化 padded prefix-coverage AUC。该目标只回答：若一个 bounded consumer 可能只充分利用 slate 的前部，应如何让 partial/direct coverage 更早出现。

Coverage core 中每一步必须具有正 terminal marginal：

\[
\Delta_t
=
U(S_{\pi_{1:t}})-U(S_{\pi_{1:t-1}})>0.
\]

因此 core 不含 duplicate、pure context 或 zero-gain evidence。若 \(K_{\min}\) rendering floor 需要更多 evidence，`ZERO_GAIN_FILL` 只能追加在 core 后，不能插入并改变 solver order；fill 不属于 semantic coverage objective。

### 2.4 Exact ordered-state solver

Prefix state 为：

\[
\mathbf x^{(0)}=\mathbf 0,
\qquad
\mathbf x^{(t)}=\max(\mathbf x^{(t-1)},\mathbf q_{\pi_t}).
\]

启用整数 token budget 时，动态规划状态 \(D[k,\mathbf x,b]\) 记录以 \(k\) 条 core evidence、恰好成本 \(b\) 达到 ordinal state \(\mathbf x\) 的最小 acquisition time；无 token budget 时使用 \(D[k,\mathbf x]\)。终态按：

\[
\left(-U(\mathbf x),T,L,\text{token cost},\text{stable-key sequence}\right)
\]

做严格词典序选择。Stable-key tie-break 保证 candidate input permutation 不影响 canonical output。Prefix-conditioned marginal greedy 仅作为 scalable baseline，不是主 solver。

### 2.5 Solver、selected slate 与 verifier-visible prompt 的边界

新稿必须区分：

- \(K_{\mathrm{core}}\)：exact solver 产生的正覆盖增益 core 长度；
- \(K_{\mathrm{sel}}\)：追加可选 `ZERO_GAIN_FILL` 后冻结的 selected slate 长度；
- \(K_{\mathrm{final}}\)：tokenizer-aware context guard 后实际进入 verifier 的 evidence 数。

\(K_{\max}\) 与 \(B\) 是 solver 的上界；\(K_{\min}\) 是 soft rendering floor；model context length 是 solver 后约束。三者不得混写为一个 budget。主 `selected_set` policy 直接消费冻结 slate，不再由旧 `target_resolved`/minmax controller 二次截取。

### 2.6 困难性、精确性与保证边界

Terminal utility

\[
U(S)=\sum_iw_i\max_{e_j\in S}q_{ij}
\]

是 monotone submodular set function；当 \(q_{ij}\in\{0,2\}\) 时包含 weighted budgeted maximum coverage 特例，因此一般规模为 NP-hard。当前 atomizer 产生 \(m\le6\)，ordinal state 最多为 \(3^m\)，且正增益 core 长度满足：

\[
r\le\min(K_{\max},2m).
\]

因此 exact ordered-state DP 的 arithmetic relaxation complexity 为：

\[
O(nr3^mB)
\]

（有整数 token budget，故对 \(B\) 为 pseudo-polynomial），或 \(O(nr3^m)\)（无 token budget）。其 exactness 只覆盖 canonical pool、冻结 \(Q\)、weights、costs 与 lexicographic objective；不保证人工事实充分、verifier correctness 或 order-induced accuracy gain。

---

## 3. Novelty 的主张层级与近邻边界

### 3.1 主张层级

建议按以下优先级组织全文：

1. **Primary formulation claim：** BACES 将事实核查中的证据组织分成 permutation-invariant terminal set coverage 与 verifier-facing prefix presentation，并以 terminal-first 的词典序目标明确二者优先级。
2. **Method claim：** 对 claim decomposition 诱导的 bounded atom universe，exact ordered-state DP 在不使用 selector supervision 或 verifier feedback 的情况下精确求解冻结目标。
3. **Empirical claim：** 在 matched candidate pool 与 budget 下，BACES 改善 externally evaluated terminal coverage 或性能—成本 Pareto；same-set controls 单独检验 presentation order 是否带来额外收益。[TO VERIFY]
4. **Application claim：** 冻结 slate 可作为 provenance-preserving verifier input，并在不重训 selector 的情况下跨 verifier/数据域迁移。[TO VERIFY]

Atomization、claim-aware chunking、Atom-Union 和 LLM evidence map 是 BACES 的输入接口与实现基础，不再各自包装成同等强度的独立 novelty。任何 `first` claim 必须落在完整组合上：typed ordinal incidence coverage、count/token feasibility、terminal-first prefix objective 与 bounded-atom exact optimization；不能对 atomization、maximum coverage、DP 或 evidence ordering 任一单项声称首次提出。

### 3.2 最危险的近邻与可守差异

定稿 Related Work 时至少逐篇核对以下近邻：

| 近邻方向 | 已有覆盖 | 本文不能单独声称的新意 | 需要守住的差异 |
|---|---|---|---|
| [ClaimDecomp](https://aclanthology.org/2022.emnlp-main.229/) / claim decomposition | 显式或隐式子问题分解 | claim atomization | 在 typed atom–evidence incidence 上优化预算化 graded set coverage 与次级 prefix quality |
| [Complex Claim Verification](https://aclanthology.org/2024.naacl-long.196/) | decomposition、retrieval、summary、verdict pipeline | raw-evidence pipeline | 选择 provenance-preserving raw evidence，并显式分开 terminal set objective 与 presentation objective |
| [Decomposition Dilemmas](https://aclanthology.org/2025.naacl-long.320/) | decomposition 的收益与错误传播 | “decomposition 总是有效” | 系统评价 atom/map 噪声如何传播到 evidence-set/slate construction |
| [CORRECT](https://aclanthology.org/2025.naacl-long.154/) / [HeterFC](https://ojs.aaai.org/index.php/AAAI/article/view/27760) | 图表示和学习式图推理 | evidence graph | 无图依赖语义的 typed-incidence coverage objective 与可审计 exact solver |
| [ProgramFC](https://aclanthology.org/2023.acl-long.386.pdf) | 生成并执行推理程序 | structured multi-step reasoning | 选择原始 evidence units，而非生成 QA/logic program |
| [GAVEL](https://aclanthology.org/2026.findings-acl.1789/) | atomic binding、provenance、sufficient evidence set | binding、provenance、sufficiency 本身 | 分级 terminal coverage、显式 count/token budget、terminal-first prefix objective 与 bounded-state exact optimization |

GAVEL 是当前最危险近邻。最终 novelty 不能停留在“atomic binding + sufficient evidence + provenance”，必须由 graded budgeted objective、exactness boundary、无需 selector supervision、原始证据选择，以及严格分层的 set/presentation evaluation 共同区分。

### 3.3 Reviewer objection 与所需证据

| 可能质疑 | 必须给出的回答 |
|---|---|
| “这只是若干规则拼成的 pipeline。” | 正式定义 \(U(S)\)、\(T(\pi)\)、feasible family、lexicographic priority、stable tie-break、exact DP 与 complexity；Algorithm 1 只写 solver。 |
| “coverage 是自己用 map 算的，不能证明充分。” | 明确称 `map-defined ordinal coverage`；补 gold evidence 或 blind human set-sufficiency evaluation。 |
| “收益来自多看 evidence。” | 做 selector × stopping factorial，并匹配平均 \(K\) 与 tokens。 |
| “证据本来就是无序集合，为什么把 sequencing 当任务？” | 同意 semantic sufficiency 是 set-valued；只将顺序定义为 terminal-optimal 解上的 verifier-facing secondary objective，并用 strict same-set paired tests 验证其工程效应。 |
| “无学习只是把参数藏在规则里。” | 报告离散 \(0/1/2\) contract、正 weights、词典序规则、稳定 tie-break、validation-only 冻结与完整 objective sensitivity。 |
| “LLM map 昂贵且不可靠。” | map 人评、重复性、噪声注入、逐阶段 token/latency/API cost。 |
| “这不是 reasoning chain。” | 明确同意；全文只称 evidence set / ordered slate，不声称 multi-hop reasoning、logical dependency 或 faithful proof。 |
| “内部 prefix-AUC 不代表 verifier 会更准。” | 将 prefix-AUC 作为 structural presentation objective；真实利用增益必须由 external prefix metric、human effort 或 downstream same-set result支持。 |

---

## 4. 论文表述的系统改写计划

### 4.1 题目候选

优先候选：

1. **Budgeted Atom-Coverage Evidence Sequencing for Fact Verification**
2. **Set-Optimal, Prefix-Efficient Evidence Slates for Fact Verification**
3. **Terminal Coverage First: Budgeted Evidence Selection and Presentation for Fact Verification**

题目中不建议同时堆叠 Atom-Union、MREC、LLM、selector-training-free 和 verifier；题目应突出新问题或两层目标。若 same-set order gate 未通过，优先使用第 3 个题目并删除 `Sequencing` headline。

### 4.2 Abstract 重写模板

Abstract 应从“问题缺口”开始，而不是枚举 pipeline：

> Evidence-based fact verification requires selecting a compact body of evidence and serializing it for a downstream verifier. These two decisions are often conflated in a single relevance ranking, although they have different semantics: the evidential adequacy of a fixed set is permutation-invariant, while a bounded verifier may utilize different positions unequally. We formulate **Budgeted Atom-Coverage Evidence Sequencing (BACES)** as a terminal-first lexicographic problem. BACES first maximizes the graded atom coverage of a budget-feasible evidence set over explicit typed atom–evidence alignments; only among terminal-optimal slates does it minimize coverage-unit acquisition time, thereby placing useful coverage earlier without assuming multi-hop reasoning or dependencies between evidence units. For the bounded atom universe induced by claim decomposition, we give an exact, auditable ordered-state dynamic program that requires neither selector supervision nor verifier feedback. We evaluate set quality, same-set presentation effects, and downstream verification separately under matched candidate pools and budgets. Across [datasets/verifiers], BACES [result placeholder], while strict same-set controls [order-result placeholder].

注意：当前 36.66、66.12 和 SciFact 结果来自 learned-selector 流程。BACES 完整重建 trace、训练样本与 verifier 后，才能作为新 `Ours` 数值写入摘要；若 same-set order 无稳定收益，删除摘要最后的 order claim，但不影响 terminal set formulation 与 exact solver claim。

### 4.3 Introduction 的六段逻辑与可直接使用文本

正文按“任务背景 → 集合本体 → 有限判别者 → 现有缺口 → 本文方法 → 评价闭环”展开：

**Paragraph 1 — task background.**

> Evidence-based fact verification requires a system to retrieve a small amount of source-grounded evidence before predicting a verdict. In realistic raw-report settings, the candidate pool is substantially larger than the context that can be passed to a verifier, making evidence organization a budgeted decision rather than a purely retrieval-side concern. Existing systems commonly serialize a fixed top-\(k\) ranking or otherwise couple evidence choice with its input order [CITE].

**Paragraph 2 — 观点 1：集合决定语义本体。**

> At the evidential-content level, however, we model evidence as an unordered set. For an ideal verifier that fully and permutation-invariantly consumes its input, the terminal support supplied by a fixed set should not change under permutation. Evidence sufficiency and atom coverage are therefore set-valued properties; a sequential construction procedure alone does not imply logical precedence, multi-hop composition, or dependencies between evidence units.

**Paragraph 3 — 观点 2：有限判别者带来呈现问题。**

> Practical verifiers are not ideal consumers. Both humans and language models operate with finite attention and context, may observe only a prefix after truncation, and may utilize early and late positions differently [CITE]. Consequently, two permutations of the same evidence set can differ in realized usability even though their evidential content is identical. We treat this as a verifier-facing presentation issue, not as an intrinsic ordering of the evidence itself.

**Paragraph 4 — research gap.**

> This distinction exposes two coupled but non-equivalent optimization questions: which budget-feasible set provides the strongest terminal evidence coverage, and how should a terminal-optimal solution be presented so that useful coverage becomes accessible early? Independent relevance ranking does not directly optimize atom-level coverage under joint count and token constraints, while an order-sensitive objective that is allowed to trade away terminal coverage can improve prefixes at the cost of evidential adequacy. A principled formulation should therefore make the set objective primary and the presentation objective secondary.

**Paragraph 5 — 引出本文工作。**

> We introduce **Budgeted Atom-Coverage Evidence Sequencing (BACES)**. Given claim atoms, a flat candidate pool, and typed atom–evidence alignments with invalid, partial, or direct coverage, BACES first maximizes the terminal graded coverage of a budget-feasible evidence set. Only among terminal-optimal slates does it minimize weighted coverage-unit acquisition time, which is equivalent to maximizing padded prefix-coverage AUC at a fixed horizon. This terminal-first lexicographic design produces an ordered verifier input without assigning logical dependency or reasoning-chain semantics to its elements.

**Paragraph 6 — solver and evidence.**

> Claim decomposition yields a small ordinal state space in our setting. We exploit this structure with an exact ordered-state dynamic program that globally solves the frozen map-defined lexicographic objective and uses stable keys for deterministic tie-breaking. The solver requires neither gold evidence annotations, verdict labels, selector training, nor verifier-derived rewards. Our evaluation separately measures terminal set quality, presentation quality under strict same-set controls, and downstream verdict performance under matched candidate pools and budgets [RESULT PLACEHOLDERS].

Paragraph 1 与 3 中关于现有系统和人类/LLM 位置效应的句子必须补可靠文献；不能只凭直觉作为主要方法依据。

### 4.4 可直接使用的贡献表述

建议最终控制为三项或四项：

1. We formulate **Budgeted Atom-Coverage Evidence Sequencing**, a terminal-first lexicographic problem that treats graded atom coverage as a permutation-invariant set objective and early coverage acquisition as a secondary verifier-facing presentation objective under explicit count and token budgets.
2. We develop an exact and auditable ordered-state dynamic program for the bounded atom universe induced by claim decomposition, globally optimizing the frozen map-defined objective without selector supervision or verifier-derived feedback.
3. We introduce an evaluation protocol that disentangles terminal evidence-set quality, strict same-set presentation effects, and downstream verdict performance under matched candidate pools and budgets. [TO VERIFY]
4. We show that the same frozen BACES policy transfers across [verified datasets/verifiers] without selector retraining or objective retuning. [TO VERIFY]

第 3、4 项只有在实验完成后才能写成完成时；在内部大纲中应保留 `[TO VERIFY]` 标记。

### 4.5 Method 章节的新结构

```text
3 Method
  3.1 System Interface and Overview
      atomization / chunking / Atom-Union 仅作为输入构造的压缩说明
  3.2 Typed Atom-Evidence Map
      valid gate / invalid-partial-direct projection / provenance
  3.3 BACES Problem Formulation
      3.3.1 Feasible Evidence Sets and Terminal Graded Coverage
      3.3.2 Verifier-Facing Presentation and Acquisition Time
      3.3.3 Terminal-First Lexicographic Objective
  3.4 Exact Ordered-State Dynamic Programming
  3.5 Coverage Core, Rendering Floor, and Verifier Integration
      selected slate / tokenizer-aware truncation / label-token prediction
  3.6 Hardness, Exactness, Complexity, and Scope
```

Algorithm 1 只接受 \((\mathcal A,\mathcal C,Q,\mathbf w,\mathbf c,K_{\max},B)\) 并描述 BACES exact solver。Atomization、retrieval、map generation 放 Figure 1 与 Appendix，不再占 Algorithm 1 的前六步。算法正文可以逐层回溯 sequence，但必须先定义其 terminal objective 是 set-valued，避免把 DP transition 误读成证据逻辑依赖。

### 4.6 v0.4.1 逐节迁移表

| v0.4.1 部分 | v0.5 动作 | 说明 |
|---|---|---|
| Abstract | 全部重写 | 删除 learned selector；旧数值暂不继承。 |
| Introduction | 保留 provenance 动机，重建 gap | 明确“semantic set vs verifier-facing presentation”，再引出 terminal-first BACES。 |
| Contributions | 全部重写 | BACES formulation、exact DP、set/presentation attribution、transfer。 |
| Related Work | 扩充并按差异组织 | decomposition、set/submodular evidence selection、budgeted coverage、evidence presentation、provenance/sufficiency。 |
| Overview | 压缩 pipeline | 上游只说明接口，核心篇幅从 typed map 开始。 |
| Task Definition | 拆成三层 | downstream classification、terminal set objective 与 secondary presentation objective 分开。 |
| Atomization | 压缩 | 定义和约束留主文，prompt/schema 入附录。 |
| Claim-aware Chunking | 大幅压缩 | 作为 evidence-unit construction，不再是核心 novelty。 |
| Atom-Union | 保留摘要版 | 作为 recall-supporting candidate construction；单独消融。 |
| Atom-Evidence Map | 保留并收紧 | 固定 valid gate 与 \(q_{ij}\in\{0,1,2\}\) projection；confidence 不连续入目标。 |
| Greedy Chain Construction | 删除旧语义 | 换成 ordinal max state、terminal coverage 与 acquisition time；greedy 仅作 baseline。 |
| Learned Marginal Selector | 整节删除替换 | 换成 BACES Objective、Exact Ordered-State DP、Core/Fill 与 Properties。 |
| Proxy Pairwise Learning | 从主方法移除 | 旧 learned selector 仅保留为 experimental baseline/appendix。 |
| Algorithm | 全部替换 | 只写 BACES exact solver，不写完整 pipeline。 |
| Verifier | 保留并澄清边界 | \(K_{core}\)、\(K_{sel}\)、\(K_{final}\) 分开。 |
| Training and Inference | 改名 | 仅 verifier 训练；selector 无 checkpoint、pair generation 或 optimizer。 |
| Main Results | 保留 protocol，结果重跑 | 旧 learned 结果降为 historical baseline。 |
| Selector Ablation | 全部重构 | terminal-only/exact/greedy/static × fixed budget，并另做 strict same-set order controls。 |
| Map Ablation | 全部重跑 | 直接移除结构字段，不再为每个变体重训 selector 权重。 |
| Prompt Policy | 保留研究问题，单独归因 | selected set、presentation order 与 prompt truncation 分开。 |
| Reliability | 扩展为 atom/map/slate 三层 | set sufficiency 与 order usefulness 使用不同评价协议。 |
| Appendix C/D | 全部替换 | learned features/symbols 改成 ordinal projection、objective、DP state 与 artifact mapping。 |

### 4.7 全文术语替换与禁用主张

| 旧术语 | 新术语 |
|---|---|
| learned marginal selector | BACES exact ordered-state solver |
| heuristic proxy preference | terminal coverage / acquisition-time objective |
| selector weights | atom weights and frozen ordinal contract |
| selector training | 不存在；仅描述 policy configuration |
| resolved rate | map-defined terminal ordinal coverage |
| minimum resolving chain | budget-feasible evidence set / ordered slate |
| evidence-chain order | verifier-facing presentation order |
| state transition | ordinal coverage update |
| selector-driven minmax gain | budget/core/fill effect |
| faithful explanation | provenance-preserving verifier input，除非完成因果干预 |

全文禁止使用 `evidence chain`、`reasoning path`、`logical prerequisite` 描述 BACES slate。无证据时禁止使用：`significantly improves`、`human-sufficient`、`verifier-optimal order`、`minimal`、`oracle-free`、`training-free system`、`faithful explanation` 和 `SciFact SOTA`。`optimal` 仅可限定为 `optimal for the frozen map-defined lexicographic objective`。

---

## 5. 实验逻辑重建

### 5.1 实验研究问题

- **RQ1（set）：** 在相同 candidate pool 与 count/token budget 下，BACES 是否优于 static ranking、greedy 与 learned baselines 的 external terminal evidence-set quality？
- **RQ2（exactness）：** exact ordered-state DP 相对 terminal-only exact DP 与 prefix-conditioned greedy 的 objective value、optimality gap 和成本如何？
- **RQ3（presentation）：** 固定完全相同的 selected evidence set 后，BACES early-coverage order 是否改善 external prefix coverage、人类阅读成本或 verifier 表现？
- **RQ4（downstream）：** set selection gain 与 same-set presentation gain 能否分别转化为 downstream verdict performance？
- **RQ5（transfer）：** 相同且不重训、不重调 objective 的 BACES policy 是否跨 verifier 与数据域保持收益？
- **RQ6（contract）：** binary/ordinal coverage、partial utility、relation/directness/confidence gate、token budget、dedup 与 fill 各自贡献什么？
- **RQ7（robustness）：** atom/map 噪声、distractor、duplicate 和冲突证据如何影响 selected set、presentation trajectory 与最终标签？
- **RQ8（cost）：** 去除 selector training 后，性能—成本—可迁移性之间的实际权衡是什么？

### 5.2 P0：Set policy × capacity 与 strict same-set order factorial

当前 fixed-top5 结果为：source-score 0.354、map-quality greedy 0.349、learned selector 0.339；learned 的 0.367 则来自 minmax(5,10)。因此现有结果不能推出“learning 有效”，也不能直接推出“structure-only 已经更优”，因为 ordering 与 stopping 被混在了一起。

此外，当前 selector-mechanism 表并非严格同 prompt 口径：部分 static baseline 使用 `plain + prefix_topk`，而 map/learned 条件使用 `mrec_min + fixed_topk`。正式对照必须统一成 `mrec_min` rendering、相同 evidence capacity、相同 verifier recipe，并分别训练对应 verifier；现有数字只能作为路线诊断。

最小 set-selection 矩阵为：

| Evidence policy | fixed-5 | fixed-10 | matched-token budget curve |
|---|---:|---:|---:|
| Atom-Union source score | 必须 | 必须 | 建议 |
| map-quality static score | 必须 | 必须 | 建议 |
| terminal-utility-only exact DP | 必须 | 必须 | 必须 |
| prefix-conditioned marginal greedy | 必须 | 必须 | 建议 |
| **BACES exact ordered-state DP** | **必须** | **必须** | **必须** |
| old learned marginal | 必须 | 必须 | 建议 |

历史 minmax(5,10) 作为 compatibility baseline 单列，不能作为 BACES 的主 stopping/controller。对于每个 BACES frozen selected set，另做严格 order-only paired matrix：

- BACES early-coverage order；
- retrieval-score order；
- reverse-BACES order；
- event-specific fixed-seed random order；
- candidate-pool order（若与 retrieval order 不同）。

每个 same-set pair 必须具有相同 stable-key set fingerprint、\(K_{\mathrm{sel}}\)、evidence token multiset、rendering template 与 context guard，并对新顺序重算 external/internal acquisition trajectory。当前仅有 shuffle 约 1 F1 的差异且无统计检验，不足以支撑 presentation claim。

### 5.3 三层公平评价协议

同时采用三种互不替代的协议：

1. **Terminal set evaluation。** 固定 candidate pool 与 budget，忽略排列，比较 gold/human terminal coverage、sufficiency、evidence P/R/F1、redundancy 与成本。
2. **Same-set presentation evaluation。** 固定完全相同的 selected set，只改变 permutation，比较 acquisition time、prefix coverage、人类 reading effort/preference 与 paired verifier outputs。
3. **End-to-end evaluation。** 每个主 evidence policy 使用相同 verifier backbone、训练预算与超参重新生成训练输入并训练 verifier；这是完整系统效果。

可先使用 frozen verifier 做低成本筛选，但这只能标注为 diagnostic，因为 verifier 可能已适应旧 selector 的输入分布。publication-grade 结果不能只用主方法 checkpoint 测所有变体。

### 5.4 P0：独立于内部 map 的 evidence-set 与 presentation 评价

这是本次转向后最关键的新实验。`map-defined ordinal coverage` 同时由 map 驱动并由 map 自己评估，存在自证循环，最多作为内部诊断。

#### 数据集与评价层级

- **SciFact：** 报告 candidate-pool gold recall、conditional selector recall、sentence evidence P/R/F1、abstract evidence/joint 指标；把 retrieval failure、chunk selection failure 与 chunk-to-sentence projection failure拆开。
- **HoVer：** 优先在 gold-document/gold-sentence candidate windows 上做受控 evidence-selection 评价，先隔离 selector，再把 open-domain retrieval 作为后续阶段。
- **LIAR-RAW/RAWFC：** 因缺少唯一 gold evidence set，构建只用于评价、不用于调参或训练的 blind human `Atom-Coverage Audit Set`。

#### 核心指标

- candidate-pool gold recall；
- conditional evidence recall：仅在 gold evidence 已进入候选池的样本上衡量 selector；
- evidence precision / recall / F1；
- joint label + evidence；
- human-confirmed per-atom coverage；
- claim-level set sufficiency；
- redundancy / duplicate rate；
- provenance validity；

Presentation-level 指标只在固定 set 或明确 matched-set protocol 下报告：

- external prefix coverage AUC：
  \[
  \operatorname{AUC}_{\mathrm{prefix}}
  =\frac{1}{K}\sum_{t=1}^{K}\operatorname{GoldOrHumanCoverage}(\mathcal T_{1:t});
  \]
- first relevant evidence rank / first sufficient prefix length；
- human reading effort / order preference；
- paired verifier logit、correctness 与 loss difference。

#### SciFact 风险门槛

当前 SciFact sentence Selection-only 为 40.51，而 abstract Label-only 为 72.41。论文中心改成 evidence selection/presentation 后，前者不再只是次要局限，而是 P0 soundness 风险。

必须二选一：

1. 实质提升 exact sentence localization，并将改进后的官方 evidence metrics 纳入主文；
2. 将主张明确限定为 **chunk/document-level raw-report evidence organization**，同时用 human chunk-level set sufficiency 与 provenance 评价支撑，不能再把 SciFact 作为细粒度 evidence-selection 成功案例。

### 5.5 主结果与迁移实验

主结果至少分为三张表，而不是只报 verdict F1：

1. **Terminal evidence-set table：** gold/human coverage、sufficiency、evidence P/R/F1、redundancy、平均 \(K\)、tokens。
2. **Same-set presentation table：** acquisition time、external prefix AUC、human order preference/effort 与 paired verifier effects。
3. **Downstream verification table：** LIAR-RAW、RAWFC、SciFact/HoVer 的 label 与 joint metrics。

验证 `verifier-agnostic` 的最低要求：

- 至少两个 verifier family，例如 Ministral 与 Llama；
- 对同一数据样本使用完全相同的 BACES selected set 与 order；
- 不因 verifier 变化重新调 selector；
- 比较 BACES 相对 static top-\(k\) 的增益是否保持。

跨域实验应明确记录：

```text
ordinal contract / weights / budgets selected on LIAR-RAW validation
→ freeze BACES policy
→ RAWFC / SciFact / HoVer
```

目标域若重新调 weights、budget policy 或 ordinal threshold，只能称 domain adaptation，不能称 frozen transfer。

### 5.6 Objective 与 solver 消融

在相同候选池与预算下逐项移除：

- collapsed-valid binary vs main ordinal \(0/1/2\)；
- partial utility \(\lambda\in\{0.25,0.5,0.75\}\)；
- direct-only vs partial+direct；
- terminal-utility-only exact DP vs full lexicographic BACES；
- prefix-conditioned marginal greedy vs exact ordered-state DP；
- count-only vs count+token budget；
- canonical dedup on/off；
- coverage core only vs core + `ZERO_GAIN_FILL`；
- multi-atom max-update vs legacy primary-atom-only update（仅作错误实现诊断）；
- BACES vs retrieval/map-quality static fallback。

同时报告 terminal utility、acquisition time、optimality gap、core/fill 长度、tokens 与 verifier-visible truncation loss。不得再报告已退役的 OPEN/CONTRAST/CORROBORATE/BRIDGE operation removal。

### 5.7 Evidence map 消融重做

所有变体使用同一固定 structural policy，不再“在退化特征上重新训练权重”：

- full map；
- no relation；
- no directness；
- no confidence；
- no map；
- oracle/human map upper bound，仅用于诊断，不进入主方法。

每个变体分别报告：

1. matched budget 下的 terminal evidence-set quality，隔离 map 对 evidence choice 的作用；
2. 在各变体自己的 frozen set 内做 BACES/retrieval/random same-set order，检验 map 质量是否改变 presentation trajectory；
3. realized prompt coverage 与性能—token Pareto，单独记录 truncation/fill 的影响。

不得继续使用 Macro-F1/平均 \(K\) 作为主要“证据效率”指标；改用 matched-cost 比较、Pareto frontier 或 budget-curve AUC。

### 5.8 Candidate construction 与上游消融

Atom-Union 继续作为 supporting component，使用既有受控方案重做：

- claim-level baseline only；
- atom-route only；
- union without MMR；
- full union；
- candidate pool size sensitivity。

所有变体都要区分 candidate recall 与 conditional selector quality。否则上游没有召回 gold evidence 时，不能把失败归因给 BACES。

### 5.9 人工评价与可靠性

建议建立三层评价：

1. **Atom 层：** faithfulness、completeness、atomicity。
2. **Map 层：** relation macro-F1/confusion matrix、directness weighted-\(\kappa\) 或 Spearman、confidence calibration。
3. **Slate 层：** set protocol 评价 per-atom evidence correctness、claim-level sufficiency、redundancy 与 provenance；same-set protocol 单独评价 order usefulness 与 reading effort。

推荐样本：

- 200 条随机 claim 作为 headline sample；
- 100 条 hard cases，覆盖多 atom、map 冲突、core/fill、truncation 与方法分歧；
- 500–800 个 evidence–atom pairs，按 relation 分层采样，汇总时按总体分布重加权；
- 2 位独立标注者、分歧 adjudication，第三位标注者复核子集。

Slate 层必须拆成两个 blind protocol，不能用一个 pairwise task 同时改变 set 与 order：

**Protocol A — set quality（matched cost, order-neutral rendering）：**

```text
BACES evidence set
vs.
best static/learned baseline evidence set
```

要求标注者完整查看全部 evidence card；card order 在标注者间 counterbalance，左右方法位置随机，不展示方法名与系统 verdict。评价 per-atom coverage、claim sufficiency、redundancy 与 provenance，不记录该 protocol 的 order preference。

**Protocol B — presentation quality（strict same set）：**

```text
BACES early-coverage order
vs.
retrieval/random order of the identical evidence set
```

评价 first-sufficient-prefix、reading effort 与 order preference。充分 set 通常不唯一，不建议使用 exact set/sequence match；分别报告两个 protocol 的 win/tie/loss，不能把 Protocol A 的偏好归因于 order。

当前 annotation project 的数据库和结果文件正在变化，执行前必须重新审计已有标注数、用户分配和未完成任务，不应沿用旧文档中“数据库为空”或“基础设施已就绪”的静态假设。

当前本地 SQLite 快照显示：Exp1 Atom Quality 的 Yulin/Zhiqiang/Zijie 完成数约为 38/257/16（各自目标 257），Exp2 Evidence Map 约为 48/1/37（各自目标 250）。尚不存在两位标注者完成同一完整任务集的条件，正式 A/B 导出、IAA 与 ECE evaluator 也仍待完成。因此现阶段不能在 Abstract 或 Contributions 中把 reliability study 写成既成结果。

API repeatability 还需先修复三个前置问题：atomization 请求的 `seed` 是否真正进入 HTTP payload、map cache key 是否覆盖 top-p/max-tokens/thinking/prompt hash，以及 schema retry 次数是否符合实验协议。冻结已有 map 做结构 selector 主实验不受阻塞，但稳定性实验必须在这些问题修复后开始。

### 5.10 鲁棒性与因果使用证据

P1 鲁棒性矩阵：

- map alignment drop：5/10/20/30%；
- relation flip；
- directness/confidence perturbation；
- 高 retrieval-score irrelevant distractor 注入；
- duplicate source/span 注入；
- stance-conflicting evidence 注入；
- atom drop、split、merge 或 no-atom 条件；
- 更换一个 frozen map annotator model。

若论文要进一步声称 selected slate 被 verifier 因果使用，可补充：

- 删除 top-priority step；
- 删除随机 step；
- reverse/shuffle；
- 仅保留未选 evidence；
- 用未选 evidence 替换已选 evidence；
- 比较 label flip、gold-label probability 或 decision margin。

这些干预只能支持“verifier 是否使用 selected slate”的因果证据，不能把 slate 升格为逻辑链或多跳 proof。若不做或结果不支持，全文只称其为 structured/provenance-preserving verifier input。

### 5.11 效率与成本

效率报告按阶段拆分：

| 阶段 | 报告内容 |
|---|---|
| Atomization | calls/claim、input/output tokens、latency、cost |
| Chunking/Retrieval | CPU/GPU time、candidate count |
| Evidence map | calls、tokens、latency、API cost、cache hit |
| Selector | preference generation/training 是否存在、CPU ms、memory |
| Verifier | visible tokens、GPU latency、throughput |
| End-to-end | total wall time、total cost |

无学习版本可守住的优势是：不生成 preference pairs、不训练或存储 selector checkpoint、不依赖 verifier-specific supervision、可直接迁移和审计。不能因为 selector 很便宜就把 map API 占主导的整个 pipeline 称为高效。

### 5.12 统计协议

- ordinal contract、atom weights、budget policy 与 map gate 只在 validation 上确定；test 只在冻结后评估。
- 核心 end-to-end 方法至少 3 个 verifier seeds，报告 mean ± std。
- sample-level paired bootstrap 计算 Macro-F1 差值与 95% CI。
- 多个消融比较使用 Holm correction。
- 人评 win/tie/loss 使用 bootstrap CI 或配对检验。
- 报告每个结果对应的 exact config、candidate/map cache hash、trace artifact、verifier checkpoint 和 metric file。
- frozen cache 用于主结果；API 重复调用只用于稳定性子实验。

---

## 6. 主文表格与图的展示规划

### 主文优先级

1. **Figure 1：Problem and solver。** 显示 `flat pool → typed ordinal matrix → terminal set objective → terminal-optimal family → presentation objective → ordered slate → verifier`；不再画成普通六阶段流水线或推理链。
2. **Figure 2：Set–presentation distinction。** 用同一 coverage matrix 展示 terminal-optimal set、BACES/retrieval 两种 permutation 及其 prefix coverage trajectory，并保留 source/span provenance。
3. **Table 1：问题定位。** 与 decomposition、raw-evidence pipeline、graph reasoning、program reasoning、GAVEL 的监督、状态、预算、输出结构和 provenance 对照。
4. **Table 2：Terminal evidence-set quality。** gold/human coverage、sufficiency、evidence recall、redundancy、平均 tokens。
5. **Table 3：End-to-end results。** LIAR-RAW、RAWFC 与 gold-evidence dataset 的 label/joint metrics。
6. **Table 4：Strict same-set presentation controls。** acquisition time、external prefix AUC、human effort 与 paired verifier effects。
7. **Figure 3：Performance–cost Pareto。** Macro-F1 与 set sufficiency 分别对 prompt tokens 作图。
8. **Table 5：Atom/map/slate reliability。** 人评与校准。
9. **Table 6：Cross-domain × cross-verifier transfer。** 明确 selector 是否重调。

### Appendix

- 完整 algorithm/objective 与 proof；
- prompts、schemas、thresholds、tie-break；
- relation confusion matrix 与 confidence reliability diagram；
- 完整 ordinal-objective/map/candidate-pool ablations；
- noise/distractor curves；
- stage-wise cost；
- 3–5 个成功、失败和分歧 case；
- artifact/config/hash 清单。

---

## 7. 当前实现到新方法的工程迁移计划

### 7.1 新建独立 policy，不复用旧 proxy 入口

不能把旧 `rank_candidates_by_proxy` 或 transition selector 简单改名，因为旧路径仍可能读取 `oracle_ordered_keys`，且旧 state、tie-break 与 stopping contract 不满足 BACES 定义。

冻结入口为：

```text
selection_policy: baces_lexicographic_early_coverage_v0_3
solver_version: baces_exact_ordered_state_dp_v0_3
prompt_policy: selected_set
```

旧 `learned_marginal_proxy` 只保留为实验 baseline，训练脚本和 weight file 不再是新主流程依赖。

### 7.2 P0 实现不变量与测试

1. **Oracle poison test：** 同一样本写入任意 `oracle_ordered_keys` 前后，BACES terminal set、order 与 objective tuple 必须完全相同。
2. **Label/verifier independence test：** 修改 gold label、verifier score 或 checkpoint 路径不改变 trace。
3. **Invalid alignment test：** irrelevant/background/context/无 visible key span pair 必须投影为 \(q_{ij}=0\)。
4. **Multi-atom update test：** 一条 evidence 的全部 valid aligned atoms 按 componentwise max 同步更新。
5. **Candidate permutation test：** 候选输入顺序打乱后，stable-key set 与 canonical sequence 不变。
6. **Duplicate/zero-gain test：** duplicate 与 zero-gain candidate 不得进入 coverage core；fill 只能追加。
7. **Budget test：** \(K_{\max}\)、token cost 与 \(B\) 始终满足约束。
8. **Layer-boundary test：** \(K_{core}\)、\(K_{sel}\)、\(K_{final}\) 分别写入 artifact。
9. **State replay test：** 从 trace 逐步回放可复现 ordinal state、\(U\)、\(T\)、core/fill role 与最终 tuple。
10. **Exhaustive equivalence test：** 小实例上 exact DP 与所有 distinct feasible sequences 的穷举最优 tuple 完全一致。

旧 artifact 即便仍包含 `oracle_ordered_keys` 也不能靠 wrapper “不传入”来保证结构纯净；新 policy 应在代码层根本不读取该字段，并通过 poison test 证明。

### 7.3 产物契约

每条 trace 至少记录：

- candidate stable key 与 provenance；
- canonical \(q_j\) vector 与 valid pair projection；
- ordinal coverage state before/after；
- marginal coverage units、\(\Delta U\) 与 acquisition-time increment；
- `solver_role`（CORE/FILL）与随 order control 重算的 `display_operation`；
- terminal \(U\)、\(T\)、length、cost 与 stable-key tuple；
- display/final prefix acquisition metrics；
- cumulative token/evidence cost；
- core/fill/underfill/truncation reason；
- \(K_{core}\)、\(K_{sel}\)、\(K_{final}\)；
- selected-set fingerprint 与 candidate-pool/config fingerprint；
- selector config/hash；
- 明确的 `uses_oracle=false`、`uses_verifier=false` audit 字段。

### 7.4 旧结果的处理

- v0.4.1 的 learned-selector 结果保留为 historical/diagnostic baseline。
- 不把旧 36.66、66.12、72.41 等数值直接改名为 BACES。
- 新主结果必须从 BACES traces 开始，重建 train/val/test verifier rows。
- frozen-verifier quick test 可用于路线筛选，但不进入最终主表。
- 旧 map cache 可以复用，前提是新 selector 完全忽略 oracle 字段，并记录 cache schema/hash。

### 7.5 当前可复用资产与真实缺口

当前已有的 evidence-map feature 规模为：

| 数据集 | Train | Validation | Test/official dev | 可复用方式 |
|---|---:|---:|---:|---|
| LIAR-RAW | 10,065 | 1,274 | 1,251 | 无需新增 API，可直接生成 BACES traces |
| RAWFC baseline20 | 1,612 | 200 | 200 | 无需新增 API，可直接生成 BACES traces |
| SciFact | 809 | 300 | 300 | 可重建 fixed-9/direct traces 并使用 official dev scorer |

仓库已有四类可借用实现，但没有任何一类可直接冒充新主方法：

1. `src/fact_checking/selectors/minimal_resolving_chain.py` 的 `transition_v0_1` 有状态机，但缺少拟议的完整结构边际目标。
2. `src/fact_checking/selectors/mrec_learned_marginal.py` 的 proxy ranking 最接近旧结构规则，但包含 oracle teacher 分支，只能作为反例和迁移参考。
3. `src/fact_checking/selectors/evidence_chain_graph.py` 已有 coverage、pair utility、redundancy、source 和 length 等 budgeted objective，但使用旧 graph schema，没有 MREC 的显式 atom state。
4. `src/fact_checking/selectors/map_selector_ablation.py` 已有 weighted set cover/minimal group，可作为结构 baseline。

现有 `scripts/sentence_trace_method/check_mrec_diagnostics.py` 可扩展复用 duplicate、fallback、token cost 与 prompt leak 检查；`paired_significance.py` 可复用做 paired significance。需要新增 ordinal state legality、terminal utility、acquisition time、core/fill、prefix AUC、selected-set fingerprint 与 oracle-invariance 汇总。

当前 Atom-Union 消融也不能直接标为完成：`baseline_only`、`atom_only`、`union_no_mmr` 已有较多可复用 pool/map/trace，然而 `union_full` 的三个 split 曾因 HTTP 402 insufficient balance 导致 API annotation 全部 fallback。该产物不能作为 full-union 正式结果；须补齐真实 map 或将该消融从主文移除。

### 7.6 最小新增训练包

在复用现有 map 的前提下，最小 publication-grade 新增 verifier runs 建议为：

1. LIAR BACES selected_set，\(K_{\max}=5\)；
2. LIAR BACES selected_set，\(K_{\max}=10\) 且 matched token budget；
3. RAWFC BACES，使用 LIAR validation 冻结的 budget policy；
4. SciFact BACES，\(K_{\max}=9\)；
5. LIAR 最佳 static baseline 的统一 rendering 与 matched budget；
6. LIAR direct-only / ordinal / collapsed-binary objective 消融；
7. LIAR direct no-map 或 no-directness 关键消融。

这约对应 7–8 个新 verifier training conditions；三随机种子应优先覆盖主方法与最佳 static/learned 对照，而不是平均铺给所有次要消融。完整版再增加第二 verifier、完整 map ablation 和更多 budget points。

---

## 8. 分阶段执行计划与 go/no-go 门槛

### Phase 0：定义冻结与实现审计

**任务：**

- 按 `atom_coverage_sequencing_phase0_spec_v0.3.md` 冻结 BACES 输入、ordinal \(0/1/2\) coverage contract、lexicographic sequence objective、预算、dedup、stable key 与 tie-break。
- 固定 multi-atom 同步 max-update、coverage core 与 ZERO_GAIN_FILL 分离、solver order 直接进入 selected slate、\(K_{\min}\) 仅作为 soft rendering floor。
- 实现独立 `baces_lexicographic_early_coverage_v0_3` policy；以所有 distinct candidate sequences 穷举校验 terminal utility、acquisition time 与 tie-break，并把 prefix-conditioned marginal greedy 作为 optimality-gap baseline。
- 对 train/val/test artifact 做 oracle/verifier dependency audit。

**完成判据：** 所有不变量测试通过；DP 与小实例穷举的 terminal ordinal utility、weighted acquisition time 与 stable-key sequence 一致；同一 map 与 config 可确定性复现 coverage core、selected slate 与 final visible identities；poisoned oracle 字段不改变任何输出。

### Phase 1：低成本路线判定

**任务：**

- 在 LIAR-RAW validation 上完成 BACES exact ordered-state DP、BACES greedy、static、map-quality、learned 的 matched count/token-budget 对照；历史 fixed-5 与 minmax(5,10) 作为兼容 baseline。
- 使用 frozen verifier 做 quick diagnostic。
- 统计 all/reachable-normalized ordinal coverage、weighted acquisition time、padded prefix AUC、0→1/0→2/1→2 transitions、\(K_{\mathrm{core}}/K_{\mathrm{sel}}/K_{\mathrm{final}}\)、ZERO_GAIN_FILL、tokens、redundancy、truncation loss 与 greedy optimality gap；按 \(m=1,m\ge2,m\ge3\) 分层。
- 完成 direct-only、main ordinal、collapsed-valid binary 与 partial-utility \(\lambda\in\{0.25,0.5,0.75\}\) 的 objective sensitivity。

**Go 条件：** BACES exact-DP 在 matched budget 下相对最佳 static baseline 至少 Pareto 非劣，并出现可外部验证的 terminal coverage 或 same-set early-order 优势；若只靠更大 \(K\)、fill 或 internal-map score 获益则不进入下一阶段。

### Phase 2：外部 evidence-slate 质量证据

**任务：**

- SciFact error decomposition：candidate recall、selector conditional recall、chunk-to-sentence projection。
- HoVer gold-window selector evaluation，或建立 blind Atom-Coverage Audit Set。
- 完成人工 atom/map/evidence-slate 三层评价；pair quality 使用 invalid/partial/direct 标签，并独立重算 terminal coverage 与 prefix acquisition。

**Go 条件：** 至少有一套 gold 或 blind-human 外部评价支持 terminal ordinal coverage 或 same-set prefix-order 效率；不能只依赖内部 map objective 与 verdict F1。

### Phase 3：Publication-grade end-to-end 主实验

**任务：**

- 用冻结的 BACES solver 重新构建 LIAR-RAW/RAWFC train/val/test verifier inputs。
- 使用 matched Ministral-LoRA 配置训练与评估；核心方法运行 3 seeds。
- 在第二 verifier family 上复用完全相同 traces。
- 冻结 ordinal coverage contract 与 solver policy 后做 RAWFC/SciFact/HoVer transfer。

**完成判据：** intrinsic coverage/sequencing table、downstream table、cross-verifier/domain table 均有完整 artifact 与统计区间。

### Phase 4：归因、鲁棒性与成本

**任务：**

- ordinal-objective/map/candidate-pool ablations；
- strict same-visible-set retrieval/random/reverse 与 matched-token controls；
- map noise、distractor、duplicate、atom corruption；
- stage-wise latency/token/API cost；
- evidence intervention。

**完成判据：** 每个核心设计都能由独立实验支撑；若 same-level repeated evidence 有稳定效用则触发 multi-cover 升级，若 order-only control 无稳定差异则降级 sequencing claim。

### Phase 5：写作迁移与 claim audit

**任务：**

- 新建 v0.5 正文，不在旧稿中机械替换术语。
- 先完成 Problem/Method/Experiment Protocol，再写 Abstract 与 Introduction。
- 将所有结果句连接到明确 table/metric/artifact。
- 对 `learned`、`proxy`、`oracle`、`minimal`、`optimal`、`faithful`、`significant`、`SOTA` 做全文审计。

**完成判据：** 论文从标题到结论都以 BACES problem 与 exact ordered-state solver 为中心；任意主张均能指向外部评价或受控实验。

---

## 9. 优先级清单

### 投稿必须完成（P0）

- BACES 正式问题定义、hardness boundary 与独立 Algorithm 1；
- strict valid predicate、ordinal multi-atom max-update、稳定 tie-break 和 oracle poison test；
- exact ordinal solver vs greedy/static/learned 的 matched-budget factorial；
- matched-cost selector controls 与 strict same-visible-set order controls；
- 至少一套外部 gold 或 blind-human ordinal coverage/prefix-quality 评价；
- SciFact sentence-localization 风险的实质处理或明确 scope 收缩；
- BACES 全量重建 verifier data 后的主结果；
- atom/map/slate reliability；
- 统计区间与 artifact/config 审计。

### 强烈建议（P1）

- 两个 verifier family 上的 frozen-trace 迁移；
- map noise 与 distractor robustness；
- terminal-only/full-lex、binary/ordinal 与 exact/greedy objective ablations；
- performance–token Pareto；
- evidence causal intervention；
- stage-wise API/GPU/CPU 成本。

### 可选增强（P2）

- 与实现一致的 NP-hardness/submodular proof；
- 第二个 map annotator model；
- open-domain HoVer/FEVEROUS 完整流程；
- 更多 backbone 或更大规模 candidate pool。

---

## 10. 最终投稿 go/no-go 标准

以下四项建议设为硬门槛：

1. **Terminal set 有效。** BACES 在 matched budget 下相对最佳 static/greedy baseline 至少处于 Pareto frontier，并在外部 terminal evidence-set 主指标上具有稳定优势或严格非劣的成本优势。
2. **外部有效。** 存在 gold evidence 或 blind human evidence-set 评价；不能只依赖 verdict F1 和 map-defined coverage。
3. **Presentation claim 过 gate。** 只有 strict same-set control 在 external prefix metric、human effort 或 downstream verifier 上出现稳定收益，才能把 sequencing/order 写成 headline empirical contribution；否则将其降为 deterministic presentation policy，保留 set formulation 与 exact solver。
4. **迁移与 scope 一致。** 同一个未重训、未重调的 BACES policy 至少跨两个数据域或两个 verifier family 保持主要收益，且 SciFact/HoVer 等结果不与论文的粒度主张明显冲突。

若第 2、4 项无法完成，论文应收缩为：

> selector-training-free chunk/document-level raw-report evidence organization

并将其定位为经验型系统研究，而不应继续声称通用、细粒度 evidence selection 的方法贡献。第 3 项失败不否定 primary set result，只否定“顺序带来外部可用性收益”的实证主张。

---

## 11. 推荐的最终论文骨架

```text
1 Introduction
  Evidence as a semantic set
  Bounded verifiers and presentation sensitivity
  Terminal-first BACES formulation
  Exact ordered-state solver
  Contributions

2 Related Work
  Claim decomposition
  Set/submodular evidence selection
  Budgeted coverage optimization
  Evidence ranking and presentation
  Graph/program reasoning (boundary)
  Provenance and sufficiency evaluation

3 Method
  3.1 System Interface
  3.2 Typed Atom-Evidence Map
  3.3 BACES Problem Formulation
      3.3.1 Terminal Graded Set Coverage
      3.3.2 Verifier-Facing Presentation
      3.3.3 Terminal-First Lexicographic Objective and Budgets
  3.4 Exact Ordered-State Dynamic Programming
  3.5 Coverage Core, Rendering Floor, and Verifier Integration
  3.6 Hardness, Exactness, and Scope

4 Experiments
  4.1 Tasks, Baselines, and Fair Protocol
  4.2 Terminal Evidence-Set Quality
  4.3 Strict Same-Set Presentation Controls
  4.4 Downstream Fact Verification
  4.5 Exactness, Objective Ablations, and Regret
  4.6 Cross-Domain and Cross-Verifier Transfer
  4.7 Reliability, Robustness, and Cost

5 Analysis
  Human set sufficiency and order preference
  Coverage acquisition trajectories
  Evidence intervention
  Failure cases

6 Limitations and Ethics
  LLM map errors and cost
  Source credibility not modeled
  Chunk-level vs sentence-level boundary
  API/cache reproducibility

7 Conclusion
```

该骨架的关键变化是：论文先确立“semantic set、operational presentation”的层级边界，再提出 terminal-first BACES，并解释 exact DP 如何联合求解词典序目标；完整 pipeline 只作为应用与评价载体。这样既不会把 evidence slate 误写成推理链，也能把 set-selection contribution 与 order-only empirical claim 分别审查。
