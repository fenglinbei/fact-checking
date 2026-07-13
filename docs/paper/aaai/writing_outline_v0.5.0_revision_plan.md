# AAAI v0.5 重构计划：从 learned-selector pipeline 到预算约束的原子覆盖证据排序

> **Phase 0 boundary erratum（2026-07-12）**：本文档中的 BAREC、SCSG、Evidence-Chain、状态转移、OPEN/CONTRAST/CORROBORATE/BRIDGE 与 chain-quality 主张已被正式退役，不再执行。冻结后的问题与算法边界以 [`atom_coverage_sequencing_phase0_spec_v0.3.md`](atom_coverage_sequencing_phase0_spec_v0.3.md) 为准：主对象改为 **Budgeted Atom-Coverage Evidence Sequencing（BACES）**，使用 \(q_{ij}\in\{0,1,2\}\) 表示 invalid/partial/direct structural coverage，优先最大化 terminal graded coverage，再最小化 coverage-unit acquisition time；当前 \(m\le6\) 的实例使用 \(3^m\) exact ordered-state DP。它保留 prefix-conditioned 边际有序性，但不建模 multi-hop、evidence dependency 或 conflict resolution。本文档中与 oracle isolation、pair-level map、dedup、预算对照、外部评价、实验成本和阶段门槛有关的规划仍保留；冲突时以 v0.3 为准。

本文档是基于 `writing_outline_v0.4.1.md` 与当前项目实现形成的独立重构方案。它不直接改写 v0.4.1，也不把旧结果自动继承为新方法结果。核心目标是同时完成两件事：

1. 将论文中心从“atomization、retrieval、map、learned selector、verifier 组成的 pipeline”转为“一个新的结构化证据构造问题及其求解算法”。
2. 删除 selector 学习层后，仍通过正式问题定义、bounded-state exact solver、外部 evidence-slate 质量评价和跨域可迁移性，形成可支撑 AAAI 方法论文的贡献闭环。

---

## 1. 总体判断与路线选择

### 1.1 推荐路线

建议将新问题暂定名为：

> **Budgeted Atom-Resolving Evidence-Chain Construction（BAREC）**

将直接结构求解算法暂定名为：

> **State-Conditioned Structural Greedy（SCSG）**

一句话问题定义为：

> Given noisy typed atom–evidence alignments, construct an ordered and provenance-preserving evidence chain that resolves claim atoms under an evidence budget, without gold chains, verdict labels, verifier feedback, or selector training.

新论文的中心对象应当是：

\[
(\mathcal A,\mathcal C,M,B)
\longrightarrow
\text{state-conditioned structural solver}
\longrightarrow
\mathcal T_B,
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

1. **新问题。** 将 evidence selection 从候选独立打分的静态 top-\(k\) 排序，改写为具有状态转移、角色约束、预算和可变长度输出的 evidence-chain construction。
2. **新算法。** 以 OPEN、CONTRAST、CORROBORATE、BRIDGE 等结构操作定义候选的前缀条件价值，直接执行可审计的结构策略，不再对该策略生成的偏好进行自蒸馏。
3. **新评价。** 将 downstream label performance 与 chain correctness、sufficiency、redundancy、provenance、成本和迁移性分开评价。

因此，“去掉学习层”本身既不是贡献，也不会自动降低贡献；真正决定论文强度的是 BAREC 是否被形式化为一个清晰问题、SCSG 是否是一个完整且非平凡的求解器，以及链质量是否有独立于系统内部 map 的外部证据。

### 1.3 必须保持的表述边界

- 可以说 **selector-training-free**，不能说整个系统 training-free；atomization/evidence map 仍依赖冻结 LLM，verifier 仍可能使用标签训练。
- 可以说 **verifier-agnostic**，前提是 selector 不读取 verifier score，且跨 verifier 实验不重新调 selector。
- 可以说 **provenance-preserving verifier input**，不能直接说 faithful explanation；后者需要证据删除或替换的因果干预。
- 不能说 minimum、globally optimal 或 shortest chain，除非补充严格证明。
- 结构规则直接用于推理后，不再称 `proxy`；正文统一称 `structural objective`、`structural policy` 或 `structural marginal rule`。
- `MREC` 中的 “Minimal” 建议从论文方法名中移除；仓库内部旧 artifact 名称可以保留以维持兼容。

---

## 2. 新问题与算法的完整逻辑

### 2.1 输入、输出与预算

给定 claim atoms \(\mathcal A=\{a_i\}_{i=1}^m\)、候选 evidence units \(\mathcal C=\{u_j\}_{j=1}^n\)、typed atom-evidence map 与证据成本：

\[
M(u_j,a_i)=(r_{ij},d_{ij},\gamma_{ij}),
\qquad
\ell_j=\operatorname{cost}(u_j),
\]

其中 \(r_{ij}\) 为 relation，\(d_{ij}\) 为 directness，\(\gamma_{ij}\) 为 map confidence。使用 \(\gamma\) 而非 \(c\) 表示 confidence，以免与 claim \(c\) 混淆。

一个 BAREC 实例写作：

\[
\mathcal I=(\mathcal A,\mathcal C,M,\ell,
B_{\mathrm{sel}},k_{\min},k_{\max},\rho_{\mathrm{target}}).
\]

输出为有序 evidence chain：

\[
\mathcal T_B=[u_1,\ldots,u_K],
\quad K\le k_{\max},
\quad \sum_{u\in\mathcal T_B}\ell(u)\le B_{\mathrm{sel}}.
\]

这里的核心任务是构造 \(\mathcal T_B\)，事实核查标签只是该链的一个下游用途。

### 2.2 严格区分 resolving、context 与 invalid alignment

必须先定义可用于 atom 状态解析的严格谓词：

\[
V^{\mathrm{resolve}}_{ij}
=
\mathbf 1[
r_{ij}\in\{\mathrm{support,refute,qualify}\}
\land d_{ij}\ge\tau_d
\land \gamma_{ij}\ge\tau_\gamma
].
\]

实现和论文均需遵循以下不变量：

- `irrelevant` 不得产生 atom coverage、new relation、resolution、novelty 或任何其他 structural gain。
- `background/context` 只能触发 BRIDGE，不得将 atom 从 unresolved 更新为 resolved。
- directness、confidence 或 relation 无效的 alignment 不得通过其他字段间接触发 OPEN。
- duplicate evidence 不得伪造 corroboration；corroboration 至少需要新的 provenance/source 或独立 span。
- “map-implied structurally resolved” 不能写成“该 atom 的事实真值已被确认”。

这一步同时解决当前实现中 background/irrelevant 可能泄漏到 coverage、无效 alignment 可能提供 directness，以及 context 被错误计入 partial resolution 的风险。

### 2.3 Atom state 与结构操作

建议将每个 atom 的状态表示为可审计的集合状态，而不必继续依赖旧 learned selector 的 soft state：

\[
H_i^{(t)}=
(R_i^{(t)},Q_i^{(t)},S_i^{(t)},z_i^{(t)}),
\]

其中：

- \(R_i^{(t)}\)：已观察到的 resolving relation 集合；
- \(Q_i^{(t)}\)：当前最高 map quality/directness；
- \(S_i^{(t)}\)：已使用的 provenance/source 集合；
- \(z_i^{(t)}\)：该 atom 是否达到 map-implied resolution 条件。

候选在当前 prefix 下被归入以下操作层级：

\[
\mathrm{DIRECT\text{-}OPEN}
\succ
\mathrm{PARTIAL\text{-}OPEN}
\succ
\mathrm{CONTRAST}
\succ
\mathrm{CORROBORATE}
\succ
\mathrm{BRIDGE}
\succ
\mathrm{FALLBACK}.
\]

- **DIRECT-OPEN**：以高直接性、高置信的 resolving relation 首次解析一个 unresolved atom。
- **PARTIAL-OPEN**：首次提供有效但较弱的 resolving alignment。
- **CONTRAST**：为已观察 atom 增加新的、相反或限定性的 relation。
- **CORROBORATE**：由新来源或独立 span 对既有 relation 提供支持。
- **BRIDGE**：提供必要背景，但不改变 hard resolution state。
- **FALLBACK**：没有有效结构增益时，使用 retrieval relevance 与成本进行兜底。

推荐让一个 evidence 同时更新其所有 valid aligned atoms；primary atom 只用于 trace 展示和 diagnostic anchor。当前实现主要显式更新 primary atom，因此这是一项必须先改实现、再写进论文的 P0 决策。

### 2.4 直接结构选择规则

为避免“删除学习后换成一组任意手工权重”，推荐采用词典序结构边际：

\[
\mathbf\Delta_t(u)=
\Big(
P(o_t(u)),
\Delta_{\mathrm{resolved}},
\Delta_{\mathrm{atom}},
\Delta_{\mathrm{relation}},
\Delta_{\mathrm{source}},
-\Delta_{\mathrm{redundancy}},
q_{\mathrm{map}},
s_{\mathrm{retrieval}},
-\ell(u),
\operatorname{stable\_key}(u)
\Big).
\]

第 \(t\) 步选择：

\[
u_t=
\operatorname*{arg\,max}^{\mathrm{lex}}_{
u\in\mathcal C\setminus\mathcal T_{<t},\,\ell(u)\le B_t
}
\mathbf\Delta_t(u).
\]

该定义需要满足：

- 操作类型先于 retrieval score；
- 新 atom 解析先于对已解析 atom 的重复支持；
- relation diversity 与 source diversity 分开；
- 最终 tie-break 使用稳定 evidence key，而不是输入 candidate index；
- 交换候选输入顺序不应改变输出链。

这样可以明确回答“为什么不再训练权重”：算法直接优化预先声明、可审计的结构偏序，而不是先用同一偏序产生 winner-vs-rest 偏好，再拟合一个线性近似。

### 2.5 排序、停止与 context guard 的层级边界

新稿必须把三个数量分开：

- \(K_{\mathrm{order}}\)：结构算法物化出的最大有序 trace 长度；
- \(K_{\mathrm{sel}}\)：sufficiency-aware budget controller 选择的 prefix 长度；
- \(K_{\mathrm{final}}\)：经过 verifier 最大上下文保护后真正可见的 evidence 数。

建议保留当前实现中“先物化 ordering、后选择 prefix”的工程方式，但在方法上将 prefix 规则定义为：

\[
K^*=\min\{t\ge k_{\min}:\rho_t^{\mathrm{map}}\ge\rho_{\mathrm{target}}\},
\quad K^*\le k_{\max},
\]

同时允许在预算耗尽、达到 \(k_{\max}\) 或不存在正结构增益时停止。\(B_{\mathrm{ctx}}\) 仅是 rendering guard，不得与 \(B_{\mathrm{sel}}\) 混写为一个“token budget”。

排序器与停止策略必须单独消融。否则无法区分收益来自 evidence order、可变 evidence 数，还是更大的平均上下文。

### 2.6 理论与算法性质计划

建议论文至少给出以下性质：

1. **确定性。** 固定 \((\mathcal A,\mathcal C,M)\) 与参数后，稳定 tie-break 保证输出唯一。
2. **标签与 verifier 独立性。** 选择规则不访问 verdict label、gold evidence、verifier logits 或 verifier-specific reward。
3. **状态有效性不变量。** invalid/background alignment 不会推进 atom resolution。
4. **复杂度。** 若每步遍历所有未选 evidence–atom pairs，时间复杂度为
   \[
   O(k_{\max}|\mathcal C||\mathcal A|).
   \]
5. **困难性与受限精确性。** Terminal ordinal utility \(U(S)=\sum_iw_i\max_{e_j\in S}q_{ij}\) 在 \(q_{ij}\in\{0,2\}\) 时包含 weighted budgeted maximum coverage 特例，因此一般规模为 NP-hard；在当前 \(m\le6\) 的 bounded atom universe 上，使用 \(O(nr3^mB)\) 的 pseudo-polynomial ordered-state DP 精确求解冻结目标。标准 greedy guarantee 只能针对匹配假设下的 terminal set objective，不能自动延伸到 acquisition-time secondary objective、联合 token 约束或 prompt-truncated slate。

理论部分必须经过独立 proof audit。若 full objective 包含无法满足单调次模条件的 contrast synergy 或 redundancy penalty，则只报告受限子问题性质和完整算法复杂度，不为全文算法声称近似保证。

---

## 3. Novelty 的主张层级与近邻边界

### 3.1 主张层级

建议按以下优先级组织全文：

1. **Primary claim：** BAREC 是一个区别于静态 relevance ranking 的预算化、状态化 evidence-chain construction 问题。
2. **Method claim：** SCSG 在不使用 selector supervision 的情况下，直接依据 typed atom-evidence state 构造有序链。
3. **Empirical claim：** 相同 structural policy 在 matched budget 下改善链质量或形成更好的性能—成本 Pareto，并能跨 verifier/数据域迁移。
4. **Application claim：** 该链可提升或保持 downstream fact-checking performance，同时保留 provenance。

Atomization、claim-aware chunking、Atom-Union 和 LLM evidence map 是 BAREC 的输入接口与实现基础，不再各自包装成同等强度的独立 novelty。

### 3.2 最危险的近邻与可守差异

定稿 Related Work 时至少逐篇核对以下近邻：

| 近邻方向 | 已有覆盖 | 本文不能单独声称的新意 | 需要守住的差异 |
|---|---|---|---|
| [ClaimDecomp](https://aclanthology.org/2022.emnlp-main.229/) / claim decomposition | 显式或隐式子问题分解 | claim atomization | 从 typed alignment 构造预算化、有序、状态化 evidence chain |
| [Complex Claim Verification](https://aclanthology.org/2024.naacl-long.196/) | decomposition、retrieval、summary、verdict pipeline | raw-evidence pipeline | 不生成替代原文的 summary，保留原始 span/provenance，并明确优化 chain state |
| [Decomposition Dilemmas](https://aclanthology.org/2025.naacl-long.320/) | decomposition 的收益与错误传播 | “decomposition 总是有效” | 系统评价 atom/map 噪声如何传播到链构造 |
| [CORRECT](https://aclanthology.org/2025.naacl-long.154/) / [HeterFC](https://ojs.aaai.org/index.php/AAAI/article/view/27760) | 图表示和学习式图推理 | evidence graph | 外部可审计、无 selector 训练的 ordered chain policy |
| [ProgramFC](https://aclanthology.org/2023.acl-long.386.pdf) | 生成并执行推理程序 | structured multi-step reasoning | 选择原始 evidence units，而非生成 QA/logic program |
| [GAVEL](https://aclanthology.org/2026.findings-acl.1789/) | atomic binding、provenance、sufficient evidence set | binding、provenance、sufficiency 本身 | 显式状态转移、预算化确定性选择、raw-report setting、无需 multi-agent debate 或 verifier supervision |

GAVEL 是当前最危险近邻。最终 novelty 不能停留在“atomic binding + sufficient evidence + provenance”，必须由状态化预算算法、无 selector supervision、原始证据选择和受控链质量评价共同区分。

### 3.3 Reviewer objection 与所需证据

| 可能质疑 | 必须给出的回答 |
|---|---|
| “这只是若干规则拼成的 pipeline。” | 正式定义 state/action/transition/objective/budget/tie-break/complexity；Algorithm 1 只写 solver，不写六阶段 pipeline。 |
| “resolved rate 是自己算的，不能证明充分。” | 将其改名 `map-implied resolution`；补 gold evidence 或 blind human sufficiency。 |
| “收益来自多看 evidence。” | 做 selector × stopping factorial，并匹配平均 \(K\) 与 tokens。 |
| “顺序不重要，LLM 只看同一集合。” | 原顺序、shuffle、reverse、retrieval-order 共用相同 evidence set，做 paired test。 |
| “无学习只是把参数藏在规则里。” | 词典序规则、稳定 tie-break、validation-only 参数冻结、跨域不调 selector、完整敏感性。 |
| “LLM map 昂贵且不可靠。” | map 人评、重复性、噪声注入、逐阶段 token/latency/API cost。 |
| “证据链不是 faithful explanation。” | 若无因果干预则降低主张；若要使用 faithful，补 top-step deletion、random deletion、unselected substitution。 |

---

## 4. 论文表述的系统改写计划

### 4.1 题目候选

优先候选：

1. **Budgeted Atom-Resolving Evidence Chains for Fact Verification**
2. **Training-Free State-Conditioned Evidence-Chain Construction for Fact Verification**
3. **From Static Evidence Ranking to Atom-Resolving Chain Construction**

题目中不建议同时堆叠 Atom-Union、MREC、LLM、training-free 和 verifier；题目应突出新问题或算法对象。

### 4.2 Abstract 重写模板

Abstract 应从“问题缺口”开始，而不是枚举 pipeline：

> Existing evidence-based fact verification systems commonly rank evidence units independently and pass a fixed top-\(k\) subset to a verifier. This formulation overlooks that the value of a candidate depends on which claim atoms have already been addressed and whether it opens, contrasts, or corroborates the current evidence state. We formalize Budgeted Atom-Resolving Evidence-Chain Construction, which seeks an ordered, provenance-preserving chain from noisy typed atom–evidence alignments under an evidence budget. We introduce a selector-training-free structural solver that updates atom states and greedily prioritizes resolving, contrasting, corroborating, and bridging operations, together with a sufficiency-aware prefix rule. Across [datasets/verifiers], the same frozen policy [result placeholders], while human/gold evaluation shows [chain-quality placeholders] at [cost placeholders].

注意：当前 36.66、66.12 和 SciFact 结果来自 learned-selector 流程。结构 selector 完整重建 trace、训练样本与 verifier 后，才能作为新 `Ours` 数值写入摘要。

### 4.3 Introduction 的五段逻辑

1. **任务背景。** Evidence-based fact verification 需要从原始报道中选择少量证据供 verifier 使用。
2. **核心缺口。** 主流 static top-\(k\) 将候选独立评分；但同一 evidence 在空 prefix 中可能用于 OPEN，在已有支持后只用于 CORROBORATE，在出现相反立场时又可能用于 CONTRAST，其价值天然依赖当前状态。
3. **新问题。** 定义 BAREC：在 noisy typed alignments 和预算下，构造有序且 provenance-preserving 的 chain。
4. **新方法。** SCSG 直接执行结构偏序并更新 atom state；无需 gold chain、verdict label、verifier feedback 或 selector training。
5. **证据与结论。** 用 intrinsic chain quality、end-to-end performance、matched-cost、跨 verifier/数据集、人评和鲁棒性共同验证。

“人类核查者通常如何操作”的叙述需补可靠文献，不应只凭直觉作为主要方法依据。

### 4.4 可直接使用的贡献表述

建议最终控制为三项或四项：

1. We formulate **Budgeted Atom-Resolving Evidence-Chain Construction**, a stateful evidence organization problem that constructs an ordered chain from typed atom–evidence alignments under an explicit evidence budget.
2. We propose **SCSG**, a selector-training-free structural solver that models resolving, contrasting, corroborating, and bridging operations, preserves source provenance, and deterministically selects evidence according to prefix-dependent structural gain.
3. We introduce an evaluation protocol that separates intrinsic chain quality from downstream verdict accuracy, including gold/human sufficiency, matched-cost ordering controls, provenance, redundancy, and robustness to map noise.
4. We show that a single frozen structural policy transfers across [verified datasets/verifiers] without selector retraining or verifier-derived supervision.

第 3、4 项只有在实验完成后才能写成完成时；在内部大纲中应保留 `[TO VERIFY]` 标记。

### 4.5 Method 章节的新结构

```text
3 Method
  3.1 System Interface and Overview
      atomization / chunking / Atom-Union 仅作为输入构造的压缩说明
  3.2 Typed Atom-Evidence Map
      relation / directness / confidence / provenance
      resolving vs context vs invalid predicate
  3.3 Budgeted Atom-Resolving Evidence-Chain Construction
      3.3.1 Problem Definition
      3.3.2 Atom States and Typed Operations
      3.3.3 Structural Marginal Objective
      3.3.4 State-Conditioned Structural Greedy Solver
      3.3.5 Sufficiency-Aware Prefix and Complexity
  3.4 Verifier Integration
      evidence rendering / label-token prediction / context guard
```

Algorithm 1 只接受 \((\mathcal A,\mathcal C,M,\ell,B)\) 并描述 BAREC solver。Atomization、retrieval、map generation 放 Figure 1 与 Appendix，不再占 Algorithm 1 的前六步。

### 4.6 v0.4.1 逐节迁移表

| v0.4.1 部分 | v0.5 动作 | 说明 |
|---|---|---|
| Abstract | 全部重写 | 删除 learned selector；旧数值暂不继承。 |
| Introduction | 保留 provenance 动机，重建 gap | 从“复杂 pipeline”改为“static ranking 不建模 prefix-dependent value 与 sufficiency budget”。 |
| Contributions | 全部重写 | BAREC problem、SCSG algorithm、chain evaluation、transfer。 |
| Related Work | 扩充并按差异组织 | decomposition、evidence selection/graph、structured/budgeted construction、provenance/sufficiency。 |
| Overview | 压缩 pipeline | 上游只说明接口，核心篇幅从 typed map 开始。 |
| Task Definition | 拆成两层 | downstream classification task 与 BAREC problem 分开。 |
| Atomization | 压缩 | 定义和约束留主文，prompt/schema 入附录。 |
| Claim-aware Chunking | 大幅压缩 | 作为 evidence-unit construction，不再是核心 novelty。 |
| Atom-Union | 保留摘要版 | 作为 recall-supporting candidate construction；单独消融。 |
| Atom-Evidence Map | 保留并收紧 | 新增 strict resolving predicate 与 provenance 字段；内部 resolution 改名。 |
| Greedy Chain Construction | 重写为问题/状态 | 删除 \(U_{\theta_{sel}}\)，引入 action、transition、budget。 |
| Learned Marginal Selector | 整节删除替换 | 换成 Structural Objective、SCSG、Stopping、Properties。 |
| Proxy Pairwise Learning | 从主方法移除 | 旧 learned selector 仅保留为 experimental baseline/appendix。 |
| Algorithm | 全部替换 | 只写 BAREC solver，不写完整 pipeline。 |
| Verifier | 保留并澄清边界 | \(K_{order}\)、\(K_{sel}\)、\(K_{final}\) 分开。 |
| Training and Inference | 改名 | 仅 verifier 训练；selector 无 checkpoint、pair generation 或 optimizer。 |
| Main Results | 保留 protocol，结果重跑 | 旧 learned 结果降为 historical baseline。 |
| Selector Ablation | 全部重构 | static/stateful × fixed/adaptive × order controls。 |
| Map Ablation | 全部重跑 | 直接移除结构字段，不再为每个变体重训 selector 权重。 |
| Prompt Policy | 保留研究问题，单独归因 | 停止/预算与 ordering 机制分开。 |
| Reliability | 扩展为 atom/map/chain 三层 | chain sufficiency 必须有外部 gold 或盲审人工依据。 |
| Appendix C/D | 全部替换 | learned features/symbols 改成 structural predicates/actions/objective/symbol mapping。 |

### 4.7 全文术语替换与禁用主张

| 旧术语 | 新术语 |
|---|---|
| learned marginal selector | state-conditioned structural solver |
| heuristic proxy preference | structural priority / structural objective |
| selector weights | fixed structural rule and thresholds |
| selector training | 不存在；仅描述 policy configuration |
| resolved rate | map-implied resolution rate |
| minimum resolving chain | budgeted atom-resolving chain |
| selector-driven minmax gain | sufficiency-aware prefix effect |
| faithful explanation | provenance-preserving chain，除非完成因果干预 |

全文禁止在无证据时使用：`significantly improves`、`optimal`、`minimal`、`oracle-free`、`training-free system`、`faithful explanation` 和 `SciFact SOTA`。

---

## 5. 实验逻辑重建

### 5.1 实验研究问题

- **RQ1：** 在相同 evidence budget 下，state-conditioned structural construction 是否优于 static ranking？
- **RQ2：** 改善来自 ordering、state update，还是 adaptive stopping？
- **RQ3：** BAREC 是否改善外部验证的 evidence coverage、sufficiency、redundancy 与 provenance，而非只提高内部 map-implied resolution？
- **RQ4：** 相同且不重训的 structural policy 是否跨 verifier 与数据域保持收益？
- **RQ5：** relation、directness、confidence、contrast、corroboration 和 provenance novelty 各自贡献什么？
- **RQ6：** atom/map 噪声、distractor、duplicate 和冲突证据如何影响链与最终标签？
- **RQ7：** 去除 selector training 后，性能—成本—可迁移性之间的实际权衡是什么？

### 5.2 P0：Selector × stopping 严格 factorial

当前 fixed-top5 结果为：source-score 0.354、map-quality greedy 0.349、learned selector 0.339；learned 的 0.367 则来自 minmax(5,10)。因此现有结果不能推出“learning 有效”，也不能直接推出“structure-only 已经更优”，因为 ordering 与 stopping 被混在了一起。

此外，当前 selector-mechanism 表并非严格同 prompt 口径：部分 static baseline 使用 `plain + prefix_topk`，而 map/learned 条件使用 `mrec_min + fixed_topk`。正式对照必须统一成 `mrec_min` rendering、相同 evidence capacity、相同 verifier recipe，并分别训练对应 verifier；现有数字只能作为路线诊断。

最小必跑矩阵为：

| Selector | fixed-5 | minmax(5,10) | matched-token budget |
|---|---:|---:|---:|
| Atom-Union source score | 必须 | 必须 | 建议 |
| map-quality static score | 必须 | 必须 | 建议 |
| atom-coverage-only greedy | 必须 | 必须 | 建议 |
| state-free structural score | 必须 | 必须 | 建议 |
| **SCSG stateful structural** | **必须** | **必须** | **必须** |
| old learned marginal | 必须 | 必须 | 建议 |

另外对 SCSG 生成的同一 evidence set 做：

- 原顺序；
- shuffle；
- reverse；
- retrieval-score reorder。

必须控制候选池、map cache、最大 evidence 数、平均 \(K\) 与平均 prompt tokens。当前仅有 shuffle 约 1 F1 的差异且无统计检验，不足以支撑 ordered-chain claim。

### 5.3 两层公平评价协议

同时采用两种协议：

1. **Intrinsic selector evaluation。** 固定 candidate pool 与 map，直接比较 gold/human chain metrics，不依赖 verifier；这是验证新问题的主证据。
2. **End-to-end evaluation。** 每个主 evidence policy 使用相同 verifier backbone、训练预算与超参重新生成训练输入并训练 verifier；这是完整系统效果。

可先使用 frozen verifier 做低成本筛选，但这只能标注为 diagnostic，因为 verifier 可能已适应旧 selector 的输入分布。publication-grade 结果不能只用主方法 checkpoint 测所有变体。

### 5.4 P0：独立于内部 map 的 chain-quality 评价

这是本次转向后最关键的新实验。`map-implied resolution` 同时由 map 驱动并由 map 自己评估，存在自证循环，最多作为内部诊断。

#### 数据集与评价层级

- **SciFact：** 报告 candidate-pool gold recall、conditional selector recall、sentence evidence P/R/F1、abstract evidence/joint 指标；把 retrieval failure、chunk selection failure 与 chunk-to-sentence projection failure拆开。
- **HoVer：** 优先在 gold-document/gold-sentence candidate windows 上做受控 chain-construction 评价，先隔离 selector，再把 open-domain retrieval 作为后续阶段。
- **LIAR-RAW/RAWFC：** 因缺少 gold chain，构建只用于评价、不用于调参或训练的 blind human `Atom-Chain Audit Set`。

#### 核心指标

- candidate-pool gold recall；
- conditional evidence recall：仅在 gold evidence 已进入候选池的样本上衡量 selector；
- evidence precision / recall / F1；
- joint label + evidence；
- human-confirmed per-atom resolution；
- claim-level sufficiency；
- redundancy / duplicate rate；
- provenance validity；
- prefix coverage AUC：
  \[
  \operatorname{AUC}_{\mathrm{prefix}}
  =\frac{1}{K}\sum_{t=1}^{K}\operatorname{GoldOrHumanCoverage}(\mathcal T_{1:t});
  \]
- first relevant evidence rank / first complete prefix length。

#### SciFact 风险门槛

当前 SciFact sentence Selection-only 为 40.51，而 abstract Label-only 为 72.41。论文中心改成 evidence-chain construction 后，前者不再只是次要局限，而是 P0 soundness 风险。

必须二选一：

1. 实质提升 exact sentence localization，并将改进后的官方 evidence metrics 纳入主文；
2. 将主张明确限定为 **chunk/document-level raw-report evidence organization**，同时用 human chunk-level sufficiency 与 provenance 评价支撑，不能再把 SciFact 作为细粒度 evidence-selection 成功案例。

### 5.5 主结果与迁移实验

主结果应分为两张表，而不是只报 verdict F1：

1. **Chain construction table：** gold/human evidence quality、prefix AUC、redundancy、平均 \(K\)、tokens。
2. **Downstream verification table：** LIAR-RAW、RAWFC、SciFact/HoVer 的 label 与 joint metrics。

验证 `verifier-agnostic` 的最低要求：

- 至少两个 verifier family，例如 Ministral 与 Llama；
- 对同一数据样本使用完全相同的 SCSG trace；
- 不因 verifier 变化重新调 selector；
- 比较 SCSG 相对 static top-\(k\) 的增益是否保持。

跨域实验应明确记录：

```text
structural thresholds selected on LIAR-RAW validation
→ freeze selector policy
→ RAWFC / SciFact / HoVer
```

目标域若重新调 stopping threshold，只能称 domain adaptation，不能称 frozen transfer。

### 5.6 结构算法消融

在相同候选池与预算下逐项移除：

- no state update：每步使用初始 state；
- no direct/partial distinction；
- no CONTRAST；
- no CORROBORATE；
- no BRIDGE；
- no source novelty；
- no duplicate/redundancy control；
- no multi-atom update；
- no adaptive prefix；
- retrieval-only fallback。

同时报告各 operation 的触发频率。如果某一 operation 极少触发，就不能仅凭概念设计将其写成核心贡献。

### 5.7 Evidence map 消融重做

所有变体使用同一固定 structural policy，不再“在退化特征上重新训练权重”：

- full map；
- no relation；
- no directness；
- no confidence；
- no map；
- oracle/human map upper bound，仅用于诊断，不进入主方法。

每个变体分别报告：

1. fixed-\(K\) 下的排序质量，隔离 map 对 evidence choice 的作用；
2. adaptive prefix 下的性能—token Pareto，隔离 map 对 sufficiency judgement 的作用。

不得继续使用 Macro-F1/平均 \(K\) 作为主要“证据效率”指标；改用 matched-cost 比较、Pareto frontier 或 budget-curve AUC。

### 5.8 Candidate construction 与上游消融

Atom-Union 继续作为 supporting component，使用既有受控方案重做：

- claim-level baseline only；
- atom-route only；
- union without MMR；
- full union；
- candidate pool size sensitivity。

所有变体都要区分 candidate recall 与 conditional selector quality。否则上游没有召回 gold evidence 时，不能把失败归因给 SCSG。

### 5.9 人工评价与可靠性

建议建立三层评价：

1. **Atom 层：** faithfulness、completeness、atomicity。
2. **Map 层：** relation macro-F1/confusion matrix、directness weighted-\(\kappa\) 或 Spearman、confidence calibration。
3. **Chain 层：** per-atom evidence correctness、claim-level sufficiency、redundancy、order usefulness、provenance validity。

推荐样本：

- 200 条随机 claim 作为 headline sample；
- 100 条 hard cases，覆盖多 atom、冲突、early-stop 与方法分歧；
- 500–800 个 evidence–atom pairs，按 relation 分层采样，汇总时按总体分布重加权；
- 2 位独立标注者、分歧 adjudication，第三位标注者复核子集。

Chain 层采用 blind matched-cost pairwise comparison：

```text
SCSG chain
vs.
best static baseline chain
```

左右顺序随机，不展示方法名与系统 verdict。充分 chain 通常不唯一，不建议使用 exact sequence match；应报告 win/tie/loss、set sufficiency 与 order preference。

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

若论文要称 evidence chain 为 faithful verifier explanation，补充：

- 删除 top-priority step；
- 删除随机 step；
- reverse/shuffle；
- 仅保留未选 evidence；
- 用未选 evidence 替换已选 evidence；
- 比较 label flip、gold-label probability 或 decision margin。

若这些干预不做或结果不支持，则全文只称链为 structured/provenance-preserving input，不称 faithful explanation。

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

- structural rule 与阈值只在 validation 上确定；test 只在冻结后评估。
- 核心 end-to-end 方法至少 3 个 verifier seeds，报告 mean ± std。
- sample-level paired bootstrap 计算 Macro-F1 差值与 95% CI。
- 多个消融比较使用 Holm correction。
- 人评 win/tie/loss 使用 bootstrap CI 或配对检验。
- 报告每个结果对应的 exact config、candidate/map cache hash、trace artifact、verifier checkpoint 和 metric file。
- frozen cache 用于主结果；API 重复调用只用于稳定性子实验。

---

## 6. 主文表格与图的展示规划

### 主文优先级

1. **Figure 1：Problem and solver。** 显示输入 typed map、当前 atom state、候选 operation、预算与输出 chain；不再画成普通六阶段流水线。
2. **Figure 2：State timeline case study。** 一条 claim 展示 DIRECT-OPEN → CONTRAST/CORROBORATE → prefix stop，并保留 source/span provenance。
3. **Table 1：问题定位。** 与 decomposition、raw-evidence pipeline、graph reasoning、program reasoning、GAVEL 的监督、状态、预算、输出结构和 provenance 对照。
4. **Table 2：Intrinsic chain quality。** gold/human sufficiency、evidence recall、prefix AUC、redundancy、平均 tokens。
5. **Table 3：End-to-end results。** LIAR-RAW、RAWFC 与 gold-evidence dataset 的 label/joint metrics。
6. **Table 4：Selector × stopping factorial。** 核心归因表。
7. **Figure 3：Performance–cost Pareto。** Macro-F1 与 chain sufficiency 分别对 prompt tokens 作图。
8. **Table 5：Atom/map/chain reliability。** 人评与校准。
9. **Table 6：Cross-domain × cross-verifier transfer。** 明确 selector 是否重调。

### Appendix

- 完整 algorithm/objective 与 proof；
- prompts、schemas、thresholds、tie-break；
- relation confusion matrix 与 confidence reliability diagram；
- 完整 map/operation/candidate-pool ablations；
- noise/distractor curves；
- stage-wise cost；
- 3–5 个成功、失败和分歧 case；
- artifact/config/hash 清单。

---

## 7. 当前实现到新方法的工程迁移计划

### 7.1 新建独立 policy，不复用旧 proxy 入口

仓库已有直接 transition selector 的起点，但不能把旧 `rank_candidates_by_proxy` 简单改名，因为旧路径仍可能读取 `oracle_ordered_keys`，并且旧 tie-break/状态更新未必满足新定义。

建议新建明确的：

```text
selection_policy: structural_state_greedy
selector_name: barec_scsg_v0_1
```

旧 `learned_marginal_proxy` 只保留为实验 baseline，训练脚本和 weight file 不再是新主流程依赖。

### 7.2 P0 实现不变量与测试

1. **Oracle poison test：** 同一样本写入任意 `oracle_ordered_keys` 前后，SCSG 输出必须完全相同。
2. **Label/verifier independence test：** 修改 gold label、verifier score 或 checkpoint 路径不改变 trace。
3. **Invalid alignment test：** irrelevant/background/context 不推进 resolution。
4. **Multi-atom update test：** 一条 evidence 的所有 valid alignments 被更新，primary atom 仅影响展示。
5. **Permutation invariance test：** 候选输入顺序打乱，稳定 key 下输出不变。
6. **Duplicate test：** 同 source/span duplicate 不触发新 corroboration。
7. **Budget test：** \(K\)、token cost 与 \(B_{sel}\) 始终满足约束。
8. **Layer-boundary test：** \(K_{order}\)、\(K_{sel}\)、\(K_{final}\) 分别写入 artifact。
9. **State replay test：** 从 trace 逐步回放可复现最终 state 与 operation。

旧 artifact 即便仍包含 `oracle_ordered_keys` 也不能靠 wrapper “不传入”来保证结构纯净；新 policy 应在代码层根本不读取该字段，并通过 poison test 证明。

### 7.3 产物契约

每条 trace 至少记录：

- candidate stable key 与 provenance；
- all valid aligned atoms；
- primary display atom；
- operation；
- state before/after；
- 每个词典序分量；
- map-implied resolution；
- cumulative token/evidence cost；
- stop reason；
- \(K_{order}\)、\(K_{sel}\)、\(K_{final}\)；
- selector config/hash；
- 明确的 `uses_oracle=false`、`uses_verifier=false` audit 字段。

### 7.4 旧结果的处理

- v0.4.1 的 learned-selector 结果保留为 historical/diagnostic baseline。
- 不把旧 36.66、66.12、72.41 等数值直接改名为 SCSG。
- 新主结果必须从 structural traces 开始，重建 train/val/test verifier rows。
- frozen-verifier quick test 可用于路线筛选，但不进入最终主表。
- 旧 map cache 可以复用，前提是新 selector 完全忽略 oracle 字段，并记录 cache schema/hash。

### 7.5 当前可复用资产与真实缺口

当前已有的 evidence-map feature 规模为：

| 数据集 | Train | Validation | Test/official dev | 可复用方式 |
|---|---:|---:|---:|---|
| LIAR-RAW | 10,065 | 1,274 | 1,251 | 无需新增 API，可直接生成 SCSG traces |
| RAWFC baseline20 | 1,612 | 200 | 200 | 无需新增 API，可直接生成 SCSG traces |
| SciFact | 809 | 300 | 300 | 可重建 fixed-9/direct traces 并使用 official dev scorer |

仓库已有四类可借用实现，但没有任何一类可直接冒充新主方法：

1. `src/fact_checking/selectors/minimal_resolving_chain.py` 的 `transition_v0_1` 有状态机，但缺少拟议的完整结构边际目标。
2. `src/fact_checking/selectors/mrec_learned_marginal.py` 的 proxy ranking 最接近旧结构规则，但包含 oracle teacher 分支，只能作为反例和迁移参考。
3. `src/fact_checking/selectors/evidence_chain_graph.py` 已有 coverage、pair utility、redundancy、source 和 length 等 budgeted objective，但使用旧 graph schema，没有 MREC 的显式 atom state。
4. `src/fact_checking/selectors/map_selector_ablation.py` 已有 weighted set cover/minimal group，可作为结构 baseline。

现有 `scripts/sentence_trace_method/check_mrec_diagnostics.py` 可扩展复用 transition legality、duplicate、fallback、resolved rate、token cost 与 prompt leak 检查；`paired_significance.py` 可复用做 paired significance。需要新增 valid coverage、source diversity、redundancy、prefix AUC 与 oracle-invariance 汇总。

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
- operation usage + removal；
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

1. **方法非劣。** SCSG 在 matched budget 下相对最佳 static baseline 至少处于 Pareto frontier，并在外部 chain-quality 主指标上具有稳定优势。
2. **外部有效。** 存在 gold evidence 或 blind human evidence 评价；不能只依赖 verdict F1 和 map-implied resolution。
3. **迁移成立。** 同一个未重训、未重调的 structural policy 至少跨两个数据域或两个 verifier family 保持主要收益。
4. **主张与 gold-evidence 结果一致。** SciFact/HoVer 等结果不再与“高质量 evidence-chain construction”主张明显冲突。

若第 2、4 项无法完成，论文应收缩为：

> training-free chunk/document-level raw-report evidence organization

并将其定位为经验型系统研究，而不应继续声称通用、细粒度的 evidence-chain construction 方法贡献。

---

## 11. 推荐的最终论文骨架

```text
1 Introduction
  Static ranking gap
  BAREC problem
  SCSG solution
  Contributions

2 Related Work
  Claim decomposition
  Evidence selection and graph reasoning
  Structured/budgeted evidence construction
  Provenance and sufficiency evaluation

3 Method
  3.1 System Interface
  3.2 Typed Atom-Evidence Map
  3.3 BAREC Problem Definition
  3.4 Atom States and Structural Operations
  3.5 SCSG Solver and Sufficiency-Aware Prefix
  3.6 Properties, Complexity, and Verifier Integration

4 Experiments
  4.1 Tasks, Baselines, and Fair Protocol
  4.2 Intrinsic Chain Construction Quality
  4.3 Downstream Fact Verification
  4.4 Selector × Stopping Attribution
  4.5 Cross-Domain and Cross-Verifier Transfer
  4.6 Reliability, Robustness, and Cost

5 Analysis
  Human chain preference
  Operation usage
  Evidence intervention
  Failure cases

6 Limitations and Ethics
  LLM map errors and cost
  Source credibility not modeled
  Chunk-level vs sentence-level boundary
  API/cache reproducibility

7 Conclusion
```

该骨架的关键变化是：论文首先提出 BAREC，再解释 SCSG 如何求解，最后把完整 pipeline 作为应用该算法的实验载体。这样即使 selector 不学习，论文仍然以一个可定义、可求解、可独立评价的新问题为中心，而不是退化为组件堆叠。
