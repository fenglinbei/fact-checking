# AAAI 论文初稿 v0.5.0

本稿固定 Budgeted Atom-Coverage Evidence Sequencing（BACES）的问题定义、exact ordered-state solver 与贡献表述。实验部分暂时保留 v0.4.1 的表格结构和历史 learned-selector 结果；这些数值仅作为对照占位，BACES 主结果以重新物化的 traces、verifier inputs 与 validation artifacts 为准。

# Abstract

基于证据的事实核查需要从大规模候选池中选择紧凑、可追溯的证据，并将其提交给下游 verifier。证据内容与证据呈现具有不同的数学结构：对完整读取且 permutation-invariant 的理想判别者，固定证据集合的终态覆盖能力不随排列改变；现实判别流程面临有限的阅读或上下文预算，LLM verifier 还表现出截断和位置敏感性，同一集合的不同排列因而可能具有不同的可利用性。本文据此提出 Budgeted Atom-Coverage Evidence Sequencing（BACES）。给定 claim atoms、平铺候选证据池和 typed atom--evidence alignments，BACES 首先在显式 evidence-count 与 token budgets 下最大化证据集合的 terminal graded atom coverage，随后在所有 terminal-optimal slates 中最小化 coverage-unit acquisition time，使有效覆盖更早进入 verifier-visible prefix。该顺序承担 verifier-facing presentation 功能，不引入多跳推理、证据依赖或冲突消解语义。针对 claim decomposition 形成的 bounded atom universe，本文给出一个精确、确定且可审计的 ordered-state dynamic program；求解过程不使用 selector supervision、verdict labels 或 verifier-derived rewards。实验评价分别覆盖 terminal evidence-set quality、strict same-set presentation effects 与 downstream verification performance，并在匹配的 candidate pools 和 budgets 下进行归因。

# Introduction

近年来，自动事实核查逐渐从 claim-only classification 扩展到 evidence-based verification。此类系统接收待核查声明及其相关报道，经过检索、筛选和组织后，将有限证据提交给分类器或生成式 verifier。现有方法采用级联证据特征 [CofCED]、LLM-generated defenses [L-defense]、辩护图 [G-defense] 或多代理核查流程 [DelphiAgent]。复杂的证据变换提高了系统能力，同时增加了证据来源、选择依据和最终判定之间的追踪难度。

在 evidential-content 层面，本文将证据建模为无序集合。设一个理想 verifier 能够完整读取输入，并对 evidence permutation 保持不变，则固定集合提供的终态 atom coverage 也应保持不变。集合表示承载 map-defined terminal structural coverage；该结构 surrogate 与人工事实充分性的对应关系由外部评价检验。逐步构造过程只是一种求解方式，不自动产生逻辑先后、多跳组合或 evidence-to-evidence dependency。

实际判别流程具有有限的阅读或上下文预算。对于 LLM verifier，prompt truncation 可能使后部证据不可见，长上下文位置敏感性也会影响不同 evidence units 的实际利用程度 [Liu et al., 2024, *Lost in the Middle*]。本文不假设人类与 LLM 共享同一种认知机制，仅使用有限消费预算这一共同接口条件。在这一条件下，证据顺序影响信息进入前缀的时间。本文将该现象建模为 verifier-facing presentation：集合目标衡量终态证据内容，顺序目标衡量有效覆盖在 slate 中的可访问时机。

上述区分形成两个相互关联的优化层级。第一层在 evidence-count 与 token budgets 下寻找 terminal graded atom coverage 最优的证据集合族。第二层在所有 terminal-optimal slates 中优化 coverage-unit acquisition time。Terminal objective 保证终态证据质量优先，presentation objective 在终态质量相同的解之间组织前缀。该层级结构也为 selection gain 与 order-only gain 提供了独立的实验归因接口。

本文提出 Budgeted Atom-Coverage Evidence Sequencing（BACES）。方法首先将 claim 分解为少量 atomic propositions，并通过 claim-level 与 atom-level retrieval 构建 provenance-preserving candidate pool；随后将 typed atom--evidence map 投影为 invalid/partial/direct 三态结构覆盖；最后使用 exact ordered-state dynamic programming 联合求解 terminal coverage 和 acquisition-time secondary objective。求解器直接消费冻结的结构目标，无需学习 selector weights，也不读取 gold evidence、verdict labels 或 verifier feedback。

本文的主要贡献如下：

1. 本文形式化 **Budgeted Atom-Coverage Evidence Sequencing**：以 permutation-invariant 的 terminal graded atom coverage 作为 primary set objective，以 coverage-unit acquisition time 作为 secondary presentation objective，并显式纳入 evidence-count 与 token budgets。
2. 本文针对 claim decomposition 诱导的 bounded ordinal state space 提出 exact ordered-state dynamic programming，在冻结的 map-defined lexicographic objective 上获得全局最优解和确定性 stable-key tie-break，同时保持 selector-training-free 与 verifier-agnostic。
3. 本文建立分层评价协议，将 terminal evidence-set quality、strict same-set presentation effects 与 downstream verdict performance 分开测量，并结合 exact evaluator、regret decomposition、matched-budget factorial 和 trace replay 支持方法归因。

# Related Work

少量早期 BERT 之前的 AFC 方法。

部分重要的 BERT 类方法。

以重要的篇幅讨论使用 LLM 的各种方法：基于辩护的、基于智能体的、基于内在表示的、基于图的等，并围绕这一系列方法复杂度高、证据 provenance 不易追踪的问题展开。

本节当前保留类别级综述骨架。完整版本覆盖各类代表工作，并围绕监督来源、集合目标、顺序语义、预算、provenance 与 verifier coupling 展开对照。

# Methodology

## Overview

本文提出一种以 BACES 为核心的 atom-aware fact-checking 方法。给定声明 $c$ 及其相关 reports，完整流程包含六个阶段：

1. **Claim atomization and query rendering.** 首先将声明分解为少量可独立验证的 claim atoms $\mathcal A=\{a_i\}_{i=1}^{m}$，并为每个 atom 生成 query rendering $q_i$。
2. **Claim-aware chunking.** 将相关 reports 切分为粒度适中的 evidence chunks $\mathcal U=\{u_j\}$。
3. **Atom-Union candidate pool construction.** 使用整条 claim 与各 atom queries 分别检索 chunks，两路结果融合、去重并经 MMR 得到候选池 $\mathcal C$。
4. **Typed atom--evidence map construction.** 对每个 candidate--atom pair 构建 relation、directness、confidence 与 key-span 标注，并通过固定 valid gate 投影为 $q_{ij}\in\{0,1,2\}$。
5. **BACES optimization.** 在 count/token budgets 下先最大化 terminal graded atom coverage，再在 terminal-optimal solutions 中最小化 coverage-unit acquisition time，得到 positive-gain coverage core 与有序 evidence slate。
6. **Verifier rendering and prediction.** Rendering layer 在 coverage core 后追加可选的 zero-gain fill，执行 tokenizer-aware context guard，并将 verifier-visible slate 交给指令微调的 LLM verifier。

BACES 的 primary objective 定义证据集合的终态质量，secondary objective 定义有限判别者下的呈现质量。Ordered slate 只携带 verifier-facing presentation semantics。多跳组合、证据逻辑依赖、立场冲突消解和 source credibility 均位于本文范围之外。

Claim atomization、chunking、retrieval 与 Atom-Union 提供 BACES 的输入接口。核心方法从 typed ordinal incidence matrix 开始，并以 exact ordered-state dynamic programming 结束。求解器不包含可训练 selector parameters；verifier training 构成独立的下游阶段。

## Task Definition

**Downstream task.** 给定一条待核查声明 $c$ 及其相关原始报道 $\mathcal R$，完整系统输出事实核查标签 $\hat y\in\mathcal Y$。标签集合 $\mathcal Y$ 随数据集而异：LIAR-RAW 使用六类细粒度真值，RAWFC 使用三类真值，SciFact 使用 SUPPORT / CONTRADICT / NOINFO。BACES 位于 label prediction 之前，其优化目标与具体标签体系解耦。

**Claim atoms.** 本文将原始 claim $c$ 分解为少量可独立验证的 atomic propositions：
$$
A(c)=\{(a_1,q_1),\dots,(a_m,q_m)\},\quad 1\le m\le 6.
$$
其中 $a_i$ 表示第 $i$ 个原子命题，$q_i$ 是对应的检索查询。Claim atoms 定义后续 ordinal coverage state 的有限坐标空间。

**Evidence units.** 每篇 report 被切分为 evidence chunks，记为 $\mathcal U=\{u_j\}$。每个 evidence unit 保留原始 report id、句子跨度、文本内容、retrieval route 信息与 provenance metadata。

**BACES interface.** Canonical candidate pool 记为 $\mathcal C=\{e_j\}_{j=1}^{n}$。每条 candidate 具有 additive cost $c_j$ 和 ordinal atom-coverage vector $\mathbf q_j=(q_{1j},\ldots,q_{mj})\in\{0,1,2\}^{m}$。一个 BACES solution 同时具有集合表示与序列表示：
$$
S_\pi=\{e_{\pi_1},\ldots,e_{\pi_L}\},
\qquad
\pi=(e_{\pi_1},\ldots,e_{\pi_L}).
$$
$S_\pi$ 用于定义 terminal structural coverage，$\pi$ 用于定义 prefix presentation quality。序列中的 evidence identities 两两互异，且可行解满足：
$$
L\le K_{\max},
\qquad
\sum_{e_j\in S_\pi}c_j\le B.
$$

**Prediction.** Ordered evidence slate 经 rendering 与 context guard 后得到 verifier-visible sequence $\pi_{\mathrm{final}}$。它与声明拼接为：
$$
x=\mathrm{Render}(c,\pi_{\mathrm{final}}),
$$
并送入指令微调的 LLM verifier：
$$
\hat y=\arg\max_y p_{\theta_{\mathrm{ver}}}(y\mid x).
$$

## Claim Atomization and Query Rendering

人工声明验证通常先识别 claim 中可独立验证的事实单元。本文将原始 claim $c$ 转换为少量语义完整、可独立验证的 atomic propositions，并将其用于 atom-conditioned retrieval、evidence map 与 ordinal coverage state construction。

输入是一条 claim，输出为：
$$
A(c)=\{(a_1,q_1),\dots,(a_m,q_m)\},\quad 1\le m\le 6.
$$
其中 $a_i$ 是原子命题，$q_i$ 是面向检索的 query rendering。实现上，atomization 通过 LLM API 完成，并在 prompt 中加入以下约束：只使用 claim 本身，不引入外部知识；只在 claim 含有多个可分别验证的事实断言时拆分；日期、数量、否定、比较对象、地点、范围、归因等必须保留在 proposition 内；每个 atom 不能只是关键词片段，而应是一个最小、可验证、语义完整的命题。若 LLM 输出超过最大 atom 数 $m_{\max}=6$，系统按 prompt 返回的重要性与可验证性字段进行合并或截断；完整 prompt、JSON schema、invalid-output retry 规则与后处理细节见 Appendix B。

每个 atom 同时附带 query rendering $q_i$。$q_i$ 是面向 retrieval 的短查询，其措辞可与 $a_i$ 不同，用于提高对细粒度事实的召回。实现上，query rendering 与 atomization 在同一 LLM API 调用中生成，附带关键词、实体、时间、数量与比较对象等字段。本文后续所有训练、验证与测试样本均使用同一 atomization 与 query rendering 流程，避免训练/推理流程不一致。

Claim atoms 决定 BACES state space 的坐标与规模。实验部分通过人工可信度评估检查 LLM-generated atoms 的 claim faithfulness、可验证断言覆盖和 atomicity（见 Reliability Study）。

## Claim-aware Evidence Chunking

在构建候选证据池之前，需要确定 evidence unit 的粒度。若粒度过小（如单句），单个 evidence unit 可能缺少必要上下文；若粒度过大（如整段 report），则会引入过多噪声并增加 verifier 上下文成本。因此本文采用 claim-aware chunking，将 reports 切分为与 claim 相关且局部连贯的 evidence chunks。

设 claim 为 $c$，一篇 report 包含句子 $s_1,\dots,s_n$，句向量为 $e_i$，claim 向量为 $e_c$。首先对每个句子计算 claim-aware relevance：
$$
r_i = \alpha_1\,\mathrm{norm}(\cos(e_i,e_c)) + \alpha_2\,\mathrm{norm}(\mathrm{LexF1}(c,s_i)) + \alpha_3\,\mathrm{norm}(\mathrm{BM25}(c,s_i)).
$$
随后对相邻句子边界 $i|i+1$ 打分：
$$
b_i = w_{\mathrm{sem}}(1-\cos(e_i,e_{i+1})) + w_{\mathrm{rel}}|r_i-r_{i+1}| - d_{\mathrm{coref}}.
$$
其中 $d_{\mathrm{coref}}$ 为指代惩罚项，用于降低错误切开跨句指代的风险。边界分数越高，该位置越可能成为 chunk boundary。

在确定切分点时，本文采用局部峰值法：仅保留同时满足“分数为局部峰值”且“超过动态阈值 $\mathrm{mean}(b)+\lambda\cdot\mathrm{std}(b)$”的边界点。为保证 chunk 粒度均匀，切分后还进行两步后处理：对超过最大长度的 chunk 沿其内部最高边界分数递归切开，对过短的 chunk 依据邻接语义与相关性相似度向相邻 chunk 合并。最终每个 chunk 记录其原文跨度：
$$
u_j=[s_a,\dots,s_b].
$$
主文只保留上述精炼描述；具体 $\alpha$ 权重、$\lambda$、最大句数、最小句数、chunk overlap 与数据集特化配置见 Appendix A。关于 chunk 粒度对最终性能与证据可读性的影响，将在实验消融中报告。

## Atom-Union Candidate Pool Construction

每个 claim 经 chunking 后通常得到数十个 evidence chunks。为了降低后续 evidence map 与 BACES optimization 的计算成本，同时保证对细粒度 atoms 的召回，本文构造一个 Atom-Union candidate pool。该过程发生在 atomization 之后，因此 candidate pool 同时利用整条 claim 与各 atom queries。

给定 chunks $\mathcal U=\{u_j\}$，系统首先使用整条 claim $c$ 进行 claim-level retrieval，得到 baseline pool：
$$
\mathcal B=\mathrm{TopK}_{k_b}\{u_j:s(c,u_j)\}.
$$
同时，对每个 atom $a_i$，使用其 query rendering $q_i$ 检索 evidence chunks：
$$
s(q_i,u_j)=\beta_1\,\mathrm{norm}(\cos(e_{q_i},e_{u_j}))
+\beta_2\,\mathrm{norm}(\mathrm{LexF1}(q_i,u_j))
+\beta_3\,\mathrm{norm}(\mathrm{BM25}(q_i,u_j)),
$$
并保留每个 atom route 的 top-$k_a$ 结果：
$$
R_i=\operatorname{TopK}_{k_a}\{u_j:s(q_i,u_j)\}.
$$
所有 atom routes 的并集记为：
$$
\mathcal A_{\mathrm{route}}=\bigcup_{i=1}^{m} R_i.
$$

多个 atom routes 的结果采用 reciprocal rank fusion（RRF）聚合，并记录每个 evidence unit 被哪些 atoms 命中、命中次数、最大 hybrid retrieval score 与 route-level rank。随后将 claim-level baseline pool 与 atom-route pool 按文本规范化 key 去重，并使用 MMR 在相关性与多样性之间做折中，得到送入 evidence map 与 BACES solver 的候选池：
$$
\mathcal C=\mathrm{MMR}\big(\mathrm{Dedup}(\mathcal B\cup\mathcal A_{\mathrm{route}})\big).
$$
其中 $\mathcal B$ 保证全局相关性，$\mathcal A_{\mathrm{route}}$ 提高对细粒度 atoms 的覆盖。重复项合并属于上游 candidate-construction contract，并在进入 BACES 之前完成。当前 solver adapter 不执行 duplicate-class representative selection；它验证每条 retained candidate 具有唯一、非空的 `candidate_uid`，再按该 stable identity 排序。由此得到的 $\mathcal C$ 进入 atom--evidence map，并由 BACES 在显式预算下构造 verifier-facing slate。Retrieval encoder、$k_b/k_a$、RRF 参数、上游 dedup policy、MMR 的 $\lambda_{\mathrm{mmr}}$ 与候选池大小 $n$ 记录于 Appendix A；retrieval-route ablation 用于分析 Atom-Union 的贡献。

## Atom-Evidence Map

给定 canonical candidate pool $\mathcal C=\{e_j\}_{j=1}^{n}$ 与 claim atoms $\mathcal A=\{a_i\}_{i=1}^{m}$，atom--evidence map 为每个 pair 提供：
$$
M(e_j,a_i)=(r_{ij},d_{ij},\gamma_{ij},s_{ij}),
$$
其中 $r_{ij}$ 为 relation，$d_{ij}$ 为 directness，$\gamma_{ij}$ 为 confidence，$s_{ij}$ 为 evidence 中对应的 key span。LLM API 通过固定 prompt 与 JSON schema 生成这些字段；schema adapter 将模型或数据集特化的 relation/directness aliases 映射到 canonical vocabulary。

可计数的 relation 集合定义为：

$$
\mathcal R_{\mathrm{cov}} =
\{\texttt{support},\texttt{refute},\texttt{qualify},\texttt{mixed}\}.
$$

Pair-level valid predicate 为：

$$
V_{ij} =
\mathbf 1\left[ r_{ij}\in\mathcal R_{\mathrm{cov}}
\land d_{ij}\in\{\texttt{direct},\texttt{partial}\}
\land \gamma_{ij}>0
\land s_{ij}\ne\varnothing
\right].
$$

在 valid gate 之上定义 ordinal structural coverage：

$$
q_{ij}
=
\begin{cases}
2, & V_{ij}=1\land d_{ij}=\texttt{direct},\\
1, & V_{ij}=1\land d_{ij}=\texttt{partial},\\
0, & \text{otherwise}.
\end{cases}
$$

整数 0/1/2 同时充当三态 state codes 与冻结的 coverage-unit values：partial 提供一个 unit，direct 提供两个 units，state 从 1 升至 2 时新增一个 unit。等间距是显式的 utility modeling choice，相关 scale sensitivity 在统一 state transition 下单独评价。

当同一 $(a_i,e_j)$ 具有多条 canonical rows 时，$q_{ij}$ 取这些 rows 的最大值。每条 evidence 由 ordinal vector
$$
\mathbf q_j=(q_{1j},\ldots,q_{mj})\in\{0,1,2\}^{m}
$$
表示，并同步作用于全部 valid aligned atoms。

Confidence 只参与 valid gate，连续 confidence value 不进入 coverage utility、dedup 或 canonical tie-break。Relation 决定 pair 的可计数性，directness 决定 coverage level。Support/refute conflict 由 downstream verifier 处理，BACES state 对每个 atom 仅保留当前最高 ordinal level。Pair quality $q_{ij}$ 构成 BACES structural objective 的唯一 map input；原始 relation、confidence、key span 和 provenance 保留在 trace sidecar 中供审计与人工评价。

## Budgeted Atom-Coverage Evidence Sequencing

### Terminal graded set objective

BACES 将 evidential content 表示为集合。对任意 evidence set \(S\subseteq\mathcal C\)，atom \(a_i\) 的 terminal ordinal state 定义为：

$$
x_i(S)
=
\max\left(\{0\}\cup\{q_{ij}:e_j\in S\}\right).
$$
Componentwise max 保留当前集合对每个 atom 达到的最高 evidence level。Partial evidence 将 state 提升至 1，direct evidence 将 state 提升至 2；两条 partial evidence 不自动合成为 direct，same-level repeated evidence 也不累加隐藏效用。

每个 atom 具有正权重 \(w_i\)。主配置固定 \(w_i=1\)；非均匀权重在 solver 前以冻结精度量化为正整数。Terminal graded coverage 为：

$$
U(S)=\sum_{i=1}^{m}w_i x_i(S).
$$

对任意 ordinal state $\mathbf x$，本文同样记 $U(\mathbf x)=\sum_iw_ix_i$。
令 evidence-count 与 token budgets 诱导的 feasible family 为：

$$
\mathcal F=
\left\{
S\subseteq\mathcal C:
|S|\le K_{\max},
\ \sum_{e_j\in S}c_j\le B
\right\}.
$$

对应的可行序列域定义为：

$$
\Pi_{\mathcal F}
=
\left\{
\pi=(e_{\pi_1},\ldots,e_{\pi_L}):
\begin{array}{l}
0\le L\le K_{\max},\\
e_{\pi_s}\ne e_{\pi_t}\ \text{for }s\ne t,\\
\sum_{t=1}^{L}c_{\pi_t}\le B
\end{array}
\right\}.
$$
未启用 token constraint 时取 $B=+\infty$。
Primary objective 及其 terminal-optimal set family 定义为：
$$
U^\star=\max_{S\in\mathcal F}U(S),
\qquad
\mathcal S^\star=\arg\max_{S\in\mathcal F}U(S).
$$
\(U(S)\) 对 evidence permutation 保持不变，并构成 monotone submodular set function。它度量 map-defined structural coverage；gold evidence 与人工 sufficiency evaluation 用于检验该 surrogate 的外部有效性。

### Verifier-facing presentation objective

对 ordered core

$$
\pi=(e_{\pi_1},\ldots,e_{\pi_L})
$$

记其底层集合为 \(S_\pi\)，前缀 state 为：

$$
x_i^{(t)}
=
\max_{1\le s\le t}q_{i,\pi_s},
\qquad
\mathbf x^{(0)}=\mathbf 0.
$$

对 terminal state 中实际取得的 ordinal unit \(\ell\in\{1,2\}\)，其首次 acquisition position 为：

$$
\tau_{i,\ell}(\pi)
=
\min\left\{
t:x_i^{(t)}\ge\ell
\right\}.
$$

Weighted coverage-unit acquisition time 定义为：

$$
T(\pi)
=
\sum_{i=1}^{m}
\sum_{\ell=1}^{x_i(S_\pi)}
w_i\tau_{i,\ell}(\pi).
$$

更小的 \(T(\pi)\) 表示有效 coverage units 更早进入 slate prefix。令：

$$
\Omega^\star
=
\{\pi\in\Pi_{\mathcal F}:S_\pi\in\mathcal S^\star\}.
$$
Secondary objective 在 \(\Omega^\star\) 上选择：
$$
\pi^\star
\in
\arg\min_{\pi\in\Omega^\star}
\left(
T(\pi),
|\pi|,
\sum_{e_j\in S_\pi}c_j,
\operatorname{keys}(\pi)
\right)_{\mathrm{lex}}.
$$

该形式允许 presentation objective 在 terminal-equivalent 的不同 sets 及其 permutations 之间共同决胜。等价的单一 lexicographic objective 为：

$$
\min_{\pi}^{\mathrm{lex}}
\left(
-U(S_\pi),
T(\pi),
|\pi|,
\sum_{e_j\in S_\pi}c_j,
\operatorname{keys}(\pi)
\right).
$$

令第 \(t\) 步的 marginal 为：

$$
\Delta_t
=
U(S_{\pi_{1:t}})-U(S_{\pi_{1:t-1}}).
$$
若某一步满足 \(\Delta_t=0\)，正权重与 componentwise-max transition 蕴含该 candidate 已被先前 state 支配。删除该步保持 terminal state 与 \(U\) 不变，使 cost 和 length 下降，并使后续 coverage units 的 acquisition positions 不增加。因此，完整 lexicographic objective 至少存在一个每步均满足 \(\Delta_t>0\) 的最优序列；exact solver 可将搜索域限制为 positive-gain coverage cores。

取固定 horizon \(H=K_{\max}\)，并在 core 结束后以 terminal state padding，定义：

$$
\operatorname{AUC}_{H}(\pi)
=
\sum_{t=1}^{H}
U\!\left(\mathbf x^{(\min(t,L))}\right).
$$

两者满足：

$$
\operatorname{AUC}_{H}(\pi)
=
(H+1)U(S_\pi)-T(\pi).
$$
因此，在 terminal coverage 相同的解之间，最小化 \(T(\pi)\) 等价于最大化 padded prefix-coverage AUC。该顺序刻画结构覆盖的早期可访问性。Evidence-to-evidence dependency、multi-hop composition、conflict resolution 和 verifier-specific utility 不进入该目标。

Direct alignment 提供两个 ordinal coverage units，partial alignment 提供一个。最终顺序同时受 atom weights、多 atom coverage、budget feasibility 和已有 prefix state 影响，因此 directness 不诱导跨全部 candidates 的固定全序。

## Exact Ordered-State Dynamic Programming

一般规模的 terminal set problem 包含 weighted budgeted maximum coverage 特例，因而为 NP-hard。当前 claim decomposition 满足 \(m\le6\)，ordinal state 数最多为 \(3^m\)。这一 bounded atom universe 支持对冻结的 lexicographic objective 进行精确求解。

启用整数 token budget 时，动态规划状态：

$$
D[k,\mathbf x,b]
$$

记录使用 \(k\) 条 positive-gain core evidence、总成本恰为 \(b\)、到达 ordinal state \(\mathbf x\) 时的最小 acquisition time。初始化为：

$$
D[0,\mathbf 0,0]=0.
$$

对 candidate \(e_j\)，定义：

$$
\mathbf x'=\max(\mathbf x,\mathbf q_j),
\qquad
\Delta_j(\mathbf x)=U(\mathbf x')-U(\mathbf x).
$$

Coverage core 仅保留 \(\Delta_j(\mathbf x)>0\) 的 transition。递推为：

$$
D[k+1,\mathbf x',b+c_j]
=
\min
\left\{
D[k+1,\mathbf x',b+c_j],
D[k,\mathbf x,b]+(k+1)\Delta_j(\mathbf x)
\right\}.
$$
新增 coverage units 首次出现在 position \(k+1\)，所以 acquisition-time increment 为 \((k+1)\Delta_j(\mathbf x)\)。启用 \(B\) 时，exact-cost cell 内按 \((T,\operatorname{keys})\) 保留 canonical sequence。未启用 \(B\) 时使用 \(D[k,\mathbf x]\)，total token cost 作为 cell value 的一部分，并按 \((T,\text{cost},\operatorname{keys})\) 保留 canonical sequence。当前 reference solver 在 cell 内存储完整 key sequence，以支持逐 cell 审计。

设：
$$
r=\min(K_{\max},2m).
$$
每个 core step 至少增加一个 ordinal unit，而总 unit 数不超过 \(2m\)，因此 \(L\le r\)。带整数 token budget 时，arithmetic state relaxations 的数量为：
$$
O(nr3^mB),
$$
使用 rolling layers 与 persistent backpointers 时，数值 state space 可降至 \(O(3^mB)\)，回溯存储另计。当前可审计 reference implementation 保留所有 reachable cells，并在每个 cell 中显式复制和比较长度至多为 \(r\) 的 stable-key sequence，因此包含 canonical tie-break 的保守时间界为：
$$
O(nr^2 3^mB).
$$
若使用可在 \(O(1)\) 时间比较的 persistent backpointer rank，则时间界恢复为 arithmetic relaxation bound。当前实现中保存 full sequences 的保守空间界为 \(O(r^2 3^mB)\)。无 token budget 时，arithmetic relaxations 为 \(O(nr3^m)\)，当前 reference implementation 的保守时间界与空间界分别为 \(O(nr^2 3^m)\) 和 \(O(r^2 3^m)\)。带 \(B\) 的复杂度为 pseudo-polynomial。

启用 \(B\) 时，终态在所有 \(k\le r\) 与 \(b\le B\) 的 reachable exact-cost cells 中选择；未启用 \(B\) 时，终态在所有 reachable \(D[k,\mathbf x]\) cells 中选择。两种情形均使用完整 tuple：
$$
\left(
-U(\mathbf x),
T,
k,
b,
\operatorname{keys}(\pi)
\right)
$$
进行 lexicographic selection。当前 reference solver 直接读取 winning cell 中保留的 sequence 得到 \(\pi^\star\)；persistent-backpointer 实现可通过回溯得到同一结果。该 solver 在 upstream-deduplicated canonical candidate pool、冻结 ordinal vectors、正量化 weights、additive integer costs 与 count/token upper bounds 下，全局优化 map-defined objective。该 exactness guarantee 不外推到人工事实充分性或 verifier correctness。

## Coverage Core, Rendering Floor, and Selected Slate

Exact solver 输出 positive-gain coverage core \(\pi^\star\)，其长度记为：

$$
K_{\mathrm{core}}=|\pi^\star|.
$$

Core 中每个 step 均具有严格正的 ordinal marginal。Duplicate、pure context 与 zero-gain evidence 不进入 core。

\(K_{\min}\) 构成 soft rendering floor。当 \(K_{\mathrm{core}}<K_{\min}\) 时，rendering layer 从 canonical unselected candidates 中筛选满足 \(\mathbf q_j\le\mathbf x^\star\) 的 zero-gain evidence，并在剩余 count/token budget 内按 deterministic fill key 追加：

$$
\left(
c_j,
-D_j,
-P_j,
-r_j^{\mathrm{retr}},
\operatorname{stablekey}(e_j)
\right),
$$

其中：

$$
D_j=\sum_iw_i\mathbf 1[q_{ij}=2],
\qquad
P_j=\sum_iw_i\mathbf 1[q_{ij}=1].
$$

Cost-first ordering优先满足 soft floor，随后使用 direct coverage、partial coverage、retrieval score 与 stable key 决胜。Selected slate 为：

$$
\pi_{\mathrm{sel}}
=
\pi^\star\oplus\operatorname{Fill},
\qquad
K_{\mathrm{sel}}=|\pi_{\mathrm{sel}}|.
$$
Fill 只追加在 core 末尾，不参与 terminal coverage 与 acquisition-time objective。Pool 或 token budget 无法达到 \(K_{\min}\) 时允许 underfill，并在 trace 中记录原因。

Solver 输出之后，tokenizer-aware context guard 可能从 slate 尾部移除 evidence。实际进入 verifier 的数量记为 \(K_{\mathrm{final}}\)。本文分别记录 \(K_{\mathrm{core}}\)、\(K_{\mathrm{sel}}\) 与 \(K_{\mathrm{final}}\)，从而隔离 optimization、rendering 和 prompt realization 三个层级。

## Algorithm

~~~text
Algorithm 1: Exact BACES Ordered-State Solver

Input:
    claim atoms A, canonical candidates C,
    ordinal matrix Q, atom weights w,
    evidence costs c, K_max, optional token budget B,
    optional soft rendering floor K_min
Output:
    coverage core pi*, selected slate pi_sel

1:  Validate unique non-empty candidate UIDs and sort by stable UID
2:  r = min(K_max, 2 * |A|)
3:  if token budget B is enabled then
4:      Initialize exact-cost cell D_B[0, 0, 0] = (T=0, keys=())
5:  else
6:      Initialize cost-carrying cell D_0[0, 0] = (T=0, cost=0, keys=())
7:  for k = 0, ..., r - 1 do
8:      for each reachable cell with state x and accumulated cost b do
9:          for each candidate e_j in C do
10:             x' = componentwise_max(x, q_j)
11:             delta = U(x') - U(x)
12:             if delta <= 0: continue
13:             b' = b + c_j
14:             if B is enabled and b' > B: continue
15:             T' = T + (k + 1) * delta
16:             if B is enabled then
17:                 Relax D_B[k+1, x', b'] by (T', keys + key_j)
18:             else
19:                 Relax D_0[k+1, x'] by (T', b', keys + key_j)
20: Select the retained cell minimizing (-U(x), T, k, b, keys)
21: Read its retained key sequence as positive-gain core pi*
22: Append deterministic zero-gain fill up to soft K_min,
        preserving K_max and the enabled token budget
23: return pi*, pi_sel
~~~

Algorithm 1 接受 BACES 的 canonical optimization inputs。Atomization、retrieval 与 map annotation 构成上游 input construction，并在系统图中单独展示。

## Verifier Rendering and Prediction

BACES rendering 直接消费冻结的 selected slate \(\pi_{\mathrm{sel}}\)，并保持 solver core order 与 provenance。给定模型最大上下文长度 \(L_{\max}\)，tokenizer-aware context guard 从 slate 尾部移除超出上下文的 evidence，得到：

$$
\pi_{\mathrm{final}}
=
\operatorname{ContextGuard}
(c,\pi_{\mathrm{sel}},L_{\max}),
\qquad
K_{\mathrm{final}}=|\pi_{\mathrm{final}}|.
$$

Prompt input 为：

$$
x=\operatorname{Render}(c,\pi_{\mathrm{final}}).
$$
Rendering template 为每条 evidence 保留 stable identity、report/source identifier、原始文本和可选 key span。Map rationale、oracle metadata 与 solver-internal objective values 不进入 verifier prompt。

Verifier 在标签集合 \(\mathcal Y\) 上执行分类：

$$
\hat y
=
\arg\max_{y\in\mathcal Y}
p_{\theta_{\mathrm{ver}}}(y\mid x).
$$

每个 dataset label 映射到固定的 single-token answer choice。训练时在 answer token 位置计算：

$$
\mathcal L_{\mathrm{verifier}}
=
-\log p_{\theta_{\mathrm{ver}}}(z^\ast\mid x),
$$

其中 \(z^\ast\) 为 gold label 对应的 answer token。Label mapping、rendering template 与 context guard 在 training、validation 和 test splits 上保持一致。

BACES trace 同时记录 solver core、selected slate 与 final visible identities。Prompt realization loss 通过比较 \(K_{\mathrm{core}}\)、\(K_{\mathrm{sel}}\)、\(K_{\mathrm{final}}\) 以及各层 terminal/prefix coverage 得到。Strict same-set controls 固定 selected stable-key set、evidence token multiset 和 rendering template，仅替换 presentation permutation，并重新计算 display acquisition trajectory。

## Training and Inference Pipeline

整个系统包含冻结的上游 annotation、deterministic BACES optimization 与 supervised verifier training 三个阶段。

**Upstream annotation.** Training、validation 与 test samples 使用同一 atomization、query rendering、candidate construction 和 atom--evidence map schema。LLM-generated annotations 经过 canonical adapter、deduplication 与 fingerprinting 后冻结。

**BACES materialization.** Exact solver 对每个 sample 的 \((\mathcal A,\mathcal C,Q,\mathbf w,\mathbf c,K_{\max},B)\) 生成 coverage core、selected slate 和 replayable trace。主配置使用 \(w_i=1\) 与冻结的 count/token budget policy。该阶段不存在 preference generation、selector checkpoint 或 optimizer。Historical learned-marginal selector 保留为 matched experimental baseline。

**Verifier training.** BACES selected slates 经统一 rendering 与 context guard 转换为 verifier training rows。Verifier 使用 gold fact-checking labels 和 label-token cross-entropy 训练；其 gradients 仅更新 verifier parameters。

**Inference.** 测试样本依次经过 frozen upstream annotation、exact BACES solver、selected-set rendering、context guard 与 fixed verifier prediction。BACES objective、budgets、stable-key policy 和 map adapter 在 test evaluation 前冻结。Cross-domain 或 cross-verifier evaluation 复用相同 BACES traces 时，不执行 selector retraining。

## Evaluation and Regret Decomposition

Exact evaluator 使用 stable candidate identity 对齐 evidence-map features、selection traces 与 verifier-build rows。对每个 sample 定义六个 terminal utilities：$U_{\mathrm{ideal}}$ 表示全部 atoms 达到 direct level 的理论上界，$U_{\mathrm{pool}}$ 表示 candidate pool 可达到的无预算上界，$U_{\mathrm{opt}}$ 表示共享可行域 $\mathcal F$ 下的 BACES exact optimum，$U_{\mathrm{full}}$ 表示待评 policy 经相同 count/token budgets 约束后的完整可行输出，$U_{\mathrm{pre}}$ 与 $U_{\mathrm{final}}$ 分别表示 prompt truncation 前和 verifier-visible slate 的 utility。

有效分解要求 $S_{\mathrm{full}}\in\mathcal F$，并满足序列前缀关系：
$$
\pi_{\mathrm{final}}\preceq\pi_{\mathrm{pre}}\preceq\pi_{\mathrm{full}},
$$
其中 $\preceq$ 表示左侧序列是右侧序列的前缀。于是：
$$
U_{\mathrm{ideal}}
\ge U_{\mathrm{pool}}
\ge U_{\mathrm{opt}}
\ge U_{\mathrm{full}}
\ge U_{\mathrm{pre}}
\ge U_{\mathrm{final}}.
$$

总 terminal loss 分解为：

$$
\begin{aligned}
L_{\mathrm{pool}} &= U_{\mathrm{ideal}}-U_{\mathrm{pool}},\\
L_{\mathrm{budget}} &= U_{\mathrm{pool}}-U_{\mathrm{opt}},\\
L_{\mathrm{selector}} &= U_{\mathrm{opt}}-U_{\mathrm{full}},\\
L_{\mathrm{stop}} &= U_{\mathrm{full}}-U_{\mathrm{pre}},\\
L_{\mathrm{realization}} &= U_{\mathrm{pre}}-U_{\mathrm{final}}.
\end{aligned}
$$

在上述 alignment invariants 下，各分量均为非负，并满足 conservation identity：

$$
U_{\mathrm{ideal}}-U_{\mathrm{final}}
=
L_{\mathrm{pool}}
+L_{\mathrm{budget}}
+L_{\mathrm{selector}}
+L_{\mathrm{stop}}
+L_{\mathrm{realization}}.
$$

该分解将 candidate reachability、预算、set selection、capacity policy 与 prompt realization 的损失分开报告。任何共享预算、identity alignment 或 prefix nesting 失败都会使对应 row 失去 regret 解释；exact evaluator 将其标记为 invalid artifact，不纳入 loss aggregation。

对任意冻结 evidence set $S$，exact fixed-set orderer 给出：

$$
T^\star(S)
=
\min_{\pi:\,S_\pi=S}T(\pi).
$$

Presentation order regret 为：

$$
R_{\mathrm{order}}(\pi)
=
T(\pi)-T^\star(S_\pi)\ge0.
$$

Strict same-set controls 保持 $S_\pi$、count、token multiset、rendering template 与 verifier checkpoint 不变，只替换 permutation。该协议隔离 presentation effect，并与改变 selected set 的 terminal-coverage comparison 分开统计。

Factorial protocol 将 evidence policy（retrieval/static、terminal-only exact、prefix greedy、BACES exact 与 historical learned baseline）和 capacity policy（fixed count 与 matched token budget）交叉。Trace replay 在保持原 solver role 与 selected-set fingerprint 的条件下重算 display ordinal states、acquisition time 和 padded prefix AUC，从而为 selection、capacity、order 与 prompt realization 提供统一的 paired evaluation interface。

# Experiments

> **v0.5 experiment status.** 本节的表格结构与数值沿用 v0.4.1，记录 historical learned-marginal pipeline 的实验事实。表中的 `ours`、`主方法` 与 `Atom-Union MREC` 均指该历史条件。BACES rows 将由冻结 solver 重新物化 train/validation/test inputs 后替换；当前数值不用于支撑 v0.5 的 BACES empirical claims。

## Main Evaluation

### Evaluation Protocol and Reporting Scope

**LIAR-RAW 与 RAWFC。** 两个数据集均使用官方 train/validation/test 划分，在 test split 上报告 accuracy 以及 macro-precision、macro-recall 和 macro-F1。主文献对比表统一以百分数报告 macro-P/R/F1。Macro-F1 定义为各类别 F1 的算术平均；macro-P 与 macro-R 的调和平均不适用。LIAR-RAW 的 ordinal MAE 与 extreme error rate 作为补充诊断另行报告。

**SciFact。** 使用原始 5,183 篇摘要语料和官方 300-claim development split，不使用 gold `cited_doc_ids` 或 gold rationale 构造候选池。按照 SciFact 官方 full-pipeline scorer 报告 sentence Selection-only、sentence Selection+Label、abstract Label-only 和 abstract Label+Rationale 四项 micro-F1。由于官方 hidden-test 提交服务和联系渠道在实验定稿时不可用，SciFact 结果统一标记为 **official development-set results**，不承载 hidden-test 或整体 SOTA 主张。该 development split 同时参与 checkpoint selection，其结论定位为跨领域迁移证据。

**训练口径。** 本文方法的主要贡献位于 evidence construction 与 selector，而 verifier backbone 和参数更新方式均可能显著影响最终分类性能。为避免按数据集分别挑选 LoRA/FullFT 和不同 backbone 造成选择偏差，LIAR-RAW/RAWFC 主结果采用同时覆盖两个数据集、并由 validation macro-F1 选择的 matched configuration。当前完成的两组成对配置中，Ministral-3-8B + LoRA 的跨数据集平均 validation macro-F1 高于 Llama-3.1-8B + LoRA，因此本稿暂以 R1/R2 为统一主口径；二者均使用 label-token CE 和 minmax$(5,10)$ prompt evidence policy。允许针对数据集调节学习率、验证间隔与 retrieval pool 宽度，并在实现细节中完整披露；RAWFC 使用 baseline20 候选设置。SciFact 是独立的跨领域迁移实验，使用 Ministral-3-8B + LoRA 和 fixed minmax$(9,9)$。

### LIAR-RAW and RAWFC Controlled Results

下表给出当前可作为正文主结果的统一 backbone 与 adaptation-family 口径。所有数值均来自完整 test artifact。

| 数据集 | Run | Verifier / adaptation | Evidence setting | Acc. | Macro-P | Macro-R | Macro-F1 |
|---|---:|---|---|---:|---:|---:|---:|
| LIAR-RAW | R1 | Ministral-3-8B / LoRA | Atom-Union, minmax$(5,10)$ | 35.97 | 38.37 | 35.87 | **36.66** |
| RAWFC | R2 | Ministral-3-8B / LoRA | Atom-Union baseline20, minmax$(5,10)$ | 66.00 | 67.17 | 65.98 | **66.12** |

**Table X: LIAR-RAW/RAWFC test-set comparison under raw-report or near raw-report settings.** 外部结果沿用其论文公开的最佳变体；由于部分方法使用额外知识、不同 LLM 规模或 agent protocol，`Context` 和 `Upper reference` 行只作为上下文，不用于“同设置最佳”结论。

| Method | Comparison scope | LIAR-RAW / LIAR P / R / F1 | RAWFC P / R / F1 |
|---|---|---:|---:|
| CofCED | Direct; raw reports | 29.48 / 29.55 / 28.93 | 52.99 / 50.99 / 51.07 |
| L-Defense | Direct; raw reports + competing wisdom | 31.63 / 31.71 / 31.40 | 61.72 / 61.01 / 61.20 |
| G-Defense | Direct; graph-enhanced defense | 34.17 / 32.37 / 32.49 | 66.29 / 65.49 / 65.50 |
| DeReC-qwen | Direct; dense retrieval + DeBERTa classifier | 35.94 / 32.24 / 33.13 | 65.58 / 64.56 / 64.60 |
| FFRR(d+q) | Direct; feedback-trained retrieval + reader | 34.50 / 32.60 / 33.50 | 56.50 / 57.40 / 57.00 |
| **Atom-Union MREC (ours)** | **Direct; Ministral-3-8B + LoRA on both datasets (R1/R2)** | **38.37 / 35.87 / 36.66** | **67.17 / 65.98 / 66.12** |
| FactLLaMAKnow | Context; LLaMA LoRA + external knowledge | 32.46 / 32.05 / 30.44 | 56.11 / 55.50 / 55.65 |
| DelphiAgent GPT-4o | Context; training-free multi-agent | 31.33 / 28.36 / 28.36 | 68.05 / 68.03 / 68.04 |

> 表述边界：`Direct` 行仅覆盖同资源口径的公开结果。FactLLaMAKnow 与部分后续工作沿用 `LIAR` 命名，其与 LIAR-RAW 的数据处理一致性记录在最终表注中。外部结果的原始来源与转录记录见 `docs/Z-cross-cutting/202606011907_paper_data_and_todo_record.md`。

### Training-Regime Audit and Final Reporting Rule

截至本稿更新，八组 verifier 实验的状态如下。该表是内部写作审计，最终投稿时只保留选定的 matched configuration 和必要的 backbone/adaptation ablation。

| Run | Dataset | Backbone | Adaptation | Acc. | Macro-P | Macro-R | Macro-F1 | Status / role |
|---:|---|---|---|---:|---:|---:|---:|---|
| R1 | LIAR-RAW | Ministral-3-8B | LoRA | 35.97 | 38.37 | 35.87 | 36.66 | complete; controlled main |
| R2 | RAWFC | Ministral-3-8B | LoRA | 66.00 | 67.17 | 65.98 | 66.12 | complete; controlled main |
| R3 | LIAR-RAW | Llama-3.1-8B | LoRA | 33.33 | 37.71 | 33.16 | 34.31 | complete |
| R4 | RAWFC | Llama-3.1-8B | LoRA | 62.50 | 63.54 | 62.49 | 62.83 | complete |
| R5 | RAWFC | Ministral-3-8B | FullFT | 67.00 | 71.77 | 66.98 | 67.64 | complete |
| R6 | RAWFC | Llama-3.1-8B | FullFT | 69.00 | 69.09 | 69.06 | 69.00 | complete; current RAWFC best |
| R7 | LIAR-RAW | Llama-3.1-8B | FullFT | -- | -- | -- | -- | running; do not report yet |
| R8 | LIAR-RAW | Ministral-3-8B | FullFT | -- | -- | -- | -- | queued/not materialized; do not report yet |

当前 matched-configuration 的 validation 审计如下。只有两个数据集均有完整 validation artifact 的配置才可参与主配置选择。

| Matched configuration | LIAR-RAW val Macro-F1 | RAWFC val Macro-F1 | Mean val Macro-F1 | Eligible now |
|---|---:|---:|---:|---|
| **Ministral-3-8B + LoRA (R1/R2)** | **36.59** | **64.59** | **50.59** | yes; provisional main |
| Llama-3.1-8B + LoRA (R3/R4) | 34.72 | 58.83 | 46.78 | yes |
| Ministral-3-8B + FullFT (R8/R5) | -- | 64.57 | -- | no |
| Llama-3.1-8B + FullFT (R7/R6) | -- | 67.09 | -- | no |

当前逐数据集最优结果是 LIAR-RAW 的 R1（Ministral + LoRA，36.66 F1）和 RAWFC 的 R6（Llama + FullFT，69.00 F1）。这两个数值在附加表中标记为 **best observed per dataset**；统一主结果保留 matched configuration。正文口径如下：

> To isolate the contribution of evidence construction from verifier training choices, our primary LIAR-RAW/RAWFC comparison uses the matched Ministral-3-8B + LoRA configuration (R1/R2). For completeness, the best score observed on each dataset is 36.66 macro-F1 on LIAR-RAW with Ministral-LoRA and 69.00 on RAWFC with Llama-FullFT. These per-dataset maxima form a performance envelope rather than a single shared verifier configuration.

R7/R8 完成后，将四种 matched configuration 记为
$\mathcal Q=\{\text{Ministral-LoRA},\text{Llama-LoRA},\text{Ministral-FullFT},\text{Llama-FullFT}\}$，只使用两个数据集的 validation macro-F1 选择最终主配置：
$$
q^\ast=\arg\max_{q\in\mathcal Q_{\mathrm{complete}}}
\frac{1}{2}\left(F^{\mathrm{LIAR}}_{1,\mathrm{val}}(q)+F^{\mathrm{RAWFC}}_{1,\mathrm{val}}(q)\right).
$$
随后冻结 $q^\ast$，一次性报告两个 test 结果。不得使用各数据集 test 最优值反向选择不同配置。若 R7/R8 在截稿前仍未完成，则保留 R1/R2 为主口径，将 R6 仅列作训练方式敏感性结果。

### SciFact Full-Pipeline Development Results

SciFact 表使用原始 5,183-abstract corpus 和官方 300-claim development split，所有值均为官方 full-pipeline micro-F1 百分数。`Selection-only` 评价证据句定位，`Selection+Label` 要求句级证据与标签同时正确，`Label-only` 评价摘要级标签，`Label+Rationale` 则要求摘要标签正确且至少给出一组有效 rationale。所有候选证据均来自 open-corpus Atom-Union retrieval，不进行 gold evidence 回填。

| Method | Year | Sent. Selection-only | Sent. Selection+Label | Abstract Label-only | Abstract Label+Rationale |
|---|---:|---:|---:|---:|---:|
| VeriSci | 2020 | 48.30 | 43.10 | 52.10 | 50.00 |
| VerT5erini | 2021 | 60.87 | 57.10 | 65.07 | 61.72 |
| ParagraphJoint | 2021 | 64.70 | 55.20 | 65.10 | 59.90 |
| ARSJoint | 2021 | 66.20 | 57.80 | <u>66.70</u> | 62.40 |
| QMUL-SDS | 2021 | <u>67.83</u> | <u>60.54</u> | 63.40 | 61.10 |
| RerrFact | 2022 | **76.37** | **63.76** | 64.59 | <u>64.02</u> |
| PrunE | 2025 | 62.96 | 53.29 | 63.21 | 60.10 |
| **Atom-Union MREC (ours)** | -- | 40.51 | 39.23 | **72.41** | **65.82** |

本文结果由 SciFact 官方 `verisci/evaluate/pipeline.py` scorer 复算，precision/recall/F1 与本地 exporter 完全一致：

| Official metric | Precision | Recall | F1 |
|---|---:|---:|---:|
| Sentence Selection-only | 38.16 | 43.17 | 40.51 |
| Sentence Selection+Label | 36.96 | 41.80 | 39.23 |
| Abstract Label-only | 76.88 | 68.42 | 72.41 |
| Abstract Label+Rationale | 69.89 | 62.20 | 65.82 |

**结果分析。** 在该 development-set 对比中，本文方法的 abstract Label-only F1 为 72.41，较次高结果 66.70 提升 5.71；abstract Label+Rationale F1 为 65.82，较次高结果 64.02 提升 1.80。这说明完整迁移后，Atom-Union MREC 能够较好地完成科学摘要级标签判断，并为正确摘要附加至少一组有效 rationale。Sentence Selection-only 和 Selection+Label F1 分别为 40.51 和 39.23，较 RerrFact 低 35.86 和 24.53，显示出 exact sentence localization 短板。该结果归纳为 **strong abstract-level cross-domain transfer with limited exact rationale localization**，整体 SciFact SOTA 不在主张范围内。

表中 VeriSci 至 RerrFact 的 development-set 数值来自 RerrFact Table 4；PrunE 数值来自其 Table 2。PrunE 使用 top-150 bigram-TF-IDF universe，并在 development inference 中采样 12 个候选摘要，该设置差异记录在正式表注中。MultiVerS/BEVERS 的 hidden-test 结果、evidence-provided verification-only 方法、BEIR retrieval-only 指标和 SciFact-Open 结果均不混入此表。完整转录、排除规则和 BibTeX 见 `docs/paper/aaai/scifact_dev_comparison_table.md`；官方 scorer 审计 artifact 为 `outputs/sentence_trace_method/scifact__ministral3_8b__atom_union_fullpool_minmax9_9_lora_ebs16_lr2em5_ep12_eval100_pat8/submission/scifact_official_scorer_metrics_val.json`。

## Ablation Study

本节从两个维度展开消融，所有消融均在 LIAR-RAW 上进行，使用与主方法相同的 verifier 训练设置（Ministral-3-8B + LoRA）。除非特别说明，selector 与 evidence map 的配置与主方法一致。

### Selector Mechanism Ablation

**动机与设计。** 本消融验证证据选择机制的各组件（候选池来源、学习与否、证据顺序）的必要性。所有变体共享同一 verifier，固定 top-5 证据容量，仅改变证据来源与排序方式。

- **s0 no\_evidence**：verifier 不接收任何证据，仅看 claim，作为性能下界。
- **s1 random\_top5**：从 Atom-Union 池随机选 5 条。
- **s2 claim\_hybrid\_top5**：仅用整条 claim 检索（baseline pool），hybrid 排序取 top5。
- **s3 claim\_hybrid\_mmr\_top5**：在 s2 基础上加 MMR 去冗余。
- **s4 atom\_union\_source\_score\_top5**：完整 Atom-Union 池，按来源融合分数排序（无 map、无学习）。
- **s4b atom\_route\_only\_top5**：仅用 atom-route 检索结果（去掉 baseline pool），验证 union 的必要性。
- **s5 map\_quality\_greedy**：完整 map 驱动的贪心构链，但 selector 权重为手工设定（无学习）。
- **s6 trace\_shuffle**：用主方法的 learned selector 生成 trace 后**打乱顺序**，验证证据有序性的价值。
- **主方法（learned\_proxy, fixed\_top5）**：完整 learned marginal selector（minmax5\_5，即 $K^\ast{=}5$），作为同容量对照。

**结果。**

| 变体 | Acc. | Macro-F1 | $\Delta$F1 vs 主方法 |
|---|---|---|---|
| s0 no\_evidence | 0.307 | 0.310 | $-2.9$ |
| s1 random\_top5 | 0.313 | 0.317 | $-2.2$ |
| s3 claim\_hybrid\_mmr\_top5 | 0.318 | 0.318 | $-2.1$ |
| **s6 trace\_shuffle** | 0.317 | **0.329** | $-1.0$ |
| s2 claim\_hybrid\_top5 | 0.338 | 0.345 | $-0.6$ |
| s4b atom\_route\_only\_top5 | 0.311 | 0.321 | $-1.8$ |
| s5 map\_quality\_greedy | 0.349 | 0.349 | $+1.0$ |
| s4 atom\_union\_source\_score\_top5 | 0.350 | 0.354 | $+1.5$ |
| 主方法 (learned\_proxy, fixed\_top5) | 0.333 | 0.339 | — |

**分析。**

1. **证据本身有用（s0 $\to$ s1）。** 从无证据（F1 0.310）到随机证据（0.317），即使随机选 5 条也能带来提升，说明证据对 LIAR-RAW 判别有基础价值。

2. **Atom-Union 池优于 claim-level 检索。** s4（atom-union，0.354）显著优于 s2（claim-only，0.345）和 s3（claim+MMR，0.318），且 s4b（仅 atom-route，0.321）明显低于 s4，证明 baseline pool 与 atom-route 的融合（union）是必要的——两者互补，缺一不可。

3. **map + 学习进一步提升。** s4（无 map 无学习，0.354）$\to$ s5（有 map 无学习，0.349）$\to$ 主方法（有 map 有学习，0.339）。此处的历史“主方法”固定 top5（minmax5\_5）以与 s0–s6 对齐容量；minmax5\_10 条件达到 0.367（见 Main Evaluation）。Fixed\_top5 结果中 s4 略高于 learned condition，因此该历史实验的收益主要来自解析驱动的动态截断；固定容量结果未显示 learned ordering 优势。

4. **证据有序性有贡献。** s6（shuffle，0.329）相比主方法下降 1.0 F1，证明 selector 生成的证据顺序对 verifier 判别有意义——打乱顺序后性能显著下降。

### Evidence Map Ablation

**动机与设计。** Atom-evidence map $M(u_j,a_i)=(r_{ij},d_{ij},c_{ij})$ 是 selector 决策的结构化中间表示：resolution gain $\phi_{\mathrm{res}}$、stance tension $\phi_{\mathrm{tension}}$、corroboration $\phi_{\mathrm{corr}}$ 等 6 个边际特征均依赖它，且 greedy chain 的 atom 状态转移与 minmax 截断位置 $K^\ast$ 也由 relation 与 directness 驱动。本消融逐元去除 map 的三个信号，检验每个信号对最终判别的贡献。

**变体定义。** 所有变体共享同一 Atom-Union 候选池、同一 verifier 训练设置（Ministral-3-8B + LoRA）与同一 minmax$(k_{\min}{=}5,k_{\max}{=}10)$ prompt evidence policy，仅改变 map 信号输入；selector 权重在各自退化特征上重新训练。具体变体如下：

- **full\_map**（主方法）：完整 map，无退化，作为基准。
- **no\_relation**：将每个 pair 的 relation $r_{ij}$ 退化为 background，使 atom 永不进入 $S/R/Q$。
- **no\_directness**：将 directness $d_{ij}$ 退化为中性档，使解析增益 $\delta(d)$ 下降，且无法独立触发 OPEN 转移。
- **no\_confidence**：将 confidence $c_{ij}$ 强制为 1.0，使 $\max(c,0.5)$ clip 失效，解析增益被人为放大。
- **no\_map**：$r,d,c$ 全部退化，6 个 map 相关特征置零，selector 仅靠 retrieval/text novelty。

**结果。** 下表报告 LIAR-RAW test（$n{=}1251$）上的 verifier 性能，以及各变体 trace 的平均 atom 解析率 $\bar\rho$ 与 minmax 截断后的平均证据容量 $\bar{K^\ast}$。$\bar{K^\ast}$ 直观反映 map 对证据容量的调控：解析率越高，越多样本在 $k_{\min}{=}5$ 处提前停止。

| 变体 | Acc. | Macro-F1 | $\bar\rho$ | $\bar{K^\ast}$ | $K^\ast{=}5$ 占比 |
|---|---|---|---|---|---|
| **full\_map** | **0.360** | **0.367** | 0.774 | 6.23 | 68.9% |
| no\_relation | 0.353 | 0.355 | 0.000 | 9.69 | 0.8% |
| no\_directness | 0.344 | 0.342 | 0.000 | 9.69 | 0.8% |
| no\_confidence | 0.330 | 0.332 | 0.774 | 6.23 | 68.6% |
| no\_map | 0.336 | 0.343 | 0.000 | 9.69 | 0.8% |

**观察与分析。**

1. **map 整体有贡献。** 完整 map 的 full\_map 在 macro-F1 上优于 no\_map（0.367 vs 0.343，+2.4），证明 LLM 结构化标注为 selector 提供了 retrieval score 之外的有效信号。

2. **证据容量调控是 map 价值的主要通道。** full\_map 有 68.9% 的样本在 $K^\ast{=}5$ 提前停止（解析达标），平均仅用 6.23 条证据；而 no\_map / no\_relation / no\_directness 因解析失效，92.6% 的样本跑满 $K^\ast{=}10$。这说明 map 的核心作用不只是"选更好的证据"，更是"判断何时证据已足够"，从而在证据充分性与上下文噪声之间取得平衡。

3. **directness 是解析能力的关键。** no\_directness 使解析率从 0.774 降至 0.000、$K^\ast$ 从 6.23 膨胀至 9.69，说明 directness 直接控制 atom 能否被独立证据解析。去除后 macro-F1 下降 2.5，验证了 $\delta(d)$ 权重设计的必要性。

4. **confidence 的作用需谨慎解读。** no\_confidence 将 $c_{ij}$ 强制为 1.0，使解析率保持 0.774（$\max(c,0.5)$ 下界 clip 被绕过），但 macro-F1 降至最低的 0.332。Uniform high confidence 使历史 selector 丧失高/低可信 pair 的区分能力；$\max(c,0.5)$ clip 在该旧实现中承担鲁棒化作用。

5. **map 提升的是证据效率而非裸精度。** no\_relation 在 macro-F1 上看似接近 full\_map（0.355 vs 0.367），但这一表现是以 $\bar{K^\ast}$ 从 6.23 膨胀至 9.69（多消耗 56\% 的证据容量）为代价的——去除 relation 后解析失效，selector 无法判断"证据已足够"，只能让 verifier 看到近乎翻倍的证据来弥补信号缺失。若以单位证据容量的判别产出衡量（macro-F1 / $\bar{K^\ast}$），full\_map 的证据效率（0.059）约为 no\_relation（0.037）的 1.6 倍。这说明 map 的核心价值在于让 selector 用更少的证据、更低的上下文成本达到同等甚至更优的判别，而非仅在最终 F1 上拉开差距；在证据预算受限或长上下文噪声敏感的场景下，这一效率优势更为关键。

### Prompt Evidence Policy and Budget Sensitivity

**动机与设计。** Prompt evidence policy 决定 verifier 实际看到的证据容量 $K^\ast$。本消融在 LIAR-RAW 上比较三类策略及其参数取值，所有变体共享同一 learned selector 生成的 trace，仅改变截断方式：

- **fixed\_topk（$K^\ast{=}k$）**：固定证据数，对应 minmax$(k,k)$。取 $k\in\{5,7,9\}$。
- **minmax$(k_{\min},k_{\max})$**：达到 $k_{\min}$ 后若解析率达标则提前停止，否则最多到 $k_{\max}$。取 $(k_{\min},k_{\max})\in\{(3,10),(5,10),(7,12)\}$，其中 $(5,10)$ 为主方法配置。
- **budget**：按 token 预算截断（$B{=}1024$）。
- **two\_pass\_uncertainty**：两阶段不确定性决策，先粗判再决定是否补充证据。

**结果。**

| Policy | $k_{\min}/k_{\max}$ 或参数 | Acc. | Macro-F1 |
|---|---|---|---|
| fixed\_topk | $k{=}5$ | 0.333 | 0.339 |
| fixed\_topk | $k{=}7$ | 0.341 | 0.340 |
| fixed\_topk | $k{=}9$ | 0.348 | 0.348 |
| minmax | $(3,10)$ | 0.336 | 0.339 |
| **minmax（主方法）** | $(5,10)$ | **0.360** | **0.367** |
| minmax | $(7,12)$ | 0.333 | 0.345 |
| budget | $B{=}1024$ | 0.345 | 0.350 |
| two\_pass\_uncertainty | — | 0.347 | 0.350 |

**分析。**

1. **解析驱动的动态截断优于固定容量。** minmax$(5,10)$（0.367）显著优于同等最大容量的 fixed\_topk $k{=}5$（0.339）和 $k{=}9$（0.348）。其原因是：minmax 允许解析率达标时提前停止（69% 的样本在 $K^\ast{=}5$ 停止），既避免了固定 $k{=}5$ 时部分样本证据不足的问题，又避免了固定 $k{=}9$ 时证据过多引入噪声的问题。

2. **$k_{\min}$ 的选择敏感。** minmax$(3,10)$（0.339）和 minmax$(7,12)$（0.345）都不如 minmax$(5,10)$（0.367）。$k_{\min}{=}3$ 时过早允许停止，部分样本证据不充分；$k_{\min}{=}7$ 时强制看较多证据，抵消了提前停止的效率优势。$k_{\min}{=}5$ 在证据充分性与效率间取得最佳平衡。

3. **budget 与 two\_pass 表现居中。** 两者（F1 约 0.350）略逊于主方法但优于多数 fixed\_topk，说明无论是按 token 还是按不确定性动态调整容量，都比固定容量好，但不如 atom 解析率驱动的 minmax——后者直接利用了 map 的结构化信号。

### Historical Planned Ablations

以下条件来自 v0.4.1 实验计划，BACES-compatible rerun 尚未物化：

1. Chunking 粒度消融（sentence / ctx\_window / semantic / abc\_claim\_aware / raw）：已有 LIAR-RAW historical test 数据；BACES selected-set 条件仍待统一评估。
2. Atomization 移除（仅用 claim-level retrieval）与 Atom-Union 路由消融。
3. 单项 marginal feature 移除（tension、corroboration、text novelty、cost 等）。
4. $\rho_{\mathrm{target}}$ 与 candidate pool size 的超参敏感性。

## Reliability Study: Claim Atomization

**目的。** 检验 LLM-generated atoms 的 claim faithfulness、verifiable-assertion coverage 与 atomicity。该实验评价 BACES ordinal state space 的输入质量。

**数据。** 从训练/验证集中抽取 200 条 claim，其中 LIAR-RAW 100 条、RAWFC 100 条。

**标注协议。** 由 2 位标注者独立逐条评估。每条 claim 的每个 atom 在三个维度打分：

- **忠实性（faithfulness）**：atom 是否能从 claim 推出，是否引入外部知识或幻觉，binary。
- **完整性（completeness）**：atoms 是否覆盖 claim 中所有可独立验证的事实断言，记录漏检断言数。
- **原子性（atomicity）**：atom 是否最小可验证，是否把多个事实黏在一起，binary。

标注前提供统一指导书与 20 条 calibration 样本。

**测量指标。** 报告 atom hallucination rate、claim omission rate、overall acceptable rate，以及两位标注者在各维度上的 Cohen's $\kappa$。其中：

$$
\mathrm{HallucinationRate}=\frac{\#\ \text{hallucinated atoms}}{\#\ \text{all atoms}},
$$
$$
\mathrm{OmissionRate}=\frac{\#\ \text{claims with at least one omitted assertion}}{\#\ \text{claims}}.
$$

**判断标准。** Hallucination rate 与 omission rate 分别作为 atomization faithfulness 和 coverage 的主指标。Error analysis 将 atom split/merge/drop 与 candidate-pool reachable coverage、terminal utility 和 downstream prediction 对齐。

## Reliability Study: Evidence Map Annotation

**目的。** 检验 $M(e_j,a_i)=(r_{ij},d_{ij},\gamma_{ij},s_{ij})$ 的 relation、directness 与 key-span 标注质量，并评价由 valid gate 导出的 ordinal level $q_{ij}$。

**数据。** 从 Reliability Study 1 对应的 candidate pools 中分层抽取 evidence--atom pairs，覆盖 support、refute、qualify、mixed、background 与 irrelevant，并覆盖 direct、partial 与 invalid 三种 projected levels。

**标注协议。** 2 位标注者独立标注 canonical relation、direct/partial/invalid level 及 key span validity。分歧经 adjudication 形成 evaluation-only gold annotation。该 annotation 不进入 BACES optimization 或 verifier training。

**测量指标。** 报告 relation macro-F1/confusion matrix、ordinal level macro-F1、directness weighted-$\kappa$、key-span validity、human-recomputed terminal coverage 与 annotator IAA。内部 map coverage 与 human-recomputed coverage 分开报告。

**判断标准。** Error analysis 按 relation confusion、direct-to-partial downgrade、invalid false positive 与 missing key span 分类，并量化每类错误对 $U_{\mathrm{pool}}$、$U_{\mathrm{opt}}$ 和 selected slate 的影响。

# Appendix A: Candidate Pool and Retrieval Details Placeholder

本附录记录 embedding model、BM25 实现、LexF1 定义、$\alpha/\beta$ 权重、chunk 最大/最小句数、chunk overlap、$k_b/k_a$、RRF 参数 $k_{\mathrm{rrf}}$、dedup 文本规范化规则、duplicate threshold、MMR 的 $\lambda_{\mathrm{mmr}}$、candidate pool size $n$、token budget 与 dataset-specific configurations。

# Appendix B: LLM Prompts and Schemas Placeholder

本附录记录 claim atomization prompt、query rendering schema、atom--evidence map prompt、relation/directness/confidence/key-span definitions、canonical adapter、JSON validation、retry/fallback rules、temperature、LLM API model version，以及一条完整 map output。

# Appendix C: BACES Objective and Solver Properties

## Monotonicity and submodularity

For each atom \(a_i\), define:
$$
f_i(S)
=
\max\left(\{0\}\cup\{q_{ij}:e_j\in S\}\right),
\qquad
f_i(\varnothing)=0.
$$
For \(S\subseteq T\) and candidate \(e\notin T\), the marginal increase of \(f_i\) is:
$$
f_i(S\cup\{e\})-f_i(S)
=
\max(0,q_i(e)-f_i(S)).
$$
Since \(f_i(S)\le f_i(T)\), this marginal cannot increase when the context set grows. Each \(f_i\) is therefore monotone submodular, and the positive weighted sum
$$
U(S)=\sum_iw_if_i(S)
$$
inherits monotonicity and submodularity.

## NP-hardness boundary

Weighted max-\(k\)-cover is embedded by mapping each universe element to an atom and each source subset to an evidence candidate. Set:
$$
q_{ij}
=
\begin{cases}
2, & \text{candidate }e_j\text{ covers atom }a_i,\\
0, & \text{otherwise},
\end{cases}
$$
retain the original atom weights, set \(K_{\max}=k\), and disable the token constraint. Every feasible source-subset choice corresponds to a feasible evidence set, and BACES terminal utility equals twice the weighted covered-universe value. The reduction preserves optimal solutions, establishing NP-hardness of the general terminal optimization problem.

Weighted budgeted maximum coverage is obtained by assigning each evidence cost \(c_j\) the integer cost of its source subset, setting the token budget to the source budget \(B\), and taking \(K_{\max}=n\) so that the count bound is inactive. Feasible solutions and objectives are again preserved up to the constant factor of two. This separate reduction establishes the budgeted maximum-coverage special case.

## Exactness of the bounded-state solver

For a fixed layer \(k\), ordinal state \(\mathbf x\), and exact token cost \(b\), future positive marginal gains depend only on \(\mathbf x\). A previously selected candidate satisfies \(\mathbf q_j\le\mathbf x\), so selecting it again has zero marginal and cannot enter a positive-gain core. This property removes the need for an explicit used-candidate subset in the DP state.

More generally, deleting any zero-gain step preserves the componentwise-max terminal state, decreases length and cost, and cannot increase the acquisition position of any remaining coverage unit. Every lexicographic optimum therefore has a positive-gain core representative. Restricting recurrence transitions to positive gain preserves at least one global optimum over \(\Pi_{\mathcal F}\).

The recurrence enumerates every reachable positive-gain transition up to
$$
r=\min(K_{\max},2m).
$$
Within each \((k,\mathbf x,b)\) exact-cost cell, the canonical retained sequence minimizes acquisition time and then stable-key sequence. Without token budget, each \((k,\mathbf x)\) cell additionally minimizes total cost before the stable-key tie-break. Terminal selection applies the complete tuple
$$
\left(-U,T,L,C,\operatorname{keys}\right).
$$
Consequently, the retained core globally optimizes the frozen lexicographic objective over the upstream-deduplicated canonical candidate pool.

## Presentation identity

With fixed horizon \(H\) and terminal-state padding:
$$
\operatorname{AUC}_{H}(\pi)
=
(H+1)U(S_\pi)-T(\pi).
$$
For equal terminal utility, minimizing acquisition time and maximizing padded prefix AUC induce the same ranking. This identity supplies an independent exact evaluator for trace replay and exhaustive small-instance testing.

# Appendix D: Symbol and Implementation Term Mapping

| 论文符号 | 代码术语 / 配置项 | 说明 |
|---|---|---|
| \(c\) | claim | 原始声明 |
| \(\mathcal R\) | reports | 与 claim 相关的原始报道 |
| \(\mathcal A=\{a_i\}\) | claim_atoms | 原子命题集合 |
| \(q_i\) | query_rendering | atom retrieval query |
| \(e_j\) | candidate / chunk | canonical evidence unit |
| \(\mathcal C\) | atom_union_pool | upstream-deduplicated pool；adapter 验证 unique candidate UID 并排序 |
| \(M(e_j,a_i)\) | evidence_map | relation/directness/confidence/key-span row |
| \(V_{ij}\) | valid-pair projection | pair-level valid gate |
| \(q_{ij}\) | pair_coverage_levels | invalid/partial/direct ordinal level |
| \(\mathbf q_j\) | ordinal_quality_vector | candidate 的 multi-atom coverage vector |
| \(w_i\) | atom_weights | positive quantized atom weight |
| \(c_j\) | token_cost | additive evidence cost |
| \(K_{\min}\) | k_min | soft rendering floor |
| \(K_{\max}\) | k_max | selected-slate count upper bound |
| \(B\) | token_budget | optional additive token upper bound |
| \(\mathbf x\) | terminal_ordinal_state | componentwise-max ordinal state |
| \(U\) | terminal_ordinal_coverage_units | terminal graded coverage |
| \(T\) | core_weighted_coverage_acquisition_time | weighted acquisition time |
| \(D[k,\mathbf x,b]\) | exact DP cell | count/state/cost cell |
| \(\pi^\star\) | coverage_core_keys | exact positive-gain core |
| \(K_{\mathrm{core}}\) | k_core | core evidence count |
| \(\operatorname{Fill}\) | ZERO_GAIN_FILL | deterministic rendering fill |
| \(\pi_{\mathrm{sel}}\) | display_ordered_keys | selected ordered slate |
| \(K_{\mathrm{sel}}\) | k_sel | selected evidence count |
| \(K_{\mathrm{final}}\) | final prompt manifest | verifier-visible evidence count |
| \(\operatorname{AUC}_{H}\) | core_padded_prefix_auc | fixed-horizon padded prefix AUC |
| — | selected_set_fingerprint | strict same-set identity |
| — | solver_objective_tuple | \((-U,T,L,C,\mathrm{keys})\) |
| — | baces_sequence_trace_v0_3 | canonical trace schema |
| — | baces_exact_ordered_state_dp_v0_3 | solver version |
| — | baces_lexicographic_early_coverage_v0_3 | selection policy |
| \(z^\ast\) | label_choice_token | gold label 的 single-token choice |
| \(\mathcal L_{\mathrm{verifier}}\) | label_token_trainer CE loss | verifier training objective |
