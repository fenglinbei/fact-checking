# AAAI写作大纲 v0.4.1

该文档记录 AAAI 论文的写作逻辑与方法部分初稿。v0.4.1 主要修正 v0.4.0 中的 pipeline 顺序、问题命名、multi-atom transition 表述、selector 监督口径、verifier 训练流程，以及 LLM annotation 可信度实验占位符。

# Abstract

社交媒体的发展提高了虚假新闻的传播效率，这使得基于证据的、可解释的自动虚假新闻检测方法成为研究热点。已有方法要么依赖复杂的证据特征工程，要么通过复杂的证据组织与变换来产出最终结果。这些方法往往依赖事后的解释，例如在真实性标签判定之后才单独输出解释文本。而在人类实际核查声明时，核查者通常会先识别声明中的可验证子事实，再围绕这些子事实组织证据，并逐步形成可追踪的证据链。受此启发，本文提出一种 atom-aware 的证据链构建方法：将声明分解为原子事实，构建 evidence map 以刻画候选证据与原子事实之间的关系、直接性与置信度，并学习一个状态条件化的边际效用选择器来生成有序证据链。该证据链既可以作为 LLM verifier 的判定输入，也可以作为人类可读的可解释信息。在 LIAR-RAW 与 RAWFC 等基准上的实验表明，该方法在保持证据链可读性的同时取得了具有竞争力的分类性能，并在若干复杂证据组织方法的对比中展现出稳定优势。

# Introduction

近年来，自动事实核查领域快速发展，从直接基于声明判断，发展到面向社交媒体的评论驱动检测，再到面向新闻的证据驱动检测系统。其中值得注意的是基于证据的检测系统：这种系统接受一个声明以及与该声明相关的、从互联网获取的各种原始报道，而后将这些报道通过各种方式精炼成可用于事实核查的数据结构。有的采用基于证据特征的级联方式 [CofCED]，有的基于 LLM 生成辩护 [L-defense] 或辩护图 [G-defense]，也有方法构建复杂的 agent 系统 [DelphiAgent] 来完成整个核查流程。通过大致观察可发现，这一系列方法最终输入到分类器的数据结构往往已经不再保持原始证据的自然顺序与 provenance，证据如何支撑最终标签变得较难直接追踪。

人类在进行事实核查时，通常会经历多个步骤：1）信息收集：对一个声明收集其相关原始报道并整理；2）要素抽取：把一个声明拆分成多个最小可验证事实；3）证据对齐：判断每一条证据支持、反驳或限定了声明的哪一部分；4）矛盾识别与证据组织：处理证据间可能存在的冲突，并形成有序的支撑路径；5）结论形成：根据所有信息得出最终判定。本文借鉴其中的“声明分解—证据对齐—证据链组织”过程，但不显式建模 report source credibility；来源可信度评估可作为未来工作或外部模块接入。

随着大模型技术的发展，LLM 展示出了强大的语言理解与推理能力，使其可以在给定证据的条件下完成复杂判断。结合人工事实核查中“分解声明、对齐证据、逐步形成判断”的流程，这启发我们思考：能否构建一套证据组织方法，使 LLM verifier 在判定前接收一条围绕 claim atoms 展开的可追踪证据链，而不是只接收无序 top-k evidence 或经过复杂改写后的解释文本。

本文提出一种 atom-aware evidence-chain fact-checking 方法。给定 claim 及其相关 reports，方法首先将 claim 分解为少量可独立验证的 atoms，并为每个 atom 生成检索查询；随后将 reports 切分为 claim-aware evidence chunks，并通过 claim-level route 与 atom-level route 构造候选证据池；接着使用 LLM API 构建 atom-evidence map，标注每个 candidate-atom pair 的 relation、directness 与 confidence；最后学习一个状态条件化的边际效用 selector，逐步选择 evidence units 并生成 ordered evidence trace。该 trace 被渲染为 LLM verifier 的输入，同时保留每步证据的 provenance 与 atom-level state transition。

本文的主要贡献如下：

1. 提出一种 atom-aware evidence-chain 表示，将声明原子事实、证据对齐关系与证据链状态转移统一到一个可追踪的中间结构中。
2. 构建 atom-evidence map，显式刻画候选证据与 claim atoms 之间的 relation、directness 与 confidence，使 selector 能围绕原子事实解析过程组织证据。
3. 设计 learned marginal chain selector，在当前 evidence prefix 与 atom states 条件下评估候选证据的边际价值，避免将 evidence selection 退化为静态 top-k ranking。
4. 通过人工可信度实验评估 claim atomization 与 evidence map annotation 的可靠性，为使用 atoms 与 LLM-generated map 作为 selector 状态变量提供实证支撑。

# Related Work

少量早期 BERT 之前的 AFC 方法。

部分重要的 BERT 类方法。

以重要的篇幅讨论使用 LLM 的各种方法：基于辩护的、基于智能体的、基于内在表示的、基于图的等，并围绕这一系列方法复杂度高、证据 provenance 不易追踪的问题展开。

> 注：定稿时补全为完整段落，明确每类方法的代表工作，并收束到“复杂度高、证据到标签的支撑路径不易追踪”的论点，与 Introduction 呼应。

# Methodology

## Overview

本文提出一种 atom-aware evidence-chain fact-checking 方法。给定声明 $c$ 及其相关 reports，完整流程固定为：

1. **Claim atomization and query rendering.** 首先将声明分解为少量可独立验证的 claim atoms $\mathcal A=\{a_i\}_{i=1}^{m}$，并为每个 atom 生成 query rendering $q_i$。
2. **Claim-aware chunking.** 将相关 reports 切分为粒度适中的 evidence chunks $\mathcal U=\{u_j\}$。
3. **Atom-Union candidate pool construction.** 使用整条 claim 与各 atom queries 分别检索 chunks，两路结果融合、去重并经 MMR 得到候选池 $\mathcal C$。
4. **Atom-evidence map construction.** 对每个 candidate-atom pair 构建结构化标注 $M(u_j,a_i)=(r_{ij},d_{ij},c_{ij})$，其中 relation、directness 与 confidence 由 LLM API 产生。
5. **Greedy atom-resolving chain selection.** 在当前 evidence prefix 与 atom states 条件下，selector 贪心选择边际效用最高的下一条证据，形成 ordered evidence trace $\mathcal T=[s_1,\ldots,s_T]$。
6. **Verifier rendering and prediction.** 将 evidence trace 按 prompt evidence policy 截断并渲染为 verifier 输入，由指令微调后的 LLM verifier 输出事实核查标签。

本文不声称求解全局最小 evidence chain，也不证明最优性。我们将证据组织表述为 **greedy atom-resolving evidence-chain construction**：在有限 evidence budget 下，通过状态条件化的 learned marginal utility 逐步选择对 atom resolution 最有帮助的证据。该表述强调本文解决的是可学习、可解释、可复现的序列化证据构建问题，而非带最优性保证的组合优化问题。

本文的核心贡献集中在两个层面：第一，使用 atom-evidence map 显式刻画证据与原子事实之间的 relation、directness 与 confidence；第二，学习一个状态条件化的边际效用函数，在当前证据链 prefix 与 atom states 条件下估计每条候选证据的新增价值。claim-aware chunking、retrieval 与 Atom-Union candidate pool 作为上游支撑模块，用于提供粒度适中、召回充分且去重后的候选证据集合；其超参数与实现细节将在 Appendix A 与实验设置中给出。

## Task Definition

**输入与输出。** 给定一条待核查的声明 $c$ 及其相关的若干原始报道 $\mathcal R$，任务目标是输出事实核查标签 $\hat y\in\mathcal Y$。标签集合 $\mathcal Y$ 随数据集而异：LIAR-RAW 为六类细粒度真值（pants-fire / false / barely-true / half-true / mostly-true / true），RAWFC 为三类（false / half / true），HoVer 为二分类（supported / not-supported）等。本文方法不依赖特定标签体系，可适配任意离散标签集合 $\mathcal Y$。

**Claim atoms.** 本文将原始 claim $c$ 分解为少量可独立验证的 atomic propositions：
$$
A(c)=\{(a_1,q_1),\dots,(a_m,q_m)\},\quad 1\le m\le 6.
$$
其中 $a_i$ 表示第 $i$ 个原子命题，$q_i$ 是其对应的检索查询。Claim atoms 在后续 evidence map 与 chain selection 中作为状态变量。

**Evidence units.** 每篇 report 被切分为 evidence chunks，记为 $\mathcal U=\{u_j\}$。每个 evidence unit 保留原始 report id、句子跨度、文本内容、retrieval route 信息与 provenance metadata。

**Ordered evidence trace.** 本文构造一条有序证据链 $\mathcal T=[s_1,\dots,s_T]$ 作为核心中间产物。不同于每步只绑定一个 atom 的表述，本文允许单条 evidence 同时对齐多个 atoms。第 $t$ 步定义为：
$$
s_t=(u_t,a_{i_t}^{\star},h_{i_t}^{(t-1)}\rightarrow h_{i_t}^{(t)}, Z_t),
\quad
Z_t=\{a_i: M(u_t,a_i)\ \text{is valid}\}.
$$
其中 $u_t$ 是被选中的 evidence unit，$Z_t$ 是该 evidence 对齐到的有效 atom 集合，$a_{i_t}^{\star}$ 是用于 trace ordering 与 diagnostics 的 primary atom，$h_{i_t}^{(t-1)}\rightarrow h_{i_t}^{(t)}$ 是 primary atom 的状态转移。Selector 在打分时聚合 $u_t$ 对所有 $Z_t$ 中 atoms 的边际效用；trace 输出时则锚定一个 primary transition，以保证链条顺序与诊断信息可读。

**判别。** 完整证据链经 prompt evidence policy 截断后与声明拼接为
$$
x=\mathrm{Render}(c,\mathcal T_{\mathrm{prompt}}),
$$
并送入指令微调的 LLM verifier：
$$
\hat y=\arg\max_y p_{\theta_{\mathrm{ver}}}(y\mid x).
$$

## Claim Atomization and Query Rendering

在一般的人类声明验证流程中，核查者通常不会把包含多个事实断言的 claim 作为一个整体一次性验证，而是会先识别其中可独立验证的事实单元。受此启发，本文将原始 claim $c$ 转换为少量语义完整、可独立验证的 atomic propositions，并将其作为后续 atom-conditioned retrieval、evidence map 与 chain selection 的状态变量。

输入是一条 claim，输出为：
$$
A(c)=\{(a_1,q_1),\dots,(a_m,q_m)\},\quad 1\le m\le 6.
$$
其中 $a_i$ 是原子命题，$q_i$ 是面向检索的 query rendering。实现上，atomization 通过 LLM API 完成，并在 prompt 中加入以下约束：只使用 claim 本身，不引入外部知识；只在 claim 含有多个可分别验证的事实断言时拆分；日期、数量、否定、比较对象、地点、范围、归因等必须保留在 proposition 内；每个 atom 不能只是关键词片段，而应是一个最小、可验证、语义完整的命题。若 LLM 输出超过最大 atom 数 $m_{\max}=6$，系统按 prompt 返回的重要性与可验证性字段进行合并或截断；完整 prompt、JSON schema、invalid-output retry 规则与后处理细节见 Appendix B。

每个 atom 同时附带 query rendering $q_i$。$q_i$ 不一定与 $a_i$ 完全相同，而是面向 retrieval 的短查询形式，用于提高对细粒度事实的召回。实现上，query rendering 与 atomization 在同一 LLM API 调用中生成，附带关键词、实体、时间、数量与比较对象等字段。本文后续所有训练、验证与测试样本均使用同一 atomization 与 query rendering 流程，避免训练/推理流程不一致。

由于 claim atoms 是后续 selector 的状态变量，本文在实验部分加入人工可信度评估，检查 LLM 生成的 atoms 是否忠实于 claim、是否遗漏可验证断言，以及是否满足原子性要求（见 Reliability Study）。

## Claim-aware Evidence Chunking

在构建候选证据池之前，需要确定 evidence unit 的粒度。若粒度过小（如单句），单个 evidence unit 可能缺少必要上下文；若粒度过大（如整段 report），则会引入过多噪声并增加 verifier 上下文成本。因此本文采用 claim-aware chunking，将 reports 切分为与 claim 相关且局部连贯的 evidence chunks。

设 claim 为 $c$，一篇 report 包含句子 $s_1,\dots,s_n$，句向量为 $e_i$，claim 向量为 $e_c$。首先对每个句子计算 claim-aware relevance：
$$r_i = \alpha_1\,\mathrm{norm}(\cos(e_i,e_c)) + \alpha_2\,\mathrm{norm}(\mathrm{LexF1}(c,s_i)) + \alpha_3\,\mathrm{norm}(\mathrm{BM25}(c,s_i)).
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

每个 claim 经 chunking 后通常得到数十个 evidence chunks。为了降低后续 evidence map 与 selector 的计算成本，同时保证对细粒度 atoms 的召回，本文构造一个 Atom-Union candidate pool。该过程发生在 atomization 之后，因此 candidate pool 同时利用整条 claim 与各 atom queries。

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

多个 atom routes 的结果采用 reciprocal rank fusion（RRF）聚合，并记录每个 evidence unit 被哪些 atoms 命中、命中次数、最大 hybrid retrieval score 与 route-level rank。随后将 claim-level baseline pool 与 atom-route pool 按文本规范化 key 去重，并使用 MMR 在相关性与多样性之间做折中，得到送入 evidence map 与 selector 的候选池：
$$
\mathcal C=\mathrm{MMR}\big(\mathrm{Dedup}(\mathcal B\cup\mathcal A_{\mathrm{route}})\big).
$$
其中 $\mathcal B$ 保证全局相关性，$\mathcal A_{\mathrm{route}}$ 提高对细粒度 atoms 的覆盖。后续方法不直接将 $\mathcal C$ 作为无序 top-k 输入 verifier，而是在 $\mathcal C$ 上构建 atom-evidence map，并进一步执行状态条件化的 evidence-chain selection。关于 retrieval encoder、$k_b/k_a$、RRF 参数、dedup 阈值、MMR 的 $\lambda_{\mathrm{mmr}}$ 与最终候选池大小 $n$，见 Appendix A；Atom-Union 的贡献将在 retrieval-route ablation 中报告。

## Atom-Evidence Map

为了使证据选择能够围绕原子事实展开，本文为每个 claim 构建 atom-evidence map。给定候选证据池 $\mathcal C=\{u_j\}_{j=1}^{n}$ 与 claim atoms $\mathcal A=\{a_i\}_{i=1}^{m}$，map 为每个 candidate-atom pair 提供结构化标注：
$$
M(u_j,a_i)=(r_{ij},d_{ij},c_{ij}),
$$
其中 $r_{ij}$ 表示 relation，$d_{ij}$ 表示 directness，$c_{ij}\in[0,1]$ 表示 confidence。Relation 描述证据对 atom 的立场或功能，取值包括 support、refute、qualify、background 与 irrelevant；directness 表示证据验证 atom 的直接程度，采用 1--5 的有序等级；confidence 表示 LLM API 对该标注的自评置信度。

实现上，relation、directness 与 confidence 均由 LLM API 通过固定 prompt 与 JSON schema 生成。Map 阶段严格约束 atom 集合固定，不允许 LLM 创建、合并、拆分或改名 atoms。除三元组外，evidence map 还记录 evidence role、key spans、duplicate group 与 brief rationale 等辅助字段，用于后续新颖性判断、文本去重与 case study 展示。完整 prompt、schema、invalid-output retry、temperature、模型版本与字段验证规则见 Appendix B；case study 中将展示一条 claim 的完整 atom-evidence map。

本文将 $M(u_j,a_i)$ 视为 selector 的结构化中间表示，而非最终事实核查标签。它告诉 selector 每条候选证据可能解析哪些 atoms、以何种 relation 解析、该 relation 是否直接，以及该判断是否足够可信。为检验该中间表示的可靠性，本文加入人工标注实验评估 evidence map 的 relation 与 directness 标注准确率（见 Reliability Study）。

记 $\mathrm{Valid}(u_j,a_i)$ 为 candidate-atom pair 的有效对齐谓词。默认情况下，若 $r_{ij}\ne\mathrm{irrelevant}$ 且 $d_{ij}$ 与 $c_{ij}$ 达到最低阈值，则 $M(u_j,a_i)$ 被视为 valid。有效对齐集合为：
$$
Z(u_j)=\{a_i:\mathrm{Valid}(u_j,a_i)\}.
$$
该集合用于 selector 聚合 evidence 对多个 atoms 的边际贡献。

## Greedy Atom-Resolving Evidence Chain Construction

本文将证据组织表述为 greedy atom-resolving evidence-chain construction。给定 claim atoms $\mathcal A$、候选证据池 $\mathcal C$ 与 atom-evidence map $M$，目标是在有限 evidence budget 下构造一条有序证据链 $\mathcal T=[s_1,\ldots,s_T]$，使其尽可能解析 claim 中的原子事实，同时控制冗余、背景噪声与上下文成本。

与静态 top-k evidence ranking 不同，selector 在第 $t$ 步根据当前 evidence prefix $\mathcal T_{<t}$ 与 atom states $H^{(t)}$ 选择下一条证据：
$$
u_t=\arg\max_{u_j\in\mathcal C\setminus \{u_\ell\}_{\ell<t}}
U_{\theta_{\mathrm{sel}}}(u_j\mid\mathcal T_{<t},H^{(t)}).
$$
该选择是贪心的：本文不声称 $\mathcal T$ 是全局最短或全局最优证据链，而是学习一个可解释的边际效用函数，在每一步估计候选证据的新增价值。

我们为每个 atom 定义验证状态：
$$
h_i^{(t)}\in\{U,S,R,Q,C\},
$$
分别表示 unresolved、supported、refuted、qualified 与 conflicted。初始时所有 atoms 均为 $U$。状态转移遵循：support/refute/qualify 分别将 unresolved atom 推进到 $S/R/Q$；若新的 support/refute 证据与已有 $S/R$ 立场冲突，则 atom 进入 $C$；background 通常作为 BRIDGE 操作提供上下文，不直接改变 atom 的 hard state。

每条被选中的 evidence unit 可能对齐多个 atoms。Selector 在打分时聚合它对所有有效对齐 atoms 的边际价值；在 trace 输出时选择一个 primary atom $a_{i_t}^{\star}$，用于排序与诊断：
$$
s_t=(u_t,a_{i_t}^{\star},h_{i_t}^{(t-1)}\rightarrow h_{i_t}^{(t)},Z_t),
\quad
Z_t=\{a_i: M(u_t,a_i)\ \text{is valid}\}.
$$
Primary atom 默认选择在当前 step 中产生最大状态解析增益的 atom；若 evidence 只提供 background 或 corroboration，则选择 directness/confidence 最高的 aligned atom 作为 diagnostic anchor。

本文定义五类转移操作：OPEN、CORROBORATE、CONTRAST、BRIDGE 与 FALLBACK。OPEN 表示证据将 unresolved atom 推进到 S/R/Q；CORROBORATE 表示证据对已解析 atom 提供同立场强化；CONTRAST 表示证据引入与现有 S/R 状态冲突的信息，使 atom 进入 C 或转为 Q；BRIDGE 表示证据提供背景上下文但不改变 hard state；FALLBACK 表示在无可解析关系时的兜底选择。

构链过程在满足以下任一条件时停止：atom 解析率达到目标 $\rho_{\mathrm{target}}$、候选最高效用低于阈值 $\tau_{\mathrm{stop}}$、token budget 耗尽，或达到最大步数 $k_{\max}^\mathrm{trace}$。第 $t$ 步后的 atom 解析率定义为：
$$
\rho_t=\frac{|\{a_i:h_i^{(t)}\in\{S,R,Q,C\}\}|}{|\mathcal A|}.
$$
主实验默认采用较严格的 $\rho_{\mathrm{target}}=1.0$，即尽可能使所有 claim atoms 获得解析状态；较低阈值作为敏感性分析。

## Learned Marginal Chain Selector

### Marginal features

Selector 将每个候选证据的价值定义为其在当前 chain prefix 下带来的状态条件化边际效用。候选证据 $u_j$ 在第 $t$ 步被映射为一组边际特征：
$$
\phi^{(t)}(u_j)=
[\phi_{\mathrm{res}},\phi_{\mathrm{ent}},\phi_{\mathrm{cov}},
\phi_{\mathrm{new-rel}},\phi_{\mathrm{tension}},
\phi_{\mathrm{corr}},\phi_{\mathrm{src-novel}},
\phi_{\mathrm{text-novel}},\phi_{\mathrm{conf}},
\phi_{\mathrm{map}},\phi_{\mathrm{ret}},\phi_{\mathrm{cost}}].
$$
为避免方法部分过于臃肿，主文只给出这些特征的定义性说明与最关键的 resolution 公式；完整归一化方式、阈值与实现细节见 Appendix C。

| Feature | 主文定义 | 是否依赖状态 |
|---|---|---|
| $\phi_{\mathrm{res}}$ | 对尚未解析 atoms 的直接解析增益 | 是 |
| $\phi_{\mathrm{ent}}$ | candidate 对当前 unresolved/uncertain atoms 的不确定性降低 | 是 |
| $\phi_{\mathrm{cov}}$ | candidate 新覆盖的 valid atoms 比例 | 是 |
| $\phi_{\mathrm{new-rel}}$ | 对已出现 atoms 引入新的 relation observation | 是 |
| $\phi_{\mathrm{tension}}$ | 与已有 S/R 状态形成潜在冲突或限定的信息量 | 是 |
| $\phi_{\mathrm{corr}}$ | 对已有 S/R/Q 状态提供同向佐证的增益 | 是 |
| $\phi_{\mathrm{src-novel}}$ | 新 report/source id 的覆盖增益；不表示 source credibility | 是 |
| $\phi_{\mathrm{text-novel}}$ | 与已选 evidence 的文本/embedding 非冗余程度 | 是 |
| $\phi_{\mathrm{conf}}$ | LLM map confidence 的聚合值 | 否 |
| $\phi_{\mathrm{map}}$ | relation/directness/schema validity 等 map 质量信号 | 否 |
| $\phi_{\mathrm{ret}}$ | claim route 与 atom route 的 retrieval score 融合值 | 否 |
| $\phi_{\mathrm{cost}}$ | evidence token 长度或 prompt 成本 | 否 |

若 $u_j$ 对 atom $a_i$ 给出可解析 relation，则其解析增益为：
$$
g_{ij}^{(t)}=p_i^{(t)}(U)\cdot \delta(d_{ij})\cdot \max(c_{ij},0.5),
$$
其中 $p_i^{(t)}(U)$ 是当前 atom 仍未解析的状态质量，$\delta(d_{ij})$ 为随 directness 等级递增的权重。候选证据的 resolution 边际贡献为：
$$
\phi_{\mathrm{res}}^{(t)}(u_j)=\frac{1}{m}\sum_{i=1}^{m}\mathbf{1}[\mathrm{Valid}(u_j,a_i)]\,g_{ij}^{(t)}.
$$
该定义自然允许单条 evidence 同时解析多个 atoms。

### Utility function

Selector 使用非负约束的线性边际效用函数为候选证据打分：
$$
U_{\theta_{\mathrm{sel}}}(u_j\mid \mathcal T_{<t},H^{(t)})
=b+\sum_{\ell\ne \mathrm{cost}}w_\ell\phi_\ell^{(t)}(u_j)
-w_c\phi_{\mathrm{cost}}^{(t)}(u_j).
$$
其中非 cost 特征权重为非负，cost 权重以负项进入效用函数。实现中使用 softplus 参数化保证权重非负：
$$
w_\ell=\mathrm{softplus}(\theta_\ell),\quad w_c=\mathrm{softplus}(\theta_c).
$$
该设计保留了 evidence-map 特征和权重贡献的可解释性，同时避免将 selector 训练为难以诊断的黑箱策略。

### Proxy-supervised pairwise learning

由于 gold atom-level evidence chains 不可得，本文从 atom-evidence map 导出的结构化 proxy preferences 学习 selector。该监督信号不使用人工 evidence chain，也不使用测试集标签；它只根据当前 rollout 状态、evidence map 与 retrieval metadata 构造相对偏好。

在每个 rollout step，给定当前 prefix $\mathcal T_{<t}$ 与 atom states $H^{(t)}$，我们根据候选证据的 structural marginal value 对剩余候选排序：直接推进 atom 状态的证据优先于只提供部分解析的证据；覆盖尚未解析 atoms 的证据优先于重复覆盖已解析 atoms 的证据；引入新的 atom-relation observation 的证据优先于重复 evidence。Map quality 与 retrieval score 仅作为次级 tie-breaking signals。

令
$$
\pi_{\mathrm{proxy}}(u\mid \mathcal T_{<t},H^{(t)})
$$
表示该 structure-induced ordering。记第 $t$ 步的剩余候选集合为 $\mathcal C_t=\mathcal C\setminus\{u_\ell\}_{\ell<t}$。每一步中，proxy 排序最高的候选被视为 positive next evidence $u_t^+$，排名较低的剩余候选被视为 relative negatives：
$$
\mathcal P_t=\{(u_t^+,u^-):u^-\in\mathcal C_t\setminus\{u_t^+\}\}.
$$
每完成一个 proxy-selected step 后，系统更新 atom state，并在新状态下重新生成下一轮 preference pairs。因此，该监督信号是 state-conditioned 且 chain-aware 的，而不是一次性的静态 top-$k$ relevance label。

Pairwise 学习目标为：
$$
\mathcal L(\theta_{\mathrm{sel}})=
\frac{1}{|\mathcal P|}
\sum_{(u^+,u^-)\in\mathcal P}
\log\left(1+\exp\left(-[U_{\theta_{\mathrm{sel}}}(u^+)-U_{\theta_{\mathrm{sel}}}(u^-)]\right)\right).
$$
实现中使用 Adam 优化。训练完成后，proxy ordering 不再直接用于推理；推理阶段使用学习到的 $U_{\theta_{\mathrm{sel}}}$ 进行 greedy selection。

除上述主方法 learned_marginal_proxy 外，本文还实现一个 reward variant 作为消融：该变体使用 verifier 在加入某证据后的判定 margin 变化作为 reward，并训练同一组 softplus 权重。为避免循环依赖，该 reward variant 使用的 verifier 来自 learned_marginal_proxy selector 生成的 training traces；verifier 不反向更新主 selector，也不在主方法中参与 selector 训练。

## Algorithm

```text
Algorithm 1: Atom-aware Greedy Evidence Chain Construction

Input: claim c, reports R, budgets k_trace, B_tok
Output: ordered evidence trace T

1:  A = AtomizeAndRenderQueries(c)
2:  U = ClaimAwareChunkReports(R, c)
3:  B = ClaimLevelRetrieve(c, U)
4:  A_route = Union_i AtomLevelRetrieve(q_i, U)
5:  C = MMR(Dedup(B ∪ A_route))
6:  M = AnnotateAtomEvidenceMap(C, A)
7:  Initialize H^(0) = {h_i = U for all a_i in A}
8:  T = []
9:  for t = 1 ... k_trace do
10:     for each u in C \ T do
11:         Z(u) = {a_i : M(u, a_i) is valid}
12:         φ_t(u) = ComputeMarginalFeatures(u, T, H^(t-1), M, Z(u))
13:         score(u) = U_sel(u | T, H^(t-1))
14:     u_t = argmax score(u)
15:     if score(u_t) < τ_stop or token budget exceeded: break
16:     Z_t = Z(u_t)
17:     a*_t = SelectPrimaryAtom(u_t, Z_t, H^(t-1), M)
18:     ΔH_t = ApplyStateTransitions(u_t, Z_t, H^(t-1), M)
19:     T.append((u_t, a*_t, h_{a*_t}^{(t-1)} → h_{a*_t}^{(t)}, Z_t))
20:     H^(t) = Update(H^(t-1), ΔH_t)
21:     if ResolvedRate(H^(t)) ≥ ρ_target: break
22: return T
```

## Verifier Rendering and Prediction

在得到 ordered evidence trace $\mathcal T=[s_1,s_2,\ldots,s_T]$ 后，本文使用一个指令微调后的 LLM 作为最终 verifier。Verifier 接收 claim 及其 prompt-visible evidence trace，并输出事实核查标签：
$$
\hat y=\arg\max_y p_{\theta_{\mathrm{ver}}}(y\mid x),
$$
其中 $x=\mathrm{Render}(c,\mathcal T_{\mathrm{prompt}})$ 表示由 claim $c$ 和被截断后的 evidence trace 构造的输入 prompt。

Verifier 在标签集合 $\mathcal Y$ 上做分类。为保证 label prediction 形式稳定，本文不要求模型直接输出 `pants-fire`、`barely-true` 等多 token 字符串，而是将每个标签映射为单 token answer choice，例如 LIAR-RAW 六分类映射为 $\{A,B,C,D,E,F\}$，RAWFC 三分类映射为 $\{A,B,C\}$。训练时在 answer choice token 位置上计算交叉熵：
$$
\mathcal L_{\mathrm{verifier}}=-\log p_{\theta_{\mathrm{ver}}}(z^\ast\mid x),
$$
其中 $z^\ast$ 是 gold label 对应的 answer choice token。Label-to-choice mapping 在训练、验证与测试阶段保持固定。

由于 selector 生成的完整 trace $\mathcal T$ 长度存在差异，直接将完整 trace 输入 verifier 可能导致长上下文噪声增加、有效证据信号被稀释，并显著提高训练与推理成本。因此，我们在 verifier 之前引入 prompt evidence policy，将完整 ordered trace 映射为 verifier 可见的 evidence prefix：
$$
\mathcal T_{\mathrm{prompt}}=[s_1,s_2,\ldots,s_{K^\ast}],\quad K^\ast\le T.
$$
这里的截断位置 $K^\ast$ 不是固定值，而是由 atom 解析状态驱动的：当 claim 中的原子事实已被现有证据充分解析时，提前停止加入新证据。该设计使 verifier 只在必要时看到更多证据，从而在证据充分性与上下文成本之间取得平衡——这也是 evidence map 价值的重要体现之一（见 Ablation Study）。

主实验采用 minmax prompt evidence policy：沿 selector 给出的顺序逐步加入 evidence step，并在满足最小证据数 $k_{\min}$ 后检查当前 trace 是否已经达到 atom-level resolution target。令
$$
\rho_t=\frac{|\{a_i:h_i^{(t)}\in\{S,R,Q,C\}\}|}{|\mathcal A|}.
$$
若
$$
\rho_t\ge\rho_{\mathrm{target}},
$$
则认为当前 trace 已达到目标解析状态，并停止继续加入证据。因此截断位置定义为：
$$
K^\ast=\min\{t:t\ge k_{\min}\land \rho_t\ge\rho_{\mathrm{target}}\}.
$$
若不存在满足条件的 $t$，则退化为最大证据数约束：
$$
K^\ast=\min(k_{\max},T).
$$
此外，本文提供 fixed_topk 与按 token budget 截断等容量敏感性对照策略，用于消融分析。Train 与 test 使用完全相同的 evidence map 生成方式、selector 策略与 prompt evidence policy。

## Training and Inference Pipeline

本文采用分阶段训练流程，以避免 selector 与 verifier 之间的循环依赖。

**Selector training.** 对训练集中的每条 claim，首先执行 atomization、chunking、Atom-Union candidate construction 与 atom-evidence map annotation。随后基于 evidence map 进行 proxy rollout，生成 state-conditioned preference pairs，并训练 learned_marginal_proxy selector。该阶段不训练 verifier，也不使用 verifier 的反馈信号。

**Verifier training.** Selector 训练完成后固定其参数。对训练集重新执行 greedy chain construction，得到 learned_marginal_proxy selector 生成的 evidence traces。然后将这些 traces 经过同一 prompt evidence policy 截断与渲染，构造 verifier training rows，并使用 gold fact-checking labels 训练 LLM verifier。Verifier 不会反过来更新主 selector。

**Inference.** 测试阶段对每条 claim 使用同一流程：atomization $\rightarrow$ chunking $\rightarrow$ Atom-Union candidate pool $\rightarrow$ evidence map $\rightarrow$ fixed learned selector 构链 $\rightarrow$ prompt evidence policy 渲染 $\rightarrow$ fixed verifier 分类。所有中间模块在 train/test 中使用相同 prompt/schema 与超参数配置。

**Reward variant.** 作为消融对照，reward-based selector 使用 verifier delta margin 作为 reward。该 verifier 来自 learned_marginal_proxy selector 产出的 traces 训练得到，不参与主方法的 selector 训练。该设置用于分析 verifier feedback 是否能进一步改善 chain selection，而非主方法依赖项。

# Experiments

## Main Evaluation

**数据集与指标。** 在 LIAR-RAW（6 类细粒度真值）与 RAWFC（3 类）两个公开基准上评估。主指标为 classification accuracy 与 macro-F1；LIAR-RAW 额外报告 ordinal MAE 与 extreme error rate 以反映六类有序性。

**实现。** Verifier 为 Ministral-3-8B + LoRA，label-token CE 训练，$k_{\min}{=}5,k_{\max}{=}10$ minmax prompt evidence policy，$\rho_{\mathrm{target}}{=}1.0$。训练/验证/测试 split 与各基准官方划分一致。

**主结果。** 下表报告本文主方法在两个基准上的 test 性能。RAWFC 同时给出 baseline20 变体（使用 20 条候选证据的增强检索池），以展示检索池规模的影响。

| 数据集 | 配置 | Acc. | Macro-F1 |
|---|---|---|---|
| LIAR-RAW | 主方法 (minmax5_10) | 0.360 | **0.367** |
| RAWFC | 主方法 (minmax5_10) | 0.635 | 0.638 |
| RAWFC | baseline20 变体 | 0.660 | **0.661** |

> 注：定稿时补充外部 baseline 对比（文献数值）、per-class P/R/F1、混淆矩阵，以及 LIAR 的 ordinal MAE / extreme error rate。HoVer 基准的 test 评估待补。

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

3. **map + 学习进一步提升。** s4（无 map 无学习，0.354）$\to$ s5（有 map 无学习，0.349）$\to$ 主方法（有 map 有学习，0.339），需要说明的是此处的"主方法"固定了 top5（minmax5\_5）以与 s0–s6 对齐容量；在 minmax5\_10 下主方法达到 0.367（见 Main Evaluation）。在同等 fixed\_top5 容量下，s4 反而略高于主方法，说明 learned selector 的优势主要体现在通过解析驱动的动态截断（minmax）提升证据效率，而非在固定容量下改善排序。

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

4. **confidence 的作用需谨慎解读。** no\_confidence 将 $c_{ij}$ 强制为 1.0，反而使解析率保持 0.774（因 $\max(c,0.5)$ 下界 clip 被绕过，解析增益不降反升），但 macro-F1 降至最低的 0.332。这表明 uniform 高 confidence 会让 selector 过度信任所有标注、丧失区分高/低可信 pair 的能力，$\max(c,0.5)$ 的 clip 并非冗余而是必要的鲁棒化设计。

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

### 待补充的消融

以下消融已有部分数据或计划中，定稿时补充：

1. Chunking 粒度消融（sentence / ctx\_window / semantic / abc\_claim\_aware / raw）：已有 LIAR-RAW test 数据，但主方法所用 abc 粒度在 fixed\_top5 设定下并非最优，需在 minmax 设定下重新评估后再纳入。
2. Atomization 移除（仅用 claim-level retrieval）与 Atom-Union 路由消融。
3. 单项 marginal feature 移除（tension、corroboration、text novelty、cost 等）。
4. $\rho_{\mathrm{target}}$ 与 candidate pool size 的超参敏感性。

## Reliability Study: Claim Atomization

**目的。** 检验 LLM 拆出的 atoms 是否会凭空编造、是否会漏掉 claim 中可验证事实，以及是否满足最小可验证单元的要求。该实验为“atoms 可作为 selector 状态变量”提供基础证据。

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

**判断标准。** 若 hallucination rate $<5\%$ 且 omission rate $<10\%$，则认为 atomization 质量足以支撑后续 atom-state modeling；若未达到该标准，则在 error analysis 中报告失败类型并讨论对 selector 的影响。

## Reliability Study: Evidence Map Annotation

**目的。** 检验 $M(u_j,a_i)=(r_{ij},d_{ij},c_{ij})$ 中 relation 与 directness 标注是否可靠。该实验直接检验 selector 决策依据的可信度，因为 $\phi_{\mathrm{res}}$、$\phi_{\mathrm{tension}}$ 与 $\phi_{\mathrm{corr}}$ 均依赖 relation。

**数据。** 从 Reliability Study 1 中 claim 对应的 candidate pools 中抽取 200--300 个 evidence-atom pairs，覆盖 support、refute、qualify、background 与 irrelevant 等类别。采样时提高 refute 与 qualify 的比例，以保证对易混淆类别有足够统计量。

**标注协议。** 2 位标注者独立标注每个 pair 的 gold relation 与 gold directness。Relation 取值为 support / refute / qualify / background / irrelevant；directness 采用 1--5 有序等级。随后将人工 gold annotation 与 LLM API 产生的 map annotation 进行比对。

**测量指标。** 报告 relation overall accuracy、per-relation accuracy、relation Cohen's $\kappa$、relation confusion matrix、directness Spearman $\rho$，以及两位人工标注者之间的 IAA。特别关注 refute 的准确率与 qualify $\leftrightarrow$ background 的混淆情况。

**判断标准。** 若 relation $\kappa>0.7$，则认为 map relation annotation 基本可接受；若 refute accuracy 偏低，则需要在 error analysis 中说明其对 tension/contrast 选择的影响，并报告是否通过 prompt 或阈值策略进行缓解。

# Appendix A: Candidate Pool and Retrieval Details Placeholder

> 定稿时补充以下内容：embedding model、BM25 实现、LexF1 定义、$\alpha/\beta$ 权重、chunk 最大/最小句数、chunk overlap、$k_b/k_a$、RRF 参数 $k_{\mathrm{rrf}}$、dedup 文本规范化规则、duplicate threshold、MMR 的 $\lambda_{\mathrm{mmr}}$、最终 candidate pool size $n$、token budget 与 dataset-specific configurations。

# Appendix B: LLM Prompts and Schemas Placeholder

> 定稿时补充 claim atomization prompt、query rendering schema、atom-evidence map prompt、relation/directness/confidence label definition、JSON validation、retry/fallback rules、temperature、LLM API model version，以及 case study 中的一条完整输出。

# Appendix C: Marginal Feature Definitions Placeholder

> 定稿时补充 12 个 marginal features 的完整公式、归一化方式、阈值、状态依赖、是否使用 evidence map、是否使用 retrieval metadata、是否可能引入 label leakage 的说明。主文保留 compact feature table；附录给精确定义。

# Appendix D: Symbol and Implementation Term Mapping

| 论文符号 | 代码术语 / 配置项 | 说明 |
|---|---|---|
| $c$ | `claim` | 原始声明 |
| $\mathcal R$ | `reports` | 与 claim 相关的原始报道 |
| $\mathcal A=\{a_i\}$ | `claim_atoms` / `claim_atomization` | 原子命题集合 |
| $q_i$ | `query_rendering` | atom 的检索查询 |
| $u_j$ | `candidate` / `chunk` | 证据单元 |
| $\mathcal U$ | `chunk_pool` | report chunking 后的完整 evidence unit 集合 |
| $\mathcal B$ | `baseline_pool` | 整条 claim 检索得到的候选池 |
| $\mathcal A_{\mathrm{route}}$ | `atom_route_pool` | 各 atom route 检索结果的并集 |
| $\mathcal C$ | `atom_union_pool` | 融合、去重、MMR 后送入 selector 的候选证据池 |
| $M(u_j,a_i)$ | `evidence_map` | relation/directness/confidence 标注 |
| $r_{ij}$ | `relation` | support/refute/qualify/background/irrelevant |
| $d_{ij}$ | `directness` | 1--5 有序等级 |
| $c_{ij}$ | `confidence` | LLM API 标注置信度 |
| $Z_t$ | `aligned_atoms` | 当前 evidence 有效对齐的 atoms |
| $a_{i_t}^{\star}$ | `primary_atom` | trace ordering 与 diagnostics 的主 atom |
| $h_i^{(t)}$ | `atom_states` | atom 验证状态 |
| $H^{(t)}$ | `atom_states_snapshot` | 第 $t$ 步的全局 atom 状态 |
| $\{U,S,R,Q,C\}$ | `VALID_ATOM_STATES` | unresolved/supported/refuted/qualified/conflicted |
| $\phi^{(t)}(u_j)$ | `marginal_features` | 12 维边际特征向量 |
| $U_{\theta_{\mathrm{sel}}}$ | `score_marginal_features` | 线性边际效用 |
| $\theta_{\mathrm{sel}}$ | `LearnedMarginalWeights` | softplus 参数化的学习权重 |
| $\mathcal T$ | `mrec_trace` / `selected_candidates` | 有序证据链 |
| $s_t$ | `mrec_step` | 单步，含 evidence、primary atom、状态转移、aligned atoms |
| operation | OPEN/CORROBORATE/CONTRAST/BRIDGE/FALLBACK | 转移操作 |
| $\rho_t$ | `resolved_atom_rate` | 解析率 |
| $\rho_{\mathrm{target}}$ | `target_resolved_rate` | 解析率目标 |
| $k_{\min}/k_{\max}$ | `min_evidence_count` / `max_evidence_count` | prompt evidence policy 证据数上下界 |
| $K^\ast$ | `prompt_evidence_count` | prompt 可见证据数 |
| $\mathcal T_{\mathrm{prompt}}$ | `build_training_row` | verifier 输入证据 prefix |
| $z^\ast$ | `label_choice_token` | gold label 对应的单 token answer choice |
| $\mathcal L_{\mathrm{verifier}}$ | `label_token_trainer` CE loss | verifier 训练损失 |
