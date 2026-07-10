# AAAI写作大纲

该文档记录AAAI论文的写作大致逻辑，以及部分实际原文，作为论文的初稿中文版，下面的章节都按正式论文的章节排列。

# Abstract

社交媒体的发展提高了虚假新闻的传播效率，这使得基于证据的、可解释的自动虚假新闻检测方法成为研究热点。已有方法要么依赖复杂的证据特征工程，要么通过复杂的证据组织与变换来产出最终结果。这些方法往往依赖事后的解释，例如在真实性标签判定之后才单独输出解释文本。而在人类实际检测虚假新闻的场景中，核查者通常会围绕声明中的原子级事实组织证据，并逐步形成可追踪的证据链。受此启发，本文提出一种 atom-aware 的证据链构建方法：将声明分解为原子事实，构建 evidence map 以刻画候选证据与原子事实之间的关系，并学习一个状态条件化的边际效用选择器来生成有序证据链。该证据链既可以作为 LLM 的判定输入，也可以作为人类可读的可解释信息。在 LIAR-RAW 与 RAWFC 等基准上的实验表明，该方法在保持证据链可读性的同时取得了具有竞争力的分类性能，并在若干复杂证据组织方法的对比中展现出稳定优势。

# Introduction

近年来，自动事实核查领域快速发展，从直接基于声明判断，发展到面向社交媒体的评论驱动检测，再到面向新闻的证据驱动检测系统。其中值得注意的是基于证据的检测系统：这种系统接受一个声明以及与该声明相关的、从互联网获取的各种原始报道，而后将这些报道通过各种方式精炼成可用于事实核查的数据结构。有的采用基于证据特征的级联方式 [CofCED]，有的基于 LLM 生成辩护 [L-defense] 或辩护图 [G-defense]，又或者构建复杂的 Agent 系统 [DelphiAgent] 来完成整个过程。而这些方法往往需要一套复杂的证据处理流程：如 [CofCED] 使用一套证据特征聚合机制，将完整证据池构建成特征向量供后续判别器分类；[L-defense] 则借助群体智慧思想，利用 LLM 生成对证据的正反辩护，而后再输入后续分类器；[G-defense] 则更进一步，该方法把一条完整声明拆成多个子声明构成图，而后为每个声明都生成正反辩护，最后将图序列化接入分类器。通过大致观察可发现，这一系列方法最终输入到分类器的数据结构都已经不再保持原始证据的自然顺序与 provenance，证据如何支撑最终标签变得较难直接追踪。

人类在进行事实核查时，会经过多个步骤：1）信息收集：对一个声明，收集其所有相关的原始报告并整理；2）要素抽取：把一个声明拆分成多个原子事实，每个原子事实都包含最小的子命题；3）证据组织：判断每一条证据支持了声明的哪一部分，并组织成一条逻辑链；4）来源评估及矛盾识别：检查 report 来源的可信度，以及处理证据中存在的矛盾；5）结论形成：根据所有信息得出最终判定。

随着大模型技术的发展，大模型展示出了强大的语言理解与推理能力，使其可以在给定证据的条件下完成复杂判断。结合人工事实核查中“分解声明、对齐证据、逐步形成判断”的流程，这启发我们思考：能否构建一套证据组织方法，使 LLM verifier 在判定前接收一条围绕 claim atoms 展开的可追踪证据链。

# Related Work

少量早期 BERT 之前的 AFC 方法。

部分重要的 BERT 类方法。

以重要的篇幅讨论使用 LLM 的各种方法：基于辩护的、基于智能体的、基于内在表示的、基于图的等，并围绕这一系列方法复杂度高、证据 provenance 不易追踪的问题展开。

> 注：定稿时补全为完整段落，明确每类方法的代表工作，并收束到“复杂度高、证据到标签的支撑路径不易追踪”的论点，与 Introduction 呼应。

# Methodology

## Overview

本文提出一种 atom-aware evidence-chain fact-checking 方法。方法从原始声明 $c$ 及其相关 reports 出发，先构造候选证据池 $\mathcal C$，再将声明分解为原子级事实 $\mathcal A=\{a_i\}_{i=1}^{m}$，并建立候选证据与 claim atoms 之间的结构化 evidence map。与直接对候选证据做静态 top-k 排序不同，本文将证据组织定义为一个 minimum atom-resolving evidence chain selection 问题：在有限 evidence budget 下，逐步选择能够解析 claim atoms 的证据，形成一条有序证据链 $\mathcal T=[s_1,\ldots,s_T]$。

本文的核心贡献集中在两个层面：第一，使用 atom-evidence map 显式刻画证据与原子事实之间的 relation、directness 与 confidence；第二，学习一个状态条件化的边际效用函数，在当前证据链 prefix 与 atom states 条件下估计每条候选证据的新增价值。claim-aware chunking、retrieval 与 Atom-Union candidate pool 作为上游支撑模块，用于提供粒度适中、召回充分且去重后的候选证据集合。最终，构建得到的证据链被渲染为 verifier 可见的 prompt evidence prefix，并由指令微调后的 LLM verifier 输出事实核查标签。

## Task Definition

**输入与输出。** 给定一条待核查的声明 $c$（claim）及其相关的若干原始报道，任务的目标是输出一个事实核查标签 $\hat y\in\mathcal{Y}$。标签集合 $\mathcal{Y}$ 随数据集而异：LIAR-RAW 为六类细粒度真值（pants-fire / false / barely-true / half-true / mostly-true / true），RAWFC 为三类（false / half / true），HoVer 为二分类（supported / not-supported）等。本文方法不依赖特定标签体系，可适配任意离散标签集合 $\mathcal{Y}$。

**Claim atoms.** 本文将原始 claim $c$ 分解为少量可独立验证的 atomic propositions：
$$A(c)=\{(a_1,q_1),\dots,(a_m,q_m)\},\quad 1\le m\le 6.$$
其中 $a_i$ 表示第 $i$ 个原子命题，$q_i$ 是其对应的检索查询（query rendering）。该分解只使用 claim 本身，不引入外部知识；日期、数量、否定、比较对象、地点、范围、归因等必须保留在 proposition 内，避免将可验证事实拆成孤立关键词。Claim atoms 在后续 evidence map 与 chain selection 中作为状态变量。

**有序证据链。** 本文构造一条有序证据链 $\mathcal{T}=[s_1,\dots,s_T]$ 作为核心中间产物。每一步
$$s_t=(u_t,a_{i_t},h_{i_t}^{(t-1)}\!\to h_{i_t}^{(t)})$$
记录所选证据 $u_t$、关联的原子事实 $a_{i_t}$ 及其验证状态 $h_{i_t}^{(t)}\in\{U,S,R,Q,C\}$（unresolved / supported / refuted / qualified / conflicted）的转移。这条证据链既作为 verifier 的判定输入，也作为人类可读的可解释信息。

**判别。** 完整证据链经 prompt evidence policy 截断后与声明拼接为 $x=\mathrm{Render}(c,\mathcal{T}_{\mathrm{prompt}})$，送入指令微调的 LLM verifier：
$$\hat y=\arg\max_y p_{\theta_{\mathrm{ver}}}(y\mid x).$$

## Minimum Atom-Resolving Evidence Chain Selection

本文的核心问题可以表述为 minimum atom-resolving evidence chain selection：给定 claim atoms $\mathcal A=\{a_i\}_{i=1}^m$、候选证据池 $\mathcal C=\{u_j\}_{j=1}^n$，以及候选证据与 atoms 之间的结构化 evidence map $M(u_j,a_i)$，目标是在有限 evidence budget 下选择一条有序证据链
$$\mathcal T=[s_1,\ldots,s_T],$$
使其尽可能解析 claim 中的原子事实，同时控制冗余、背景噪声与上下文成本。每个 step $s_t$ 不只是一个被选中的 evidence unit，而是一次 atom-level verification state transition：
$$s_t=(u_t,a_{i_t},h_{i_t}^{(t-1)}\rightarrow h_{i_t}^{(t)}).$$

因此，selector 的目标不是静态地挑选相关性最高的 top-k evidence，而是在当前已选 prefix $\mathcal T_{<t}$ 和 atom states $H^{(t)}$ 条件下，选择边际贡献最大的下一条证据：
$$u_t=\arg\max_{u_j\in \mathcal C\setminus \mathcal T_{<t}}
U_{\theta_{\mathrm{sel}}}(u_j\mid \mathcal T_{<t},H^{(t)}).$$

构链过程在满足以下任一条件时停止：atom 解析率达到目标 $\rho_{\mathrm{target}}$、候选的最高效用低于阈值 $\tau_{\mathrm{stop}}$、token 预算耗尽、或达到最大步数 $k_{\max}$。在主实验配置中，我们采用更严格的 $\rho_{\mathrm{target}}=1.0$，即尽可能使所有 claim atoms 获得解析状态；较低阈值可作为敏感性分析。该目标把事实核查中的证据选择从“相关 evidence ranking”转化为“面向 atom resolution 的序列化证据链构建”。

## Atom-Evidence Map

为了使证据选择能够围绕原子事实展开，本文为每个 claim 构建 atom-evidence map。给定候选证据池 $\mathcal C=\{u_j\}_{j=1}^{n}$ 和 claim atoms $\mathcal A=\{a_i\}_{i=1}^{m}$，map 为每个候选证据与 atom 的配对提供结构化标注：
$$M(u_j,a_i)=(r_{ij},d_{ij},c_{ij}),$$
其中 $r_{ij}$ 表示 relation，$d_{ij}$ 表示 directness，$c_{ij}\in[0,1]$ 表示置信度。Relation 用于描述证据对 atom 的立场或功能，例如 support、refute、qualify、background 或 irrelevant；directness 用于区分证据是否直接验证 atom，还是只提供上下文；confidence 表示该标注的可信程度。实现上，该标注由 LLM 通过固定 prompt 给出，并约束 atom 集合固定，不允许在 map 阶段创建、合并、拆分或改名 atoms。

除三元组外，evidence map 还记录 evidence role、key spans、duplicate group 等辅助字段，用于后续新颖性判断与去重。需要强调的是，evidence map 并不直接给出最终事实核查标签，而是作为 selector 的结构化中间表示：它告诉 selector 每条候选证据可能解析哪些 atoms、以何种关系解析、以及该关系是否足够直接。基于该 map，selector 能够显式追踪 atom states，并把证据组织为一条有状态的 verification chain。

## Learned Marginal Chain Selector

为求解上述证据链选择问题，本文设计了一个 learned marginal chain selector。该 selector 将每个候选证据的价值定义为其在当前 chain prefix 下带来的状态条件化边际效用，而不是仅依赖静态 retrieval score 或人工优先级。我们为每个 atom 定义验证状态：
$$h_i^{(t)}\in\{U,S,R,Q,C\},$$
分别表示 unresolved, supported, refuted, qualified, conflicted。初始时所有 atom 均为 $U$。selector 维护由当前 hard atom state 导出的 one-hot 状态质量 $p_i^{(t)}(s)$，用于计算状态条件化边际特征；学习过程只优化线性效用函数的非负权重，状态转移本身不作为可微模块端到端训练。状态转移遵循：support/refute/qualify 分别映射到 $S/R/Q$，若新证据与已有 $S/R$ 立场冲突则进入 $C$。

在第 $t$ 步，候选证据 $u_j$ 会根据当前 atom states、已选证据集合以及 evidence map 标注被映射为边际特征：
$$\phi^{(t)}(u_j)=
[\phi_{\mathrm{res}},\phi_{\mathrm{ent}},\phi_{\mathrm{cov}},
\phi_{\mathrm{new-rel}},\phi_{\mathrm{tension}},
\phi_{\mathrm{corr}},\phi_{\mathrm{src-novel}},
\phi_{\mathrm{text-novel}},\phi_{\mathrm{conf}},
\phi_{\mathrm{map}},\phi_{\mathrm{ret}},\phi_{\mathrm{cost}}].$$
其中 resolution、coverage、new relation、stance tension 和 corroboration 等特征刻画该证据是否能推进尚未解析的 atoms，或为已解析 atoms 提供新的支持、反驳、限定或冲突信息。若 $u_j$ 对 atom $a_i$ 给出可解析关系，则其解析增益可写为：
$$g_{ij}^{(t)} = p_i^{(t)}(U)\cdot \delta(d_{ij})\cdot \max(c_{ij},0.5),$$
其中 $p_i^{(t)}(U)$ 是当前 atom 仍未解析的状态质量，$\delta(d_{ij})$ 为 directness 权重。候选的解析边际贡献为：
$$\phi_{\mathrm{res}}^{(t)}(u_j)=\frac{1}{m}\sum_{i=1}^{m}\max_{M(u_j,a_i)}g_{ij}^{(t)}.$$

selector 使用非负约束的线性边际效用函数进行打分：
$$U_{\theta_{\mathrm{sel}}}(u_j\mid \mathcal T_{<t},H^{(t)})
=b+\sum_{\ell\ne \mathrm{cost}}w_\ell\phi_\ell^{(t)}(u_j)
-w_c\phi_{\mathrm{cost}}^{(t)}(u_j).$$

权重 $\theta_{\mathrm{sel}}$ 通过 proxy-supervised pairwise ranking 学习得到。训练时，在 evidence-map feature rows 上模拟 rollout，构造偏好对 $(u^+,u^-)$，其中 $u^+$ 是 proxy 排序更优的候选，$u^-$ 是较差候选。proxy 只用于构造训练偏好对，而推理时 selector 依赖学习到的边际效用函数进行逐步选择。优化目标为：
$$\mathcal L(\theta_{\mathrm{sel}})=
\frac{1}{|\mathcal P|}
\sum_{(u^+,u^-)\in\mathcal P}
\log\left(1+\exp\left(-[U_{\theta_{\mathrm{sel}}}(u^+)-U_{\theta_{\mathrm{sel}}}(u^-)]\right)\right).$$
实现中使用 Adam 优化，并用 softplus 参数化保证权重非负：
$$w_\ell=\mathrm{softplus}(\theta_{\ell}),\quad w_c=\mathrm{softplus}(\theta_c).$$

除上述基于 proxy 规则的学习器外，本文还实现了一个以 verifier 反馈为信号的 reward 版学习器：它用 verifier 在加入某证据后的判定边际变化（delta margin）作为 reward，以 pairwise + listwise + 平滑正则的复合损失训练同一组 softplus 权重。本文主方法采用更简单、可复现性更强的 proxy-supervised 版本，reward 变体作为消融对照报告（具体见 XXX）。这样，方法保留了 evidence-map 特征的可解释性，同时避免把证据链构建退化为固定规则排序。

## Candidate Pool Construction

候选证据池构建并非本文的主要优化对象，而是为后续证据链选择提供粒度适中、召回充分且去重后的 evidence units。具体地，我们首先将每篇 report 切分为 claim-aware evidence chunks，以避免单句证据信息不足或整段证据噪声过高。chunking 同时考虑句子与 claim 的相关性以及相邻句子之间的语义连贯性，并通过局部边界峰值和长度约束得到最终证据单元。

随后，系统同时从两条路径构造候选证据：一条路径使用整条 claim 进行 baseline retrieval，以保证全局相关性；另一条路径使用 claim atoms 的 query rendering 分别检索 evidence chunks，以提高对细粒度事实的覆盖。两路候选经过去重、route-level fusion 与 MMR 去冗余后，得到最终候选池：
$$\mathcal C=\mathrm{MMR}(\mathrm{Dedup}(\mathcal B\cup \mathcal A_{\mathrm{route}})).$$
其中 $\mathcal B$ 表示 claim-level baseline pool，$\mathcal A_{\mathrm{route}}$ 表示 atom-route pool。后续方法不直接将 $\mathcal C$ 作为无序 top-k 输入 verifier，而是在其上构建 atom-evidence map 并进一步学习证据链选择策略。关于 chunking 粒度、retrieval route 与 Atom-Union 组件的消融见 XXX。

## Verifier Rendering and Prediction

在得到 ordered evidence trace $\mathcal T=[s_1,s_2,\ldots,s_T]$ 后，我们使用一个指令微调后的 LLM 作为最终 verifier。Verifier 接收 claim 及其 prompt-visible evidence trace，并输出事实核查标签：
$$\hat y=\arg\max_y p_{\theta_{\mathrm{ver}}}(y\mid x),$$
其中 $x=\mathrm{Render}(c,\mathcal T_{\mathrm{prompt}})$ 表示由 claim $c$ 和被截断后的证据 trace $\mathcal T_{\mathrm{prompt}}$ 构造的输入 prompt。verifier 在 Task Definition 所定义的标签集合 $\mathcal Y$ 上做分类，每个标签以单 token 形式编码，verifier 在标签 token 位置上输出预测；可选地附加一个 coverage 辅助头预测证据覆盖程度。训练阶段采用监督微调，最小化标准交叉熵损失：
$$\mathcal L_{\mathrm{verifier}}=-\log p_{\theta_{\mathrm{ver}}}(y^\ast\mid x),$$
其中 $y^\ast$ 为样本的 gold fact-checking label。

由于 selector 生成的完整 trace $\mathcal T$ 长度存在显著差异，直接将完整 trace 输入 verifier 可能导致长上下文噪声增加、有效证据信号被稀释，并显著提高训练与推理成本。因此，我们在 verifier 之前引入 prompt evidence policy，将完整 ordered trace 映射为 verifier 可见的 evidence prefix：
$$\mathcal T_{\mathrm{prompt}}=[s_1,s_2,\ldots,s_{K^\ast}],\quad K^\ast\le T.$$

主实验采用 minmax prompt evidence policy：我们沿 selector 给出的顺序逐步加入 evidence step，并在满足最小证据数 $k_{\min}=5$ 后检查当前 trace 是否已经达到 atom-level resolution target。令
$$\rho_t=\frac{|\{a_i:h_i^{(t)}\in\{S,R,Q,C\}\}|}{|\mathcal A|},$$
其中 $\rho_t$ 是第 $t$ 步后的 resolved atom rate。若
$$\rho_t\ge\rho_{\mathrm{target}},$$
则认为当前 trace 已经达到目标解析状态，并停止继续加入证据。因此截断位置定义为：
$$K^\ast=\min\{t:t\ge k_{\min}\land\rho_t\ge\rho_{\mathrm{target}}\}.$$
若不存在满足条件的 $t$，则退化为最大证据数约束。主实验中 $k_{\max}=10$，即 verifier 至少接收 5 个 MREC steps；若 5 步后 claim atoms 已达到目标解析状态，则停止，否则继续补充，最多到 10 步：
$$K^\ast=\min(k_{\max},T).$$
此外，本文还提供固定 top-k（fixed_topk）与按 token 预算截断（budget）等容量敏感性对照策略，用于消融分析（见 XXX）。其中 fixed_topk 可视作 $k_{\min}=k_{\max}$ 的 minmax 特例；resolve-stop 信号已并入 minmax policy，不在主文中单独报告。

---

## 附录：符号与实现术语对照

为便于复现，下表给出论文符号与代码/配置术语的对应关系。

| 论文符号 | 代码术语 / 配置项 | 说明 |
|---|---|---|
| $c$ | `claim` | 原始声明 |
| $\mathcal{A}=\{a_i\}$ | `claim_atoms` / `claim_atomization` | 原子命题集合 |
| $q_i$ | `query_rendering` | atom 的检索查询 |
| $u_j$ | `candidate` / chunk | 证据单元 |
| $\mathcal{B}$ | baseline pool | 整条 claim 检索得到的候选池 |
| $\mathcal{A}_{\mathrm{route}}$ | atom-route pool | 各 atom route 检索结果的并集 |
| $\mathcal{C}$ | atom-union pool | 融合、去重、MMR 后送入 selector 的候选证据池 |
| $M(u_j,a_i)$ | `evidence_map` (`relation`/`directness`/`confidence`) | 证据-atom 标注 |
| $h_i^{(t)}$ | `atom_states` | atom 验证状态 |
| $H^{(t)}$ | `atom_states` snapshot | 第 $t$ 步的全局 atom 状态 |
| $\{U,S,R,Q,C\}$ | `VALID_ATOM_STATES` | 状态集合 |
| $\phi^{(t)}(u_j)$ | `marginal_features` (12 维) | 边际特征向量 |
| $U_{\theta_{\mathrm{sel}}}$ | `score_marginal_features` | 线性效用 |
| $\theta_{\mathrm{sel}}$ (softplus) | `LearnedMarginalWeights` | 学习权重 |
| $\mathcal{T}$ | `mrec_trace` / `selected_candidates` | 有序证据链 |
| $s_t$ | `mrec_step` | 单步（含 operation/状态转移） |
| operation | OPEN/CORROBORATE/CONTRAST/BRIDGE/FALLBACK | 转移操作 |
| $\rho_t$ | `resolved_atom_rate` | 解析率 |
| $\rho_{\mathrm{target}}$ | `target_resolved_rate` | 解析率目标 |
| $k_{\min}/k_{\max}$ | `min_evidence_count`/`max_evidence_count`（prompt 层）；`min_steps`/`max_steps`（trace 层） | 证据数上下界 |
| $K^\ast$ | `prompt_evidence` 截断位置 | prompt 可见证据数 |
| $\mathcal{T}_{\mathrm{prompt}}$ | `build_training_row` 渲染输入 | verifier 输入 |
| $\hat y$ | `gold_label` / label token | 判定标签 |
| $\mathcal{L}_{\mathrm{verifier}}$ | `label_token_trainer` CE loss | verifier 损失 |
