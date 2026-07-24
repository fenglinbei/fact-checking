# AAAI 写作大纲 v0.4.2（Structure-only）

本文档以 v0.4.1 为基线，给出一版围绕 **pre-verdict evidence organization** 的论文初稿。正文中的内部数值只来自已通过 clean structure-only artifact contract 的实验；尚未完成独立核验的分析不进入当前结果或贡献表述。外部论文已公开的基线数值保留，并在方法行和表注中分别标明方法来源与数值来源。

# Abstract

基于原始报道的事实核查不仅要检索相关文本，更要将分散证据组织为适合判别器使用的声明视图。我们的核心洞见是，证据组织并非中性的输入格式化：typed claim--evidence map 随证据前缀演化产生的结构增量，本身就能构成有效的 selector supervision——每个候选的价值可以由其如何解决 claim atoms、扩展结构覆盖并补充当前证据状态来刻画，而无需读取事实标签、gold/teacher-provided evidence supervision 或 verifier feedback。基于这一洞见，我们提出一个 **decompose--align--organize** 框架：将声明分解为可核查 atoms，在 Atom-Union 候选池上构建保留来源映射的 typed Evidence Map，再从 map-induced state transitions 中诱导训练偏好并学习 state-conditioned evidence selector。该框架同时生成紧凑的 verifier-facing evidence prefix，以及完整记录 atom alignment、证据来源和选择过程的可审计轨迹。独立人工双标与第三人盲化仲裁进一步显示，生成的 claim atoms 在 faithfulness、atomicity 与 claim-level complete coverage 上的最终 gold 通过率分别为 99.22%、95.72% 和 99.00%，表明自动 claim atomization 在本研究审计样本中高度符合人工质量判断。在 LIAR-RAW 与 RAWFC 官方测试集上，我们分别取得 35.40 和 65.08 macro-F1，在已核实的 original associated-report setting 比较中超过最强可比基线 2.27 和 0.48 个百分点；在 SciFact official-development comparison 中，同一监督原则也在两项 abstract-level 指标上取得表中最高结果。进一步的 controlled fixed-$K$ crossover 表明，证据组织是实质性影响 verifier 的建模选择，而不是无关紧要的输入格式。我们在匿名仓库 \url{ANONYMIZED_URL} 开源完整实现、实验配置，以及全部生成的中间产物、过程数据与审计轨迹。

# Introduction

自动事实核查已经从仅根据声明或元数据预测真实性，发展到显式检索外部证据并联合完成判定。LIAR 提供真实政治声明及细粒度标签，但不包含用于检索的 gold evidence \citep{Wang2017LIAR}；FEVER 将证据句检索与 SUPPORTS/REFUTES/NOT-ENOUGH-INFO 判定结合起来 \citep{Thorne2018FEVER}，HoVer 进一步扩展到跨多篇文档的多跳验证 \citep{Jiang2020HoVer}，SciFact 则要求在科学摘要语料中同时识别证据和判定立场 \citep{Wadden2020SciFact}。在真实世界场景中，CofCED 从与声明关联的原始报道中级联选择 report 和 sentence，并据此构建 LIAR-RAW 与 RAWFC \citep{Yang2022CofCED}；AVeriTeC 则将证据来源扩展到开放网页 \citep{Schlichtkrull2023AVeriTeC}。这些任务的发展使系统不仅需要“找到相关内容”，还需要决定最终判别器应当看到哪些证据。

已有方法从多个方向解决证据使用问题。CofCED 使用 coarse-to-fine cascaded selectors 对原始报道进行压缩 \citep{Yang2022CofCED}；GEAR 与 KGAT 通过图结构传播多条候选证据的交互信息 \citep{Zhou2019GEAR,Liu2019KGAT}；L-Defense 和 G-Defense 分别生成相互竞争的解释与图式防御结构 \citep{Wang2024LDefense,Wang2026GDefense}；DelphiAgent 使用多智能体反馈形成判定 \citep{Xiong2025DelphiAgent}，KG-CRAFT 则结合知识图谱与对比问题组织多阶段推理 \citep{Lourenco2026KGCRAFT}。这些系统在判定前利用证据，经过不同的处理流程得到供最终判别器使用的证据形式，但是，`evidence unit -- claim subfact -- source span` 的显式对应关系以及 evidence set 的构造路径，并不总是作为一等输出被保留。本文据此采用一个 **decompose--align--organize** 的设计视角，构造一套可显式观察且可审计的自动事实核查证据选择流程。

声明分解已被用于生成可核查子问题、构造程序化验证步骤，以及在开放语料中提高复杂声明的检索与验证能力 \citep{Chen2022ClaimDecomp,Pan2023ProgramFC,Chen2023WildEvidence}。我们将声明分解嵌入整套证据选择流程，既作为上游输入，也作为可审计结构的起点：先得到可独立检查的 claim atoms，再显式记录每条候选证据与 atoms 的 typed alignments，最后在判定前形成一条 source-linked ordered evidence trace。

序列化证据选择同样已有充分先例。DQN-based evidence selection 将已选证据纳入序列决策状态 \citep{Wan2021DQN}；natural-logic-guided retrieval 根据既有证据更新形式证明状态并动态终止检索 \citep{Aly2022NaturalLogic}；证据缺失研究直接分析 verifier 何时错误地把不完整输入视为充分 \citep{Atanasova2022Insufficient}；近期 user-centric evidence ranking 比较 one-shot 与 incremental ranking，并直接优化让互补且充分的信息更早出现 \citep{Alt2026EvidenceRanking}。在 raw-report setting 中，FFRR 使用黑盒 LLM 的细粒度反馈训练检索策略 \citep{Zhang2024FFRR}。因此，本文并非首次提出 sequential selection、context-conditioned ranking 或 sufficient-prefix construction。我们研究的是另一种监督边界：能否以 typed Evidence Map 在不同 prefix state 下发生的结构变化为主要信号，并仅用 map quality、retrieval relevance 与冻结候选索引作确定性区分，自动产生不依赖 verdict/gold/verifier feedback 的 selector 训练偏好。

本文聚焦如下研究问题：

> Can a typed claim--evidence map provide label- and verifier-independent supervision for constructing an atom-indexed, source-linked evidence trace before verdict prediction?

为回答该问题，本文提出一种 atom-aware、structure-induced 的证据组织方法。给定声明及其相关报道，系统依次执行 claim atomization、claim-aware chunking、Atom-Union retrieval 和 Evidence Map annotation。训练阶段在每个 rollout state 中根据 direct resolving、partial resolving、new atom coverage、new relation、map quality 与 retrieval score 形成确定性的 structure-induced ordering，并由 winner-vs-rest pairs 学习非负线性 structural marginal score。推理阶段，固定 selector 访问完整候选池并产生 audit ordering；独立的 verifier-visible prefix policy 再根据 evidence count、structural coverage proxy 与 tokenizer context guard 产生判别器实际可见的 evidence prefix，并用于最终 verifier 的训练。

本文的主要贡献如下：

1. 提出一种 typed、source-linked 的中间表示，将 claim atoms、candidate--atom alignments、稳定 report/chunk identity 与逐步 map-induced state transition 组织为可审计的 ordered evidence trace。
2. 提出一种 structure-induced state-conditioned selector：它以 Evidence Map 的状态转移为主要信号重建 winner-vs-rest preferences，并学习 prefix-dependent structural marginal score，而不使用事实标签、gold evidence trace、gold/teacher-provided evidence order fields 或 verifier feedback。
3. 通过独立人工双标与第三人盲化仲裁，验证本研究审计样本中的自动 claim atomization；三项主要质量维度的最终 gold 通过率均不低于 95.72%，为在该审计范围内将其作为后续 retrieval 与 Evidence Map 构建的可靠上游结构输入提供直接人工证据。
4. 在 LIAR-RAW、RAWFC 与 SciFact 上评估该方法，并通过 LIAR-RAW two-seed fixed-$K$ matched-verifier crossover 分析 one-shot 与 state-conditioned evidence organization 对 verifier 与训练种子的方向敏感性。

# Related Work

## Evidence-grounded Fact Verification

Evidence-grounded fact verification 通常包含 document retrieval、sentence selection 与 verdict prediction。FEVER 建立了从 Wikipedia 检索证据句并预测 SUPPORTS、REFUTES 或 NOT-ENOUGH-INFO 的经典任务设置 \citep{Thorne2018FEVER}。DeClarE 将声明、证据文章及来源信息联合编码，为外部证据感知的虚假声明识别提供了早期实现 \citep{Popat2018DeClarE}。后续工作进一步建模多证据之间的交互：GEAR 将候选证据构造成全连接图并进行信息传播 \citep{Zhou2019GEAR}，KGAT 则通过 node/edge kernels 与图注意力完成细粒度证据聚合 \citep{Liu2019KGAT}。HoVer 把检索扩展到最多四篇 Wikipedia 文档上的多跳验证 \citep{Jiang2020HoVer}；SciFact 要求系统从科学摘要中识别 rationale sentences 并预测科学声明立场 \citep{Wadden2020SciFact}。这些工作表明，在判定前检索或聚合证据已经是成熟做法。本文关注在候选检索之后，能否显式构造 atom-indexed、source-linked 的证据组织 artifact。

## Fact Checking over Raw Reports

Raw-report fact checking 使用与声明相关、但尚未被人工核查报告整理成最终解释的新闻报道。CofCED 首先从 top-$K$ reports 中级联筛选解释性句子，并提出 LIAR-RAW 与 RAWFC 两个数据集 \citep{Yang2022CofCED}。DeReC 使用 dense sentence retrieval 与紧凑的 DeBERTa classifier，强调检索方法在该场景下的效率与可扩展性 \citep{Qazi2025DeReC}。LLM 时代的方法进一步改变了 evidence-to-verdict interface：L-Defense 生成相互竞争的正反解释 \citep{Wang2024LDefense}，RAFTS 通过检索与对比论证完成判定 \citep{Yue2024RAFTS}，DelphiAgent 以多角色、多轮反馈形成共识 \citep{Xiong2025DelphiAgent}，G-Defense 与 KG-CRAFT 分别引入 subclaim graph 和 knowledge-graph-guided contrastive reasoning \citep{Wang2026GDefense,Lourenco2026KGCRAFT}。FFRR 则以黑盒 LLM 的细粒度任务反馈训练 retrieval policy \citep{Zhang2024FFRR}。相比于一些复杂证据处理方法，本文不生成新的 arguments 或 summaries，而是将注意力落在原始证据上。同时，与依赖 downstream feedback 监督的 selector 方法相比，我们以 typed Evidence Map 的结构状态变化为主要偏好信号，map quality、retrieval relevance 与冻结候选索引只用于确定性区分，再据此构造 verifier-facing ordering。

## Claim Decomposition and Structured Evidence

Claim decomposition 常被用于降低复杂声明的检索和验证难度。Literal/implied subquestion generation 将复杂声明转化为可回答的子问题 \citep{Chen2022ClaimDecomp}，ProgramFC 将复杂核查过程表示为由多个子任务组成的程序 \citep{Pan2023ProgramFC}，wild-evidence verification 结合声明分解、两阶段检索与 claim-focused summaries 处理开放语料 \citep{Chen2023WildEvidence}。另一方面，GEAR、KGAT、G-Defense 与 KG-CRAFT 已展示图结构在证据聚合、subclaim dependency 和对比推理中的作用 \citep{Zhou2019GEAR,Liu2019KGAT,Wang2026GDefense,Lourenco2026KGCRAFT}。本文的 Evidence Map 并不承担 verdict reasoning，而是在方法中作为服务于证据选择的 typed intermediate representation，显式连接 claim atoms、evidence units 与 source spans，并为每个 prefix state 计算结构变化。

## Sequential Evidence Ordering and Sufficiency

已有工作已经研究逐步证据选择、紧凑证据集和充分前缀。DQN-based selection 将精确证据搜索建模为序列决策 \citep{Wan2021DQN}；natural-logic-guided retrieval 根据已检索证据的证明状态选择后续文档并决定是否停止 \citep{Aly2022NaturalLogic}；insufficient-evidence analysis 说明 verifier 可能无法可靠识别缺失信息 \citep{Atanasova2022Insufficient}。User-centric evidence ranking 更直接地研究 complementary evidence 和 early sufficient prefix，并显示 incremental ranking 与一次性排序具有不同能力 \citep{Alt2026EvidenceRanking}；FFRR 则利用 verdict label 与黑盒 LLM task feedback 训练 raw-report retrieval policy \citep{Zhang2024FFRR}。

这些工作表明，候选证据的贡献取决于当前已选上下文。本文的区别在于监督来源：我们的 selector 不依赖 gold evidence、verdict-conditioned rewards、formal proof states 或 downstream feedback，而是以固定 candidate pool 上 typed claim--evidence map 的状态转移为主要信号，并用 map quality、retrieval relevance 与冻结候选索引作确定性区分，产生随 prefix 变化的结构选择偏好。

# Methodology

## Overview

给定声明 $c$ 及相关报道集合 $\mathcal R$，整体方法由三个层级组成：

1. **Upstream input construction.** 上游模块首先将声明分解为 claim atoms $\mathcal A=\{a_i\}_{i=1}^{m}$，并为每个 atom 生成检索 query；随后将相关报道切分为保留 report/span identity 的 evidence units $\mathcal U=\{u_j\}$。Claim route 与 atom routes 经 RRF、dedup 和 MMR 融合为 Atom-Union candidate pool $\mathcal C$。因此，核心算法接收经过上游处理得到的 claim atoms、evidence units，以及覆盖不同检索视角的候选池。

2. **Structure-induced evidence organization.** 核心算法先构建 typed Evidence Map，显式记录 candidate--atom relation、directness、confidence 与 source linkage。基于当前 prefix 已覆盖的 atoms、已有 relations 和重复程度，算法在每个 rollout state 重新诱导 winner-vs-rest preferences，并训练 structural marginal scorer。推理时，固定 scorer 在完整候选池上逐步更新结构状态并重新评分，最终产生保留 state transitions 与 provenance 的 ordered audit trace $\mathcal T^{\mathrm{audit}}$。

3. **Downstream prefix projection and verification.** Audit trace 形成后，独立的 verifier-visible prefix policy 仅沿既定顺序执行容量投影，根据 evidence-count constraint、structural coverage proxy 与 tokenizer context guard 得到 $\mathcal T^{\mathrm{vis}}$，而不重新评分或排序。Verifier training 只在这一层使用事实标签；推理阶段则将 claim、atom cues 与可见 evidence text 渲染为输入，由冻结的 selector、固定 prefix policy 和 label-token verifier 共同输出最终标签。

## Task Definition

给定声明 $c$、相关报道集合 $\mathcal R$ 和离散标签空间 $\mathcal Y$，系统需要预测其事实标签 $y\in\mathcal Y$。由报道得到候选证据集合 $\mathcal C=\{u_j\}_{j=1}^{n}$；每个候选均具有稳定来源映射 $p(u_j)$。分解后声明的结构单元集合记为 $\mathcal A$，候选与结构单元之间的 typed alignment 记为 $M$。因此，证据组织问题的输入实例可写为

$$
\mathcal X=(c,\mathcal A,\mathcal C,M,p).
$$

为表达“高结构贡献证据应尽早出现”的设计意图，我们定义如下概念性序列目标：

$$
J(\pi)
=\sum_{t=1}^{L}\gamma_t\,
\Delta_{\mathrm{str}}\!\left(u_{\pi_t}\mid \mathcal T_{<t},\mathcal X\right),
\qquad \gamma_1\ge\gamma_2\ge\cdots\ge 0,
$$

其中 $\mathcal T^{\mathrm{audit}}=(u_{\pi_1},\ldots,u_{\pi_L})$ 是保留 source linkage 的 ordered evidence trace，$\Delta_{\mathrm{str}}$ 表示候选相对于既有前缀的结构边际贡献。本文并不求解 $\max_\pi J(\pi)$；实际算法用学习到的 state-conditioned scorer 逐步执行 greedy approximation，因此不保证全局最优。

在有限上下文预算 $B$ 或者证据数量范围 $(K_{min}, K_{max})$ 限制下，下游策略从 $\mathcal T^{\mathrm{audit}}$ 投影出 verifier-visible prefix $\mathcal T^{\mathrm{vis}}$，最终预测为

$$
\mathcal T^{\mathrm{vis}}=P_B(\mathcal T^{\mathrm{audit}}),
\qquad
\hat y=\arg\max_{y\in\mathcal Y}
p_{\theta_{\mathrm{ver}}}(y\mid c,\mathcal A,\mathcal T^{\mathrm{vis}}).
$$

因此，本文精确研究的是：在标签与 verifier 信号均不参与证据排序监督的约束下，如何构造一个 prefix-dependent、source-linked 的证据序列，使其有限前缀能够支持后续事实判定。

## Upstream Input Construction

**Claim Atomization and Query Rendering.**

Claim decomposition 已被用于生成核查子问题和构造复杂声明验证流程 \citep{Chen2022ClaimDecomp,Pan2023ProgramFC,Chen2023WildEvidence}。本文采用这一思想的目的不是生成 reasoning program，而是定义后续 retrieval 与 selection 的有界状态空间。Atomizer 只读取 claim，不读取外部报道或事实标签，并遵循以下约束：不引入 claim 之外的事实；只在存在可分别核查的断言时拆分；保留数量、日期、否定、比较对象、地点、范围和归因；每个 atom 必须是语义完整的 proposition。该模块通过受约束的结构化生成得到 atoms 与 queries；调用参数在 Experiments 中报告，prompt/schema hash、异常输出处理与缓存 checksum 随冻结产物保存。

**Claim-aware Evidence Chunking.**

Chunking 作为上游支撑模块，为每篇 report 切分出 evidence units $u_j=[s_a,\ldots,s_b]$。切分目标是在保留局部语境的同时避免把大量无关文本送入检索器和 map annotator。每条 chunk 保存 report ID、source key、原始句子跨度、文本、retrieval route 和稳定 candidate identity；切分与去重参数随发布配置冻结。

**Atom-Union Candidate Pool Construction.**

Atom-Union 的作用是提供兼顾全局相关性与细粒度 atom coverage 的候选池，使 selector 接收到有界且受控的候选池规模，并减少不必要的 Evidence Map 标注成本。实际实现中，系统分别使用原始 claim 与各 atom queries 检索 evidence units。Claim-level route 为：

$$
\mathcal B=\operatorname{TopK}_{k_b}\{u_j:s(c,u_j)\}.
$$

对每个 atom query $q_i$，multi-signal retrieval score 为：

$$
s(q_i,u_j)=\beta_1\,\operatorname{norm}(\cos(e_{q_i},e_{u_j}))
+\beta_2\,\operatorname{norm}(\operatorname{LexF1}(q_i,u_j))
+\beta_3\,\operatorname{norm}(\operatorname{BM25}(q_i,u_j)),
$$

其中 BM25 遵循经典 probabilistic relevance framework \citep{Robertson2009BM25}。每个 atom route 保留 top-$k_a$：

$$
R_i=\operatorname{TopK}_{k_a}\{u_j:s(q_i,u_j)\},\qquad
\mathcal A_{\mathrm{route}}=\bigcup_i R_i.
$$

多个 atom routes 使用 reciprocal rank fusion 聚合 \citep{Cormack2009RRF}，再与 claim-level pool 合并、按 stable text/source key 去重，并用 maximal marginal relevance 控制相关性与多样性 \citep{Goldstein1998MMR}：

$$
\mathcal C=\operatorname{MMR}\!\left(\operatorname{Dedup}(\mathcal B\cup\mathcal A_{\mathrm{route}})\right).
$$

主配置使用 BGE-base-en-v1.5 产生 dense representations \citep{Xiao2023BGE}。Atom-Union 的作用是提供兼顾全局相关性与细粒度 atom coverage 的候选池；它不直接决定 verifier 最终看到的 evidence order。

## Structure-Induced Evidence Organization

**Typed Evidence Map and Source Linkage.**

图式证据表示已被用于多证据聚合与 subclaim reasoning \citep{Zhou2019GEAR,Liu2019KGAT,Wang2026GDefense,Lourenco2026KGCRAFT}。本文的 Evidence Map 用途不同：它不执行 verdict-oriented message passing，而是为 candidate selection 提供 typed state。对每个 candidate--atom pair：

$$
M(u_j,a_i)=(r_{ij},d_{ij},c_{ij}),
$$

其中 $r_{ij}\in\{\text{support},\text{refute},\text{qualify},\text{background},\text{irrelevant}\}$，$d_{ij}$ 是有序 directness，$c_{ij}\in[0,1]$ 是 map annotator 的自评置信度。Map 同时引用 $p(u_j)$，使每个 alignment 能回到原始 report 和 span。

Relation、directness 与 confidence 由受约束的 LLM annotator 生成。调用设置在 Experiments 中报告，prompt/schema hash 与输出校验元数据随冻结产物保存。

有效 alignment 定义为：

$$
Z(u_j)=\{a_i:\operatorname{Valid}(r_{ij},d_{ij},c_{ij})\}.
$$

每个 atom 的 map-induced state 记为 $h_i^{(t)}\in\{U,S,R,Q,C\}$，整体状态为 $H^{(t)}=\{h_i^{(t)}\}_{i=1}^{m}$。这些状态描述 evidence acquisition progress，分别记录尚未覆盖、观察到 support、refute、qualify 或相互冲突的 map relations。

**Structure-induced Preferences.**

由于 gold atom-level trace 不可得，本文以 Evidence Map 的状态变化为主要信号诱导偏好。词典序依次优先 direct resolving、partial resolving、新 atom coverage 与新 relation，再用 map quality 和 retrieval relevance 区分候选，最终以冻结 candidate/input index 破除同分。该索引来自固定候选池输入顺序，而不是 gold 或 teacher-provided evidence order。

在每个 prefix state，最高优先级候选作为 winner，并与其余候选构成 winner-vs-rest pairs：

$$
\mathcal P_t=\{(u_t^+,u^-):u^-\in\mathcal C_t\setminus\{u_t^+\}\}.
$$

选中 winner 后更新结构状态与 prefix，再生成下一轮 preferences，因此同一候选的偏好关系可以随 prefix 改变。该过程不读取 verdict label、gold/teacher-provided evidence order fields 或 verifier feedback。

**Structure-induced State-conditioned Selector.**

上述确定性规则只用于构造 pairwise preference constraints，而不直接充当最终排序策略。词典序规则能够表达清晰的结构优先级，但其决策边界较为刚性：一旦高优先级条件触发，coverage、relation novelty、redundancy、source diversity 与 evidence cost 等其他信号便难以共同参与权衡。

为此，本文将离散的规则偏好蒸馏为连续、state-conditioned 的 structural scorer。该 scorer 在保留结构监督方向的同时，对多个弱信号进行联合加权，并在 prefix 更新后重新估计候选的边际贡献。这里的“soft”指从硬词典序约束到连续多因素评分的松弛，而不是随机化采样，也不意味着学习到了 verifier utility 或真实任务效用。

每个候选在 state $t$ 下被映射为若干组结构边际特征：

$$
\phi^{(t)}(u)=\left[
\phi_{\mathrm{state}},
\phi_{\mathrm{coverage/relation}},
\phi_{\mathrm{novelty}},
\phi_{\mathrm{quality}},
\phi_{\mathrm{retrieval}},
\phi_{\mathrm{cost}}
\right].
$$

这些 feature groups 分别描述当前结构状态的推进、新 atom/relation 信息、来源与文本新颖性、map annotation quality、检索相关性和上下文成本。实现中的 12 个标量特征为 `resolution_delta`、`entropy_reduction`、`new_atom_coverage`、`new_relation_for_atom`、`stance_tension`、`corroboration_gain`、`source_novelty`、`text_novelty`、`map_confidence`、`map_quality`、`retrieval_score` 与 `cost_ratio`。前十一项以非负权重进入 structural marginal score，cost 以惩罚项进入：

$$
S_\theta^{\mathrm{str}}(u\mid\mathcal T_{<t},H^{(t)})
=b+\sum_{\ell\ne\mathrm{cost}}w_\ell\phi_\ell^{(t)}(u)
-w_c\phi_{\mathrm{cost}}^{(t)}(u),
$$

其中 $w_\ell=\operatorname{softplus}(\theta_\ell)$、$w_c=\operatorname{softplus}(\theta_c)$，保证非 cost 特征的方向可解释。所有正向特征与 cost 权重均从 equal-weight neutral initialization 开始，避免把手工权重先验伪装成学习结果。

State-conditioned selection 本身已有序列决策与 incremental ranking 先例 \citep{Wan2021DQN,Alt2026EvidenceRanking}；本文的区别在于训练偏好以 typed map transitions 为主要信号，map quality、retrieval relevance 与冻结候选索引只用于确定性区分，而不使用 gold evidence 或 verdict-conditioned feedback。参数学习使用经典 pairwise logistic ranking objective \citep{Burges2005RankNet}：

$$
\mathcal L(\theta)=\frac{1}{|\mathcal P|}
\sum_{(u^+,u^-)\in\mathcal P}
\log\left(1+\exp\left[-\left(S_\theta^{\mathrm{str}}(u^+)-S_\theta^{\mathrm{str}}(u^-)\right)\right]\right).
$$

训练输入即使为审计目的保留其他字段，structure-only loader 也只复制候选文本、map alignments、retrieval metadata、token cost、source identity 与固定候选索引。训练 manifest 进一步记录允许字段的投影结果，并确认 verdict label、gold evidence trace、gold/teacher-provided order fields 和 verifier-derived feedback 均未被 selector preference generation 或参数学习读取。

**State-Conditioned Full-Pool-Access Evidence Ordering.**

训练完成后，固定 $S_\theta^{\mathrm{str}}$ 在每一步对剩余候选重新提取 state-dependent features：

$$
u_t=\arg\max_{u\in\mathcal C_t}
S_\theta^{\mathrm{str}}(u\mid\mathcal T_{<t}^{\mathrm{audit}},H^{(t)}).
$$

选中 $u_t$ 后，系统更新 atom state、已见 relation、source coverage 与 novelty statistics，再重新计算下一步分数。运行配置设置 `candidate_top_n=0`、`max_steps=0`，并将 absolute-score threshold 设为极低值；因此 selector 可以访问完整候选池，不会因 atom coverage threshold、固定 trace length 或分数阈值提前停止。实现仍会跳过规范化后重复或无效的候选，因此输出长度 $L$ 满足 $L\le |\mathcal C|$，本文将其称为 **full-pool-access audit ordering**，而不是候选池的严格全排列。该设计与在 retrieval rollout 内依据 formal proof state 或 early-sufficiency objective 动态终止的方法不同 \citep{Aly2022NaturalLogic,Alt2026EvidenceRanking}；本文把可见容量决策留给后续 prefix policy。

每一步的审计记录包括 candidate identity、primary atom、所有 aligned atoms、选择前后 state snapshot、structural score、feature contributions 与 provenance record。完整 audit ordering 用于分析和 prefix projection，不会把所有 metadata 写入 verifier prompt。

## Downstream Prefix Projection and Verification

**Verifier-visible Prefix Policy.**

Prefix policy 只沿 audit ordering 选择前缀，**不重新打分、不重新排序**。对 LIAR-RAW 与 RAWFC，主策略为 minmax$(5,10)$：至少尝试保留五条 evidence units；达到 $k_{\min}=5$ 后，若 map-induced covered-atom rate 达到 $\rho_{\mathrm{target}}=1.0$，则在当前位置投影前缀，否则继续到最多十条。对 SciFact，主配置为 minmax$(9,9)$。令 $L=|\mathcal T^{\mathrm{audit}}|$、$k_0=\min(k_{\min},L)$、$k_1=\min(k_{\max},L)$，先由结构策略确定目标长度：

$$
\widetilde K=\min\left(
\left\{k\in\{k_0,\ldots,k_1\}:\rho(k)\ge\rho_{\mathrm{target}}\right\}
\cup\{k_1\}
\right),
$$

随后，renderer 在构造包含 claim、atom cues 与 evidence text 的完整输入后，执行 tokenizer-aware 1024-token auto-length tail truncation：

$$
K^\ast=\max\left\{k\in\{0,\ldots,\widetilde K\}:
\operatorname{FitsContext}\big(\operatorname{Render}(c,\mathcal A,\mathcal T_{1:k}^{\mathrm{audit}})\big)
\right\},
$$

并令 $\mathcal T^{\mathrm{vis}}=\mathcal T_{1:K^\ast}^{\mathrm{audit}}$。其中 $\rho(k)$ 是 audit-order prefix $1{:}k$ 的 map-induced covered-atom rate。该两阶段定义保证 coverage policy 决定“希望展示到哪里”，renderer 只从尾部删除 evidence units，既不重新评分也不改变剩余顺序；独立的 max-length guard 仅审计并报告残余超长样本。当 audit ordering 本身较短或 auto-length truncation 触发时，实际 evidence count 可以低于 $k_{\min}$。这里的 $\rho_{\mathrm{target}}$ 是 structural coverage threshold，而不是对事实充分性的保证；本文不把该阈值解释为人工或逻辑意义上的 evidence sufficiency \citep{Atanasova2022Insufficient,Alt2026EvidenceRanking}。

**End-to-end Algorithm.**

```text
Algorithm 1: Structure-Induced Full-Pool-Access Evidence Ordering

Input: claim c, reports R, learned weights theta, prefix policy P
Output: audit trace T_audit, verifier-visible prefix T_vis, label y_hat

1: A <- AtomizeAndRenderQueries(c)
2: U <- ClaimAwareChunkReports(R, c)
3: C <- BuildAtomUnionPool(c, A, U)
4: M <- AnnotateTypedEvidenceMap(c, A, C)
5: H <- InitializeMapInducedStates(A)
6: T_audit <- []
7: C_remaining <- C
8: while C_remaining is not empty do
9:     score every valid candidate with S_str(candidate | T_audit, H)
10:    u <- deterministic argmax with structural-feature/token-cost cascade and frozen candidate-index tie break
11:    if u duplicates an already selected canonical unit then
12:        remove u from C_remaining and continue
13:    end if
14:    append candidate, atom transition, feature scores, and provenance to T_audit
15:    H <- UpdateMapInducedState(H, M(u, .))
16:    remove u from C_remaining
17: end while
18: T_vis <- PrefixPolicy(T_audit, P)   # no re-scoring or re-ordering
19: x <- RenderClaimAtomCuesAndEvidence(c, A, T_vis)
20: y_hat <- LabelTokenVerifier(x)
21: return T_audit, T_vis, y_hat
```

**Verifier Rendering and Prediction.**

除 claim 外，verifier renderer 对每个可见 step 只输出简短的 atom cue 与 evidence 原文。Operation、map relation、directness、confidence、state transition、structural score 和 provenance metadata 均留在审计 artifact，不进入 verifier text。该边界避免把 selector diagnostics 当成额外推理结论，同时仍可通过 stable IDs 回查原始来源。

Verifier 将标签映射为单 token answer choices。若标签集合为 $\mathcal Y=\{y_1,\ldots,y_C\}$，映射 $g:\mathcal Y\rightarrow\{A,B,\ldots\}$，训练目标为：

$$
\mathcal L_{\mathrm{ver}}=-\sum_n
\log p_{\theta_{\mathrm{ver}}}\left(g(y_n)\mid
\operatorname{Render}(c_n,\mathcal A_n,\mathcal T_n^{\mathrm{vis}})\right).
$$

Verifier 使用参数高效适配的自回归语言模型实现；LoRA 通过低秩更新降低微调成本 \citep{Hu2022LoRA}。具体 backbone、checkpoint、rendering style 与训练超参数统一在 Experiments 中报告。

**Training and Inference Pipeline.**

**Structure-only selector training.** 对训练集依次生成 atoms、Atom-Union pool 和 Evidence Map；从每个 state 产生 winner-vs-rest pairs 并训练 selector weights，validation target 是 structure-winner pair accuracy。训练过程不启动 verifier，也不读取事实标签。

**Verifier training.** 固定 selector 后，为训练样本构建 audit ordering，再通过与推理阶段相同的 prefix policy 与 renderer 生成 verifier rows。只有该阶段读取 gold fact-checking labels。Selector weights 在 verifier training 中保持冻结。

**Inference.** 对 validation/test claim 执行相同的 atomization、retrieval、map annotation、full-pool-access ordering、prefix projection 与 label-token prediction。上游 procedure/configuration 与已训练参数在 split 之间固定，各 split 独立生成样本级 atoms、candidate pools 与 maps；verdict labels 不回流到任何上游模块。

# Experiments

## Research Questions

实验围绕五个问题展开：

- **RQ1：** structure-only 方法在 LIAR-RAW 与 RAWFC 上相对于可核实的同任务基线表现如何？
- **RQ2：** 相同的结构监督边界能否适用于标签体系和证据来源均不同的 SciFact？
- **RQ3：** 在这一选择器流程中，主要收益来自什么组件？
- **RQ4：** Map标注能够多大程度影响最终结果？
- **RQ5：** 在固定 evidence ordering 的条件下，证据容量与 verifier-visible prefix policy 如何影响判别性能和上下文成本？

## Datasets and Evaluation Protocol

### LIAR-RAW

LIAR 提供 12.8K 条真实政治声明及六类细粒度标签 \citep{Wang2017LIAR}；LIAR-PLUS 为其增加了从 fact-check articles 中抽取的 justifications \citep{Alhindi2018LIARPlus}。LIAR-RAW 则进一步为声明关联原始报道，并由 CofCED 论文正式提出 \citep{Yang2022CofCED}。这三者的 evidence source 不应混写：本文使用 LIAR-RAW associated reports 和官方 train/validation/test split，不把 LIAR-PLUS justification 当作候选证据或 selector supervision。我们报告 test accuracy、macro-precision、macro-recall 和 macro-F1。

### RAWFC

RAWFC 同样由 CofCED 提出，包含来自 Snopes 的三分类声明以及相关 raw reports \citep{Yang2022CofCED}。本文使用其官方 split，报告 test accuracy 与 macro-P/R/F1。LIAR-RAW 与 RAWFC 采用相同的 backbone/LoRA adaptation family；各数据集的训练配置与 checkpoint 均只依据 validation 选择，test 不参与选择。

### SciFact

SciFact 包含 1.4K expert-written claims、5,183 篇 scientific abstracts，以及 abstract-level stance 与 sentence-level rationale annotations \citep{Wadden2020SciFact}。本文使用 original 300-claim development split 和完整 5,183-abstract corpus，不使用 gold `cited_doc_ids` 或 gold rationale 构造候选池。我们使用官方 full-pipeline scorer 报告 sentence Selection-only、sentence Selection+Label、abstract Label-only 与 abstract Label+Rationale 四项 micro-F1。Verifier 在 SciFact training split 上训练，development split 同时用于 checkpoint selection，因此结果只用于评估该方法在 **dataset-specific training** 下对 scientific fact-checking setting 的适用性，不作为 zero-shot transfer、hidden-test 或整体 SciFact SOTA 声明。

## Implementation Details

**LLM-based structural annotation.** 结构标注调用配置如下：

| Component | Dataset | API model string | Temperature | Top-$p$ | Thinking | Max output tokens |
|---|---|---|---:|---:|---|---:|
| Claim atomization | All | `deepseek-v4-flash` | 0 | 1.0 | disabled | 2,048 |
| Evidence Map | LIAR-RAW / RAWFC | `deepseek-v4-flash` | 0 | 1.0 | disabled | 2,048 |
| Evidence Map | SciFact | `deepseek-v4-flash` | 0 | 1.0 | disabled | 4,096 |

该名称是可能随服务更新的 API alias，而非不可变 checkpoint。因此，实验冻结全部标注产物，并在 release manifest 中保存调用日期、服务模型字符串、prompt/schema hash、retry/fallback 元数据与缓存 checksum；DeepSeek 模型家族报告仅作为背景引用 \citep{DeepSeekAI2024V3}。

**Retrieval and selector.** LIAR-RAW/RAWFC 使用 BGE-base-en-v1.5 retrieval 与 Atom-Union candidate pool；SciFact 使用相同 dense encoder 的 open-corpus retrieval。Preference training 在每个样本的 top-20 candidates 上执行五步 rollout，训练 30 epochs，learning rate 为 0.05，并从 equal-weight neutral initialization 开始；推理阶段的 audit ordering 则访问通过去重与有效性检查的完整候选池。

**Verifier and context policy.** 三个数据集均使用 Ministral-3-8B-Instruct-2512 \citep{MistralAI2025Ministral3} 和 LoRA verifier。LIAR-RAW/RAWFC 使用 minmax$(5,10)$ prefix policy，SciFact 使用 minmax$(9,9)$；renderer 采用 `mrec_min` style，完整输入最长 1,024 tokens。Verifier 通过候选 label tokens 的 logits 作确定性判定，不使用生成式 sampling temperature 或 thinking mode。共同设置为四卡 ZeRO-2、BF16、per-device batch size 1、gradient accumulation 4（effective batch size 16）、AdamW、weight decay 0.01、warmup ratio 0.03、cosine-with-restarts scheduler，以及作用于 attention/MLP projections 的 LoRA $r=16$、$\alpha=32$、dropout 0.1。各数据集差异如下：

| Dataset | Learning rate | Epoch cap | Eval/save interval | Early-stop patience | Canonical seed (trainer default) | Checkpoint selection |
|---|---:|---:|---:|---:|---:|---|
| LIAR-RAW | $2\times10^{-5}$ | 12 | 100 steps | 12 | 42 | validation macro-F1 |
| RAWFC | $1\times10^{-5}$ | 12 | 50 steps | 8 | 42 | validation macro-F1 |
| SciFact | $2\times10^{-5}$ | 12 | 100 steps | 12 | 42 | development macro-F1 |

本文内部数值按逐项 clean audit contract 放行，并在结果段记录来源 artifact checksum。已发表 baseline 数字保留其原论文口径，不将 dataset-family comparison 描述为严格相同资源、模型规模或训练监督的 apples-to-apples comparison。

## Claim Atomization Reliability Study

为审计 claim atomization 这一上游输入，我们从 LIAR-RAW 与 RAWFC validation data 各抽取 100 条 claims（70% 随机、30% 困难优先），得到 257 个 atoms。两位标注者独立评估 faithfulness、atomicity 与 completeness；所有主维度 exact mismatches 以及一个 claim-level 内部冲突均由第三位标注者在看不到 A/B 标签时独立仲裁。

| Dimension | Unit / Gold N | Final gold pass | Pre-adj. Exact | Cohen's $\kappa$ | Gwet AC1 |
|---|---:|---:|---:|---:|---:|
| Faithfulness | atom / 257 | 99.22% | 95.72% | 0.133 | 0.955 |
| Atomicity | atom / 257 | 95.72% | 88.72% | 0.306 | 0.866 |
| Complete coverage | claim / 200 | 99.00% | 95.48% | -0.023 | 0.953 |

表X表明，由LLM完成的 claim atomization 在本次 LIAR-RAW/RAWFC 审计样本上高度符合独立人工质量判断。最终 human gold 中，99.22% 的 atoms 被判定为 faithful，95.72% 满足单一、可独立核验的 atomicity 要求，且 99.00% 的 claims 获得 complete atom coverage；对应的 dataset-stratified claim-cluster bootstrap 95% 区间分别为 98.02%--100.00%、92.80%--98.12% 和 97.50%--100.00%。即使要求同一 claim 的所有 atoms 同时通过 faithfulness 与 atomicity，并完整覆盖原 claim，仍有 187/200（93.50%）达到 strict pass。因此，该实验支持将当前 atomization 视为质量较高且可用于后续 retrieval 与 Evidence Map 构建的可靠上游结构输入。

## Evidence Map Annotation Reliability Study (Exp2; Placeholder)

**占位说明。** Exp2 将独立审计 Evidence Map 中 candidate--atom pair 的 `relation`、`directness` 与 `confidence` 标注。当前版本只冻结结果位置与报告口径；在正式双标、分歧仲裁和 artifact audit 完成前不填入数值，也不据此扩展本文的结果或贡献表述。

| Field | Human reliability | Planned LLM comparison | Diagnostic |
|---|---|---|---|
| Relation | Cohen's $\kappa$ | Overall and per-relation accuracy | Confusion matrix |
| Directness | Spearman $\rho$ | Ordinal agreement (TBD) | Ordinal error analysis |
| Confidence | TBD after target definition | TBD after target definition | Calibration / ECE target TBD |

表注：Exp2 衡量结构标注本身的可靠性；RQ4 的 component ablation 衡量 map signals 对下游结果的敏感性，二者不是同一个问题。`gold_confidence` 记录的是人工标注者自信度，填入结果前需另行冻结其与 LLM confidence 的比较及校准目标，不能预先把它等同于事实 gold。

## Main Results

### LIAR-RAW and RAWFC Controlled Results

#### Matched Internal Results

| Dataset | Verifier / adaptation | Evidence setting | Acc. | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---:|---:|---:|---:|
| LIAR-RAW | Ministral-3-8B / LoRA | Structure-only, full-pool-access ordering, minmax$(5,10)$ | 35.17 | 38.38 | 34.83 | 35.40 |
| RAWFC | Ministral-3-8B / LoRA | Structure-only, full-pool-access ordering, minmax$(5,10)$ | 65.00 | 65.26 | 65.01 | 65.08 |

**结果表述。** 在相同 Ministral-3-8B/LoRA adaptation family、但按数据集独立训练并仅依赖各自 validation 选择 checkpoint 的设置下，structure-only 方法在 LIAR-RAW 与 RAWFC 上分别取得 35.40 和 65.08 macro-F1。RAWFC 结果略高于表内原始 associated-report setting 的 DeReC-qwen（64.60）和 G-Defense default（64.31），因此支持“在该比较范围内具有竞争力”的表述；不同方法的模型规模、训练监督与运行次数并不完全一致，本文不据此宣称整体 SOTA。两个数据集的结果共同说明 typed structure 可以在不使用 verdict-conditioned selector supervision 的情况下支撑 raw-report fact checking；它们不单独证明 audit trace 是模型预测的忠实解释。


#### Published Raw-report Comparisons

下表按资源与协议分块。方法名逐行引用原论文，数值统一显示为 percentages。CofCED、L-Defense 与 G-Defense 明确报告 macro P/R/F1；DeReC、FFRR、FactLLaMA 与 KG-CRAFT 的原表只标注 P/R/F1，未进一步说明 classification averaging，因此本文保留原论文指标名而不替其补写 macro 口径。

| Method | Comparison scope | LIAR-RAW P / R / F1 (FactLLaMA: LIAR) | RAWFC P / R / F1 |
|---|---|---:|---:|
| **Same dataset family and original associated-report setting** | | | |
| CofCED \citep{Yang2022CofCED} | Cascaded report/sentence selection | 29.48 / 29.55 / 28.93 | 52.99 / 50.99 / 51.07 |
| L-Defense LLaMA2 \citep{Wang2024LDefense} | Raw reports + generated competing explanations; best of 10 runs | 31.63 / 31.71 / 31.40 | 60.95 / 60.00 / 60.12 |
| L-Defense ChatGPT \citep{Wang2024LDefense} | Raw reports + generated competing explanations; best of 10 runs | 30.55 / 32.20 / 30.53 | 61.72 / 61.01 / 61.20 |
| G-Defense default \citep{Wang2026GDefense} | Multi-stage graph defense; 3-run mean | 33.09$\pm$1.01 / 31.55$\pm$0.84 / 31.55$\pm$0.51 | 64.99$\pm$0.78 / 64.52$\pm$1.02 / 64.31$\pm$1.20 |
| DeReC-qwen \citep{Qazi2025DeReC} | Dense top-10 retrieval + DeBERTa-v3-large | 35.94 / 32.24 / 33.13 | 65.58 / 64.56 / 64.60 |
| **Ours** | Ministral-3-8B + LoRA; dataset-specific training | 38.38 / 34.83 / 35.40 | 65.26 / 65.01 / 65.08 |
| **Modified or external evidence setting** | | | |
| FFRR(d+q) \citep{Zhang2024FFRR} | Cleaned corpus + label-grounded LLM feedback | 34.50 / 32.60 / 33.50 | 56.50 / 57.40 / 57.00 |
| FactLLaMA (instruction-tuning with external knowledge) \citep{Cheung2023FactLLaMA} | Google-retrieved evidence + LoRA | 32.46 / 32.05 / 30.44 | 56.11 / 55.50 / 55.65 |
| **Multi-stage / large-LLM contextual reference** | | | |
| KG-CRAFT$_{\mathrm{L3.3}}$ \citep{Lourenco2026KGCRAFT} | Claude KG extraction + Llama-3.3-70B multi-stage reasoning | 77.38 / 70.67 / 73.87 | 81.63 / 81.53 / 81.58 |

数值转录来源固定如下：CofCED 来自 \citet{Yang2022CofCED} Table 2；L-Defense 两个 backbone variants 来自 \citet{Wang2024LDefense} Table 2，并按其附录说明报告 10 次运行中的 best；默认 G-Defense 来自 \citet{Wang2026GDefense} Table 2，为 3-run mean；DeReC-qwen 来自 \citet{Qazi2025DeReC} Table 3，并以正式表格中的 F1 而非摘要中存在冲突的数值为准；FFRR(d+q) 来自 \citet{Zhang2024FFRR} Table 2；FactLLaMA 的 RAWFC 与 LIAR 数字分别来自 \citet{Cheung2023FactLLaMA} Tables I and II；KG-CRAFT$_{\mathrm{L3.3}}$ 来自 \citet{Lourenco2026KGCRAFT} Table 1。

FFRR 使用移除潜在泄漏文档后的清洗语料，因此不被视为完全相同 candidate corpus。FactLLaMA 原论文将其第一项数据称为 LIAR，并通过 Google API 重新检索 external knowledge，而非直接使用 benchmark associated-report pool；该行只作 contextual comparison。KG-CRAFT 使用 Llama-3.3-70B，且其 KG 抽取和多阶段调用预算显著更高。DelphiAgent 保留在 Related Work，但在无法从原论文直接核实完整 backbone 与结果表之前不列入数值对比。

### SciFact Full-Pipeline Development Results

| Method | Year | Sent. Selection-only | Sent. Selection+Label | Abstract Label-only | Abstract Label+Rationale |
|---|---:|---:|---:|---:|---:|
| VeriSci \citep{Wadden2020SciFact} | 2020 | 48.30 | 43.10 | 52.10 | 50.00 |
| VerT5erini \citep{Pradeep2021VerT5erini} | 2021 | 60.87 | 57.10 | 65.07 | 61.72 |
| ParagraphJoint \citep{Li2021ParagraphJoint} | 2021 | 64.70 | 55.20 | 65.10 | 59.90 |
| ARSJoint \citep{Zhang2021ARSJoint} | 2021 | 66.20 | 57.80 | 66.70 | 62.40 |
| QMUL-SDS \citep{Zeng2021QMULSDS} | 2021 | 67.83 | 60.54 | 63.40 | 61.10 |
| RerrFact \citep{Rana2022RerrFact} | 2022 | 76.37 | 63.76 | 64.59 | 64.02 |
| PrunE \citep{Fang2025PrunE} | 2025 | 62.96 | 53.29 | 63.21 | 60.10 |
| **Ours: structure-only** | -- | 39.95 | 38.65 | 71.98 | 64.78 |

方法归属逐行引用其原论文；VeriSci 至 RerrFact 的 development-set 数值转录自 \citet{Rana2022RerrFact} Table 4，PrunE 数值来自 \citet{Fang2025PrunE} Table 2。PrunE 使用 top-150 bigram-TF-IDF universe，并在 development inference 中采样 12 个候选摘要，因此只导入其自身结果行，不混入其重训 baseline rows。

**结果表述。** 已完成的 structure-only SciFact evaluation 获得 71.98 abstract Label-only F1 和 64.78 abstract Label+Rationale F1；sentence Selection-only 与 Selection+Label F1 分别为 39.95 和 38.65。该结果说明相同的结构监督流程能够在 SciFact official-development setting 中完成端到端评估，但也显示 abstract-level label prediction 与 exact sentence localization 之间仍有明显差距。单独报告的 verifier-only claim-label validation diagnostic 为 82.97 macro-F1，不替代上述官方 full-pipeline 指标。由于同一 development split 参与 checkpoint selection，本文不据此宣称 zero-shot transfer、hidden-test 或整体 SciFact SOTA。

## Selector Mechanism Ablation

为了回答RQ3，从而验证组件增益来源，本消融验证证据选择机制的各组件（候选池来源、学习与否、证据顺序）的必要性。所有实验均在 LIAR-RAW 上进行，使用与主方法相同的 verifier 训练设置（Ministral-3-8B + LoRA）。设计如下：

- **s0 no_evidence**：verifier 不接收任何证据，仅看 claim，作为性能下界。
- **s1 random**：从未经检索的原始证据池中随机选择证据。
- **s2 baseline_pool_only**：selector 仅用raw claim 产生的候选池 baseline pool。
- **s3 atom_route_only**：仅用 atom-route 产生的候选池，不用 baseline pool，验证 union 的必要性。
- **s4 no_selector**：完整 Atom-Union 池，但只使用原始检索分数，不经过selector。
- **s5 no_learned**：完整 map 驱动的贪心构链，但 selector 权重为手工设定（无学习）。
- **s6 shuffle**：用主方法的 learned selector 生成 trace 后打乱顺序，验证证据有序性的价值。

| 变体 | Acc. | Macro-F1 |
|---|---|---|
| s0 no_evidence | 30.70 | 31.02 |
| s1 random | 31.25 | 31.68 |
| s2 baseline_pool_only | 33.81 | 34.51 |
| s3 atom_route_only | 31.15 | 32.11 |
| s4 no_selector | 35.01 | 35.37 |
| s5 no_learned | 34.85 | 34.92 |
| s6 trace\_shuffle | 31.7 | 32.90 |
| full | 35.97 | 36.66 |

表X报告了各消融设置在 LIAR-RAW 上的 verifier 性能，结果支持以下关键结论：

1. 证据本身有用，random优于no_evidence，即使随机证据也能带来提升，说明证据对判别有基础价值。
2. baseline_pool_only与atom_route_only都优于random，表明证据经过一定程度的处理后，已经能取得不错的性能，同时raw claim 产生的候选池起到的是主导作用
3. s4与s5则证明证据经过进一步组织，能得到更近一步的性能。同时表明，仅靠map驱动构链而不经过权重学习，达不到最优性能。
4. 而s6的结果则指明了，证据的排序方式对最终结果有较大影响，这说明在目前基于注意力机制的判别器的制约下，需要关注证据的排序而不能只关注证据选择。

### Evidence Map Ablation

Atom-evidence map $M(u_j,a_i)=(r_{ij},d_{ij},c_{ij})$ 是 selector 决策的结构化中间表示，多个边际特征均依赖它，且 greedy chain 的 atom 状态转移与 minmax 截断位置 $K^\ast$ 也由 relation 与 directness 驱动。为了回答RQ4，本消融逐元去除 map 的三个信号，检验每个信号对最终判别的贡献。

本实验设置如下：所有变体共享与主方法完全相同的设置，仅改变 map 信号输入；selector 权重在各自退化特征上重新训练。具体变体如下：

- **full\_map**（主方法）：完整 map，无退化，作为基准。
- **no\_relation**：将每个 pair 的 relation $r_{ij}$ 退化为 background，使 atom 永不进入 $S/R/Q$。
- **no\_directness**：将 directness $d_{ij}$ 退化为中性档，使解析增益 $\delta(d)$ 下降，且无法独立触发 OPEN 转移。
- **no\_confidence**：将 confidence $c_{ij}$ 强制为 1.0，使 $\max(c,0.5)$ clip 失效，解析增益被人为放大。
- **no\_map**：$r,d,c$ 全部退化，6 个 map 相关特征置零，selector 仅靠 retrieval/text novelty。

| 变体 | Acc. | Macro-F1 | $\bar\rho$ | $\bar{K^\ast}$ | $K^\ast{=}5$ 占比 |
|---|---|---|---|---|---|
| **full\_map** | 35.97 | 36.66 | 0.774 | 6.23 | 68.9% |
| no\_relation | 35.25 | 35.53 | 0.000 | 9.69 | 0.8% |
| no\_directness | 34.37 | 34.16 | 0.000 | 9.69 | 0.8% |
| no\_confidence | 33.01 | 33.19 | 0.774 | 6.23 | 68.6% |
| no\_map | 33.57 | 34.32 | 0.000 | 9.69 | 0.8% |

表X报告 LIAR-RAW 上的 verifier 性能，以及各变体 trace 的平均 atom 解析率 $\bar\rho$ 与 minmax 截断后的平均证据容量 $\bar{K^\ast}$。$\bar{K^\ast}$ 直观反映 map 对证据容量的调控：解析率越高，越多样本在 $k_{\min}{=}5$ 处提前停止。有以下观察：

1. 完整 map 优于 no\_map，证明 LLM 结构化标注为 selector 提供了 retrieval score 之外的有效信号。
2. 证据容量调控是 map 价值的主要通道。 full\_map 有 68.9% 的样本在 $K^\ast{=}5$ 提前停止（解析达标），平均仅用 6.23 条证据；而 no\_map / no\_relation / no\_directness 因解析失效，多数样本跑满 $K^\ast{=}10$，显示为 $\bar{K^\ast}$ 接近10。这说明 map 的核心作用不只是"选更好的证据"，更是"判断何时证据已足够"，从而在证据充分性与上下文噪声之间取得平衡。
3. relation / directness / confidence分别提供了不同程度的增益，值得注意的是，尽管no\_relation数值上接近full_map，但是其证据容量增加了50%，这说明 map 的核心价值在于让 selector 用更少的证据、更低的上下文成本达到同等甚至更优的判别，而非仅在最终 F1 上拉开差距；在证据预算受限或长上下文噪声敏感的场景下，这一效率优势更为关键。

### Verifier-Visible Evidence Capacity and Prefix-Policy Sensitivity

Prompt evidence policy 将 learned selector 生成的 audit ordering 投影为 verifier 实际可见的 evidence prefix，并由此决定最终证据容量 $K^\ast$。为隔离下游容量控制与上游证据排序的影响，本消融固定 claim atoms、candidate pool、Evidence Map、learned selector weights 及其生成的 full-pool-access ordering，仅改变 verifier-visible prefix policy。每种 policy 均一致地用于构建 train、validation 和 test prompts，并在相同训练配置下分别训练 verifier，使用 validation performance 选择 checkpoint。

本文比较以下三类 policy：

- **fixed-$k$**：保留 audit ordering 的前 $k$ 项，对应 minmax$(k,k)$，其中 $k\in\{3,5,7,9\}$。
- **minmax$(k_{\min},k_{\max})$**：至少保留 $k_{\min}$ 项；此后若 map-induced structural coverage proxy 达到目标则停止，否则最多扩展至 $k_{\max}$。取 $(k_{\min},k_{\max})\in\{(3,8),(3,10),(5,10),(5,12),(7,12)\}$，其中 $(5,10)$ 为主方法配置。
- **budget-$B$**：依据累计 evidence-side token cost 截取不超过预算 $B$ 的最长前缀，其中 $B\in\{512,768,1024\}$。

上述 policy 首先给出目标前缀，renderer 随后统一执行 1,024-token context guard；当 audit trace 本身短于目标容量或 context guard 删除尾部 evidence 时，最终 verifier-visible evidence count $K^\ast$ 会小于 policy 的目标容量。

| Prefix policy | 参数 | Acc. (%) | Macro-F1 (%) | $\bar{K^\ast}$ | $\bar{T}_{\mathrm{prompt}}$ | Prompt 截断率 |
|---|---:|---:|---:|---:|---:|---:|
| fixed | $k{=}3$ | 33.17 | 34.12 | 2.99 | 436.33 | 0.00% |
| fixed | $k{=}5$ | 33.25 | 33.91 | 4.95 | 603.45 | 0.16% |
| fixed | $k{=}7$ | 34.13 | 34.03 | 6.84 | 751.29 | 3.60% |
| fixed | $k{=}9$ | 34.77 | 34.81 | 8.36 | 867.65 | 25.18% |
| minmax | $(3,8)$ | 33.17 | 33.84 | 4.24 | 540.19 | 4.16% |
| minmax | $(3,10)$ | 33.57 | 33.86 | 4.54 | 562.51 | 12.07% |
| **minmax（主方法）** | **$(5,10)$** | **35.97** | **36.66** | 5.96 | 677.90 | 12.15% |
| minmax | $(5,12)$ | 35.17 | 35.22 | 6.07 | 686.65 | 20.62% |
| minmax | $(7,12)$ | 33.25 | 34.50 | 7.47 | 798.46 | 22.70% |
| budget | $B{=}512$ | 35.01 | 36.05 | 9.17 | 923.07 | 30.46% |
| budget | $B{=}768$ | 34.05 | 34.93 | 9.47 | 949.29 | 93.37% |
| budget | $B{=}1024$ | 34.45 | 35.03 | 9.47 | 949.29 | 93.45% |

表X报告 LIAR-RAW test set（$n{=}1251$）上的 accuracy、Macro-F1 与实际上下文成本。$\bar{K^\ast}$ 表示经过 context guard 后 verifier 实际接收的平均 evidence 数量；$\bar{T}_{\mathrm{prompt}}$ 表示不含 target 的完整输入 prompt 的平均 tokenizer token 数；Prompt 截断率表示 policy 给出的目标前缀被统一的 1,024-token guard 删除尾部 evidence 的样本比例。

该容量实验体现出几个关键观察:1) 固定容量增加并未带来稳定的 Macro-F1 增益: fixed-$k$ 的 accuracy 随 $k$ 从 3 增至 9 整体上升，但 Macro-F1 在 34.12、33.91、34.03 和 34.81 之间波动；与此同时，$\bar{K^\ast}$ 从 2.99 增至 8.36，$\bar{T}_{\mathrm{prompt}}$ 从 436.33 增至 867.65，Prompt 截断率从 0.00% 增至 25.18%。这表明扩大固定 prefix 会持续增加上下文成本，却不保证相应的 Macro-F1 提升; 2) 结构条件停止呈现出更有利的性能—成本组合: 相比表现最好的 fixed 配置 $k{=}9$，minmax$(5,10)$ 的 Macro-F1 数值上高 1.85 个百分点，同时平均少使用 2.40 条 evidence 和 189.75 个 prompt tokens；相比 budget-$B{=}512$，其 Macro-F1 数值上高 0.61 个百分点，同时平均少使用 3.22 条 evidence 和 245.17 个 prompt tokens。

总体而言，在上游 learned selector ordering 固定时，判别性能并不随 verifier-visible evidence 数量或 prompt 长度单调提高，prefix 的选择依据与有效容量需要共同考虑。在本实验网格中，利用结构覆盖状态进行样本级停止的 minmax$(5,10)$ 取得了最高性能，并以中等上下文成本优于固定取证和单纯扩大 token budget。

# Limitations

首先，claim decomposition 并不总能稳定提高事实核查表现，错误拆分、遗漏限定条件或过度细分仍可能向 retrieval 与 Evidence Map 传播 \citep{Hu2025DecompositionDilemmas}。为审计并量化这一风险，我们在 200 条 claims、257 个 atoms 上完成两位标注者的独立双标与第三人盲化仲裁。最终 gold 的 faithfulness、atomicity 与 complete coverage 通过率分别为 99.22%、95.72% 和 99.00%。这些结果支持本研究审计样本内的 atomization artifact 高度符合人工质量判断，并将 atomicity 定位为主要残余风险。由于样本来自两个 validation set 的等量抽样且过采样困难样本，该结论不能外推为 claim decomposition 普遍可靠，也不能证明它会因果性地改善 downstream verification。

第二，Evidence Map 仍依赖 LLM API，且 Exp1 只覆盖 claim atomization，不能外推为对 relation、directness 或 confidence 标注的验证。Exp2 的独立双标、仲裁与校准分析仍待完成；冻结缓存、prompt/schema hash、调用日期与调用元数据提高了可复现性和 artifact-level 可审计性，但不能将这些结构标注等同于人工 gold supervision。

第三，source linkage 不等于 source credibility。Trace 能回到原始 report 和 span，但本文没有判断新闻来源是否可信，也没有防止多个报道复制同一错误信息。上游 candidate pool 的召回上限同样限制所有后续选择。

第四，map-induced covered-atom rate 是结构代理，不等价于人工或逻辑意义上的 evidence sufficiency。既有研究表明 verifier 可能在缺失信息时仍做出高置信度判断 \citep{Atanasova2022Insufficient}。本文仅将 covered-atom rate 用作 prefix projection 代理，不对其与真实性、充分性或下游性能之间的关系作经验性结论。

第五，ordered evidence trace 是系统构造过程的 inspectable artifact，不自动说明 verifier 的因果决策依据。Rationale 与解释研究区分可读性、sufficiency、comprehensiveness 和 faithfulness \citep{Lei2016Rationalizing,DeYoung2020ERASER,Jacovi2020Faithfulness}，而这些指标本身也可能受到分布偏移和优化目标的影响 \citep{Hsia2024Goodhart}。本文没有将 audit trace 表述为模型推理解释。

最后，greedy state-conditioned ordering 不保证全局最短或全局最优。本文的机制证据仅来自两个种子上的 fixed-$K$ validation diagnostics，且 verifier 与 input organization 的训练分布同时变化；它不支持显著性、因果、普遍顺序优势或稳健 co-adaptation 主张。主结果表以各数据集的单个 canonical run 为准，不提供多种子均值或方差。SciFact 结果来自 official development split 且该 split 参与 checkpoint selection，因而只支持该方法在相应 scientific fact-checking setting 中的描述性适用性结论。精确句级证据定位仍明显弱于 abstract-level verification。

# Conclusion

本文研究了在最终事实判定之前，如何显式构造 atom-indexed、source-linked evidence trace。方法将 claim atomization、Atom-Union retrieval 与 typed Evidence Map 连接起来，并以 map-induced state transitions 为主要偏好信号；map quality、retrieval relevance 与冻结候选索引只用于确定性区分，不读取 verdict label、gold/teacher-provided evidence order 或 verifier feedback。学习到的 state-conditioned structural scorer 访问完整候选池并产生 full-pool-access audit ordering，独立 prefix policy 再产生 verifier-visible evidence。Exp1 的独立双标与第三人仲裁显示，生成 atoms 在 faithfulness、atomicity 与 complete coverage 上分别有 99.22%、95.72% 和 99.00% 被最终 human gold 判为合格，支持自动 atomization 在本研究审计范围内高度符合人工质量判断，并可作为后续结构构建的可靠上游输入；这一证据只覆盖 claim atomization，不验证 Evidence Map 的 relation、directness 或 confidence。LIAR-RAW 与 RAWFC test evaluation、SciFact official-development evaluation，以及 LIAR-RAW two-seed fixed-$K$ diagnostics，共同支持这一 selector supervision boundary 在不同事实核查场景中的可行性；crossover 同时表明 organization effect 的方向和幅度会随 verifier training distribution 与训练种子变化。系统保留可回查的 source-linked evidence organization artifact，但本文不把它等同于 verifier 的忠实解释，也不主张普遍顺序优势。
