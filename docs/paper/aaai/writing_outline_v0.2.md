# AAAI写作大纲

该文档记录AAAI论文的写作大致逻辑，以及部分实际原文，作为论文的初稿中文版，下面的章节都按正式论文的章节排列。

# Abstract

社交媒体的发展提高了虚假新闻的传播效率，这使得基于证据的、可解释的自动虚假新闻检测方法成为研究热点。已有方法要么依赖复杂的证据特征工程，要么通过复杂的证据组织与变换来产出最终结果。这些方法往往依赖事后的解释，例如在真实性标签判定之后才单独输出解释文本。而在人类实际检测虚假新闻的场景中，人类往往按照证据一步一步地组织构成可解释的证据链，进而得到最终判定。这启发我们：将某条声明可得的所有证据按照特定方式切分，并按声明中的原子级事实逐个解释的顺序编排，从而得到一条有序证据链。该证据链既可以作为 LLM 的判定输入，也可以作为人类可读的可解释信息。经过这种简单的编排后，在 LLM 上的微调结果表明，我们的方法超越了大部分包含复杂证据组织形式的自动事实核查方法，同时在与引入额外外部证据以及完全使用闭源模型的 agent 方法的比较中也具有竞争性。

# Introduction

近年来，自动事实核查领域快速发展，从直接基于声明判断，发展到面向社交媒体的评论驱动检测，再到面向新闻的证据驱动检测系统。其中值得注意的是基于证据的检测系统：这种系统接受一个声明以及与该声明相关的、从互联网获取的各种原始报道，而后将这些报道通过各种方式精炼成可用于事实核查的数据结构。有的采用基于证据特征的级联方式 [CofCED]，有的基于 LLM 生成辩护 [L-defense] 或辩护图 [G-defense]，又或者构建复杂的 Agent 系统 [DelphiAgent] 来完成整个过程。而这些方法往往需要一套复杂的证据处理流程：如 [CofCED] 使用一套证据特征聚合机制，将完整证据池构建成特征向量供后续判别器分类；[L-defense] 则借助群体智慧思想，利用 LLM 生成对证据的正反辩护，而后再输入后续分类器；[G-defense] 则更进一步，该方法把一条完整声明拆成多个子声明构成图，而后为每个声明都生成正反辩护，最后将图序列化接入分类器。通过大致观察可发现，这一系列方法最终输入到分类器的数据结构都已经不再是原始证据的形式，导致判别过程黑箱化，可解释性受限。

人类在进行事实核查时，会经过多个步骤：1）信息收集：对一个声明，收集其所有相关的原始报告并整理；2）要素抽取：把一个声明拆分成多个原子事实，每个原子事实都包含最小的子命题；3）证据组织：判断每一条证据支持了声明的哪一部分，并组织成一条逻辑链；4）来源评估及矛盾识别：检查 report 来源的可信度，以及处理证据中存在的矛盾；5）结论形成：根据所有信息得出最终判定。

随着大模型技术的发展，大模型展示出了强大的逻辑推理能力，使其可以完成许多以前人类才能完成的推理流程。结合人类事实核查流程，这启发我们思考：能否借助大模型的推理能力，构建一套模仿人类事实核查流程的方法。

# Related Work

少量早期 BERT 之前的 AFC 方法。

部分重要的 BERT 类方法。

以重要的篇幅讨论使用 LLM 的各种方法：基于辩护的、基于智能体的、基于内在表示的、基于图的等，并围绕这一系列方法复杂度高、黑箱化的问题展开。

> 注：定稿时补全为完整段落，明确每类方法的代表工作，并收束到“复杂度高、黑箱化”的论点，与 Introduction 呼应。

# Methodology

## Overview

本文提出了一套 AFC 方法：从原始声明出发，将其分解成最小的、独立的、语义完整的原子事实，基于这些原子事实完成三件事：1）对原始 report 的基于 atom 语义的 chunking；2）利用 atom 检索分解后的 chunks 扩充基础证据池；3）围绕 atom 逐步构建证据链。而后，将构建得到的证据链接入大模型进行推理与分类，得到最终判定；该证据链对人类与大模型均统一可读，既保证可解释性，又能充分利用大模型的推理能力。

## 符号约定

> 注：定稿时补全核心符号表，包括 \(c, \mathcal{A}, \mathcal{R}, \mathcal{T}, u_j, a_i, h_i^{(t)}\) 等。

## 声明与证据的前处理

### claim-aware Chunking

在做后续证据组织之前，如何确定证据单元的粒度是一个关键问题。若粒度过小（如单句），则可能导致单个证据单元得不到有效信息；若粒度过大（如整段 report），单个证据单元会包含过多信息，为后续处理引入噪声。因此本文实现了一个同时关注证据单元对 claim 的相关性以及在段落内连贯性的 chunking 策略，具体关于证据粒度的实验见 XXX。

设 claim 为 \(c\)，一篇 report 包含的所有句子为 \(s_1,\dots,s_n\)，句向量为 \(e_i\)，claim 向量为 \(e_c\)，本文使用统一的 embedding 模型得到向量 \(E(s)=e\)。

首先对每个句子计算 claim-aware relevance：
\[
r_i = 0.70\,\mathrm{norm}(\cos(e_i,e_c)) + 0.20\,\mathrm{norm}(\mathrm{LexF1}(c,s_i)) + 0.10\,\mathrm{norm}(\mathrm{BM25}(c,s_i)).
\]

然后，对相邻句子边界打分 \(i|i+1\)：
\[
b_i = w_{\text{sem}}(1-\cos(e_i,e_{i+1})) + w_{\text{rel}}|r_i-r_{i+1}| - d_{\text{coref}}.
\]
其中，\(w_{\text{sem}}=0.75\)、\(w_{\text{rel}}=0.25\)，\(d_{\text{coref}}\) 惩罚连续指代，降低错误切开的概率。此时 \(b_i\) 越大，该边界越有可能成为分段边界。得到一系列分数后，通过计算局部峰值来拆分组合句子。

最终每个 chunk 记录：
\[
u_j = [s_a,\dots,s_b].
\]

### Claim Decomposing

在一般的人类声明验证流程中，通常不会同时验证多个事实，这样不仅会互相干扰，还会导致边界模糊不清。因此，参考该流程以及一般的做法（参考文献），我们把原始 claim \(c\) 转成少量可独立验证的 atomic propositions，作为后续 atom-conditioned retrieval、evidence map、MREC 状态转移的“状态变量”。

输入是一条 claim，输出是：
\[
A(c)=\{(a_1,q_1),\dots,(a_m,q_m)\},\quad 1\le m\le 6
\]
其中 \(a, q\) 分别为分解后的原子命题及其对应的查询，该查询在后面用于检索证据单元。实现上，该分解通过 LLM 完成，我们在 prompt 中要求：只使用 claim 本身，不引入外部知识；只在 claim 含有多个可分别验证的事实断言时拆分；日期、数量、否定、比较对象、地点、范围、归因等必须保留在 proposition 内，不能拆成孤立片段。即每个 atom 不能只是关键词片段，而是一个个可验证的、语义完整的原子命题。

### Atom-Union Evidence Pool

每个 claim 经 chunking 后得到的证据单元可能多达数十个，而为了降低后续证据链的构造难度，我们对完整的证据池进行处理，为每个 claim 检索其对应的有限个证据单元。在实现上，为了加大召回率，我们把原来“用整条 claim 检索证据”改成“每个 atom 各自检索一批证据”，再合成一个 atom-aware candidate pool。

对每个 atom \(a_i\)，取它的查询 \(q_i\)，在该 claim 的 evidence chunk pool \(\mathcal{U}=\{u_j\}\) 上打分：
\[
s(q_i,u_j)
=0.70\,\mathrm{norm}(\cos(e_{q_i},e_{u_j}))
+0.20\,\mathrm{norm}(\mathrm{LexF1}(q_i,u_j))
+0.10\,\mathrm{norm}(\mathrm{BM25}(q_i,u_j)).
\]

然后每个 atom 保留 top-\(k_1\) routes：
\[
R_i=\operatorname{TopK}_{k_1}\{u_j:s(q_i,u_j)\}.
\]

多个 atom 的 top-\(k_1\) 结果会按证据单元去重、聚合以及排序得到基础候选池。

记所有 atom 检索结果的并集，即最后得到的候选池为：
\[
\mathcal{R}=\bigcup_{i=1}^{m} R_i.
\]

我们把 \(\mathcal{R}\) 定义为 Atom-Union 后送入 selector 的候选证据池。值得说明的是，本文的 learned marginal selector 可细化为“状态条件化的边际效用选择器”：它不是对 \(\mathcal R\) 做一次静态 top-k 排序，而是在每一步根据已经选过的证据、当前 atom 状态和候选的 evidence-map 标注，重新估计每个候选的边际贡献。具体见下一节。

## Evidence Selector

完成证据候选池的构建后，开始进行证据链的选择。我们的选择器遵从一个简单原则：“在每一步证据选择时，优先选对解决所有 claim atom 最有帮助的证据”。这种帮助在本文中被定义为：
1）该证据直接支持/反对了一条尚未被 solve 的 claim atom；
2）该证据为已解析的 atom 提供新的支持/反对立场；
3）该证据为已有的 evidence 提供有效的背景信息/反对意见。

根据这一原则，本文设计了一种证据选择器。

对于某个 claim，给定 Atom-Union pool \(\mathcal R=\{u_j\}_{j=1}^{n}\) 和 claim atoms \(\mathcal A=\{a_i\}_{i=1}^{m}\)，我们为所有 claim atom 以及 evidence 构建一个 map，该 map 为每个候选证据和 atom 提供结构化标注：
\[
M(u_j,a_i)=(r_{ij}, d_{ij}, c_{ij}),
\]
其中 \(r_{ij}\) 是关系类型，\(d_{ij}\) 是 directness，\(c_{ij}\in[0,1]\) 是置信度。实现上，该标注由 LLM 通过设计的 prompt 给出。

我们为每个 atom 定义了一组验证状态，通过 selector 维护：
\[
h_i^{(t)}\in\{U,S,R,Q,C\},
\]
分别表示 unresolved, supported, refuted, qualified, conflicted。初始时所有 atom 均为 \(U\)。

在第 \(t\) 步，selector 对每个未选择候选 \(u_j\) 计算状态条件化的边际特征。若 \(u_j\) 对 atom \(a_i\) 给出可解析关系，则其解析增益可写为：
\[
g_{ij}^{(t)}
=
p_i^{(t)}(U)\cdot \delta(d_{ij})\cdot \max(c_{ij},0.5),
\]
其中 \(p_i^{(t)}(U)\) 是当前 atom 仍未解析的概率质量，\(\delta(d_{ij})\) 为一组根据 directness 程度递增的手动权重。于是候选的解析边际贡献为：
\[
\phi_{\text{res}}^{(t)}(u_j)
=
\frac{1}{m}\sum_{i=1}^{m}
\max_{M(u_j,a_i)} g_{ij}^{(t)}.
\]

同时计算信息增益、覆盖增益、新关系增益、立场张力、佐证增益、来源/文本新颖性、map 质量、检索分数和长度代价：
\[
\phi^{(t)}(u_j)=
[
\phi_{\text{res}},
\phi_{\text{ent}},
\phi_{\text{cov}},
\phi_{\text{new-rel}},
\phi_{\text{tension}},
\phi_{\text{corr}},
\phi_{\text{src-novel}},
\phi_{\text{text-novel}},
\phi_{\text{conf}},
\phi_{\text{map}},
\phi_{\text{ret}},
\phi_{\text{cost}}
].
\]

Selector 用一个线性边际效用函数给候选打分：
\[
U_\theta(u_j\mid \mathcal T_{<t},H^{(t)})
=
b+
\sum_{\ell\neq \text{cost}} w_\ell \phi_{\ell}^{(t)}(u_j)
-
w_c \phi_{\text{cost}}^{(t)}(u_j).
\]

我们为这组效用特征构建了一个 pairwise 学习器，其目的是让 selector 学习到哪些证据是“好”的，此处的好即为本节开头所说的简单原则。训练时，在 evidence-map feature rows 上模拟 rollout，构造偏好对 \((u^+,u^-)\)，其中 \(u^+\) 是 proxy 排序更优的候选，\(u^-\) 是较差候选。proxy 来自一组手工定义的规则，根据 resolving / 新 atom 覆盖 / 新 relation 等来获得排序。

pairwise 学习器的优化目标为：
\[
\mathcal L(\theta)
=
\frac{1}{|\mathcal P|}
\sum_{(u^+,u^-)\in\mathcal P}
\log\left(
1+\exp\left(
-\left[
U_\theta(u^+) - U_\theta(u^-)
\right]
\right)
\right).
\]

实现中使用 Adam 优化，并用 softplus 参数化保证权重非负：
\[
w_\ell=\mathrm{softplus}(\theta_\ell),\quad
w_c=\mathrm{softplus}(\theta_c).
\]

推理时，selector 进行贪心构链：
\[
u_t
=
\arg\max_{u_j\in \mathcal R\setminus \mathcal T_{<t}}
U_\theta(u_j\mid \mathcal T_{<t},H^{(t)}).
\]

选中 \(u_t\) 后，selector 会生成一个 step：
\[
s_t=
(u_t,a_{i_t},h_{i_t}^{(t-1)}\rightarrow h_{i_t}^{(t)}).
\]
并更新 atom 状态。例如 support/refute/qualify 分别映射到 \(S/R/Q\)，若新证据与已有 \(S/R\) 状态冲突，则进入 \(C\)。最终输出的是一个 ordered evidence trace：
\[
\mathcal T=
[s_1,s_2,\ldots,s_T].
\]

值得注意的是，本文在 selector 部分使用了不少人工/启发式规则。实际测试表明，一些更复杂的规则（如根据 verifier 反馈学习 selector、人工标注学习）效果与该方式相当；出于简化原则，本文采用当前手动方法。具体实验与细节见 XXX。

## Verifier

在得到 ordered evidence trace \(\mathcal T=[u_1,u_2,\ldots,u_T]\) 后，我们使用一个指令微调后的 LLM 作为最终 verifier。Verifier 接收 claim 及其 prompt-visible evidence trace，并输出事实核查标签：
\[
\hat y
=
\arg\max_y p_\theta(y\mid x),
\]
其中 \(x=\mathrm{Render}(c,\mathcal T_{\mathrm{prompt}})\) 表示由 claim \(c\) 和被截断后的证据 trace \(\mathcal T_{\mathrm{prompt}}\) 构造的输入 prompt。训练阶段采用监督微调，最小化标准交叉熵损失：
\[
\mathcal L_{\mathrm{verifier}}
=
-\log p_\theta(y^\ast\mid x),
\]
其中 \(y^\ast\) 为样本的 gold fact-checking label。

由于 selector 生成的完整 trace \(\mathcal T\) 长度存在显著差异，直接将完整 trace 输入 verifier 可能导致长上下文噪声增加、有效证据信号被稀释，并显著提高训练与推理成本。因此，我们在 verifier 之前引入 prompt evidence policy，将完整 ordered trace 映射为 verifier 可见的 evidence prefix：
\[
\mathcal T_{\mathrm{prompt}}
=
[u_1,u_2,\ldots,u_{K^\ast}],
\quad
K^\ast \le T.
\]

具体地，我们沿 selector 给出的顺序逐步加入 evidence step，并在满足最小证据数 \(k_{\min}\) 后检查当前 trace 是否已经达到 atom-level resolution target。令
\[
\rho_t
=
\frac{
|\{a_i:h_i^{(t)}\in\{S,R,Q,C\}\}|
}{
|\mathcal A|
},
\]
其中 \(\rho_t\) 是第 \(t\) 步后的 resolved atom rate。若
\[
\rho_t \ge \rho_{\mathrm{target}},
\]
则认为当前 trace 已经达到目标解析状态，并停止继续加入证据。因此截断位置定义为：
\[
K^\ast
=
\min
\left\{
t:
t\ge k_{\min}
\land
\rho_t\ge \rho_{\mathrm{target}}
\right\}.
\]
若不存在满足条件的 \(t\)，则退化为最大证据数约束：
\[
K^\ast=\min(k_{\max},T).
\]
