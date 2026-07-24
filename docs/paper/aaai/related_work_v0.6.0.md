# v0.6.0 相关工作与潜在贡献定位

> **状态：文献核对初稿，2026-07-15。** 本文档服务于 `writing_outline_v0.6.0_plan.md`。它优先讨论与 Verifier-Grounded Marginal Evidence Selection（VGMES）直接构成 novelty collision 的工作，而不是追求自动事实核查的全面综述。正式投稿前仍需做一次系统检索并补齐 BibTeX。

## 1. 文献地图与核心判断

v0.6.0 位于五条已经相当成熟的研究线交叉处：

1. 自动事实核查中的 evidence retrieval、sentence selection 与 listwise evidence aggregation；
2. 使用下游 reader/verifier 反馈优化 retriever 或 context filter；
3. 从 relevance 转向 passage utility，并建模 utility 的上下文依赖；
4. iterative/adaptive retrieval、证据充分性判断与动态停止；
5. rationale selection、输入干预和模型相对的 faithfulness。

因此，以下单点均**不能**再单独作为 v0.6.0 的 novelty claim：

- “使用 verifier feedback 训练 evidence retriever”；
- “用正确输出 likelihood 衡量 context utility”；
- “passage utility 依赖已有 context”；
- “根据当前置信度决定是否继续检索”；
- “选择紧凑、对预测足够的 rationale”。

更需要注意的是，近期工作已经几乎覆盖 VGMES 的数学原语：Iterative ICL 直接使用
“已选集合加入新项前后的正确输出概率差”作为逐步 reward；InfoGain-RAG 用单文档信息增益训练 reranker 并通过阈值过滤；FES-RAG 则已将 gold-target likelihood gain 蒸馏到轻量证据选择器。因此，**add-one likelihood gain、student selector 和 threshold stopping 都不是本文可单独主张的新意**。

v0.6.0 可能成立的差异必须是一个更窄、也更任务化的组合：在 raw-report 事实核查候选池内，将 **typed atom--role Evidence Map** 作为当前证据 bundle 的显式状态，学习 corroboration、opposition、context、source independence 和 redundancy 如何改变候选证据的 verifier-relative 条件边际收益，再用该估计联合决定下一条证据与实例级容量。这一组合能否形成贡献，取决于实验能否证明结构状态和集合条件化都有不可替代的作用，而不是只超过 BM25/top-K。

## 2. 自动事实核查中的证据检索与选择

### 2.1 Pipeline evidence retrieval

FEVER 将自动事实核查明确拆为文档检索、句子证据检索与标签预测，并为 SUPPORT/REFUTE 样本提供必要证据句标注；这奠定了后续将 evidence selection 作为独立子任务的主流范式（Thorne et al., 2018，[FEVER](https://aclanthology.org/N18-1074/)）。传统方法主要以 claim--sentence relevance 或 gold evidence classification 训练检索器，最终以 evidence precision/recall、label accuracy 与 FEVER score 评价。

后续研究逐步超越独立 pointwise sentence ranking。DOMLIN 采用两阶段句子选择，使第二阶段证据依赖先前检索结果，已经体现 prefix/context-conditioned retrieval 的早期形式（Stammbach and Neumann, 2019，[Team DOMLIN](https://aclanthology.org/D19-6616/)）。Jiang et al. 使用 T5 进行 listwise sentence selection 和 label prediction，强调多条 evidence 的联合推理及对噪声的鲁棒性（2021，[Exploring Listwise Evidence Reasoning](https://aclanthology.org/2021.acl-short.51/)）。DQN-FV 则将 claim、当前已选证据集和标签预测编码为状态，通过 DQN 逐句构造精简证据集（Wan et al., 2021，[A DQN-based Approach to Finding Precise Evidences](https://aclanthology.org/2021.acl-long.83/)）。因此，“多证据联合建模”、“后续证据依赖已有证据”和“序列式精简证据集”都并非新问题。

最新的 User-Centric Evidence Ranking 直接将目标设为“让充分证据尽早出现在排序前缀”，比较 one-shot 与 incremental ranking，并通过用户实验验证早期充分性和阅读成本（Alt et al., 2026，[User-Centric Evidence Ranking](https://aclanthology.org/2026.eacl-long.340/)）。它已显式研究 redundancy、complementarity 与 early sufficient prefix；VGMES 的差异只能放在 verifier-relative conditional utility 和 Evidence Map 结构状态上，不能声称首次进行有序证据选择。

### 2.2 Raw-report evidence distillation

与 FEVER 的封闭 Wikipedia 证据不同，CofCED 从与 claim 相关的 raw reports 中先选 top-K reports，再以 coarse-to-fine cascaded selectors 选取解释性句子，并联合完成虚假新闻判别（Yang et al., 2022，[CofCED](https://aclanthology.org/2022.coling-1.230/)）。它与本文共享 LIAR-RAW/RAWFC 的 raw-report 场景，但其 selector 主要学习 report/sentence importance，没有 claim-atom Evidence Map，也不将单条证据相对当前前缀的 verdict improvement 定义为监督。

RAV 将 hybrid retriever、ranker 和 fact verifier 端到端连接，使 claim-verification gradient 能反向优化 evidence selection（Zheng et al., 2024，[Evidence Retrieval is almost All You Need](https://aclanthology.org/2024.findings-acl.551/)）。因此，v0.6.0 不能将“让下游 verification signal 作用于 evidence selector”写成首创。它与 RAV 的差异应定位为：VGMES 冻结 teacher，将每个 prefix--candidate intervention 物化为可审计的 signed scalar label；selector 与 verifier 解耦训练，并显式评价 teacher-to-verifier transfer。

### 2.3 LLM modular fact checking

HiSS 将 claim 分解为 subclaims，并在模型对当前问题缺乏信心时调用搜索，再继续分层验证（Zhang and Gao, 2023，[HiSS](https://aclanthology.org/2023.ijcnlp-main.64/)）。Self-Checker 由 claim processor、query generator、evidence seeker 和 verdict counselor 等可插拔 LLM 模块组成（Li et al., 2024，[Self-Checker](https://aclanthology.org/2024.findings-naacl.12/)）。Chen et al. 则面向真实政治声明构建 claim decomposition、web retrieval、fine-grained retrieval、claim-focused summarization 与 verdict 的完整流程（2024，[Complex Claim Verification with Evidence Retrieved in the Wild](https://aclanthology.org/2024.naacl-long.196/)）。

这些系统支持 claim decomposition 和 iterative verification 的合理性，但多数通过 prompting/agent policy 决定检索行为。VGMES 的计划差异不是“也分解 claim”，而是把固定 atom universe 与 typed candidate--atom alignments 转成一个部署时可计算的 prefix state，并对每个候选预测反事实判别收益。

## 3. 从 relevance 到 verifier-derived utility

这是 v0.6.0 最关键、也最危险的直接近邻。

### 3.1 FER：事实核查中的 verifier feedback

FER 已经明确提出自动事实核查应从 relevance 转向 verifier-derived utility。它先获得 claim 的候选句集合，再用冻结 verifier 比较 gold evidence 与 retrieved evidence 下的 gold-label probability，以 utility-divergence loss 和 evidence-classification loss共同训练 fine-grained retriever（Zhang et al., 2023，[From Relevance to Utility](https://aclanthology.org/2023.findings-emnlp.422/)）。其核心 utility loss 可概括为：

\[
\mathcal L_{uti}
=y^{*\top}D_\phi(c,E^*)
-y^{*\top}D_\phi(c,R_\theta(c,S)).
\]

FER 与 v0.6.0 共享三个关键思想：verifier 冻结、gold label 参与 utility training、retriever 适配 verifier。因此，v0.6.0 不能声称首次提出 verifier-grounded evidence utility。

计划中的实质差异应明确为：

- FER 以 retrieved set 与 gold-evidence set 的整体 utility divergence 训练静态 fine-grained retriever；VGMES 不要求 gold evidence，而对同一 prefix 下每个候选执行 \(S\rightarrow S\cup\{e\}\) 配对干预；
- FER 的监督主要使最终 retrieved set 接近 gold-evidence verdict behavior；VGMES 学习 signed conditional marginal gain，允许一条证据在不同 prefix 下从正收益变成零或负收益；
- FER 不以同一 learned scalar 联合定义证据顺序和自适应停止；VGMES 计划用预测净增益同时决定 next evidence 和 capacity；
- VGMES 额外使用 typed atom-level Evidence Map 表示方向、directness、context、来源与重复状态，并要求相应消融。

这些差异必须通过 `FER-style static/set-level feedback` baseline 实证体现；只与普通 relevance retriever 比较不足以支撑贡献。

### 3.2 FFRR：黑盒 LLM 的强化检索

FFRR 是 v0.6.0 在任务和数据上最必须正面比较的近邻。它在 RAWFC/LIAR-RAW 新闻声明核查上，使用黑盒 LLM 对 gold label 的分数构造单文档、问题级和最终证据集 reward，再通过 REINFORCE 顺序采样证据（Zhang and Gao, 2024，[FFRR](https://aclanthology.org/2024.lrec-main.1209/)）。因此，VGMES 不能声称首次在 LIAR-RAW 上使用 verifier/LLM feedback、gold-label utility 或序列式证据策略。

两者仍有一个可检验的 credit-assignment 差异：FFRR 的单文档奖励近似 $F(\{e\})$，不依赖当前前缀 $S$；最终集合奖励 $F(S)$ 又没有显式分解为每个候选的局部边际。VGMES 计划缓存可重放的

\[
F(S\cup\{e\})-F(S)
\]

监督，并用 Evidence Map 显式解释该边际何时由重复、异源佐证、反向证据或 context 改变。此外，FFRR 的主要推理设定仍是 top-K；VGMES 拟将同一个校准的 candidate-level 净边际同时用于排序与容量分配。这些差异必须由 FFRR-style reward baseline 支撑，不能只作文字区分。

### 3.3 RAG/reader-supervised retrieval

更广泛的 RAG 文献早已使用下游生成目标训练 retriever。RAG 将文档视为 latent variables 并联合微调 retriever 与 generator（Lewis et al., 2020，[RAG](https://papers.nips.cc/paper_files/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)）；EMDR² 让 multi-document reader 的训练信号流向 retriever（Sachan et al., 2021，[EMDR²](https://proceedings.neurips.cc/paper/2021/hash/da3fde159d754a2555eaa198d2d105b2-Abstract.html)）；Atlas 在多种 knowledge-intensive tasks 上联合训练 retriever-reader（Izacard et al., 2023，[Atlas](https://jmlr.org/papers/v24/23-0037.html)）。

REPLUG 更接近冻结 teacher 设定：它将 LM 当作 black box，用各文档带来的 language-modeling likelihood 监督可训练 retriever，使 retriever 偏好降低 LM perplexity 的文档（Shi et al., 2024，[REPLUG](https://aclanthology.org/2024.naacl-long.463/)）。因此，“冻结 LM 给 retriever 打分”也不是 v0.6.0 的新颖点。本文必须强调其任务是 multi-evidence fact-checking decision、监督是 gold-verdict marginal intervention、状态是 atom-role structured prefix，而不是一般 language modeling likelihood。

## 4. Context filtering 与 passage utility prediction

### 4.1 FILCO：likelihood-based context usefulness

FILCO 先通过 string inclusion、lexical overlap 或 conditional cross-mutual information（CXMI）识别有用 context，再训练 context filter。其 CXMI 直接衡量给定 context 后 canonical output likelihood 的增加，并在包括 FEVER 在内的六个知识密集任务上减少 44--64% prompt 长度（Wang et al., 2023/2024，[Learning to Filter Context for RAG](https://arxiv.org/abs/2311.08377)）。这与：

\[
\Delta(e\mid S)
=\log P(y^*\mid S\cup\{e\})
-\log P(y^*\mid S)
\]

在数学形式上非常接近。

VGMES 不能把“用 likelihood difference 产生 context labels”作为核心创新。可辩护的差异是 FILCO 主要进行 sentence-wise context filtering，oracle context 通常相对原始 passage/canonical output构造；VGMES 对多个 sampled prefixes 重复进行 candidate intervention，显式学习同一 evidence 随集合状态变化的 signed marginal utility，并将其用于 sequential acquisition 与 stopping。实验中需要加入 `empty-prefix CXMI/static utility` baseline，证明 prefix conditioning 确有额外价值。

### 4.2 独立 passage utility judgments

Zhang et al. 系统研究 LLM 是否能区分 passage relevance 与 downstream QA utility，并比较 pointwise、pairwise、listwise 和 repeated-sampling judgments（2024，[Are LLMs Good at Utility Judgments?](https://doi.org/10.1145/3626772.3657784)）。其结果也提醒本文：直接让 LLM 自评 usefulness 会受到 prompt form、顺序和 counterfactual passage 的影响。因此 v0.6.0 选择行为差值而非语言化 utility rating 是一个合理设计，但必须验证 teacher log-probability 本身的稳定性。

Perez-Beltrachini and Lapata 训练轻量模型预测 passage 对目标 QA 模型的 utility，并用它量化回答不确定性（2025，[Uncertainty Quantification in Retrieval Augmented QA](https://arxiv.org/abs/2502.18108)）。这说明 target-model-specific utility distillation 已形成独立研究方向。v0.6.0 应将自己的 predictor 称为 fact-checking-specific conditional utility model，而不能泛称首个 passage utility learner。

### 4.3 Contextual passage utility

Jain and Garimella 明确提出 passage utility 依赖 preceding passages，记为：

\[
U(p_i\mid P_{<i},q).
\]

他们用 reasoning model 生成带 passage citations 的 trace，再让 GPT-4o 基于完整 trace 给每条 passage 1--5 utility rating，随后训练轻量 RoBERTa scorer用于 multi-hop QA reranking（2025，[Modeling Contextual Passage Utility](https://aclanthology.org/2025.ijcnlp-short.37/)）。这是“prefix-conditioned utility”最直接的概念近邻。

两者的关键区别应准确表述：

- 该工作用 reasoning trace + LLM ordinal judgment 生成 utility；VGMES 用实际 verifier prediction intervention 生成连续 signed utility；
- 该工作论文层面强调前文依赖，但其主 RoBERTa predictor直接输入 question--current passage，prefix 信息主要进入标签生成；VGMES 计划将 Evidence Map prefix state显式输入 predictor；
- 该工作聚焦 multi-hop QA reranking；VGMES 聚焦 support/refute/qualify/insufficient/conflict 共存的事实核查，并联合 adaptive capacity。

这篇论文必须进入 Related Work 与实验 baseline 讨论，否则审稿人很容易认为 v0.6.0 忽略了最直接的 concurrent/prior work。

### 4.4 Information-gain supervision 与 stateful selection

三项近期工作使“下游正确输出的增益监督”不能再被当作 VGMES 的数学新意。

1. Iterative ICL 将已选 demonstrations 编码为状态，用 PPO 学习逐步检索策略；其增量 reward 直接使用 $p(y^*\mid x,S\cup\{e\})-p(y^*\mid x,S)$（Chen et al., 2024，[Learning to Retrieve Iteratively for In-Context Learning](https://aclanthology.org/2024.emnlp-main.406/)）。这几乎精确覆盖“set-conditioned downstream marginal + stateful selector”。它选择的是 ICL exemplars，使用固定长度，且没有 fact-checking Evidence Map；但这些是任务化差异，不能改变该公式已有的事实。
2. InfoGain-RAG 定义 singleton Document Information Gain，即加入某文档前后正确答案生成信心的差值，用该信号训练 reranker 并阈值过滤低增益文档，实验还包含 FM2 事实验证任务（Wang et al., 2025，[InfoGain-RAG](https://aclanthology.org/2025.emnlp-main.365/)）。它与 VGMES 的关键差异是主要估计 $\Delta(e\mid\varnothing)$，而非随当前 bundle 改变的 $\Delta(e\mid S)$。
3. FES-RAG 用 fragment 对 gold-answer log-likelihood 的边际贡献生成 Fragment Information Gain，并将高容量多模态模型的 utility judgment 蒸馏到轻量 selector（Wang et al., 2026，[FES-RAG](https://arxiv.org/abs/2604.27600)）。它主要是 singleton fragment purification 与 fixed top-K，但已直接覆盖“teacher likelihood gain → lightweight evidence selector”的训练逻辑。

因此，VGMES 必须把自己表述为：**将已有的 downstream conditional-gain learning 引入具有 typed atom--role state 的多证据事实核查 bundle，并检验该结构是否能学到佐证、冲突、context 和过饱和的条件效应**，而不是宣称提出了新的 information-gain 公式。

### 4.5 Set selection 与 coalition-aware utility

SetR 已从独立 passage ranking 转向由查询信息需求驱动的整体集合选择（Lee et al., 2025，[Shifting from Ranking to Set Selection](https://aclanthology.org/2025.acl-long.861/)）。更直接的 concurrent work RepoShapley 在 repository-level code completion 中使用 teacher-forced probing 估计 signed per-chunk effects，通过轻量 surrogate game 表示 saturation 与 interference，对小检索集计算 Shapley-style contribution，再将 KEEP/DROP 与 retrieval trigger 蒸馏到单模型（Huo et al., 2026，[RepoShapley](https://aclanthology.org/2026.findings-acl.505/)）。

RepoShapley 已同时覆盖 signed marginal、higher-order interaction、saturation/interference、frozen-generator probing、轻量代理与 retrieval trigger，是 v0.6.0 最强的方法级 novelty collision。VGMES 相对它可能保留的只是：

- support/refute/qualify/context 共存的事实核查 decision utility；
- atom × role × direction × provenance 结构对 coalition state 的显式因子化；
- 候选池中“下一条证据”与“当前是否足够”的统一校准；
- 对 held-out verifier 和 human sufficiency 的任务特定验证。

这意味着，如果 v0.6.0 只实现一阶 greedy marginal，而 Evidence Map 不显著优于 text-only state，该方法很容易被评为 InfoGain-RAG 或 Iterative ICL 在 LIAR-RAW 上的结构化应用，并在 interaction modeling 上弱于 RepoShapley。

## 5. 前缀条件检索、证据充分性与动态停止

### 5.1 Autoregressive evidence acquisition

多跳事实核查已有显式 prefix-conditioned retrieval。DOMLIN 的第二阶段依赖先前证据；Natural Logic-guided autoregressive retrieval 联合评分文档与已有证据，并在 proof system 判断证据充分时动态终止，在 FEVER、HoVer 和 FEVEROUS-S 上评价（Aly and Vlachos, 2022，[Natural Logic-guided Autoregressive Retrieval](https://aclanthology.org/2022.emnlp-main.411/)）。

因此，VGMES 不应声称首次在 fact verification 中做 sequential retrieval 或 dynamic termination。差异在于：上述方法的终止来自自然逻辑 proof/sufficiency state，VGMES 的停止来自验证集校准的预测净增益，并在固定候选池中权衡 verdict improvement 与 token/step cost。

### 5.2 Evidence sufficiency

Atanasova et al. 通过删除 constituent/sentence 检验 fact-checking 模型何时将剩余证据视为不充分，并构建 SufficientFacts 诊断数据及 Evidence Sufficiency Prediction 任务（2022，[Fact Checking with Insufficient Evidence](https://aclanthology.org/2022.tacl-1.43/)）。这项工作与 VGMES 的 stopping 合理性直接相关：停止不只是“当前 label confidence 高”，还应意味着没有重要的缺失信息。

VGMES 的 adaptive evaluation 应至少增加：过早停止率、遗漏关键 atom/role 的比例，以及在证据删除/补充后的行为。仅报告平均 K 和 Macro-F1，无法说明停止代表 evidence sufficiency。

### 5.3 LLM-era adaptive retrieval

FLARE 在长文本生成中根据 upcoming sentence 的低置信 token 动态触发检索（Jiang et al., 2023，[FLARE](https://aclanthology.org/2023.emnlp-main.495/)）；Self-RAG 学习 retrieval、generation 与 critique reflection tokens，实现 retrieval on demand（Asai et al., 2024，[Self-RAG](https://openreview.net/forum?id=hSyW5go0v8)）；Adaptive-RAG 根据问题复杂度在 no-retrieval、single-step 和 iterative retrieval 之间路由（Jeong et al., 2024，[Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/)）。这些工作表明 adaptive retrieval 是成熟方向。

事实核查中，FIRE 进一步将 atomic-claim verification 与 iterative search 结合，基于当前判断置信度决定输出最终答案或继续生成搜索 query，并报告显著的 LLM/search 成本降低（Xie et al., 2025，[FIRE](https://aclanthology.org/2025.findings-naacl.158/)）。这是 v0.6.0 adaptive stopping 的最强直接 baseline。

VGMES 与 FIRE 的预期差异是：FIRE 决定“是否继续搜索以及搜索什么”，VGMES 在一个已构造的 provenance-preserving union pool 中估计每个候选相对当前 slate 的收益，因而能显式比较下一条证据、重复/反向/context 角色和 token cost。若 v0.6.0 最终也改成开放式 query-generation agent，则其与 FIRE 的差异会明显减弱。

### 5.4 成本约束与理论边界

“按边际收益逐项加入”在子模优化、value of information 和 active feature acquisition 中有长期理论传统。经典贪心算法可对单调子模目标给出 (1-1/e) 近似保证（Nemhauser et al., 1978，[Submodular Maximization](https://doi.org/10.1007/BF01588971)），adaptive submodularity 则在逐步观测物品状态的设定下支持 adaptive greedy（Golovin and Krause, 2011，[Adaptive Submodularity](https://research.google/pubs/adaptive-submodularity-theory-and-applications-in-active-learning-and-stochastic-optimization/)）。

但 VGMES 不能直接借用这些保证。对

\[
F_\phi(S)=\log p_\phi(y^*\mid c,S),
\]

有害或冲突证据会使 $\Delta_\phi(e\mid S)<0$，多跳/context--conclusion 互补还可能违反 diminishing returns。同时，VGMES 在选择前已知候选文本和 Evidence Map，与选择后才观测物品随机状态的 adaptive-submodular 设定不同。因此建议使用 **instance-adaptive sequential evidence selection** 或 **adaptive context-capacity allocation**，不要在未验证单调性和子模性前称为“adaptive submodular selection”。

## 6. Rationale learning、faithfulness 与干预评价

Rationalizing Neural Predictions 将 rationale generator 与 task predictor 联合训练，在紧凑性和预测充分性约束下选择输入子集（Lei et al., 2016，[Rationalizing Neural Predictions](https://aclanthology.org/D16-1011/)）。ERASER 则系统化评价 rationale 与人类证据的一致性，以及删除/保留 rationale 对模型预测的 comprehensiveness 和 sufficiency（DeYoung et al., 2020，[ERASER](https://aclanthology.org/2020.acl-main.408/)）。从这一视角看，VGMES 的 prefix--candidate intervention 属于 model-relative input attribution/selection，而非对客观证据价值的直接观测。

这一联系带来两个写作边界：

1. \(\Delta_\phi(e\mid S)\) 只能称为 teacher-grounded decision utility；它表明证据如何改变某个 verifier 的 gold-label belief，不自动等于 human utility、truthfulness 或 faithful causal explanation。
2. 输入删除/添加可能产生 off-distribution contexts。后续工作已指出 sufficiency/comprehensiveness 指标可能被分布外输入操纵（Hsia et al., 2024，[Goodhart's Law Applies to NLP Explanation Benchmarks](https://aclanthology.org/2024.findings-eacl.88/)）。因此需要自然 prefix sampling、prompt visibility audit、cross-fitting、重复推理稳定性和人类/跨 verifier 验证。

## 7. 最近邻对照矩阵

| 工作 | 下游 target 反馈 | 集合/前缀条件 | typed atom/role state | signed candidate marginal | 可变容量 | 主要边界 |
|---|---:|---:|---:|---:|---:|---|
| DQN-FV 2021 | 终局验证 + gold evidence | 是 | 否 | 否 | 最大步数 | 精简证据的序列 RL |
| FER 2023 | frozen verifier | 整体集合 | 否 | 否 | 否 | gold-set vs selected-set utility divergence |
| FFRR 2024 | 黑盒 LLM gold-label score | 策略是，candidate reward 否 | 否 | 否 | 主要为 top-K | LIAR-RAW 上的 document/set RL reward |
| Iterative ICL 2024 | frozen LLM target probability | **是** | 通用隐状态 | **是** | 否 | 已有几乎相同的 incremental reward |
| FILCO 2023/24 | canonical-output likelihood | 主要为 singleton | 否 | 静态 CXMI | 过滤后变长 | context filter |
| InfoGain-RAG 2025 | answer confidence | singleton | 否 | $\Delta(e\mid\varnothing)$ | 阈值过滤 | 已有 information-gain reranker，含 FV |
| Contextual Passage Utility 2025 | LLM trace rating | 概念上是 | 否 | 否 | 否 | QA trace + ordinal utility distillation |
| FES-RAG 2026 | teacher gold-target likelihood | singleton | multimodal fragment type | 静态 FIG | fixed top-K | 已有 gain-to-selector distillation |
| User-Centric Evidence Ranking 2026 | gold/logical sufficiency | **是** | 否 | 否 | 用户可早停 | FV 前缀排序、互补与阅读成本 |
| FIRE 2025 | verifier confidence | 是 | atomic claims | 否 | 是 | iterative search/query generation |
| RepoShapley 2026 | frozen generator likelihood | **coalition-aware** | 任务无关 surrogate game | **是，含交互** | retrieval trigger | signed effect、饱和、干扰与蒸馏 |
| **VGMES（计划）** | frozen fact verifier | **是** | **typed Evidence Map** | **是** | **是** | role-aware fact-checking bundle acquisition |

该矩阵只能作为定位工具，不能直接当作 novelty proof。Iterative ICL 覆盖了 set-conditioned marginal reward，InfoGain-RAG/FES-RAG 覆盖了 likelihood gain 及其蒸馏，RepoShapley 进一步覆盖了 coalition interaction 和 retrieval trigger。VGMES 必须通过组合消融证明 typed Evidence Map、fact-checking role dynamics 和 joint next/stop calibration 都有不可替代的实证作用。

## 8. 可直接放入论文的 Related Work 草稿

### Evidence retrieval for fact verification

Evidence-based fact verification commonly separates document retrieval, sentence selection, and verdict prediction. FEVER established this pipeline and evidence-level supervision, while subsequent work improved sentence retrieval through evidence-aware ranking, prefix-conditioned second-stage selection, listwise aggregation, and sequential reinforcement learning. In raw-report settings, CofCED introduced coarse-to-fine report and sentence distillation, RAV propagated verification gradients into evidence ranking, and FFRR used black-box LLM scores as document- and set-level rewards on RAWFC/LIAR-RAW. Recent user-centric evidence ranking further targets early sufficient prefixes and explicitly studies redundancy and complementarity. These methods establish both sequential evidence construction and downstream-feedback retrieval; our question is narrower: whether a typed atom--role state can predict the verifier-relative conditional contribution of a candidate to a partially constructed fact-checking bundle.

### Utility-aligned retrieval and context filtering

Several studies replace relevance with downstream utility. FER trains a fact-verification retriever using the discrepancy between verifier behavior on retrieved and gold evidence; FFRR derives document- and set-level black-box LLM rewards; and RAV propagates verification gradients into evidence selection. Beyond fact verification, RAG, EMDR², Atlas, and REPLUG adapt retrievers using reader or language-model objectives. FILCO and InfoGain-RAG identify useful context from the change in correct-output likelihood, while FES-RAG distills target-likelihood gains into a lightweight fragment selector. Crucially, iterative ICL retrieval already uses the target-probability improvement from adding an item to the current set as a stepwise policy reward. We therefore do not claim a new marginal-gain objective; we instantiate this supervision in multi-evidence fact checking and ask whether explicit atom, stance, role, provenance, and redundancy state improves conditional utility prediction over text-only and singleton alternatives.

### Context-dependent utility and adaptive acquisition

Passage utility is not necessarily intrinsic: evidence may become redundant, corroborative, contradictory, or useful only after other context has been observed. Contextual passage utility, stateful iterative retrieval, autoregressive fact-verification retrieval, and user-centric evidence ranking all condition later decisions on preceding context. Coalition-aware filtering such as RepoShapley goes further by modeling signed effects, saturation, interference, and retrieval triggering. VGMES is thus differentiated neither by contextual utility nor by adaptive retrieval alone. Its proposed task-specific contribution is to factor the current fact-checking slate through typed atom--evidence relations and to test whether this state supports calibrated next-evidence and stopping decisions under matched count/token budgets and held-out verifiers.

### Evidence sufficiency and model-relative faithfulness

Rationale learning studies select compact subsets that preserve task predictions, and ERASER evaluates their sufficiency and comprehensiveness through input interventions. Fact-checking work on insufficient evidence additionally shows that models often fail to recognize omitted information. These findings motivate our intervention-based labels but also delimit their interpretation: the resulting utility is relative to a frozen verifier and may reflect its biases. We therefore separate teacher-relative utility prediction from human evidence sufficiency and evaluate transfer to held-out verifiers, together with evidence-map ablations and cost-matched fixed/adaptive controls.

## 9. v0.6.0 的潜在贡献：强、中、弱三档判断

### 9.1 最强且可辩护的潜在贡献

#### A. 事实核查特有的 typed evidence-bundle utility

下式本身已被 Iterative ICL 等工作覆盖，不再是 novelty：

\[
\Delta_\phi(e\mid S)
=\log P_\phi(y^*\mid c,S\cup\{e\})
-\log P_\phi(y^*\mid c,S).
\]

可辩护的贡献是将该测量变成一个 **atom × role × direction × provenance 条件化的事实核查 bundle 学习问题**：同一条证据可随当前 slate 在 primary support、independent corroboration、opposition、qualification、context 或 redundancy 之间呈现不同净效应。它必须同时超过 singleton InfoGain、Iterative-ICL-style text state 和 FFRR-style rewards 才能成立。

#### B. Typed Evidence Map 作为 conditional utility state，而非人工 utility

Evidence Map 的贡献不应写成“提出 relation/directness/confidence 标注”，而应写成：用 atom × role × direction × provenance 的显式状态预测 verifier marginal response，使旧 proxy 中人工定义的 corroboration/opposite/context 权重转为可学习、可审计的条件效应。要成立必须证明：

- full map 优于 no-map/text-only；
- prefix map state 优于 static candidate map；
- relation/source/context 消融分别损害相应条件 bucket 的 marginal prediction；
- 人工 map alignment 有足够可靠性。

由于 RepoShapley 已经建模一般的 saturation/interference，Evidence Map 贡献还应进一步表现为 **typed interaction efficiency**：利用 map 只在高可能互补或冲突的 atom--role pair 上估计二阶作用，在更低标注/推理成本下接近通用 coalition method。若仍只有一阶贪心且 map 不提高 interaction bucket，本项贡献会明显变弱。

#### C. Candidate-specific acquisition 与 adaptive capacity 的统一

动态检索、阈值过滤和 stop action 都有先例。VGMES 可能保留的是，相对 FIRE 的全局 confidence stop 或 InfoGain-RAG 的独立文档阈值，通过：

\[
\max_{e\notin S}g_\theta(e\mid S)\le\tau
\]

判断“是否还存在值得加入的具体证据”。这将 `what next` 与 `whether to continue` 放在同一个结构化、candidate-specific 净收益尺度上。若能在 matched-F1 下显著减少 K/tokens，或在 matched-cost 下提高 F1，并优于 FIRE-like confidence stop、InfoGain threshold、resolve-stop 和最优验证集固定 K，这可以成为一项任务级方法贡献。

#### D. 跨 teacher 的 utility transfer protocol

由于 utility learning 极易变成 teacher hacking，一套严格的 cross-fitting、held-out verifier、cross-backbone transfer 与人工 sufficiency 评价具有方法论价值。但 Iterative ICL 等工作已报告 cross-LLM transfer，所以“迁移评价”不能单独成为新意。它在本文中的作用是支撑更强主张：若 VGMES 在未参与 utility labeling 的 verifier 上仍稳定提升，才能说明 Evidence Map 状态学到了较一般的证据价值规律，而不只是适配单个 teacher。

### 9.2 中等强度贡献

- 在 LIAR-RAW/RAWFC 的 raw-report 候选池上系统研究 evidence saturation、corroboration、opposition 和 context 的实际 verifier marginal effects；
- 建立可重放 prefix--candidate intervention cache 和对应分析基准；
- 对 fixed-K、adaptive-K、token budget、teacher-specific 与 transfer settings 做统一 paired evaluation；
- 量化负边际率、diminishing-return violation、pair synergy、一阶 greedy 对二阶 complementarity 的 regret，以及 learned stop 相对 oracle stop 的 regret。

这些贡献需要发布数据/代码或至少提供完整 artifact contract 才有分量。

### 9.3 不能单独成立的弱贡献

- claim atomization；
- union retrieval；
- 用 LLM 标 evidence relation；
- 用 verifier feedback；
- 用 log probability difference；
- greedy marginal ranking；
- confidence-based early stopping；
- 固定 K 改成 adaptive K。

这些均已有明确近邻，只能作为完整方法组件。

## 10. 推荐的最终贡献表述

在结果尚未完成前，建议使用条件性版本：

1. **We study state-conditioned evidence-bundle acquisition for fact checking.** Building on downstream marginal-utility supervision, we instantiate paired verifier interventions over diverse evidence prefixes and analyze how corroborating, opposing, contextual, and redundant evidence changes decision utility without requiring gold evidence sets.
2. **We introduce an atom- and role-conditioned utility surrogate.** A typed Evidence Map represents which claim atoms, directions, contextual roles, duplicate groups, and independent sources have already been observed, replacing hand-set proxy weights with learnable and auditable conditional effects.
3. **We use the same structured net-gain estimate for next-evidence selection and instance-level capacity allocation.** The selector acquires the highest-gain feasible candidate and stops when no remaining candidate is predicted to justify its token and attention cost.
4. **We provide leakage-controlled, interaction-aware, and transfer-oriented evaluation.** Cross-fitted labels, singleton and stateful nearest-neighbor controls, small-pool coalition oracles, and held-out-verifier tests distinguish structured evidence utility learning from teacher adaptation.

如果实验只支持其中部分，应相应删减：

- 没有跨 verifier 提升：删除“generalizable evidence utility”，改为 verifier-adaptive selection；
- adaptive 不形成 Pareto 优势：固定 K 作为主方法，停止策略降为附录；
- full map 不优于 text-only：Evidence Map 只能作为解释 sidecar，不能列作方法贡献；
- prefix model 不优于 static delta：不能使用 conditional marginal 作为 headline；
- 一阶 marginal 对 interaction bucket 明显失效：必须启用 typed pair/coalition 扩展，不能把一阶 greedy 定为最终方法；
- 只超过 retrieval top-K 或旧 proxy，而未超过 FFRR、InfoGain-RAG-style singleton 和 Iterative-ICL-style stateful controls：整体只能评价为在 LIAR-RAW 上的增量性应用。

## 11. 必须加入的最近邻 baseline

为了让贡献经得起审稿，建议最低包含：

1. relevance/retrieval top-K；
2. direct/partial manual greedy；
3. learned marginal proxy v0.2；
4. BACES exact v0.3；
5. FER-style static/set-level verifier feedback；
6. FFRR-style single-document + final-set RL reward；
7. FILCO/InfoGain-RAG-style $\Delta(e\mid\varnothing)$ + threshold；
8. Iterative-ICL-style text-state conditional policy，fixed K；
9. contextual/incremental LLM ranking（Contextual Utility 或 User-Centric Ranking 可复现近似）；
10. global confidence/resolve stop（FIRE-like）；
11. VGMES static、VGMES prefix、VGMES prefix+adaptive；
12. VGMES first-order vs typed-pair interaction；
13. text-only vs map-only vs text+map；
14. teacher-oracle marginal greedy，以及小候选池上的 exhaustive/beam/coalition oracle。

最关键的三条证据链是：

1. `singleton delta → text-state conditional delta → Evidence-Map conditional delta`，证明不是 InfoGain-RAG 的结构化复刻；
2. `FFRR/FER feedback → explicit candidate credit assignment`，证明局部边际监督真的改善了顺序构造；
3. `first-order → typed pair/coalition`，并报告与小池 oracle 的 regret，证明方法没有回避互补与冲突。

如果第 1 条不显著，Evidence Map 不能承担核心贡献；如果第 2 条不显著，VGMES 没有必要性；如果第 3 条显示一阶方法 regret 较大，就应将 typed interaction 升为主方法，而不是局限性附注。

## 12. 当前版本的总体贡献判断

目前方法的价值是明确的，但算法新意尚不稳固。若只实现“单 teacher 的一阶 $\Delta(e\mid S)$ + greedy + threshold”，最多是一个较完整的 **fact-checking adaptation/system contribution**：其数学原语分别可由 Iterative ICL、InfoGain-RAG、FES-RAG 和 RepoShapley 覆盖。

若要形成更强的方法论贡献，建议把主线收紧为：

> **Structure-conditioned evidence-bundle acquisition for fact checking:** use a typed Evidence Map to learn and diagnose how atom coverage, evidence role, stance, provenance, redundancy, and selected interactions alter verifier-relative utility, then allocate evidence capacity on the same calibrated scale.

对应的实验升级顺序应是：

1. 先证明 utility landscape 确实存在显著的过饱和、负边际、异源佐证、冲突和 context 效应；
2. 再证明 `singleton → text-prefix → Evidence-Map prefix → typed interaction` 逐层带来可解释改善；
3. 最后将 adaptive capacity 作为效用校准的外在验证，要求它在 matched-cost 或 matched-performance 下形成 Pareto 优势；
4. 用 held-out verifier 与人工 sufficiency 评价排除单 teacher hacking。

如果这条证据链成立，贡献不再是“又一个 utility selector”，而是：**对多角色事实核查证据 bundle 的条件效用进行结构化学习与系统验证**。正式论文中建议避免绝对的“first”，除非投稿前的系统检索能完整支持。
