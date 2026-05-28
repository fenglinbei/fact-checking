# 研究思路分享 PPT 大纲：从原始证据到 Utility-aware Selector

文档更新时间：2026-05-23 14:15 CST

本文用于课程邀请分享场景，目标不是完整复盘所有实验流水账，而是把当前 fact-checking 项目的研究思路讲清楚：如何从原始 evidence label 的低效使用出发，把任务重构为 RAG，再沿着 MMR lambda、oracle evidence set、selector utility 逐步收敛到当前问题边界。听众大致了解 LIAR-RAW 和 fact-checking，因此不需要花太多时间介绍数据集；但需要对 RAG 和 MMR 做短导入。

## 1. 分享定位

### 1.1 推荐标题

主标题：

```text
从 Evidence 到 Selector：事实核查系统中的研究问题重构
```

备选标题：

```text
当 Gold Evidence 不够用：事实核查 RAG 系统的证据选择路线
```

```text
从调参到重定义监督信号：Fact-checking Evidence Selection 的实验路径
```

### 1.2 一句话主线

这场分享应围绕一个问题展开：

```text
事实核查模型判断错了，究竟是 verifier 不会判，还是 evidence 没有选对？
```

最终落点：

```text
当前最值得学习的不是单个分类器、也不是一个 scalar lambda，而是 verifier-aware 的 evidence set utility。
```

### 1.3 受众假设

- 课程听众大致了解 LIAR-RAW、fact-checking 任务和 evidence / verdict 的基本语义。
- 课程听众不一定熟悉 RAG、dense retrieval、BM25、MMR 和 relevance-diversity trade-off。
- 分享前 5-7 分钟需要补一个“面向 fact-checking 的 RAG/MMR 最小概念包”，范围限定在本项目所需概念。
- 需要讲清楚研究决策，而不是堆所有指标。
- 每个阶段都要回答三个问题：为什么做、做了什么、为什么转向。

### 1.4 风格建议

若后续使用 `guizang-ppt-skill` 生成网页 PPT，推荐：

- 风格：瑞士国际主义 / Swiss Style。
- 原因：这份分享更偏工程研究路线、指标对比、实验决策树，适合网格、数据大字报、流程图。
- 视觉锚点：少量大数字 + pipeline 图 + stop/go 决策表。

### 1.5 45 分钟内容组织原则

- 45 分钟版本的额外时间主要用于概念铺垫、关键转向和指标解释，而不是增加更多实验细节。
- RAG/MMR 需要讲直觉，不需要讲完整 IR 背景。
- Oracle set 和 oracle-direct verifier 是全场最关键转折，必须留足时间解释“upper-bound diagnostic”而不是可部署结果。
- Selector Step1-4 不适合逐模型展开；重点是从“学不到”中提炼出“监督信号不对 / utility 表示不足”。

## 2. 研究脉络与材料边界

### 2.1 早期探索定位

早期探索从原始 evidence label 的直接使用开始。系统使用 LIAR-RAW `reports[].tokenized[].is_evidence=1` 句子级 label，从标注句子中取 top5 微调 LLM verifier，但实验效果不理想。无论按 hybrid_score 重新排序，还是保留原始 report / sentence order，test Macro-F1 都低于 fixed-MMR baseline。该阶段说明：人工 evidence relevance label 与当前 verifier 所需的 evidence utility 之间存在差异。

ModernBERT 分类器属于同期的轻量验证实验，发生在学校服务器停机期间。该实验可以作为判别式路径的辅助背景，但不构成后续研究主线的关键因果节点。

### 2.2 主线陈述

PPT 主线应按以下逻辑展开：

```text
原始 gold evidence label 并不能直接转化为强 verifier。
因此任务被重构为 RAG：R 负责从 report 中检索证据，G/verifier 负责根据证据判断真假。
在这个系统里，MMR baseline 已经有竞争力，于是研究自然转向：如何选择更好的 evidence set。
```

后续转向 evidence selection 的依据来自四类证据：

- 原始 evidence label top5 微调效果不理想。
- MMR-RAG baseline 已形成有竞争力的固定比较对象。
- learned-lambda / DPO-lambda 路线未能产生可部署收益。
- oracle evidence set 上界显著大于 oracle lambda 上界。

其中最核心的方法动机是检索目标错位：普通 RAG 检索主要依据 evidence 与 claim 的相似度，而 fact-checking 需要判断该 evidence 是否能提供有效证明、反驳或限定信息。相似度检索只能完成候选初筛，后续必须建模 claim-evidence 之间的 evidence utility 交互。

### 2.3 可量化材料状态

早期 `is_evidence top5 -> LLM verifier` 阶段适合作为研究背景写入 PPT。它的作用不是证明某个具体模型失败，而是给出负向对照：gold relevance label 不能直接替代 verifier utility。

| 方法 | 来源 / 设置 | LIAR-RAW Acc | LIAR-RAW Macro-P | LIAR-RAW Macro-R | LIAR-RAW Macro-F1 | PPT 用法 |
|---|---|---:|---:|---:|---:|---|
| L-Defense | 文献对照 | - | 30.55 | 32.20 | 30.53 | 外部参照 |
| G-Defense | 文献对照 | - | 33.09 | 31.55 | 31.55 | 外部参照 |
| fixed-MMR baseline | `lambda=0.7, top_k=5` | 27.02 | 28.13 | 28.44 | 27.69 | 本项目固定对照 |
| raw `is_evidence` + hybrid | 标注 evidence 句子按 hybrid_score 排序 | 25.74 | 27.44 | 24.49 | 23.51 | 负向对照 |
| raw `is_evidence` + original order | 标注 evidence 句子保留原始顺序 | 25.42 | 29.95 | 24.72 | 22.71 | 负向对照 |

表中 L-Defense / G-Defense 数值来自周报中的 LIAR-RAW P/R/F1；本项目三行使用 test inference 的 accuracy 与 macro P/R/F1。raw `is_evidence` 两种排序均低于 fixed-MMR，说明“标出来的 evidence”并不自动等于“对当前 verifier 有用的 evidence set”。

## 3. 总体叙事弧

```text
Hook:       Gold evidence 不等于可用 evidence
Context:    听众已了解 fact-checking，快速补 RAG：先找证据，再让 verifier 判断
Bridge:     快速补 MMR：不只取最相关 top-k，还要控制冗余
Tension:    相似度检索只能初筛，fact-checking 需要有效证据
Baseline:   构建 dense + BM25 + overlap + MMR 的检索器
Question:   MMR lambda 能不能自适应？
Pivot 1:    lambda 有上界，但不可学
Pivot 2:    oracle evidence set 上界远大于 oracle lambda
Diagnosis:  verifier 能吸收好证据，selector 选不到好证据
Current:    从 imitation selector 转向 verifier-aware utility
Takeaway:   研究推进的关键是不断重定义监督信号
```

## 4. 细化 Slide 大纲

分享时长约 45 分钟，建议做 28 页左右：22 页主线 + 6 页展开页。主线页负责讲清研究转向，展开页用于给不熟 RAG/MMR 的听众补直觉、解释关键指标和承接问答。

### 4.1 45 分钟时间分配

| 段落 | 页数 | 时间 | 目标 |
|---|---:|---:|---|
| 开场与原始 evidence label 问题 | 1-4 | 7 分钟 | 让听众接受“gold evidence 不等于 verifier utility” |
| RAG / MMR 最小概念包 | 5-10 | 10 分钟 | 补齐 RAG、Top-k、相似度检索错位、MMR、lambda、hybrid retrieval 的必要背景 |
| Baseline 与 lambda 路线 | 11-15 | 9 分钟 | 解释为什么从 fixed-MMR 走到 learned-lambda，又为什么停止 |
| Oracle set 转向 | 16-19 | 9 分钟 | 用 +3pp vs +18.76pp 和 oracle-direct verifier 锁定 selector gap |
| Selector 结果与下一步 | 20-22 | 7 分钟 | 说明 Step1-4 的有效增量和当前 VIG / utility 方向 |
| 收束与问答缓冲 | 23-28 或备份页 | 3 分钟 + Q&A | 根据现场反馈展开 RAG/MMR 或 selector metrics |

### 4.2 22 页主线

| 页 | 标题 | 这一页要回答的问题 | 主要内容 | 图示建议 | 证据来源 |
|---|---|---|---|---|---|
| 1 | 从 Evidence 到 Selector | 这场分享讲什么？ | 事实核查不是单纯分类，而是 evidence selection + verifier 的耦合问题。 | 标题页，claim/evidence/verdict 三元素。 | 本文叙事定位 |
| 2 | 一个看似简单的问题 | 模型错了，到底错在哪里？ | 提出核心问题：verifier 不会判，还是 evidence 没选对？ | 二分问题图：Verifier vs Evidence。 | 本文主线 |
| 3 | 起点：Gold Evidence 并不自动好用 | 为什么没有直接沿用数据集证据标签？ | 使用原始 `is_evidence` 句子级 label，取 top5 微调 LLM verifier。两种排序方式 test Macro-F1 均低于 fixed-MMR baseline，说明 gold relevance label 不能直接替代 verifier utility。强调这是研究动机，不是最终指标页。 | “Gold label -> top5 -> LLM verifier -> weak result”的流程图 + 五行指标表。 | 实验进度文档，周报 |
| 4 | 第一轮反思：Evidence Relevance 不等于 Verifier Utility | 为什么人工证据标签未必够？ | `is_evidence` 更像 relevance / support 标注，但 verifier 需要的是能让 label decision 更稳的 evidence set。 | 两列对比：relevance label vs utility signal。 | `docs/analysis/202605200216_selector_experiment_plan_and_literature_review.md` |
| 5 | RAG 在这里是什么意思 | 为什么要补 RAG？ | 不是泛泛介绍 RAG，而是说明：先从 report 中取证据，再让 verifier 只基于这些证据判断。R 是 evidence retrieval，G/verifier 是 label decision。 | `claim + report -> retrieval -> evidence -> verifier -> label`。 | `docs/analysis/202605201437_experiment_progress_timeline.md` |
| 6 | 相似度检索与有效证据的错位 | fact-checking 的检索目标有什么特殊性？ | 普通 RAG 的检索口径主要是 evidence 与 claim 的相似度；fact-checking 的目标是 evidence 是否能提供有效证明、反驳或限定信息。因此 RAG 检索只能做初筛，后续需要特殊方法建模 claim-evidence utility 交互。 | 两列对比：similarity retrieval vs evidence utility。 | 本文研究动机，`docs/analysis/202605161449_oracle_set_gap_analysis.md` |
| 7 | MMR 一分钟介绍 | MMR 解决什么？ | MMR = relevance - redundancy。每次选下一条 evidence 时，既看它和 claim 的相关性，也惩罚它和已选证据的相似性。 | 公式简化版 + 已选/候选示意。 | `docs/implementation/202605111255_mmr-lambda-sweep-pipeline.md` |
| 8 | lambda 是 MMR 的旋钮 | lambda 控制什么？ | 高 lambda 偏相关性，低 lambda 偏多样性；lambda=1 类似纯相关 top-k，lambda=0 更偏去冗余。 | lambda slider：diversity <- -> relevance。 | `docs/implementation/202605111255_mmr-lambda-sweep-pipeline.md` |
| 9 | 轻量插曲：ModernBERT 分类器 | ModernBERT 在故事中放哪里？ | 服务器停机时做的轻量判别式验证。可以说明“单独换分类器没有打开局面”，但不把它作为主因证据。 | 小号 side note，不做主转折页。 | `docs/analysis/202605111212_classifier-collapse-analysis.md` |
| 10 | 检索器：Dense 主导，词面信号兜底 | R 具体怎么做？ | BGE dense score 0.70，lexical overlap 0.20，BM25 0.10；句子/chunk 编码，hybrid score 排序。 | 三路分数融合图。 | `docs/implementation/202605111255_mmr-lambda-sweep-pipeline.md` |
| 11 | Baseline 的意义：不是弱起点 | 为什么后续都要和 fixed-MMR 比？ | fixed lambda=0.7 是强 baseline。它的 test Macro-F1 为 27.69，和 L-Defense / G-Defense 的 30.53 / 31.55 保持同一量级，因此可以作为后续 raw evidence 与 selector 实验的固定对照。 | 大字报：fixed-MMR = strong baseline。 | `docs/analysis/202605151453_RL_MMR_direction_summary.md`，周报对照表 |
| 12 | 于是问题变成：lambda 能不能自适应？ | 为什么从 lambda 开始？ | 如果不同 claim 需要不同相关性/多样性权衡，那么 lambda 应该可以变成 claim-adaptive policy。 | claim 简单/复杂两个例子。 | `docs/analysis/202605141045_RL_MMR_research_review.md` |
| 13 | 第一组实验：k sweep / fixed-lambda / oracle-lambda | lambda 方向有没有理论价值？ | fixed lambda 建 baseline；oracle lambda 比 fixed 约 +3pp，说明 adaptive lambda 有上界但不大。 | 小表：fixed vs oracle lambda。 | `docs/learned_lambda/202605141045_verification_experiment.md`，timeline |
| 14 | Learned-lambda 为什么失败 | 为什么不继续堆模型预测 lambda？ | predictor 近似均值预测；R2 约 0.01；72.6% 样本最优与次优 lambda margin < 0.05；oracle lambda 是不稳定 hard label。 | 大数字：R2~0.01，72.6%。 | `docs/learned_lambda/202605141052_analysis.md` |
| 15 | 强化 / DPO 式 lambda 也没有打开 | RL 思路为什么没有直接解决？ | soft-label 退化，DPO step-wise 多轮坍缩到 lambda=0.7。说明问题不只是训练方法，而是 scalar lambda 表达空间有限。 | Stop/Go 表：soft-label stop，DPO stop。 | `docs/analysis/202605201437_experiment_progress_timeline.md` |
| 16 | 关键转向：从 Scalar Lambda 到 Evidence Set | 为什么转向 set？ | lambda 只能沿 MMR 单轴调节；真正目标是选择能让 verifier 做对的 K-subset。 | 左：lambda slider；右：combinatorial set selector。 | `docs/plan/202605161147_oracle_evidence_selection.md` |
| 17 | Oracle Evidence Set 上界 | set selection 有多大空间？ | Oracle set 相比 fixed-MMR：accuracy +18.76pp，macro-F1 +13.00pp，远大于 oracle lambda 约 +3pp。 | 大数字对比：+3pp vs +18.76pp。 | `docs/analysis/202605161449_oracle_set_gap_analysis.md` |
| 18 | Verifier 到底是不是瓶颈？ | 如果给好证据，verifier 能不能学会？ | Oracle sentence direct verifier 在 val oracle evidence 上 accuracy 0.7111 / macro-F1 0.7169。说明好 evidence 可被 verifier 吸收。 | 上界柱状图：fixed-MMR vs oracle direct。 | `docs/analysis/202605192113_oracle_direct_verifier_result_and_next_plan.md` |
| 19 | 因此瓶颈锁定到 Selector | 当前缺口在哪里？ | 同一个 oracle-direct verifier 换成 fixed-MMR / pointwise evidence 后回到 0.26-0.27；selector 没选到接近 oracle distribution 的证据。 | 漏斗图：oracle evidence 高，普通 evidence 低。 | `docs/analysis/202605192141_oracle_direct_val_evidence_checks.md`，timeline |
| 20 | Selector Step1-4：学得到排序，但选不准集合 | 已经试过哪些 selector？ | pairwise/listwise No-Go；sequential pointer 改善 order metrics，但 recall@5 仍约 0.385，full pipeline 提升有限。 | 表：Step1/3/4 指标和 Stop/Go。 | `docs/analysis/202605201437_experiment_progress_timeline.md` |
| 21 | 下一步：Verifier-aware Utility | 为什么不是继续上更复杂 RL？ | 当前不是 exposure bias 优先，而是 evidence utility 表示不足。VIG / oracle-margin distillation / prefix-level contribution 是更合理下一步。 | 公式：delta_margin = margin(prefix+cand)-margin(prefix)。 | `docs/implementation/202605221430_oracle_vig_utility_analysis.md` |
| 22 | 方法论 Takeaways | 听众需要带走什么？ | 先建强 baseline；用 oracle 上界判断空间；用 Stop/Go 管理路线；失败实验要收敛问题定义。 | 四条 takeaways。 | 本文综合 |

### 4.3 45 分钟建议增加的 6 页展开页

这些页不改变主线，只是在 45 分钟场景下让概念和关键转向更稳。建议插入到对应位置，而不是全部放到最后。

| 插入位置 | 标题 | 用途 | 主要内容 | 图示建议 |
|---|---|---|---|---|
| 第 5 页后 | RAG 和传统分类器的差别 | 防止听众把系统理解成 claim-only classifier | claim-only classifier 只看声明；RAG verifier 先看 report evidence，再做判断。强调 evidence selection 是可优化模块。 | 两条 pipeline 对照 |
| 第 6 页后 | Top-k 为什么会失败 | 给 MMR 做动机铺垫 | 相关性最高的 5 条可能近重复；也可能全部支持同一侧，缺少反驳/限定证据。更深层的问题是：相似不等于对 verdict 有效。 | 一个 claim + 5 条重复证据示意 |
| 第 10 页后 | Hybrid Retrieval 的三种信号 | 让 dense/BM25/overlap 不显得黑箱 | dense 管语义相近，BM25 管关键词命中，overlap 管 claim 中实体/数字/词面对应。 | 三路信号三角图 |
| 第 13 页后 | Oracle Lambda 怎么定义 | 避免听众误解 oracle-lambda 是人工标注 | 枚举多个 lambda，分别生成 evidence set，让 verifier 对 gold label 打分，取得分最高的 lambda。 | lambda 网格 -> 多个 evidence set -> verifier score |
| 第 17 页后 | Oracle Set 不是可部署结果 | 防止把 0.71 误解成最终系统指标 | oracle set 用 gold label 条件搜索，是 upper-bound diagnostic；它说明改进空间在哪里，不代表线上可用。 | diagnostic upper bound 标识 |
| 第 20 页后 | Selector 指标怎么读 | 让 Step1-4 的 No-Go 更可理解 | recall@5 / jaccard@5 看集合是否选对；top1_match / NDCG / pairwise order 看顺序；full pipeline 看最终转化。 | 指标分层表 |

45 分钟版推荐最终页数：

```text
22 页主线 + 6 页展开 = 28 页
```

## 5. 每个章节的内容要点

### 5.1 开场：把问题从分类改成证据选择

内容陈述：

```text
研究起点不是直接设计复杂 selector，而是先尝试使用数据集中标好的 evidence 句子。该路径效果不理想，暴露出一个关键问题：数据集里标注为 evidence 的句子，不一定就是对当前 LLM verifier 最有用的 evidence set。

实验对照显示，直接使用 raw `is_evidence` 的两种排序方式都低于 fixed-MMR baseline。结论不是“gold evidence 没价值”，而是“人工 relevance label 与 verifier utility target 不一致”。
```

表达边界：

- Gold evidence 仍然有研究价值。
- 本页强调的是：gold evidence relevance label 与 verifier utility target 不一致。

### 5.2 RAG 导入：只讲本项目需要的最小定义

听众大致了解 fact-checking，因此 RAG 导入从项目闭环开始，而不是从 LIAR-RAW 定义开始：

```text
RAG = 先检索外部证据，再让模型基于证据回答。
在这里，外部知识不是网页百科，而是每条 claim 对应的 fact-checking reports。
```

需要讲清楚三个角色：

- `claim`：要判断真假的陈述。
- `retriever`：从 report 句子中选出候选 evidence。
- `verifier`：只基于 claim + selected evidence 输出 6 类 label。

本页范围限定在“研究对象不是单个分类器，而是检索-验证闭环”。Embedding、BM25、MMR 放在后续页展开。

### 5.3 RAG 化：系统闭环比单点模型更重要

需要强调：

- Fact-checking 的输入不是 claim alone，而是 claim + report evidence。
- 如果 evidence 选择错了，后面的 verifier 再强也只能基于错误上下文判断。
- 普通 RAG 的第一步通常是相似度检索；在 fact-checking 中，相似度只能说明候选 evidence 与 claim 相关，不能说明它能有效改变或支撑 verdict。
- 因此检索阶段提供的是候选池初筛，后续 selector 需要继续建模 claim-evidence utility 交互。
- 所以研究对象从 `classifier(c)` 变成 `verifier(c, S_K)`，关键变量是 `S_K`。

可放公式：

```text
S_K = Selector(c, C_N(c))
y_hat = Verifier(c, S_K)
```

### 5.4 MMR 导入：从 Top-k 到低冗余 Evidence Set

MMR 导入先用 fact-checking 场景解释直觉，再给出简化公式：

```text
如果 top-5 全是同一句话的改写，它们都很相关，但对 verifier 新增信息很少。
MMR 的作用是：在相关性之外，惩罚和已选证据太相似的候选。
```

MMR 解决的是相似度初筛之后的低冗余选择问题，但它仍然没有直接建模“该 evidence 是否能形成有效证明/反驳”。这也是后续从 scalar lambda 转向 evidence set utility 的原因。

再给简化公式：

```text
MMR score = lambda * relevance - (1 - lambda) * redundancy
```

其中：

- `relevance`：候选 evidence 与 claim 的相关性。
- `redundancy`：候选 evidence 与已选 evidence 的相似度。
- `lambda`：相关性和多样性之间的旋钮。

### 5.5 MMR 阶段：为什么 lambda 是自然切入点

内容陈述：

```text
在 RAG 系统里，证据选择需要在相关性和多样性之间做权衡。MMR 提供了一个很自然的控制旋钮：lambda。它不是凭空调参，而是对应 evidence set 的两个需求：相关性和非冗余。
```

该阶段的作用：

- 建 baseline。
- 固定评估闭环。
- 确认 lambda 对 evidence selection 有影响。
- 为后续 learned-lambda 和 oracle-lambda 提供研究问题。

### 5.6 Learned-lambda 阶段：失败原因要讲成“监督信号不稳定”

本节重点是监督信号不稳定，而不是模型能力不足：

```text
预测 lambda 的失败，不是简单因为模型小，也不是特征不够复杂。核心问题是 oracle lambda 本身像一个噪声 hard label：大量样本的 utility curve 很平，最优 lambda 和次优 lambda 差别很小。
```

可展示三条证据：

- R2 vs mean baseline 约 0.01。
- 72.6% 的 claim 最优-次优 margin < 0.05。
- 预测器输出向均值收缩。

### 5.7 Oracle set 阶段：全场关键转折

这是整场 PPT 的核心页。建议用最大视觉权重展示：

```text
Oracle lambda: 约 +3pp
Oracle evidence set: +18.76pp accuracy / +13.00pp macro-F1
```

讲述逻辑：

1. 如果 oracle lambda 上界只有 +3pp，继续沿 scalar lambda 做复杂策略，收益空间有限。
2. 如果 oracle evidence set 上界有 +18.76pp，说明真正的大空间在 set selection。
3. 因此研究问题从“预测 lambda”转为“学习 evidence set utility”。

### 5.8 Verifier 校准：拆开 verifier 和 selector

需要避免听众误解：

```text
oracle set 高，并不自动说明可部署系统高。它只是诊断：如果证据足够好，系统有没有可能做得更好？
```

接着用 oracle sentence direct verifier 说明：

- 直接用 oracle-selected sentence evidence 训练 verifier，val oracle evidence accuracy 0.7111。
- 这说明 verifier 能吸收好证据。
- 但普通 selector evidence 下回到 0.26-0.27，说明可部署缺口仍在 selector。

### 5.9 Selector 阶段：失败要讲出增量

内容陈述：

```text
Selector 实验不是简单失败。Pairwise 和 listwise 说明：单候选相关性和普通 listwise ranking 不够。Sequential pointer 说明：顺序建模确实有帮助，但集合选择仍不够准。
```

重点区分：

- set metrics：recall@5 / jaccard@5。
- order metrics：top1_match / oracle_rank_ndcg@5 / pairwise_order_acc@5。
- full pipeline：最终 accuracy / macro-F1。

### 5.10 结尾：当前研究问题

建议结尾句：

```text
所以当前问题不再是“哪个模型更强”，而是“什么监督信号能让 selector 学到 verifier 真正需要的 evidence utility”。
```

当前下一步：

- VIG utility analysis。
- oracle-margin distillation。
- prefix-level evidence contribution。
- final-set counterfactual。

## 6. 建议保留的关键数字

| 数字 | 含义 | 放在哪页 |
|---|---|---|
| dense / lexical / BM25 = 0.70 / 0.20 / 0.10 | hybrid retrieval 配置 | 第 10 页 |
| lambda = 0.7 | fixed-MMR 强 baseline | 第 8、11-12 页 |
| oracle lambda 约 +3pp | adaptive lambda 有上界但有限 | 第 13 页 |
| R2 约 0.01 | learned-lambda predictor 基本等价均值预测 | 第 14 页 |
| 72.6% margin < 0.05 | oracle lambda hard label 不稳定 | 第 14 页 |
| oracle set +18.76pp accuracy / +13.00pp macro-F1 | evidence set 空间远大于 lambda 空间 | 第 17 页 |
| oracle direct verifier 0.7111 / 0.7169 | verifier 能吸收好 evidence | 第 18 页 |
| fixed-MMR / pointwise oracle-direct eval 约 0.26-0.27 | selector gap 仍存在 | 第 19 页 |
| sequential recall@5 约 0.3852 | Step4 set metrics 未过 gate | 第 20 页 |
| sequential full val 0.3132 / 0.3026 | selector 有增益但远低于 oracle 上界 | 第 20 页 |

## 7. 可视化素材清单

### 7.1 必备图

1. **RAG pipeline 图**
   - claim
   - report sentences
   - hybrid retrieval
   - MMR / selector
   - verifier
   - label

2. **MMR lambda slider**
   - 左侧 diversity
   - 右侧 relevance
   - 标出 fixed lambda=0.7

3. **研究转向对比图**
   - oracle lambda +3pp
   - oracle evidence set +18.76pp

4. **瓶颈拆分图**
   - oracle evidence + oracle-direct verifier 高
   - fixed-MMR / pointwise evidence 低
   - 箭头指向 selector gap

5. **Stop/Go 路线图**
   - original evidence label: weak start
   - ModernBERT: side validation
   - learned-lambda: stop
   - oracle set: go
   - pairwise/listwise: stop
   - sequential: diagnostic value
   - VIG: next

### 7.2 可选图

- learned-lambda prediction collapse histogram。
- selector metrics radar chart。
- evidence relevance label vs utility label 的二维示意图。
- prefix marginal utility 公式页。

## 8. 主线外材料

以下内容适合放入问答或备份页，不进入主线叙事：

- 过细的 Hydra / cache fingerprint / sharding 实现细节。
- 所有 selector 模型结构的超参数。
- VIG partial cache 的未最终核查指标。
- ModernBERT 分类器的长表格。
- 过多 val/test gap debug 细节。

## 9. 备份页建议

若课程问答偏技术，可准备以下 backup slides：

| 备份页 | 内容 |
|---|---|
| A1 | LIAR-RAW 6 类标签定义 |
| A2 | Hybrid score 公式和 min-max normalization |
| A3 | MMR 伪代码 |
| A4 | learned-lambda predictor 结构 |
| A5 | oracle set greedy search 伪代码 |
| A6 | selector evaluation metrics 定义 |
| A7 | Step1/Step3/Step4 详细指标表 |
| A8 | VIG utility analysis 公式与 self-check |

## 10. 后续生成 PPT 时的 TODO

1. 按 45 分钟版本制作时，建议主 deck 为 28 页：22 页主线 + 6 页展开页；额外技术细节放 backup。
2. 决定 PPT 风格：建议 Swiss Style，便于承载指标和路线图。
3. 若生成网页 PPT，优先做第 5、7、10、17、19、21 页的图示资产；45 分钟版还要补 Top-k failure 和 selector metrics 分层图。
