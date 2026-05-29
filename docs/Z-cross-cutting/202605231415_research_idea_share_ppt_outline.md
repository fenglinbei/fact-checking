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

### 2.2 主线陈述

PPT 主线应按以下逻辑展开：

```text
原始 gold evidence label 并不能直接转化为强 verifier。
因此任务被重构为 RAG：R 负责从 report 中检索证据，G/verifier 负责根据证据判断真假。
在这个系统里，MMR baseline 已经有竞争力，并且 lambda 对最终指标影响明显。
但传统 MMR 使用固定 lambda，于是研究问题变成：不同 claim 是否应该使用不同的 relevance-diversity 权衡？
```

后续转向 evidence selection 的依据来自四类证据：

- 原始 evidence label top5 微调效果不理想。
- MMR-RAG baseline 已形成有竞争力的固定比较对象。
- Oracle lambda 证明 claim-adaptive lambda 有理论价值，但 learned-lambda / DPO-lambda 路线未能产生可部署收益。
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
Knob:       lambda 控制 MMR 在 relevance 和 diversity 之间的取舍
Question:   固定 lambda 是否适合所有 claim？
Intuition:  观点密集型 claim 需要高相似证据，观点分散型 claim 需要多样证据
Probe:      oracle lambda 有约 +3pp 上界，说明方向有价值
Failure:    learned-lambda / soft-label / DPO-lambda 都学不到稳定策略
Pivot 1:    lambda 方向收尾，问题不是更强预测器，而是信号太弱、表达太窄
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
| RAG / MMR 最小概念包 | 5-9 | 9 分钟 | 补齐 RAG、相似度检索错位、MMR、lambda，以及 lambda 对指标的影响 |
| Learned-lambda 路线 | 10-15 | 10 分钟 | 解释为什么从 fixed-MMR 走到 claim-adaptive lambda，又为什么停止 |
| Oracle set 转向 | 16-18 | 8 分钟 | 用 +3pp vs +18.76pp 和 oracle-direct verifier 锁定 selector gap |
| Selector 结果与下一步 | 19-22 | 8 分钟 | 说明 Step1-4 的有效增量和当前 VIG / utility 方向 |
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
| 8 | lambda 是 MMR 的旋钮 | lambda 控制什么？ | 高 lambda 偏相关性，低 lambda 偏多样性；lambda=1 类似纯相关 top-k，lambda=0 更偏去冗余。补一张 0.0 / 0.3 / 0.7 / 1.0 的 Acc / Macro-F1 小表，说明这个旋钮会真实改变 verifier 指标。 | lambda slider + 四行指标表。 | `docs/implementation/202605111255_mmr-lambda-sweep-pipeline.md`，20260519 周报 |
| 9 | 旋钮有效，但传统 MMR 是固定旋钮 | fixed lambda 暴露出什么研究问题？ | λ 从 0.0/0.3 到 0.7 指标明显上升，但 fixed-MMR 只能给所有 claim 一个全局 λ。于是问题变成：每个 claim 是否都适合同一个 relevance-diversity 权衡？ | 大字报：one global λ for all claims? | `docs/Z-cross-cutting/202605201437_experiment_progress_timeline.md` |
| 10 | Claim-adaptive lambda 的直觉 | 为什么不同 claim 可能需要不同 λ？ | 观点密集型 claim 更需要稠密、高相似度 evidence set；观点分散型 claim 更需要多来源、多角度 evidence 辅助判断。两种情况分别对应更高 / 更低的 λ。 | 两个 claim 卡片：dense evidence vs diverse evidence。 | `docs/C-mmr-learned-lambda/202605141045_RL_MMR_research_review.md`，本文归纳 |
| 11 | Oracle lambda：先验证方向有没有价值 | adaptive λ 是否值得学？ | 对每个 claim 枚举 21 个 λ，用 verifier 对 gold label 的 logprob 选 oracle λ。Oracle λ 相比 fixed λ=0.7：Accuracy 30.40% -> 33.48%，Macro-F1 30.60% -> 33.84%，48.4% 样本改变 evidence 选择。 | 小表：fixed λ vs oracle λ。 | `docs/learned_lambda/202605141045_verification_experiment.md` |
| 12 | Learned-lambda 实验链 | 做了哪些尝试？ | 从 hard predictor 开始，尝试 chunk embedding、73 维手工特征、classification、high-margin 过滤、3-bin 粗分类、soft-label λ policy、sensitivity-gated MMR、DPO step-wise λ。 | 路线图：oracle λ -> predictor -> 修复实验 -> DPO。 | `docs/learned_lambda/202605141052_analysis.md`，timeline |
| 13 | 为什么 learned-lambda 学不动 | 失败归因是什么？ | predictor 近似均值预测；R2≈0.01；72.6% 样本最优与次优 λ margin <0.05；utility curve 接近均匀；DPO 四轮坍缩到 λ=0.7。核心不是模型还不够大，而是监督信号太弱、scalar λ 表达太窄。 | 大数字：R2≈0.01，72.6%，DPO -> λ=0.7。 | `docs/learned_lambda/202605141052_analysis.md`，timeline |
| 14 | Lambda 方向收尾 | 为什么不继续调 learned lambda？ | Oracle λ 证明方向有价值，但上界只有约 +3pp，且可学习信号不足。低成本 `log(n)` / sensitivity-gated 只能作为弱 baseline；主线停止继续预测单个 scalar λ。 | Stop/Go 表：oracle λ 有价值，learned λ stop。 | `docs/Z-cross-cutting/202605201437_experiment_progress_timeline.md` |
| 15 | 关键转向：从 Scalar Lambda 到 Evidence Set | 为什么转向 set selector？ | lambda 只能沿 MMR 的单轴调节；真正目标是选择能让 verifier 做对的 K-subset。下一步不再问“λ 取多少”，而是问“哪一组 evidence 最有用”。 | 左：lambda slider；右：combinatorial set selector。 | `docs/D-oracle-evidence/202605161147_oracle_evidence_selection.md` |
| 16 | Oracle Evidence Set 上界 | set selection 有多大空间？ | Oracle set 相比 fixed-MMR：accuracy +18.76pp，macro-F1 +13.00pp，远大于 oracle lambda 约 +3pp。 | 大数字对比：+3pp vs +18.76pp。 | `docs/D-oracle-evidence/202605161449_oracle_set_gap_analysis.md` |
| 17 | Verifier 到底是不是瓶颈？ | 如果给好证据，verifier 能不能学会？ | Oracle sentence direct verifier 在 val oracle evidence 上 accuracy 0.7111 / macro-F1 0.7169。说明好 evidence 可被 verifier 吸收。 | 上界柱状图：fixed-MMR vs oracle direct。 | `docs/D-oracle-evidence/202605192113_oracle_direct_verifier_result_and_next_plan.md` |
| 18 | 因此瓶颈锁定到 Selector | 当前缺口在哪里？ | 同一个 oracle-direct verifier 换成 fixed-MMR / pointwise evidence 后回到 0.26-0.27；selector 没选到接近 oracle distribution 的证据。 | 漏斗图：oracle evidence 高，普通 evidence 低。 | `docs/D-oracle-evidence/202605192141_oracle_direct_val_evidence_checks.md`，timeline |
| 19 | Selector Step1-4：学得到排序，但选不准集合 | 已经试过哪些 selector？ | pairwise/listwise No-Go；sequential pointer 改善 order metrics，但 recall@5 仍约 0.385，full pipeline 提升有限。 | 表：Step1/3/4 指标和 Stop/Go。 | `docs/Z-cross-cutting/202605201437_experiment_progress_timeline.md` |
| 20 | 下一步：Verifier-aware Utility | 为什么不是继续上更复杂 RL？ | 当前不是 exposure bias 优先，而是 evidence utility 表示不足。VIG / oracle-margin distillation / prefix-level contribution 是更合理下一步。 | 公式：delta_margin = margin(prefix+cand)-margin(prefix)。 | `docs/F-feature-diagnostics/202605221430_oracle_vig_utility_analysis.md` |
| 21 | 当前路线图：Set-aware Selector | set selector 接下来怎么推进？ | 用 sentence-level oracle candidate pool、order-aware gate 和 verifier-aware utility，把目标从 imitation oracle indices 改成学习 prefix / set-level utility。 | 三阶段路线图：candidate pool -> utility scorer -> ordered top5。 | `docs/E-selectors/202605200216_selector_experiment_plan_and_literature_review.md` |
| 22 | 方法论 Takeaways | 听众需要带走什么？ | 先建强 baseline；用 oracle 上界判断空间；用 Stop/Go 管理路线；失败实验要收敛问题定义。 | 四条 takeaways。 | 本文综合 |

### 4.3 45 分钟建议增加的 6 页展开页

这些页不改变主线，只是在 45 分钟场景下让概念和关键转向更稳。建议插入到对应位置，而不是全部放到最后。

| 插入位置 | 标题 | 用途 | 主要内容 | 图示建议 |
|---|---|---|---|---|
| 第 5 页后 | RAG 和传统分类器的差别 | 防止听众把系统理解成 claim-only classifier | claim-only classifier 只看声明；RAG verifier 先看 report evidence，再做判断。强调 evidence selection 是可优化模块。 | 两条 pipeline 对照 |
| 第 6 页后 | Top-k 为什么会失败 | 给 MMR 做动机铺垫 | 相关性最高的 5 条可能近重复；也可能全部支持同一侧，缺少反驳/限定证据。更深层的问题是：相似不等于对 verdict 有效。 | 一个 claim + 5 条重复证据示意 |
| 第 8 页后 | Fixed lambda sweep 细节 | 让“lambda 会影响指标”有数据支撑 | 展开 λ=0.0~1.0 曲线，强调 λ=0.7 是强 baseline，但不是证明所有 claim 都适合固定 λ。 | 折线图 + λ=0.7 高亮 |
| 第 11 页后 | Oracle Lambda 怎么定义 | 避免听众误解 oracle-lambda 是人工标注 | 枚举多个 lambda，分别生成 evidence set，让 verifier 对 gold label 打分，取得分最高的 lambda。 | lambda 网格 -> 多个 evidence set -> verifier score |
| 第 16 页后 | Oracle Set 不是可部署结果 | 防止把 0.71 误解成最终系统指标 | oracle set 用 gold label 条件搜索，是 upper-bound diagnostic；它说明改进空间在哪里，不代表线上可用。 | diagnostic upper bound 标识 |
| 第 19 页后 | Selector 指标怎么读 | 让 Step1-4 的 No-Go 更可理解 | recall@5 / jaccard@5 看集合是否选对；top1_match / NDCG / pairwise order 看顺序；full pipeline 看最终转化。 | 指标分层表 |

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

第 8 页要先讲清楚这个旋钮本身，再立刻给出 fixed-MMR sweep 的代表点：

| λ | Test Acc | Test Macro-F1 |
|---:|---:|---:|
| 0.0 | 19.98 | 16.39 |
| 0.3 | 21.58 | 17.71 |
| 0.7 | 26.62 | 26.20 |
| 1.0 | 24.70 | 22.79 |

该阶段的作用：

- 建 baseline。
- 固定评估闭环。
- 确认 lambda 对 evidence selection 和 verifier 指标都有影响。
- 暴露出传统 MMR 的限制：它通常给所有 claim 使用同一个全局 lambda。
- 引出后续 learned-lambda 和 oracle-lambda 的研究问题。

### 5.6 Claim-adaptive lambda：要先讲直觉，再讲实验

第 10 页不要直接跳到模型和脚本，而是先讲一个可理解的直觉：

```text
观点密集型 claim：
需要稠密、高相似度、集中在关键事实周围的 evidence set。
更可能偏高 lambda。

观点分散型 claim：
需要来自不同来源、不同角度的 evidence 来限定或反驳 claim。
更可能偏低 lambda。
```

这一步的作用是让 learned-lambda 不像“为了调参而调参”，而是像一个自然研究问题：

```text
如果 claim 的证据需求不同，为什么所有 claim 都要共用同一个 lambda？
```

### 5.7 Learned-lambda 阶段：先证明方向有价值，再解释为什么失败

Oracle lambda 页要先回答“是否值得学”：

| 指标 | Fixed λ=0.70 | Oracle λ | Δ |
|---|---:|---:|---:|
| Accuracy | 0.3040 | 0.3348 | +0.0308 |
| Macro-F1 | 0.3060 | 0.3384 | +0.0324 |

同时补一句：48.4% 的样本在 oracle λ 和 fixed λ 下产生不同 evidence 选择。这说明 lambda 不是纯粹的形式参数，它确实改变了 evidence set。

然后再讲 learned-lambda 系列实验：

- hard predictor：chunk embedding / 73 维手工特征 / classification。
- 修复实验：high-margin 过滤、3-bin 粗粒度分类。
- 替代策略：sensitivity-gated MMR、soft-label λ policy、DPO step-wise λ。

### 5.8 Learned-lambda 失败归因：监督信号太弱

本节重点是监督信号不稳定，而不是模型能力不足：

```text
预测 lambda 的失败，不是简单因为模型小，也不是特征不够复杂。核心问题是 oracle lambda 本身像一个噪声 hard label：大量样本的 utility curve 很平，最优 lambda 和次优 lambda 差别很小。
```

可展示五条证据：

- R2 vs mean baseline 约 0.01。
- 72.6% 的 claim 最优-次优 margin < 0.05。
- 预测器输出向均值收缩。
- soft-label 的 utility curve 接近均匀，expected λ 退化为 fixed。
- DPO step-wise 四轮训练都坍缩到 λ=0.7。

收尾时要明确：

```text
Oracle λ 证明 adaptive λ 有约 +3pp 上界；
但 learned λ 证明这个上界很难从当前可见特征中稳定学出来。
```

### 5.9 Lambda 方向收尾：转向 Evidence Set

这是整场 PPT 的核心页。建议用最大视觉权重展示：

```text
Oracle lambda: 约 +3pp
Oracle evidence set: +18.76pp accuracy / +13.00pp macro-F1
```

讲述逻辑：

1. Oracle lambda 已经证明 claim-adaptive 权衡有价值，但上界只有约 +3pp。
2. learned-lambda / soft-label / DPO 都说明 scalar λ 的可学习信号太弱。
3. Oracle evidence set 上界有 +18.76pp accuracy / +13.00pp macro-F1，说明真正的大空间在 set selection。
4. 因此研究问题从“预测 lambda”转为“学习 evidence set utility”。

### 5.10 Verifier 校准：拆开 verifier 和 selector

需要避免听众误解：

```text
oracle set 高，并不自动说明可部署系统高。它只是诊断：如果证据足够好，系统有没有可能做得更好？
```

接着用 oracle sentence direct verifier 说明：

- 直接用 oracle-selected sentence evidence 训练 verifier，val oracle evidence accuracy 0.7111。
- 这说明 verifier 能吸收好证据。
- 但普通 selector evidence 下回到 0.26-0.27，说明可部署缺口仍在 selector。

### 5.11 Selector 阶段：失败要讲出增量

内容陈述：

```text
Selector 实验不是简单失败。Pairwise 和 listwise 说明：单候选相关性和普通 listwise ranking 不够。Sequential pointer 说明：顺序建模确实有帮助，但集合选择仍不够准。
```

重点区分：

- set metrics：recall@5 / jaccard@5。
- order metrics：top1_match / oracle_rank_ndcg@5 / pairwise_order_acc@5。
- full pipeline：最终 accuracy / macro-F1。

### 5.12 结尾：当前研究问题

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
| λ=0.0 / 0.3 / 0.7 / 1.0 的 Macro-F1 = 16.39 / 17.71 / 26.20 / 22.79 | lambda 改变会明显影响 verifier 指标 | 第 8-9 页 |
| lambda = 0.7 | fixed-MMR 强 baseline，但仍是全局固定值 | 第 8-9 页 |
| oracle lambda +3.08pp accuracy / +3.24pp Macro-F1 | adaptive lambda 有理论上界但有限 | 第 11 页 |
| 48.4% 样本 evidence 选择改变 | λ 变化确实改变 evidence set | 第 11 页 |
| R2 约 0.01 | learned-lambda predictor 基本等价均值预测 | 第 13 页 |
| 72.6% margin < 0.05 | oracle lambda hard label 不稳定 | 第 13 页 |
| DPO step-wise 四轮坍缩到 λ=0.7 | scalar λ policy 没有形成可用策略 | 第 13-14 页 |
| oracle set +18.76pp accuracy / +13.00pp macro-F1 | evidence set 空间远大于 lambda 空间 | 第 16 页 |
| oracle direct verifier 0.7111 / 0.7169 | verifier 能吸收好 evidence | 第 17 页 |
| fixed-MMR / pointwise oracle-direct eval 约 0.26-0.27 | selector gap 仍存在 | 第 18 页 |
| sequential recall@5 约 0.3852 | Step4 set metrics 未过 gate | 第 19 页 |
| sequential full val 0.3132 / 0.3026 | selector 有增益但远低于 oracle 上界 | 第 19 页 |

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
   - 下方放 λ sweep 代表点小表

3. **Claim-adaptive lambda 直觉图**
   - 观点密集型 claim -> 高相似 evidence set -> higher λ
   - 观点分散型 claim -> 多角度 evidence set -> lower λ

4. **研究转向对比图**
   - oracle lambda +3pp
   - oracle evidence set +18.76pp

5. **瓶颈拆分图**
   - oracle evidence + oracle-direct verifier 高
   - fixed-MMR / pointwise evidence 低
   - 箭头指向 selector gap

6. **Stop/Go 路线图**
   - original evidence label: weak start
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
- hybrid retrieval 的 dense / BM25 / overlap 细节长表。
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
3. 若生成网页 PPT，优先做第 5、8、10、16、18、21 页的图示资产；45 分钟版还要补 Top-k failure、Oracle Lambda 定义和 selector metrics 分层图。
