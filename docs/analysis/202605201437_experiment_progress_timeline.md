# 实验总进度时间线与文档索引

文档更新时间：2026-05-22 23:59 CST

本文整理 `docs/` 目录下的计划、分析、实现与运维文档，按实验线程归纳总体进度。本文只汇总已有文档与已记录结果，不新增实验结论。

命名约定：从 2026-05-19 起，新增或实质更新后重新落盘的分析文档，文件名前缀使用文档更新时间 `YYYYMMDDHHMM`，而不是实验启动时间或首次创建时间。

## 1. 线程导航

| 线程 | 时间范围 | 核心问题 | 状态 | 一句话结论 |
|---|---|---|---|---|
| [A. 分块与基础设施](#4a-基础设施与分块05-08--05-13) | 05-08 ~ 05-13 | chunking 策略选择与 pipeline 固化 | 已固定 | sentence chunking 为主线，semantic 为对照；训练/推理范式已统一 |
| [B. 分类器塌陷](#4b-分类器塌陷与证据质量瓶颈05-11) | 05-11 | 判别式分类器为何失败 | 已停止 | 证据质量而非损失函数是主要瓶颈 |
| [C. MMR λ 与 Learned-λ](#4c-mmr-λ-扫描与-learned-λ05-11--05-16) | 05-11 ~ 05-16 | adaptive λ 能否提升 verifier 准确率 | 已停止 | oracle λ 有 +3pp 上界，但无法从文本特征预测；scalar λ 路线全部失败 |
| [D. Oracle Evidence Set 与 Verifier 校准](#4d-oracle-evidence-set-与-verifier-校准05-16--05-19) | 05-16 ~ 05-19 | 最优证据集上界多大，verifier 能否吸收 | 进行中 | oracle set gap +18.76pp；direct verifier 在 oracle evidence 上 accuracy 0.7111 |
| [E. Selector 实验 Step1-4](#4e-selector-实验-step1-405-19--05-22) | 05-19 ~ 05-22 | 能否训练模型选出 oracle 级证据 | 继续诊断 | Step1/3 No-Go；Step4 sequential pointer 改善 order，但 full pipeline 转化有限；VIG 仅部分运行 |
| [F. Targeted Feature 前置诊断](#4f-targeted-feature-前置诊断05-21--05-22) | 05-21 ~ 05-22 | stance/aspect 特征能否辅助 selector | 已停止 | NLI / rule_aspect / LLM decomp+ 三者均 No-Go；claim-aspect coverage 主线关闭 |

## 2. 全局时间线概览

| 日期 | 关键事件 | 线程 | 决策/结论 |
|---|---|---|---|
| 05-08 | 固定 5 种 chunking 策略 | A | sentence 为默认，semantic 进入后续对比 |
| 05-11 | b4 ModernBERT 分类器塌陷 | B | 瓶颈转向 evidence selection |
| 05-11 | MMR λ sweep pipeline 固化 | C | 11 个 λ 值的统一 build→train→infer 流水线 |
| 05-12~13 | val/test gap 诊断、chunking 可视化 | A, C | LoRA vLLM 动态加载导致 parse error；代码路径差异贡献小 |
| 05-14 | oracle λ 验证 +3pp；hard predictor 全失败 | C | adaptive λ 有价值但不可从特征预测；启动 sensitivity-gated / soft-label / DPO |
| 05-15~16 | DPO step-wise 4 轮全坍缩到 λ=0.7；scalar λ 路线关闭 | C | fixed λ=0.7 是强 baseline；转向 oracle evidence set |
| 05-16 | Oracle evidence set gap 计算：+18.76pp accuracy | D | 比 oracle λ gap (~3pp) 大 6 倍；启动 oracle-set supervision |
| 05-17 | Pointwise selector V1 完成；下游 verifier test 0.2230 < fixed 0.2702 | D | 旧 selection-only gate 无效（cache 口径不一致）；启动 verifier 校准 |
| 05-18 | Stage1 label-token weighted CE verifier 完成；V1b true-side anchor 微升 | D | weighted CE 阻止 false bias；但 selector 仍未超 fixed-MMR |
| 05-19 | Stage2 margin re-oracle 完成；粒度决策：sentence > semantic | D | sentence-level oracle supervised 进入主线 |
| 05-19 | Oracle sentence direct verifier 强阳性：val accuracy 0.7111 | D | verifier 可吸收 oracle evidence；瓶颈确认在 selector |
| 05-20 | Step1 cross-encoder pairwise 三种模型均 No-Go | E | recall@5 最高 0.3739，远低于 0.50 gate |
| 05-20 | Step3 listwise selector No-Go | E | 最好 recall@5=0.3826；rank prior 非唯一问题 |
| 05-20~21 | Step4 sequential pointer：order 改善但 set 未过 | E | top1_match 0.1664, recall@5=0.3852；后续 full pipeline probe 转化有限 |
| 05-22 | Selector trace full pipeline：hybrid vs sequential | E | sequential val 0.3132/0.3026 > hybrid 0.2834/0.2878；recall@5 +0.0418 未充分转成 macro-F1 |
| 05-22 | Step4.1-D VIG utility analysis 部分运行 | E | 诊断链路已实现；当前产物不是最终版本，不据此做 Stop/Go |
| 05-21~22 | stance NLI / rule aspect / LLM decomp+ 前置诊断 | F | NLI / rule_aspect 均 stop；LLM decomp+ plain 重跑生成质量合格但 coverage No-Go，claim-aspect 主线关闭 |

## 3. 系统基础设施

在展开各线程前，先梳理贯穿所有实验的共享基础设施与符号约定。

### 3.1 数据流

```
原始 LIAR-RAW JSON
  → SentenceRecord / SampleRecord（分句、解析 gold label）
  → Build: Chunking → Candidate Pool → Hybrid Scoring → MMR Selection → Prompt Construction
  → JSONL（含 prompt + target + gold label + evidence list）
  → Train: accelerate + DeepSpeed ZeRO-2, LoRA SFT on Qwen2.5-7B-Instruct
  → best checkpoint（LoRA adapter）
  → Infer: vLLM OpenAI-compatible API, temperature=0.0, max_tokens=8
  → Label Parsing → Accuracy, Macro-F1, Confusion Matrix
```

### 3.2 标签体系

所有实验均使用 LIAR-RAW 6 类标签（`src/fact_checking/data/constants.py`），映射为单字母 token：

| ID | 标签名 | 字母 | 定义 |
|----|--------|------|------|
| 0 | pants-fire | A | completely false and implausible |
| 1 | false | B | false based on the available evidence |
| 2 | barely-true | C | mostly false, with only a small element of truth |
| 3 | half-true | D | partly true and partly false |
| 4 | mostly-true | E | mostly true, with minor missing context or caveats |
| 5 | true | F | accurate based on the available evidence |

3 分类折叠（仅用于 b4 分类器诊断实验）：`{A,B}→false, {C,D}→mixed, {E,F}→true`。

### 3.3 核心检索 Pipeline（Hybrid Scoring + MMR）

**Hybrid Score 计算**（三路融合，权重 α_dense=0.70, α_lexical=0.20, α_bm25=0.10）：
1. **Dense**：BGE-base-en-v1.5 将 claim 和 chunk 编码为 768 维向量，内积相似度（等价余弦相似度）
2. **Lexical**：基于内容词的 F1 分数（去除停用词后 token overlap 的 precision/recall/F1）
3. **BM25**：简化版 BM25（k1=1.2, b=0.75），idf × tf 加权

三路分数各自 Min-Max 归一化到 [0,1] 后加权求和。

**MMR 选择算法**（`src/fact_checking/retrieval/mmr.py`）：
- λ=1.0 等价于按 hybrid_score 降序取 top-K；λ=0.0 纯粹最大化与已选集合的差异
- 首轮选 hybrid_score 最高的 chunk；迭代选 mmr_score 最大的未选 chunk

### 3.4 训练与推理基础设施

- **基座模型**：Qwen2.5-7B-Instruct
- **微调方式**：LoRA（r=16, α=32, dropout=0.05），目标模块所有线性投影
- **训练框架**：accelerate + DeepSpeed ZeRO-2，4 GPU，bf16，FlashAttention-2
- **有效 batch size**：32（per_device=4 × grad_accum=2 × 4 GPU）
- **学习率**：1e-5，cosine scheduler，warmup_ratio=0.03
- **推理引擎**：vLLM 0.8.5.post1，tensor_parallel_size=4，gpu_memory_utilization=0.90
- **Prompt 模板**：Qwen2.5 ChatML 格式。system prompt 固定，user content 包含 Labels 定义 + Claim + Evidence
- **解码策略**：temperature=0，max_tokens=8，guided choice 限定 A-F

### 3.5 符号约定

| 符号 | 含义 |
|------|------|
| $c$ | 政治声明（claim） |
| $y^*$ | 真实 6 类 veracity 标签 |
| $C_N(c)$ | claim $c$ 的候选证据池，含 $N$ 个候选 chunk |
| $K$ | 需选择的证据数量（即 `top_k`） |
| $S_t$ | 经 $t$ 步选择后的已选证据集合 |
| $\lambda$ | MMR 的多样性惩罚系数，λ∈[0,1] |
| $\operatorname{Rel}(c,d)$ | claim $c$ 与候选 $d$ 的相关性分数（hybrid score） |
| $\operatorname{Red}(d,S)$ | 候选 $d$ 与已选集合 $S$ 的冗余度 |

---

## 4. 分线程详细展开

### 4.A 基础设施与分块（05-08 ~ 05-13）

**动机**：需要统一 evidence retrieval 的 chunking 策略，作为所有后续实验的基础。

[`../plan/202605080025_chunking.md`](../plan/202605080025_chunking.md) 定义了五类策略（均继承 `ChunkingStrategy` 抽象基类，实现于 `src/fact_checking/build/chunking.py`）：

| 策略 | 行为 |
|------|------|
| `sentence` | 返回每个非空句子作为独立 chunk |
| `raw` | 不切分，返回整个 report 全文 |
| `ctx_window` | 以 sent_idx 为中心取上下各 k 句窗口 |
| `semantic` | 按相邻句子对的 BGE 余弦相似度合并（θ=0.7） |
| `ctx_semantic` | 先按窗口 k 分组，再按窗口平均向量做 semantic 合并 |

配置入口 `configs/build/default.yaml`：默认 `strategy=sentence`。semantic chunking（θ=0.5）下 val 候选池 $N$ 中位数约 16，远小于 sentence chunking 的 ~51，显著减少了搜索空间但也可能丢失细粒度信息。

[`../visualization/202605121618_chunking_evidence_examples.md`](../visualization/202605121618_chunking_evidence_examples.md) 用真实样本展示了不同策略的证据跨度差异。

[`../plan/202605131653_dvc-warm-minsky.md`](../plan/202605131653_dvc-warm-minsky.md) 规划了 DVC 数据版本控制，保证跨机器复现。

[`../implementation/202605171700_tailscale_container_ssh_vscode.md`](../implementation/202605171700_tailscale_container_ssh_vscode.md) 记录了 Tailscale 远程连接方案。

**当前状态**：sentence chunking 为主线，semantic 保留为对照。chunk-level candidate pool 是所有后续 MMR、learned-λ、oracle search 的共同基础。

---

### 4.B 分类器塌陷与证据质量瓶颈（05-11）

**动机**：尝试用判别式分类器（ModernBERT-large）替代生成式 SFT，看能否以更小模型达到可比性能。

[`202605111212_classifier-collapse-analysis.md`](202605111212_classifier-collapse-analysis.md) 记录了 b4 实验的系统性塌陷：

| 实验 | 损失函数 | 类别数 | Val Macro-F1 | 塌陷表现 |
|---|---|---|---|---|
| 原始 b4 (CE) | CrossEntropyLoss | 6 | 0.152 | 61.9% 预测→half-true |
| b4 + CORAL | CORAL ordinal | 6 | 0.117 | 64.8%→half-true；仅 3/6 类被预测 |
| b4 + 3class | CrossEntropyLoss | 3 | 0.391 | false 召回 17.3%，mixed 召回 57.5% |
| b4 + no_evidence | CrossEntropyLoss | 6 | — | 去掉 evidence，仅用 claim |

随机基线对照：6 类 random Macro-F1≈0.167；3 类 random Macro-F1≈0.33。原始 b4 的 CE 结果仅略高于随机。

**决策**：停止 ModernBERT 判别式分类器作为主线。根因是证据质量而非损失函数——MMR 检索返回的证据判别力不足，多条证据可能相互矛盾，2048 tokens 拼接 16 条证据使模型难以聚焦。**主矛盾从分类头/损失函数推向 evidence selection。**

---

### 4.C MMR λ 扫描与 Learned-λ（05-11 ~ 05-16）

**动机**：固定 λ=0.7 是否最优？能否为每条 claim 自适应选择 λ？

#### 4.C.1 固定 λ sweep（05-11）

[`../implementation/202605111255_mmr-lambda-sweep-pipeline.md`](../implementation/202605111255_mmr-lambda-sweep-pipeline.md) 建立了统一的 λ sweep 流水线：两阶段 GPU 缓存（Pre-MMR → Chunk-MMR），λ 扫描范围 0.0~1.0（11 个值），prompt 自动截断。

[`202605112128_val_test_Mismatch.md`](202605112128_val_test_Mismatch.md) 诊断了 `mmr_lambda_sweep_1024`（max_length=1024, top_k=16）的问题：
- val 在 epoch 1 末达峰后回落~5pp（经典过拟合）
- val/test 严重不一致：best val F1≈0.27, API test F1≈0.15, parse_error_rate≈22.6%
- 根因：LoRA 在 vLLM 动态加载模式下 `modules_to_save` 丢失

[`202605131520_val-test-f1-gap-diagnosis.md`](202605131520_val-test-f1-gap-diagnosis.md) 修正结论：代码路径差异贡献很小，gap 主要是真实泛化差距。修复方案：推理前 `merge_and_unload()` 出临时 dense 模型。

#### 4.C.2 Oracle λ 验证（05-14）

[`../learned_lambda/202605141045_verification_experiment.md`](../learned_lambda/202605141045_verification_experiment.md) 先回答"oracle λ 是否有价值"：在 21 个 λ 值的网格上，oracle λ 相比 fixed λ=0.7 将 accuracy 从 30.40% 提到 33.48%（**+3.08 pp**），macro-F1 从 30.60% 提到 33.84%（**+3.24 pp**）。48.4% 的样本产生了不同的 evidence 选择。

**结论**：adaptive λ 不是无意义方向，但收益有限（+3pp vs oracle set 的 +19pp）。

#### 4.C.3 Hard predictor 全失败（05-14）

[`../learned_lambda/202605141052_analysis.md`](../learned_lambda/202605141052_analysis.md) 给出反向结论：hard oracle λ 作为监督目标非常弱。

| 变体 | Val MAE | Val RMSE | Target Std |
|---|---|---|---|
| Chunk embedding (256-dim attention) | 0.256 | 0.294 | 0.296 |
| Handcrafted 73 features (regression) | 0.250 | 0.283 | 0.299 |
| Handcrafted 73 features (classification) | 0.250 | 0.282 | 0.299 |

对比均值预测 λ=0.445 的 MAE=0.262。$R^2$ vs 均值基线 ≈ 0.01。

**五重根因**：(1) oracle λ margin 极小（72.6% 样本 <0.05，中位数 0.0185）；(2) tie-break 向 0.7 收缩；(3) oracle λ 与 claim 语义无关（6 类均值均在 0.42-0.50）；(4) BGE embedding 不编码"什么 λ 对 MMR 最优"的信息；(5) 唯一可观察的系统模式是 corr(log(n_candidates), λ_oracle) = -0.13。

两个修复实验也全失败：
- **High-margin 过滤**（margin≥0.05）：预测值坍缩至 0.30-0.32，$R^2$ = -0.167
- **3-bin 粗粒度分类**：准确率 0.336（随机 0.333），全预测为同一类

#### 4.C.4 替代 λ 策略全失败（05-14 ~ 05-16）

[`../implementation/202605141531_sensitivity-gated-mmr.md`](../implementation/202605141531_sensitivity-gated-mmr.md) 实现 sensitivity-gated MMR：用 Jaccard 差异和池冗余度做二值门控（λ_low vs λ_base）。保留为弱 adaptive baseline（test +0.0040）。

[`../plan/202605141828_soft_label_policy.md`](../plan/202605141828_soft_label_policy.md) 设计 soft-label λ policy：用 utility curve 的 softmax 作为软目标。结果 utility curve 接近均匀，expected λ 退化到 fixed。

[`../plan/202605151936_dpo_step_wise_lambda.md`](../plan/202605151936_dpo_step_wise_lambda.md) 和 [`../plan/202605152032_dpo_stepwise_lambda.md`](../plan/202605152032_dpo_stepwise_lambda.md) 实现 DPO step-wise λ：

| 版本 | 特征维度 | β | 结果 |
|---|---|---|---|
| V1 | 20-dim (pool+step) | 1.0 | λ=0.7: 99.97% |
| V2 | 20-dim | 3.0 | λ=0.7: 99.87% |
| V3 | 13-dim (step+prev_lambda) | 3.0 | λ=0.7: 99.87% |
| V4 | 13-dim, K=1 (claim-level) | 3.0 | λ=0.7: 100% |

四轮训练全部坍缩到 λ=0.7。根因：reference policy 已与多数 winner 一致，有效 DPO 梯度太少；utility gap 中位数仅 2.34。

#### 4.C.5 Scalar λ 路线总决算（05-16）

[`../plan/202605161049_RL_MMR_experiment_plan_v2.md`](../plan/202605161049_RL_MMR_experiment_plan_v2.md) 汇总：

| 方法 | test accuracy | test macro-F1 | Δ vs fixed | 状态 |
|---|---|---|---|---|
| fixed λ=0.7 (b3, top_k=5) | 0.2702 | 0.2769 | — | locked baseline |
| log(n_candidates) heuristic | 0.2766 | 0.2799 | +0.0064 | 保留弱 baseline |
| sensitivity-gated MMR | 0.2742 | 0.2795 | +0.0040 | 保留弱 baseline |
| soft-label λ | — | — | expected≈fixed | 停止 |
| DPO step-wise λ | — | — | 全坍缩到 λ=0.7 | 停止 |

**决策**：Scalar λ 路线已充分探索并阶段性停止。**转向 oracle evidence set——直接搜索最优 K-subset，跳过 λ 和 MMR 两个间接层。**

---

### 4.D Oracle Evidence Set 与 Verifier 校准（05-16 ~ 05-19）

**动机**：Oracle λ 仍受 MMR greedy selection 约束；最优 evidence set 可能无法被任何单一 λ 的 MMR 选中。Oracle evidence set 直接回答"哪个 K-子集最好"。

#### 4.D.1 Oracle set gap 计算（05-16）

[`../plan/202605161147_oracle_evidence_selection.md`](../plan/202605161147_oracle_evidence_selection.md) 规划直接搜索最优 K-subset。搜索算法（`scripts/oracle_evidence/search_optimal_evidence.py`）：N≤15 穷举，N>15 贪婪前向选择。

[`202605161449_oracle_set_gap_analysis.md`](202605161449_oracle_set_gap_analysis.md) 给出关键结果（val split, semantic chunking θ=0.5, top_k=5, b3 LoRA checkpoint）：

| 指标 | Oracle (greedy) | MMR (λ=0.7) | Gap |
|---|---|---|---|
| Accuracy | **48.43%** | 29.67% | **+18.76 pp** |
| Macro F1 | **43.03%** | 30.03% | **+13.00 pp** |

这个 gap 比 oracle λ 的约 +3 pp 大得多（~6x），说明瓶颈已经不只是 λ，而是 MMR 单轴表达能力和 evidence set selection 本身。

事件级分解（1274 val 样本）：Oracle only correct 占 **31.0%**（selector 可直接修复），Both correct 仅 17.4%，Neither correct 占 39.3%（verifier 硬瓶颈）。

Per-label 分析暴露严重 **false-side bias**：pants-fire oracle accuracy 88.7%、false 89.6%，但 mostly-true 仅 21.9%、true 仅 1.2%（甚至低于 MMR）。verifier 有强烈的 false bias，即使选出了最优证据集，也不相信 claim 是真的。

[`202605161516_oracle_set_supervision_next_steps.md`](202605161516_oracle_set_supervision_next_steps.md) 将下一阶段目标改为 oracle evidence set supervision，选择 pointwise selector 作为最小可行实验。

#### 4.D.2 Pointwise selector V1（05-17）

[`../implementation/202605171203_oracle_pointwise_supervision_v1.md`](../implementation/202605171203_oracle_pointwise_supervision_v1.md) 实现 NumPy logistic regression selector。

Selection-only 结果（与 oracle set 的 overlap）：Recall@5=0.8495, Jaccard@5=0.7706（val full）。但这些指标后来被判定为**无效强信号**——候选池是重建的（先注入 oracle positives 再补 negatives），不代表正式 pipeline 的候选空间。

**下游 vLLM verifier 给出了反向结论**：

| run | split | accuracy | macro_f1 |
|---|---|---|---|
| pointwise_oracle_full (新训练 verifier) | **test** | **0.2230** | **0.2059** |
| fixed λ=0.7 baseline | test | 0.2702 | 0.2769 |

V1a 在 test 集上显著劣于 fixed-MMR（accuracy -4.72 pp, macro_f1 -7.10 pp）。mostly-true 的 per-class F1 仅 0.0685。

[`../implementation/202605180045_pointwise_v1b_true_side_anchor.md`](../implementation/202605180045_pointwise_v1b_true_side_anchor.md) 加入 low-weight true/mostly-true anchor 后 val 仅微升至 0.2630/0.2632，仍低于 fixed-MMR。

**决策**：停止 pointwise oracle selector V1。问题不在 selector 吸收能力，而在旧 oracle set 继承了旧 verifier 的 false-side bias。转向 verifier 校准 + re-oracle。

#### 4.D.3 Verifier 校准四阶段（05-18 ~ 05-19）

[`../plan/202605180118_oracle_calibration_reoracle_four_stage_plan.md`](../plan/202605180118_oracle_calibration_reoracle_four_stage_plan.md) 定义四阶段：

1. **Stage 1 — Label-token Weighted CE Verifier**：把训练从完整 target SFT loss 改成 `prompt + "Label:"` 后 A-F label token 的 weighted CE，加重 true-side 权重
2. **Stage 2 — Calibration-aware Re-Oracle**：用 margin objective（$P(y_{\text{gold}}) - \max_{y \neq y_{\text{gold}}} P(y)$）替代 gold_logprob
3. **Stage 3 — Filtered Preference / Utility Supervision**
4. **Stage 4 — Selector Training + Full Pipeline Evaluation**

**Stage 1 已完成**（[`../implementation/202605180118_label_token_ce_verifier_stage1.md`](../implementation/202605180118_label_token_ce_verifier_stage1.md)）：

类别权重：true=3.0, mostly-true=2.0, half-true=1.2, barely-true=1.2, false=1.0, pants-fire=1.0。Checkpoint 选择按 `macro_f1 + 0.5 × true_side_macro_f1`。

| 指标 | Stage 1 (val) | 旧 verifier MMR baseline (val) |
|---|---|---|
| accuracy | **0.3006** | 0.2967 |
| macro_f1 | **0.3015** | 0.3003 |

true-side 未退化：mostly-true F1=0.3419、true F1=0.3298，为全部 6 类中最高。但 barely-true（0.2707）和 half-true（0.2445）仍然偏低。

**Stage 2 粒度决策已完成**（[`202605191945_sentence_vs_semantic_stage2_oracle_decision.md`](202605191945_sentence_vs_semantic_stage2_oracle_decision.md)）：

| 指标 | sentence-level | semantic-level |
|---|---|---|
| paired accuracy (n=1709) | 0.6192 | 0.5407 |

sentence-level 高出 +7.85 pp。**主线转回 sentence-level oracle evidence supervision。**

#### 4.D.4 Oracle sentence direct verifier 强阳性（05-19）

[`202605192113_oracle_direct_verifier_result_and_next_plan.md`](202605192113_oracle_direct_verifier_result_and_next_plan.md) 直接把 sentence-level Stage2 oracle selected evidence 渲染成 verifier 训练样本（不训练 selector）：

| 口径 | accuracy | macro-F1 |
|---|---|---|
| 去重后 val | **0.7111** | **0.7169** |

高于 Stage1 verifier oracle correct rate 0.6593，净提升约 +5.18 pp。**verifier 可以吸收 sentence-level oracle evidence supervision。**

但[非 oracle evidence 对照](202605192141_oracle_direct_val_evidence_checks.md)确认了 selector gap：同一 oracle-direct verifier 在 fixed-MMR sentence evidence 上仅 0.2716/0.2663，在 pointwise evidence 上仅 0.2637/0.2596。**当前瓶颈从 verifier 能否吸收 evidence，转向 selector 能否在正式候选池中近似 oracle evidence set。**

---

### 4.E Selector 实验 Step1-4（05-19 ~ 05-22）

**动机**：Oracle sentence direct verifier 已验证 evidence supervision 可被吸收。下一步是训练 selector 模型在正式候选池中选出接近 oracle 质量的 evidence。

[`../plan/202605192148_new_selector_model_research_brief.md`](../plan/202605192148_new_selector_model_research_brief.md) 和 [`202605200216_selector_experiment_plan_and_literature_review.md`](202605200216_selector_experiment_plan_and_literature_review.md) 固定了路线：Step1 cross-encoder pairwise → Step3 listwise → Step4 sequential pointer → Step5 OPD → Step6 GRPO。

**统一合同**：chunk_mmr_fingerprint=`432dfc970e75`（sentence chunking），候选池为 Stage2 oracle 的 candidate_pool top15，输出 ordered top5。

[`202605200004_oracle_direct_order_sensitivity.md`](202605200004_oracle_direct_order_sensitivity.md) 将 selector gate 从纯 set overlap 扩展为 order-aware：

```
set metrics:  recall@5 ≥ 0.50, jaccard@5 ≥ 0.35
order metrics: top1_match, oracle_rank_ndcg@5, pairwise_order_acc@5
controls: hybrid_score top5, candidate_pool_order top5, random-order seeds
```

**对照基线**（所有 Step 共用，1274 claim val set）：

| 对照 | recall@5 | jaccard@5 | top1_match | oracle_rank_ndcg@5 | pairwise_order_acc@5 | 说明 |
|---|---|---|---|---|---|---|
| `hybrid_score_top5` | 0.3435 | 0.2294 | 0.1028 | 0.2872 | 0.5271 | 按 hybrid_score 降序取 top-5，纯检索基线 |
| `candidate_pool_order_top5` | 0.3435 | 0.2294 | 0.1028 | 0.2872 | 0.5271 | 与 hybrid_score_top5 等价（候选池本身按 hybrid_score 降序存储） |

`same_set_random_order_mean`：将模型预测的 evidence **集合**用 5 个随机种子（0-4）重排后取平均。其 recall@5 / jaccard@5 等于模型自身的 set metrics（因为是同一集合），order metrics 反映"给定模型选出的集合，随机排序 vs 模型排序的差距"——用于衡量模型的排序能力是否超过随机。

#### 4.E.1 Step1 Cross-encoder Pairwise → No-Go（05-20）

[`../implementation/202605201045_cross_encoder_pairwise_selector_step1.md`](../implementation/202605201045_cross_encoder_pairwise_selector_step1.md) 实现了三种 cross-encoder pairwise selector：

| 模型 | recall@5 | jaccard@5 | oracle_rank_ndcg@5 | top1_match | Gate |
|---|---|---|---|---|---|
| `deberta_pairwise` | 0.3739 | 0.2522 | 0.2646 | 0.0667 | No-Go |
| `modernbert_pairwise` | 0.3733 | 0.2536 | 0.2635 | 0.0659 | No-Go |
| `bge_reranker_base_pairwise` | 0.3736 | 0.2528 | 0.2593 | 0.0667 | No-Go |

对照（同一候选池，1274 claim）：

| 对照 | recall@5 | jaccard@5 | oracle_rank_ndcg@5 | top1_match | pairwise_order_acc@5 |
|---|---|---|---|---|---|
| `hybrid_score_top5` | 0.3435 | 0.2294 | 0.2872 | 0.1028 | 0.5271 |
| `same_set_random_order_mean` (deberta) | 0.3739 | 0.2522 | 0.2574 | 0.0647 | 0.5033 |
| `same_set_random_order_mean` (modernbert) | 0.3744 | 0.2546 | 0.2685 | 0.0747 | 0.5101 |
| `same_set_random_order_mean` (bge) | 0.3736 | 0.2528 | 0.2545 | 0.0666 | 0.5089 |

三组 cross-encoder 在所有 gate 条件上均不达标。set metrics 略高于 hybrid_score_top5（+0.03 recall@5），但远低于 0.50 gate。order metrics 中 top1_match 全部低于 hybrid_score（0.103），模型未能可靠识别最优证据；pairwise_order_acc@5 与 hybrid_score（0.5271）接近甚至略低，说明模型学到的排序信号不比纯 hybrid_score 强。`same_set_random_order_mean` 的 order metrics 全部低于 hybrid_score_top5——即使用模型选的集合，随机排序也比 hybrid_score 排序差，说明模型集合的"可排序性"不如 hybrid_score 集合。

#### 4.E.2 Step3 Listwise Selector → No-Go（05-20）

[`../implementation/202605201519_set_aware_listwise_selector_step3.md`](../implementation/202605201519_set_aware_listwise_selector_step3.md) 实现了 set-aware listwise 15-candidate reranker：

| 模型 | recall@5 | jaccard@5 | top1_match | oracle_rank_ndcg@5 | pairwise_order_acc@5 | Gate |
|---|---|---|---|---|---|---|
| `deberta_listwise` | 0.3689 | 0.2484 | 0.1162 | 0.3072 | 0.5633 | No-Go |
| `deberta_listwise_shuffle03` | 0.3732 | 0.2518 | 0.1279 | 0.3131 | 0.5592 | No-Go |
| `deberta_listwise_rank_ablation` | 0.3826 | 0.2588 | 0.1122 | 0.3108 | 0.5168 | No-Go |

去掉 rank prior 后仅小幅改善（recall@5 从 0.3689→0.3826），说明 rank shortcut 不是唯一问题。listwise set encoder 本身仍没学到足够强的 oracle set utility。

对照：

| 对照 | recall@5 | jaccard@5 | oracle_rank_ndcg@5 | top1_match | pairwise_order_acc@5 |
|---|---|---|---|---|---|
| `hybrid_score_top5` | 0.3435 | 0.2294 | 0.2872 | 0.1028 | 0.5271 |
| `same_set_random_order_mean` (listwise) | 0.3689 | 0.2484 | 0.2919 | 0.0785 | 0.4941 |
| `same_set_random_order_mean` (shuffle03) | 0.3732 | 0.2518 | 0.2974 | 0.0912 | 0.5018 |
| `same_set_random_order_mean` (rank_ablation) | 0.3826 | 0.2588 | 0.2922 | 0.0738 | 0.4857 |

三种 listwise 模型的 pairwise_order_acc@5 均高于其对应的 random 基线（如 rank_ablation 0.5168 vs random 0.4857），说明模型学到了一定的排序能力；但与 hybrid_score_top5（0.5271）相比增量有限，且 top1_match 全部低于 hybrid_score（0.1028）。

#### 4.E.3 Step4 Sequential Pointer Selector（05-20 ~ 05-21）

[`../implementation/202605202145_sequential_pointer_selector_step4.md`](../implementation/202605202145_sequential_pointer_selector_step4.md) 实现了 supervised Sequential Pointer Selector——按 oracle greedy order 做 teacher forcing，逐步选出 evidence。

| 模型 | recall@5 | jaccard@5 | top1_match | oracle_rank_ndcg@5 | pairwise_order_acc@5 | step0 acc | Gate |
|---|---|---|---|---|---|---|---|
| `deberta_sequential_deep` | 0.3852 | 0.2615 | 0.1664 | 0.3306 | 0.5871 | 0.1664 | No-Go |
| `deberta_sequential_deep_mask02` | 0.3758 | 0.2531 | 0.1625 | 0.3322 | 0.5777 | 0.1625 | No-Go |
| `deberta_sequential_deep_mask05` | 0.3662 | 0.2472 | 0.1499 | 0.3270 | 0.5764 | 0.1499 | No-Go |

**相对 Step3 的改善在 order metrics**：top1_match 从 0.1122 → 0.1664，oracle_rank_ndcg@5 从 0.3108 → 0.3306，pairwise_order_acc@5 从 0.5168 → 0.5871。但 set metrics 仍停在 recall@5≈0.385、Jaccard@5≈0.262，离 gate 很远。

对照：

| 对照 | recall@5 | jaccard@5 | oracle_rank_ndcg@5 | top1_match | pairwise_order_acc@5 |
|---|---|---|---|---|---|
| `hybrid_score_top5` | 0.3435 | 0.2294 | 0.2872 | 0.1028 | 0.5271 |
| `same_set_random_order_mean` (deep) | 0.3852 | 0.2615 | 0.3005 | 0.0724 | 0.4862 |
| `same_set_random_order_mean` (mask02) | 0.3758 | 0.2531 | 0.2996 | 0.0683 | 0.4706 |
| `same_set_random_order_mean` (mask05) | 0.3662 | 0.2472 | 0.3001 | 0.0760 | 0.4890 |

Sequential pointer 的 pairwise_order_acc@5（deep 0.5871）显著高于 random 基线（0.4862）和 hybrid_score（0.5271），说明 sequential modeling 确实学到了有意义的 evidence ordering——这也是 Step4 相对 Step3 的核心增益。top1_match（0.1664）已超过 hybrid_score（0.1028），但 set metrics 未突破。关键在于：模型排序能力在改善，但第一步仍大量选错（step0 acc=0.1664），后续步骤受 prefix drift 影响进一步偏离。

Step4.1-A 的 mask BCE 变体（mask02/mask05）均未超过 deep-only set metrics，也未达到低成本参考线（recall@5≥0.40 / jaccard@5≥0.275）。

**决策**（[`../implementation/202605202008_sequential_pointer_selector_step4_plan.md`](../implementation/202605202008_sequential_pointer_selector_step4_plan.md)）：Sequential pointer 是有效结构（改善了顺序建模），但 set gate 未突破。后续已补跑轻量 full pipeline probe（见 4.E.4），结论是有增益但不足以支持进入正式 Step5 OPD。step0 accuracy 仅 0.1664，说明即使在 oracle-prefix 条件下，模型也无法可靠地选出第一个 evidence。优先补 evidence utility 表示而非解决 exposure bias。

#### 4.E.4 Selector Trace Full Pipeline 验证（05-22）

为检查 selection-only 指标能否转化为最终 verifier 指标，补跑 `outputs/runs/b3_selector_trace_full_pipeline`：从 selector/control trace 构造 train/val evidence，重新训练 label-token CE LoRA verifier，再在 val 上 infer。两个 run 都覆盖 1274 条 val claim，`parse_error_rate=0.0`，`val_predictions.jsonl` 无重复 `sample_idx`。

| Evidence 来源 | recall@5 | jaccard@5 | top1_match | oracle_rank_ndcg@5 | pairwise_order_acc@5 | full val accuracy | full val macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hybrid_score_top5` | 0.3435 | 0.2294 | 0.1028 | 0.2872 | 0.5271 | 0.2834 | 0.2878 |
| `deberta_sequential_deep` | 0.3852 | 0.2615 | 0.1664 | 0.3306 | 0.5871 | 0.3132 | 0.3026 |

`deberta_sequential_deep` 相对 `hybrid_score_top5` 的 selection-only 提升为 recall@5 +0.0418、jaccard@5 +0.0321、top1_match +0.0636、oracle_rank_ndcg@5 +0.0434、pairwise_order_acc@5 +0.0600；但 full pipeline 只提升 accuracy +0.0298、macro-F1 +0.0148。说明当前 selector 的 set/order 增益可以带来下游改善，但转化效率偏弱，远低于 oracle sentence direct verifier 的 0.7111/0.7169 上界。

阶段结论：full pipeline probe 没有推翻 Step4 No-Go。Sequential pointer 的排序能力确实优于 hybrid baseline，但 recall@5≈0.385 仍太低，且新增 evidence overlap 未稳定转化为 verifier 可用的判别信息。下一步仍应转向 verifier-aware utility、oracle-margin distillation 或 prefix-level evidence contribution，而不是直接进入 OPD/GRPO。

#### 4.E.5 Step4.1-D VIG Utility Analysis（05-22，部分运行）

[`../implementation/202605221430_oracle_vig_utility_analysis.md`](../implementation/202605221430_oracle_vig_utility_analysis.md) 实现了 VIG（verifier information gain / verifier-induced gain）诊断链路。该实验不把 oracle selected evidence 直接当黑盒 imitation target，而是重新打分 oracle prefix 下的候选 evidence，把 verifier margin 变化拆成可审计信号：

```text
prefix marginal utility:
u_t(i) = margin_verifier(claim, prefix_t + candidate_i) - margin_verifier(claim, prefix_t)

final-set counterfactual:
contribution(i) = margin_verifier(claim, oracle_final_set)
                - margin_verifier(claim, oracle_final_set without / replaced i)
```

新增入口为 `scripts/selectors/generate_oracle_vig_cache.py`、`scripts/selectors/analyze_oracle_vig_utility.py` 和 `scripts/selectors/run_oracle_vig_utility_analysis.sh`。默认输出目录为 `outputs/selectors/vig_utility/stage2_margin_val/`；完整 val 预期覆盖 1274 条 claim，约 82810 条 prefix marginal rows 与 70070 条 final-set counterfactual rows。

当前只记录为**部分运行**：已有产物不是最终版本，不能作为 Stop/Go 指标。最终全量版本需要至少核查以下口径后再写结论：

| 诊断口径 | 作用 | 判定口径 |
|---|---|---|
| `true_delta_margin_oracle_probe` | 检查 VIG 重新打分是否复现 Stage2 oracle 的 greedy choice | `step_top1_match >= 0.90` 才能认为 cache / prompt / LoRA / max length 基本对齐 |
| Delta decomposition | 区分 evidence 是提高 gold logprob、压低 best-wrong logprob，还是二者同时发生 | 汇总 `delta_margin / delta_gold_logprob / delta_best_wrong_logprob` 的 target-vs-nontarget 差异 |
| Feature-group probe | 判断 retrieval、text_overlap、prefix_state、single_verifier 等可解释特征能否解释 oracle margin gain | `all_feature target AUROC >= 0.60`，且 step_top1_match 比 hybrid-rank baseline 高至少 3pp 才考虑 utility distillation |
| Final-set counterfactual | 判断 greedy oracle final set 中是否存在有害、冗余或可替代 evidence | 重点看 `selected_harmful_final_rate` 与 `selected_replaceable_rate` |

阶段结论：VIG 是下一步 verifier-aware utility / oracle-margin distillation 的必要诊断，但目前还没有最终全量结果。当前不能据部分运行结果推进新 selector 训练，也不能把 partial cache 的指标写成最终结论；全量 VIG self-check 通过后，再决定是否进入 utility feature distillation。

---

### 4.F Targeted Feature 前置诊断（05-21 ~ 05-22）

**动机**：Step4 的 deep semantic interaction 不足以突破 set gate。在将 stance/aspect 特征接入 selector 前，先离线诊断这些特征是否包含 selection signal。

#### 4.F.1 Stance NLI 诊断 → stop_or_calibrate

[`../implementation/202605212330_oracle_stance_nli_diagnostic.md`](../implementation/202605212330_oracle_stance_nli_diagnostic.md) 用 DeBERTa-v3-base 在 MNLI/FEVER/ANLI 上做 NLI stance 打分：

| 指标 | 当前值 | Go 阈值 |
|---|---|---|
| support/refute selected-vs-pool lift | -0.60 pp | ≥ +5 pp |
| best stance feature separability AUROC | 0.5090 | ≥ 0.57 |

当前 NLI 模型对 LIAR-RAW sentence candidate 的 claim-evidence 关系过度判为 neutral（79.20%），stance scalar 暂不接入 selector。

#### 4.F.2 Rule-based aspect coverage → stop_or_refine

[`../implementation/202605212350_oracle_aspect_coverage_diagnostic.md`](../implementation/202605212350_oracle_aspect_coverage_diagnostic.md) 用规则提取 claim 的原子化 aspects，计算 evidence 对 aspects 的 coverage：

| 指标 | 当前值 | Go 阈值 |
|---|---|---|
| uncovered_gain AUROC | 0.4820 | ≥ 0.57 |
| oracle vs hybrid coverage lift | -0.07 pp | ≥ +3 pp |

更换为 BGE encoder 的 100 条 sample probe 也未打开信号（AUROC=0.4670），说明问题不只是裸 DeBERTa encoder。

#### 4.F.3 LLM decomp+ aspect coverage → No-Go（05-22 最终结论）

[`../implementation/202605212359_llm_decomp_plus_aspect_coverage.md`](../implementation/202605212359_llm_decomp_plus_aspect_coverage.md) 用 Qwen2.5-7B-Instruct + vLLM 做 claim decomposition 生成 sub-claims，再计算 coverage。

**第一次尝试（guided JSON）**：parse_failed=655/1274，大量输出为 prompt/schema 残片，生成质量严重失效。

**第二次尝试（plain JSON, max_tokens=256, tensor_parallel_size=2）**：全量生成 1274/1274 条，生成质量已满足 stop/go 判定线：

```text
parse_failures = 1 / 1274
claims_with_no_local_aspects = 1 / 1274
n_local_aspects = 2962
valid_subclaims_per_claim_mean = 2.33
parse_status.ok = 1192 / 1274
fewer_than_min_valid_subclaims = 81 / 1274
```

但 coverage gate 仍明确 No-Go：

| 指标 | 当前值 | Go 阈值 |
|---|---|---|
| uncovered_gain AUROC | 0.4730 | ≥ 0.57 |
| oracle vs hybrid coverage lift | -1.51 pp | ≥ +3 pp |
| oracle coverage mean | 0.8743 | — |
| hybrid top5 coverage mean | 0.8893 | — |
| oracle_beats_hybrid_rate | 8.16% | — |

step-wise probe 也没有可用选择信号：

| 指标 | 当前值 |
|---|---|
| uncovered_gain positive_mean | 0.1759 |
| covered_overlap AUROC | 0.5127 |
| max_aspect_score AUROC | 0.4983 |
| mean_aspect_score AUROC | 0.4953 |

oracle set 在该 proxy 下反而比 hybrid top5 覆盖更低（0.8743 vs 0.8893），只有 8.16% 的样本 oracle 超过 hybrid。

**最终决策**：这次不是解析或缓存失败——Qwen decomp+ plain 生成已经基本可用，但 claim-aspect semantic coverage 仍不能解释 Stage2 oracle selected evidence。**停止 claim-aspect coverage 作为 Step4 主线，不进入 `deberta_sequential_aspect` 训练。** 不建议继续投入更强 LLM、闭源 API 或规则增强作为主线优化。只保留可选的小样本 sanity check（把 aspect-candidate alignment 从 embedding cosine 换成 cross-encoder / NLI entailment scorer）。下一步主线转向 verifier-aware utility、prefix-level evidence contribution 或 oracle-margin distillation。

---

## 5. 总体结论

1. **Evidence selection 是核心瓶颈**：从 classifier collapse 到 direct verifier 对照实验，一致指向 evidence quality/distribution 而非模型结构或损失函数。

2. **Fixed λ=0.7 是强 baseline**（test accuracy=0.2702, macro-F1=0.2769），不应被视为容易击败的弱方法。

3. **Scalar λ 路线已充分探索并关闭**：oracle λ 证明 adaptive λ 有约 3pp 理论收益，但 hard predictor（$R^2 \approx 0.01$）、soft-label（退化）、DPO step-wise（4 轮全坍缩）均无法形成可用策略。`log(n_candidates)`（+0.0064）和 sensitivity-gated（+0.0040）仅保留为弱 baseline。

4. **Oracle evidence set 上界远大于 oracle λ**：+18.76pp accuracy / +13.00pp macro-F1（vs oracle λ 的 +3pp），说明应学习 evidence set/selector/utility 而非继续预测 λ。

5. **Verifier 校准有效但不足以独立解决**：Label-token weighted CE 阻止了 false-side bias（mostly-true F1=0.3419, true F1=0.3298），但中间类仍是难点。

6. **Oracle sentence direct verifier 强阳性**（accuracy 0.7111）：verifier 可吸收 oracle evidence supervision；瓶颈在 selector 能否在正式候选池中近似 oracle evidence set。

7. **Step1-3 Selector 均 No-Go**：cross-encoder pairwise 和 listwise 都停在 recall@5≈0.38、Jaccard@5≈0.26，远低于 0.50/0.35 gate。

8. **Step4 sequential pointer 改善 order 但 set 未突破**：top1_match 0.1664、recall@5=0.3852。full pipeline probe 中 sequential evidence 训练出的 verifier 达到 0.3132/0.3026，高于 hybrid_score_top5 的 0.2834/0.2878，但增益仍远小于 oracle evidence 上界。当前 priority 是 evidence utility 表示而非 exposure bias（OPD）；VIG 诊断已启动但仅部分运行，尚未形成最终 utility 结论。

9. **Stance/aspect targeted features 全部 No-Go**：NLI stance 过度 neutral（AUROC=0.5090）；rule aspect coverage 信号接近随机（AUROC=0.4820）；LLM decomp+ plain 重跑生成质量合格但 coverage 仍 No-Go（AUROC=0.4730, oracle 覆盖率反低于 hybrid）。claim-aspect coverage 主线已关闭。

10. **sentence-level 优于 semantic-level**：paired 对比 +7.85pp。后续主线使用 sentence-level oracle evidence supervision。

## 6. Stop / Go 决策状态

| 方向 | 状态 | 原因 |
|---|---|---|
| ModernBERT 判别式分类器 | 停止 | 指标接近随机/塌陷，证据质量瓶颈更明显 |
| hard learned-λ predictor | 停止 | oracle λ margin 极小（72.6%<0.05），$R^2\approx 0.01$ |
| high-margin / 3-bin λ 修复 | 停止 | 过滤和粗分类仍未提供稳定可学信号 |
| soft-label scalar λ | 停止 | utility curve 接近均匀，expected 退化 |
| DPO step-wise scalar λ | 停止 | 4 轮全坍缩到 λ=0.7 |
| GRPO refinement | 暂不跑 | 前置 DPO 或 stable offline policy 不满足 |
| `log(n)` / sensitivity-gated | 保留弱 baseline | +0.004-0.006，不继续深挖 |
| multi-weight MMR | 待定 | re-oracle 后再决定 |
| Pointwise oracle selector (V1a/V1b) | 停止 | downstream test 低于 fixed-MMR；旧 gate 无效 |
| Cross-encoder pairwise (Step1) | 停止 | 三组模型均未过 gate |
| Set-aware listwise (Step3) | 停止 | 最好 recall@5=0.3826，未接近 0.50 gate |
| Sequential pointer (Step4) | 停止当前结构，保留诊断价值 | order 改善且 full pipeline 略升，但 recall@5 仍低、macro-F1 转化有限 |
| Step4.1-A mask BCE | 停止当前变体 | 未超过 deep-only set metrics |
| Step4.1-D VIG utility analysis | 部分运行，待全量 | 诊断链路已实现；partial 产物不是最终版本，需先过 true-delta self-check 再做 utility distillation |
| Stance NLI scalar | 暂停，先校准 | AUROC=0.5090，过度 neutral |
| Rule-based aspect coverage | 停止，先 refine | AUROC=0.4820，负 lift |
| LLM decomp+ aspect coverage | 停止 | plain 重跑生成质量合格，但 coverage 指标全部未过 gate（AUROC=0.4730，oracle 覆盖率反低于 hybrid）；claim-aspect 主线关闭 |
| Oracle-set supervision | 继续，sentence-level | direct verifier 已验证可吸收 |
| Oracle sentence direct verifier | Upper-bound probe | 非 oracle evidence 回到 0.26-0.27 |
| Selector trace full pipeline probe | 已完成 | sequential 0.3132/0.3026 > hybrid 0.2834/0.2878，但未接近 oracle 上界 |
| Semantic-level oracle | 诊断保留 | paired 低于 sentence +7.85pp |
| Label-token CE verifier | 已完成 Stage1 | true-side 未退化，待补 test infer |
| Stage2 margin re-oracle | 转向 sentence 主线 | sentence train oracle 完成 |

## 7. 下一步行动

### 7.1 已完成里程碑

- [x] Stage 1 label-token CE verifier 的 val 指标确认（accuracy 0.3006, macro_f1 0.3015）
- [x] Stage 2 粒度决策（sentence > semantic，主线使用 `outputs/oracle_evidence/stage2_margin_train_sharded`）
- [x] Oracle sentence evidence direct verifier（val accuracy 0.7111, macro-F1 0.7169）
- [x] Oracle-direct verifier 的非 oracle evidence 对照（fixed-MMR 0.2716, pointwise 0.2637）
- [x] Step1 cross-encoder pairwise selector（三组模型 No-Go）
- [x] Step3 set-aware listwise selector（No-Go）
- [x] Step4 supervised sequential pointer selector（第一轮完成，order 改善但 set gate 未过）
- [x] Selector trace full pipeline probe（sequential 0.3132/0.3026 > hybrid 0.2834/0.2878，但转化有限）
- [x] VIG utility analysis 诊断链路实现（当前仅部分运行，最终指标待全量产物）
- [x] LLM decomp+ full-val plain 重跑并给出最终结论（生成质量合格，coverage 全部 No-Go，claim-aspect 主线关闭）

### 7.2 当前活跃事项

1. **[P0]** 完成 VIG utility analysis 全量运行并先核查 `true_delta_margin_oracle_probe` self-check。当前 partial 产物不是最终版本；只有 cache / prompt / LoRA / max length 与 Stage2 oracle 对齐后，才解释 feature-group probe 和 final-set counterfactual。

2. **[P0]** 转向 verifier-aware utility 或 oracle-margin distillation 作为 selector 的监督信号。当前 claim-aspect coverage 主线已关闭（NLI/rule_aspect/LLM decomp+ 三者均 No-Go），应直接从 oracle 构造目标（margin objective）出发定义 evidence utility；VIG 全量结果是是否进入 utility distillation 的前置依据。

3. **[P1]** 修正 eval metric 去重：`val_predictions.jsonl` / distributed gather 输出按唯一 `sample_idx` 去重后再算正式 eval 指标，避免 padding 样本影响 checkpoint 选择与报告口径。

4. **[P1]** 新 selector 仍应先过 order-aware selection-only gate（recall@5≥0.50, jaccard@5≥0.35），再把 full pipeline 作为确认实验。当前 sequential full pipeline 虽优于 hybrid_score_top5，但 recall@5≈0.386、Jaccard@5≈0.262 的增益只带来 macro-F1 +0.0148。

5. **[P2]** 暂不进入正式 Step5 OPD / GRPO。OPD 只解决 on-policy prefix drift；当前 step0 accuracy 仍低（0.1664），且 full pipeline 转化有限，优先补 evidence utility 表示。

6. **[P2]** Semantic-level oracle 仅保留 paired diagnostic 或报告对照，不建议等权推进完整 train oracle。

7. **[P3]** 待补 Stage1 label-token CE verifier 的 test infer。

## 8. 关键文件索引

### Pipeline 核心

| 文件 | 作用 |
|---|---|
| `src/fact_checking/pipeline/run.py` | Hydra 入口（`@hydra.main`），PipelineRunner 编排 |
| `src/fact_checking/pipeline/runner.py` | build → train → infer 三阶段编排 |
| `src/fact_checking/build/candidates.py` | Build 主逻辑（chunking、检索、MMR、Prompt 构建） |
| `src/fact_checking/build/chunking.py` | 5 类 chunking 策略实现 |
| `src/fact_checking/retrieval/mmr.py` | MMR 算法 |
| `src/fact_checking/retrieval/embedder.py` | BGE-base-en-v1.5 文本嵌入封装 |
| `src/fact_checking/retrieval/text_utils.py` | 词汇分（F1）、BM25 分计算 |
| `src/fact_checking/data/constants.py` | 标签常量（6 类、3 类、字母映射） |
| `src/fact_checking/infer/api.py` | vLLM API 推理与 server 管理 |
| `src/sft/trainer.py` | 生成式 SFT 训练器 |
| `src/sft/parser.py` | 标签解析 |
| `src/sft/metrics.py` | 分类指标计算 |

### Learned Lambda / RL-MMR

| 文件 | 作用 |
|---|---|
| `src/fact_checking/learned_lambda/predictor.py` | `ChunkEmbeddingLambdaEncoder`、`LambdaPredictor` |
| `src/fact_checking/learned_lambda/features.py` | 73 维手工特征提取 |
| `scripts/learned_lambda/compute_oracle_lambda.py` | oracle λ 计算（21 网格） |
| `src/fact_checking/rl_mmr/sensitivity.py` | 敏感度特征提取 + 门控决策 |
| `src/fact_checking/rl_mmr/dpo_policy.py` | `StepLambdaPolicy`, `dpo_loss()` |
| `src/fact_checking/rl_mmr/trajectory.py` | `Trajectory`, `PreferencePair` dataclasses |

### Oracle Evidence / Selector

| 文件 | 作用 |
|---|---|
| `scripts/oracle_evidence/search_optimal_evidence.py` | Oracle evidence search 主脚本 |
| `src/fact_checking/oracle_evidence/scorer.py` | vLLM scoring（gold_logprob / margin） |
| `src/fact_checking/oracle_evidence/search.py` | Search 主逻辑 |
| `src/fact_checking/selectors/stage2_oracle.py` | Stage2 oracle 契约检查与数据加载 |
| `src/fact_checking/selectors/metrics.py` | Ordered selector metrics 与 controls |
| `src/fact_checking/selectors/cross_encoder.py` | Step1 cross-encoder pairwise selector |
| `src/fact_checking/selectors/listwise.py` | Step3 set-aware listwise selector |
| `src/fact_checking/selectors/sequential.py` | Step4 sequential pointer selector |
| `src/fact_checking/selectors/base.py` | 通用 selector 契约与 registry |
| `src/fact_checking/selectors/aspects.py` | Rule-based / LLM claim aspect 提取 |
| `scripts/selectors/train_cross_encoder_pairwise.py` | Step1 训练入口 |
| `scripts/selectors/train_listwise_selector.py` | Step3 训练入口 |
| `scripts/selectors/train_sequential_selector.py` | Step4 训练入口 |
| `scripts/selectors/analyze_oracle_stance_distribution.py` | Stance/NLI 诊断 |
| `scripts/selectors/analyze_oracle_aspect_coverage.py` | Aspect coverage 诊断 |
| `scripts/selectors/generate_llm_claim_decomp_aspects.py` | LLM decomp+ aspect 生成 |

### Verifier Calibration

| 文件 | 作用 |
|---|---|
| `src/sft/label_token_dataset.py` | `prompt + Label:` 训练样本构造 |
| `src/sft/label_token_trainer.py` | A-F label token weighted CE 训练 |
| `configs/experiment/b3_label_token_ce_1024.yaml` | Stage 1 实验配置 |
| `configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml` | Oracle direct verifier 配置 |
| `scripts/verifier/run_label_token_ce_stage1.sh` | Stage 1 一键 wrapper |
| `scripts/oracle_evidence/run_reoracle_stage2.sh` | Stage 2 margin re-oracle wrapper |

### 核心配置文件

| 文件 | 作用 |
|---|---|
| `configs/build/default.yaml` | Build 默认配置 |
| `configs/train/default.yaml` | 训练默认配置 |
| `configs/infer/vllm_api.yaml` | 推理默认配置 |
| `configs/pipeline/default.yaml` | Pipeline 编排默认配置 |
| `configs/experiment/b0.yaml` | 生成式 SFT 基线 |
| `configs/experiment/b4.yaml` | 判别式分类器基线 |
| `configs/experiment/mmr_lambda_sweep.yaml` | MMR λ sweep |
| `configs/experiment/mmr_sensitivity_gated.yaml` | Sensitivity-gated MMR |
| `configs/experiment/mmr_dpo_step_lambda.yaml` | DPO step-wise λ |
| `configs/experiment/b3_cross_encoder_stage2_sentence_1024.yaml` | Cross-encoder selector build 配置 |
| `configs/experiment/b3_listwise_stage2_sentence_1024.yaml` | Listwise selector build 配置 |
| `configs/experiment/b3_sequential_stage2_sentence_1024.yaml` | Sequential selector build 配置 |

### 文档索引

当前 `docs/` 下文档按实验线程归类：

| 线程 | 文档 |
|---|---|
| A. 分块与基础设施 | [`../plan/202605080025_chunking.md`](../plan/202605080025_chunking.md), [`../visualization/202605121618_chunking_evidence_examples.md`](../visualization/202605121618_chunking_evidence_examples.md), [`../plan/202605131653_dvc-warm-minsky.md`](../plan/202605131653_dvc-warm-minsky.md), [`../implementation/202605171700_tailscale_container_ssh_vscode.md`](../implementation/202605171700_tailscale_container_ssh_vscode.md) |
| B. 分类器塌陷 | [`202605111212_classifier-collapse-analysis.md`](202605111212_classifier-collapse-analysis.md) |
| C. MMR / Learned-λ | [`../implementation/202605111255_mmr-lambda-sweep-pipeline.md`](../implementation/202605111255_mmr-lambda-sweep-pipeline.md), [`202605112128_val_test_Mismatch.md`](202605112128_val_test_Mismatch.md), [`202605131520_val-test-f1-gap-diagnosis.md`](202605131520_val-test-f1-gap-diagnosis.md), [`../learned_lambda/`](../learned_lambda/) (3 篇), [`../implementation/202605141531_sensitivity-gated-mmr.md`](../implementation/202605141531_sensitivity-gated-mmr.md), [`../plan/202605141828_soft_label_policy.md`](../plan/202605141828_soft_label_policy.md), [`202605141045_RL_MMR_research_review.md`](202605141045_RL_MMR_research_review.md), [`202605151453_RL_MMR_direction_summary.md`](202605151453_RL_MMR_direction_summary.md), 以及 `../plan/` 下的 RL-MMR / DPO 计划 (6 篇) |
| D. Oracle Evidence Set + Verifier 校准 | [`../plan/202605161147_oracle_evidence_selection.md`](../plan/202605161147_oracle_evidence_selection.md), [`202605161449_oracle_set_gap_analysis.md`](202605161449_oracle_set_gap_analysis.md), [`202605161516_oracle_set_supervision_next_steps.md`](202605161516_oracle_set_supervision_next_steps.md), [`../implementation/202605171203_oracle_pointwise_supervision_v1.md`](../implementation/202605171203_oracle_pointwise_supervision_v1.md), [`../implementation/202605180045_pointwise_v1b_true_side_anchor.md`](../implementation/202605180045_pointwise_v1b_true_side_anchor.md), [`../plan/202605180118_oracle_calibration_reoracle_four_stage_plan.md`](../plan/202605180118_oracle_calibration_reoracle_four_stage_plan.md), [`../implementation/202605180118_label_token_ce_verifier_stage1.md`](../implementation/202605180118_label_token_ce_verifier_stage1.md), [`../implementation/202605181113_calibration_aware_reoracle_stage2.md`](../implementation/202605181113_calibration_aware_reoracle_stage2.md), [`202605191945_sentence_vs_semantic_stage2_oracle_decision.md`](202605191945_sentence_vs_semantic_stage2_oracle_decision.md), [`../implementation/202605192010_oracle_sentence_direct_verifier.md`](../implementation/202605192010_oracle_sentence_direct_verifier.md), [`202605192113_oracle_direct_verifier_result_and_next_plan.md`](202605192113_oracle_direct_verifier_result_and_next_plan.md), [`202605192141_oracle_direct_val_evidence_checks.md`](202605192141_oracle_direct_val_evidence_checks.md) |
| E. Selector Step1-4 | [`../plan/202605192148_new_selector_model_research_brief.md`](../plan/202605192148_new_selector_model_research_brief.md), [`202605200004_oracle_direct_order_sensitivity.md`](202605200004_oracle_direct_order_sensitivity.md), [`202605200216_selector_experiment_plan_and_literature_review.md`](202605200216_selector_experiment_plan_and_literature_review.md), [`../implementation/202605201045_cross_encoder_pairwise_selector_step1.md`](../implementation/202605201045_cross_encoder_pairwise_selector_step1.md), [`../implementation/202605201519_set_aware_listwise_selector_step3.md`](../implementation/202605201519_set_aware_listwise_selector_step3.md), [`../implementation/202605202008_sequential_pointer_selector_step4_plan.md`](../implementation/202605202008_sequential_pointer_selector_step4_plan.md), [`../implementation/202605202145_sequential_pointer_selector_step4.md`](../implementation/202605202145_sequential_pointer_selector_step4.md), [`202605212200_sequential_selector_architecture_notes.md`](202605212200_sequential_selector_architecture_notes.md) |
| F. Targeted Feature 诊断 | [`../implementation/202605212330_oracle_stance_nli_diagnostic.md`](../implementation/202605212330_oracle_stance_nli_diagnostic.md), [`../implementation/202605212350_oracle_aspect_coverage_diagnostic.md`](../implementation/202605212350_oracle_aspect_coverage_diagnostic.md), [`../implementation/202605212359_llm_decomp_plus_aspect_coverage.md`](../implementation/202605212359_llm_decomp_plus_aspect_coverage.md) |
