# 实验总进度时间线与文档索引

生成日期：2026-05-19

本文整理 `docs/` 目录下截至当前的计划、分析、实现与运维文档，按实验推进时间线归纳总体进度。本文只汇总已有文档与已记录结果，不新增实验结论，不重新解释未验证产物。

## 1. 文档版图

当前 `docs/` 下文档可分为 6 类：

| 类别 | 文档 | 作用 |
|---|---|---|
| 检索与分块基础 | [`../plan/202605080025_chunking.md`](../plan/202605080025_chunking.md), [`../visualization/202605121618_chunking_evidence_examples.md`](../visualization/202605121618_chunking_evidence_examples.md) | 固定 evidence chunk 作为后续检索单元，并用示例比较 `sentence/raw/ctx_window/semantic/ctx_semantic` 的效果。 |
| 早期训练与评估诊断 | [`202605111212_classifier-collapse-analysis.md`](202605111212_classifier-collapse-analysis.md), [`202605112128_val_test_Mismatch.md`](202605112128_val_test_Mismatch.md), [`202605131520_val-test-f1-gap-diagnosis.md`](202605131520_val-test-f1-gap-diagnosis.md) | 诊断分类器塌陷、MMR λ sweep 的 val/test 差距、训练/推理路径差异与泛化差距。 |
| MMR / learned-λ 主线 | [`../implementation/202605111255_mmr-lambda-sweep-pipeline.md`](../implementation/202605111255_mmr-lambda-sweep-pipeline.md), [`../learned_lambda/202605122054_generate_oracle_prompts.md`](../learned_lambda/202605122054_generate_oracle_prompts.md), [`../learned_lambda/202605141045_verification_experiment.md`](../learned_lambda/202605141045_verification_experiment.md), [`../learned_lambda/202605141052_analysis.md`](../learned_lambda/202605141052_analysis.md), [`../plan/202605141045_Improving_Learned_Lambda.md`](../plan/202605141045_Improving_Learned_Lambda.md), [`../implementation/202605141531_sensitivity-gated-mmr.md`](../implementation/202605141531_sensitivity-gated-mmr.md), [`../plan/202605141828_soft_label_policy.md`](../plan/202605141828_soft_label_policy.md) | 从固定 λ 扫描到 oracle λ 验证、hard predictor 失败、sensitivity-gated、soft-label λ 的完整探索。 |
| RL-MMR / DPO 计划与阶段结论 | [`202605141045_RL_MMR_research_review.md`](202605141045_RL_MMR_research_review.md), [`../plan/202605111704_RL_MMR_experiment_plan_v1.md`](../plan/202605111704_RL_MMR_experiment_plan_v1.md), [`202605151453_RL_MMR_direction_summary.md`](202605151453_RL_MMR_direction_summary.md), [`../plan/202605151936_dpo_step_wise_lambda.md`](../plan/202605151936_dpo_step_wise_lambda.md), [`../plan/202605152032_dpo_stepwise_lambda.md`](../plan/202605152032_dpo_stepwise_lambda.md), [`../plan/202605161049_RL_MMR_experiment_plan_v2.md`](../plan/202605161049_RL_MMR_experiment_plan_v2.md) | 形成 fixed → heuristic → gated → soft-label → DPO step-wise → multi-weight → GRPO 的有序实验路线，并记录 scalar λ 方向的停止条件。 |
| Oracle evidence set / selector 主线 | [`../plan/202605161147_oracle_evidence_selection.md`](../plan/202605161147_oracle_evidence_selection.md), [`202605161449_oracle_set_gap_analysis.md`](202605161449_oracle_set_gap_analysis.md), [`202605161516_oracle_set_supervision_next_steps.md`](202605161516_oracle_set_supervision_next_steps.md), [`../plan/202605170118_oracle_set_supervision_next_steps.md`](../plan/202605170118_oracle_set_supervision_next_steps.md), [`../plan/202605171203_oracle_pointwise_supervision_v1.md`](../plan/202605171203_oracle_pointwise_supervision_v1.md), [`../implementation/202605171203_oracle_pointwise_supervision_v1.md`](../implementation/202605171203_oracle_pointwise_supervision_v1.md), [`../implementation/202605171322_oracle_search_output_contract.md`](../implementation/202605171322_oracle_search_output_contract.md), [`../implementation/202605171430_pointwise_oracle_pipeline.md`](../implementation/202605171430_pointwise_oracle_pipeline.md), [`../implementation/202605180045_pointwise_v1b_true_side_anchor.md`](../implementation/202605180045_pointwise_v1b_true_side_anchor.md) | 从 oracle 上界计算转向 oracle-set supervision，完成 pointwise V1 selection-only probe、候选池输出契约与 pipeline 接入，并设计 V1b true-side anchor。 |
| Verifier 校准与复现实验基础设施 | [`../plan/202605180118_oracle_calibration_reoracle_four_stage_plan.md`](../plan/202605180118_oracle_calibration_reoracle_four_stage_plan.md), [`../implementation/202605180118_label_token_ce_verifier_stage1.md`](../implementation/202605180118_label_token_ce_verifier_stage1.md), [`../implementation/202605181113_calibration_aware_reoracle_stage2.md`](../implementation/202605181113_calibration_aware_reoracle_stage2.md), [`../plan/202605131653_dvc-warm-minsky.md`](../plan/202605131653_dvc-warm-minsky.md), [`../implementation/202605171700_tailscale_container_ssh_vscode.md`](../implementation/202605171700_tailscale_container_ssh_vscode.md) | 修复 verifier false-side bias、用 margin objective 重跑 oracle，同时补齐 DVC/Tailscale 等跨机复现与访问能力。 |

## 2. 系统基础设施

在展开时间线前，先梳理贯穿所有实验的共享基础设施与符号约定，避免后文重复描述。

### 2.1 数据流

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

### 2.2 标签体系

所有实验均使用 LIAR-RAW 6 类标签（`src/fact_checking/data/constants.py`），映射为单字母 token 以避免多 token bias：

| ID | 标签名 | 字母 | 定义 |
|----|--------|------|------|
| 0 | pants-fire | A | completely false and implausible |
| 1 | false | B | false based on the available evidence |
| 2 | barely-true | C | mostly false, with only a small element of truth |
| 3 | half-true | D | partly true and partly false |
| 4 | mostly-true | E | mostly true, with minor missing context or caveats |
| 5 | true | F | accurate based on the available evidence |

3 分类折叠（仅用于 b4 分类器诊断实验）：`{A,B}→false, {C,D}→mixed, {E,F}→true`。

### 2.3 核心检索 Pipeline（Hybrid Scoring + MMR）

所有实验的 evidence selection 均遵循统一的 Hybrid Scoring → MMR Selection 范式：

**Hybrid Score 计算**（三路融合）：
1. **Dense（α=0.70）**：内积相似度。BGE-base-en-v1.5 将 claim（query，带指令前缀 `"Represent this sentence for searching relevant passages: "`）和 chunk 文本分别编码为 768 维向量，已 L2 归一化，等价于余弦相似度。
2. **Lexical（α=0.20）**：基于内容词的 F1 分数。去除停用词后，计算 claim 与 chunk 间 token overlap 的 precision/recall/F1。
3. **BM25（α=0.10）**：简化版 BM25（`k1=1.2, b=0.75, avgdl=18.0`），在 query 的每个词上计算 `idf × tf/(tf + k1×(1-b + b×dl/avgdl))`。

三路分数各自 Min-Max 归一化到 [0,1] 后加权求和。

**MMR 选择算法**（`src/fact_checking/retrieval/mmr.py::maximal_marginal_relevance`）：
1. 首轮选 hybrid_score 最高的 chunk。
2. 迭代：$\text{mmr\_score}[i] = \lambda \cdot \text{hybrid\_score}[i] - (1-\lambda) \cdot \max\_\text{sim\_to\_selected}[i]$，每轮选出 mmr_score 最大的未选 chunk。
3. $\lambda \in [0, 1]$：$\lambda=1.0$ 等价于按 hybrid_score 降序取 top-K，$\lambda=0.0$ 纯粹按最大化与已选集合的差异来选。

### 2.4 训练与推理基础设施

- **基座模型**：Qwen2.5-7B-Instruct
- **微调方式**：LoRA（`r=16, α=32, dropout=0.05`），目标模块 `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **训练框架**：accelerate + DeepSpeed ZeRO-2，4 GPU，bf16，FlashAttention-2
- **有效 batch size**：32（per_device=4 × grad_accum=2 × 4 GPU）
- **学习率**：1e-5，cosine scheduler，warmup_ratio=0.03，max_grad_norm=1.0
- **推理引擎**：vLLM 0.8.5.post1，tensor_parallel_size=4，gpu_memory_utilization=0.90
- **Prompt 模板**：Qwen2.5 ChatML 格式。system prompt 固定为 `"You are a careful fact-checking assistant..."`，user content 包含 `Labels: A-F 定义 + Claim + Evidence [1]...[K]"`，target 为 `"Label: <letter>"`，自动截断至 max_length
- **解码策略**：temperature=0，max_tokens=8，guided choice 限定 A-F

### 2.5 符号约定

| 符号 | 含义 |
|------|------|
| $c$ | 一条政治声明（claim），即待核查的文本 |
| $y^*$（或 $y_{\text{gold}}$） | $c$ 的真实 6 类 veracity 标签 |
| $\hat{y}$ | Verifier 对 $c$ 的预测标签 |
| $C$（或 $C_N(c)$） | claim $c$ 的候选证据池（candidate pool），包含 $N$ 个候选 chunk |
| $d$（或 $d_i$） | 候选池中的单个 evidence chunk |
| $K$ | 需选择的证据数量（即 `top_k`） |
| $S_t$ | 经过 $t$ 步选择后的已选证据集合，$S_0 = \emptyset$ |
| $S_K$ | 最终选出的 $K$ 个证据组成的集合 |
| $\lambda$ | MMR 的多样性惩罚系数，$\lambda \in [0, 1]$。$\lambda=1$ 等价于纯相关性排序，$\lambda=0$ 等价于纯多样性 |
| $\lambda_t$ | 第 $t$ 步选择时使用的 $\lambda$ 值。若所有步共用同一个 $\lambda$，则为 scalar λ；若每步独立，则构成 step-wise λ schedule |
| $\Lambda$ | $\lambda$ 的离散候选值集合，如 `{0.1, 0.3, 0.5, 0.7, 0.9}` |
| $\tau$ | 一条 trajectory：$\tau = ((\lambda_1, d_1), \ldots, (\lambda_K, d_K))$，记录了每一步的 λ 选择与选出的证据 |
| $s_t$ | 第 $t$ 步时的状态，包含 claim $c$、候选池 $C$、已选集合 $S_{t-1}$、步号 $t$、以及相关的分数与相似度统计量 |
| $\pi_\theta$ | 参数为 $\theta$ 的 policy（策略函数），输入状态 $s_t$，输出 $\lambda_t$ 的分布 |
| $\pi_{\text{ref}}$ | DPO 训练中的 reference policy（通常是 fixed λ=0.7 或已有最优策略） |
| $U$（或 $U_i$） | Utility 函数：给定 claim $c$ 和证据集 $S_K$，量化该选择的下游效用 |
| $\operatorname{Rel}(c, d)$ | claim $c$ 与候选 $d$ 的相关性分数（hybrid score） |
| $\operatorname{Red}(d, S)$ | 候选 $d$ 与已选集合 $S$ 的冗余度：$\operatorname{Red}(d, S) = \max_{s \in S} \operatorname{Sim}(d, s)$ |
| $\operatorname{Sim}(d_i, d_j)$ | 两个候选 chunk 的余弦相似度（基于 BGE embedding） |

下文所有公式均沿用上述符号，不再重复定义。

## 3. 时间线总进度

### 2026-05-08：固定 chunking 作为检索单元

起点是 [`../plan/202605080025_chunking.md`](../plan/202605080025_chunking.md)：把 report 切分方式固定为后续统一的检索单元。实现位于 `src/fact_checking/build/chunking.py`，定义了五类策略（均继承 `ChunkingStrategy` 抽象基类）：

| 策略 | 类 | 行为 |
|------|-----|------|
| `sentence` | `SentenceChunking` | 返回 `sent_idx` 对应的单句。`chunks_from_presplit` 返回每个非空句子作为独立 `ChunkRecord(text=text, sent_indices=(idx,))` |
| `raw` | `RawChunking` | 不切分，返回整个 report 全文 |
| `ctx_window` | `ContextWindowChunking(k)` | 以 `sent_idx` 为中心取上下各 k 句，窗口 2k+1 句，边界处取至实际可用句子 |
| `semantic` | `SemanticChunking(θ)` | 按相邻句子对的余弦相似度合并：相似度 > θ 则合并为连续 chunk，< θ 则切分。需额外传入 BGE embeddings |
| `ctx_semantic` | `CtxSemanticChunking(k, θ)` | 先将句子按窗口 k 分组，再按窗口平均向量的余弦相似度执行 semantic 合并 |

配置入口 `configs/build/default.yaml`：`build.retrieval.chunking.strategy=sentence`，`context_k=1`，`theta=0.7`。semantic chunking 推荐使用 CPU 设备（避免 GPU OOM 于 embedding 批处理）。

之后的 MMR、learned-λ、oracle search 都围绕 chunk-level candidate pool 展开。Semantic chunking（θ=0.5）下 val 候选池 $N$ 中位数约 16，远小于 sentence chunking 的 ~51，显著减少了搜索空间但也可能丢失细粒度信息。

### 2026-05-11：分类器塌陷暴露证据质量瓶颈

[`202605111212_classifier-collapse-analysis.md`](202605111212_classifier-collapse-analysis.md) 记录了 b4 判别式分类器实验的系统性塌陷。

**实验配置**：ModernBERT-large（~400M 参数）作为基座，LIAR-RAW 训练集约 10k 条。使用 top_k=16 evidence chunks（按 fixed-MMR 检索），拼接后送入分类头做 6 类判别。

**实验矩阵与结果**：

| 实验 | 损失函数 | 类别数 | Epochs | Val Macro-F1 | 塌陷表现 |
|---|---|---|---|---|---|
| 原始 b4 (CE) | `CrossEntropyLoss` | 6 | 3 | 0.152 | 61.9% 预测 → `half-true` (788/1274)；pants-fire 仅被预测 1 次 |
| b4 + CORAL | CORAL ordinal regression | 6 | 6 | 0.117 | 64.8% → `half-true`；仅 3/6 类被预测 |
| b4 + 3class | `CrossEntropyLoss` | 3 | 3 | 0.391 (test: 0.402) | `false` 召回 17.3%，`mixed` 召回 57.5% |
| b4 + no_evidence | `CrossEntropyLoss` | 6 | 3 | — | 去掉 evidence，仅用 claim 做分类 |

随机基线对照：6 类 random accuracy≈16.7%, Macro-F1≈0.167；3 类 random accuracy≈33.3%, Macro-F1≈0.33。原始 b4 的 CE 结果仅略高于随机。

**CORAL 有序回归细节**：CORAL（consistent rank logits）将 6 类有序分类分解为 5 个二元分类器（$P(y > \text{class}_k)$），共享 backbone 并在每个二元头上用 BCE。结果比 CE 更差（Macro-F1 0.117），原因是多数类（half-true）的梯度信号淹没了少数类。

**根因诊断**：结论是仅靠分类头或损失函数难以突破，主要瓶颈转向 evidence selection 质量。具体原因：(1) MMR 检索返回的 16 条证据判别力不足；(2) 多条证据可能相互矛盾，输入信号混乱；(3) 2048 tokens 拼接 16 条证据，模型难以聚焦；(4) LIAR 数据集中许多 claim 是非事实性陈述（观点、承诺等），难以仅凭检索证据判断。

同日，[`../implementation/202605111255_mmr-lambda-sweep-pipeline.md`](../implementation/202605111255_mmr-lambda-sweep-pipeline.md) 梳理了 MMR λ sweep 的 build → train → infer 逻辑，形成后续实验的统一流水线基础。关键设计：

- **两阶段 GPU 缓存**：Pre-MMR cache（BGE 句子级嵌入，排除 `mmr_lambda` 指纹，11 个 λ 值共享）→ Chunk-MMR cache（chunk 级嵌入，排除 `top_k` 和 `mmr_lambda` 指纹）。只有最终 MMR + Prompt 构建（CPU-only）因 λ 不同而独立执行。
- **Hybrid Scoring 权重**：α_dense=0.70, α_lexical=0.20, α_bm25=0.10
- **λ 扫描范围**：0.0, 0.1, ..., 1.0（11 个值）
- **Prompt 自动截断**：若 prompt+target 超 max_length，从尾部（得分最低的 evidence）逐个移除
- **Label 输出格式**：`label_only` + `letter`，target=`"Label: A"` 等

[`202605112128_val_test_Mismatch.md`](202605112128_val_test_Mismatch.md) 进一步诊断 `mmr_lambda_sweep_1024`（max_length=1024, top_k=16, λ=0.0/0.1/0.2）：

- **val 先升后降**：三组 λ 均在 epoch 1 末（step-300）附近达峰，第 2 个 epoch 全面回落 ~5pp，是经典过拟合（10k 训练样本 + LoRA r=16）。
- **val/test 严重不一致**：best val F1≈0.27，但 API test F1≈0.15，parse_error_rate≈22.6%。根因在于训练时 online val 走的是 `model.merge_adapter() → vLLM LLM.generate()` 路径，而推理 test 走的是 `vLLM OpenAI server + --enable-lora --lora-modules` 路径——两条路径的 LoRA 加载方式不同，导致 API 推理下输出格式漂移（模型生成 `"A"` 而非 `"Label: A"`）。

### 2026-05-12 至 2026-05-13：chunking 可视化、learned-λ 数据流和 val/test gap 收敛

[`../visualization/202605121618_chunking_evidence_examples.md`](../visualization/202605121618_chunking_evidence_examples.md) 用真实样本展示不同 chunking 策略的证据跨度，帮助确认 semantic chunking 不是抽象配置，而会显著改变候选池形态。

[`../learned_lambda/202605122054_generate_oracle_prompts.md`](../learned_lambda/202605122054_generate_oracle_prompts.md) 固化 learned-λ 第一步：按 λ 网格（0.00, 0.05, ..., 1.00，共 21 个值）从 chunk-MMR cache 生成 per-lambda prompt JSONL，并保留 prompt 截断、token 统计与自动缓存解析。

[`202605131520_val-test-f1-gap-diagnosis.md`](202605131520_val-test-f1-gap-diagnosis.md) 把 val/test gap 结论修正为：代码路径差异贡献很小——`mmr_lambda_sweep` 的 API-val 与训练 val 差距约 0.36 pp，`b3_mmr_topk_sweep_1024` 的 3-5 pp gap 主要是真实泛化差距。22.6% parse error rate 的根因确认为 LoRA 在 vLLM 动态加载模式下 `modules_to_save` 丢失。诊断方案：(1) 用现有 best checkpoint 跑 API-val 复现，(2) 推理前 `merge_and_unload()` 出临时 dense 模型。

同日 [`../plan/202605131653_dvc-warm-minsky.md`](../plan/202605131653_dvc-warm-minsky.md) 规划 DVC：不接管现有 Hydra/fingerprint pipeline，只按需同步 raw data、build/pre-MMR cache 与重型推理产物，保证跨机器复现实验。

### 2026-05-14：验证 adaptive λ 有上界，但 hard λ predictor 失败

**Oracle λ 定义**：在 21 个 λ 值 (0.00, 0.05, ..., 1.00) 的网格上，SFT verifier 对正确 label token 给出最高 log-probability 的那个 λ 值。计算脚本为 `scripts/learned_lambda/compute_oracle_lambda.py`，通过 vLLM 离线推理对每个 λ 值下的 build prompt 评分 `Label: <A-F>` 的 logprob。

[`../learned_lambda/202605141045_verification_experiment.md`](../learned_lambda/202605141045_verification_experiment.md) 先验证 oracle λ 的价值：在 2013 个 predictor hold-out 样本上，oracle λ 相比 fixed λ=0.70 将 accuracy 从 30.40% 提到 33.48%（**+3.08 pp**），macro-F1 从 30.60% 提到 33.84%（**+3.24 pp**）。所有 6 个类别均有正向提升，改善最大的三个类别是 `true`（+5.15%）、`pants-fire`（+4.85%）、`false`（+3.35%）。48.4% 的样本在 oracle λ 和固定 λ 下产生了不同的 evidence 选择。推理使用 b3 的 LoRA checkpoint + vLLM guided_choice（限定输出 A-F letter tokens）。

这个结果说明 adaptive λ 不是无意义方向。

但 [`../learned_lambda/202605141052_analysis.md`](../learned_lambda/202605141052_analysis.md) 给出反向结论：hard oracle λ 作为监督目标非常弱。

**Predictor 架构**：`ChunkEmbeddingLambdaEncoder`（`src/fact_checking/learned_lambda/predictor.py`），输入为 BGE 768 维 chunk embeddings 经 hybrid scoring 排序后的 top-k，通过 256 维 attention 池化后做回归。另有 73 维手工特征版本（`src/fact_checking/learned_lambda/features.py`）和分类版本。

**三个变体全失败**：

| 变体 | Val MAE | Val RMSE | Target Std |
|---|---|---|---|
| Chunk embedding (256-dim attention, regression) | 0.256 | 0.294 | 0.296 |
| Handcrafted 73 features (regression) | 0.250 | 0.283 | 0.299 |
| Handcrafted 73 features (classification) | 0.250 | 0.282 | 0.299 |

对比基线：均值预测 $\lambda=0.445$ 的 MAE=0.262。$R^2$ vs 均值基线 ≈ 0.01（最高仅 0.058），预测 std=0.057 vs oracle std=0.296（方差比仅 19%）。

**五重根因**：
1. Oracle λ 信号极弱：72.6% 样本的 margin（最优与次优 λ 的 logprob 差异）< 0.05，38.5% < 0.01，中位数 margin 仅 0.0185。每个 claim 平均有 7.9/21 个 λ 值在最优值 0.1 logprob 范围内——oracle λ 本质上是平坦 logprob 曲面上的噪声点。
2. Tie-break 偏差：`compute_oracle_lambda.py:269` 中当多个 λ 在 0.01 logprob 范围内时，优先选离 `DEFAULT_LAMBDA=0.7` 最近的，进一步向 0.7 收缩。
3. Oracle λ 与 claim 语义无关：6 个 label 的 oracle λ 均值全在 0.42-0.50。
4. 特征-目标语义鸿沟：BGE embedding 编码文本语义，oracle λ 反映 SFT 模型对不同 evidence ordering 的内部响应——两者之间不存在系统性映射。
5. 唯一可观察到的系统模式：$\text{corr}(\log(n_{\text{candidates}}), \lambda_{\text{oracle}}) = -0.13$（候选越多 → 越需 diversity），但不足以支撑有效预测。

**两个修复实验也全失败**：
- **High-margin 过滤**（仅保留 margin ≥ 0.05 的 2761 样本）：预测值坍缩至 0.30-0.32 极窄区间，$R^2$ vs 均值 = -0.167。
- **3-bin 粗粒度分类**（diversity/balanced/relevance）：准确率 0.336（随机基线 0.333），全部 10065 样本被预测为同一类别（"balanced"）。

三个实验的一致失败表明问题不在数据质量、问题粒度或模型架构，而在于 BGE 文本 embedding 根本不编码"什么 λ 对 MMR 最优"这个信息。

[`../plan/202605141045_Improving_Learned_Lambda.md`](../plan/202605141045_Improving_Learned_Lambda.md) 因此提出 high-margin 与 3-bin 两个修复实验；后续分析证明它们不能改变结论。

[`../implementation/202605141531_sensitivity-gated-mmr.md`](../implementation/202605141531_sensitivity-gated-mmr.md) 实现 sensitivity-gated MMR。核心思想：不预测 oracle λ，而是判断当前 claim 是否对 λ 敏感，用简单的二值门控决定使用 $\lambda_{\text{low}}$ 还是 $\lambda_{\text{base}}$。

**两阶段设计**：
- **Stage A（离线搜索）**：不重新训练模型，逐 λ（`0.2, 0.3, ..., 0.8`）做 build → infer，收集预测结果矩阵。然后在 $(\theta_s, \theta_r, \lambda_{\text{low}}, \text{gating\_mode}, \varepsilon)$ 的网格上穷举搜索，查表计算 accuracy/macro_F1，按 $w_{\text{acc}} \times \text{accuracy} + w_{\text{f1}} \times \text{macro\_f1}$ 排序。
- **Stage B（在线流水线）**：读取 Stage A 最优参数，在 Build 阶段对每个 claim 实时计算门控 λ。

**门控决策树**（`src/fact_checking/rl_mmr/sensitivity.py`）：
1. 计算敏感度特征：$\text{sens} = 1 - \operatorname{Jaccard}(S_{\text{low}}, S_{\text{base}})$，三个 λ（probe, low, base）分别执行 MMR 后比较 overlap。
2. 计算池冗余度：$\text{pool\_redundancy} = \frac{2}{N(N-1)}\sum_{1 \le i < j \le N} \operatorname{Sim}(d_i, d_j)$，即候选池 top-N 句子的平均 pairwise cosine similarity。
3. 决策：若 $n < \text{min\_n\_candidates}$ → $\lambda_{\text{base}}$；若 $\text{sens} \ge \theta_s \land \text{pool\_redundancy} \ge \theta_r$ → $\lambda_{\text{low}}$；否则 → $\lambda_{\text{base}}$。Conservative 模式额外检查 relevance floor（低 λ 下平均相关性下降不超过 $\varepsilon$）。

搜索空间：$\theta_s$ ∈ {0.2,0.4,0.6,0.8}, $\theta_r$ ∈ {0.3,0.4,0.5,0.6}, $\lambda_{\text{low}}$ ∈ {0.2,0.3,0.4}, `gating_mode` ∈ {basic, conservative}。

这条路线保留为弱自适应 baseline。

[`../plan/202605141828_soft_label_policy.md`](../plan/202605141828_soft_label_policy.md) 设计 soft-label λ policy。核心思想：不再用 hard argmax λ 作为监督目标，而是对每条 claim 的完整 utility curve $U_i(\lambda)$ 做 softmax（with temperature $\tau$），得到概率分布 $q_i(\lambda)$ 作为 soft target：

$$q_i(\lambda_j) = \frac{\exp(U_i(\lambda_j) / \tau)}{\sum_k \exp(U_i(\lambda_k) / \tau)}$$

低 margin 样本自然产生平坦分布，自动降低对训练的影响。样本权重 $w_i = \max_\lambda U_i(\lambda) - U_i(0.7)$，使模型将容量集中在对 λ 敏感的 claim 上。损失函数为加权 KL 散度：$\mathcal{L} = \sum_i w_i \cdot \operatorname{KL}(q_i(\lambda) \,\|\, p_\theta(\lambda \mid x_i))$。特征加入 interventional features（如 $\operatorname{Jaccard}(S_{0.3}, S_{0.7})$、不同 λ 下的 meanRel/meanRed 差异等），不再只用 BGE embedding。

**推理策略**：三种——`argmax`（取最大概率 λ）、`expected`（按概率加权 λ 期望值）、`sample`（从概率分布采样）。

同日 [`202605141045_RL_MMR_research_review.md`](202605141045_RL_MMR_research_review.md) 完成研究综述，明确从 fixed-MMR、learned-λ、DPO/GRPO 到 reranker + adaptive diversity policy 的方法地图。

### 2026-05-15 至 2026-05-16：scalar λ 路线阶段性关闭

[`202605151453_RL_MMR_direction_summary.md`](202605151453_RL_MMR_direction_summary.md) 将研究目标从"预测最优 scalar λ"调整为"学习 evidence diversity policy"。推荐探索顺序：`fixed 0.7 → log(n_candidates) → sensitivity-gated → soft-label → DPO step-wise → multi-weight → GRPO`，并强调 fixed λ=0.7 是强 baseline。定义了 5 个关键决策门槛（Gate 1-5）和分桶分析框架（按候选数、敏感度、冗余度、oracle margin、label 类型分桶）。

[`../plan/202605151936_dpo_step_wise_lambda.md`](../plan/202605151936_dpo_step_wise_lambda.md) 和 [`../plan/202605152032_dpo_stepwise_lambda.md`](../plan/202605152032_dpo_stepwise_lambda.md) 将 DPO step-wise λ 方案具体化。

**DPO 实现细节**（最终实现于 2026-05-15~16）：
- **Trajectory 定义**：每步 $\lambda_t \sim \pi_\theta(\lambda \mid s_t)$，$d_t = \operatorname{MMR\_select}(\lambda_t, c, C, S_{t-1})$，最终形成 $\tau = ((\lambda_1, d_1), \ldots, (\lambda_K, d_K))$。
- **动作空间**：5 个离散 λ 值 `{0.1, 0.3, 0.5, 0.7, 0.9}`。
- **Trajectory 生成**：7 个手工 schedule（`[0.7,0.7,0.7,0.7,0.7]`、`[0.3×5]`、`[0.5×5]`、`[0.9,0.7,0.5,0.3,0.3]`、`[1.0,0.7,0.5,0.3,0.1]`、`[0.5,0.5,0.7,0.7,0.9]`、`[0.7,0.5,0.3,0.5,0.7]`）+ 30 个随机 schedule = 每 claim 37 条 trajectory，train 集约 372,000 条。
- **Policy 模型**：`StepLambdaPolicy`（MLP，`hidden_dims=[64, 32]`），输入为 20 维 state features（8 维 pool 特征 + 12 维 step 特征）。后续版本 V3 精简为 13 维（仅 step + `prev_lambda`）。
- **DPO Loss**：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\!\Big(\beta \cdot \big[(\log \pi_\theta(\tau^+) - \log \pi_\theta(\tau^-)) - (\log \pi_{\text{ref}}(\tau^+) - \log \pi_{\text{ref}}(\tau^-))\big]\Big)$$

其中 $\tau^+ \succ \tau^-$ 为 preference pair（$U(\tau^+) - U(\tau^-) \ge \delta$），reference policy 固定偏向 $\lambda=0.7$。
- **Preference pair 构造**：train 78,510 pairs（来自 7,851/10,065 claim），val 11,030 pairs。63.2% 的 claim 存在优于 fixed λ=0.7 的 λ schedule。Utility gap 均值=4.54, std=3.53。最优 λ 分布：0.3 (20.9%), 0.5 (39.9%), 0.7 (39.1%)。

**四次 DPO 训练全部失败**：

| 版本 | 特征维度 | β | 关键差异 | 结果 |
|---|---|---|---|---|
| V1 | 20-dim (pool+step) | 1.0 | 原始实现 | λ=0.7: 99.97%, entropy=1.32 |
| V2 | 20-dim | 3.0 | 过滤 sentinel rows | λ=0.7: 99.87%, entropy=1.57 |
| V3 | 13-dim (step+prev_lambda) | 3.0 | 去掉无信息的 pool features | λ=0.7: 99.87%, entropy=1.56 |
| V4 | 13-dim, K=1 (claim-level) | 3.0 | 只用 step-0，等价于预测 majority λ | λ=0.7: 100% |

**三层失败根因**：
1. **数据质量**：2.98% 的 trajectory utility 为 `-100` sentinel 值，产生虚假的大 gap。过滤仅部分改善。
2. **特征问题**：pool features（8 维）对同一 claim 的 winner/loser 完全相同（纯噪声）；step features 中 12 维仅有 `top_mmr_score` 和 `mmr_score_gap` 有 winner/loser 差异，且这些差异是 λ 选择的**结果**而非**原因**（内生性）；Step 0 时所有 trajectory 状态完全一样，无法区分。
3. **信号本质**：reference policy 已与多数 winner 一致，有效 DPO 梯度太少；utility gap 中位数仅 2.34，信噪比太低。

[`../plan/202605161049_RL_MMR_experiment_plan_v2.md`](../plan/202605161049_RL_MMR_experiment_plan_v2.md) 汇总了前五个方向的阶段结论：

| 方法 | test accuracy | test macro-F1 | Δ vs fixed | 当前状态 |
|---|---|---|---|---|
| fixed λ=0.7 (b3, top_k=5) | 0.2702 | 0.2769 | — | locked baseline |
| `log(n_candidates)` heuristic | 0.2766 | 0.2799 | +0.0064 | 保留弱 baseline |
| sensitivity-gated MMR | 0.2742 | 0.2795 | +0.0040 | 保留弱 baseline |
| soft-label λ | — | — | expected≈fixed, argmax/sample 更差 | 停止 |
| DPO step-wise λ | — | — | 4 轮训练坍缩到 λ=0.7 | 停止 |

此时 scalar λ 的 claim-level 与 step-wise 路线均已触发停止条件，GRPO 因前置 DPO 无收益也不跑。唯一尚有意义的 λ 框架内扩展是 multi-weight MMR，但随后 oracle set gap 结果把主线进一步推向 set-level supervision。

### 2026-05-16：Oracle evidence set 显示更大上界 gap

[`../plan/202605161147_oracle_evidence_selection.md`](../plan/202605161147_oracle_evidence_selection.md) 规划直接搜索最优 K-subset，而不是只搜索 λ。

**核心区别**：Oracle λ 仍受 MMR greedy selection 约束——最优 evidence set 可能无法被任何单一 λ 的 MMR 选中；MMR 的 objective（$\lambda \cdot \operatorname{Rel} - (1-\lambda) \cdot \operatorname{Red}$）只是 set utility 的粗糙代理。Oracle evidence set 直接回答"哪个 K-子集最好"，跳过了 λ 和 MMR 两个间接层。

**搜索算法**（`scripts/oracle_evidence/search_optimal_evidence.py`）：
- $N \le 15$ 时使用穷举搜索（$\binom{15}{5}=3003$）。
- $N > 15$ 时使用贪婪前向选择：从空集开始，每步枚举剩余候选，用 vLLM 评分 $(c, S_{t-1} \cup \{d\})$ 的正确 label logprob，选最大者加入。复杂度 $O(NK)$ 次 verifier 调用（每样本平均约 235 次，$N \approx 50$）。

评分使用 vLLM 离线推理（`prompt_logprobs=0`），与 `compute_oracle_lambda.py` 的实现模式一致。

[`202605161449_oracle_set_gap_analysis.md`](202605161449_oracle_set_gap_analysis.md) 给出关键实验结果（val split, semantic chunking θ=0.5, top_k=5, b3 LoRA checkpoint）：

**总览**：

| 指标 | Oracle (greedy) | MMR (λ=0.7) | Gap |
|---|---|---|---|
| Accuracy | **48.43%** | 29.67% | **+18.76 pp** |
| Macro F1 | **43.03%** | 30.03% | **+13.00 pp** |

这个 gap 比 oracle λ 的约 +3 pp 大得多（~6x），说明瓶颈已经不只是 λ，而是 MMR 单轴表达能力和 evidence set selection 本身。

**事件级 Gap 分解**（1274 个 val 样本）：

| 类别 | 样本数 | 占比 | 含义 |
|---|---|---|---|
| Both correct | 222 | 17.4% | 当前 MMR 已足够 |
| Oracle only correct | 395 | **31.0%** | evidence selection 可直接修复 |
| MMR only correct | 156 | 12.2% | Oracle 目标函数缺陷（logprob ≠ accuracy） |
| Neither correct | 501 | 39.3% | Verifier 硬瓶颈，selector 无法解决 |

**Per-label 分析揭示严重 false-side bias**：

| Class | Oracle Acc | MMR Acc | Gap |
|---|---|---|---|
| pants-fire | **88.7%** | 40.0% | +48.7 pp |
| false | **89.6%** | 27.0% | +62.5 pp |
| barely-true | 40.7% | 34.7% | +5.9 pp |
| half-true | 53.3% | 28.7% | +24.6 pp |
| mostly-true | 21.9% | 27.1% | **−5.2 pp** |
| true | 1.2% | 24.9% | **−23.7 pp** |

pants-fire 和 false 的 oracle 召回率接近 90%（候选池中存在强力反驳证据），但 mostly-true 和 true 的 oracle 准确率**低于 MMR**——verifier 有强烈的 false bias，即使选出了最优证据集，verifier 也不相信 claim 是真的。pants-fire 和 false 的 median logprob 几乎为 0，而 true 的 median logprob 为 −7.6（正确标签概率极低）。

**核心矛盾**：Oracle 贪婪地最大化正确标签 logprob，但 true 类标签的 logprob 即使被最大化后仍远低于 false 类标签，argmax 仍选错。所以证据选择有空间，但旧 verifier 的 calibration 必须并行修复。

[`202605161516_oracle_set_supervision_next_steps.md`](202605161516_oracle_set_supervision_next_steps.md) 和 [`../plan/202605170118_oracle_set_supervision_next_steps.md`](../plan/202605170118_oracle_set_supervision_next_steps.md) 将下一阶段目标改成 Oracle evidence set supervision：先做 pointwise/sequential selector 与 preference supervision，再考虑 multi-weight MMR 和 GRPO；但不能直接把旧 oracle set 当无条件硬标签蒸馏。

**为什么先选 Pointwise selector V1？** 在 Oracle evidence set gap 分析揭示 +18.76 pp accuracy / +13.00 pp macro-F1 的巨大上界后，Scalar λ 路线已被充分探索并阶段性停止（oracle λ 仅 +3 pp 且五种 scalar λ 方法全部失败或退化）。接下来的核心问题是：**Oracle evidence set 的监督信号是否可被模型吸收？** 如果连最简单的 pointwise selector 都无法从 oracle set 中学会有用的选择模式，那么更复杂的 sequential selector、DPO、GRPO 就更不可能。因此需要一个最小可行实验来回答这个前提问题。选择 Pointwise 作为第一步的理由：

1. **最简模型**：NumPy logistic regression，无 GPU 训练依赖，快速迭代（800 epochs, 纯 CPU, ~1 min）。
2. **最小假设**：不建模 evidence 间的交互（coverage/redundancy/order），只学"每个候选单独看是否值得选"——如果这个都学不会，说明特征或监督信号有问题。
3. **Selection-only probe**：不跑完整的 build→train→infer（成本高），先用 Recall@5 / Jaccard@5 衡量"模型选出的 evidence 与 oracle evidence 的重叠度"，作为快速 gate。
4. **可分析性强**：logistic regression 的 feature importance 可直接解释哪些特征驱动了选择，便于诊断。

### 2026-05-17：Pointwise selector V1 — selection-only gate 通过，但下游 verifier 评估反而低于 fixed-MMR

> 2026-05-19 复盘修正：这里的 “selection-only gate 通过” 已判定为无效强信号，只能作为历史弱参考。V1 gate 使用了与正式 b3 semantic pipeline 不一致的 Chunk-MMR cache，并且候选池构造会先注入 oracle positives 再补 negatives；它没有评估正式 pipeline 的 `dedup -> hybrid top candidate_pool_size -> selector topK` 候选空间。

**Selection-only 评估指标说明**：Recall@5 和 Jaccard@5 是衡量 selector 选出的 top-5 evidence set（$S_{\text{pred}}$）与 oracle evidence set（$S_{\text{oracle}}$）之间重叠度的指标，定义如下：

$$\text{Recall@}K = \frac{|S_{\text{pred}} \cap S_{\text{oracle}}|}{|S_{\text{oracle}}|}$$

$$\text{Jaccard@}K = \frac{|S_{\text{pred}} \cap S_{\text{oracle}}|}{|S_{\text{pred}} \cup S_{\text{oracle}}|} = \frac{|S_{\text{pred}} \cap S_{\text{oracle}}|}{2K - |S_{\text{pred}} \cap S_{\text{oracle}}|}$$

Recall@5 衡量"oracle 选的 K 个 evidence 中有多少被找回"（越高越好，上限 1.0）。Jaccard@5 同时惩罚多选和漏选（两个集合越一致分越高，上限 1.0）。两者结合可判断 selector 是否学到了 oracle evidence 的排序模式。但这些指标只衡量与 oracle set 的集合重叠度，**不等于**下游 verifier accuracy / macro-F1 提升——oracle set 本身继承了旧 verifier 的 false bias，且 evidence order 对 SFT 模型的影响也会被忽略。

[`../plan/202605171203_oracle_pointwise_supervision_v1.md`](../plan/202605171203_oracle_pointwise_supervision_v1.md) 定义了 V1 轻量闭环：只在高置信 retained labels 上验证 oracle-selected evidence 是否可被吸收。

**V1a 过滤条件**：
- `oracle_correct == true`
- `gold_label` ∈ {pants-fire, false, barely-true, half-true}（排除 true-side，因为旧 oracle 继承了 false bias）
- `final_logprob >= -0.5`
- `n_candidates > 5`

[`../implementation/202605171203_oracle_pointwise_supervision_v1.md`](../implementation/202605171203_oracle_pointwise_supervision_v1.md) 实现并运行了 NumPy logistic regression selector。新增文件：`src/fact_checking/oracle_pointwise.py`（共享工具模块）、`scripts/selectors/build_pointwise_oracle_dataset.py`、`train_pointwise_oracle_selector.py`、`eval_pointwise_oracle_selector.py`。

**训练细节**：
- 模型：标准化特征上的 logistic regression，纯 NumPy 实现（无 sklearn/lightgbm 依赖）。
- 监督：candidate-level weighted BCE，权重含 label-level inverse frequency + claim-level positive/negative balance。
- 切分：按 `event_id` 分 train/dev（避免同一 claim 的 candidate 泄漏），按 label 做分层切分。
- 训练：800 epochs, lr=0.05, patience=80。
- 过滤后数据：3259 claims, 40470 candidate rows, positive rate≈40.24%。

**重要约束：当前候选池是重建的**。旧 oracle 输出只保存了 `selected_texts`/`selected_indices`/`search_steps`，没有完整 `candidate_pool`。本轮通过文本匹配在可用 Chunk-MMR cache 中匹配 positives（match rate 99.94%），用 hybrid score 最高的候选补 negatives。所以结果是 **reconstructed-pool selection-only probe**，不是严格复现。

**Selection-only 结果（与 oracle set 的 overlap）**：

| 评估集 | Scorer | Recall@5 | Jaccard@5 |
|---|---|---|---|
| dev | pointwise logreg | 0.8288 | 0.7402 |
| dev | hybrid_score baseline | 0.1926 | 0.1261 |
| val (retained) | pointwise logreg | 0.8382 | 0.7502 |
| val (retained) | hybrid_score | 0.1496 | 0.0948 |
| val (full) | pointwise logreg | 0.8495 | 0.7706 |
| val (full) | hybrid_score | 0.2622 | 0.2070 |

Candidate-level AUPRC=0.8971, AUROC=0.8908。Per-label Jaccard@5 在 four retained labels 上均衡（0.72-0.81）。当时记录为 selection-only gate 通过；2026-05-19 复盘后，该结论降级为无效强 gate，因为它来自 reconstructed / positive-injected pool，不代表正式 build pipeline 的候选池。

**但下游 vLLM verifier 评估显示指标反而低于 fixed-MMR baseline**。产物位于 `outputs/runs/b3_pointwise_oracle_selector_1024/`，共三个 run：

| run | 类型 | split | accuracy | macro_f1 |
|---|---|---|---|---|
| `pointwise_oracle_eval_val` | evaluation-only（复用旧 verifier ckpt `79d8b34809bb`） | val | 0.2582 | 0.2582 |
| `pointwise_oracle_full` | 完整 build→train→infer（**新训练 verifier**） | val (train best) | 0.2438 | 0.2346 |
| `pointwise_oracle_full` | 同上 | **test** | **0.2230** | **0.2059** |

对比 fixed λ=0.7 baseline（test accuracy=0.2702, macro_f1=0.2769）：**V1a 完整流水线在 test 集上显著劣于 fixed-MMR**（accuracy -4.72 pp, macro_f1 -7.10 pp）。`mostly-true` 的 per-class F1 仅 0.0685，说明排除 true-side 训练后 selector 对 true-side 证据排序信号完全缺失。

**Selection-only overlap 高但下游 verifier 差的原因**：
1. V1 gate 的 cache / chunking 与正式 b3 semantic pipeline 不一致，selection-only overlap 被 sentence-level 重建候选池抬高。
2. V1 gate 先把 oracle positives 注入候选池再评估 selector，隐藏了正式 pipeline 中 positives 可能不在 hybrid top candidate_pool_size 内的问题。
3. Oracle set 继承了旧 verifier 的 false-side bias——oracle-selected evidence 对 false-side labels 的区分力强但对 true-side 无效。
4. Pointwise independent selection 不考虑 evidence 间的交互（coverage/redundancy/order），选出的 top-5 可能信息高度重叠。
5. Evidence order 效应：pointwise 按 score 降序排列，而 MMR 按逐步选择排列——SFT 模型对 evidence 顺序敏感。

[`../implementation/202605171322_oracle_search_output_contract.md`](../implementation/202605171322_oracle_search_output_contract.md) 随后修正 oracle search 输出契约：保存完整 `candidate_pool`、`candidate_scores`、`candidate_pool_fingerprint`、两阶段剪枝元数据，并明确 `selected_indices` 是相对于 `candidate_pool` 的索引，解决后续 selector 训练的候选池可追溯问题。

[`../implementation/202605171430_pointwise_oracle_pipeline.md`](../implementation/202605171430_pointwise_oracle_pipeline.md) 把 pointwise selector 接入正式 build pipeline：新增 `selection_method=pointwise_oracle`，并支持 `pipeline.steps=[build,infer]` 用同一个 verifier checkpoint 做 evaluation-only。该阶段已完成 compile/config/sanity check。

同日 [`../implementation/202605171700_tailscale_container_ssh_vscode.md`](../implementation/202605171700_tailscale_container_ssh_vscode.md) 记录了公网服务器、容器服务器和本地 VS Code 的 Tailscale 连接方案。

### 2026-05-18：V1b true-side anchor 微弱改善，但仍低于 fixed-MMR；转向 verifier calibration + re-oracle

[`../implementation/202605180045_pointwise_v1b_true_side_anchor.md`](../implementation/202605180045_pointwise_v1b_true_side_anchor.md) 针对 V1a 的 mostly-true F1=0.0685 问题，加入 low-weight true / mostly-true anchor：

- `gold_label` ∈ {mostly-true, true} + `oracle_correct == true` + `n_candidates > 5`
- mostly-true supervision_weight=0.25，true supervision_weight=0.10（低权重锚点，只防止完全遗忘 true-side 模式，不主导训练）

**V1b vLLM Verifier 评估结果**（`outputs/runs/b3_pointwise_oracle_selector_v1b_1024/pointwise_oracle_v1b_eval_val__ec28d138/`，evaluation-only, val split）：

| 指标 | V1b (val) | V1a (val, eval-only) | fixed λ=0.7 (val) |
|---|---|---|---|
| accuracy | 0.2630 | 0.2582 | 0.2967 |
| macro_f1 | 0.2632 | 0.2582 | 0.3003 |

V1b 相比 V1a 有微弱提升（accuracy +0.0048, macro_f1 +0.0050），但**仍显著低于 fixed λ=0.7**（accuracy 差距约 3.4pp）。低权重 anchor 未能将 true-side 指标拉回 MMR 水平。

**V1a/V1b 的总体结论**：Pointwise oracle selector 的旧 selection-only overlap 指标不能作为强结论，原因是 cache/chunking 和 candidate-pool 构造口径与正式 pipeline 不一致。下游 verifier evaluation 给出了反向结论——accuracy 和 macro-F1 均低于 fixed-MMR baseline。后续 selector 监督必须基于 re-oracle 保存的完整 candidate pool，并共享同一个 chunk cache fingerprint。

[`../plan/202605180118_oracle_calibration_reoracle_four_stage_plan.md`](../plan/202605180118_oracle_calibration_reoracle_four_stage_plan.md) 把下一轮主线明确调整为四阶段：

1. **Stage 1 — Label-token Weighted CE Verifier**：把 verifier 训练从完整 target SFT loss 改成 `prompt + "Label:"` 后 A-F label token 的 weighted CE。用 `macro_f1 + 0.5 × true_side_macro_f1` 选 checkpoint。目标：修复 false-side bias。
2. **Stage 2 — Calibration-aware Re-Oracle**：用 Stage 1 verifier 重跑 oracle search，目标从 `gold_logprob` 扩展为 margin（$P(y_{\text{gold}} \mid c, S) - \max_{y \neq y_{\text{gold}}} P(y \mid c, S)$）。目标：避免旧 verifier false bias 被 oracle set 继承。
3. **Stage 3 — Filtered Preference / Utility Supervision**：用 re-oracle 的完整候选池重建 pointwise / preference supervision。
4. **Stage 4 — Selector Training + Full Pipeline Evaluation**：端到端对比 fixed-MMR vs calibrated selector。

核心判断是：当前 pointwise selector 不稳定超过 fixed-MMR 的主要风险，不一定在 selector 吸收能力，而在旧 oracle set 继承了旧 verifier 的 false-side bias。

[`../implementation/202605180118_label_token_ce_verifier_stage1.md`](../implementation/202605180118_label_token_ce_verifier_stage1.md) 实现 Stage 1。

**训练范式改变**：不再对完整 target 序列做 causal LM loss，而是取 `Label:` 之后下一 token 的 logits：

```text
input  = row["prompt"].rstrip() + "Label:"
target = correct single label token: " A" ... " F"
label_logits = logits_at_last_input_position[:, token_ids(" A"..." F")]
loss = WeightedCrossEntropy(label_logits, gold_label_id)
```

**类别权重**（加重 true-side 以对抗 false bias）：

```yaml
class_weights:
  "pants-fire": 1.0
  "false": 1.0
  "barely-true": 1.2
  "half-true": 1.2
  "mostly-true": 2.0
  "true": 3.0
```

**Checkpoint 选择**：$\text{selection\_score} = \text{macro\_f1} + 0.5 \times \text{true\_side\_macro\_f1}$，其中 $\text{true\_side\_macro\_f1} = \frac{1}{2}(\text{F1}_{\text{mostly-true}} + \text{F1}_{\text{true}})$。关闭 `logit_adjust`（避免 weighted CE 与 prior correction 双重校正）。

**产物兼容性**：模型仍是 `AutoModelForCausalLM`，LoRA 保存格式仍走 `sft.data.io.save_model`，因此 `train/best` 可直接给现有 vLLM infer pipeline 使用。

新增文件：`src/sft/label_token_dataset.py`、`src/sft/label_token_trainer.py`、`configs/experiment/b3_label_token_ce_1024.yaml`、`scripts/verifier/run_label_token_ce_stage1.sh`。Pipeline runner 支持 `train.kind=label_token_ce`。

**Stage 1 已完成。实验结果**（run: `label_token_ce_stage1__0ee9b55f`，2026-05-18 完成）：

训练：accelerate + DeepSpeed ZeRO-2 × 4 GPU，600 steps（~2 epochs），eval 每 50 steps。

Checkpoint 选择曲线（按 `selection_score = macro_f1 + 0.5 × true_side_macro_f1`）：

| step | macro_f1 | true_side_macro_f1 | selection_score |
|---|---|---|---|
| 200 | 0.2511 | 0.3431 | 0.4227 |
| 350 | 0.2800 | 0.3148 | 0.4374 |
| 450 | 0.2930 | 0.3224 | 0.4543 |
| **550** | **0.2961** | 0.3574 | **0.4748** ★ |
| 600 | 0.2949 | 0.3543 | 0.4721 |

★ = best，step-550 的 selection_score 最高（macro_f1 接近峰值且 true_side_macro_f1 最高）。

**Val vLLM infer 结果**（`infer/val/best/3773899c0e40/api/metrics.json`）：

| 指标 | Stage 1 (val) | 旧 verifier MMR baseline (val) |
|---|---|---|
| accuracy | **0.3006** | 0.2967 |
| macro_f1 | **0.3015** | 0.3003 |
| parse_error_rate | 0.0 | 0.0 |

Per-class F1：

| label | Stage 1 F1 |
|---|---|
| pants-fire | 0.3152 |
| false | 0.3065 |
| barely-true | 0.2707 |
| half-true | 0.2445 |
| mostly-true | **0.3419** |
| true | **0.3298** |

**关键发现**：
- 整体指标与旧 verifier 基本持平（accuracy +0.39pp, macro_f1 +0.12pp），差异在噪声范围内。
- **true-side 未出现退化**：mostly-true F1=0.3419、true F1=0.3298 是全部 6 类中最高的两个，说明 weighted CE（true=3.0, mostly-true=2.0）成功防止了旧 verifier 的 false-side bias。
- 但 `barely-true`（F1=0.2707）和 `half-true`（F1=0.2445）仍然偏低，中间类的判别力是整个 6-class verifier 的持续难点。
- 截至目前只跑了 val infer，test 尚未跑。

### 2026-05-19：Stage 2 Calibration-aware Re-Oracle 进行中

[`../implementation/202605181113_calibration_aware_reoracle_stage2.md`](../implementation/202605181113_calibration_aware_reoracle_stage2.md) 实现 Stage 2。

**核心改变：oracle objective 扩展为 calibration-aware margin**：

$$\text{margin} = \log P(y_{\text{gold}} \mid c, S_K) - \max_{y \neq y_{\text{gold}}} \log P(y \mid c, S_K)$$

如果 margin > 0，说明 verifier 不仅给正确标签高概率，而且正确标签超过所有错误标签——直接对应 argmax 正确。这比旧 `gold_logprob` 更严格，可天然过滤 false-side bias 导致的 oracle set 选择错误。

**代码改动**：
- `src/fact_checking/oracle_evidence/scorer.py`：新增 all-label logprob scoring（同时评分 A-F 六个 token）。
- `src/fact_checking/oracle_evidence/search.py`：新增 `objective=margin` 搜索目标，记录 `gold_logprob`、`best_wrong_logprob`、`margin`、`label_logprobs`。
- `scripts/oracle_evidence/search_optimal_evidence.py`：新增 `--objective {gold_logprob,margin}`、`--save-candidate-pool`、`--save-search-step-scores`。
- `scripts/oracle_evidence/run_reoracle_stage2.sh`：Stage 2 默认运行脚本。
- `scripts/oracle_evidence/merge_shards.py`：合并 sharded oracle JSONL 并重算合并指标。

**输出字段**（每条 `oracle_results_<split>.jsonl`）：

| 字段 | 含义 |
|---|---|
| `search_objective` | `gold_logprob` 或 `margin` |
| `final_objective` | 当前 objective 下最终 set 的分数 |
| `gold_logprob` | 最终 set 的正确标签 logprob |
| `best_wrong_logprob` | 最终 set 的最高错误标签 logprob |
| `margin` | `gold_logprob - best_wrong_logprob` |
| `label_logprobs` | A-F 六个 label token 的 logprob |
| `pred_label` | label-token argmax 预测标签 |
| `candidate_pool` | 完整候选池（JSON 数组） |
| `candidate_scores` | 每个候选的 hybrid/dense/lexical/bm25 分数 |
| `candidate_pool_fingerprint` | 候选池的 SHA1 指纹 |

**Sharding 与断点续跑**：shard 由 `sha1(event_id) % NUM_SHARDS` 稳定分配，支持并行 + 断点续跑（自动跳过已完成的 event_id，处理异常行截断）。建议先用 `MAX_SAMPLES=32` 或 `128` smoke test，再用 sharding 跑完整 train。

**成本注意**：`objective=margin` 需对每个候选 set 评分 A-F 六个 label token，vLLM scoring 成本约为旧 `gold_logprob` 的 6 倍。

**当前状态（2026-05-19）**：Stage 2 正在 train set 上跑 margin re-oracle。使用 Stage 1 checkpoint `label_token_ce_stage1__0ee9b55f/train/best` 作为 verifier，`objective=margin`，`search_method=greedy`，`top_k=5`。train set 约 10,065 条 claim，shard 并行，预计耗时较长（每样本约 235 次 vLLM 评分，每次评分 6 个 label token）。

## 4. 当前总体结论

1. 早期 classifier collapse 已经把主要矛盾从"分类头/损失函数"推到"证据选择质量"。单纯换 CE/CORAL/3-class 无法解决问题。
2. `fixed λ=0.7` 是当前强 baseline（test accuracy=0.2702, macro-F1=0.2769），不应被视为容易击败的弱方法。
3. Oracle λ 证明 adaptive λ 有约 3 pp 的理论收益（accuracy 30.40% → 33.48%），但 hard predictor（$R^2 \approx 0.01$）、high-margin 过滤（$R^2 = -0.17$）、3-bin 分类（全坍缩）、soft-label（utility curve 接近均匀）、DPO step-wise（4 轮全部坍缩到 λ=0.7）都没有形成可用策略。
4. Scalar λ 路线已经足够充分地探索并阶段性停止；`log(n)`（test +0.0064）和 sensitivity-gated（test +0.0040）只作为弱 adaptive baseline 保留。
5. Oracle evidence set 的上界 gap（accuracy +18.76 pp, macro-F1 +13.00 pp）远大于 oracle λ（~3 pp），说明后续更应学习 evidence set / selector / utility，而不是继续精确预测单一 λ。
6. Pointwise V1 旧 selection-only overlap 指标已判定为无效强 gate：它混用了非正式 cache，并在候选池中注入 oracle positives。下游 vLLM verifier evaluation 给出了反向结论：V1a test accuracy 0.2230、macro-F1 0.2059，均显著低于 fixed-MMR（0.2702/0.2769）；V1b 加入 true-side anchor 后 val 仅微升至 0.2630/0.2632，仍低于 fixed-MMR。
7. 当前最合理主线是先校准 verifier（label-token weighted CE + margin objective），再用 calibration-aware objective 重跑 oracle，最后基于可追溯 candidate pool 构造 filtered supervision。
8. **Stage 1 已完成**：Label-token weighted CE verifier 在 val 上整体指标与旧 verifier 持平（accuracy 0.3006 vs 0.2967），但 true-side 未退化（mostly-true F1=0.3419, true F1=0.3298，为全部 6 类最高），说明加权有效防止了 false-side bias。中间类（half-true/barely-true）仍是难点。
9. **Stage 2 正在进行**：用 Stage 1 verifier + margin objective 在 train set 上重跑 oracle search。

## 5. 当前 Stop / Go 状态

| 方向 | 状态 | 原因 |
|---|---|---|
| ModernBERT 判别式分类器 | 停止作为主线 | 指标接近随机/塌陷（val macro-F1≈0.15），证据质量瓶颈更明显。 |
| hard learned-λ predictor | 停止 | oracle λ margin 太小（72.6% < 0.05），$R^2\approx 0.01$，预测坍缩到均值。 |
| high-margin / 3-bin λ 修复 | 停止 | 过滤和粗分类仍未提供稳定可学信号。 |
| soft-label scalar λ | 停止 | utility curve 接近均匀（entropy≈1.6 vs uniform 1.61），expected 退化。 |
| DPO step-wise scalar λ | 停止 | 4 轮训练全部坍缩到 λ=0.7，信噪比太低。 |
| GRPO refinement | 暂不跑 | 前置 DPO 或 stable offline policy 不满足。 |
| `log(n)` / sensitivity-gated | 保留 | 作为弱 adaptive baseline（+0.004-0.006），不继续深挖。 |
| multi-weight MMR | 待定 | 仍可能解决 scalar λ 表达能力不足，但应在 re-oracle 后再决定。 |
| Pointwise oracle selector (V1a) | 停止 | 旧 selection-only gate 无效，只能作弱参考；下游 test accuracy 0.2230 显著低于 fixed 0.2702。 |
| Pointwise oracle selector (V1b) | 停止 | True-side anchor 仅微升（val +0.005），仍低于 fixed-MMR；旧 gate 口径不能支撑继续放大。 |
| Oracle-set supervision | 继续 | 上界 gap 大（+13 pp macro-F1），但需要先修 verifier false bias + 换用 preference/sequential 范式。 |
| Label-token CE verifier + margin re-oracle | 当前主线 | Stage 1 完成（val 持平旧 verifier，true-side 未退化）；Stage 2 train set re-oracle 进行中。 |

## 6. 建议下一步

1. ~~先完成 Stage 1 label-token CE verifier 的 val/test 指标确认~~ → **已完成**。Val 上 accuracy 0.3006, macro_f1 0.3015，true-side 未退化。待补 test infer。
2. **Stage 2 进行中**：用 Stage 1 verifier + margin objective 在 train set 上跑 re-oracle shard。完成后先 val smoke 确认 true/mostly-true 的 oracle accuracy 是否不再系统性低于 fixed-MMR，再合并 train shard 产出完整 `oracle_results_train.jsonl`。
3. 对比旧 oracle（gold_logprob objective, 旧 verifier）与 margin re-oracle（Stage 1 verifier）：按 label 看 oracle accuracy、margin-positive 占比、oracle-only-correct 规模，确认 true-side 不再系统性低于 fixed-MMR。
4. 用 re-oracle 的完整候选池重建 pointwise / preference supervision，避免继续依赖 reconstructed negatives。鉴于 V1a/V1b pointwise 在下游 verifier evaluation 上均低于 fixed-MMR，建议优先使用 preference 或 sequential 范式替代 pointwise independent selection。
5. 先做 evaluation-only：同一个 Stage 1 verifier checkpoint + fixed-MMR vs calibrated selector；val 过 gate 后再跑 test。重点确认 calibrated oracle 的 true-side 不再继承旧 false bias。
6. 若 calibrated pointwise overlap 高但 verifier 指标仍低，再检查 evidence order、prompt truncation、candidate fingerprint、infer decoding；之后再决定是否推进 sequential selector 或 multi-weight MMR。

## 7. 关键文件索引

### Pipeline 核心

| 文件 | 作用 |
|---|---|
| `src/fact_checking/pipeline/run.py` | Hydra 入口（`@hydra.main`），PipelineRunner 编排 |
| `src/fact_checking/pipeline/runner.py` | build → train → infer 三阶段编排 |
| `src/fact_checking/build/candidates.py` | Build 主逻辑（chunking、检索、MMR、Prompt 构建） |
| `src/fact_checking/build/chunking.py` | 5 类 chunking 策略实现 |
| `src/fact_checking/retrieval/mmr.py` | MMR 算法（`maximal_marginal_relevance` + `stepwise`） |
| `src/fact_checking/retrieval/embedder.py` | BGE-base-en-v1.5 文本嵌入封装 |
| `src/fact_checking/retrieval/text_utils.py` | 词汇分（F1）、BM25 分计算 |
| `src/fact_checking/data/constants.py` | 标签常量（6 类、3 类、字母映射） |
| `src/fact_checking/infer/api.py` | vLLM API 推理与 server 管理 |
| `src/sft/trainer.py` | 生成式 SFT 训练器 |
| `src/sft/classifier_trainer.py` | 判别式分类器训练器 |
| `src/sft/parser.py` | 标签解析（三级优先级：精确字母 → 标签名 → 全文扫描） |
| `src/sft/metrics.py` | 分类指标计算 |

### Learned Lambda / RL-MMR

| 文件 | 作用 |
|---|---|
| `src/fact_checking/learned_lambda/predictor.py` | `ChunkEmbeddingLambdaEncoder`、`LambdaPredictor`、`LambdaClassifier` |
| `src/fact_checking/learned_lambda/embedding_features.py` | chunk embedding 特征构建 |
| `src/fact_checking/learned_lambda/features.py` | 73 维手工特征提取 |
| `scripts/learned_lambda/compute_oracle_lambda.py` | oracle λ 计算（21 网格，tie-break 在 L269） |
| `scripts/learned_lambda/train_predictor.py` | λ predictor 训练 |
| `src/fact_checking/rl_mmr/sensitivity.py` | 敏感度特征提取 + 门控决策函数 |
| `src/fact_checking/rl_mmr/gated_selector.py` | Build 阶段批量调用门控逻辑 |
| `src/fact_checking/rl_mmr/dpo_policy.py` | `StepLambdaPolicy` (MLP), `FixedReferencePolicy`, `dpo_loss()` |
| `src/fact_checking/rl_mmr/trajectory.py` | `Trajectory`, `MMRStep`, `PreferencePair` dataclasses |
| `src/fact_checking/rl_mmr/step_features.py` | 每步 state 特征提取（13 维最终版） |

### Oracle Evidence / Pointwise Selector

| 文件 | 作用 |
|---|---|
| `scripts/oracle_evidence/search_optimal_evidence.py` | Oracle evidence search 主脚本（greedy + exhaustive） |
| `src/fact_checking/oracle_evidence/scorer.py` | vLLM scoring（gold_logprob / all-label / margin） |
| `src/fact_checking/oracle_evidence/search.py` | Search 主逻辑（`objective=margin/gold_logprob`） |
| `src/fact_checking/oracle_pointwise.py` | 共享工具模块：数据加载、特征构建、指标计算 |
| `scripts/selectors/build_pointwise_oracle_dataset.py` | 从 oracle 输出构造 pointwise 训练数据 |
| `scripts/selectors/train_pointwise_oracle_selector.py` | NumPy logistic regression 训练 |
| `scripts/selectors/eval_pointwise_oracle_selector.py` | Selection-only 评估 |

### Verifier Calibration

| 文件 | 作用 |
|---|---|
| `src/sft/label_token_dataset.py` | 将 build rows 转成 `prompt + Label:` 训练样本 |
| `src/sft/label_token_trainer.py` | A-F label token weighted CE 训练 |
| `configs/experiment/b3_label_token_ce_1024.yaml` | Stage 1 实验配置 |
| `scripts/verifier/run_label_token_ce_stage1.sh` | Stage 1 一键 build/train/infer |
| `scripts/oracle_evidence/run_reoracle_stage2.sh` | Stage 2 margin re-oracle 运行脚本 |
| `scripts/oracle_evidence/merge_shards.py` | 合并 sharded oracle JSONL |

### 配置文件

| 文件 | 作用 |
|---|---|
| `configs/build/default.yaml` | Build 默认配置（chunking、retrieval 参数） |
| `configs/train/default.yaml` | 训练默认配置（LoRA、DeepSpeed、学习率） |
| `configs/infer/vllm_api.yaml` | 推理默认配置（vLLM server、decoding） |
| `configs/pipeline/default.yaml` | Pipeline 编排默认配置 |
| `configs/experiment/b0.yaml` | 生成式 SFT 基线配置 |
| `configs/experiment/b4.yaml` | 判别式分类器基线配置（`kind=classifier`） |
| `configs/experiment/b4_3class.yaml` | 3 分类判别式实验配置 |
| `configs/experiment/mmr_lambda_sweep.yaml` | MMR λ sweep 实验配置 |
| `configs/experiment/mmr_sensitivity_gated.yaml` | Sensitivity-gated MMR 配置 |
| `configs/experiment/mmr_dpo_step_lambda.yaml` | DPO step-wise λ 配置 |
