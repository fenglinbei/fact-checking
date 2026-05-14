# Learned Lambda Predictor 问题诊断分析

## 概述

"Learned lambda" 实验的目标是用神经网络预测器替换全局固定的 MMR `mmr_lambda` 超参数，为每个 claim 预测最优的 relevance-vs-diversity 权衡参数。但训练出的预测器几乎没有学到有效特征。

## 问题现象

`ChunkEmbeddingLambdaEncoder` 训练后，各变体在验证集上的表现：

| 变体 | val MAE | val RMSE | target std |
|------|---------|----------|------------|
| chunk_embedding (regression) | 0.256 | 0.294 | 0.296 |
| handcrafted 73 features (regression) | 0.250 | 0.283 | 0.299 |
| handcrafted 73 features (classification) | 0.250 | 0.282 | 0.299 |

对比基线：

| 基线方法 | MAE | RMSE |
|----------|-----|------|
| 均值预测 (λ=0.445) | 0.262 | 0.296 |
| 最优固定 λ (0.45) | 0.262 | 0.296 |
| **log(n_candidates) 线性回归** | **0.253** | **0.290** |
| ChunkEmbeddingLambdaEncoder (256维 attention) | 0.256 | 0.294 |

R² vs 均值基线 ≈ **0.01**（最高仅 0.058），模型几乎等价于预测均值。

### 预测器输出崩溃

- 预测 std = 0.057，oracle std = 0.296 → **方差比仅 19%**
- 预测值全部集中在 0.41-0.50 附近（均值 0.468）
- 所有 6 个 label 的预测值几乎完全一致（0.464-0.479）

```
Predicted λ histogram:
  [0.30, 0.35):     28 ( 0.3%) 
  [0.35, 0.40):    970 ( 9.6%)
  [0.40, 0.45):   3164 (31.4%)  ← 集中区
  [0.45, 0.50):   3346 (33.2%)  ← 集中区
  [0.50, 0.55):   1693 (16.8%)
  [0.55, 0.60):    615 ( 6.1%)
  [0.60, 0.65):    221 ( 2.2%)
```

## 根因分析

### 根因 1：Oracle λ 信号本身极弱（核心问题）

Oracle λ 的定义：在 21 个 λ 值 (0.00, 0.05, ..., 1.00) 中，SFT 模型对 **正确 label token** 给出最高 log-probability 的那个 λ。

关键数据：

- **72.6%** 的 claim 的 margin（最优 λ 与次优 λ 的 logprob 差异）< 0.05
- **38.5%** 的 claim margin < 0.01
- 中位数 margin 仅 **0.0185**
- 每个 claim 平均有 **7.9/21 个** λ 值在最优值 0.1 logprob 范围内

这意味着对大多数 claim，**不同 λ 值下 MMR 选出的证据在 SFT 模型眼中几乎没区别**。Oracle λ 本质上是噪声标签 — 在平坦的 logprob 曲面上人为挑出的一个点。

`compute_oracle_lambda.py:269` 行的 tie-break 逻辑进一步引入偏差：
```python
# 当多个 λ 在 0.01 logprob 范围内时，挑离 DEFAULT_LAMBDA=0.7 最近的
candidates = [l for l, lp in lp_by_lam.items() if best_lp - lp < 0.01]
oracle_lam = min(candidates, key=lambda l: abs(l - args.default_lambda))
```

这导致 oracle λ 向 0.7 收缩，降低了分布的自然方差。

Logprob 曲面的平坦程度示例：

```
# 典型低 margin claim
eid=2013.json, label=true, oracle=1.00, margin=0.08
  λ=0.90: logprob=-1.7042
  λ=0.95: logprob=-1.7241
  λ=1.00: logprob=-1.6287  <-- BEST (仅比 λ=0.90 高 0.08)
  λ=0.70: logprob=-1.7795
```

### 根因 2：Oracle λ 与 Claim 语义无关

6 个 veracity label 的 oracle λ 分布几乎完全一致：

```
barely-true:  n=1611 mean=0.437 std=0.299
false:        n=1958 mean=0.420 std=0.299
half-true:    n=2087 mean=0.440 std=0.295
mostly-true:  n=1950 mean=0.450 std=0.297
pants-fire:   n= 812 mean=0.502 std=0.275
true:         n=1647 mean=0.458 std=0.292
```

所有 label 的 oracle λ 均值都在 **0.42-0.50**，标准差都在 **0.28-0.30**。无论 claim 的真假、难易，oracle λ 分布都相同。

### 根因 3：特征与目标之间的语义鸿沟

`ChunkEmbeddingLambdaEncoder` 输入：BGE embedding（768维文本语义向量），经过 hybrid scoring 排序后取 top-k candidate chunks。

但 oracle λ 的实际决定因素：
1. SFT 模型的内部行为（不同 evidence 排序下给正确 label 的 logprob）
2. 不同 λ 下 MMR 证据选择的差异
3. 模型对不同证据顺序的敏感度

**预测器只能看到文本语义，无法看到 SFT 模型的内部状态。** 文本语义中不存在"什么 λ 能让我更好地做对这道题"的信息。

### 根因 4：仅有的可学习信号太弱

候选数量与 oracle λ 的弱相关是**唯一可观察到的系统模式**：

```
候选数 1-2:   oracle_mean=0.589 (偏好 relevance)
候选数 3-4:   oracle_mean=0.596
候选数 5-6:   oracle_mean=0.517
候选数 7-10:  oracle_mean=0.399 (偏好 diversity)
候选数 11-15: oracle_mean=0.390
候选数 21-30: oracle_mean=0.418
候选数 31-60: oracle_mean=0.467
```

`corr(log(n_candidates), oracle_lambda) = -0.13`

模型确实部分捕获了这个模式（预测值从 1-3 候选的 0.572 降到 21+ 候选的 0.413），但这是它能学到的几乎全部内容。

### 根因 5：特征相关性普遍很弱

在 73 维人工特征中，与 oracle λ 的最高相关系数（全量数据）：

| 特征 | 相关系数 |
|------|----------|
| mmr_overlap_lambda_0_00_1_00 | +0.213 |
| score_entropy | -0.205 |
| score_concentration | +0.188 |
| top10_mass | +0.186 |
| n_sentences | -0.135 |

没有任何特征的相关系数超过 0.22，即 **最高解释力 < 5%**。

### 根因 6：Oracle λ 相对固定 λ 的改善有限

Oracle λ 相对默认 λ=0.7 的正确标签概率比：

| 指标 | 值 |
|------|-----|
| 中位数概率比 | **1.14x** |
| 几何平均概率比 | 6.25x（受尾部极端值驱动） |
| 概率比 > 2x 的 claim | 10.6% |
| 概率比 > 5x 的 claim | 2.2% |

**对一半的 claim，oracle λ 相比默认 λ=0.7 的正确标签概率提升不到 14%。** 只有约 10% 的 claim 能从 optimal λ 中获得显著的 (>2x) 正确概率提升。

### 根因 7：高 Margin 子集的特殊性质

在高 margin 子集上（margin >= 0.05，占 27.4%），oracle λ 的分布发生了偏移：

| 阈值 | n | oracle mean | oracle std |
|------|---|-------------|------------|
| margin >= 0.00 | 10065 | 0.445 | 0.296 |
| margin >= 0.01 | 6193 | 0.355 | 0.296 |
| margin >= 0.05 | 2761 | 0.282 | 0.259 |
| margin >= 0.10 | 1256 | 0.207 | 0.189 |

**当 λ 确实有影响时（高 margin），最优 λ 明显偏向 diversity（低值）。** 这意味着对于"λ 敏感的 claim"，提供低 λ（更多样化的证据）比高 λ（更相关的证据）更好。

在这个子集上，特征相关性有所提升（`mmr_rel_lambda_0_30` 相关系数达到 -0.317），但仍然有限。

## 结论

### 核心结论

**Learned lambda 预测器学不到东西的根本原因是 oracle λ 标签本身噪声太大。** 对于大多数 claim，不同 λ 值的效果几乎无差异，使得训练目标基本是随机的。这不是模型架构或特征工程的问题 — 三个完全不同的模型变体（chunk embedding attention、handcrafted features regression、classification）表现几乎完全一致，都只学到了预测均值。

### 问题的本质

Learned lambda 试图用 **证据的文本语义** 来预测 **SFT 模型对不同证据排序的敏感度**。这是两个本质上不相关的域 — 前者是"claim 和 evidence 在讲什么"，后者是"如果证据以不同方式呈现，模型会不会改变判断"。除非这两者之间存在系统性的关联（目前数据表明不存在），否则这个方向在理论上就是困难的。

## 改进方向建议

### 短期：使用简单启发式

```python
λ = -0.073 * log(n_candidates) + 0.613
```

这个仅用候选数量的线性回归，表现与复杂神经网络相当（MAE=0.253 vs 0.256），零维护成本。

### 中期：改进 Oracle 定义

1. **过滤低质量训练数据**：只保留 margin >= 0.05 的 claim（约 27% 样本），在这些"λ 有影响"的样本上训练
2. **粗粒度分类**：将 21 个 λ 合并为 3 个区间（如 diversity / balanced / relevance），分类准确率可能更高
3. **软标签**：用 logprob 分布作为软目标，让模型学到"多个 λ 都可以"的不确定性

### 长期：先验证方向

在进行更多预测器优化之前，必须先在 test 集上回答这个问题：

> **Oracle λ（理论最优，不可实现）相比固定 λ=0.7 能提升多少 end-to-end fact-checking 准确率？**

如果 oracle λ 的准确率提升很小（例如 <1%），那么即使完美预测 oracle λ 也没有意义。如果提升显著，再考虑如何缩小预测器与 oracle 之间的差距。

## 关键文件索引

| 文件 | 说明 |
|------|------|
| `src/fact_checking/learned_lambda/predictor.py` | 模型定义 (ChunkEmbeddingLambdaEncoder, LambdaPredictor, LambdaClassifier) |
| `src/fact_checking/learned_lambda/embedding_features.py` | chunk embedding 特征构建 (build_chunk_embedding_arrays) |
| `src/fact_checking/learned_lambda/features.py` | 73 维人工特征提取 (extract_features) |
| `scripts/learned_lambda/train_predictor.py` | 训练脚本 |
| `scripts/learned_lambda/evaluate_predictor.py` | 评估脚本 |
| `scripts/learned_lambda/compute_oracle_lambda.py` | oracle λ 计算（tie-break 逻辑在第 269 行） |
| `scripts/learned_lambda/generate_oracle_prompts.py` | 生成各 λ 值的 prompt JSONL |
| `outputs/learned_lambda/oracle_lambda_train.jsonl` | oracle λ 标签（10065 条） |
| `outputs/learned_lambda/training_meta.json` | 训练元数据与结果 |
| `outputs/learned_lambda/predictor.pt` | 训练好的 chunk_embedding 模型 (7.3MB) |
| `configs/experiment/learned_lambda_mmr.yaml` | learned lambda 实验配置 |

## 复现步骤

```bash
# 环境
conda activate /data/liaozijie/conda/accelerate-fc/

# 步骤 1: 生成各 λ 值的 prompt
bash scripts/learned_lambda/run_generate_oracle_prompts.sh

# 步骤 2: 用 vLLM 计算 oracle λ
bash scripts/learned_lambda/run_compute_oracle_lambda.sh

# 步骤 3: 训练预测器
bash scripts/learned_lambda/run_train_predictor.sh

# 步骤 4: 评估预测器
bash scripts/learned_lambda/run_evaluate_predictor.sh
```

## 分析日期

2026-05-13
