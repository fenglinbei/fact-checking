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

## 实验验证

基于上述分析，我们设计了两个实验来尝试改善预测效果：

1. **实验 1：高 Margin 过滤** — 只在 margin >= 0.05 的 2761 条（27.4%）数据上训练，过滤低质量标签
2. **实验 2：3-Bin 粗粒度分类** — 将 21 个 λ 值合并为 3 个区间（diversity/balanced/relevance），降低分类难度

### 实验 1：高 Margin 过滤（margin >= 0.05）

**设计思路**：只在高 margin 子集上训练预测器。低 margin claim fallback 到默认 λ=0.7。减小模型复杂度应对数据量减少（hidden_dim=128, dropout=0.2）。

**训练集特征**：

| 指标 | 全量 | High-margin (>=0.05) |
|------|------|----------------------|
| N | 10065 | 2761 |
| Oracle mean | 0.445 | 0.282 |
| Oracle std | 0.296 | 0.259 |
| Oracle 在 [0, 0.3] 占比 | 36.4% | 61.8% |

**结果**：

```
val MAE=0.214, val RMSE=0.270, target_std=0.259
R² vs mean: 负值（比预测均值更差）
```

在全量数据上的最终评估：

```
MAE=0.267, RMSE=0.319
Pearson r=0.181, Spearman r=0.178
R² vs mean: -0.167（负值）
Pred std/oracle std: 0.129（预测极度压缩）
```

**关键发现**：
- 预测值坍缩到 0.30-0.32 极窄区间（std=0.038），对所有 oracle λ 值的预测几乎相同
- `corr(pred, n_candidates) = -0.348` — 模型学到了候选数量信号，但预测范围被 sigmoid 严重压缩
- `corr(pred, oracle) = 0.181` — 与 oracle 几乎无关
- 预测范围：1-2 候选时 pred=0.356，31-60 候选时 pred=0.301（总跨度仅 0.055），而 oracle 在同样区间从 0.589 到 0.467（跨度 0.122）

**结论**：过滤低质量数据不能让模型学到 oracle λ，因为即使在高 margin 子集中，BGE embedding 仍然不包含预测 optimal λ 所需的信息。

### 实验 2：3-Bin 粗粒度分类

**设计思路**：将 21 个细粒度 λ 值合并为 3 个粗粒度区间，降低分类难度：

| Bin | 区间 | 代表值 | N | % | High-margin % |
|-----|------|--------|---|-----|---------------|
| diversity | [0.0, 0.3] | 0.15 | 3667 | 36.4% | 46.4% |
| balanced | (0.3, 0.7) | 0.50 | 3382 | 33.6% | 22.4% |
| relevance | [0.7, 1.0] | 0.85 | 3016 | 30.0% | 10.1% |

使用 `ChunkEmbeddingLambdaEncoder` 的 classification 模式（`lambda_grid=[0.15, 0.50, 0.85]`）。

**结果**：

```
val MAE=0.256, val RMSE=0.294, target_std=0.296
R² vs mean: 0.05
```

在全量数据上的分类评估：

```
3-Bin Classification:
  Accuracy: 0.336 (n=3382/10065, random=0.333)

  Confusion matrix (rows=oracle, cols=pred):
                             diversity balanced relevance
    diversity [0,.3]                 0     3667        0
    balanced (.3,.7)                 0     3382        0
    relevance [.7,1]                 0     3016        0

  Per-bin metrics:
    diversity [0,.3]    precision=0.000 recall=0.000
    balanced (.3,.7)    precision=0.336 recall=1.000
    relevance [.7,1]    precision=0.000 recall=0.000
```

**所有 10065 个样本全部被预测为 "balanced" 类别。**

通过 softmax 原始输出进一步分析：

```
Softmax 概率（均值）: diversity=0.457, balanced=0.287, relevance=0.256
Per-bin 的 diversity 概率:
  oracle=diversity:  0.482
  oracle=balanced:   0.453
  oracle=relevance:  0.432
  → 跨度仅 0.05，std=0.086，三个类别的概率分布几乎完全重叠
```

- Softmax argmax 分布：88.8% diversity, 0.2% balanced, 11.0% relevance
- 按 argmax 计算准确率：40.7%（略高于 random 33.3%，但模型严重偏向 diversity 类）
- Per-class 概率差异远小于类间方差：模型无法通过 BGE embedding 区分这三个类别

**结论**：即使将问题简化为 3 分类，BGE embedding 仍然没有包含区分最优 λ 区间所需的信息。

### 实验总结

| 实验 | val MAE | R² vs mean | 核心失败模式 |
|------|---------|------------|-------------|
| 原版 (全量回归) | 0.256 | 0.01 | 预测坍缩到均值 |
| 实验 1 (high-margin) | 0.267 | -0.17 | 预测坍缩到 0.31 |
| 实验 2 (3-bin 分类) | 0.250 | 0.05 | 所有样本预测同一类 |

三个实验的一致失败表明：**问题不在数据质量、问题粒度或模型架构，而在于 BGE 文本 embedding 根本不编码 "什么 λ 对 MMR 最优" 这个信息。**

Oracle λ 取决于 SFT 模型对证据排列顺序的内部响应。BGE embedding 编码的是文本的语义内容 — 两者之间不存在系统性关联。

## 改进方向建议

### 短期可行：简单启发式

```python
λ = -0.073 * log(n_candidates) + 0.613
```

这个仅用候选数量的线性回归，表现与复杂神经网络相当（MAE=0.253 vs 0.256），零维护成本。

### 中期探索：替代 Oracle 定义

1. **直接搜索替代预测** — 推理时对每个 claim 在少量候选 λ（如 0.0, 0.5, 1.0）上分别做 MMR，用 vLLM 评估 logprob，选最优。成本 3x 但无需训练预测器。
2. **更粗的 Oracle 网格** — 将 oracle λ 的计算从 21 个值减为 3-5 个（如 0.0, 0.5, 1.0），降低多重比较引入的噪声
3. **端到端信号替代 logprob** — 用下游 accuracy 而非 label logprob 作为 λ 优劣的标准

### 长期前提：验证 Oracle 价值

Oracle λ 在 val 集上相比固定 λ=0.7 已验证 **+3.1%** 准确率提升（33.48% vs 30.40%，2013 样本）。这确认了 adaptive λ 有理论价值。但当前基于 BGE embedding 的预测路径不可行，需要重新设计预测器使用的特征（如直接使用 MMR sensitivity 模拟结果而非文本语义），或采用不需要预测器的替代方案。

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
| `scripts/learned_lambda/run_train_predictor_high_margin.sh` | 实验 1 启动脚本（高 margin 过滤训练） |
| `scripts/learned_lambda/run_train_predictor_coarse.sh` | 实验 2 启动脚本（3-bin 粗粒度分类训练） |
| `outputs/learned_lambda_high_margin/` | 实验 1 输出目录 |
| `outputs/learned_lambda_coarse/` | 实验 2 输出目录 |
| `outputs/learned_lambda/comparison/oracle_predictions.jsonl` | Oracle λ 端到端推理结果（val, N=2013） |
| `outputs/learned_lambda/comparison/fixed_0.70_predictions.jsonl` | 固定 λ=0.7 端到端推理结果（val, N=2013） |

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

# 实验 1: 高 margin 过滤训练
bash scripts/learned_lambda/run_train_predictor_high_margin.sh

# 实验 2: 3-bin 粗粒度分类训练
bash scripts/learned_lambda/run_train_predictor_coarse.sh

# 实验 2 评估 (含 classification accuracy 和 confusion matrix)
MODEL=outputs/learned_lambda_coarse/predictor.pt \
FEATURE_STATS=outputs/learned_lambda_coarse/feature_stats.json \
bash scripts/learned_lambda/run_evaluate_predictor.sh
```

## 分析日期

2026-05-14
