# 实验4: Soft-Label Lambda Policy 实施计划

## Context

前三个实验（fixed λ=0.7、log(n) heuristic、sensitivity-gated MMR）在 test set 上的
accuracy/macro-F1 均在 ~0.27-0.28 区间，delta ≤ 0.006，无显著优势。这说明 hard oracle
lambda（单点 argmax）的监督信号噪声过大，模型难以学到有效的 lambda 预测策略。

实验 4 的目标是修复这个噪声监督问题：不再让模型学习单点 argmax lambda，而是学习完整
utility curve 的 soft target 分布。低 margin 样本不应贡献高权重 hard label。

## 核心设计

### Lambda 网格
使用粗网格 5 点：{0.1, 0.3, 0.5, 0.7, 0.9}（不先用 21 点网格）

### Soft target 构造
```
q_i(λ_j) = exp(U_i(λ_j) / τ) / Σ_k exp(U_i(λ_k) / τ)
```
其中 U_i(λ) = correct label logprob，τ = softmax temperature

### 样本权重
两种方式均可测试：
1. margin-based: w_i = max U_i - U_i(0.7)
2. gap-based: w_i = top1_utility - top2_utility

### 模型（轻量 tabular，不用大模型）
1. Logistic Regression — 可解释 baseline
2. Gradient Boosting (LightGBM/XGBoost) — 非线性 tabular baseline
3. 小 MLP (2-3 hidden layers, hidden_dim=64-128) — neural baseline

### 推理策略（三种都要测试）
1. argmax: λ = argmax_j p_θ(λ_j | x)
2. expected: λ = Σ_j p_θ(λ_j | x) * λ_j
3. sample: λ ~ p_θ(λ | x), temperature=0.5

## 实施阶段

### Stage 1: 生成效用曲线（utility curves）

**目的**: 对每个 claim 计算所有 λ 网格点下的 correct label logprob

**复用**:
- `scripts/learned_lambda/generate_oracle_prompts.py` — 已支持 `--lambda-grid` 参数，改为 5 点网格
- `scripts/learned_lambda/compute_oracle_lambda.py` — 直接可用，输出 `logprobs_by_lambda`

**步骤**:
1. 对 train/val/test 三个 split 分别生成 per-λ 的 prompt JSONL（复用 chunk-mmr cache）
2. 用已有 verifier checkpoint (infer_id=79d8b34809bb) 跑 vLLM 推理，获取 logprobs
3. 输出 `outputs/rl_mmr/oracle_logprobs/{split}.jsonl`

**启动命令**:
```bash
# 生成 prompt（train/val/test 都要）
PYTHONPATH=src python scripts/learned_lambda/generate_oracle_prompts.py \
  --experiment b3_mmr_topk_sweep_1024 \
  --split-name train \
  --output-dir outputs/rl_mmr/oracle_prompts/ \
  --lambda-grid "0.1,0.3,0.5,0.7,0.9" \
  --top-k 5

# 计算 oracle logprobs
PYTHONPATH=src python scripts/learned_lambda/compute_oracle_lambda.py \
  --prompts-dir outputs/rl_mmr/oracle_prompts/ \
  --model /data/models/Qwen2.5-7B-Instruct \
  --lora-adapter <train_best_checkpoint_path> \
  --output outputs/rl_mmr/oracle_logprobs/train.jsonl \
  --split-name train
```

**计算成本估计**: 
- 3 splits × 5 lambdas × ~1000-1300 claims ≈ 15000-19000 vLLM 推理
- 用单 GPU vLLM，每 prompt ~1-2s，共约 4-8 GPU-hours
- 如果有之前实验的 logprobs 缓存可直接复用其中的 λ=0.7 数据

**输出格式**: 
```json
{"event_id": "xxx.json", "gold_label": "false", "oracle_lambda": 0.7, 
 "best_logprob": -1.2, "logprobs_by_lambda": {"0.10": -3.5, "0.30": -2.1, ...}}
```

### Stage 2: 特征提取

**目的**: 对每个 claim 提取三类特征向量用于训练 lambda policy

**新增文件**: `src/fact_checking/rl_mmr/soft_label_features.py`

**复用**:
- `src/fact_checking/learned_lambda/features.py::extract_features()` — 70+ 维度 feature vector
- `src/fact_checking/rl_mmr/sensitivity.py::sensitivity_features()` — interventional MMR 特征

**实现内容**:

```python
# soft_label_features.py 核心函数

def extract_soft_label_features(
    chunk_sample: ChunkMMRSample,
    hybrid_scores: np.ndarray,
    chunk_emb: np.ndarray,
    lambda_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
    top_k: int = 5,
) -> dict[str, float | list]:
    """
    返回 dict 包含三类特征:
    A. Pool features (10+ features)
    B. Interventional MMR features (15+ features)
    C. Claim features (8+ features)
    共计 ~35-40 维 tabular 特征
    """
```

**特征清单**:

A. Pool features（复用 features.py 子集）:
- n_candidates, log_n_candidates
- score_entropy, score_std, score_gini
- top1_top2_gap, top5_score_std
- mean_pairwise_sim, max_pairwise_sim
- score_q10, q50, q90

B. Interventional MMR features（复用 sensitivity.py）:
- Jaccard(S_0.1, S_0.7), Jaccard(S_0.3, S_0.7), Jaccard(S_0.1, S_0.9)
- mean_rel(S_0.7) - mean_rel(S_0.3)
- mean_red(S_0.7) - mean_red(S_0.3)
- Kendall_tau(rank_0.3, rank_0.7)
- n_selected_changes across grid
- pool_redundancy
- selected_redundancy at λ=0.7

C. Claim features（复用 features.py）:
- claim_token_count, claim_word_count
- entity_count, number_count, time_expression_count
- negation_flag, comparison_flag, superlative_flag

### Stage 3: 训练数据集构建

**新增文件**: `src/fact_checking/rl_mmr/soft_label_dataset.py`

**核心功能**:
```python
class SoftLabelDataset:
    """从 oracle logprobs + chunk cache 构建训练数据"""
    features: np.ndarray   # [N, D] 特征矩阵
    soft_targets: np.ndarray  # [N, 5] soft target 分布
    sample_weights: np.ndarray  # [N] 样本权重
    event_ids: list[str]
    lambda_grid: np.ndarray  # [5]
    
    @classmethod
    def from_oracle_and_cache(
        cls,
        oracle_jsonl: Path,
        chunk_cache_pkl: Path,
        lambda_grid: list[float],
        temperature: float = 1.0,
        weight_mode: str = "margin",  # "margin" or "gap"
    ) -> "SoftLabelDataset":
        ...
```

**数据处理流程**:
1. 加载 oracle logprobs JSONL → {event_id: logprobs_by_lambda}
2. 加载 chunk-mmr cache → list[ChunkMMRSample]
3. 为每个 ChunkMMRSample:
   - 调用 `compute_hybrid_scores()` 获取 scores
   - 调用 `extract_soft_label_features()` 获取特征向量
   - 从 oracle 读取 U_i(λ) 构造 soft target
   - 计算 sample weight
4. 标准化特征（z-score），保存 mean/std
5. 输出 `outputs/rl_mmr/soft_label/train_dataset.npz`

**特征标准化**: 用train集mean/std对val/test进行标准化

### Stage 4: 模型训练

**新增文件**: `scripts/rl_mmr/train_soft_label_policy.py`

**三种模型**:

1. **Logistic Regression** (baseline, sklearn)
   - Multi-class softmax via `sklearn.linear_model.LogisticRegression`
   - 超参: C in {0.01, 0.1, 1.0, 10.0}
   - 支持 sample_weight
   
2. **Gradient Boosting** (LightGBM)
   - `lgb.LGBMClassifier` with soft targets
   - 超参: num_leaves, learning_rate, n_estimators
   
3. **MLP** (PyTorch, 复用 LambdaClassifier 结构调整)
   - Input → 128 → 64 → 5 (softmax)
   - KL-divergence loss with sample weights
   - Early stopping on val set

**训练脚本参数**:
```
--oracle-logprobs outputs/rl_mmr/oracle_logprobs/train.jsonl
--chunk-mmr-cache <chunk_cache_path>
--lambda-grid 0.1,0.3,0.5,0.7,0.9
--temperature 1.0
--weight-mode margin
--model-type lightgbm  # lr, lightgbm, mlp
--output-dir outputs/rl_mmr/soft_label/
```

**训练评估指标**:
- KL divergence on val set
- ECE (expected calibration error)
- Predicted λ distribution (不应坍缩到单一类别)
- Prediction entropy per sample

### Stage 5: 评估（离线/回顾式）

**新增文件**: `scripts/rl_mmr/evaluate_soft_label_policy.py`

**目的**: 不跑完整 pipeline，直接用已有 MMR 选择 + oracle logprobs 进行回顾式评估

**两种评估方式**:

**A. 回顾式评估（快速，不需要额外 GPU）**:
对 val/test set 的每个 claim:
1. 模型预测 p_θ(λ | x)
2. 分别用三种推理策略选出 λ
3. 用 pre-computed oracle logprobs 查表评估: 这个 λ 对应的 U_i(λ) 是多少？
4. 统计 mean utility per inference strategy
5. 与 fixed 0.7 对比: mean utility 是否有提升？

**B. 分桶分析**:
- candidate count bucket
- sensitivity bucket (high/low)
- pool redundancy bucket
- oracle margin bucket
- 重点观察: high sensitivity + high redundancy bucket

**输出**:
```
outputs/rl_mmr/soft_label/
  eval_summary.json        # 整体指标
  eval_by_bucket.json       # 分桶指标
  predictions.jsonl         # per-sample 预测结果
  calibration.png           # reliability diagram
```

### Stage 6: 管线集成（完整 pipeline 运行）

**目的**: 将训练好的模型接入 build 管线，运行 build → train → infer 完整流程，获取真正的 fact-checking accuracy

**新增/修改文件**:

1. `src/fact_checking/rl_mmr/soft_label_selector.py` — 类似 `gated_selector.py`:
```python
def build_lambda_overrides_from_soft_label(
    chunk_samples: list[ChunkMMRSample],
    model,  # 训练好的分类器
    feature_stats: dict,
    lambda_grid: list[float],
    inference_mode: str = "argmax",  # argmax/expected/sample
) -> dict[str, float]:
    """返回 {event_id: predicted_lambda}"""
```

2. `configs/experiment/mmr_soft_label.yaml`:
```yaml
# @package _global_
defaults:
  - /experiment/b3_mmr_topk_sweep_1024

build:
  retrieval:
    mmr_lambda: 0.7  # fallback
    top_k: 5
    learned_lambda:
      enabled: true
      mode: soft_label
      soft_label:
        model_path: outputs/rl_mmr/soft_label/${model_type}/
        lambda_grid: [0.1, 0.3, 0.5, 0.7, 0.9]
        inference_mode: argmax  # argmax | expected | sample
        sample_temperature: 0.5
```

3. `src/fact_checking/build/candidates.py` 的 `run_build()` — 在 learned_lambda 分发处添加 `"soft_label"` 分支:
```python
elif mode == "soft_label":
    from fact_checking.rl_mmr.soft_label_selector import build_lambda_overrides_from_soft_label
    lambda_overrides = build_lambda_overrides_from_soft_label(...)
```

4. `scripts/rl_mmr/run_soft_label_full.sh`:
```bash
#!/bin/bash
# Full pipeline: build → train → infer with soft-label policy
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=mmr_soft_label \
  pipeline.mode=full \
  build.retrieval.learned_lambda.soft_label.inference_mode=$1
```

### Stage 7: 超参搜索

**搜索空间**:
- temperature τ ∈ {0.5, 1.0, 2.0, 3.0}
- weight_mode ∈ {margin, gap, none}
- model_type ∈ {lr, lightgbm, mlp}
- inference_mode ∈ {argmax, expected, sample}

**策略**: Stage 5 回顾式评估快速筛选（不需要 GPU），选出 top-3 配置后再跑 Stage 6 完整 pipeline

## 文件清单

### 新增文件
| 文件 | 用途 |
|------|------|
| `src/fact_checking/rl_mmr/soft_label_features.py` | 特征提取函数 |
| `src/fact_checking/rl_mmr/soft_label_dataset.py` | 数据集构建 |
| `src/fact_checking/rl_mmr/soft_label_selector.py` | 管线集成（build_lambda_overrides） |
| `scripts/rl_mmr/train_soft_label_policy.py` | 训练脚本 |
| `scripts/rl_mmr/evaluate_soft_label_policy.py` | 回顾式评估脚本 |
| `scripts/rl_mmr/run_soft_label_full.sh` | 完整 pipeline 脚本 |
| `configs/experiment/mmr_soft_label.yaml` | 实验配置 |

### 修改文件
| 文件 | 改动 |
|------|------|
| `src/fact_checking/build/candidates.py` | 在 `run_build()` 添加 "soft_label" mode 分支 |

## 评估与对比

### 主表指标
- accuracy, macro_f1, macro_precision, macro_recall
- 各推理策略 (argmax/expected/sample) 对比
- λ 预测分布统计（mean, std, histogram）

### 分桶分析（与实验计划 §10 统一）
- 每个 bucket 下与 fixed 0.7 的 accuracy delta
- 重点看 high sensitivity + high redundancy bucket

### 与前面实验对比
| System | 说明 |
|--------|------|
| fixed λ=0.7 | Exp1 baseline |
| log(n) heuristic | Exp2 simple adaptive |
| sensitivity-gated | Exp3 hard gate |
| soft-label LR | Exp4 可解释 baseline |
| soft-label MLP | Exp4 neural |
| soft-label GBDT | Exp4 非线性 |

## 停止条件（来自原始实验计划 §15）

满足以下条件之一则停止该方向，转入 DPO 路线（实验 5）：
1. Dev accuracy 不超过 log(n) heuristic 或 sensitivity-gated
2. 预测分布坍缩到单一 λ（95%+ 样本预测同一 λ）
3. Soft-label policy 在 dev set 上的 U(λ_pred) 不超过 U(0.7)

## 计算成本估计

| 阶段 | GPU 需求 | 时间估计 | 备注 |
|------|----------|----------|------|
| Stage 1 生成 prompts | CPU only | ~10 min/split | 复用 chunk cache |
| Stage 1 vLLM 推理 | 1 GPU | ~4-8 GPU-hr | ~19000 prompts × ~1.5s |
| Stage 2 特征提取 | CPU only | ~5 min/split | numpy 运算 |
| Stage 3 数据集构建 | CPU only | ~1 min | 纯粹数据整理 |
| Stage 4 模型训练 | CPU only | ~5-30 min | tabular 数据很小 |
| Stage 5 评估 | CPU only | ~1 min | 查表式 |
| Stage 6 完整 pipeline | 2-4 GPUs | ~2-4 GPU-hr | 同常规实验 |

总计: ~6-12 GPU-hours, ~1-2 hours wall time

## 验证方式

1. `PYTHONPATH=src python scripts/rl_mmr/train_soft_label_policy.py --help` — 确认参数正确
2. 小批量训练（100 条样本）验证 pipeline 跑通
3. 回顾式评估输出与预期一致（predicted λ 在 [0.1, 0.9] 范围内）
4. 分桶分析脚本输出格式与现有 `infer_metrics_summary_all.csv` 兼容
5. 完整 pipeline 输出的 metrics.json 格式与其他实验一致，可直接加入 `overall_run_analysis`
