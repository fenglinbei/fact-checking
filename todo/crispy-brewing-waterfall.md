# 两个 Learned Lambda 改进实验的详细计划

## 背景

- Oracle λ 在 val 集上相比固定 λ=0.7 提升 **3.1%** 准确率（33.48% vs 30.40%，2013 样本）
- 但当前 `ChunkEmbeddingLambdaEncoder` 预测器 R² vs 均值基线仅 0.058，几乎没学到东西
- 根因：72.6% 的 claim 的 oracle logprob margin < 0.05（λ 选择几乎不影响结果）
- 策略：缩小问题空间，降低噪声

## 公共前提：生成 val/test oracle

当前只有 `oracle_lambda_train.jsonl`（10065 条）。实验需要 val oracle 进行独立评估。

**步骤**（在执行实验训练之前完成）：
```bash
# 1. 生成 val split 的 prompt（已有 train 的，补 val 的）
SPLIT_NAME=val bash scripts/learned_lambda/run_generate_oracle_prompts.sh

# 2. 计算 val split 的 oracle λ
SPLIT_NAME=val bash scripts/learned_lambda/run_compute_oracle_lambda.sh
```
产出：`outputs/learned_lambda/oracle_lambda_val.jsonl`

---

## 实验 1：高 Margin 过滤（margin >= 0.05）

### 思路

只在高 margin 子集上训练预测器。对低 margin 的 claim（λ 不重要），fallback 到默认 λ=0.7。

### 数据分析

| 子集 | N | % | Oracle mean | Oracle std |
|------|---|----|-------------|------------|
| 全量 | 10065 | 100% | 0.445 | 0.296 |
| margin >= 0.05 | 2761 | 27.4% | 0.282 | 0.259 |
| margin < 0.05 | 7304 | 72.6% | 0.507 | 0.289 |

高 margin 子集 oracle λ 分布显著偏向低值（diversity）：61.8% 的 oracle λ 在 [0.0, 0.3] 区间。

### 修改内容

#### 1.1 修改 `scripts/learned_lambda/train_predictor.py`

新增 `--margin-threshold` 参数：

在 `parse_args()` 中添加：
```python
p.add_argument("--margin-threshold", type=float, default=0.0,
               help="Only train on claims with oracle logprob margin >= this value. "
                    "0.0 means use all data (default). 0.05 keeps ~27%% of data.")
```

在加载 oracle 数据后（第 219 行之后），增加过滤逻辑：
```python
# 计算 margin 并过滤
if args.margin_threshold > 0:
    n_before = len(oracle_by_eid)
    keep_eids = []
    for eid, rec in oracle_by_eid.items():
        lp = rec.get("logprobs_by_lambda", {})
        if isinstance(lp, dict) and len(lp) >= 2:
            vals = sorted([float(v) for v in lp.values()], reverse=True)
            margin = vals[0] - vals[1]
            if margin >= args.margin_threshold:
                keep_eids.append(eid)
    keep_set = set(keep_eids)
    oracle_by_eid = {eid: rec for eid, rec in oracle_by_eid.items() if eid in keep_set}
    _log(f"Margin >= {args.margin_threshold}: kept {len(oracle_by_eid)} / {n_before} records",
         show_progress=show_progress)
```

推荐超参调整（应对数据量减少）：
- `HIDDEN_DIM`：256 → 128
- `DROPOUT`：0.1 → 0.2
- `BATCH_SIZE`：256 → 128
- `PATIENCE`：30 → 20
- `EPOCHS`：200 → 150

#### 1.2 修改 `scripts/learned_lambda/evaluate_predictor.py`

在评估输出中增加按 margin 分组的指标：
- 全量数据上的 MAE/RMSE（主要指标）
- 高 margin 子集（margin >= 0.05）上的 MAE/RMSE
- 低 margin 子集上的 MAE/RMSE 及 fallback-to-default (0.7) 的对比

#### 1.3 新增 Shell 脚本 `scripts/learned_lambda/run_train_predictor_high_margin.sh`

基于 `run_train_predictor.sh` 复制修改，默认值：
- `MARGIN_THRESHOLD=0.05`
- `HIDDEN_DIM=128`
- `DROPOUT=0.2`
- `BATCH_SIZE=128`
- `PATIENCE=20`
- `EPOCHS=150`
- `OUTPUT_DIR=outputs/learned_lambda_high_margin`

#### 1.4 Pipeline 集成（可选增强）

在 `src/fact_checking/build/candidates.py` 中，当预测 λ 与默认 λ 差距小于阈值时 fallback 到默认值：
```python
predicted_lambda = lambda_overrides.get(sample.event_id, mmr_lambda)
if abs(predicted_lambda - mmr_lambda) < 0.10:
    effective_lambda = mmr_lambda
else:
    effective_lambda = predicted_lambda
```

#### 1.5 预期结果

- Train 集高 margin 子集上：Pearson r 目标 > 0.35
- 端到端准确率：目标比固定 0.7 高 ≥ 1.0%

---

## 实验 2：3-bin 粗粒度分类

### 思路

将 21 个细粒度 λ 值合并为 3 个粗粒度区间：
- **Bin 0 (diversity)**: λ ∈ [0.0, 0.3]，代表值 0.15
- **Bin 1 (balanced)**: λ ∈ (0.3, 0.7)，代表值 0.50
- **Bin 2 (relevance)**: λ ∈ [0.7, 1.0]，代表值 0.85

### 数据分析

| Bin | 区间 | N | % | Oracle mean | High-margin % |
|-----|------|---|----|-------------|---------------|
| diversity | [0.0, 0.3) | 3667 | 36.4% | 0.123 | 46.4% |
| balanced | [0.3, 0.7) | 3382 | 33.6% | 0.477 | 22.4% |
| relevance | [0.7, 1.0] | 3016 | 30.0% | 0.802 | 10.1% |

关键发现：relevance bin 中仅 10.1% 是 high-margin，但 diversity bin 中 46.4% 是 high-margin。

### 修改内容

#### 2.1 `train_predictor.py` — 基本无需修改

现有的 `--objective classification --lambda-grid "0.15,0.50,0.85"` 已经支持 3 类分类。

`_nearest_grid_indices()` 会将 oracle λ 映射到最近的 grid 值对应的 class index（0/1/2）。

`ChunkEmbeddingLambdaEncoder.forward()` 在 classification 模式下输出 weighted sum of bin centers × softmax probs。

推荐使用 `soft_classification` 模式：
- `--objective soft_classification --lambda-grid "0.15,0.50,0.85"`
- 用 full logprob distribution 做软标签，让模型学到每个 bin 的可信度
- `--softmax-temperature 2.0` 软化分布

#### 2.2 修改 `evaluate_predictor.py`

当 `model_type == "classifier"` 时，增加分类评估指标：

```python
def _print_classification_metrics(pred_arr, oracle_arr, lambda_grid):
    """Print 3-bin accuracy, confusion matrix, per-bin precision/recall."""
    boundaries = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
    bin_names = ["diversity [0,0.3]", "balanced (0.3,0.7)", "relevance [0.7,1.0]"]
    
    oracle_bins = _assign_bins(oracle_arr, boundaries)
    pred_bins = _assign_bins(pred_arr, boundaries)
    
    acc = (oracle_bins == pred_bins).mean()
    print(f"\n3-Bin Classification:")
    print(f"  Accuracy: {acc:.4f} (random=0.333)")
    
    print(f"\n  Confusion matrix (rows=oracle, cols=pred):")
    for i, name in enumerate(bin_names):
        row = [f"{(oracle_bins[pred_bins == j] == i).sum():4d}" for j in range(3)]
        print(f"    {name:25s} {' '.join(row)}")
    
    print(f"\n  Per-bin metrics:")
    for i, name in enumerate(bin_names):
        tp = ((oracle_bins == i) & (pred_bins == i)).sum()
        prec = tp / max((pred_bins == i).sum(), 1)
        rec = tp / max((oracle_bins == i).sum(), 1)
        print(f"    {name:25s} precision={prec:.3f} recall={rec:.3f}")


def _assign_bins(lambdas, boundaries):
    bins = np.zeros(len(lambdas), dtype=int)
    for i, (lo, hi) in enumerate(boundaries):
        if hi >= 1.0 - 1e-8:
            bins[(lambdas >= lo) & (lambdas <= hi)] = i
        else:
            bins[(lambdas >= lo) & (lambdas < hi)] = i
    return bins
```

#### 2.3 新增 Shell 脚本 `scripts/learned_lambda/run_train_predictor_coarse.sh`

基于 `run_train_predictor.sh` 复制修改，默认值：
- `OBJECTIVE=classification`（或 `soft_classification`）
- `LAMBDA_GRID=0.15,0.50,0.85`
- `OUTPUT_DIR=outputs/learned_lambda_coarse`
- `SOFTMAX_TEMPERATURE=2.0`（soft_classification 模式）
- 其余超参与基线一致（hidden_dim=256, dropout=0.1, epochs=200, lr=1e-4）

#### 2.4 Pipeline 集成

无需修改。`ChunkEmbeddingLambdaEncoder.forward()` 在 classification 模式下输出 expected λ（weighted sum），可直接用于 `lambda_overrides`。

#### 2.5 预期结果

- 3-bin 分类准确率：目标 > 45%（random = 33.3%）
- MAE：~0.22-0.25（粗粒度有下限约 0.1-0.15）
- 端到端准确率：预期介于固定 λ 和 oracle λ 之间，目标 ≥ 1.0% 提升

### 可选增强：bin + margin 联合

在 3-bin 分类基础上，同时预测"是否 high-margin"（二分类）。实际使用时：
- 预测为 low-margin → 使用默认 λ=0.7
- 预测为 high-margin → 使用预测的 bin 代表值

这相当于结合实验 1 和实验 2 的思路，可作为实验 3。

---

## 验证计划

> 目前服务器可用显存不多，只能跑轻量训练，vllm推理应该暂时用不了，端到端实验暂时先不做

### 阶段 1：训练

```bash
# 实验 1: high-margin
bash scripts/learned_lambda/run_train_predictor_high_margin.sh

# 实验 2: 3-bin coarse
bash scripts/learned_lambda/run_train_predictor_coarse.sh
```

### 阶段 2：评估（train oracle 上）

```bash
# 实验 1
MODEL=outputs/learned_lambda_high_margin/predictor.pt \
FEATURE_STATS=outputs/learned_lambda_high_margin/feature_stats.json \
bash scripts/learned_lambda/run_evaluate_predictor.sh

# 实验 2
MODEL=outputs/learned_lambda_coarse/predictor.pt \
FEATURE_STATS=outputs/learned_lambda_coarse/feature_stats.json \
bash scripts/learned_lambda/run_evaluate_predictor.sh
```

关键观察指标：
- **实验 1**：高 margin 子集上的 Pearson r（目标 > 0.35）
- **实验 2**：3-bin 分类准确率（目标 > 45%），confusion matrix
- **两者**：全量 train 上的 MAE/RMSE

### 阶段 3：端到端验证

```bash
# 实验 1
PYTHONPATH=src python -m fact_checking.pipeline.run \
    experiment=learned_lambda_mmr \
    build.retrieval.learned_lambda.model_path=outputs/learned_lambda_high_margin/predictor.pt \
    build.retrieval.learned_lambda.feature_stats_path=outputs/learned_lambda_high_margin/feature_stats.json \
    pipeline.mode=infer

# 实验 2
PYTHONPATH=src python -m fact_checking.pipeline.run \
    experiment=learned_lambda_mmr \
    build.retrieval.learned_lambda.model_path=outputs/learned_lambda_coarse/predictor.pt \
    build.retrieval.learned_lambda.feature_stats_path=outputs/learned_lambda_coarse/feature_stats.json \
    pipeline.mode=infer
```

### 成功标准

| 实验 | 最低目标 | 期望目标 |
|------|---------|---------|
| 实验 1 | 端到端比固定 0.7 高 ≥ 1.0% | ≥ 2.0% |
| 实验 2 | 3-bin 准确率 ≥ 40%；端到端 ≥ 固定 0.7 | 端到端 ≥ 1.5% 提升 |
| 两者均失败 | 回退到 log(n_candidates) 启发式 | — |

---

## 涉及修改的文件

1. `scripts/learned_lambda/train_predictor.py` — 新增 `--margin-threshold` 参数
2. `scripts/learned_lambda/evaluate_predictor.py` — 新增分类指标（accuracy, confusion matrix, per-bin metrics），按 margin 分组报告
3. `scripts/learned_lambda/run_train_predictor_high_margin.sh` — **新建**，实验 1 的 shell 包装
4. `scripts/learned_lambda/run_train_predictor_coarse.sh` — **新建**，实验 2 的 shell 包装
5. （可选）`src/fact_checking/build/candidates.py` — 增加预测置信度 fallback 逻辑

## 无需修改的文件

- `src/fact_checking/learned_lambda/predictor.py` — `ChunkEmbeddingLambdaEncoder` 已支持 classification + lambda_grid
- `src/fact_checking/learned_lambda/embedding_features.py` — 特征构建无需修改
- `src/fact_checking/learned_lambda/features.py` — 不涉及
- `configs/experiment/learned_lambda_mmr.yaml` — 仅通过命令行 override model_path
