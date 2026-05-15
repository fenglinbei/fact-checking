# 实验 5: DPO Step-wise λ Policy 实现文档

## 动机

实验 1-4 的结论表明：claim-level scalar λ 的自适应信号很弱，fixed λ=0.7 仍是稳定 baseline。核心问题是"为每条 claim 预测一个 λ"的粒度太粗——MMR 选择是逐步进行的，每步面对的候选池状态不同，理应使用不同的 λ。

实验 5 将学习粒度从 **claim-level scalar λ** 推进到 **step-wise λ trajectory**：

$$ \tau = ((\lambda_1, d_1), (\lambda_2, d_2), \ldots, (\lambda_K, d_K)) $$

核心假设：trajectory-level preference learning 比预测单个 hard oracle λ 更稳定——低 utility gap 的 pair 可以过滤，高 gap pair 提供清晰监督信号。

## 整体架构

```
Chunk-MMR Cache ──→ Trajectory 生成 ──→ Utility 计算 (vLLM) ──→ Preference Pairs
                                                                        │
                                                                        ▼
Build Pipeline ←── DPO Policy Checkpoint ←── DPO 训练 (StepLambdaPolicy)
      │
      ▼
   Train (SFT) ──→ Infer (vLLM) ──→ Metrics
```

五个独立阶段，每阶段产物可缓存复用：

| 阶段 | 脚本 | 输入 | 输出 |
|---|---|---|---|
| Trajectory 生成 | `generate_trajectories.py` | Chunk-MMR cache | `trajectories_{split}.jsonl` |
| Utility 计算 | `compute_trajectory_utility.py` | Trajectories + vLLM | 同上 + `utility` 字段 |
| Preference Pair | `build_preference_pairs.py` | Scored trajectories | `{split}_pairs.npz` |
| DPO 训练 | `train_dpo_step_lambda.py` | Preference pairs | `model_best.pt` |
| 评估 | `evaluate_dpo_step_lambda.py` | Policy + test cache | 指标报告 |

## 核心数据结构

### Trajectory

```python
@dataclass
class Trajectory:
    event_id: str              # claim ID
    claim: str                 # claim 文本
    gold_label: str            # 真实标签
    steps: list[MMRStep]       # K 步选择记录
    selected_ids: list[int]    # 最终选中的候选索引
    lambda_schedule: list[float]  # 使用的 λ 序列
    schedule_type: str         # "handcrafted" | "random"
    utility: float | None      # vLLM 计算的 utility
    evidence_set_key: str      # 去重 key（排序后的 selected_ids）
    state_features: list[np.ndarray] | None  # 每步的特征向量
```

### MMRStep

```python
@dataclass
class MMRStep:
    step_idx: int              # 第几步 (0-based)
    lambda_val: float          # 该步使用的 λ
    selected_idx: int          # 选中的 candidate 索引
    hybrid_score: float        # 选中项的 relevance 分数
    max_sim_to_selected: float # 选中项与已选集合的最大相似度
    mmr_score: float           # 该步的 MMR 分数
```

### PreferencePair

```python
@dataclass
class PreferencePair:
    event_id: str
    traj_win: Trajectory       # τ⁺ (高 utility)
    traj_lose: Trajectory      # τ⁻ (低 utility)
    utility_gap: float         # U(τ⁺) - U(τ⁻)
    evidence_set_diff: bool    # 两个轨迹选择的证据集是否不同
```

## 核心逻辑详解

### 1. Step-wise MMR（`mmr.py`）

与标准 `maximal_marginal_relevance()` 的区别：每步使用不同的 λ。

```python
def maximal_marginal_relevance_stepwise(
    query_scores: np.ndarray,        # [N] relevance scores
    sentence_vectors: np.ndarray,    # [N, D] embeddings
    lambda_weights: list[float],     # 长度 = K，每步的 λ
) -> tuple[list[int], list[dict]]:
```

核心循环：

```
for t in range(K):
    lam = lambda_weights[t]
    if 第一步:
        mmr_scores = query_scores  # 第一步只看 relevance
    else:
        mmr_scores = lam * query_scores - (1 - lam) * max_sim_to_selected
    mmr_scores[已选中的] = -inf
    选中 mmr_scores 最大的候选
    更新 max_sim_to_selected
    记录该步的状态（candidate_mask_before, mmr_scores_before）
```

每步记录 `candidate_mask_before` 和 `mmr_scores_before`，供后续特征提取使用。

### 2. State 特征提取（`step_features.py`）

Policy 在每个 selection step 前观察 state，输出 λ 的选择概率。State 由两部分拼接（共 20 维）：

**Pool 特征（8 维，静态，对所有 step 相同）：**

| 特征 | 含义 |
|---|---|
| `n_candidates` | 候选池大小 |
| `log_n_candidates` | log(1 + N) |
| `score_mean` | relevance 分数均值 |
| `score_std` | relevance 分数标准差 |
| `score_entropy` | softmax(score) 的熵 |
| `top1_top2_gap` | 最高分与次高分的差距 |
| `mean_pairwise_sim` | 候选池平均 pairwise 相似度 |
| `max_pairwise_sim` | 候选池最大 pairwise 相似度 |

**Step 特征（12 维，动态，随选择进程变化）：**

| 特征 | 含义 |
|---|---|
| `step_fraction` | t / K，当前进度 |
| `n_already_selected` | 已选数量 |
| `mean_rel_selected` | 已选中的平均 relevance |
| `mean_red_selected` | 已选中的平均 pairwise 相似度 |
| `max_red_in_pool` | 剩余候选中到已选集合的最大相似度 |
| `mean_red_in_pool` | 剩余候选中到已选集合的平均相似度 |
| `last_selected_rel` | 上一步选中项的 relevance |
| `last_selected_red` | 上一步选中项的 max_sim_to_selected |
| `remaining_score_entropy` | 剩余候选 score 的熵 |
| `remaining_n` | 剩余候选数 |
| `top_mmr_score` | 当前步最高 MMR 分数 |
| `mmr_score_gap` | 当前步 top1 与 top2 MMR 分数差 |

### 3. Trajectory 生成（`generate_trajectories.py`）

对每条 claim，使用手工设计的 schedules + 随机 schedules 生成候选轨迹。

**手工 schedules（7 个，K=5）：**

| Schedule | 含义 |
|---|---|
| `[0.7, 0.7, 0.7, 0.7, 0.7]` | 固定 baseline |
| `[0.3, 0.3, 0.3, 0.3, 0.3]` | 恒定低 λ（强多样性） |
| `[0.5, 0.5, 0.5, 0.5, 0.5]` | 恒定中等 |
| `[0.9, 0.7, 0.5, 0.3, 0.3]` | 递减（先重相关性，后重多样性） |
| `[1.0, 0.7, 0.5, 0.3, 0.1]` | 剧烈递减 |
| `[0.5, 0.5, 0.7, 0.7, 0.9]` | 递增（先多样性，后重相关性） |
| `[0.7, 0.5, 0.3, 0.5, 0.7]` | U 型 |

**随机 schedules（每条 claim 30 个）：** 每步 λ_t ~ Uniform({0.1, 0.3, 0.5, 0.7, 0.9})

对每个 schedule，调用 `Trajectory.from_chunk_sample()` 执行 step-wise MMR 并记录完整轨迹。

### 4. Utility 计算（`compute_trajectory_utility.py`）

Utility = correct label logprob，由 vLLM verifier 计算。

**优化策略：去重 + oracle 复用**

1. 多个 trajectory 可能选到相同 evidence set → 去重后只需计算一次
2. 单 λ 的 trajectory（如 `[0.7, 0.7, 0.7, 0.7, 0.7]`）可直接复用已有的 oracle logprob（来自 `compute_oracle_lambda.py`），无需重新 vLLM 推理
3. 混合 λ 的 trajectory 产生新 evidence set，需要 vLLM scoring

vLLM scoring 方式：对每个 unique evidence set 构造 prompt（复用 `_build_prompt_for_evidence()`），在 prompt 末尾加上 `Label: <gold_letter>`，用 `prompt_logprobs` 获取最后 token 的 logprob。

### 5. Preference Pair 构造（`build_preference_pairs.py`）

对每条 claim：
1. 将 trajectory 按 utility 降序排列
2. 对所有 utility gap ≥ δ(=0.05) 的 pair，构造 PreferencePair
3. 优先保留 evidence_set_diff=True 的 pair（选出的证据集不同）
4. 优先保留 utility gap 大的 pair
5. 每 claim 最多保留 10 个 pair

输出 `train_pairs.npz`，包含：`win_features` [N, K, 20], `win_lambdas` [N, K], `lose_features` [N, K, 20], `lose_lambdas` [N, K], `utility_gaps` [N].

### 6. DPO Policy 模型（`dpo_policy.py`）

**Policy 模型 `StepLambdaPolicy`：** 小型 MLP

```
state_features [B, 20] → Linear(20, 64) → ReLU → Dropout(0.1)
                       → Linear(64, 32) → ReLU → Dropout(0.1)
                       → Linear(32, 5) → logits over {0.1, 0.3, 0.5, 0.7, 0.9}
```

**Reference Policy `FixedReferencePolicy`：** 固定偏向 λ=0.7

```
π_ref(λ) ∝ exp(-|λ - 0.7| / 0.3)
```

这是一个 frozen policy，提供 DPO 训练中的 KL 正则化基准。

**DPO Loss：**

$$ \mathcal{L} = -\log\sigma\left(\beta\left[(\log\pi_\theta(\tau^+) - \log\pi_\theta(\tau^-)) - (\log\pi_{ref}(\tau^+) - \log\pi_{ref}(\tau^-))\right]\right) $$

其中 $\log\pi(\tau) = \sum_{t=1}^{K}\log\pi(\lambda_t | s_t)$，即每步 λ 选择的 log 概率之和。

**训练监控指标：**
- **DPO loss** — 主训练损失
- **Accuracy** — policy 给 winner 分配更高概率的比例
- **Entropy** — $H(\pi_\theta) = -\sum_\lambda \pi_\theta(\lambda|s) \log \pi_\theta(\lambda|s)$，监控是否坍缩
- **Argmax distribution** — 每步各 λ 被选中的比例，检测是否坍缩到单一 λ

### 7. Build Pipeline 集成（`dpo_selector.py` + `candidates.py`）

**集成方式：** 新增 `learned_lambda_mode = "dpo_stepwise"`，在该模式下不走标准 `_mmr_phase_from_chunk_cache`，而是直接运行 step-wise DPO 选择。

**核心函数 `select_candidates_dpo_stepwise()`：**

```
对每个 ChunkMMRSample:
    计算 hybrid_scores 和 chunk_emb
    for t in range(K):
        1. 提取当前 state features（pool + step）
        2. 标准化特征（用训练时保存的 mean/std）
        3. policy(state) → π_θ(λ | s_t)
        4. argmax/sample 选择 λ_t
        5. MMR: mmr_scores = λ_t * relevance - (1-λ_t) * max_sim_to_selected
        6. 选择 mmr_scores 最高的候选
        7. 更新已选集合和 candidate_mask
    返回 selected candidates（与标准 MMR 相同的格式）
```

返回的 candidate 格式与 `_select_candidates_from_chunk_sample` 完全一致，后续 `_build_training_row` 可直接复用。

### 8. 评估

**离线评估（`evaluate_dpo_step_lambda.py`）：**
- 在 test set 上运行 DPO step-wise selection
- 统计每步 λ 分布
- 与 fixed λ=0.7 的 utility 对比（通过 oracle logprob 查表）
- 按 candidate count / sensitivity buckets 分桶分析

**完整 pipeline 评估：**
```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
    experiment=mmr_dpo_step_lambda pipeline.mode=full
```

## 使用流程

### 前置条件

- 已有 Chunk-MMR cache（通过任意 b3 配置的 build 阶段生成）
- 已有 vLLM 环境（用于 utility 计算和后续训练推理）
- 可选：已有 oracle logprob JSONL（可复用，减少 vLLM 计算量）

### Step 1: 生成 Trajectories

```bash
# 对 train/val/test 分别生成
for split in train val test; do
    PYTHONPATH=src python scripts/rl_mmr/generate_trajectories.py \
        --experiment b3_mmr_topk_sweep_1024 \
        --split-name ${split} \
        --output-dir outputs/rl_mmr/dpo_stepwise/trajectories \
        --top-k 5 \
        --n-random 30 \
        --seed 42
done
```

产物：
```
outputs/rl_mmr/dpo_stepwise/trajectories/
├── trajectories_train.jsonl
├── trajectories_val.jsonl
└── trajectories_test.jsonl
```

每行 JSON 包含 event_id, claim, gold_label, lambda_schedule, selected_ids, evidence_set_key, steps, state_features。

### Step 2: 计算 Utility

```bash
for split in train val test; do
    PYTHONPATH=src python scripts/rl_mmr/compute_trajectory_utility.py \
        --trajectories outputs/rl_mmr/dpo_stepwise/trajectories/trajectories_${split}.jsonl \
        --chunk-mmr-cache outputs/cache/chunk_mmr/<fingerprint>/chunk_mmr_${split}.pkl \
        --model /data/models/Qwen2.5-7B-Instruct \
        --lora-adapter outputs/runs/b3_mmr_topk_sweep_1024/<run>/train/best \
        --output outputs/rl_mmr/dpo_stepwise/trajectories/trajectories_${split}_scored.jsonl \
        --reuse-oracle outputs/learned_lambda/oracle_lambda_${split}.jsonl \
        --tensor-parallel-size 4
done
```

`--reuse-oracle` 参数可大幅减少 vLLM 计算量——对于恒定 λ 的 trajectory，直接复用已计算好的 oracle logprob。

产物：`trajectories_{split}_scored.jsonl`（在原始 trajectory 基础上增加 `utility` 字段）。

### Step 3: 构造 Preference Pairs

```bash
for split in train val; do
    PYTHONPATH=src python scripts/rl_mmr/build_preference_pairs.py \
        --trajectories outputs/rl_mmr/dpo_stepwise/trajectories/trajectories_${split}_scored.jsonl \
        --output-dir outputs/rl_mmr/dpo_stepwise/preference_pairs \
        --split-name ${split} \
        --delta 0.05 \
        --max-pairs-per-claim 10
done
```

产物：
```
outputs/rl_mmr/dpo_stepwise/preference_pairs/
├── train_pairs.npz
├── train_pair_statistics.json
├── val_pairs.npz
└── val_pair_statistics.json
```

`train_pair_statistics.json` 包含 n_claims_total, n_pairs_total, utility_gap_mean/std, evidence_set_diff_fraction 等统计。

### Step 4: 训练 DPO Policy

```bash
PYTHONPATH=src python scripts/rl_mmr/train_dpo_step_lambda.py \
    --train-pairs outputs/rl_mmr/dpo_stepwise/preference_pairs/train_pairs.npz \
    --val-pairs outputs/rl_mmr/dpo_stepwise/preference_pairs/val_pairs.npz \
    --output-dir outputs/rl_mmr/dpo_stepwise/checkpoints \
    --beta 1.0 \
    --lr 1e-3 \
    --epochs 200 \
    --batch-size 64 \
    --patience 25 \
    --hidden-dims 64 32 \
    --dropout 0.1
```

训练过程中的关键信号：
- **val loss 下降** → DPO 在学习偏好
- **accuracy > 0.5** → policy 能区分 winner/loser
- **entropy 不接近 0** → policy 未坍缩
- **argmax_max_frac < 0.8** → 未坍缩到单一 λ

产物：
```
outputs/rl_mmr/dpo_stepwise/checkpoints/
├── model_best.pt
├── feature_stats.json
└── training_metrics.json
```

### Step 5: 离线评估（可选，快速验证）

```bash
PYTHONPATH=src python scripts/rl_mmr/evaluate_dpo_step_lambda.py \
    --policy outputs/rl_mmr/dpo_stepwise/checkpoints \
    --experiment b3_mmr_topk_sweep_1024 \
    --split-name test \
    --oracle-logprobs outputs/learned_lambda/oracle_lambda_test.jsonl \
    --output-dir outputs/rl_mmr/dpo_stepwise/eval
```

输出：
- 每步 λ distribution（检查是否有合理结构，如第一步偏重 relevance，后续降低）
- 与 fixed λ=0.7 的 utility 对比
- 分桶分析（candidate count buckets）

### Step 6: 完整 Pipeline 评估

```bash
# 一键运行 build → train → infer
PYTHONPATH=src python -m fact_checking.pipeline.run \
    experiment=mmr_dpo_step_lambda pipeline.mode=full

# 或分步运行
PYTHONPATH=src python -m fact_checking.pipeline.run \
    experiment=mmr_dpo_step_lambda pipeline.mode=build

PYTHONPATH=src python -m fact_checking.pipeline.run \
    experiment=mmr_dpo_step_lambda pipeline.mode=train

PYTHONPATH=src python -m fact_checking.pipeline.run \
    experiment=mmr_dpo_step_lambda pipeline.mode=infer
```

配置入口：`configs/experiment/mmr_dpo_step_lambda.yaml`，继承自 `b3_mmr_topk_sweep_1024`，仅覆写 `learned_lambda` 块。

## 文件清单

### 新增文件

| 文件 | 说明 |
|---|---|
| `src/fact_checking/rl_mmr/trajectory.py` | `Trajectory`, `MMRStep`, `PreferencePair` dataclasses + schedules 定义 |
| `src/fact_checking/rl_mmr/step_features.py` | State 特征提取（pool 8 维 + step 12 维 = 20 维） |
| `src/fact_checking/rl_mmr/dpo_policy.py` | `StepLambdaPolicy`, `FixedReferencePolicy`, `dpo_loss()` |
| `src/fact_checking/rl_mmr/dpo_selector.py` | `select_candidates_dpo_stepwise()`, build pipeline 集成 |
| `scripts/rl_mmr/generate_trajectories.py` | Trajectory 生成脚本 |
| `scripts/rl_mmr/compute_trajectory_utility.py` | vLLM utility 计算脚本 |
| `scripts/rl_mmr/build_preference_pairs.py` | Preference pair 构造脚本 |
| `scripts/rl_mmr/train_dpo_step_lambda.py` | DPO 训练脚本 |
| `scripts/rl_mmr/evaluate_dpo_step_lambda.py` | 评估脚本 |
| `configs/experiment/mmr_dpo_step_lambda.yaml` | Hydra 实验配置 |

### 修改文件

| 文件 | 修改内容 |
|---|---|
| `src/fact_checking/retrieval/mmr.py` | 新增 `maximal_marginal_relevance_stepwise()` |
| `src/fact_checking/build/candidates.py` | `run_build()` 中新增 `dpo_stepwise` 模式分支 |

## 成功标准与 Stop Criteria

### 成功标准

1. DPO policy 在 test set 上的 accuracy 或 macro-F1 超过 fixed λ=0.7
2. Step-wise λ distribution 有合理结构（如第一步偏高 λ，后续可降低）
3. 高 sensitivity/redundancy bucket 中提升更明显
4. Policy 未坍缩到单一 λ（argmax distribution 中最高项占比 < 80%）

### Stop Criteria

- 若 preference pairs 中 utility gap ≥ δ 的高质量 pair 数量 < N_claims，说明 trajectory 池多样性不够，需增加随机 schedules
- 若 DPO policy 坍缩到 fixed λ=0.7（argmax distribution 中 λ=0.7 占比 > 80%），说明 trajectory preference 信号不足以驱动 policy 偏离 reference
- 若 dev utility 不升反降，检查 reward 是否过度依赖 logprob 尾部值

## 关键设计决策

1. **离散动作空间**：Λ = {0.1, 0.3, 0.5, 0.7, 0.9}，5 个离散值。相比连续 λ，离散动作稳定、适合 DPO loss，且与 MMR 的 λ 敏感性粒度匹配。

2. **Reference policy 固定在 λ=0.7**：将最强的 baseline 作为 reference，迫使 policy 只有在有明确偏好信号时才偏离。

3. **Utility = correct label logprob**：与 oracle λ 计算一致，是 verifier 对正确标签的置信度，连续且可比较。

4. **去重优化**：不同 trajectory 可能产生相同的 evidence set → 去重后只计算一次 utility。恒定 λ trajectory 的 evidence set 直接复用已有 oracle logprob。

5. **State features 包含 pool + step 信息**：pool 特征让 policy 感知候选池整体特性，step 特征让 policy 感知当前选择进度和冗余状态。
