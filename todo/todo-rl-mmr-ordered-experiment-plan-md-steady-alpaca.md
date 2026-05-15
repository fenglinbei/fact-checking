# 实验 5: PAMM-lite / DPO step-wise λ policy — 详细实现文档

## Context

实验 1-4 的结论是: claim-level scalar λ 的自适应信号很弱，fixed λ=0.7 仍是稳定 baseline。实验 5 将学习粒度从 claim-level scalar λ 推进到 **step-wise λ trajectory**，用 evidence set preference 训练一个逐步选择 λ 的 policy，核心假设是 trajectory-level preference learning 比预测单个 hard oracle λ 更稳定。

## 整体流程

```
Chunk-MMR Cache → 生成 Trajectories → 计算 Utility (vLLM) → 构造 Preference Pairs → 训练 DPO Policy → Build Pipeline 集成 → Train → Infer → Metrics
```

五个独立阶段，每阶段产物可缓存复用。

---

## 1. 数据结构定义

### 1.1 `MMRStep` — 单步选择记录

```python
# src/fact_checking/rl_mmr/trajectory.py

@dataclass
class MMRStep:
    step_idx: int           # 0-based, 当前是第几步
    lambda_val: float       # 该步使用的 λ
    selected_idx: int       # 选中的 candidate pool 索引
    hybrid_score: float     # 选中项的 relevance score
    max_sim_to_selected: float  # 选中项与已选集合的最大相似度
    mmr_score: float        # 该步的 MMR 分数
```

### 1.2 `Trajectory` — 完整选择轨迹

```python
@dataclass
class Trajectory:
    event_id: str
    claim: str
    gold_label: str
    steps: list[MMRStep]                    # K 步
    selected_ids: list[int]                 # 最终选中的 candidate 索引列表
    lambda_schedule: list[float]            # [λ_1, ..., λ_K]
    schedule_type: str                      # "fixed" | "handcrafted" | "random"
    utility: float | None = None            # 后续 vLLM 计算填充
    evidence_set_key: str | None = None     # frozenset(selected_ids) 的字符串表示，用于去重
    state_features: list[np.ndarray] | None = None  # 每步的 state feature vector
```

### 1.3 `PreferencePair` — 偏好对

```python
@dataclass
class PreferencePair:
    event_id: str
    traj_win: Trajectory      # τ⁺
    traj_lose: Trajectory     # τ⁻
    utility_gap: float        # U(τ⁺) - U(τ⁻)
    evidence_set_diff: bool   # 两者选择的 evidence set 是否不同
```

---

## 2. Step-wise MMR 核心实现

### 2.1 修改 `src/fact_checking/retrieval/mmr.py`

新增函数 `maximal_marginal_relevance_stepwise()`:

```python
def maximal_marginal_relevance_stepwise(
    query_scores: np.ndarray,       # [N]
    sentence_vectors: np.ndarray,   # [N, D]
    lambda_weights: list[float],    # 长度 = top_k，每步的 λ
) -> tuple[list[int], list[dict]]:
    """
    Returns:
        selected_indices: 最终选中的 K 个索引
        step_records: 每步的详细信息，包含 step_idx, lambda_val, selected_idx,
                      hybrid_score, max_sim_to_selected, mmr_score,
                      candidate_mask_before (该步选择前的可用 mask),
                      mmr_scores_before (该步选择前的 MMR 分数分布)
    """
```

实现逻辑: 与 `maximal_marginal_relevance()` 相同，但 while 循环中每步使用 `lambda_weights[len(selected)]`，并记录每步的中间状态。

### 2.2 新增 `src/fact_checking/rl_mmr/step_features.py`

提取每步的 state features，供 policy 使用:

```python
# 特征维度: POOL_FEATURES (16) + STEP_FEATURES (12) = 28

POOL_FEATURE_NAMES: list[str] = [
    "n_candidates", "log_n_candidates",
    "score_mean", "score_std", "score_entropy",
    "top1_top2_gap", "mean_pairwise_sim", "max_pairwise_sim",
    # ... (复用 soft_label_features.py 的 POOL_FEATURE_NAMES)
]

STEP_FEATURE_NAMES: list[str] = [
    "step_fraction",         # t / K
    "n_already_selected",    # t
    "mean_rel_selected",     # 已选中的平均 relevance
    "mean_red_selected",     # 已选中的平均 pairwise similarity
    "max_red_in_pool",       # 当前候选池中 max_sim_to_selected 的最大值
    "mean_red_in_pool",      # 当前候选池中 max_sim_to_selected 的平均值
    "last_selected_rel",     # 上一步选中项的 relevance
    "last_selected_red",     # 上一步选中项的 max_sim_to_selected
    "remaining_score_entropy",  # 剩余候选的 score entropy
    "remaining_n",           # 剩余候选数
    "top_mmr_score",         # 当前步最高 MMR 分数
    "mmr_score_gap",         # 当前步 top1 与 top2 MMR 分数差
]

def extract_step_features(
    scores: np.ndarray,          # hybrid_scores [N]
    sim: np.ndarray,             # pairwise similarity [N, N]
    selected_indices: list[int], # 已选中的索引
    candidate_mask: np.ndarray,  # 当前可用的候选 mask
    step_idx: int,
    total_steps: int,
) -> np.ndarray:
    """返回当前步的 state feature vector, shape [28]."""
```

**复用已有代码**: `_entropy()`, `_gini()` 从 `soft_label_features.py` import; `mean_pairwise_sim()` 从 `sensitivity.py` import。

---

## 3. Trajectory 生成 (`scripts/rl_mmr/generate_trajectories.py`)

### 3.1 输入

- Chunk-MMR cache pickle 文件（train/val/test）
- 配置: `top_k=5`, λ schedules, random seed, 每 claim 随机 trajectory 数量

### 3.2 λ Schedules

**手工 schedules (7 个):**
```python
HANDCRAFTED_SCHEDULES = [
    [0.7, 0.7, 0.7, 0.7, 0.7],   # baseline
    [0.3, 0.3, 0.3, 0.3, 0.3],   # constant low
    [0.5, 0.5, 0.5, 0.5, 0.5],   # constant mid
    [0.9, 0.7, 0.5, 0.3, 0.3],   # decreasing
    [1.0, 0.7, 0.5, 0.3, 0.1],   # steep decreasing
    [0.5, 0.5, 0.7, 0.7, 0.9],   # increasing
    [0.7, 0.5, 0.3, 0.5, 0.7],   # U-shaped
]
```

**随机 schedules:** 每条 claim 生成 N_random 条（建议 30 条），每步 λ_t ~ Uniform({0.1, 0.3, 0.5, 0.7, 0.9})

### 3.3 处理流程

对每个 ChunkMMRSample:
1. 调用 `compute_hybrid_scores()` 获取 hybrid_scores 和 chunk_emb
2. 对每个 schedule，调用 `maximal_marginal_relevance_stepwise()` 执行逐步 MMR
3. 记录每步的 state features（调用 `extract_step_features()`）
4. 生成 `Trajectory` 对象

### 3.4 输出

```text
outputs/rl_mmr/dpo_stepwise/trajectories/trajectories_train.jsonl
outputs/rl_mmr/dpo_stepwise/trajectories/trajectories_val.jsonl
outputs/rl_mmr/dpo_stepwise/trajectories/trajectories_test.jsonl
```

每行 JSON:
```json
{
  "event_id": "...",
  "claim": "...",
  "gold_label": "...",
  "schedule_type": "handcrafted",
  "schedule_id": 0,
  "lambda_schedule": [0.7, 0.7, 0.7, 0.7, 0.7],
  "selected_ids": [3, 7, 1, 12, 5],
  "selected_texts": ["text1", "text2", ...],
  "evidence_set_key": "1_3_5_7_12",
  "steps": [
    {"step_idx": 0, "lambda_val": 0.7, "selected_idx": 3, ...},
    ...
  ],
  "state_features": [[...], [...], [...], [...], [...]]
}
```

### 3.5 命令行参数

```bash
PYTHONPATH=src python scripts/rl_mmr/generate_trajectories.py \
    --chunk-mmr-cache outputs/cache/chunk_mmr/<fp>/chunk_mmr_train.pkl \
    --output-dir outputs/rl_mmr/dpo_stepwise/trajectories \
    --top-k 5 \
    --n-random 30 \
    --seed 42
```

---

## 4. Utility 计算 (`scripts/rl_mmr/compute_trajectory_utility.py`)

### 4.1 目的

对 trajectory 池中的每个 unique evidence set，用 vLLM verifier 计算 correct label logprob 作为 utility。

### 4.2 流程

1. 加载 trajectory JSONL，收集所有 unique `(event_id, evidence_set_key)` 对
2. 对每个 unique evidence set，构建 prompt（复用 `_build_training_row()` 的 prompt 构建逻辑，但从 trajectory 的 selected_texts 构造 evidence block）
3. 调用 vLLM 计算 correct label token 的 logprob（复用 `compute_oracle_lambda.py` 的 `vllm_prompt` scoring 逻辑，只计算 gold label 对应的那个 token 的 prompt logprob）
4. 将 utility 映射回所有共享同一 evidence set 的 trajectory

### 4.3 优化

- 多个 trajectory 可能产生相同 evidence set → 去重后只需计算一次
- 预期 unique evidence set 数量: N_claims × (7 handcrafted + ~30 random) ≈ 37 per claim，去重后可能 10-20 unique per claim
- 使用与 `compute_oracle_lambda.py` 相同的 vLLM scoring 接口

### 4.4 输出

在原始 trajectory JSONL 基础上增加 `utility` 字段，另存为新文件:

```text
outputs/rl_mmr/dpo_stepwise/trajectories/trajectories_train_scored.jsonl
```

---

## 5. Preference Pair 构造 (`scripts/rl_mmr/build_preference_pairs.py`)

### 5.1 流程

1. 加载 scored trajectory JSONL
2. 对每条 claim，将 trajectory 按 utility 降序排列
3. 对 utility gap ≥ δ 的 trajectory 对，构造 `PreferencePair`
4. 过滤和优先级:
   - 优先保留 evidence_set_diff=True 的 pair
   - 优先保留 utility gap 大的 pair
   - 每 claim 最多 max_pairs_per_claim=10 个 pair
5. 对每个 pair，提取 winner 和 loser 的 step-wise state features 和 λ choices

### 5.2 超参数

```python
DELTA = 0.05          # 最小 utility gap
MAX_PAIRS_PER_CLAIM = 10
```

### 5.3 输出格式

```text
outputs/rl_mmr/dpo_stepwise/preference_pairs/train_pairs.npz
```

NPZ 内容:
- `win_features`: shape [N_pairs, K, D_state] — winner 每步的 state features
- `win_lambdas`: shape [N_pairs, K] — winner 每步的 λ index (0-4)
- `lose_features`: shape [N_pairs, K, D_state]
- `lose_lambdas`: shape [N_pairs, K]
- `utility_gaps`: shape [N_pairs]
- `event_ids`: shape [N_pairs]

### 5.4 统计输出

同时输出 `pair_statistics.json`:
```json
{
  "n_claims_total": 1024,
  "n_claims_with_pairs": 800,
  "n_pairs_total": 6500,
  "utility_gap_mean": 0.12,
  "utility_gap_std": 0.08,
  "evidence_set_diff_fraction": 0.75
}
```

### 5.5 命令行

```bash
PYTHONPATH=src python scripts/rl_mmr/build_preference_pairs.py \
    --trajectories outputs/rl_mmr/dpo_stepwise/trajectories/trajectories_train_scored.jsonl \
    --output-dir outputs/rl_mmr/dpo_stepwise/preference_pairs \
    --delta 0.05 \
    --max-pairs-per-claim 10
```

---

## 6. DPO Policy 模型 (`src/fact_checking/rl_mmr/dpo_policy.py`)

### 6.1 模型架构

```python
class StepLambdaPolicy(nn.Module):
    """MLP policy: state_features → 5-class logits over λ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}"""
    
    def __init__(self, input_dim: int = 28, hidden_dims: list[int] = [64, 32], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 5))  # 5 个离散 λ 值
        self.net = nn.Sequential(*layers)
    
    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """state_features: [B, D] → logits: [B, 5]"""
        return self.net(state_features)
    
    def log_prob(self, state_features: torch.Tensor, lambda_idx: torch.Tensor) -> torch.Tensor:
        """计算 log π_θ(λ_idx | state_features) → [B]"""
        logits = self.forward(state_features)
        return F.log_softmax(logits, dim=-1).gather(1, lambda_idx.unsqueeze(-1)).squeeze(-1)
```

### 6.2 DPO Loss

```python
def dpo_loss(
    policy: StepLambdaPolicy,
    ref_policy: StepLambdaPolicy,
    win_features: torch.Tensor,    # [B, K, D]
    win_lambdas: torch.Tensor,     # [B, K]  λ index (0-4)
    lose_features: torch.Tensor,   # [B, K, D]
    lose_lambdas: torch.Tensor,    # [B, K]
    beta: float = 1.0,
) -> torch.Tensor:
    """
    L_DPO = -log σ(β * [(log π_θ(τ⁺) - log π_θ(τ⁻)) - (log π_ref(τ⁺) - log π_ref(τ⁻))])
    
    其中 log π(τ) = Σ_{t=1}^{K} log π(λ_t | s_t)
    """
    B, K, D = win_features.shape
    
    # Flatten steps → [B*K, D]
    wf = win_features.reshape(B * K, D)
    lf = lose_features.reshape(B * K, D)
    wl = win_lambdas.reshape(B * K)
    ll = lose_lambdas.reshape(B * K)
    
    # Per-step log probs → [B*K]
    logp_win = policy.log_prob(wf, wl)    # π_θ
    logp_lose = policy.log_prob(lf, ll)
    logp_win_ref = ref_policy.log_prob(wf, wl)  # π_ref
    logp_lose_ref = ref_policy.log_prob(lf, ll)
    
    # Sum over steps → [B]
    logp_win_sum = logp_win.reshape(B, K).sum(dim=-1)
    logp_lose_sum = logp_lose.reshape(B, K).sum(dim=-1)
    logp_win_ref_sum = logp_win_ref.reshape(B, K).sum(dim=-1)
    logp_lose_ref_sum = logp_lose_ref.reshape(B, K).sum(dim=-1)
    
    # DPO objective
    log_ratio = (logp_win_sum - logp_lose_sum) - (logp_win_ref_sum - logp_lose_ref_sum)
    loss = -F.logsigmoid(beta * log_ratio).mean()
    return loss
```

### 6.3 Reference Policy

Reference policy 实现为 fixed λ=0.7 的"软"版本:

```python
def make_reference_policy(lambda_grid: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9], 
                          center: float = 0.7, temperature: float = 0.3):
    """返回一个 frozen StepLambdaPolicy，其输出偏向 center lambda."""
    # 使用固定权重使得 π_ref(λ) ∝ exp(-|λ - center| / temperature)
    # 实际上是一个查表式 policy
    
class FixedReferencePolicy(nn.Module):
    def __init__(self, lambda_grid, center=0.7, temperature=0.3):
        super().__init__()
        weights = np.array([-abs(lam - center) / temperature for lam in lambda_grid])
        self.register_buffer("logits", torch.from_numpy(weights).float())
    
    def forward(self, x):
        return self.logits.unsqueeze(0).expand(x.shape[0], -1)
```

---

## 7. DPO 训练 (`scripts/rl_mmr/train_dpo_step_lambda.py`)

### 7.1 训练流程

1. 加载 `train_pairs.npz` 和 `val_pairs.npz`
2. 初始化 `StepLambdaPolicy`（policy model）和 `FixedReferencePolicy`（reference）
3. 使用 DataLoader 按 batch 加载 preference pairs
4. 优化 DPO loss
5. 在每个 eval epoch 计算:
   - val DPO loss
   - policy entropy（防止坍缩）
   - argmax λ distribution（检查是否坍缩到单一 λ）
6. Early stopping on val loss
7. 保存最佳 checkpoint

### 7.2 超参数

```python
BETA = 1.0              # DPO temperature
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EPOCHS = 200
PATIENCE = 25
HIDDEN_DIMS = [64, 32]
DROPOUT = 0.1
```

### 7.3 监控指标

- DPO loss
- Policy entropy: H(π_θ) = -Σ_λ π_θ(λ|s) log π_θ(λ|s)，按步分桶
- argmax distribution: 每步各类 λ 被选中的比例
- Accuracy improvement: 在 val 的 preference pairs 上，policy 分配更高概率给 winner 的比例

### 7.4 输出

```text
outputs/rl_mmr/dpo_stepwise/checkpoints/model_best.pt
outputs/rl_mmr/dpo_stepwise/checkpoints/feature_stats.json
outputs/rl_mmr/dpo_stepwise/checkpoints/training_metrics.json
```

### 7.5 命令行

```bash
PYTHONPATH=src python scripts/rl_mmr/train_dpo_step_lambda.py \
    --train-pairs outputs/rl_mmr/dpo_stepwise/preference_pairs/train_pairs.npz \
    --val-pairs outputs/rl_mmr/dpo_stepwise/preference_pairs/val_pairs.npz \
    --output-dir outputs/rl_mmr/dpo_stepwise/checkpoints \
    --beta 1.0 \
    --lr 1e-3 \
    --epochs 200 \
    --batch-size 64
```

---

## 8. Build Pipeline 集成 (`src/fact_checking/rl_mmr/dpo_selector.py`)

### 8.1 设计思路

DPO policy 需要在 build 阶段逐步运行 MMR，每步调用 policy 选择 λ。这比 claim-level scalar λ override 复杂。采用方案:

**新增 `learned_lambda_mode = "dpo_stepwise"`**，在该模式下不走 `_mmr_phase_from_chunk_cache`，而是走专用的 DPO step-wise 选择路径。

### 8.2 核心函数

```python
def select_candidates_dpo_stepwise(
    sample: ChunkMMRSample,
    policy: StepLambdaPolicy,
    feature_stats: dict,
    lambda_grid: np.ndarray,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
) -> dict[str, Any]:
    """
    对单个样本运行 step-wise DPO MMR:
    1. 计算 hybrid_scores 和 chunk_emb (复用 compute_hybrid_scores)
    2. 计算 pairwise similarity matrix
    3. 循环 K 步:
       a. 提取当前 state features (调用 extract_step_features)
       b. 标准化特征 (用 feature_stats)
       c. 调用 policy 预测 π_θ(λ | state) → argmax 选择 λ_t
       d. 计算 MMR scores = λ_t * relevance - (1-λ_t) * max_sim_to_selected
       e. 选择得分最高的 candidate
    4. 返回 selected candidates（与 _select_candidates_from_chunk_sample 相同格式）
    """
```

### 8.3 Build Pipeline 修改 (`src/fact_checking/build/candidates.py`)

在 `run_build()` 的 Phase 3 部分（约 line 1481），新增 `learned_lambda_mode == "dpo_stepwise"` 分支:

```python
elif learned_lambda_mode == "dpo_stepwise":
    from fact_checking.rl_mmr.dpo_selector import (
        load_dpo_step_policy,
        run_dpo_stepwise_selection,
    )
    policy, feature_stats = load_dpo_step_policy(learned_lambda_cfg["model_path"])
    lambda_grid = np.array(learned_lambda_cfg.get("lambda_grid", [0.1, 0.3, 0.5, 0.7, 0.9]))
    
    # 直接生成 training rows，不走 _mmr_phase_from_chunk_cache
    with output_path.open("w") as writer:
        for sample in tqdm(chunk_samples, ...):
            row = select_candidates_dpo_stepwise(
                sample, policy, feature_stats, lambda_grid,
                top_k=run_summary["top_k"],
                alpha_dense=..., alpha_lexical=..., alpha_bm25=...,
            )
            training_row = _build_training_row(row, tokenizer, prompt_cfg_local)
            writer.write(json.dumps(training_row, ensure_ascii=False) + "\n")
```

### 8.4 配置

配置继承自 `b3_mmr_topk_sweep_1024`，保持相同的 data paths、chunking 策略、verifier 模型、训练配置、推理配置。
仅覆写 build retrieval 的 `top_k`、`mmr_lambda` 和 `learned_lambda` 块。

```yaml
# configs/experiment/mmr_dpo_step_lambda.yaml
# @package _global_
# 实验 5: DPO step-wise λ policy — 完整 pipeline 配置
# 继承自 b3_mmr_topk_sweep_1024，仅覆写 learned_lambda 相关参数

defaults:
  - b3_mmr_topk_sweep_1024   # 继承完整的 b3 baseline 参数
  - _self_

# ============================================================================
# 实验元数据（覆写）
# ============================================================================
experiment:
  name: mmr_dpo_step_lambda

baseline:
  variant: mmr_dpo_step_lambda
  chunking_strategy: semantic
  model_name_or_path: /data/models/Qwen2.5-7B-Instruct
  top_k: 5                     # evidence selection 数量，与 fixed λ=0.7 baseline 保持一致

# ============================================================================
# Build 配置
# ============================================================================
build:
  # --- 数据路径（与 b3 相同，显式列出） ---
  output_policy: fingerprint_cache
  data:
    train_path: data/raw/LIAR-RAW/train.json
    val_path: data/raw/LIAR-RAW/val.json
    test_path: data/raw/LIAR-RAW/test.json

  # --- 检索配置 ---
  retrieval:
    embedder_model: /data/models/bge-base-en-v1.5/
    device: cuda
    max_length: 256
    batch_size: 64
    top_k: 32                   # candidate pool 大小（统一固定 32）
    alpha_dense: 0.70
    alpha_lexical: 0.20
    alpha_bm25: 0.10
    mmr_lambda: 0.70           # 默认 MMR λ（仅当 DPO policy 不可用时回退）
    precision: bf16
    num_gpus: 4
    prefetch_size: 200
    cpu_workers: 4
    selection_method: mmr       # DPO 模式走自定义 step-wise selection，但仍标记为 mmr

    # --- Chunking 配置（与 b3 相同） ---
    chunking:
      strategy: semantic
      context_k: 1
      theta: 0.5
      embedder_model: /data/models/bge-base-en-v1.5/
      device: cuda
      max_length: 256
      batch_size: 64
      precision: bf16

    # ##### DPO step-wise λ policy 配置（实验 5 核心参数） #####
    learned_lambda:
      enabled: true
      mode: dpo_stepwise
      dpo_stepwise:
        model_path: outputs/rl_mmr/dpo_stepwise/checkpoints
        lambda_grid: [0.1, 0.3, 0.5, 0.7, 0.9]
        inference_mode: argmax     # argmax | sample
        dump_trace: true

  # --- Prompt 配置（与 b3 相同） ---
  prompt:
    model_name_or_path: /data/models/Qwen2.5-7B-Instruct
    auto_length: true
    max_length: 1024
    output_mode: label_only
    label_format: letter
    system_prompt: null

# ============================================================================
# Train 配置（与 b3 完全相同，显式列出确保可复现）
# ============================================================================
train:
  model_name_or_path: /data/models/Qwen2.5-7B-Instruct
  backend: accelerate_deepspeed
  cuda_visible_devices: "0,1,2,3"
  nproc_per_node: 4
  num_machines: 1
  mixed_precision: bf16
  deepspeed_config: configs/deepspeed_zero2_bsz8_ga1.json
  checkpoint_for_infer: best
  run_dir: null

sft_train:
  max_new_tokens: 8
  temperature: 0.0
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 1
  learning_rate: 1.0e-5
  num_train_epochs: 2
  weight_decay: 0.0
  warmup_ratio: 0.03
  bf16: true
  max_length: 1024
  logging_steps: 2
  save_steps: 50
  eval_steps: 50
  dataloader_num_workers: 4
  gradient_checkpointing: true
  use_flash_attention_2: true
  lr_scheduler_type: cosine
  max_grad_norm: 1.0
  padding: longest
  use_length_bucket: true
  empty_cache_steps: 0
  empty_cache_on_eval: true
  empty_cache_on_save: true
  early_stopping_patience: 3
  # ... (lora, logit_adjust 等子块与 b3 完全相同) ...
  lora:
    enabled: true
    r: 16
    alpha: 32
    dropout: 0.05
    bias: none
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
    modules_to_save: null
  logit_adjust:
    enabled: true
    tau: 1.0

# ============================================================================
# Infer 配置（与 b3 完全相同，显式列出确保可复现）
# ============================================================================
infer:
  config_path: null
  provider: vllm_openai
  split: test
  checkpoint: best
  served_model_name: fact-checking-sft
  host: 127.0.0.1
  port: 35001
  base_url: null
  wait_seconds: 180
  request_timeout_seconds: 120
  log_predictions: 5
  max_new_tokens: 8
  temperature: 0.0
  cuda_visible_devices: "0,1,2,3"
  tensor_parallel_size: 4
  gpu_memory_utilization: 0.90
  dtype: auto
  max_model_len: null
  top_p: 1.0
  presence_penalty: 0.0
  frequency_penalty: 0.0
  repetition_penalty: 1.0
  label_decoding:
    enabled: true
    prefix: "Label:"
    guided_choice: true
    max_tokens: 1
  merge_lora_cache:
    enabled: true
    dir: outputs/cache/merged_lora
    force_rebuild: false
  server:
    manage: true
    stop_after_infer: true
    pid_file: null
    extra_args: []

# ============================================================================
# 跟踪配置（覆写实验名称）
# ============================================================================
tracking:
  enabled: true
  backend: swanlab

swanlab:
  project: fact-checking-dpo-stepwise
  experiment_name: mmr_dpo_step_lambda
```

---

## 9. 评估 (`scripts/rl_mmr/evaluate_dpo_step_lambda.py`)

### 9.1 两种评估模式

**模式 A: 离线 utility 评估（快速）**
- 在 trajectory 生成时已计算 utility
- 直接对比 DPO policy 选择的 evidence set 的 utility vs fixed λ=0.7
- 无需重新训练 verifier
- 适合快速迭代

**模式 B: 完整 build → train → infer 评估（正式）**
- 通过 Hydra 配置运行完整 pipeline
- 使用 `configs/experiment/mmr_dpo_step_lambda.yaml`
- 对比 test metrics: accuracy, macro-F1, redundancy, cost
- 这是正式评估

### 9.2 评估脚本功能

1. 加载训练好的 DPO policy
2. 对 test set Chunk-MMR cache 运行 step-wise selection
3. 统计:
   - 每步 λ distribution
   - 与 fixed λ=0.7 的 evidence overlap
   - 分桶 utility（按 sensitivity, pool redundancy, candidate count）
4. 输出分析报告

### 9.3 命令行

```bash
# 离线评估
PYTHONPATH=src python scripts/rl_mmr/evaluate_dpo_step_lambda.py \
    --policy outputs/rl_mmr/dpo_stepwise/checkpoints \
    --chunk-mmr-cache outputs/cache/chunk_mmr/<fp>/chunk_mmr_test.pkl \
    --oracle-logprobs outputs/learned_lambda/oracle_lambda_test.jsonl \
    --output-dir outputs/rl_mmr/dpo_stepwise/eval

# 完整 pipeline 评估
PYTHONPATH=src python -m fact_checking.pipeline.run \
    experiment=mmr_dpo_step_lambda pipeline.mode=full
```

---

## 10. 文件清单

### 新增文件

| 文件 | 行数估计 | 说明 |
|---|---|---|
| `src/fact_checking/rl_mmr/trajectory.py` | ~120 | `MMRStep`, `Trajectory`, `PreferencePair` dataclasses + step-wise MMR runner |
| `src/fact_checking/rl_mmr/step_features.py` | ~180 | `extract_step_features()`, `POOL_FEATURE_NAMES`, `STEP_FEATURE_NAMES` |
| `src/fact_checking/rl_mmr/dpo_policy.py` | ~200 | `StepLambdaPolicy`, `FixedReferencePolicy`, `dpo_loss()`, training utilities |
| `src/fact_checking/rl_mmr/dpo_selector.py` | ~200 | `load_dpo_step_policy()`, `select_candidates_dpo_stepwise()`, build pipeline 集成 |
| `scripts/rl_mmr/generate_trajectories.py` | ~250 | Trajectory 生成主脚本 |
| `scripts/rl_mmr/compute_trajectory_utility.py` | ~200 | vLLM utility 计算脚本（复用 compute_oracle_lambda.py 的 scoring 逻辑） |
| `scripts/rl_mmr/build_preference_pairs.py` | ~200 | Preference pair 构造脚本 |
| `scripts/rl_mmr/train_dpo_step_lambda.py` | ~350 | DPO 训练脚本 |
| `scripts/rl_mmr/evaluate_dpo_step_lambda.py` | ~300 | 评估脚本 |
| `configs/experiment/mmr_dpo_step_lambda.yaml` | ~30 | Hydra 实验配置 |

### 修改文件

| 文件 | 修改内容 |
|---|---|
| `src/fact_checking/retrieval/mmr.py` | 新增 `maximal_marginal_relevance_stepwise()` |
| `src/fact_checking/build/candidates.py` | 在 `run_build()` 中新增 `dpo_stepwise` 模式分支（约 40 行） |
| `src/fact_checking/rl_mmr/__init__.py` | 按需更新（可能不需要修改） |

---

## 11. 复用现有代码

| 现有模块 | 复用场景 |
|---|---|
| `fact_checking.retrieval.mmr.maximal_marginal_relevance` | step-wise 版本的基础逻辑 |
| `fact_checking.build.candidates.compute_hybrid_scores` | Trajectory 生成和 DPO 选择时的 scoring |
| `fact_checking.build.candidates.ChunkMMRSample` | 数据类型 |
| `fact_checking.build.candidates._build_training_row` | DPO 选择完成后构造训练行 |
| `fact_checking.rl_mmr.sensitivity.{compute_pairwise_sim,mean_pairwise_sim}` | Step features 中的 redundancy 计算 |
| `fact_checking.rl_mmr.soft_label_features.{_entropy,_gini,POOL_FEATURE_NAMES}` | Pool 统计特征 |
| `fact_checking.rl_mmr.soft_label_selector.SoftLabelMLP` | MLP 架构参考（`StepLambdaPolicy` 类似） |
| `fact_checking.learned_lambda.cache_utils.{load_experiment_build_cfg,resolve_chunk_mmr_cache_path}` | 解析 build config 和 cache 路径 |
| `scripts/learned_lambda/compute_oracle_lambda.py` 中的 vLLM scoring 逻辑 | Utility 计算 |
| `fact_checking.data.constants.{LABEL_LETTERS,LETTER_ORDER}` | Label token 映射 |

---

## 12. 成功标准与 Stop Criteria

### 成功标准
1. DPO policy 在 test set 上的 accuracy 或 macro-F1 超过 fixed λ=0.7
2. Step-wise λ distribution 有合理结构（如第一步偏高 λ，后续降低）
3. 高 sensitivity/redundancy bucket 中提升更明显
4. Policy 未坍缩到单一 λ

### Stop Criteria（来自实验计划 §15）
- 若 preference pairs 中 utility gap ≥ δ 的高质量 pair 数量不足（< N_claims），说明 trajectory 池多样性不够，需先增加随机 schedules
- 若 DPO policy 坍缩到 fixed λ=0.7（argmax distribution 中 λ=0.7 占比 > 80%），则说明 trajectory preference 信号不足以驱动 policy 偏离 reference
- 若 dev utility 不升反降，检查 reward 是否过度依赖 logprob 尾部值

---

## 13. 实现顺序建议

1. **Phase A** (基础组件): `mmr.py` step-wise 扩展 + `trajectory.py` dataclasses + `step_features.py`
2. **Phase B** (数据准备): `generate_trajectories.py` → `compute_trajectory_utility.py` → `build_preference_pairs.py`
3. **Phase C** (训练): `dpo_policy.py` + `train_dpo_step_lambda.py`
4. **Phase D** (集成): `dpo_selector.py` + `candidates.py` 修改 + `configs/experiment/mmr_dpo_step_lambda.yaml`
5. **Phase E** (评估): `evaluate_dpo_step_lambda.py` + 完整 pipeline 运行

每个 Phase 完成后验证产物存在且格式正确再进入下一 Phase。
