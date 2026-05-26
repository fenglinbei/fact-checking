# Sensitivity-Gated MMR 实验设计与实现

## 动机

固定 λ 的 MMR（Maximal Marginal Relevance）对所有样本一视同仁，但不同样本对 λ 的敏感度不同：

- **低冗余样本**：候选证据彼此差异大、重复少 → 用低 λ 即可，强去冗余反而丢失信息
- **高冗余样本**：候选证据高度相似、信息重叠 → 需要高 λ 强力去冗余

因此提出按样本动态选择 λ 的策略：对每个 claim 计算"敏感度特征"，门控决定使用 `λ_low` 还是 `λ_base`。

## 整体架构

```
Stage A（离线搜索）                    Stage B（上线流水线）
─────────────────                     ─────────────────
固定 checkpoint (λ=0.7 训练)           读取 Stage A 最优参数
  │                                     │
  ├─ λ=0.2 → build → infer              ├─ Build（实时门控选 λ）
  ├─ λ=0.3 → build → infer              ├─ Train（SFT 新模型）
  ├─ λ=0.4 → build → infer              └─ Infer（vLLM 推理）
  ├─ ...                               ═══════════════════
  └─ λ=0.8 → build → infer             最终模型 + 评估指标
  │
  └─ 网格搜索 (θ_s, θ_r, λ_low,
      gating_mode, ε) → 最优参数
```

## Stage A：离线网格搜索

**入口**：`scripts/rl_mmr/run_stage_a_search.sh`

**目的**：不重新训练模型，在 val 集上穷举搜索最优门控参数。

### 流程

#### 第一步：逐 λ 收集预测结果

对 λ 网格（默认 `0.2 0.3 0.4 0.5 0.6 0.7 0.8`）中的每个值：

1. **Build** — 用该 λ 运行证据检索 + MMR 重排序，输出 `build_val.jsonl`
2. **Infer** — 复用固定 checkpoint（λ=0.7 训练的模型），推理 val 集，输出 `val_predictions.jsonl`

这一步只改变 MMR λ，模型权重不变。不同 λ 产出不同的证据排序 → 不同的 prompt → 不同的预测结果。

第一个 λ 启动 vLLM server 后保持运行，后续 λ 复用同一个 server。

#### 第二步：网格搜索门控参数

**入口**：`scripts/rl_mmr/search_sensitivity_thresholds.py`

对每个 `(θ_s, θ_r, λ_low, gating_mode, ε)` 组合：

1. 从缓存的 `chunk_mmr.pkl` 读取每个样本的候选证据和 embedding
2. 计算敏感度特征（`sensitivity_features`）
3. 执行门控决策（`gating_decision`），决定每个样本应使用的 λ
4. 查表：用该样本在选定 λ 下的预测结果计算 accuracy / macro_f1
5. 按 `score = w_acc × accuracy + w_f1 × macro_f1` 排序

##### 搜索空间

| 参数 | 候选值 | 说明 |
|------|--------|------|
| `theta_s` | 0.2, 0.4, 0.6, 0.8 | 敏感度阈值 |
| `theta_r` | 0.3, 0.4, 0.5, 0.6 | 冗余度阈值 |
| `lambda_low` | 0.2, 0.3, 0.4 | 低 λ 候选值 |
| `gating_mode` | basic, conservative | 门控策略 |
| `epsilon` | 0.02, 0.05, 0.10 | conservative 模式相关性容差 |

注意：`epsilon` 仅在 `gating_mode=conservative` 时参与搜索；`basic` 模式下 `epsilon` 为 `None`。

##### 输出

- `dev_grid.csv` — 所有网格单元按 score 降序排列
- `dev_grid.json` — 包含固定 baseline、各 λ baseline、top-K 最佳配置的完整报告

### 关键设计决策

- **不重新训练**：复用 λ=0.7 的 checkpoint，仅改变证据排序
- **查表式计算**：预计算各 λ 下的预测结果，搜索时只需查表（O(1)），无需反复推理
- **幂等**：build / infer 阶段命中缓存即跳过

## Stage B：完整门控流水线

**入口**：`scripts/rl_mmr/run_sensitivity_gated.sh`

**目的**：用 Stage A 搜出的最优参数，运行完整 build → train → infer 流水线。

### 参数加载

启动时自动读取 `outputs/rl_mmr/sensitivity_search/dev_grid.csv` 第一行（score 最高），提取参数并转为 Hydra CLI overrides：

```
build.retrieval.learned_lambda.sensitivity.theta_s=<best>
build.retrieval.learned_lambda.sensitivity.theta_r=<best>
build.retrieval.learned_lambda.sensitivity.lambda_low=<best>
build.retrieval.learned_lambda.sensitivity.gating_mode='<best>'
build.retrieval.learned_lambda.sensitivity.relevance_floor.epsilon=<best>   # conservative only
```

### Build 阶段（核心差异）

与常规 MMR 不同，Build 阶段对每个 claim **实时计算**门控 λ。逻辑在 `src/fact_checking/rl_mmr/gated_selector.py`。

详见下方 [门控机制详解](#门控机制详解)。

### Train / Infer

与常规流水线一致：用门控 MMR 产出的证据构建 prompt，SFT 训练 Qwen2.5-7B-Instruct，vLLM 推理。

## 门控机制详解

**核心文件**：
- `src/fact_checking/rl_mmr/sensitivity.py` — 特征提取 + 门控决策（纯函数，无状态）
- `src/fact_checking/rl_mmr/gated_selector.py` — 批量处理 + 输出 trace

### 第一步：特征提取（`sensitivity_features`）

对每个 claim 的候选证据集（N 个句子 + embedding）：

1. 用三个不同的 λ 值分别执行 MMR 选择 top-k：
   - `S_low` = MMR(λ_low) 的结果
   - `S_base` = MMR(λ_base) 的结果
   - `S_probe` = MMR(λ_probe) 的结果（仅在敏感度计算中使用）

2. 计算核心敏感度指标：
   - `sens_low_base` = 1 - Jaccard(S_low, S_base)
     - Jaccard 高 → 两个 λ 选出的集合重叠大 → 低敏感
     - Jaccard 低 → λ 变化显著影响结果 → 高敏感

3. 计算池冗余度：
   - `pool_redundancy` = 候选池 Top-N 句子的平均 pairwise cosine similarity
     - 高 → 候选证据彼此相似，冗余严重

4. 辅助特征：Kendall τ 排序相关性、top1 是否变化、分数熵、top1-top2 差距等

### 第二步：门控决策（`gating_decision`）

```
输入：样本特征 feat、hybrid_scores、阈值参数
输出：(chosen_λ, gate_label, extras)
```

决策树：

```
n < min_n_candidates_for_gate?
  ├─ Yes → GATE_TRIVIAL, λ_base     # 候选太少，不冒险
  └─ No  → sens >= θ_s AND pool_redundancy >= θ_r?
              ├─ No  → GATE_BASE, λ_base    # 不敏感或低冗余，切换无益
              └─ Yes → gating_mode?
                          ├─ basic       → GATE_LOW, λ_low     # 直接切换
                          └─ conservative → relevance_floor_ok?
                                              ├─ Yes → GATE_LOW, λ_low
                                              └─ No  → GATE_FLOOR_BLOCKED, λ_base
```

**相关性地板检查（`relevance_floor_ok`）**仅 conservative 模式触发：

- `mean_delta` 模式：`mean_rel(S_base) - mean_rel(S_low) <= ε`
  - 切换到低 λ 后，证据平均相关性下降不超过 ε
- `min_quantile` 模式：`min(Rel(S_low)) >= percentile(H_pool, p_floor × 100)`
  - 低 λ 选出的最差证据也不低于池中 p_floor 分位数

### 门控类型统计

| Gate | 含义 | λ 选择 |
|------|------|--------|
| `trivial_pool` | 候选太少 (< min_n) | λ_base |
| `base_lambda` | 不满足切换条件 | λ_base |
| `low_lambda` | 通过所有条件 | λ_low |
| `relevance_floor_blocked` | 被相关性检查阻止 | λ_base |

### 第三步：执行 MMR 并输出

每个 claim 得到专属 λ 后，正常走 MMR → prompt 构建 → `build_*.jsonl` 输出。

同时输出 trace JSONL（每个样本的特征、门控结果、选中的 λ），用于后续分析。

## 配置文件

**Stage B 配置**：`configs/experiment/mmr_sensitivity_gated.yaml`

关键配置路径：

```yaml
build:
  retrieval:
    top_k: 5
    mmr_lambda: 0.70                    # 无 override 时的兜底
    learned_lambda:
      enabled: true
      mode: sensitivity_gated
      sensitivity:
        lambda_low: 0.30                # 门控选中的低 λ
        lambda_base: 0.70               # 默认 λ（训练所用）
        lambda_probe: 1.00              # 仅用于特征计算的探测 λ
        theta_s: 0.40                   # 敏感度阈值
        theta_r: 0.40                   # 冗余度阈值
        gating_mode: conservative       # basic | conservative
        relevance_floor:
          mode: mean_delta              # mean_delta | min_quantile
          epsilon: 0.05
          p_floor: 0.50
        pool_redundancy_topn: 32
        min_n_candidates_for_gate: 2
        dump_trace: true
```

注意：`theta_s`、`theta_r`、`lambda_low`、`gating_mode`、`epsilon` 会被 Stage B 脚本自动覆盖为 Stage A 的最优值。

## 运行方式

```bash
# Stage A — 离线搜索最优参数（复用已有 checkpoint）
bash scripts/rl_mmr/run_stage_a_search.sh \
    <FIXED_TRAIN_DIR>     # e.g. outputs/runs/b3_mmr_topk_sweep_1024/.../train
    "0.2 0.3 0.4 0.5 0.6 0.7 0.8"   # λ 网格
    val                   # split
    mmr_sensitivity_gated # base experiment
    outputs/rl_mmr/sensitivity_search  # 输出目录
    10909                 # vLLM 端口

# Stage B — 用 Stage A 最优参数跑完整流水线
bash scripts/rl_mmr/run_sensitivity_gated.sh \
    outputs/rl_mmr/sensitivity_search  # Stage A 输出目录
    mmr_sensitivity_gated              # experiment
    full                               # mode (build | train | infer | full)
```

## 文件索引

| 文件 | 作用 |
|------|------|
| `scripts/rl_mmr/run_stage_a_search.sh` | Stage A 入口：逐 λ build+infer + 阈值搜索 |
| `scripts/rl_mmr/search_sensitivity_thresholds.py` | Stage A 核心：网格搜索门控参数 |
| `scripts/rl_mmr/run_sensitivity_gated.sh` | Stage B 入口：读取最优参数运行完整流水线 |
| `src/fact_checking/rl_mmr/sensitivity.py` | 敏感度特征计算 + 门控决策函数 |
| `src/fact_checking/rl_mmr/gated_selector.py` | Build 阶段批量调用门控逻辑 |
| `src/fact_checking/rl_mmr/test_sensitivity.py` | 单元测试 |
| `configs/experiment/mmr_sensitivity_gated.yaml` | Stage B 实验配置 |
