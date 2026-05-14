# RL-MMR 有序实验计划

生成日期: 2026-05-14

本文档按如下顺序制定实验计划:

1. fixed `lambda = 0.7`
2. `log(n_candidates)` heuristic
3. sensitivity-gated MMR
4. soft-label lambda policy
5. PAMM-lite / DPO step-wise lambda policy
6. multi-weight MMR policy
7. GRPO refinement

本文档不展开论文内容，只定义实验目标、实现方式、评价指标、输出文件与推进 gate。

## 1. 总体实验目标

目标不是简单证明 learned lambda 一定优于 fixed lambda，而是系统验证:

```text
从 fixed diversity control 到 adaptive evidence diversity policy，是否能提升 fact-checking evidence selection 的下游 utility。
```

核心比较对象为:

```text
fixed lambda = 0.7
```

最终目标为:

```text
adaptive MMR policy > fixed-MMR
```

更长远目标为:

```text
reranker + adaptive MMR policy > reranker-only
```

## 2. 统一实验设置

### 2.1 输入与输出

输入:

- claim `c`
- candidate evidence pool `C_N(c)`
- relevance score `Rel(c,d)`
- pairwise similarity `Sim(d_i,d_j)`
- optional metadata: source、report id、time、speaker、label、stance proxy
- optional gold evidence 或 gold label

输出:

- selected evidence set `S_K`
- selected evidence order
- lambda 或 weight trajectory
- verifier prediction
- verifier confidence 或 correct label logprob
- evidence set utility
- cost statistics

### 2.2 统一 MMR 公式

基础形式:

```text
d_t = argmax_d [lambda_t * Rel(c,d) - (1 - lambda_t) * Red(d, S_{t-1})]
```

其中:

```text
Red(d, S_{t-1}) = max_{s in S_{t-1}} Sim(d,s)
```

若 `S_{t-1}` 为空，则 redundancy 项设为 0。

### 2.3 统一 candidate pool

所有实验必须使用相同 candidate pool，否则无法进行公平比较。

建议固定:

```text
retrieval.top_k = 32
final evidence budget K = 5
```

若已有实验使用不同 `K`，需在主表中保持一致，并在消融中再比较 `K = 3, 5, 8`。

### 2.4 统一 verifier

所有 selection 方法应使用同一个 SFT verifier 或同一个 inference checkpoint。推理配置应固定:

- prompt template
- label mapping
- max_new_tokens
- temperature
- decoding mode
- vLLM server setting

### 2.5 统一指标

主指标:

| 指标 | 用途 |
|---|---|
| verdict accuracy | 最终 fact-checking 正确率 |
| macro-F1 | 类别不均衡时的主补充指标 |
| correct label logprob | 连续型 verifier utility |
| evidence recall 或 evidence F1 | 若有 gold evidence |
| mean pairwise similarity | evidence set 冗余度 |
| selected evidence overlap | 与 fixed 0.7 的差异 |
| source diversity | 若有 source/report metadata |
| coverage proxy | coverage 的弱监督估计 |
| token cost | 输入 evidence 的上下文成本 |
| latency | 推理或选择耗时 |
| joint utility | composite reward 或综合分 |

建议主 utility:

```text
U = w_label * LabelUtility
  + w_ev * EvidenceUtility
  + w_cov * Coverage
  + w_div * Diversity
  - w_red * Redundancy
  - w_cost * Cost
```

### 2.6 统一报告方式

每个实验至少输出:

```text
outputs/rl_mmr/<system_name>/predictions.jsonl
outputs/rl_mmr/<system_name>/selected_evidence.jsonl
outputs/rl_mmr/<system_name>/lambda_trace.jsonl
outputs/rl_mmr/<system_name>/metrics.json
outputs/rl_mmr/<system_name>/metrics_by_bucket.json
outputs/rl_mmr/<system_name>/run_config.yaml
```

每条样本记录建议包含:

```json
{
  "id": "...",
  "claim": "...",
  "gold_label": "...",
  "pred_label": "...",
  "correct": true,
  "correct_label_logprob": -1.23,
  "selected_ids": [1, 5, 9, 11, 12],
  "lambda_trace": [0.7, 0.7, 0.7, 0.7, 0.7],
  "utility": 0.52,
  "redundancy": 0.31,
  "cost_tokens": 1530
}
```

## 3. 实验 1: fixed `lambda = 0.7`

### 3.1 目的

锁定强 baseline。后续所有方法必须与该系统做 paired comparison。

### 3.2 方法

固定:

```text
lambda_t = 0.7 for all t
```

选择:

```text
d_t = argmax_d [0.7 * Rel(c,d) - 0.3 * Red(d,S_{t-1})]
```

### 3.3 实现任务

1. 固定 `retrieval.mmr_lambda = 0.70`。
2. 使用当前共享 candidate pool 与 verifier。
3. 保存 selected evidence、verifier predictions、metrics。
4. 输出 lambda trace，所有位置均为 0.7。

建议配置:

```text
configs/experiment/mmr_fixed_07.yaml
```

建议脚本:

```text
scripts/rl_mmr/run_fixed_07.sh
```

### 3.4 指标

必须记录:

- accuracy
- macro-F1
- correct label logprob
- mean pairwise similarity
- selected evidence IDs
- token cost
- latency

### 3.5 成功标准

该实验不追求超过其他方法，而是作为 locked baseline。输出必须稳定可复现。

### 3.6 产物

```text
outputs/rl_mmr/fixed_07/
```

## 4. 实验 2: `log(n_candidates)` heuristic

### 4.1 目的

测试极简 adaptive lambda 是否能接近或超过 fixed 0.7。

### 4.2 假设

候选数量越多，候选池越可能存在冗余，需要更低 lambda 以增强 diversity。候选数量越少，应偏向 relevance。

### 4.3 方法

形式:

```text
lambda(c) = clip(a * log(n_candidates(c)) + b, lambda_min, lambda_max)
```

初始可用:

```text
lambda(c) = clip(-0.073 * log(n_candidates(c)) + 0.613, 0.0, 1.0)
```

建议在 train 或 dev 上重新拟合 `a`、`b`，并比较以下版本:

| 版本 | 说明 |
|---|---|
| log-linear-fixed | 使用已有经验系数 |
| log-linear-fit | 在 train/dev 上拟合系数 |
| binned-count | 按候选数量分桶设置 lambda |

分桶示例:

| n_candidates | lambda |
|---:|---:|
| 1 到 4 | 0.65 |
| 5 到 6 | 0.55 |
| 7 到 15 | 0.40 |
| 16 到 30 | 0.45 |
| 31 以上 | 0.50 |

分桶值只作为初始建议，实际应在 dev set 上调节。

### 4.4 实现任务

新增函数:

```text
lambda_from_candidate_count(n_candidates, mode, params)
```

建议配置:

```text
configs/experiment/mmr_log_n.yaml
```

建议脚本:

```text
scripts/rl_mmr/run_log_n.sh
```

### 4.5 对比

主对比:

```text
fixed_07 vs log_n
```

### 4.6 成功标准

满足以下任一条件即可保留为强 baseline:

1. dev accuracy 或 macro-F1 高于 fixed 0.7。
2. correct label logprob 高于 fixed 0.7。
3. 在高候选数或高冗余 bucket 中明显优于 fixed 0.7。

### 4.7 失败解释

如果无收益，说明 candidate count 的弱相关不足以转化为下游 verifier gain。

## 5. 实验 3: sensitivity-gated MMR

### 5.1 目的

测试不预测 oracle lambda，而是预测“是否需要更强 diversity”的策略。

### 5.2 假设

当不同 lambda 导致 selected evidence set 明显变化，且候选池内部冗余较高时，较低 lambda 可能优于 fixed 0.7。

### 5.3 核心特征

对每条 claim 预计算:

```text
S_03 = MMR(lambda = 0.3)
S_07 = MMR(lambda = 0.7)
S_10 = MMR(lambda = 1.0)
```

基础 sensitivity:

```text
sens_03_07 = 1 - Jaccard(S_03, S_07)
sens_07_10 = 1 - Jaccard(S_07, S_10)
sens_03_10 = 1 - Jaccard(S_03, S_10)
```

冗余特征:

```text
pool_redundancy = mean pairwise Sim among top-N candidates
selected_redundancy_07 = mean pairwise Sim within S_07
```

score 分布特征:

```text
score_entropy
top1_top2_gap
top5_score_std
top10_mass
```

### 5.4 决策规则

基础二值 gating:

```text
if sens_03_07 >= theta_s and pool_redundancy >= theta_r:
    lambda = 0.3
else:
    lambda = 0.7
```

更保守版本:

```text
if sens_03_07 >= theta_s and pool_redundancy >= theta_r and relevance_floor_ok:
    lambda = 0.3
else:
    lambda = 0.7
```

其中 `relevance_floor_ok` 可定义为:

```text
min Rel(d, c) in S_03 >= percentile(Rel top-N, p)
```

或:

```text
mean Rel(S_03) >= mean Rel(S_07) - epsilon
```

### 5.5 超参数搜索

建议搜索:

```text
theta_s in {0.2, 0.4, 0.6, 0.8}
theta_r in {0.3, 0.4, 0.5, 0.6}
lambda_low in {0.2, 0.3, 0.4}
lambda_base = 0.7
```

### 5.6 实现任务

新增模块:

```text
src/fact_checking/rl_mmr/sensitivity.py
src/fact_checking/rl_mmr/gated_selector.py
```

建议配置:

```text
configs/experiment/mmr_sensitivity_gated.yaml
```

建议脚本:

```text
scripts/rl_mmr/run_sensitivity_gated.sh
```

### 5.7 输出

额外输出:

```json
{
  "sens_03_07": 0.6,
  "pool_redundancy": 0.52,
  "gate": "low_lambda",
  "chosen_lambda": 0.3
}
```

### 5.8 成功标准

优先看 dev set:

1. 整体 accuracy 或 macro-F1 高于 fixed 0.7。
2. 高 sensitivity bucket 中明显优于 fixed 0.7。
3. 冗余下降但 label accuracy 不下降。

### 5.9 失败解释

如果冗余下降但 accuracy 下降，说明低 lambda 引入了“不同但无用”的 evidence，需要更强 relevance floor 或 coverage-aware score。

## 6. 实验 4: soft-label lambda policy

### 6.1 目的

修复 hard oracle lambda predictor 的噪声监督问题。

### 6.2 假设

完整 utility curve 比单点 argmax lambda 更稳定。低 margin 样本应贡献软标签或低权重，而不是强制模型学习一个随机 hard label。

### 6.3 训练目标

对每条 claim 已有:

```text
U_i(lambda_1), U_i(lambda_2), ..., U_i(lambda_m)
```

构造 soft target:

```text
q_i(lambda_j) = exp(U_i(lambda_j) / tau) / sum_k exp(U_i(lambda_k) / tau)
```

样本权重:

```text
w_i = max_lambda U_i(lambda) - U_i(0.7)
```

或:

```text
w_i = top1_utility - top2_utility
```

损失:

```text
L = sum_i w_i * KL(q_i(lambda) || p_theta(lambda | x_i))
```

### 6.4 lambda 网格

建议先使用粗网格:

```text
Lambda = {0.1, 0.3, 0.5, 0.7, 0.9}
```

不要一开始使用 21 点网格。粗网格可降低曲面平坦导致的多重比较噪声。

### 6.5 特征设计

不建议只用 BGE embedding。建议使用三类特征。

#### A. Candidate pool features

```text
n_candidates
log_n_candidates
score_entropy
top1_top2_gap
top5_score_std
top10_mass
mean_pairwise_sim
max_pairwise_sim
cluster_count
```

#### B. Interventional MMR features

```text
Jaccard(S_0.3, S_0.7)
Jaccard(S_0.1, S_0.9)
mean_rel(S_0.7) - mean_rel(S_0.3)
mean_red(S_0.7) - mean_red(S_0.3)
KendallDistance(rank_0.3, rank_0.7)
number_of_selected_changes_across_lambda_grid
```

#### C. Claim features

```text
claim_length
entity_count
number_count
time_expression_count
negation_flag
comparison_flag
```

### 6.6 模型

优先使用轻量模型:

| 模型 | 用途 |
|---|---|
| logistic regression | 可解释 baseline |
| gradient boosting 或 random forest | 非线性 tabular baseline |
| small MLP | neural baseline |

不建议首先使用大模型或复杂 attention encoder。

### 6.7 推理策略

三种方式都要测试:

```text
lambda = argmax p_theta(lambda | x)
lambda = sum_lambda p_theta(lambda | x) * lambda
lambda = sample from p_theta(lambda | x) with temperature 0.5
```

主评估建议用 expected lambda 或 argmax lambda。

### 6.8 实现任务

新增模块:

```text
src/fact_checking/rl_mmr/soft_label_dataset.py
src/fact_checking/rl_mmr/soft_label_policy.py
scripts/rl_mmr/train_soft_label_lambda.py
scripts/rl_mmr/evaluate_soft_label_lambda.py
```

建议配置:

```text
configs/experiment/mmr_soft_label_lambda.yaml
```

### 6.9 成功标准

1. 比 hard oracle predictor 更稳定。
2. 比 fixed 0.7 或 log heuristic 至少在某些 bucket 中更好。
3. predicted distribution 不应坍缩到单一类别。
4. Calibration 可接受，低 confidence 样本应多为低 margin 样本。

### 6.10 决策

如果 soft-label 仍无收益，但 probability distribution 可作为 reference policy，则进入 DPO。若完全坍缩，则 DPO 的 reference policy 使用 fixed 0.7 或 sensitivity-gated MMR。

## 7. 实验 5: PAMM-lite / DPO step-wise lambda policy

### 7.1 目的

把学习粒度从 claim-level scalar lambda 推进到 step-wise lambda trajectory，并用 evidence set preference 训练。

### 7.2 核心假设

训练模型偏好高 utility trajectory，比预测 hard oracle lambda 更稳。因为低 gap 样本可以过滤，高 gap pair 可以提供更清晰监督。

### 7.3 Trajectory 定义

```text
tau = ((lambda_1, d_1), ..., (lambda_K, d_K))
```

每一步:

```text
lambda_t ~ pi_theta(lambda | s_t)
d_t = MMR_select(lambda_t, c, C, S_{t-1})
S_t = S_{t-1} union {d_t}
```

state:

```text
s_t = {c, C, S_{t-1}, t, Rel, Sim, pool_stats, selected_stats}
```

### 7.4 Trajectory 生成

先使用手工 schedules 和随机 schedules 生成候选轨迹。

建议 schedules:

```text
[0.7, 0.7, 0.7, 0.7, 0.7]
[0.3, 0.3, 0.3, 0.3, 0.3]
[0.5, 0.5, 0.5, 0.5, 0.5]
[0.9, 0.7, 0.5, 0.3, 0.3]
[1.0, 0.7, 0.5, 0.3, 0.1]
[0.5, 0.5, 0.7, 0.7, 0.9]
[0.7, 0.5, 0.3, 0.5, 0.7]
```

再加入随机 schedules:

```text
lambda_t sampled from {0.1, 0.3, 0.5, 0.7, 0.9}
```

每条 claim 先生成 20 到 50 条 trajectories。

### 7.5 Preference pair 构造

对每条 trajectory 计算:

```text
U(tau) = U(c, S_K(tau))
```

构造 pair:

```text
tau_plus > tau_minus if U(tau_plus) - U(tau_minus) >= delta
```

建议:

```text
delta in {0.02, 0.05, 0.10}
max_pairs_per_claim = 10
```

优先保留:

- 与 fixed 0.7 选择结果不同的 pair
- utility gap 大的 pair
- redundancy 差异大的 pair
- verifier logprob 差异大的 pair

### 7.6 Policy 模型

动作空间:

```text
Lambda = {0.1, 0.3, 0.5, 0.7, 0.9}
```

输入特征:

```text
pool_stats
selected_stats at step t
last_selected_relevance
current_mean_redundancy
remaining_candidate_score_entropy
step index t
```

模型建议:

```text
small MLP over tabular state features
```

不要一开始使用大 Transformer policy。先验证学习信号。

### 7.7 训练目标

DPO 形式:

```text
L_DPO = -log sigmoid(beta * ((log pi_theta(tau_plus) - log pi_theta(tau_minus))
                            - (log pi_ref(tau_plus) - log pi_ref(tau_minus))))
```

其中:

```text
log pi_theta(tau) = sum_t log pi_theta(lambda_t | s_t)
```

Reference policy:

| ref | 说明 |
|---|---|
| fixed 0.7 | 最稳定 |
| log heuristic | 若实验 2 有收益 |
| sensitivity-gated | 若实验 3 有收益 |
| soft-label policy | 若实验 4 有收益 |

也可测试 margin loss:

```text
L_margin = max(0, DeltaU - (log pi_theta(tau_plus) - log pi_theta(tau_minus)))
```

### 7.8 实现任务

新增模块:

```text
src/fact_checking/rl_mmr/trajectory.py
src/fact_checking/rl_mmr/reward.py
src/fact_checking/rl_mmr/preference_dataset.py
src/fact_checking/rl_mmr/dpo_policy.py
scripts/rl_mmr/generate_trajectories.py
scripts/rl_mmr/build_preference_pairs.py
scripts/rl_mmr/train_dpo_step_lambda.py
scripts/rl_mmr/evaluate_dpo_step_lambda.py
```

建议配置:

```text
configs/experiment/mmr_dpo_step_lambda.yaml
```

### 7.9 输出

额外输出:

```text
trajectory_pool.jsonl
preference_pairs.jsonl
policy_checkpoint.pt
dpo_training_metrics.json
lambda_distribution_by_step.json
```

### 7.10 成功标准

1. 超过 fixed 0.7。
2. 超过 sensitivity-gated MMR 或 soft-label policy。
3. step-wise lambda distribution 有合理结构，例如第一步偏高 lambda，后续可降低。
4. 高 sensitivity bucket 或高 redundancy bucket 中提升更明显。
5. 不显著增加 cost。

### 7.11 失败分析

若失败，检查:

- trajectory pool 是否多样。
- preference pair 的 utility gap 是否足够。
- reward 是否过度依赖 logprob 尾部值。
- policy 是否坍缩到固定 lambda。
- MMR 选择是否对 lambda schedule 真的敏感。

## 8. 实验 6: multi-weight MMR policy

### 8.1 目的

突破 scalar lambda 的表达能力限制，学习多维 evidence selection 权重。

### 8.2 假设

事实核查 evidence selection 不只是 relevance 与 redundancy 的单轴 trade-off，还需要 coverage、source novelty、stance diversity、cost 等因素。multi-weight policy 能更好拟合 set-level utility。

### 8.3 打分函数

```text
Score(d | s_t) = w_rel,t * Rel(c,d)
               - w_red,t * Red(d,S_{t-1})
               + w_cov,t * Cov(d,S_{t-1},c)
               + w_src,t * SrcNovelty(d,S_{t-1})
               + w_stance,t * StanceNovelty(d,S_{t-1})
               - w_cost,t * Cost(d)
```

每一步:

```text
w_t = g_theta(s_t)
d_t = argmax_d Score(d | s_t)
```

### 8.4 Feature definitions

#### Relevance

```text
Rel(c,d) = hybrid retrieval score or reranker score
```

#### Redundancy

```text
Red(d,S) = max_{s in S} Sim(d,s)
```

#### Coverage proxy

可选实现:

```text
Cov(d,S,c) = 1 - max_{s in S} Sim_claim_aspect(d,s)
```

或基于 claim subcomponents:

```text
Cov(d,S,c) = number of new claim entities/numbers/time expressions covered by d
```

#### Source novelty

若有 report id 或 source metadata:

```text
SrcNovelty(d,S) = 1 if source(d) not in source(S) else 0
```

若无 source metadata，可用 report id 或 chunk parent id 代替。

#### Stance novelty

初期可不做复杂 stance classifier。可先用 simple proxy:

```text
StanceNovelty = diversity over predicted entailment/refutation/neutral proxy
```

若没有 stance signal，则先省略该项，避免引入噪声。

#### Cost

```text
Cost(d) = token_count(d) / max_token_count
```

### 8.5 Policy 输出约束

为保持可解释性，建议使用非负权重:

```text
w_t = softplus(raw_w_t)
```

或归一化:

```text
w_t = softmax(raw_w_t)
```

如果使用 softmax，需要注意负项与正项分别归一化，避免符号混乱。

### 8.6 训练方式

先复用实验 5 的 preference framework。把 trajectory 中的 action 从 `lambda_t` 换成 `w_t` 或离散 weight template。

建议先使用离散 template，降低训练难度:

| template | w_rel | w_red | w_cov | w_src | w_cost |
|---|---:|---:|---:|---:|---:|
| relevance-first | 1.0 | 0.2 | 0.1 | 0.0 | 0.0 |
| balanced | 0.7 | 0.4 | 0.3 | 0.1 | 0.0 |
| diversity-heavy | 0.5 | 0.7 | 0.4 | 0.2 | 0.0 |
| coverage-heavy | 0.6 | 0.3 | 0.8 | 0.2 | 0.0 |
| cost-aware | 0.7 | 0.4 | 0.3 | 0.1 | 0.3 |

Policy 先选择 template:

```text
a_t = template_id
```

后续再扩展为连续权重。

### 8.7 实现任务

新增模块:

```text
src/fact_checking/rl_mmr/multi_weight_selector.py
src/fact_checking/rl_mmr/weight_templates.py
scripts/rl_mmr/train_dpo_multi_weight.py
scripts/rl_mmr/evaluate_multi_weight.py
```

建议配置:

```text
configs/experiment/mmr_multi_weight.yaml
```

### 8.8 消融

必须做:

```text
Rel + Red only
Rel + Red + Cov
Rel + Red + Cov + SrcNovelty
Rel + Red + Cov + SrcNovelty + Cost
```

若 stance proxy 可用，再加:

```text
Rel + Red + Cov + SrcNovelty + StanceNovelty + Cost
```

### 8.9 成功标准

1. 超过 DPO step-wise lambda。
2. 在复杂 claim、高冗余候选池、跨 source evidence 任务上提升更明显。
3. 能降低 redundancy，同时不牺牲 relevance 或 verdict accuracy。
4. 权重分布可解释，不应所有样本坍缩到同一 template。

## 9. 实验 7: GRPO refinement

### 9.1 目的

在 DPO 或 multi-weight policy 已有收益的基础上，进一步直接优化 reward。

### 9.2 前置条件

只有满足以下条件才开始 GRPO:

1. DPO step-wise lambda 或 multi-weight policy 在 dev set 上稳定超过 fixed 0.7。
2. Reward 定义稳定，不依赖少数 logprob 极端值。
3. 每条 claim 能生成多个具有差异的 trajectories。
4. Policy 未坍缩到单一 action。

### 9.3 训练流程

```text
for each batch of claims:
    for each claim c:
        sample G trajectories tau_1, ..., tau_G from pi_theta
        compute rewards R_1, ..., R_G
        A_i = (R_i - mean(R)) / (std(R) + epsilon)
    update pi_theta with group-relative objective and KL penalty to pi_ref
```

### 9.4 Reward

推荐 reward:

```text
R = w_label * LabelUtility
  + w_ev * EvidenceUtility
  + w_cov * Coverage
  + w_div * Diversity
  - w_red * Redundancy
  - w_cost * Cost
```

必须做 reward clipping 或 normalization:

```text
R_clipped = clip(R, r_min, r_max)
```

### 9.5 KL 约束

Reference policy 使用 dev 表现最好的 offline policy:

```text
pi_ref = best DPO policy or best multi-weight policy
```

KL penalty:

```text
L_KL = beta_KL * KL(pi_theta || pi_ref)
```

### 9.6 实现任务

新增模块:

```text
src/fact_checking/rl_mmr/grpo_trainer.py
scripts/rl_mmr/train_grpo_refine.py
scripts/rl_mmr/evaluate_grpo.py
```

建议配置:

```text
configs/experiment/mmr_grpo_refine.yaml
```

### 9.7 超参数初值

```text
G = 4 or 8
learning_rate = 1e-5 to 5e-5 for neural policy
beta_KL = 0.01 to 0.1
reward_clip = [-2, 2] or [0, 1]
entropy_bonus = small, optional
max_updates = small, early stopping on dev utility
```

### 9.8 成功标准

1. 在 dev set 上超过 reference DPO policy。
2. test set 上不出现过拟合回落。
3. policy entropy 不快速坍缩。
4. reward component 没有被单一项劫持，例如只降低 redundancy 但 accuracy 下降。

### 9.9 失败处理

如果 GRPO 不稳定，应回退到 DPO 或 multi-weight DPO 作为最终主方法。GRPO 可作为负结果或附录分析，不必强行作为主贡献。

## 10. 统一分桶分析

所有系统都应按以下 bucket 报告:

| Bucket | 定义 |
|---|---|
| candidate count | 1 到 4、5 到 6、7 到 15、16 到 30、31 以上 |
| sensitivity | low、medium、high based on `1 - Jaccard(S_0.3,S_0.7)` |
| pool redundancy | low、medium、high based on mean pairwise Sim |
| oracle margin | low margin、high margin |
| label type | 各 veracity label |
| claim length | short、medium、long |
| evidence overlap shift | selected set 与 fixed 0.7 相同或不同 |

重点观察:

```text
adaptive policy 是否主要在 high sensitivity 和 high redundancy bucket 中收益最大
```

## 11. 统一统计检验

建议使用 paired test，因为每个系统处理的是同一批 claim。

可报告:

- paired bootstrap confidence interval
- McNemar test for accuracy difference
- paired t-test 或 Wilcoxon signed-rank for logprob/utility
- effect size by bucket

主表至少报告 mean 与 95% CI。

## 12. 实验推进时间线

### Phase A: Baseline consolidation

包括:

1. fixed `lambda = 0.7`
2. `log(n_candidates)` heuristic
3. sensitivity-gated MMR

目标: 在 3 个低成本方法中确定最强非学习或弱学习 baseline。

### Phase B: Supervised repair

包括:

4. soft-label lambda policy

目标: 判断 supervised lambda learning 是否可以通过 soft target 与 interventional features 修复。

### Phase C: Offline preference learning

包括:

5. PAMM-lite / DPO step-wise lambda policy
6. multi-weight MMR policy

目标: 验证 trajectory-level preference learning 是否优于 fixed 与 heuristic。

### Phase D: Online or semi-online refinement

包括:

7. GRPO refinement

目标: 在 DPO 已有收益的前提下进一步优化 reward。

## 13. 最终主表建议

| System | Accuracy | Macro-F1 | Correct label logprob | Utility | Redundancy | Cost | High-sensitivity Acc | High-redundancy Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed 0.7 | | | | | | | | |
| log(n) heuristic | | | | | | | | |
| sensitivity-gated | | | | | | | | |
| soft-label lambda | | | | | | | | |
| DPO step-wise lambda | | | | | | | | |
| multi-weight MMR | | | | | | | | |
| GRPO refine | | | | | | | | |

## 14. 关键代码接入建议

当前代码可以按 build、train、infer 流水线接入新方法。建议新增 `rl_mmr` 子模块，避免把新实验散落在原 learned-lambda 目录中。

建议目录:

```text
src/fact_checking/rl_mmr/
  __init__.py
  mmr.py
  sensitivity.py
  features.py
  utility.py
  trajectory.py
  preference_dataset.py
  soft_label_policy.py
  dpo_policy.py
  multi_weight_selector.py
  grpo_trainer.py
```

建议脚本:

```text
scripts/rl_mmr/run_fixed_07.sh
scripts/rl_mmr/run_log_n.sh
scripts/rl_mmr/run_sensitivity_gated.sh
scripts/rl_mmr/train_soft_label_lambda.py
scripts/rl_mmr/generate_trajectories.py
scripts/rl_mmr/build_preference_pairs.py
scripts/rl_mmr/train_dpo_step_lambda.py
scripts/rl_mmr/train_dpo_multi_weight.py
scripts/rl_mmr/train_grpo_refine.py
```

建议配置:

```text
configs/experiment/mmr_fixed_07.yaml
configs/experiment/mmr_log_n.yaml
configs/experiment/mmr_sensitivity_gated.yaml
configs/experiment/mmr_soft_label_lambda.yaml
configs/experiment/mmr_dpo_step_lambda.yaml
configs/experiment/mmr_multi_weight.yaml
configs/experiment/mmr_grpo_refine.yaml
```

## 15. Stop criteria

为避免在弱信号方向过度投入，建议设定停止条件。

### 停止普通 hard lambda prediction

若模型继续接近均值预测或输出方差坍缩，则停止该方向。

### 停止 soft-label lambda

若 soft-label policy 在 dev set 上不超过 log heuristic，且预测分布坍缩，则停止 supervised lambda 路线。

### 停止 DPO step-wise

若 preference pairs 中高 gap pair 数量不足，或 DPO policy 坍缩到 fixed 0.7 且无收益，则需要先重构 reward 或 trajectory generator。

### 停止 GRPO

若 GRPO dev utility 不升、policy entropy 快速下降、或 accuracy 下降，应回退到 DPO checkpoint。

## 16. 最终建议

按本计划推进时，最关键的不是让每一步都超过前一步，而是让每一步回答一个明确问题:

1. `log(n_candidates)` 能否提供最低成本 adaptive 信号。
2. sensitivity-gated MMR 能否利用 lambda 敏感性。
3. soft-label lambda 能否修复 hard oracle 标签噪声。
4. DPO step-wise policy 能否利用 trajectory preference。
5. multi-weight MMR 是否解决 scalar lambda 表达能力不足。
6. GRPO 是否能在稳定 offline policy 上进一步提升。

若最终结果是 multi-weight DPO 明显优于 fixed 0.7，而 GRPO 没有进一步提升，这仍然是一个合理且完整的研究结论。
