# RL-MMR 有序实验计划

生成日期: 2026-05-14

本文档按如下顺序制定实验计划:

1. fixed $\lambda = 0.7$
2. $\log(n_{\mathrm{candidates}})$ heuristic
3. sensitivity-gated MMR
4. soft-label $\lambda$ policy
5. PAMM-lite / DPO step-wise $\lambda$ policy
6. multi-weight MMR policy
7. GRPO refinement

本文档不展开论文内容，只定义实验目标、实现方式、评价指标、输出文件与推进 gate。

## 0. 当前实验结论快照

更新时间: 2026-05-16

已完成的前五个方向的结论:

| 实验 | 结论 | 状态 |
|---|---|---|
| fixed $\lambda=0.7$ | 稳定 baseline, test accuracy=0.2702 | locked |
| $\log(n)$ heuristic | test +0.0064, 弱 adaptive baseline | 保留, 不深挖 |
| sensitivity-gated MMR | test +0.0040, 弱 adaptive baseline | 保留, 不深挖 |
| soft-label $\lambda$ | 修复 oracle 后 `expected` 退化, `argmax/sample` 更差 | **已停止** |
| DPO step-wise $\lambda$ | 4 次训练全部坍缩到 λ=0.7, 无法学到自适应策略 | **已停止** |

DPO step-wise 详细结论: 尽管 utility 信号存在 (63.2% 的 claim 有优于 fixed 0.7 的 λ schedule, median utility range=2.34)，但 policy 始终坍缩到 reference center。根因:
1. 特征信息量不足 — pool features 对同 claim 的 winner/loser 完全相同 (纯噪声)，step-0 时所有 trajectory 状态一样 (无法区分)
2. 内生性问题 — step features 的差异是 λ 选择的**结果**而非**原因**
3. 有效 DPO 梯度太少 — reference policy (偏向 0.7) 已与多数 winner 一致，偏离 reference 的 gradient 不足以克服其吸引域

当前建议:

1. 保留 fixed $\lambda=0.7$ 作为 locked baseline。
2. $\log(n_{\mathrm{candidates}})$ 和 sensitivity-gated MMR 可作为弱 adaptive baseline 报告。
3. **停止所有 scalar $\lambda$ 方向** (experiments 1-5 覆盖了 claim-level 和 step-wise scalar $\lambda$ 的完整探索)。
4. GRPO refinement (实验 7) 不跑 — 其前置条件 (DPO 有收益) 未满足。
5. 下一步唯一有意义的方向: **实验 6: multi-weight MMR policy** — 将 scalar $\lambda$ 扩展为多维权重向量，增加 coverage、source novelty 等优化维度。

## 1. 总体实验目标

目标不是简单证明 learned $\lambda$ 一定优于 fixed $\lambda$，而是系统验证:

$$ \text{fixed diversity control} \rightarrow \text{adaptive evidence diversity policy} $$

即验证 adaptive evidence diversity policy 是否能提升 fact-checking evidence selection 的下游 utility。

核心比较对象为:

$$ \lambda_{\mathrm{fixed}} = 0.7 $$

最终目标为:

$$ \text{adaptive MMR policy} > \text{fixed-MMR} $$

更长远目标为:

$$ \text{reranker} + \text{adaptive MMR policy} > \text{reranker-only} $$

## 2. 统一实验设置

### 2.1 输入与输出

输入:

- claim $c$
- candidate evidence pool $C_N(c)$
- relevance score $\operatorname{Rel}(c,d)$
- pairwise similarity $\operatorname{Sim}(d_i,d_j)$
- optional metadata: source、report id、time、speaker、label、stance proxy
- optional gold evidence 或 gold label

输出:

- selected evidence set $S_K$
- selected evidence order
- $\lambda$ 或 weight trajectory
- verifier prediction
- verifier confidence 或 correct label logprob
- evidence set utility
- cost statistics

### 2.2 统一 MMR 公式

基础形式:

$$ d_t = \arg\max_{d \in C_N(c) \setminus S_{t-1}} \left[ \lambda_t \operatorname{Rel}(c,d) - (1-\lambda_t)\operatorname{Red}(d,S_{t-1}) \right] $$

其中:

$$ \operatorname{Red}(d,S_{t-1}) = \max_{s \in S_{t-1}}\operatorname{Sim}(d,s) $$

若 $S_{t-1}$ 为空，则 redundancy 项设为 $0$。

### 2.3 统一 candidate pool

所有实验必须使用相同 candidate pool，否则无法进行公平比较。

建议固定:

$$ \texttt{retrieval.top\_k} = 32,\qquad K = 5 $$

若已有实验使用不同 $K$，需在主表中保持一致，并在消融中再比较 $K \in \{3,5,8\}$。

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
| selected evidence overlap | 与 fixed $\lambda=0.7$ 的差异 |
| source diversity | 若有 source/report metadata |
| coverage proxy | coverage 的弱监督估计 |
| token cost | 输入 evidence 的上下文成本 |
| latency | 推理或选择耗时 |
| joint utility | composite reward 或综合分 |

建议主 utility:

$$ \begin{aligned} U ={}& w_{\mathrm{label}}\operatorname{LabelUtility}  + w_{\mathrm{ev}}\operatorname{EvidenceUtility}  + w_{\mathrm{cov}}\operatorname{Coverage}  + w_{\mathrm{div}}\operatorname{Diversity} \\ & - w_{\mathrm{red}}\operatorname{Redundancy}  - w_{\mathrm{cost}}\operatorname{Cost} \end{aligned} $$

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

## 3. 实验 1: fixed $\lambda = 0.7$

### 3.1 目的

锁定强 baseline。后续所有方法必须与该系统做 paired comparison。

### 3.2 方法

固定:

$$ \lambda_t = 0.7,\qquad t=1,\ldots,K $$

选择:

$$ d_t = \arg\max_{d \in C_N(c) \setminus S_{t-1}} \left[ 0.7\operatorname{Rel}(c,d) - 0.3\operatorname{Red}(d,S_{t-1}) \right] $$

### 3.3 实现任务

1. 固定 `retrieval.mmr_lambda = 0.70`。
2. 使用当前共享 candidate pool 与 verifier。
3. 保存 selected evidence、verifier predictions、metrics。
4. 输出 $\lambda$ trace，所有位置均为 $0.7$。

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

### 3.7 当前结果与结论

已采用 `b3_mmr_topk_sweep_1024` 中 `top_k=5, mmr_lambda=0.7` 的 run 作为当前可比 baseline:

```text
outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/infer/test/best/79d8b34809bb/api/metrics.json
```

test 结果:

| system | n | accuracy | macro-F1 | parse error |
|---|---:|---:|---:|---:|
| fixed $\lambda=0.7$, `top_k=5` | 1251 | 0.2702 | 0.2769 | 0.0000 |

结论: 该结果作为后续 RL-MMR 的 locked baseline。后续 adaptive 方法只有在同一 candidate pool、同一 verifier 与同一推理配置下超过它，才算有效提升。

## 4. 实验 2: $\log(n_{\mathrm{candidates}})$ heuristic

### 4.1 目的

测试极简 adaptive $\lambda$ 是否能接近或超过 fixed $0.7$。

### 4.2 假设

候选数量越多，候选池越可能存在冗余，需要更低 $\lambda$ 以增强 diversity。候选数量越少，应偏向 relevance。

### 4.3 方法

形式:

$$ \lambda(c) = \operatorname{clip} \left( a\log n_{\mathrm{candidates}}(c) + b,\, \lambda_{\min},\, \lambda_{\max} \right) $$

初始可用:

$$ \lambda(c) = \operatorname{clip} \left( -0.073\log n_{\mathrm{candidates}}(c) + 0.613,\, 0.0,\, 1.0 \right) $$

建议在 train 或 dev 上重新拟合 `a`、`b`，并比较以下版本:

| 版本 | 说明 |
|---|---|
| log-linear-fixed | 使用已有经验系数 |
| log-linear-fit | 在 train/dev 上拟合系数 |
| binned-count | 按候选数量分桶设置 $\lambda$ |

分桶示例:

| $n_{\mathrm{candidates}}$ | $\lambda$ |
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

$$ \mathrm{fixed}_{0.7}\ \mathrm{vs.}\ \log(n_{\mathrm{candidates}}) $$

### 4.6 成功标准

满足以下任一条件即可保留为强 baseline:

1. dev accuracy 或 macro-F1 高于 fixed $\lambda=0.7$。
2. correct label logprob 高于 fixed $\lambda=0.7$。
3. 在高候选数或高冗余 bucket 中明显优于 fixed $\lambda=0.7$。

### 4.7 失败解释

如果无收益，说明 candidate count 的弱相关不足以转化为下游 verifier gain。

### 4.8 当前结果与结论

当前可比 LoRA 版本 artifact:

```text
outputs/runs/heuristic_lambda_mmr/33088d994dd4/infer/test/best/79d8b34809bb/api/metrics.json
```

test 结果:

| system | n | accuracy | macro-F1 | parse error | vs fixed acc | vs fixed macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| fixed $\lambda=0.7$, `top_k=5` | 1251 | 0.2702 | 0.2769 | 0.0000 | 0.0000 | 0.0000 |
| $\log(n_{\mathrm{candidates}})$ heuristic | 1251 | 0.2766 | 0.2799 | 0.0000 | +0.0064 | +0.0030 |

结论: $\log(n_{\mathrm{candidates}})$ 是一个便宜的弱 adaptive baseline，但收益很小。当前结果只能说明候选数量与最优 $\lambda$ 存在弱相关，不能说明 claim-level scalar $\lambda$ 已经提供稳定下游收益。

另有 full fine-tuning 版本:

```text
outputs/runs/heuristic_lambda_mmr_fullft/484dc7dca0ad/infer/test/best/0d27dabf11a7/api/metrics.json
```

该版本 test accuracy 0.3110、macro-F1 0.3215，但它同时改变了训练方式（全参数微调、ZeRO-3、学习率等），不能用于隔离 $\log(n)$ $\lambda$ heuristic 的贡献；主线 RL-MMR 比较中不把它作为 $\lambda$ policy 结论。

## 5. 实验 3: sensitivity-gated MMR

### 5.1 目的

测试不预测 oracle $\lambda$，而是预测“是否需要更强 diversity”的策略。

### 5.2 假设

当不同 $\lambda$ 导致 selected evidence set 明显变化，且候选池内部冗余较高时，较低 $\lambda$ 可能优于 fixed $0.7$。

### 5.3 核心特征

对每条 claim 预计算:

$$ S_{0.3}=\operatorname{MMR}(\lambda=0.3),\qquad S_{0.7}=\operatorname{MMR}(\lambda=0.7),\qquad S_{1.0}=\operatorname{MMR}(\lambda=1.0) $$

基础 sensitivity:

$$ \begin{aligned} \mathrm{sens}_{0.3,0.7} &= 1-\operatorname{Jaccard}(S_{0.3},S_{0.7}),\\ \mathrm{sens}_{0.7,1.0} &= 1-\operatorname{Jaccard}(S_{0.7},S_{1.0}),\\ \mathrm{sens}_{0.3,1.0} &= 1-\operatorname{Jaccard}(S_{0.3},S_{1.0}). \end{aligned} $$

冗余特征:

$$ \mathrm{pool\_redundancy} = \frac{2}{N(N-1)}\sum_{1\le i<j\le N}\operatorname{Sim}(d_i,d_j) $$

$$ \mathrm{selected\_redundancy}_{0.7} = \frac{2}{|S_{0.7}|(|S_{0.7}|-1)} \sum_{\substack{d_i,d_j\in S_{0.7}\\i<j}} \operatorname{Sim}(d_i,d_j) $$

score 分布特征:

```text
score_entropy
top1_top2_gap
top5_score_std
top10_mass
```

### 5.4 决策规则

基础二值 gating:

$$ \lambda(c)= \begin{cases} 0.3, & \mathrm{sens}_{0.3,0.7}\ge \theta_s\ \land\ \mathrm{pool\_redundancy}\ge \theta_r,\\ 0.7, & \text{otherwise}. \end{cases} $$

更保守版本:

$$ \lambda(c)= \begin{cases} 0.3, & \mathrm{sens}_{0.3,0.7}\ge \theta_s\ \land\ \mathrm{pool\_redundancy}\ge \theta_r\ \land\ \mathrm{relevance\_floor\_ok}(c),\\ 0.7, & \text{otherwise}. \end{cases} $$

其中 `relevance_floor_ok` 可定义为:

$$ \min_{d\in S_{0.3}}\operatorname{Rel}(c,d) \ge \operatorname{Percentile}_{p} \left( \{\operatorname{Rel}(c,d): d\in C_N(c)\} \right) $$

或:

$$ \frac{1}{|S_{0.3}|}\sum_{d\in S_{0.3}}\operatorname{Rel}(c,d) \ge \frac{1}{|S_{0.7}|}\sum_{d\in S_{0.7}}\operatorname{Rel}(c,d) - \epsilon $$

### 5.5 超参数搜索

建议搜索:

$$ \theta_s \in \{0.2,0.4,0.6,0.8\},\qquad \theta_r \in \{0.3,0.4,0.5,0.6\} $$

$$ \lambda_{\mathrm{low}} \in \{0.2,0.3,0.4\},\qquad \lambda_{\mathrm{base}}=0.7 $$

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

1. 整体 accuracy 或 macro-F1 高于 fixed $\lambda=0.7$。
2. 高 sensitivity bucket 中明显优于 fixed $\lambda=0.7$。
3. 冗余下降但 label accuracy 不下降。

### 5.9 失败解释

如果冗余下降但 accuracy 下降，说明低 $\lambda$ 引入了“不同但无用”的 evidence，需要更强 relevance floor 或 coverage-aware score。

### 5.10 当前结果与结论

当前 sensitivity-gated artifact:

```text
outputs/runs/mmr_sensitivity_gated/ts0p8_tr0p3_ll0p2_basic__9a7f2ee7/infer/test/best/79d8b34809bb/api/metrics.json
```

test 结果:

| system | n | accuracy | macro-F1 | parse error | vs fixed acc | vs fixed macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| fixed $\lambda=0.7$, `top_k=5` | 1251 | 0.2702 | 0.2769 | 0.0000 | 0.0000 | 0.0000 |
| sensitivity-gated MMR | 1251 | 0.2742 | 0.2795 | 0.0000 | +0.0040 | +0.0026 |

结论: sensitivity-gated MMR 比 fixed 有轻微提升，但幅度与 $\log(n)$ heuristic 同一量级，尚不足以证明 sensitivity gating 是强策略。它可以保留为“可解释弱 adaptive baseline”，但不建议继续只围绕阈值做大规模搜索。后续若使用它，更适合作为 DPO/reference 或分桶分析对象，而不是主方法。

## 6. 实验 4: soft-label $\lambda$ policy

### 6.1 目的

修复 hard oracle $\lambda$ predictor 的噪声监督问题。

### 6.2 假设

完整 utility curve 比单点 $\arg\max$ $\lambda$ 更稳定。低 margin 样本应贡献软标签或低权重，而不是强制模型学习一个随机 hard label。

### 6.3 训练目标

对每条 claim 已有:

$$ \{U_i(\lambda_j)\}_{j=1}^{m} $$

构造 soft target:

$$ q_i(\lambda_j) = \frac{\exp\left(U_i(\lambda_j)/\tau\right)} {\sum_{k=1}^{m}\exp\left(U_i(\lambda_k)/\tau\right)} $$

样本权重:

$$ w_i = \max_{\lambda\in\Lambda}U_i(\lambda) - U_i(0.7) $$

或:

$$ w_i = U_i^{(1)} - U_i^{(2)} $$

损失:

$$ \mathcal{L} = \sum_i w_i\, \operatorname{KL} \left( q_i(\lambda)\,\|\,p_\theta(\lambda\mid x_i) \right) $$

### 6.4 $\lambda$ 网格

建议先使用粗网格:

$$ \Lambda = \{0.1,0.3,0.5,0.7,0.9\} $$

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

- $\operatorname{Jaccard}(S_{0.3},S_{0.7})$
- $\operatorname{Jaccard}(S_{0.1},S_{0.9})$
- $\operatorname{meanRel}(S_{0.7})-\operatorname{meanRel}(S_{0.3})$
- $\operatorname{meanRed}(S_{0.7})-\operatorname{meanRed}(S_{0.3})$
- $\operatorname{KendallDistance}(\operatorname{rank}_{0.3},\operatorname{rank}_{0.7})$
- $\operatorname{changes}(\{S_\lambda:\lambda\in\Lambda\})$

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

$$ \lambda_{\mathrm{argmax}} = \arg\max_{\lambda\in\Lambda}p_\theta(\lambda\mid x) $$

$$ \lambda_{\mathrm{expected}} = \sum_{\lambda\in\Lambda}p_\theta(\lambda\mid x)\lambda $$

$$ \lambda_{\mathrm{sample}} \sim p_\theta(\lambda\mid x;\,T=0.5) $$

主评估建议用 expected $\lambda$ 或 $\arg\max$ $\lambda$。

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
2. 比 fixed $\lambda=0.7$ 或 $\log(n)$ heuristic 至少在某些 bucket 中更好。
3. predicted distribution 不应坍缩到单一类别。
4. Calibration 可接受，低 confidence 样本应多为低 margin 样本。

### 6.10 决策

如果 soft-label 仍无收益，但 probability distribution 可作为 reference policy，则进入 DPO。若完全坍缩，则 DPO 的 reference policy 使用 fixed $\lambda=0.7$ 或 sensitivity-gated MMR。

### 6.11 当前结果与结论

已修复 `compute_oracle_lambda.py` 的 label-token logprob 计算，并重跑 oracle logprobs。重跑后的训练数据不再有 `-100` sentinel:

| model dir | train sentinel rows | val sentinel rows | train target entropy | val target entropy |
|---|---:|---:|---:|---:|
| `outputs/rl_mmr/soft_label/lightgbm_recomputed` | 0 | 0 | 1.6030 | 1.5968 |
| `outputs/rl_mmr/soft_label/lr_recomputed` | 0 | 0 | 1.6030 | 1.5968 |
| `outputs/rl_mmr/soft_label/mlp_recomputed` | 0 | 0 | 1.6030 | 1.5968 |

注意: $\log(5)=1.6094$。soft target entropy 接近均匀分布，说明修复 oracle 后 utility curve 本身非常平，监督信号弱。

val 离线回顾式评估:

| model | fixed utility | argmax delta | expected delta | sample delta | val KL |
|---|---:|---:|---:|---:|---:|
| LightGBM recomputed | -1.5518 | -0.0291 | -0.0001 | -0.0099 | 0.0157 |
| Logistic Regression recomputed | -1.5518 | -0.0304 | -0.0001 | -0.0120 | 0.0206 |
| MLP recomputed | -1.5518 | -0.0284 | -0.0001 | -0.0106 | 0.0150 |

分桶观察:

- `oracle_margin >= 0.10` 的样本中，argmax 有正收益；但低 margin 样本数量更多，整体抵消并转为负收益。
- `expected` 基本等价于 fixed $\lambda=0.7$，因为模型输出接近均匀分布，期望 $\lambda$ 约落在 $0.49$ 到 $0.50$。
- `sample` 增加随机性但不增加 utility。

决策: 不跑 Stage 6。当前 soft-label policy 修复了旧 oracle 的异常值问题，但没有提供可转化为下游收益的 claim-level scalar $\lambda$ 信号。若为了补实验闭环，最多只运行 `LightGBM + expected`，但不建议投入完整 build -> train -> infer 成本。

后续建议: supervised scalar $\lambda$ 路线在此停止。若继续推进 RL-MMR，应转向 trajectory-level preference learning 或 multi-weight MMR，让学习目标从“预测单个 claim-level $\lambda$”转为“选择 evidence set/trajectory 中真正有 utility gap 的行为”。

## 7. 实验 5: PAMM-lite / DPO step-wise $\lambda$ policy

### 7.1 目的

把学习粒度从 claim-level scalar $\lambda$ 推进到 step-wise $\lambda$ trajectory，并用 evidence set preference 训练。

### 7.2 核心假设

训练模型偏好高 utility trajectory，比预测 hard oracle $\lambda$ 更稳。因为低 gap 样本可以过滤，高 gap pair 可以提供更清晰监督。

### 7.3 Trajectory 定义

$$ \tau = \left((\lambda_1,d_1),\ldots,(\lambda_K,d_K)\right) $$

每一步:

$$ \lambda_t \sim \pi_\theta(\lambda\mid s_t) $$

$$ d_t = \operatorname{MMRSelect}(\lambda_t,c,C,S_{t-1}) $$

$$ S_t = S_{t-1}\cup\{d_t\} $$

state:

$$ s_t = \{c,C,S_{t-1},t,\operatorname{Rel},\operatorname{Sim}, \mathrm{pool\_stats},\mathrm{selected\_stats}\} $$

### 7.4 Trajectory 生成

先使用手工 schedules 和随机 schedules 生成候选轨迹。

建议 schedules:

- $\left[0.7,0.7,0.7,0.7,0.7\right]$
- $\left[0.3,0.3,0.3,0.3,0.3\right]$
- $\left[0.5,0.5,0.5,0.5,0.5\right]$
- $\left[0.9,0.7,0.5,0.3,0.3\right]$
- $\left[1.0,0.7,0.5,0.3,0.1\right]$
- $\left[0.5,0.5,0.7,0.7,0.9\right]$
- $\left[0.7,0.5,0.3,0.5,0.7\right]$

再加入随机 schedules:

$$ \lambda_t \sim \operatorname{Uniform}\left(\{0.1,0.3,0.5,0.7,0.9\}\right) $$

每条 claim 先生成 20 到 50 条 trajectories。

### 7.5 Preference pair 构造

对每条 trajectory 计算:

$$ U(\tau)=U\left(c,S_K(\tau)\right) $$

构造 pair:

$$ \tau^+ \succ \tau^- \quad\text{if}\quad U(\tau^+)-U(\tau^-)\ge \delta $$

建议:

$$ \delta\in\{0.02,0.05,0.10\},\qquad \mathrm{max\_pairs\_per\_claim}=10 $$

优先保留:

- 与 fixed $\lambda=0.7$ 选择结果不同的 pair
- utility gap 大的 pair
- redundancy 差异大的 pair
- verifier logprob 差异大的 pair

### 7.6 Policy 模型

动作空间:

$$ \Lambda=\{0.1,0.3,0.5,0.7,0.9\} $$

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

$$ \mathcal{L}_{\mathrm{DPO}} = -\log\sigma\left( \beta\left[ \left(\log\pi_\theta(\tau^+) - \log\pi_\theta(\tau^-)\right) - \left(\log\pi_{\mathrm{ref}}(\tau^+) - \log\pi_{\mathrm{ref}}(\tau^-)\right) \right]\right) $$

其中:

$$ \log\pi_\theta(\tau) = \sum_{t=1}^{K}\log\pi_\theta(\lambda_t\mid s_t) $$

Reference policy:

| ref | 说明 |
|---|---|
| fixed $\lambda=0.7$ | 最稳定 |
| $\log(n)$ heuristic | 若实验 2 有收益 |
| sensitivity-gated | 若实验 3 有收益 |
| soft-label policy | 若实验 4 有收益 |

也可测试 margin loss:

$$ \mathcal{L}_{\mathrm{margin}} = \max\left( 0,\, \Delta U - \left(\log\pi_\theta(\tau^+) - \log\pi_\theta(\tau^-)\right) \right) $$

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

1. 超过 fixed $\lambda=0.7$。
2. 超过 sensitivity-gated MMR 或 soft-label policy。
3. step-wise $\lambda$ distribution 有合理结构，例如第一步偏高 $\lambda$，后续可降低。
4. 高 sensitivity bucket 或高 redundancy bucket 中提升更明显。
5. 不显著增加 cost。

### 7.11 失败分析

若失败，检查:

- trajectory pool 是否多样。
- preference pair 的 utility gap 是否足够。
- reward 是否过度依赖 logprob 尾部值。
- policy 是否坍缩到固定 $\lambda$。
- MMR 选择是否对 $\lambda$ schedule 真的敏感。

### 7.12 当前实现

实现时间: 2026-05-15 ~ 2026-05-16。

已实现完整的 5 阶段 pipeline:

| 阶段 | 脚本 | 输出 |
|---|---|---|
| Trajectory 生成 | `scripts/rl_mmr/generate_trajectories.py` | `trajectories_{split}.jsonl` |
| Utility 计算 | `scripts/rl_mmr/compute_trajectory_utility.py` | 同上 + `utility` 字段 |
| Preference Pair 构造 | `scripts/rl_mmr/build_preference_pairs.py` | `{split}_pairs.npz` |
| DPO 训练 | `scripts/rl_mmr/train_dpo_step_lambda.py` | `model_best.pt` |
| 评估 | `scripts/rl_mmr/evaluate_dpo_step_lambda.py` | eval metrics |

核心模块:

| 模块 | 说明 |
|---|---|
| `src/fact_checking/retrieval/mmr.py` | 新增 `maximal_marginal_relevance_stepwise()` |
| `src/fact_checking/rl_mmr/trajectory.py` | `Trajectory`, `MMRStep`, `PreferencePair` dataclasses |
| `src/fact_checking/rl_mmr/step_features.py` | 每步 state 特征提取（最终版 13 维: 12 step + prev_lambda） |
| `src/fact_checking/rl_mmr/dpo_policy.py` | `StepLambdaPolicy` (MLP), `FixedReferencePolicy`, `dpo_loss()` |
| `src/fact_checking/rl_mmr/dpo_selector.py` | `select_candidates_dpo_stepwise()`, build pipeline 集成 |
| `src/fact_checking/build/candidates.py` | `run_build()` 新增 `dpo_stepwise` 模式分支 |
| `configs/experiment/mmr_dpo_step_lambda.yaml` | Hydra 实验配置（继承 b3） |

### 7.13 实验数据与配置

训练配置:

- Chunk-MMR cache: `432dfc970e75` (b3_mmr_topk_sweep_1024, semantic chunking, top_k=32)
- Verifier: Qwen2.5-7B-Instruct LoRA (b3 checkpoint, `79d8b34809bb`)
- Evidence budget: `K=5`
- λ 离散动作空间: `Λ = {0.1, 0.3, 0.5, 0.7, 0.9}`
- Trajectory: 7 手工 schedule + 30 随机 schedule = 37 per claim
- Train claims: ~10,000, total trajectories: ~372,000
- Preference pairs (train): 78,510 → 76,794 (sentinel 过滤后)
- Preference pairs (val): 11,030 → 11,010 (sentinel 过滤后)

### 7.14 关键数据分析

#### 7.14.1 Utility 分布

```
Utility (trajectories): mean=-7.646 std=16.968
  min=-100.000 max=0.000
  p1=-100.000 p5=-18.003 p50=-2.540
  Values < -50 (sentinel): 11,111 (2.98%)
```

发现 2.98% 的 trajectory utility 为 oracle logprob 文件中的 `-100` sentinel 值。这些无效 utility 在 preference pair 中产生 >98 的巨大 gap，污染了训练数据。修复: 在 `compute_trajectory_utility.py` 和 `build_preference_pairs.py` 中过滤 `utility <= -99` 的条目。

#### 7.14.2 Utility 信号质量

```
Per-claim utility range: mean=3.36 median=2.34
  p10=0.00 p25=0.16 p75=5.08 p90=8.63
Claims where best non-0.7 > fixed: 63.2%
Claims where fixed λ=0.7 IS best: 36.8%
Oracle (best) λ per claim:
  λ≈0.3: 20.9%  λ≈0.5: 39.9%  λ≈0.7: 39.1%
```

**结论: 信号存在**。不同 λ schedule 确实导致有意义的 utility 差异，63.2% 的 claim 存在优于 fixed 0.7 的方案。

#### 7.14.3 Feature 诊断

20 维原始特征 (8 pool + 12 step) 的 winner/loser 差异分析:

```
Step 0 (n_diff=61243): ALL dims 0-19 have |diff| = 0.0000
Step 1 (n_diff=64198): dims 0-17 = 0.0000, dim 18 (top_mmr_score) = 0.56, dim 19 = 0.04
Steps 2-4: Same pattern — only dims 18-19 (top_mmr_score, mmr_score_gap) have non-zero diff
```

Pool features (8 dims) 在同一 claim 的 winner/loser 间 **完全相同**，是纯噪声。这也是为什么去掉 pool features 后仍无效——step features 中只有 2 维 (top_mmr_score, mmr_score_gap) 在 winner/loser 间有明显差异，其余 10 维差异也很小。

Supervised λ prediction (Logistic Regression):

```
Step 0: train_acc=0.311 test_acc=0.305 baseline=0.296  ← 几乎不可预测
Steps 1-4: train_acc=0.989-0.999 test_acc=0.989-0.998  ← 近乎完美
```

Steps 1-4 的 99% accuracy **不是**模型学到了"什么状态选什么 λ 更好"，而是循环论证——λ 的选择本身决定了后续 state（哪些 item 被选中），因此从 state 可以反推 λ，但这不包含 λ → utility 的因果信息。

### 7.15 四次 DPO 训练尝试

| 版本 | 特征维度 | β | ref temp | 关键差异 | 结果 |
|---|---|---|---|---|---|
| V1 | 20 dims (pool+step) | 1.0 | 0.3 | 原始实现，含 sentinel | λ=0.7: 99.97%, H=1.32 |
| V2 | 20 dims | 3.0 | 0.8 | 过滤 sentinel | λ=0.7: 99.87%, H=1.57 |
| V3 | 13 dims (step+prev_lambda) | 3.0 | 0.8 | 去除 pool features | λ=0.7: 99.87%, H=1.56 |
| V4 | 13 dims, claim-level (K=1) | 3.0 | 0.8 | 只用 step-0 特征，预测 majority λ | λ=0.7: **100%** |

所有版本均完全坍缩到 reference policy 中心 λ=0.7，accuracy 始终 0.52-0.53（接近随机 0.50），entropy 始终接近均匀分布上限 log(5)=1.609。

### 7.16 失败根因

经过 4 轮实验 + 3 轮诊断分析，确定 DPO step-wise λ 训练失败的**三个层级的根因**:

**层级 1 —— 数据质量**: oracle logprob 文件中存在 `-100` sentinel 值（2.98% 的 trajectory），产生虚假的巨大 utility gap。过滤后有一定改善（val loss 从 0.72 降到 0.65），但未解决根本问题。

**层级 2 —— 特征问题**: 
- Pool features (8 dims) 对同一 claim 的 winner/loser 完全相同 → 纯噪声
- Step features 中只有 `top_mmr_score` 和 `mmr_score_gap` 在 winner/loser 间有显著差异
- Step 0 时任何选择尚未做出，所有 trajectory 的状态完全一样，无法区分最优 λ
- Steps 1-4 的特征差异是 λ 选择的**结果**而非**原因**（内生性问题）

**层级 3 —— 信号本质**:
- DPO 的 reference policy 固定偏向 λ=0.7
- 最优 λ≈0.5 (39.9%) 和 λ≈0.7 (39.1%) 几乎打平
- Reference policy 对 winner/loser 的偏好往往与真实 utility 排序一致
- 只有当 winner 用非 0.7、loser 用 0.7 时，DPO 才有强 gradient 推动 policy 远离 reference
- 这种"有效训练信号"的比例太低，不足以驱动 policy 离开 reference 的吸引域
- **核心矛盾**: 信号量级 (utility gap median=2.34) 虽然 > 0，但相对于 logprob 的自然方差 (std=16.97 → 过滤后 std 仍大)，信噪比太低

### 7.17 结论与决策

**DPO step-wise λ 训练在当前设置下不可行。** 这不是实现问题，而是 utility signal 相对 reference policy 的偏离太小。这个结论与实验 1-4 的结论一致: utility curve 太平，learned λ 提供的边际收益不足以超越 fixed λ=0.7。

触发了实验计划 §15 的 stop criteria: "若 DPO policy 坍缩到 fixed 0.7 且无收益，则需要先重构 reward 或 trajectory generator。"

具体建议:
1. **停止 DPO step-wise λ 方向** (已触发 stop criteria)。
2. **不跑 GRPO refinement** (实验 7，需要 DPO 已有收益作为前置条件)。
3. **转向实验 6: multi-weight MMR policy** — 将 scalar λ 的 relevance/diversity 单轴 trade-off 扩展为多维权重向量 (relevance, redundancy, coverage, source novelty, cost)。这个方向不依赖 utility curve 的"陡峭度"，而是扩展了 selection 的优化空间本身。
4. 若实验 6 仍无收益，建议整体结论为: 在 LIAR-RAW + chunk-first MMR + Qwen2.5-7B-Instruct verifier 的设置下，learned evidence selection 无法显著超越 fixed λ=0.7 MMR baseline。

## 8. 实验 6: multi-weight MMR policy

### 8.1 目的

突破 scalar $\lambda$ 的表达能力限制，学习多维 evidence selection 权重。

### 8.2 假设

事实核查 evidence selection 不只是 relevance 与 redundancy 的单轴 trade-off，还需要 coverage、source novelty、stance diversity、cost 等因素。multi-weight policy 能更好拟合 set-level utility。

### 8.3 打分函数

$$ \begin{aligned} \operatorname{Score}(d\mid s_t) = {}& w_{\mathrm{rel},t}\operatorname{Rel}(c,d) - w_{\mathrm{red},t}\operatorname{Red}(d,S_{t-1}) \\ &+ w_{\mathrm{cov},t}\operatorname{Cov}(d,S_{t-1},c) + w_{\mathrm{src},t}\operatorname{SrcNovelty}(d,S_{t-1}) \\ &+ w_{\mathrm{stance},t}\operatorname{StanceNovelty}(d,S_{t-1}) - w_{\mathrm{cost},t}\operatorname{Cost}(d) \end{aligned} $$

每一步:

$$ w_t = g_\theta(s_t) $$

$$ d_t = \arg\max_{d\in C_N(c)\setminus S_{t-1}}\operatorname{Score}(d\mid s_t) $$

### 8.4 Feature definitions

#### Relevance

$$ \operatorname{Rel}(c,d) = \text{hybrid retrieval score or reranker score} $$

#### Redundancy

$$ \operatorname{Red}(d,S) = \max_{s\in S}\operatorname{Sim}(d,s) $$

#### Coverage proxy

可选实现:

$$ \operatorname{Cov}(d,S,c) = 1-\max_{s\in S}\operatorname{Sim}_{\mathrm{claim\_aspect}}(d,s) $$

或基于 claim subcomponents:

$$ \operatorname{Cov}(d,S,c) = \#\{\text{new claim entities, numbers, or time expressions covered by }d\} $$

#### Source novelty

若有 report id 或 source metadata:

$$ \operatorname{SrcNovelty}(d,S) = \begin{cases} 1, & \operatorname{source}(d)\notin \operatorname{source}(S),\\ 0, & \text{otherwise}. \end{cases} $$

若无 source metadata，可用 report id 或 chunk parent id 代替。

#### Stance novelty

初期可不做复杂 stance classifier。可先用 simple proxy:

$$ \operatorname{StanceNovelty} = \operatorname{Diversity}\left(\text{predicted entailment/refutation/neutral proxy}\right) $$

若没有 stance signal，则先省略该项，避免引入噪声。

#### Cost

$$ \operatorname{Cost}(d) = \frac{\operatorname{token\_count}(d)}{\operatorname{max\_token\_count}} $$

### 8.5 Policy 输出约束

为保持可解释性，建议使用非负权重:

$$ w_t = \operatorname{softplus}(\tilde{w}_t) $$

或归一化:

$$ w_t = \operatorname{softmax}(\tilde{w}_t) $$

如果使用 softmax，需要注意负项与正项分别归一化，避免符号混乱。

### 8.6 训练方式

先复用实验 5 的 preference framework。把 trajectory 中的 action 从 $\lambda_t$ 换成 $w_t$ 或离散 weight template。

建议先使用离散 template，降低训练难度:

| template | $w_{\mathrm{rel}}$ | $w_{\mathrm{red}}$ | $w_{\mathrm{cov}}$ | $w_{\mathrm{src}}$ | $w_{\mathrm{cost}}$ |
|---|---:|---:|---:|---:|---:|
| relevance-first | 1.0 | 0.2 | 0.1 | 0.0 | 0.0 |
| balanced | 0.7 | 0.4 | 0.3 | 0.1 | 0.0 |
| diversity-heavy | 0.5 | 0.7 | 0.4 | 0.2 | 0.0 |
| coverage-heavy | 0.6 | 0.3 | 0.8 | 0.2 | 0.0 |
| cost-aware | 0.7 | 0.4 | 0.3 | 0.1 | 0.3 |

Policy 先选择 template:

$$ a_t = \mathrm{template\_id} $$

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

- $\operatorname{Rel}+\operatorname{Red}$
- $\operatorname{Rel}+\operatorname{Red}+\operatorname{Cov}$
- $\operatorname{Rel}+\operatorname{Red}+\operatorname{Cov}+\operatorname{SrcNovelty}$
- $\operatorname{Rel}+\operatorname{Red}+\operatorname{Cov}+\operatorname{SrcNovelty}+\operatorname{Cost}$

若 stance proxy 可用，再加:

$$ \operatorname{Rel} + \operatorname{Red} + \operatorname{Cov} + \operatorname{SrcNovelty} + \operatorname{StanceNovelty} + \operatorname{Cost} $$

### 8.9 成功标准

1. 超过 DPO step-wise $\lambda$。
2. 在复杂 claim、高冗余候选池、跨 source evidence 任务上提升更明显。
3. 能降低 redundancy，同时不牺牲 relevance 或 verdict accuracy。
4. 权重分布可解释，不应所有样本坍缩到同一 template。

## 9. 实验 7: GRPO refinement

### 9.1 目的

在 DPO 或 multi-weight policy 已有收益的基础上，进一步直接优化 reward。

### 9.2 前置条件

只有满足以下条件才开始 GRPO:

1. DPO step-wise $\lambda$ 或 multi-weight policy 在 dev set 上稳定超过 fixed $0.7$。
2. Reward 定义稳定，不依赖少数 logprob 极端值。
3. 每条 claim 能生成多个具有差异的 trajectories。
4. Policy 未坍缩到单一 action。

### 9.3 训练流程

$$ \tau_1,\ldots,\tau_G \sim \pi_\theta(\tau\mid c) $$

$$ R_i = R\left(c,S_K(\tau_i)\right),\qquad i=1,\ldots,G $$

$$ A_i = \frac{R_i-\operatorname{mean}(\{R_j\}_{j=1}^{G})} {\operatorname{std}(\{R_j\}_{j=1}^{G})+\epsilon} $$

然后使用 group-relative objective 更新 $\pi_\theta$，并加入到 $\pi_{\mathrm{ref}}$ 的 KL penalty。

### 9.4 Reward

推荐 reward:

$$ \begin{aligned} R ={}& w_{\mathrm{label}}\operatorname{LabelUtility}  + w_{\mathrm{ev}}\operatorname{EvidenceUtility}  + w_{\mathrm{cov}}\operatorname{Coverage}  + w_{\mathrm{div}}\operatorname{Diversity} \\ & - w_{\mathrm{red}}\operatorname{Redundancy}  - w_{\mathrm{cost}}\operatorname{Cost} \end{aligned} $$

必须做 reward clipping 或 normalization:

$$ R_{\mathrm{clipped}} = \operatorname{clip}(R,r_{\min},r_{\max}) $$

### 9.5 KL 约束

Reference policy 使用 dev 表现最好的 offline policy:

$$ \pi_{\mathrm{ref}} = \text{best DPO policy or best multi-weight policy} $$

KL penalty:

$$ \mathcal{L}_{\mathrm{KL}} = \beta_{\mathrm{KL}}\operatorname{KL} \left(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\right) $$

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

$$ G\in\{4,8\},\qquad \eta\in[10^{-5},5\times10^{-5}],\qquad \beta_{\mathrm{KL}}\in[0.01,0.1] $$

$$ R_{\mathrm{clip}}\in\{[-2,2],[0,1]\},\qquad \mathrm{entropy\_bonus}=\text{small optional value} $$

$$ \mathrm{max\_updates}=\text{small budget with early stopping on dev utility} $$

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
| sensitivity | low、medium、high based on $1-\operatorname{Jaccard}(S_{0.3},S_{0.7})$ |
| pool redundancy | low、medium、high based on mean pairwise Sim |
| oracle margin | low margin、high margin |
| label type | 各 veracity label |
| claim length | short、medium、long |
| evidence overlap shift | selected set 与 fixed $0.7$ 相同或不同 |

重点观察:

$$ \text{gain}(\mathrm{adaptive\ policy}) \text{ 是否主要集中在 high-sensitivity 与 high-redundancy buckets} $$

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

1. fixed $\lambda = 0.7$
2. $\log(n_{\mathrm{candidates}})$ heuristic
3. sensitivity-gated MMR

目标: 在 3 个低成本方法中确定最强非学习或弱学习 baseline。

### Phase B: Supervised repair

包括:

4. soft-label $\lambda$ policy

目标: 判断 supervised $\lambda$ learning 是否可以通过 soft target 与 interventional features 修复。

### Phase C: Offline preference learning

包括:

5. PAMM-lite / DPO step-wise $\lambda$ policy
6. multi-weight MMR policy

目标: 验证 trajectory-level preference learning 是否优于 fixed 与 heuristic。

### Phase D: Online or semi-online refinement

包括:

7. GRPO refinement

目标: 在 DPO 已有收益的前提下进一步优化 reward。

## 13. 最终主表建议

| System | Accuracy | Macro-F1 | Correct label logprob | Utility | Redundancy | Cost | High-sensitivity Acc | High-redundancy Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed $\lambda=0.7$ | 0.2702 | 0.2769 | | locked baseline | | | | |
| $\log(n)$ heuristic | 0.2766 | 0.2799 | | weak positive, small delta | | | | |
| sensitivity-gated | 0.2742 | 0.2795 | | weak positive, small delta | | | | |
| soft-label $\lambda$ | Stage 6 skipped | Stage 6 skipped | | val expected delta about -0.0001; argmax/sample worse | | | | |
| DPO step-wise $\lambda$ | **坍缩到 fixed** | **坍缩到 fixed** | | 4 轮训练全部坍缩到 λ=0.7; 不跑完整 build→train→infer | | | | |
| multi-weight MMR | | | | | | | | |
| GRPO refine | **不跑** (前置条件未满足) | | | | | | | |

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

### 停止普通 hard $\lambda$ prediction

若模型继续接近均值预测或输出方差坍缩，则停止该方向。

### 停止 soft-label $\lambda$

若 soft-label policy 在 dev set 上不超过 $\log(n)$ heuristic，且预测分布坍缩，则停止 supervised $\lambda$ 路线。

当前已触发停止条件的弱化版本: 重跑 oracle 后预测分布没有 hard collapse，但 target distribution 接近均匀，`expected` 退化为 fixed，`argmax/sample` 均低于 fixed。因此停止 claim-level supervised scalar $\lambda$，不跑 Stage 6。

### 停止 DPO step-wise

若 preference pairs 中高 gap pair 数量不足，或 DPO policy 坍缩到 fixed $0.7$ 且无收益，则需要先重构 reward 或 trajectory generator。

**当前已触发停止条件:** 经过 4 轮训练 (V1-V4)，policy 在 13 维优化特征和 claim-level 简化设置下均 100% 坍缩到 λ=0.7。utility 信号存在但相对 reference policy 的偏离太小，DPO gradient 不足以驱动 policy 离开 reference 的吸引域。停止该方向，不跑完整 build→train→infer 流程。

### 停止 GRPO

若 GRPO dev utility 不升、policy entropy 快速下降、或 accuracy 下降，应回退到 DPO checkpoint。

**当前不跑 GRPO:** 前置条件 (DPO step-wise 或 multi-weight policy 在 dev set 上稳定超过 fixed 0.7) 未满足。

## 16. 最终建议

按本计划推进时，最关键的不是让每一步都超过前一步，而是让每一步回答一个明确问题:

1. $\log(n_{\mathrm{candidates}})$ 能否提供最低成本 adaptive 信号。→ **能，但收益太小 (+0.0064)。**
2. sensitivity-gated MMR 能否利用 $\lambda$ 敏感性。→ **能，但收益太小 (+0.0040)。**
3. soft-label $\lambda$ 能否修复 hard oracle 标签噪声。→ **修复了噪声，但 utility curve 太平，预测退化。**
4. DPO step-wise policy 能否利用 trajectory preference。→ **不能。4 轮训练全部坍缩到 fixed，信噪比太低。**
5. multi-weight MMR 是否解决 scalar $\lambda$ 表达能力不足。
6. GRPO 是否能在稳定 offline policy 上进一步提升。

若最终结果是 multi-weight DPO 明显优于 fixed $0.7$，而 GRPO 没有进一步提升，这仍然是一个合理且完整的研究结论。
