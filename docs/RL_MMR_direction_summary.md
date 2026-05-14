# RL-MMR 后续研究方向整理

生成日期: 2026-05-14

本文档整理当前可学习 MMR 研究的阶段性判断与后续路线。本文只整理已有讨论、当前实验诊断与方法决策，不展开论文内容。

## 1. 当前研究状态

你的研究目标可以从“学习一个最优 lambda”调整为更稳健的表述:

> 学习一个 claim-adaptive 或 state-adaptive 的 evidence diversity policy，使 evidence set 同时兼顾相关性、覆盖度、低冗余、下游 verifier utility 与成本。

目前已有实验可概括为三点。

第一，`System 2: fixed-MMR` 已经显示 `lambda = 0.7` 是一个强基线。在多数情况下，它能在 relevance 与 diversity 之间取得稳定平衡。因此，`lambda = 0.7` 不应被视为容易击败的弱 baseline，而应作为后续所有方法的主比较对象。

第二，`System 3: learned-lambda MMR, non-RL` 的 oracle 上界有价值。oracle lambda 相比 fixed `lambda = 0.7` 在验证集上有约 3.1% 的 accuracy 提升，说明 adaptive evidence selection 不是无意义的。

第三，当前 hard oracle lambda predictor 基本失败。chunk embedding regression、73 维 handcrafted feature regression、3-bin classification 都接近均值预测。预测器输出方差明显收缩，不能有效恢复 oracle lambda 的分布。

## 2. 当前失败的核心诊断

当前 learned-lambda predictor 的失败不应简单归因于模型太弱或特征不够复杂。更准确的诊断是: hard oracle lambda 不是一个稳定监督目标。

主要依据如下。

1. Oracle lambda 曲面高度平坦。大量 claim 中，最优 lambda 与次优 lambda 的 logprob 差异很小。此时 hard argmax lambda 只是平坦曲面上的一个不稳定点。
2. 对很多 claim，不同 lambda 产生的 evidence set 对 SFT verifier 的影响很弱。即使选错 lambda，utility 变化也不大。
3. BGE 文本 embedding 编码的是语义内容，但 oracle lambda 实际上反映的是 verifier 对不同 evidence ordering 或 evidence set 的响应敏感度。两者之间没有足够稳定的可泛化映射。
4. 候选数量 `n_candidates` 是少数可观察到的弱信号。`log(n_candidates)` 线性回归虽简单，但表现已经接近复杂神经预测器。
5. 高 margin 子集呈现出一个重要现象: 当 lambda 真正有影响时，最优 lambda 明显偏低，即更偏向 diversity。这提示后续方法应关注“什么时候需要更强 diversity”，而不是强行精确预测每条 claim 的单点 lambda。

因此，下一阶段不建议继续堆叠普通文本特征或更复杂 embedding 模型来预测 hard oracle lambda。

## 3. 研究方向调整

建议从以下问题:

```text
给定 claim 和候选池，预测最优 scalar lambda
```

调整为:

```text
给定 claim、候选池和已选 evidence 状态，学习一个 evidence selection policy，使高 utility evidence trajectory 的概率高于低 utility trajectory
```

这个调整有三个好处。

1. 避免把平坦 utility 曲面压缩成噪声 hard label。
2. 可以只使用 reward gap 明显的 preference pairs，过滤无信息样本。
3. 可以自然扩展到 step-wise lambda、多权重 MMR 与 GRPO refinement。

## 4. 后续探索顺序

你计划按以下顺序探索是合理的:

1. fixed `lambda = 0.7`
2. `log(n_candidates)` heuristic
3. sensitivity-gated MMR
4. soft-label lambda policy
5. PAMM-lite / DPO step-wise lambda policy
6. multi-weight MMR policy
7. GRPO refinement

这个顺序的优点是从低成本、低方差、可解释的启发式逐步推进到更强但更复杂的 policy learning。每一步都能回答一个明确问题。

## 5. 各阶段定位

### 5.1 fixed `lambda = 0.7`

定位: 强 baseline。

作用:

- 固定当前最稳的 MMR 设置。
- 作为所有后续方法的主比较对象。
- 生成统一的 selected evidence、verifier outputs、accuracy、logprob、redundancy 和 cost 指标。

核心判断标准:

> 后续方法若不能稳定超过 `lambda = 0.7`，就不能说明 adaptive policy 有实际价值。

### 5.2 `log(n_candidates)` heuristic

定位: 极简 adaptive baseline。

当前分析显示，候选数量是少数能解释 oracle lambda 的弱信号之一。可以先使用:

```text
lambda = clip(a * log(n_candidates) + b, lambda_min, lambda_max)
```

初始可用经验形式:

```text
lambda = clip(-0.073 * log(n_candidates) + 0.613, 0.0, 1.0)
```

也可以在 dev set 上重新拟合 `a` 和 `b`。

作用:

- 验证极简 claim-adaptive 规则是否能超过 fixed lambda。
- 为后续 neural policy 提供一个强于随机的 reference policy。
- 作为 sanity check: 如果该方法也无收益，说明当前数据与 verifier 可能对 lambda 变化不敏感。

### 5.3 sensitivity-gated MMR

定位: 无监督或弱监督的 adaptive diversity baseline。

核心思想不是预测 oracle lambda，而是判断当前 claim 是否对 lambda 敏感。

可定义:

```text
S_low  = MMR(lambda = 0.3)
S_base = MMR(lambda = 0.7)
Sensitivity = 1 - Jaccard(S_low, S_base)
```

也可加入 order-level 差异、pairwise redundancy、candidate score entropy、top-k score gap 等信号。

简单策略:

```text
if Sensitivity >= threshold and candidate_pool_redundancy >= threshold:
    lambda = 0.3
else:
    lambda = 0.7
```

作用:

- 直接利用“高 margin 样本偏好低 lambda”的发现。
- 避免训练一个噪声 hard-label predictor。
- 成本低，解释性强。

### 5.4 soft-label lambda policy

定位: 对 System 3 的合理修复。

不要再把每条 claim 的目标写成单点 `lambda*`。应使用完整 utility curve 构造 soft target:

```text
q_i(lambda) = softmax(U_i(lambda) / temperature)
```

训练目标:

```text
L = weighted_cross_entropy(q_i(lambda), p_theta(lambda | features))
```

样本权重可设为:

```text
w_i = max_lambda U_i(lambda) - U_i(0.7)
```

或使用 top1-top2 margin、sensitivity score。

特征不建议只用 BGE embedding。更应加入 interventional features:

- `Jaccard(S_0.3, S_0.7)`
- `Jaccard(S_0.0, S_1.0)`
- 不同 lambda 下 selected evidence 的 overlap
- 不同 lambda 下 mean relevance 差异
- 不同 lambda 下 mean redundancy 差异
- candidate pool pairwise similarity 分布
- score entropy
- top-k score gap
- MMR top-1 或 top-K 是否变化

作用:

- 让低 margin 样本自然变成平滑标签，而不是噪声 hard label。
- 让模型学习“偏好分布”，而不是学习不稳定 argmax。
- 作为 DPO step-wise policy 的 warm start 或 reference。

### 5.5 PAMM-lite / DPO step-wise lambda policy

定位: 下一阶段主线。

核心思想: 不再学习单个 lambda，而是学习 evidence trajectory preference。

Trajectory 可写为:

```text
tau = ((lambda_1, d_1), (lambda_2, d_2), ..., (lambda_K, d_K))
```

其中每一步:

```text
lambda_t ~ pi_theta(lambda | c, C, S_{t-1}, t)
d_t = MMR_select(lambda_t, c, C, S_{t-1})
```

对同一 claim 生成多条 trajectory，并用 utility 打分:

```text
U(c, S_K) = w1 * VerdictUtility
          + w2 * EvidenceUtility
          + w3 * Coverage
          - w4 * Redundancy
          - w5 * Cost
```

构造 preference pair:

```text
tau_plus > tau_minus if U(tau_plus) - U(tau_minus) >= delta
```

可使用 DPO loss 或 margin loss:

```text
L_margin = max(0, DeltaU - [log pi_theta(tau_plus) - log pi_theta(tau_minus)])
```

```text
L_DPO = -log sigmoid(beta * ([log pi_theta(tau_plus) - log pi_theta(tau_minus)]
                             - [log pi_ref(tau_plus) - log pi_ref(tau_minus)]))
```

作用:

- 直接优化 evidence set utility。
- 只训练 reward gap 明显的 pair，降低低信号样本干扰。
- 学习 step-wise diversity schedule，例如前期偏 relevance，后期偏 diversity。
- 保留 MMR 的可解释 inductive bias。

### 5.6 multi-weight MMR policy

定位: 强化表达能力的主方法扩展。

Scalar lambda 只能表达 relevance 与 redundancy 的单轴权衡。事实核查 evidence selection 往往还涉及 coverage、source novelty、stance diversity、time、cost 等多维因素。因此可扩展为:

```text
Score(d | s_t) = w_rel,t * Rel(c,d)
               - w_red,t * Red(d,S_{t-1})
               + w_cov,t * Cov(d,S_{t-1},c)
               + w_src,t * SrcNovelty(d,S_{t-1})
               + w_stance,t * StanceNovelty(d,S_{t-1})
               - w_cost,t * Cost(d)
```

其中:

```text
w_t = g_theta(c, C, S_{t-1}, t)
```

作用:

- 解决单一 lambda 表达能力不足的问题。
- 允许模型区分 semantic redundancy 与 independent-source corroboration。
- 更适合复杂 claim、多跳 claim、冲突证据与高冗余候选池。

### 5.7 GRPO refinement

定位: 最后阶段 refinement，不建议过早开始。

流程:

```text
for each claim:
    sample G trajectories from current policy
    compute reward R_1, ..., R_G
    normalize advantage within group
    update policy with KL constraint to reference policy
```

组内 advantage:

```text
A_i = (R_i - mean(R_1, ..., R_G)) / (std(R_1, ..., R_G) + epsilon)
```

作用:

- 在 DPO policy 基础上进一步直接优化 reward。
- 使用同一 claim 的多 trajectory 组内比较，降低跨 claim reward 尺度差异。
- 适合在 reward 已相对稳定、DPO 已有收益后再做。

## 6. 推荐 utility 设计

建议不要只使用正确 label token logprob。它可以保留，但应作为 composite utility 的一部分。

推荐主 utility:

```text
U = w_label * LabelCorrectOrLogprob
  + w_ev * EvidenceOverlapOrRecall
  + w_cov * Coverage
  + w_div * SourceOrSemanticDiversity
  - w_red * Redundancy
  - w_cost * Cost
```

若 gold evidence 不完整，可以并行记录两套 utility:

1. gold-assisted utility: 使用 gold evidence overlap、gold label。
2. verifier-assisted utility: 使用 verifier logprob、verdict correctness、confidence margin。

后续 DPO 与 GRPO 应优先使用 reward gap 足够大的 pair。

## 7. 评价原则

所有系统必须共享以下条件:

- 相同 train/dev/test split。
- 相同 candidate pool。
- 相同 final evidence budget `K`。
- 相同 verifier 或 SFT checkpoint。
- 相同 prompt template 与 inference setting。
- 相同 evaluation script。

主指标:

- verdict accuracy
- macro-F1
- correct label logprob
- evidence recall 或 evidence F1
- mean pairwise similarity
- evidence redundancy
- source diversity
- coverage proxy
- token cost
- latency
- joint utility

必须做 paired comparison，因为方法之间通常只改变 evidence selection，claim 集合相同。

建议分桶:

- candidate count
- candidate redundancy
- sensitivity score
- oracle margin
- label type
- claim length
- number of entities
- evidence set overlap with fixed `lambda = 0.7`

## 8. 关键决策门槛

建议按以下 gate 推进。

### Gate 1: heuristic 是否有收益

比较:

```text
fixed 0.7 vs log(n_candidates) heuristic vs sensitivity-gated MMR
```

若 sensitivity-gated MMR 无法在 dev set 上超过 fixed 0.7，说明当前 adaptive lambda 信号较弱。此时仍可做 soft-label 和 DPO，但应降低预期，把重点放在 pairwise trajectory learning 是否能提取非线性信号。

### Gate 2: soft-label 是否修复 System 3

比较:

```text
hard oracle predictor vs soft-label lambda policy
```

若 soft-label 仍无收益，但 predicted distribution 能反映 sensitivity 或 uncertainty，也可作为 DPO reference。若完全退化为均匀或固定分布，则不应再在 supervised lambda prediction 上投入过多。

### Gate 3: DPO step-wise 是否超过 heuristic

比较:

```text
sensitivity-gated MMR vs DPO step-wise lambda policy
```

若 DPO 有收益，说明 trajectory preference learning 是正确方向。若没有收益，需要检查 preference pair 构造、reward 定义、trajectory diversity 与 policy capacity。

### Gate 4: multi-weight 是否超过 scalar lambda

比较:

```text
DPO step-wise lambda vs DPO multi-weight MMR
```

若 multi-weight 明显提升，主论文方法应从 “learned lambda” 升级为 “learned evidence diversity policy”。

### Gate 5: GRPO 是否值得

只有在 DPO policy 已经稳定超过 baseline 时再做 GRPO。若 GRPO 造成 policy collapse 或 dev reward 不升反降，应停止并保留 DPO 作为主方法。

## 9. 推荐最终论文叙事

不建议把论文主张写成:

```text
learned lambda predictor beats fixed lambda
```

更稳健的叙事是:

```text
Fixed-MMR is a strong baseline.
Hard oracle lambda is an unstable target under flat utility curves.
Adaptive evidence diversity should be learned as set-level or trajectory-level preference.
Pairwise trajectory learning and multi-weight MMR better align evidence selection with verifier utility.
```

最终主张应是:

```text
reranker + adaptive evidence diversity policy > reranker-only
```

或在当前不接 reranker 的阶段:

```text
trajectory-level adaptive MMR > fixed-MMR and hard learned-lambda predictor
```

## 10. 结论

下一步应停止沿着“预测 hard oracle scalar lambda”的路径堆模型。更合理的路线是先建立两个低成本 adaptive baseline: `log(n_candidates)` heuristic 与 sensitivity-gated MMR。随后用 soft-label lambda policy 修复 System 3，再进入 PAMM-lite / DPO step-wise trajectory preference learning。若 DPO 有收益，再升级到 multi-weight MMR。GRPO 只作为最后 refinement。
