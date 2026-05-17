# Oracle Evidence Set 监督信号使用方案与下一步实验计划

> 目的：基于当前 `oracle_set_gap_analysis.md` 的实验发现，判断是否应使用 Oracle evidence set 作为下一阶段监督信号，并给出可供后续代码 agent 执行的实验路线。  
> 核心结论：**应该使用 Oracle evidence set，但不应把它作为无条件硬标签直接蒸馏；更合理的用法是 set-level utility supervision、preference supervision 与 reward shaping。**

---

## 1. 决策摘要

当前实验已经证明，Oracle evidence set 与 fixed-MMR baseline 之间存在显著 gap：Oracle greedy evidence set 将 Accuracy 从 29.67% 提升到 48.43%，将 Macro F1 从 30.03% 提升到 43.03%，Macro F1 gap 为 13.00 个百分点。这个 gap 说明 evidence selection 本身仍有较大的可学习空间。

$$\Delta_{\mathrm{F1}}^{\mathrm{oracle\text{-}set}} = \mathrm{F1}(S_{\mathrm{oracle}}) - \mathrm{F1}(S_{\mathrm{MMR}}) = 43.03 - 30.03 = 13.00\ \mathrm{pp}$$

因此，下一阶段应从“学习 scalar lambda”转向“学习 evidence set selection policy”。Oracle evidence set 应作为训练信号，但建议以 preference / utility / reward 的形式使用，而不是对所有样本直接做硬模仿。

推荐主线如下：

1. 在 train split 上生成 Oracle evidence set。
2. 构造 filtered preference pairs：Oracle set 作为正例，fixed-MMR / reranker-only / random / low-utility set 作为负例。
3. 训练 set-aware selector 或 multi-weight MMR policy。
4. 先做 DPO 或监督式 set selection，再做 GRPO / RL refinement。
5. 并行修复 verifier calibration，尤其是 true 与 mostly-true 类别的 false bias。

---

## 2. 为什么应使用 Oracle evidence set

### 2.1 Evidence selection gap 已经足够大

当前 fixed-MMR baseline 与 Oracle evidence set 的核心结果如下：

| 指标 | Oracle evidence set | fixed-MMR lambda=0.7 | Gap |
|---|---:|---:|---:|
| Accuracy | 48.43% | 29.67% | +18.76 pp |
| Macro Precision | 57.65% | 30.39% | +27.26 pp |
| Macro Recall | 49.22% | 30.40% | +18.82 pp |
| Macro F1 | 43.03% | 30.03% | +13.00 pp |

这说明在同样候选池、同样 verifier、同样 prompt 条件下，**只要 evidence set 选得更好，系统表现就可以大幅提升**。

### 2.2 Oracle lambda 的收益远小于 Oracle set

此前 Oracle lambda 相比 fixed lambda 只带来约 +3.1% Accuracy，而 Oracle evidence set 相比 fixed-MMR 带来约 +18.8% Accuracy。这说明当前瓶颈不是“如何预测更好的 scalar lambda”，而是“单一 MMR 公式本身无法覆盖最优 evidence set”。

$$\Delta_{\mathrm{Acc}}^{\mathrm{oracle\text{-}\lambda}} \approx 33.48 - 30.40 = 3.08\ \mathrm{pp}$$

$$\Delta_{\mathrm{Acc}}^{\mathrm{oracle\text{-}set}} = 48.43 - 29.67 = 18.76\ \mathrm{pp}$$

因此，下一阶段不应继续围绕以下路径展开：

$$\lambda(c) \rightarrow \mathrm{MMR}(C, \lambda) \rightarrow S_K$$

而应转向更直接的 evidence set learning：

$$(c, C) \rightarrow S_K$$

或者 sequential evidence selection：

$$(c, C, S_{t-1}) \rightarrow d_t$$

### 2.3 Scalar lambda 路线已经被充分探索

已有实验显示，learned-lambda hard regression、soft-label lambda policy、DPO step-wise lambda policy 均失败或退化：

| 方法 | 结果 | 解释 |
|---|---|---|
| Hard oracle lambda regression | 近似均值预测 | Oracle lambda 曲面高度平坦，hard argmax 标签噪声大 |
| Soft-label lambda policy | 退化为固定 lambda 约 0.5 | Utility curve 接近均匀，soft target 无有效区分度 |
| DPO step-wise lambda | 多次坍缩至 lambda=0.7 | Reference policy 吸引域过强，step features 内生性严重 |
| Oracle lambda sweep | 有小幅收益 | 证明 adaptive 思想有效，但 scalar lambda 表达能力不足 |

结论是：**lambda 是 utility 的间接代理；Oracle evidence set 才是更直接的监督对象。**

---

## 3. 但不能直接硬蒸馏所有 Oracle set

Oracle evidence set 是高价值监督信号，但不是无偏标签。它是相对于当前 verifier、当前 prompt、当前候选池、当前 label format 的 privileged label。若直接硬蒸馏，可能把 verifier 的偏差一起蒸馏进 selector。

### 3.1 事件级分解显示 Oracle set 不是绝对正确

1,274 个 validation 样本可以分成四类：

| 类别 | 样本数 | 占比 | 含义 |
|---|---:|---:|---|
| Both correct | 222 | 17.4% | MMR 已足够，Oracle 与 MMR 都能选出有效证据 |
| Oracle only correct | 395 | 31.0% | 更好 evidence selection 可以直接修复 |
| MMR only correct | 156 | 12.2% | Oracle objective 与 argmax accuracy 不一致 |
| Neither correct | 501 | 39.3% | Verifier 或候选池瓶颈，selector 难以单独修复 |

最值得利用的是 Oracle only correct 的 395 个样本。它们代表了 selector 的直接可学习空间。

### 3.2 Oracle objective 存在 calibration 问题

Oracle set 搜索最大化的是 verifier 对正确标签的 log-probability：

$$S_K^* = \arg\max_{S \subseteq C, |S|=K} P_{\mathrm{verifier}}(y^* \mid c, S)$$

但最大化正确标签 log-probability 并不总是等价于让 argmax prediction 正确。特别是在当前 verifier 对 false 类过度自信、对 true 类过度不自信的情况下，Oracle set 可能提高了正确标签 logprob，但 argmax 仍然落在错误类别上。

### 3.3 True / mostly-true 类别需要谨慎使用 Oracle supervision

按 label 分桶后，Oracle 对 false / pants-fire 类别提升巨大，但对 true / mostly-true 类别甚至低于 MMR baseline：

| Class | Oracle Acc | MMR Acc | Gap | 使用建议 |
|---|---:|---:|---:|---|
| pants-fire | 88.7% | 40.0% | +48.7 pp | 高权重使用 |
| false | 89.6% | 27.0% | +62.5 pp | 高权重使用 |
| barely-true | 40.7% | 34.7% | +5.9 pp | 中等权重 |
| half-true | 53.3% | 28.7% | +24.6 pp | 高权重使用 |
| mostly-true | 21.9% | 27.1% | -5.2 pp | 谨慎使用，不做硬监督 |
| true | 1.2% | 24.9% | -23.7 pp | 暂缓硬监督，优先修 verifier |

因此，Oracle set 应经过样本过滤、类别加权或 preference 化处理，而不是直接当成全量硬标签。

---

## 4. 推荐监督信号形式

### 4.1 Set-level utility supervision

将 Oracle set 视作高 utility set，而不是唯一正确答案。训练目标不应是“完全复制这 K 个 evidence”，而应是学习什么样的 evidence set 对 verifier 有用。

可以为每个 evidence set 记录 utility：

$$r(c,S) = \log P_{\mathrm{verifier}}(y^* \mid c,S)$$

或者结合 argmax correctness、logprob、冗余惩罚与成本惩罚：

$$r(c,S) = \mathbb{1}[\hat{y}=y^*] + \alpha \log P_{\mathrm{verifier}}(y^* \mid c,S) - \beta \mathrm{Red}(S) - \gamma \mathrm{Cost}(S)$$

该信号可用于 utility regression、reward model training 或 RL reward shaping。

### 4.2 Preference supervision

优先推荐使用 preference learning。构造如下偏好对：

$$S^+ = S_{\mathrm{oracle}}$$

$$S^- \in \{S_{\mathrm{MMR}}, S_{\mathrm{reranker}}, S_{\mathrm{random}}, S_{\lambda=0.3}, S_{\lambda=0.5}, S_{\lambda=0.7}, S_{\mathrm{low\text{-}utility}}\}$$

训练目标不是直接预测 Oracle set，而是学习：

$$S^+ \succ S^-$$

这比 hard imitation 更稳，因为 evidence set 通常不是唯一的。多个不同证据集合可能都足以让 verifier 判对。

### 4.3 Reward shaping

后续做 GRPO / PPO / RL refinement 时，Oracle set 可作为 reward shaping 信号。奖励函数可以写成：

$$R(c,S) = w_1 R_{\mathrm{verdict}} + w_2 R_{\mathrm{oracle\text{-}overlap}} + w_3 R_{\mathrm{coverage}} - w_4 R_{\mathrm{redundancy}} - w_5 R_{\mathrm{cost}}$$

其中：

$$R_{\mathrm{verdict}} = \mathbb{1}[\hat{y}=y^*]$$

$$R_{\mathrm{oracle\text{-}overlap}} = \frac{|S \cap S_{\mathrm{oracle}}|}{|S_{\mathrm{oracle}}|}$$

$$R_{\mathrm{redundancy}} = \frac{1}{|S|(|S|-1)} \sum_{i \neq j} \mathrm{Sim}(d_i,d_j)$$

注意：Oracle overlap 不应成为唯一 reward，否则模型会变成 Oracle imitation，而不是学习 verifier utility。

---

## 5. 样本过滤策略

建议优先构造高置信训练子集。核心条件是：Oracle set 明确优于 baseline，且差值足够大。

$$\mathcal{D}_{\mathrm{strong}} = \{i : \mathrm{OracleCorrect}_i = 1 \land \mathrm{BaselineCorrect}_i = 0 \land \Delta r_i > \tau\}$$

其中：

$$\Delta r_i = r(c_i,S_{\mathrm{oracle}}) - r(c_i,S_{\mathrm{baseline}})$$

推荐初始过滤规则：

| 过滤条件 | 建议 |
|---|---|
| Oracle correct / MMR wrong | 强保留 |
| Oracle correct / MMR correct | 保留但降权 |
| MMR correct / Oracle wrong | 不作为正例，可用于分析 objective failure |
| Oracle wrong / MMR wrong | 不用于 selector imitation，转入 verifier / candidate pool 分析 |
| true / mostly-true | 降权或暂缓硬监督 |
| false / pants-fire / half-true | 高权重使用 |

如果要构造全量 preference 数据，也应给不同样本设置权重：

$$w_i = \sigma\left(\frac{\Delta r_i}{T}\right) \cdot \mathbb{1}[\mathrm{OracleCorrect}_i=1]$$

---

## 6. 推荐模型路线

### 6.1 Pointwise evidence utility model

训练一个模型为每个候选 evidence 打分：

$$u_\theta(c,d_i,C) \rightarrow \mathbb{R}$$

最简单的监督标签是：

$$y_i = \mathbb{1}[d_i \in S_{\mathrm{oracle}}]$$

推理时选择 top-K：

$$S_K = \mathrm{TopK}_{d_i \in C}\ u_\theta(c,d_i,C)$$

优点是实现简单，能快速验证 Oracle-set supervision 是否可被吸收。缺点是它把 set selection 拆成独立点预测，不能显式建模 coverage、redundancy 与 step-wise interaction。

### 6.2 Sequential evidence selector

将 evidence selection 建模为逐步决策：

$$d_t \sim \pi_\theta(d \mid c,C,S_{t-1})$$

每一步选择后更新状态：

$$S_t = S_{t-1} \cup \{d_t\}$$

最终得到：

$$S_K = \{d_1,d_2,\ldots,d_K\}$$

训练方式可以是 teacher forcing、behavior cloning、DPO 或后续 GRPO。相比 pointwise model，sequential selector 更贴近 MMR 与 RL-MMR 的本质。

### 6.3 Multi-weight MMR policy

这是当前最推荐的主线之一。它保留 MMR 的可解释性，但不再使用单一 scalar lambda，而是让模型输出多维权重：

$$w_t = (w_{\mathrm{rel}}, w_{\mathrm{red}}, w_{\mathrm{cov}}, w_{\mathrm{src}}, w_{\mathrm{stance}}, w_{\mathrm{cost}})$$

每一步选择：

$$d_t = \arg\max_{d \in C \setminus S_{t-1}} \left[w_{\mathrm{rel}}\mathrm{Rel}(c,d) - w_{\mathrm{red}}\mathrm{Red}(d,S_{t-1}) + w_{\mathrm{cov}}\mathrm{Cov}(d,S_{t-1}) + w_{\mathrm{src}}\mathrm{SrcDiv}(d,S_{t-1}) + w_{\mathrm{stance}}\mathrm{StanceDiv}(d,S_{t-1}) - w_{\mathrm{cost}}\mathrm{Cost}(d)\right]$$

相较于 scalar lambda MMR：

$$\mathrm{MMR}(d) = \lambda \mathrm{Rel}(c,d) - (1-\lambda)\max_{s \in S_{t-1}}\mathrm{Sim}(d,s)$$

multi-weight MMR 可以表达更多因素：coverage、source diversity、stance diversity、cost、position-dependent strategy 等。

### 6.4 Set scorer + search

训练一个 set-level scorer：

$$U_\theta(c,S_K) \approx P_{\mathrm{verifier}}(y^* \mid c,S_K)$$

推理时通过 beam search、greedy search 或 Monte Carlo search 找到高分集合：

$$S_K^\theta = \arg\max_{S \subseteq C, |S|=K} U_\theta(c,S)$$

这个方法最接近 Oracle search，但推理成本可能更高。它适合作为后续强模型，不建议作为第一步。

---

## 7. DPO / GRPO 训练设计

### 7.1 DPO for evidence set preference

对于每个 claim，构造正负 evidence set：

$$(c, S^+, S^-)$$

其中：

$$S^+ = S_{\mathrm{oracle}}, \quad S^- = S_{\mathrm{baseline}}$$

DPO 损失可以写成：

$$\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\left(\beta \left[\log \pi_\theta(S^+|c) - \log \pi_\theta(S^-|c) - \log \pi_{\mathrm{ref}}(S^+|c) + \log \pi_{\mathrm{ref}}(S^-|c)\right]\right)$$

如果 policy 是 step-wise selector，则：

$$\log \pi_\theta(S|c) = \sum_{t=1}^{K} \log \pi_\theta(d_t \mid c,C,S_{t-1})$$

如果 policy 是 lambda 或 multi-weight MMR policy，则可以将 evidence set 的生成概率近似为对应 action trajectory 的概率：

$$\log \pi_\theta(S|c) \approx \sum_{t=1}^{K} \log \pi_\theta(a_t \mid s_t)$$

其中 action 可以是 lambda、multi-weight vector、或候选 evidence id。

### 7.2 GRPO refinement

在 DPO 或监督 warm start 后，再做 GRPO refinement。对每个 claim 采样 G 条 trajectories：

$$\tau_1, \tau_2, \ldots, \tau_G \sim \pi_\theta(\cdot \mid c)$$

计算每条 trajectory 的 reward：

$$R_i = R(c,S_i)$$

使用 group-relative advantage：

$$A_i = \frac{R_i - \mathrm{mean}(R_1,\ldots,R_G)}{\mathrm{std}(R_1,\ldots,R_G) + \epsilon}$$

GRPO 适合当前任务，因为同一个 claim 可以生成多组 evidence set，并在组内比较哪一组更有用。它比直接 PPO 更适合小型 evidence selector，因为不一定需要额外训练 value model。

### 7.3 为什么不建议马上上 GRPO

GRPO / RL 会同时引入 reward noise、verifier calibration bias、exploration variance 与 credit assignment 问题。当前更重要的是先确认 Oracle-set supervision 是否可被模型吸收。

推荐顺序是：

1. Supervised pointwise / sequential selector。
2. DPO evidence set preference。
3. Multi-weight MMR distillation。
4. GRPO refinement。

---

## 8. 训练数据构造

### 8.1 必须先在 train split 运行 Oracle search

Validation / test 的 Oracle result 只能用于诊断和上界分析，不能作为训练监督，否则会发生标签泄漏。

推荐生成：

```text
outputs/oracle_evidence/<run_id>/oracle_results_train.jsonl
outputs/oracle_evidence/<run_id>/oracle_metrics_train.json
```

每条样本建议包含：

```json
{
  "claim_id": "...",
  "claim": "...",
  "label": "...",
  "candidate_pool": [...],
  "candidate_scores": {...},
  "oracle_indices": [...],
  "oracle_logprob": ...,
  "oracle_pred": "...",
  "oracle_correct": true,
  "baseline_indices": [...],
  "baseline_logprob": ...,
  "baseline_pred": "...",
  "baseline_correct": false,
  "delta_logprob": ...,
  "delta_correct": ...,
  "label_bucket": "false"
}
```

### 8.2 三类训练数据

#### 数据 A：Positive set supervision

$$(c, C, S_{\mathrm{oracle}})$$

用于 pointwise selector、sequential selector、multi-weight MMR distillation。

#### 数据 B：Preference pairs

$$(c, S^+, S^-)$$

其中：

$$S^+ = S_{\mathrm{oracle}}$$

$$S^- \in \{S_{\mathrm{MMR}}, S_{\mathrm{reranker}}, S_{\mathrm{random}}, S_{\mathrm{low\text{-}utility}}\}$$

用于 DPO / preference learning。

#### 数据 C：Utility regression

$$(c, S, r)$$

其中：

$$r = \log P_{\mathrm{verifier}}(y^* \mid c,S)$$

或者：

$$r = \mathbb{1}[\hat{y}=y^*] + \alpha \log P_{\mathrm{verifier}}(y^* \mid c,S) - \beta \mathrm{Red}(S)$$

用于 reward model 或 set scorer。

---

## 9. 下一步实验计划

### E1. Train Oracle Search

目标：在 train split 上生成 Oracle evidence set，作为后续监督信号。

输出：

```text
oracle_results_train.jsonl
oracle_metrics_train.json
```

注意：仅 train split 可用于训练；val/test oracle 只用于上界分析。

### E2. Filtered Oracle-vs-MMR DPO Selector

目标：验证 Oracle set preference 是否能训练出优于 fixed-MMR 的 selector。

正例：

$$S^+ = S_{\mathrm{oracle}}$$

负例：

$$S^- = S_{\mathrm{MMR}\text{-}0.7}$$

优先使用高置信子集：

$$\mathrm{OracleCorrect}=1 \land \mathrm{BaselineCorrect}=0 \land \Delta r > \tau$$

评价：

- 是否超过 fixed-MMR。
- 是否超过 reranker-only。
- 是否提升 Oracle only correct 子集。
- 是否恶化 true / mostly-true 类别。

### E3. Pointwise / Sequential Selector

目标：验证直接学习 Oracle indices 是否有效。

Pointwise baseline：

$$p_\theta(d_i \in S_{\mathrm{oracle}} \mid c,C)$$

Sequential selector：

$$\pi_\theta(d_t \mid c,C,S_{t-1})$$

若 pointwise selector 都无法接近 fixed-MMR，说明 Oracle set supervision 的可学习性存在问题，需要检查特征、candidate representation 或 verifier bias。

### E4. Multi-weight MMR Distillation

目标：保留 MMR 可解释性，同时突破 scalar lambda 的表达限制。

训练目标：让 multi-weight MMR 产生的 evidence set 接近 Oracle set，或在 preference 上优于 fixed-MMR set。

核心公式：

$$w_t = f_\theta(c,C,S_{t-1})$$

$$d_t = \arg\max_{d \in C \setminus S_{t-1}} \mathrm{Score}_{w_t}(c,d,S_{t-1})$$

评价：

- 相比 fixed-MMR 是否提升。
- 相比 scalar learned-lambda 是否提升。
- 相比 DPO selector 是否更稳定。
- 是否具备可解释的权重变化，例如前几步高 relevance，后几步高 diversity / coverage。

### E5. Verifier Calibration + Re-Oracle

目标：修复 true / mostly-true 类别的 false bias，避免 Oracle supervision 被 verifier 偏差污染。

可选方法：

- class-balanced loss。
- label prior correction。
- temperature scaling。
- logit adjustment。
- true / mostly-true 样本重采样。
- explanation-then-label prompt。

重新计算小规模 Oracle set，检查：

- true / mostly-true Oracle accuracy 是否提升。
- Oracle set 是否仍然低于 MMR baseline。
- false / pants-fire 类别是否受损。

### E6. Reranker + Oracle-supervised Selector

目标：与原始六组系统对齐，验证最强组合。

需要比较：

| 系统 | 描述 |
|---|---|
| reranker-only | cross-encoder 或 LLM reranker top-K |
| fixed-MMR | lambda=0.7 |
| learned-lambda MMR | 非强化学习 lambda predictor |
| RL/DPO/GRPO learned-lambda MMR | 原始 RL-lambda 路线，作为对照 |
| reranker + learned-lambda MMR | reranker 分数作为 relevance input |
| reranker + RL-MMR / Oracle-supervised selector | 主方法 |

最终目标不是证明 Oracle-supervised selector 替代 reranker，而是证明：

$$\mathrm{reranker} + \mathrm{Oracle\text{-}supervised\ selector} > \mathrm{reranker\text{-}only}$$

---

## 10. 评价指标

### 10.1 常规指标

- Accuracy。
- Macro Precision。
- Macro Recall。
- Macro F1。
- Per-class F1。
- Confusion matrix。

### 10.2 Evidence selection 指标

- Oracle overlap。
- Redundancy rate。
- Source diversity。
- Candidate rank distribution。
- Evidence set semantic diversity。
- Selected evidence average relevance。

### 10.3 Gap closed ratio

建议把 gap closed ratio 作为核心指标。它衡量方法向 Oracle 上界逼近了多少。

$$\mathrm{GapClosed}_{\mathrm{F1}} = \frac{\mathrm{F1}_{\mathrm{method}} - \mathrm{F1}_{\mathrm{baseline}}}{\mathrm{F1}_{\mathrm{oracle}} - \mathrm{F1}_{\mathrm{baseline}}}$$

以当前 Macro F1 为例，如果新方法达到 34.00：

$$\mathrm{GapClosed}_{\mathrm{F1}} = \frac{34.00 - 30.03}{43.03 - 30.03} \approx 30.5\%$$

这个指标比单独报告 F1 更能说明方法是否真正利用了 Oracle gap。

### 10.4 分桶指标

必须按以下维度分桶：

| Bucket | 目的 |
|---|---|
| label bucket | 观察 false / true 偏差 |
| Oracle only correct | 判断 selector 是否学到可修复样本 |
| MMR only correct | 检查是否破坏原本能做对的样本 |
| candidate pool size | 判断候选池复杂度影响 |
| redundancy level | 验证 diversity policy 是否有用 |
| claim length / entity count | 观察复杂 claim 收益 |
| oracle delta logprob | 评估高置信样本与低置信样本差异 |

---

## 11. 风险与处理

### 11.1 Oracle set 是 verifier-relative label

Oracle set 不是真实世界绝对最优 evidence set，而是相对于当前 verifier 的最优 set。如果 verifier、prompt、label format 或 candidate pool 改变，Oracle set 可能也会改变。

处理方式：

- 记录 verifier version、prompt version、candidate config。
- 在 verifier calibration 后重新抽样计算 Oracle set。
- 不把 Oracle set 当成永久 gold label。

### 11.2 Verifier false bias 会污染 selector

当前 true / mostly-true 类别的 Oracle 表现低于 MMR，说明 verifier 对真类严重不校准。

处理方式：

- 对 true / mostly-true 降权或暂缓硬监督。
- 先做 verifier calibration。
- 使用 preference 而不是 hard imitation。
- 在评价中单独监控 true / mostly-true 的性能变化。

### 11.3 Gold evidence 与 Oracle evidence 不一定一致

Oracle evidence set 最大化 verifier utility，不一定等价于人工 gold evidence 或事实充分性。

处理方式：

- 如果有 gold evidence，额外报告 gold evidence recall。
- 抽样人工检查 Oracle set。
- 引入 source reliability、stance balance、coverage 等约束。

### 11.4 Preference pairs 可能过于容易或过于嘈杂

如果 S+ 与 S- 差异过小，DPO 信号弱；如果差异过大，模型可能学到 shortcut。

处理方式：

- 过滤 delta reward 太小的 pair。
- 加入 hard negatives。
- 分开训练 easy / hard preference。
- 使用 pair weight。

### 11.5 直接优化 verifier reward 可能过拟合

Selector 可能学会 exploit 当前 verifier，而不是选择真正充分的证据。

处理方式：

- 使用 held-out verifier 或 prompt variant 做鲁棒性评估。
- 加入 evidence coverage 与 redundancy penalty。
- 人工抽样检查高 reward set。
- 报告 cross-verifier transfer。

---

## 12. 推荐实施顺序

### Phase 1：确认 Oracle supervision 可学习

1. 在 train split 运行 Oracle evidence set search。
2. 构造 filtered Oracle-vs-MMR preference pairs。
3. 训练 pointwise selector 与 sequential selector 两个简单模型。
4. 与 fixed-MMR、reranker-only 比较。
5. 使用 gap closed ratio 作为主指标。

### Phase 2：回到 MMR 框架但突破 scalar lambda

1. 训练 multi-weight MMR policy。
2. 比较 scalar lambda、fixed-MMR、pointwise selector、sequential selector。
3. 分析权重随 step 与 claim 类型的变化。
4. 检查是否在高冗余候选池、多子事实 claim、half-true claim 上收益更明显。

### Phase 3：加入偏好优化与强化学习

1. 用 DPO 训练 evidence set preference。
2. 在 DPO 稳定收益后再做 GRPO refinement。
3. Reward 结合 verdict correctness、Oracle overlap、coverage、redundancy 与 cost。
4. 避免直接从随机 policy 开始 RL。

### Phase 4：修复 verifier 并重新估计上界

1. 做 verifier calibration。
2. 重新计算小规模 Oracle set。
3. 检查 true / mostly-true 类别是否恢复。
4. 若 verifier 上界提升，再重新训练 selector。

---

## 13. 建议的代码模块边界

后续代码 agent 可按以下模块组织：

```text
scripts/oracle_evidence/search_optimal_evidence.py
scripts/oracle_evidence/build_train_oracle_sets.py
scripts/oracle_evidence/build_preference_pairs.py
scripts/selectors/train_pointwise_oracle_selector.py
scripts/selectors/train_sequential_oracle_selector.py
scripts/selectors/train_dpo_evidence_selector.py
scripts/selectors/train_multiweight_mmr_policy.py
scripts/selectors/eval_selector_gap_closed.py
scripts/verifier/calibrate_verifier.py
scripts/analysis/analyze_oracle_supervision_quality.py
```

建议所有 selector 输出统一格式：

```json
{
  "claim_id": "...",
  "selected_indices": [...],
  "selected_scores": [...],
  "method": "...",
  "metadata": {...}
}
```

这样可以复用现有 verifier inference 与 metrics pipeline。

---

## 14. 最小可行实验闭环

下一轮最小闭环建议只做四个实验：

| 实验 | 目的 | 成功标准 |
|---|---|---|
| E1: train Oracle search | 生成训练监督 | 得到稳定 `oracle_results_train.jsonl` |
| E2: pointwise Oracle selector | 验证 Oracle indices 可学习 | 超过 fixed-MMR 或至少在 Oracle-only subset 提升 |
| E3: filtered DPO selector | 验证 preference supervision | Macro F1 提升且 gap closed ratio > 10% |
| E4: verifier calibration probe | 检查 true / mostly-true bias | true / mostly-true Oracle Acc 不再显著低于 MMR |

若 E2 / E3 无法超过 fixed-MMR，应优先分析：

1. Oracle set 是否过拟合 verifier。
2. 训练特征是否不足以区分 Oracle evidence。
3. Candidate pool 是否过小或语义分块损失信息。
4. Preference pairs 是否信号太弱或类别偏置过强。

---

## 15. 最终结论

应该使用 Oracle evidence set 作为下一阶段核心监督信号。当前实验已经证明，evidence set selection 有 13.00 pp Macro F1 的上界 gap，而 scalar lambda 路线收益有限且多次退化。继续学习 lambda 的边际价值较低，下一步应转向 set-level selector、preference learning、multi-weight MMR policy 与 RL refinement。

但 Oracle evidence set 不应被当作无条件硬标签。它应以三种形式进入训练：

1. **Set-level utility supervision**：学习什么样的 evidence set 对 verifier 有用。
2. **Preference supervision**：学习 Oracle set 优于 MMR / reranker / random set。
3. **Reward shaping**：在 GRPO / RL 中作为辅助 reward，而不是唯一 reward。

最推荐的路线是：

$$\mathrm{fixed\text{-}MMR} \rightarrow \mathrm{Oracle\text{-}supervised\ selector} \rightarrow \mathrm{DPO\ preference\ tuning} \rightarrow \mathrm{multi\text{-}weight\ MMR} \rightarrow \mathrm{GRPO\ refinement}$$

同时，必须并行修复 verifier calibration，尤其是 true 与 mostly-true 类别。否则 Oracle supervision 可能把当前 verifier 的 false bias 蒸馏进 selector。

最终研究目标应从“学习 lambda”更新为：

$$\mathrm{Learning\ Oracle\text{-}guided\ Evidence\ Set\ Selection\ for\ Fact\ Checking}$$

更具体地说，应证明：

$$\mathrm{reranker} + \mathrm{Oracle\text{-}guided\ RL/MMR\ selector} > \mathrm{reranker\text{-}only}$$

而不是证明 RL-MMR 完全替代 reranker。
