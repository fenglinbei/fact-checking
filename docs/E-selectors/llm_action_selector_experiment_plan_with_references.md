# LLM Action Selector / Fact-checking Evidence Selector 实验路线与文献参考

生成日期：2026-05-25  
适用对象：当前 Stage2 `LLM action selector` / `fact-checking evidence selector` 实验线  
核心问题：模型能够学习 train set 的 selection/action 逻辑，但没有稳定转化到 validation set 的 evidence selection 能力。

---

## 0. 当前问题定位

当前任务可以描述为：给定一条 claim 和一个固定候选池 `C(c) = {e_1, ..., e_n}`，其中 `n <= 15`，selector 需要输出一个长度为 `k` 的有序 evidence list：

```text
L_k = (e_{i_1}, e_{i_2}, ..., e_{i_k})
```

LLM selector 当前采用 sequential action policy：

```text
state_t = claim + selected_prefix + remaining_candidates
a_t ~ pi_phi(a | state_t)
```

目前主要观察是：

1. overfit sanity 已证明模型、LoRA、action-token score path 可以拟合训练样本；
2. 中等规模训练里 train action accuracy 能上升，但 validation action accuracy / `Jaccard@5` / `NDCG@5` 没有稳定跟随提升；
3. 单纯提高 learning rate、增加 hard CE 拟合、调整 soft/set loss，主要加快 train-set 拟合，不能解决 val 泛化；
4. action-token `A..O` selector 天然存在 label prior / option bias / candidate-position shortcut 风险；
5. teacher-forced oracle prefix 与真实 rollout prefix 存在分布错配，导致错误 prefix 后恢复能力不足。

因此，后续主线不应继续围绕 `train/action_accuracy` 调参，而应围绕以下问题设计实验：

```text
selector 是否能在 val claim 上选择 verifier-useful evidence？
selector 是否能在 bad prefix / self-rollout prefix 下恢复？
selector 是否避免了 label/position/candidate-index shortcut？
selector 输出的 evidence ordering 是否真的提升 verifier？
```

---

## 1. 总体实验目标

### 1.1 目标

把当前 hard next-action imitation 转换为更接近最终任务的 evidence utility optimization。建议按以下优先级推进：

1. **固定诊断面板与强 baseline**：确认当前 selector 相对 saved-score / utility baseline 的真实位置。
2. **robust-prefix SFT**：验证 bad-prefix recovery 是否能提升 validation selection。
3. **multi-positive / pairwise utility ranking**：弱化唯一 oracle next action，改为学习 evidence utility preference。
4. **DAgger-lite on-policy aggregation**：用当前 policy rollout 生成真实错误状态，再重新标注 utility。
5. **pairwise / setwise / permutation decode**：降低 action-token label prior 和 position bias。
6. **selection-order 拆分**：先选 evidence set，再排序给 verifier。
7. **DPO / GRPO**：仅在能构造稳定 list-level preference reward 后再上。

### 1.2 不再作为主判据的指标

以下指标仍可记录，但不应作为主 checkpoint criterion：

```text
train/action_accuracy
val/action_accuracy under only teacher-forced oracle prefix
raw hard CE loss
```

原因：当前问题已经显示 train action imitation 与 val evidence utility 不一致。

### 1.3 主判据

建议主判据按优先级排序：

```text
val/selection/jaccard@5
val/selection/ndcg@5
val/selection/recall@5
val/bad_prefix/remaining_oracle_hit@1
val/bad_prefix/positive_prob
full verifier accuracy / macro-F1 / gold-label margin
selected candidate original-rank entropy
selected action-label entropy
```

如果 verifier 顺序敏感，还需要额外记录：

```text
same selected set + different ordering 的 verifier variance
selected evidence redundancy score
coverage over subclaims / atomic constraints
```

---

## 2. 核心假设

### H1: Train-val gap 主要来自 exposure bias / state distribution mismatch

Teacher-forced 训练只看 oracle prefix：

```text
prefix = gold/oracle prefix
next action = oracle next evidence
```

但推理时 prefix 来自模型自身 rollout：

```text
prefix = model-selected prefix
```

一旦前面某步错选，后续 state 就落到训练中很少见的区域。robust-prefix / DAgger-style data aggregation 应能改善这个问题。

参考：

- Scheduled Sampling for Sequence Prediction with RNNs: https://arxiv.org/abs/1506.03099
- DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning: https://proceedings.mlr.press/v15/ross11a.html

### H2: Oracle next action 不是唯一正解，也未必是 verifier-optimal 正解

当前 oracle sequence 来自 margin-search 或 saved-score artifacts。它是一条可用路径，不一定是唯一最优路径。若 hard CE 强迫模型只学唯一 next action，容易惩罚其他同样有用的 evidence。

改进方向是：

```text
single positive CE -> multi-positive utility ranking
oracle action match -> verifier utility / delta-margin preference
```

参考：

- From Relevance to Utility: Evidence Retrieval with Feedback for Fact Verification: https://aclanthology.org/2023.findings-emnlp.422/
- BERT for Evidence Retrieval and Claim Verification: https://arxiv.org/abs/1910.02655

### H3: Action-token selector 存在 label prior / option bias

当前 `A..O` action label 是 multiple-choice style prediction。即使 local choice label 和 candidate-order augmentation 已经削弱固定绑定，模型仍可能偏向某些字母 token 或 prompt position。

改进方向是：

```text
raw action-token score -> calibrated score
single candidate order -> permutation self-consistency
A..O listwise choice -> pairwise / setwise comparison
```

参考：

- Large Language Models Are Not Robust Multiple Choice Selectors: https://arxiv.org/abs/2309.03882
- Calibrate Before Use: https://arxiv.org/abs/2102.09690
- Permutation Self-Consistency Improves Listwise Ranking in LLMs: https://aclanthology.org/2024.naacl-long.129/

### H4: Selection 和 ordering 可能是两个不同目标

当前任务输出 ordered top-k。这个设计是合理的，因为 verifier 对 evidence 顺序敏感。但从训练角度看，`选哪些 evidence` 与 `怎样排序 evidence` 可能需要不同模型或不同 loss。

改进方向是：

```text
stage A: order-invariant set selector
stage B: verifier-aware orderer
```

参考：

- User-Centric Evidence Ranking for Attribution and Fact Verification: https://aclanthology.org/2026.eacl-long.340/
- Evidence Retrieval for Fact Verification using Multi-stage Reranking: https://aclanthology.org/2024.findings-emnlp.428/

---

## 3. 推荐实验路线

## Phase 0: 固定诊断面板与数据切片

### 目标

先固定一个小而稳定的 eval panel，避免每次实验因 sample / seed / candidate order 改变而无法比较。

### 建议配置

```text
TRAIN_SAMPLE_LIMIT = 2048 或 4096
VAL_SAMPLE_LIMIT   = 1024
EVAL_SAMPLE_LIMIT  = 512 或 1000
SEED               = 固定
EVAL_CANDIDATE_ORDER = candidate_pool
TRAIN_ORDER_AUGMENTATION = dynamic_random only for training
```

### 必须落盘

```text
metrics/latest_val.json
metrics/best_val.json
metrics/latest_selection_val.json
metrics/latest_selection_eval.json
evals/val/*.jsonl
selected evidence list per claim
selected label distribution
selected original candidate rank distribution
```

### 诊断切片

建议额外按以下维度切 val：

```text
oracle evidence length: 1 / 2 / 3+
candidate pool size: <=5 / 6-10 / 11-15
gold label: true / false / half-true / etc.
retrieval difficulty: oracle evidence rank high / middle / low
claim complexity: atomic / multi-constraint / temporal / numeric / entity-relation
```

### Pass / fail 判据

这一阶段不追求提升，只追求可复现比较。若同一 checkpoint 在相同 seed 下 selection eval 波动明显，需要先排查 eval order、randomness、generation/scoring path。

---

## Phase 1: 建立 baseline ladder

### 目标

明确 LLM selector 目前相对强基线的位置。不要只和 random 或旧 action selector 比。

### Baseline 列表

| Baseline | 说明 | 作用 |
|---|---|---|
| Random top-k | 随机选 remaining candidates | 确认指标下限 |
| Candidate-pool order top-k | 直接按候选池原始排序选 top-k | 检查 upstream retriever 强度 |
| Saved-score utility top-k | 按 `delta_margin` / verifier utility 排序 | 作为近似上界或强 oracle-side baseline |
| Current action-token LLM | 现有 robust-prefix 前版本 | 当前主线基线 |
| Current LLM + calibration | `raw_score - alpha * label_bias` | 检查 label prior 影响 |
| Current LLM + permutation decoding | 多次 shuffle candidate order 后聚合 | 检查 position sensitivity |

### 推荐记录

```text
selection/jaccard@5
selection/ndcg@5
selection/recall@5
top1_match
full verifier metric
label entropy
candidate original-rank entropy
```

### 关键判断

1. 如果 `candidate-pool order top-k` 已经很强，说明候选池排序本身含有强 prior，LLM selector 容易学位置 shortcut。
2. 如果 `saved-score utility top-k` 明显强于 LLM selector，说明训练目标没有充分吸收 verifier utility。
3. 如果 `LLM + permutation` 明显强于 raw LLM，说明 position / label bias 是主要问题之一。

参考：

- RankGPT / Is ChatGPT Good at Search?: https://aclanthology.org/2023.emnlp-main.923/
- Permutation Self-Consistency: https://aclanthology.org/2024.naacl-long.129/
- Large Language Models Are Not Robust Multiple Choice Selectors: https://arxiv.org/abs/2309.03882

---

## Phase 2: Robust-prefix preference-style SFT

### 目标

验证当前最新主线：bad-prefix samples + multi-positive pairwise loss 是否能改善真实 rollout 下的恢复能力。

### 推荐启动配置

```bash
OUTPUT_DIR=outputs/selectors/llm_action_selector/qwen25_3b_robust_prefix_v1 \
LR=1e-4 \
SOFT_LOSS_WEIGHT=0.01 \
SET_LOSS_WEIGHT=0.02 \
PAIRWISE_LOSS_WEIGHT=0.05 \
HARD_LOSS_WEIGHT=1.0 \
BAD_PREFIX_HARD_LOSS_WEIGHT=0 \
TRAIN_ORDER_AUGMENTATION=dynamic_random \
BUILD_BAD_PREFIX_DATA=true \
BAD_PREFIX_SOURCES=hybrid,random_corrupt \
BAD_PREFIX_MAX_REPLACEMENTS=2 \
BAD_PREFIX_SAMPLE_RATIO=1.0 \
TRAIN_SAMPLE_LIMIT=2048 \
VAL_SAMPLE_LIMIT=1024 \
EVAL_SAMPLE_LIMIT=512 \
EPOCHS=10 \
EVAL_EVERY=50 \
bash scripts/selectors/run_llm_action_selector_vig_soft.sh
```

### Loss 设计

Oracle-prefix 样本：

```text
loss = hard_action_ce + soft_loss + set_loss + pairwise_loss
```

Bad-prefix 样本：

```text
hard CE weight = 0
positive set = remaining oracle candidates 或 positive-utility candidates
loss = pairwise positive-vs-negative + optional set/listwise loss
```

Pairwise loss：

```text
loss_pairwise = mean softplus(-(score_pos - score_neg))
```

### 必须看哪些指标

```text
val/bad_prefix/remaining_oracle_hit@1
val/bad_prefix/positive_prob
val/selection/jaccard@5
val/selection/ndcg@5
selected label distribution
selected original candidate rank distribution
```

### 关键判断

| 现象 | 解释 | 下一步 |
|---|---|---|
| bad-prefix hit@1 上升，Jaccard/NDCG 也上升 | robust-prefix 有效 | 扩大数据、调 pairwise weight |
| bad-prefix hit@1 上升，Jaccard/NDCG 不升 | remaining oracle 与 verifier utility 不一致 | 转 Phase 3 utility ranking |
| train 继续上升，val 不动 | 仍在背 train action path | 转 DAgger-lite / utility preference |
| label 分布坍缩 | option bias 或 soft/set loss 噪声 | 加 calibration / permutation / pairwise decode |

参考：

- Scheduled Sampling: https://arxiv.org/abs/1506.03099
- DAgger: https://proceedings.mlr.press/v15/ross11a.html
- LOLS: https://proceedings.mlr.press/v37/changb15.html

### 2026-05-25 smoke 结果记录：`qwen25_3b_robust_prefix_v1_smoke`

运行目录：

```text
outputs/selectors/llm_action_selector/qwen25_3b_robust_prefix_v1_smoke
```

配置与本阶段 smoke 设计基本一致：`TRAIN_SAMPLE_LIMIT=2048`，`VAL_SAMPLE_LIMIT=1024`，`EVAL_SAMPLE_LIMIT=512`，`BAD_PREFIX_SOURCES=hybrid,random_corrupt`，`BAD_PREFIX_HARD_LOSS_WEIGHT=0`，`PAIRWISE_LOSS_WEIGHT=0.05`，主 checkpoint 按 `jaccard@5` 保存。

最终 post-train selection eval 使用 `checkpoints/best_selection`，在 512 条 val claims 上得到：

| 指标 | robust-prefix LLM | candidate-pool / hybrid control | 差值 |
|---|---:|---:|---:|
| `recall@5` | 0.3910 | 0.3465 | +0.0445 |
| `jaccard@5` | 0.2659 | 0.2321 | +0.0338 |
| `oracle_rank_ndcg@5` | 0.2803 | 0.2874 | -0.0071 |
| `top1_match` | 0.0664 | 0.0938 | -0.0273 |
| `pairwise_order_acc@5` | 0.5149 | 0.5495 | -0.0346 |
| `ordered_exact_match@5` | 0.0039 | 0.0039 | +0.0000 |

与 saved-score utility reference 的差距仍然很大：`single_margin_step0_static` 的 `jaccard@5=0.3761`，本次 `delta_vs_single_margin_step0_static=-0.1102`，eval 侧自动判定为 `no_go`。

训练过程中出现了 train-side 拟合，但没有形成稳定的 val recovery：最后 50 个 train log 的平均 `remaining_oracle_hit@1` 为 0.5944，而 best-selection checkpoint 的 val `remaining_oracle_hit@1=0.3242`、`bad_prefix_remaining_oracle_hit@1=0.3945`。初始 eval 的 bad-prefix hit 已有 0.4121，训练后没有实质性提高。

Action / position 分布显示 label prior 没有消失，只是从 base model 的 `A` / top-candidate 偏置迁移到新的中后段偏置：post-train 512-claim rollout 中 `G` 占 24.0%，`J` 占 20.6%，`I` 占 10.4%，而 `O` 仅占 0.2%；选中的 candidate idx 也集中在 7-13，明显高于 oracle idx 的近似均匀分布。

阶段结论：**No-Go / 不建议直接扩大同一 robust-prefix action-token SFT 配方**。本次结果只说明模型学到了一点 set-overlap 信号，但没有解决 bad-prefix recovery、top1/order quality 或 action-label bias。下一步应转向 Phase 3 的 utility ranking / multi-positive target，或先做 decode-time permutation/calibration 诊断确认 raw action-token bias 的影响；不建议继续单纯增加 epoch、数据量或 hard CE 权重。

---

## Phase 3: Multi-positive utility ranking

### 目标

把训练目标从“模仿唯一 oracle next action”改成“学习当前 state 下每个 candidate 对 verifier 的 utility”。

### 数据构造

对每个 state `(claim, prefix, remaining_candidates)`，为每个 candidate 构造 utility：

```text
u(e_i | claim, prefix) = verifier_margin(prefix + e_i) - verifier_margin(prefix)
```

或使用已有 saved-score VIG rows 中的 `delta_margin`。

候选标签可分为：

```text
positive: delta_margin > 0 或 top percentile
neutral: delta_margin 近似 0
negative: delta_margin < 0 或 high retrieval score but low utility
```

不要只把 oracle next action 视为 positive。建议 positive set 包括：

```text
remaining oracle evidence
positive delta-margin evidence
能提升 gold-label probability 的 evidence
能补充未覆盖 subclaim 的 evidence
```

### Loss 选择

优先顺序：

1. Pairwise logistic loss：`positive > negative`
2. Listwise CE / ListNet：soft label = `softmax(delta_margin / tau)`
3. LambdaRank-style NDCG surrogate：直接优化 top-k ranking
4. Hard CE：仅作为 auxiliary，不超过主 loss

### 推荐配置

```text
HARD_LOSS_WEIGHT = 0.2 ~ 0.5
PAIRWISE_LOSS_WEIGHT = 0.1 ~ 0.5
SOFT_LOSS_WEIGHT = 0.02 ~ 0.1
SOFT_TAU = 0.2 ~ 0.5 initially
BAD_PREFIX_HARD_LOSS_WEIGHT = 0
```

### Hard negative 设计

重点构造下面几类 negatives：

```text
高 lexical overlap 但 verifier utility 低
高 dense/BM25 score 但 delta_margin <= 0
与 prefix 冗余的 evidence
同实体但关系/时间/数值错配的 evidence
支持相反 label 的 evidence
来自同页面但不支撑 claim 的 evidence
```

### 关键判断

如果 utility ranking 提升 `NDCG@5`，但 full verifier 不提升，说明 selection set 已改善，但 ordering 或 redundancy 仍有问题，应进入 Phase 6。

参考：

- From Relevance to Utility: https://aclanthology.org/2023.findings-emnlp.422/
- BERT for Evidence Retrieval and Claim Verification: https://arxiv.org/abs/1910.02655
- Evidence Retrieval for Fact Verification using Multi-stage Reranking: https://aclanthology.org/2024.findings-emnlp.428/

---

## Phase 4: DAgger-lite on-policy data aggregation

### 目标

让训练数据覆盖模型自己真实会访问的 states，而不是只覆盖 oracle prefix 或人工 corrupt prefix。

### 过程

```text
Round 0: train current robust-prefix SFT model
Round 1: 用当前 model 在 train claims 上 rollout top-k
Round 2: 收集 rollout prefix states
Round 3: 对这些 states 重新计算 / 读取 candidate utility
Round 4: 将 on-policy states 聚合进训练集
Round 5: 继续训练或重新训练
Round 6: 重复 2~3 轮
```

### 标注策略

对于 rollout prefix `p_t`，不再问“oracle 下一步是什么”，而是问：

```text
在当前错误/部分正确 prefix 下，哪个 remaining candidate 最能提升 verifier utility？
```

标注可以来自：

```text
saved-score delta_margin
verifier gold-label margin
final correctness improvement
remaining oracle overlap
subclaim coverage improvement
```

### 关键 ablation

| 配置 | 作用 |
|---|---|
| offline bad-prefix only | 当前 robust-prefix 近似 |
| self-rollout prefix only | 真实 policy distribution |
| hybrid + self-rollout | 推荐主配置 |
| oracle-prefix only | teacher-forced baseline |

### 关键判断

如果 on-policy aggregation 后 bad-prefix / rollout-prefix recovery 明显提升，说明主要瓶颈确实是 exposure bias。如果仍不提升，问题更可能是 utility label 噪声、候选池证据不足或 verifier 本身顺序敏感。

参考：

- DAgger: https://proceedings.mlr.press/v15/ross11a.html
- Scheduled Sampling: https://arxiv.org/abs/1506.03099
- LOLS: https://proceedings.mlr.press/v37/changb15.html

---

## Phase 5: Decode-time bias control：pairwise / setwise / permutation

### 目标

降低 action-token `A..O` 分类的 label prior 和 prompt-position sensitivity。

### 三个 decode baseline

#### 5.1 Symmetric pairwise comparison

对候选 `e_i, e_j` 做双向比较：

```text
prompt A: A=e_i, B=e_j
prompt B: A=e_j, B=e_i
score(i > j) = average / debiased pairwise preference
```

再用 Bradley-Terry、Copeland score、tournament 或 sorting 聚合成 ranking。

#### 5.2 Setwise tournament

每次让 LLM 在一个小集合中选 winner：

```text
group size = 4 or 5
repeat tournament until top-k selected
```

相比 full listwise，它降低了 15-way selection 难度；相比 pairwise，它减少了比较次数。

#### 5.3 Permutation self-consistency

对同一 claim 重复 `m` 次：

```text
shuffle candidate order
run selector
map local labels back to original candidate ids
aggregate ranking / vote / score
```

推荐先用：

```text
m = 8 or 16
aggregation = Borda count / reciprocal rank fusion / mean calibrated score
```

### 必须记录

```text
score variance across permutations
selected set agreement across permutations
label distribution before/after calibration
position distribution before/after permutation aggregation
```

### 关键判断

如果 permutation aggregation 明显提升 val selection，说明模型本身有 evidence signal，但 raw decode 被 label/position bias 干扰。如果 pairwise/setwise 显著强于 action-token，则应考虑把训练目标也改成 pairwise/setwise。

参考：

- Pairwise Ranking Prompting: https://arxiv.org/abs/2306.17563
- Setwise Prompting: https://arxiv.org/abs/2310.09497
- RankGPT: https://aclanthology.org/2023.emnlp-main.923/
- Permutation Self-Consistency: https://aclanthology.org/2024.naacl-long.129/

---

## Phase 6: Selection 和 ordering 拆分

### 目标

避免 ordered action sequence 把两个问题混在一起：

```text
Q1: 哪些 evidence 应该被选？
Q2: 这些 evidence 应该以什么顺序给 verifier？
```

### 推荐结构

#### Stage A: Order-invariant set selector

输入：

```text
claim + candidate pool
```

输出：

```text
unordered selected set S_k
```

训练目标：

```text
set recall
Jaccard@k
utility@k
coverage over atomic constraints
```

#### Stage B: Verifier-aware orderer

输入：

```text
claim + selected evidence set S_k
```

输出：

```text
ordered list L_k
```

训练目标：

```text
verifier margin
gold-label probability
final correctness
sufficiency early in ranked list
```

### 实验设计

```text
A1: current sequential selector directly outputs ordered L_k
A2: set selector + original candidate order
A3: set selector + verifier-margin greedy ordering
A4: set selector + LLM orderer
A5: set selector + permutation order ensemble
```

### 关键判断

1. 如果 `A2` 比当前 sequential selector 强，说明 early action error 污染后续选择。
2. 如果 `A3/A4` 明显提升 full verifier，但 selection metrics 差异不大，说明 bottleneck 在 ordering。
3. 如果 set selector selection metrics 高但 verifier 不升，说明 evidence 冗余或 contradiction management 有问题。

参考：

- User-Centric Evidence Ranking: https://aclanthology.org/2026.eacl-long.340/
- FEVEROUS: https://aclanthology.org/2021.fever-1.1/
- M-ReRank: https://aclanthology.org/2024.findings-emnlp.428/

---

## Phase 7: Preference optimization：DPO / GRPO

### 前提

不要直接把当前 hard oracle sequence 用作 DPO preference。DPO/GRPO 应建立在稳定的 list-level reward 上。

### List-level reward 建议

对同一 claim 采样多个 evidence lists：

```text
L_1, L_2, ..., L_m
```

为每个 list 计算 reward：

```text
R(L) = w1 * verifier_gold_margin
     + w2 * final_correctness
     + w3 * evidence_utility_ndcg
     + w4 * oracle_overlap
     + w5 * diversity_or_coverage
     - w6 * redundancy
```

然后构造 preference pair：

```text
L_good > L_bad if R(L_good) - R(L_bad) >= margin_threshold
```

### DPO 路线

```text
Step 1: 用 robust-prefix / utility-ranking SFT 初始化
Step 2: 采样 candidate lists
Step 3: 用 verifier reward 产生 pairwise preferences
Step 4: DPO fine-tune
Step 5: selection-only eval + full verifier eval
```

### GRPO 路线

```text
Step 1: 对同一 claim sample group of evidence lists
Step 2: group-relative reward normalization
Step 3: policy update with KL constraint
Step 4: monitor reward hacking / evidence collapse
```

### 关键风险

```text
reward noise 会被 RL 放大
verifier bias 会被 selector 学走
如果候选池缺少有效 evidence，RL 只会优化 shortcut
如果 reward 没有 coverage/diversity，容易选冗余 evidence
```

### 建议顺序

```text
robust-prefix SFT -> utility pairwise ranking -> DAgger-lite -> DPO -> small-step GRPO
```

参考：

- Direct Preference Optimization: https://arxiv.org/abs/2305.18290
- DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300
- FFRR: https://aclanthology.org/2024.lrec-main.1209/

---

## 4. 推荐实验矩阵

### 4.1 Minimal matrix

| ID | Model / Training | Decode | 主要目的 |
|---|---|---|---|
| B0 | random | random | 下限 |
| B1 | candidate-pool order | top-k | 检查候选池 prior |
| B2 | saved-score utility | top-k | verifier-utility 上界/强基线 |
| M0 | current action-token SFT | raw A..O | 当前主线基线 |
| M1 | current action-token SFT | calibrated A..O | label prior 诊断 |
| M2 | robust-prefix SFT | raw A..O | bad-prefix SFT 主实验 |
| M3 | robust-prefix SFT | permutation self-consistency | position sensitivity 诊断 |
| M4 | utility pairwise ranking | raw / calibrated | 验证 utility preference |
| M5 | DAgger-lite utility ranking | raw / calibrated | 验证 on-policy state aggregation |
| M6 | set selector + orderer | verifier-aware order | selection-order 拆分 |

### 4.2 Ablation matrix

| 变量 | 候选值 | 关注指标 |
|---|---|---|
| `PAIRWISE_LOSS_WEIGHT` | 0 / 0.05 / 0.1 / 0.2 / 0.5 | bad-prefix hit@1, NDCG@5 |
| `BAD_PREFIX_SAMPLE_RATIO` | 0 / 0.5 / 1.0 / 2.0 | rollout-prefix recovery |
| `BAD_PREFIX_MAX_REPLACEMENTS` | 1 / 2 / 3 | robustness vs noise |
| `TRAIN_ORDER_AUGMENTATION` | static / random / dynamic_random | position collapse |
| `HARD_LOSS_WEIGHT` | 1.0 / 0.5 / 0.2 / 0 | oracle overfitting |
| calibration `alpha` | 0 / 0.25 / 0.5 / 1.0 | label entropy, val selection |
| decode | action-token / pairwise / setwise / permutation | bias-robust ranking |
| target | oracle action / remaining oracle / delta-margin utility / list-level reward | objective alignment |

---

## 5. 判据与排错决策树

### 5.1 主要验收标准

一个实验应至少满足下面之一，才值得扩大：

```text
val/selection/jaccard@5 提升 >= 2~3 percentage points
val/selection/ndcg@5 提升 >= 2~3 percentage points
bad_prefix_remaining_oracle_hit@1 明显高于 random / current selector
full verifier macro-F1 或 gold-label margin 有稳定提升
label entropy 与 candidate-rank entropy 没有明显 collapse
```

### 5.2 决策树

```text
train/action_acc ↑, val/selection 不动
  -> 不再调 LR；检查 action imitation shortcut
  -> 转 utility ranking 或 DAgger-lite

bad-prefix hit@1 ↑, Jaccard/NDCG 不动
  -> remaining oracle 不等于 verifier utility
  -> 用 delta-margin / verifier feedback 重新定义 positive

Jaccard/NDCG ↑, full verifier 不动
  -> ordering / redundancy / verifier sensitivity 是瓶颈
  -> 拆 selection 和 ordering

permutation decode 明显优于 raw decode
  -> label/position bias 是主要问题
  -> 加 calibration、pairwise/setwise decode，训练也改 pairwise

saved-score utility baseline 远强于 LLM selector
  -> LLM 没学到 utility signal
  -> 强化 utility labels、hard negatives、listwise NDCG loss

candidate-pool order baseline 很强
  -> upstream retrieval rank 是强 shortcut
  -> 必须用 dynamic candidate order 和 rank entropy 诊断

所有 selector 都弱，包括 saved-score utility baseline
  -> 候选池本身可能缺 evidence，或 verifier reward 噪声大
  -> 回到 candidate generation / retriever / verifier 校准
```

---

## 6. 与实验路线直接对应的论文

| 实验方向 | 参考论文 | 链接 | 可借鉴点 |
|---|---|---|---|
| Fact-checking evidence selection 基础 | FEVER | https://aclanthology.org/N18-1074/ | evidence-based verification benchmark；label + evidence joint evaluation |
| Evidence retrieval baseline | BERT for Evidence Retrieval and Claim Verification | https://arxiv.org/abs/1910.02655 | pointwise/pairwise retrieval loss；hard negative mining |
| Multi-hop evidence | HoVer | https://aclanthology.org/2020.findings-emnlp.309/ | 多 evidence 组合、跨文档 hop、prefix-conditioned selection 动机 |
| Structured + unstructured evidence | FEVEROUS | https://aclanthology.org/2021.fever-1.1/ | evidence retrieval 与 label accuracy 联合评价；复杂证据形态 |
| Real-world claim verification | AVeriTeC | https://arxiv.org/abs/2305.13117 | QA-style evidence、subclaim coverage、真实世界证据不足问题 |
| Utility-aware retrieval | From Relevance to Utility / FER | https://aclanthology.org/2023.findings-emnlp.422/ | 从 relevance 改成 verifier utility；用 verifier feedback 训练 retriever |
| Reinforcement retrieval | FFRR | https://aclanthology.org/2024.lrec-main.1209/ | LLM/verifier fine-grained feedback 作为 retrieval reward |
| Multi-stage reranking | M-ReRank | https://aclanthology.org/2024.findings-emnlp.428/ | 多阶段 reranking 比单阶段 selector 更稳 |
| Iterative retrieval + verification | FIRE | https://aclanthology.org/2025.findings-naacl.158/ | 检索和验证循环交互；根据 confidence 决定是否继续检索 |
| Evidence ranking | User-Centric Evidence Ranking | https://aclanthology.org/2026.eacl-long.340/ | 把 evidence selection 改成 ranking；强调 early sufficiency 和减少阅读成本 |
| LLM listwise reranking | RankGPT | https://aclanthology.org/2023.emnlp-main.923/ | LLM 可做 reranker，但 ranking objective 与 LM objective 有 mismatch |
| Pairwise LLM ranking | PRP | https://arxiv.org/abs/2306.17563 | pairwise comparison 降低 LLM ranking 难度 |
| Setwise LLM ranking | Setwise prompting | https://arxiv.org/abs/2310.09497 | 小集合 tournament，平衡效果和计算成本 |
| Position-bias decode | Permutation Self-Consistency | https://aclanthology.org/2024.naacl-long.129/ | 多次 candidate order shuffle 后聚合，边缘化位置偏置 |
| Exposure bias | Scheduled Sampling | https://arxiv.org/abs/1506.03099 | teacher-forced training 与 self-rollout inference 的 discrepancy |
| On-policy imitation | DAgger | https://proceedings.mlr.press/v15/ross11a.html | rollout 当前 policy，收集 induced states，再查询 expert/utility label |
| Suboptimal teacher | LOLS | https://proceedings.mlr.press/v37/changb15.html | teacher/reference policy 不一定最优；允许改进 teacher |
| Label calibration | Calibrate Before Use | https://arxiv.org/abs/2102.09690 | content-free input 估计 label prior，减少 prompt/label bias |
| Option-ID bias | LLMs Are Not Robust MC Selectors | https://arxiv.org/abs/2309.03882 | option token prior / selection bias；适合 action-token selector 诊断 |
| Preference optimization | DPO | https://arxiv.org/abs/2305.18290 | 用 preference pairs 直接优化 LM policy |
| Group-relative RL | DeepSeekMath / GRPO | https://arxiv.org/abs/2402.03300 | group-relative reward；适合同 claim 多 list reward 优化 |
| AFC survey | A Survey on Automated Fact-Checking | https://arxiv.org/abs/2108.11896 | 自动 fact-checking 任务定义、数据集、pipeline 总览 |
| LLM fact-checking survey | Generative LLMs in Automated Fact-Checking | https://arxiv.org/abs/2407.02351 | LLM 在 fact-checking 中的 prompting / fine-tuning / limitation 综述 |

---

## 7. 论文分组笔记

### 7.1 Fact-checking evidence retrieval / selection

#### FEVER: a Large-scale Dataset for Fact Extraction and VERification

链接：https://aclanthology.org/N18-1074/

与当前实验的关系：

- 奠定 claim + evidence + verdict 的标准设置；
- evidence retrieval 与 claim verification 不应分开看；
- 可以参考 FEVER score 的思想：最终评价应同时考虑 evidence 和 label。

#### BERT for Evidence Retrieval and Claim Verification

链接：https://arxiv.org/abs/1910.02655

与当前实验的关系：

- 直接 relevant：evidence retriever + verifier pipeline；
- 使用 pointwise / pairwise loss 和 hard negative mining；
- 说明 hard negatives 对 evidence selector 很关键。

建议借鉴：

```text
当前 LLM selector 也应引入 pairwise positive-vs-hard-negative loss，
而不是只学 oracle next action CE。
```

#### HoVer

链接：https://aclanthology.org/2020.findings-emnlp.309/

与当前实验的关系：

- 多跳 claim verification 需要多 evidence 互补；
- prefix-conditioned selector 的动机很强；
- 单条 evidence relevance 不足以决定下一步选择。

建议借鉴：

```text
在 bad-prefix 或 partial-prefix 下，下一条 evidence 的价值应看互补性，
而不是独立 relevance。
```

#### FEVEROUS

链接：https://aclanthology.org/2021.fever-1.1/

与当前实验的关系：

- evidence 可能来自 text/table/list，selection 与 verification 强耦合；
- evaluation 结合 evidence retrieval 和 label accuracy；
- NotEnoughInfo 也要求 partial evidence，说明 evidence selection 不是简单 gold sentence match。

#### AVeriTeC

链接：https://arxiv.org/abs/2305.13117

与当前实验的关系：

- 真实世界 claim 往往需要 question-answer style evidence；
- evidence value 取决于覆盖 claim 的哪个子问题；
- 支持引入 subclaim / atomic constraint coverage signal。

---

### 7.2 Verifier-utility-aware retrieval

#### From Relevance to Utility: Evidence Retrieval with Feedback for Fact Verification

链接：https://aclanthology.org/2023.findings-emnlp.422/

与当前实验的关系：

- 最直接相关的论文之一；
- 明确提出 fact verification 的 evidence retrieval 不应只优化 relevance，而应优化 verifier utility；
- 其 FER 用 verifier feedback 训练 retriever。

建议借鉴：

```text
把 delta_margin / verifier gold-label probability / final correctness
作为 selector 的 utility label 或 reward。
```

#### FFRR: Reinforcement Retrieval Leveraging Fine-grained Feedback for Fact Checking News Claims with Black-Box LLM

链接：https://aclanthology.org/2024.lrec-main.1209/

与当前实验的关系：

- 用 LLM feedback 作为 reward 来优化 retrieval policy；
- 适合作为后续 GRPO / policy optimization 的参考；
- 说明非 retrieval-oriented label 可以转成细粒度 retrieval reward。

#### FIRE: Fact-checking with Iterative Retrieval and Verification

链接：https://aclanthology.org/2025.findings-naacl.158/

与当前实验的关系：

- 把 retrieval 和 verification 做成 iterative loop；
- 强调当前 evidence 是否足够可以由 verifier confidence 决定；
- 对当前 fixed top-k selector 的启发是：可以考虑 stop/continue 或 confidence-aware top-k，而不是固定选满 k。

#### User-Centric Evidence Ranking for Attribution and Fact Verification

链接：https://aclanthology.org/2026.eacl-long.340/

与当前实验的关系：

- 明确把 evidence selection 重构为 evidence ranking；
- 强调 sufficient evidence 应尽早出现在 ranked list；
- 与当前 ordered top-k / verifier order sensitivity 高度相关。

建议借鉴：

```text
评估不只看 selected set overlap，也看 useful evidence 是否越早出现。
可以设计 early-sufficiency NDCG 或 prefix-verifier margin curve。
```

---

### 7.3 LLM reranking / selector

#### RankGPT / Is ChatGPT Good at Search?

链接：https://aclanthology.org/2023.emnlp-main.923/

与当前实验的关系：

- LLM 可以作为 passage reranker；
- 但 LLM pretraining objective 与 ranking objective 存在 mismatch；
- 支持把 action-token SFT 改成 ranking-specific training / distillation。

#### Pairwise Ranking Prompting

链接：https://arxiv.org/abs/2306.17563

与当前实验的关系：

- pairwise ranking 降低 LLM 一次性 listwise ranking 的负担；
- 当前候选池最多 15 条，可以接受 pairwise/tournament decode；
- 可以作为 action-token selector 的强 decode baseline。

#### Setwise prompting

链接：https://arxiv.org/abs/2310.09497

与当前实验的关系：

- setwise 是 pointwise/pairwise/listwise 的折中；
- 适合候选池不大但 full pairwise 仍较贵的场景；
- 可实现 group size 4/5 的 tournament selector。

#### Permutation Self-Consistency

链接：https://aclanthology.org/2024.naacl-long.129/

与当前实验的关系：

- 直接处理 LLM listwise ranking 的 positional bias；
- 当前 candidate-order augmentation 是 training-side，permutation self-consistency 是 inference-side；
- 两者结合可以判断模型是否真正学习 evidence semantic signal。

---

### 7.4 Learning-to-search / imitation learning

#### Scheduled Sampling

链接：https://arxiv.org/abs/1506.03099

与当前实验的关系：

- 解释 teacher-forcing 与 self-rollout 之间的 exposure bias；
- 当前 selector 的错误 prefix 问题基本是 evidence sequence version 的 scheduled sampling 问题。

#### DAgger

链接：https://proceedings.mlr.press/v15/ross11a.html

与当前实验的关系：

- 当前 bad-prefix samples 是 DAgger 的 offline approximation；
- 更强版本应由当前 policy rollout 产生真实 states，再重新标注 utility。

#### LOLS

链接：https://proceedings.mlr.press/v37/changb15.html

与当前实验的关系：

- 当 teacher/oracle 本身 suboptimal 时，不应只做 behavioral cloning；
- 当前 oracle sequence 可能不是唯一最优 evidence list，因此 LOLS 的思想适合支持 utility-based deviations。

---

### 7.5 Bias calibration / option bias

#### Calibrate Before Use

链接：https://arxiv.org/abs/2102.09690

与当前实验的关系：

- 用 content-free input 估计 label prior；
- 可用于 action label `A..O` 的 bias calibration；
- 建议与 blank claim、blank evidence、shuffled candidates 三种 estimator 比较。

#### Large Language Models Are Not Robust Multiple Choice Selectors

链接：https://arxiv.org/abs/2309.03882

与当前实验的关系：

- 当前 action-token selector 本质是 MCQ selector；
- 论文指出 LLM 可能偏好特定 option ID，例如 `A`；
- PriDe 思路可用于 inference-time debiasing。

推荐诊断：

```text
blank claim + real candidates
real claim + blank candidates
random claim + real candidates
candidate content permutation
label ID permutation
```

---

### 7.6 DPO / GRPO / preference optimization

#### Direct Preference Optimization

链接：https://arxiv.org/abs/2305.18290

与当前实验的关系：

- 适合已有 list-level preference pairs 后使用；
- 可避免显式 reward model + PPO 的复杂训练；
- 但 preference 数据必须基于 verifier utility，而不是 hard oracle sequence。

#### DeepSeekMath / GRPO

链接：https://arxiv.org/abs/2402.03300

与当前实验的关系：

- GRPO 用 group-relative reward，适合同 claim 多 evidence-list 采样；
- 可以把同一 claim 下多个 selected lists 作为 group，用 verifier reward 做 relative normalization；
- 风险是 reward noise 和 reward hacking。

---

## 8. 建议的近期执行顺序

### Week-level plan

#### Step 1: 固定 eval panel 和 baseline ladder

输出：

```text
baseline_table.md
baseline_metrics.json
candidate_rank_distribution.png
label_distribution.png
```

必须包含：

```text
candidate-pool order
saved-score utility
current action-token LLM
current + calibration
current + permutation self-consistency
```

#### Step 2: 跑 robust-prefix v1 主实验

输出：

```text
robust_prefix_v1_metrics.json
bad_prefix_recovery_table.md
selection_eval.jsonl
```

判断：

```text
bad-prefix recovery 是否提升？
Jaccard/NDCG 是否同步提升？
是否出现 label / position collapse？
```

#### Step 3: 如果 robust-prefix 不够，转 utility pairwise ranking

重点改动：

```text
positive = delta_margin positive / remaining oracle / coverage-positive
negative = high score but low utility hard negatives
loss = pairwise + listwise utility
hard CE 降权
```

#### Step 4: 做 DAgger-lite

重点改动：

```text
每轮用当前 selector rollout
收集 self-induced prefix states
用 verifier utility 重新标注 candidate preference
聚合进训练集
```

#### Step 5: selection-order split

当 selection metrics 提升但 full verifier 不提升时启动。

---

## 9. 最终建议

当前最应该优先验证的是：

```text
robust-prefix + utility ranking + DAgger-lite
```

不建议优先做的是：

```text
继续提高 LR 追 train action accuracy
继续加 hard CE 拟合 oracle next action
直接上 GRPO 而没有稳定 list-level reward
只看 action accuracy 选 checkpoint
```

最有诊断价值的对照是：

```text
saved-score utility top-k vs current LLM selector
raw action-token vs calibrated action-token
single candidate order vs permutation self-consistency
oracle-prefix validation vs self-rollout-prefix validation
selection-only metrics vs full verifier metrics
```

如果这些对照跑清楚，后续方向会非常明确：

```text
若 saved-score 强：重点学 utility。
若 permutation 强：重点消 bias。
若 DAgger 强：重点解决 exposure bias。
若 selection 强但 verifier 弱：重点做 ordering / redundancy / coverage。
若所有都弱：回到 candidate pool / verifier reward 质量。
```
