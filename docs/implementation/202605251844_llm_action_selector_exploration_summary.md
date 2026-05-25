# LLM Selector 方向阶段性探索总结

## 1. 基础问题描述

最基础的 fact-checking 任务可以写成：给定一条待核查声明 `claim`，系统需要预测它的真实性标签。

$$f_\theta(c) \rightarrow \hat{y},\quad y\in\mathcal{Y}$$

其中 `c` 表示 claim，`y` 是 gold factuality label，例如 `true`、`false`、`half-true` 等。对于需要外部信息支撑的 fact-checking，模型通常不应只依赖 claim 本身，而应基于检索到的 evidence 做判断：

$$f_\theta(c,E) \rightarrow \hat{y},\quad E=\{e_1,e_2,\ldots,e_m\}$$

因此，pipeline 中一个核心子问题是 evidence selection：如何从大量候选证据中选出最有用的一小组 evidence，使后续 verifier 更容易做出正确判断。

在当前 Stage2 selector 实验里，问题进一步被约束为“从固定候选池中选择 top-k”。对每条 claim，前置检索/候选构建模块已经给出一个固定候选池：

$$C(c)=\{e_1,e_2,\ldots,e_n\},\quad n\le 15$$

selector 的目标是在不改变候选池的前提下，选出一个长度为 `k` 的有序 evidence list：

$$L_k=(e_{i_1},e_{i_2},\ldots,e_{i_k}),\quad i_t\in\{1,\ldots,n\},\quad i_a\ne i_b$$

这里的“有序”很重要：后续 verifier 接收的是按顺序拼接的 evidence，上游实验已经显示同一个 evidence set 的不同排列可能带来不同 verifier 表现。因此当前 selector 不是只预测无序集合 `S_k`，而是预测有序列表 `L_k`。

如果把 selector 写成直接打分排序，它可以表示为：

$$s_\phi(c,e_i,C)\in\mathbb{R},\quad L_k=\operatorname{TopK}_{e_i\in C(c)}s_\phi(c,e_i,C)$$

而 LLM action selector 采用的是另一种序列决策形式：每一步只选择一条 next evidence。第 `t` 步状态由 claim、已选 prefix 和剩余候选组成：

$$p_t=(e_{i_1},\ldots,e_{i_{t-1}}),\quad R_t=C(c)\setminus p_t$$

$$a_t\sim\pi_\phi(a\mid c,p_t,R_t),\quad a_t\in R_t$$

最终 evidence list 由逐步 action rollout 得到：

$$L_k=(a_1,a_2,\ldots,a_k)$$

当前 LLM selector 方向探索的核心问题就是：能否训练一个策略 `π_φ`，让它不仅能在 teacher-forced 的正确 prefix 下模仿 oracle 下一步选择，还能在真实 rollout 或错误 prefix 下继续选择有用 evidence。

当前实验使用的主要数据边界是：

- 候选池来自 Stage2 oracle artifacts，通常为每条 claim 最多 15 条候选 evidence。
- oracle 输出 `selected_indices`，表示 margin-search 下较优的 evidence 序列。
- selector 训练目标不是直接预测真假标签，而是学习如何从候选池里选择 evidence。
- 评估以 selection-only gate 为主，核心指标包括 `Recall@5`、`Jaccard@5`、`NDCG@5`、`top1_match`，并保留 hybrid / candidate-pool controls。

这个方向的关键难点不是“能否在训练集上复现 oracle action”，而是：

1. 能否在验证集上选到泛化有效的 evidence set；
2. 能否在顺序敏感的 verifier pipeline 中维持有用的 evidence ordering；
3. 能否避免 selector 只学习候选位置、action label 或训练样本路径；
4. 能否在前缀 prefix 选错后继续恢复，而不是依赖 teacher-forced 的正确 prefix。

## 2. 为什么选择 LLM Selector

早期 selector 实验包括 pointwise、cross-encoder、listwise、sequential pointer 等路线。这些模型可以有效利用检索分数、候选位置和文本相似度，但表达能力相对受限，尤其是在以下场景中可能不足：

- evidence 与 claim 的关系不是简单词面重合；
- claim 需要隐含实体、时间、数量、立场或语义约束；
- 多条 evidence 之间存在互补、冗余或纠错关系；
- prefix 已经选入部分 evidence 后，下一步候选的价值会发生变化。

LLM selector 的动机是利用大模型更强的文本理解和组合推理能力，让模型在 `claim + selected prefix + remaining candidates` 的上下文里判断下一条 evidence。直觉上，它应该比轻量 encoder/ranker 更擅长处理：

- claim 与 evidence 的自然语言蕴含/冲突；
- 多 evidence 之间的上下文补充；
- prefix-conditioned utility，即“当前已经选了这些 evidence，下一条应该补什么”；
- 错误 prefix 下的恢复能力。

因此，LLM selector 最初被设计为一个 action selector：

```text
输入: claim + selected prefix + remaining candidates
输出: 下一条 evidence action
```

第一版目标是先通过 selection-only eval 验证 evidence selection 能否超过已有静态/序列 selector，再决定是否接入 full pipeline verifier 或进一步做 OPD/GRPO。

## 3. 从第一版到当前版本的改进链

### V0: Teacher-forced LLM Action Selector

第一版使用 teacher-forced oracle prefix 构造 action samples。每个样本对应一个 `(claim, step)`：

```text
Selected prefix: oracle selected evidence before this step
Remaining candidates: candidates not in prefix
Target action: oracle selected evidence at this step
```

训练目标为：

```text
loss = hard_action_ce + soft_loss_weight * soft_listwise_ce
soft target = softmax(delta_margin / soft_tau)
```

其中 `delta_margin` 来自 saved-score VIG rows，含义是 verifier margin 在当前 prefix 下加入某候选后的变化。

这版很快暴露出两个工程问题：

- 原 continuation likelihood 模式需要对每个候选都 forward 一次 `prompt + action + eos`，训练慢且容易 OOM。
- 默认 `MAX_LENGTH=2048` 成本较高，而同步 manifest 显示 train/val p99 约为 907/905，因此 1024 足以覆盖绝大多数样本。

对应改进：

- 默认 `MAX_LENGTH` 改为 1024。
- 增加 `score_mode=action_token`，每个 action sample 只 forward 一次 prompt，在最后 token logits 上 gather action label 分数。
- 保留 `score_mode=continuation` 作为旧行为回退。

### V1: Action Token 加速与标签格式修正

最初 action label 使用 `E00..E14`。实际 tokenizer 检查发现：

```text
E04 -> ['E', '0', '4']
```

即 `E04` 不是单 token，无法安全用于 `action_token` 快速路径。于是 action label 改为 `A..O`，completion 使用带前导空格的单字母形式，例如：

```text
target_action = " A"
```

这一步的目的：

- 保证 action-token scoring 与 tokenizer 契约一致；
- 避免多 token action label 让 logits gather 语义不清；
- 让 fast scoring 成为默认路径。

同时补充了 selector-local observability：

- `logs/train_llm_action_selector.log`
- `train_history.jsonl`
- `val_history.jsonl`
- `selection_history.jsonl`
- `eval_history.jsonl`
- SwanLab train/val/selection metrics

### V2: Checkpoint 与 Eval 输出规范化

早期 best checkpoint 直接写在运行根目录，adapter/tokenizer 文件和日志/metrics 混在一起，不利于复现实验与后续 eval。

对应改进：

- best checkpoint 改为独立目录：
  - `checkpoints/best_action/`
  - `checkpoints/best_selection/`
- root `selector_metadata.json` 记录 checkpoint layout、best metric、best checkpoint path。
- eval script 支持输入 run root 并自动解析 best checkpoint。
- 训练期 selection eval 和训练后完整 eval 都落盘：
  - `metrics/latest_val.json`
  - `metrics/best_val.json`
  - `metrics/latest_selection_val.json`
  - `metrics/latest_selection_eval.json`
  - `evals/during_train/`
  - `evals/val/`

这一步解决的是实验管理问题：不再只看 action accuracy，而是让 `Jaccard@5/NDCG@5` 成为 checkpoint 选择和实验比较的一等指标。

### V3: Local Choice Label 与候选顺序随机化

初始 action label 存在位置绑定风险：candidate idx 固定映射到 action label。模型可能学习“某些位置/字母更容易是答案”，而不是学习 evidence 本身。

对应改进：

- 增加 `ACTION_LABEL_MODE=local_choice`。
- 每一步根据当前展示顺序重新分配 `A..O`。
- 训练样本使用 `TRAIN_CANDIDATE_ORDER=random`，评估保持 `EVAL_CANDIDATE_ORDER=candidate_pool`。
- manifest/metadata 写入 label mode、candidate order 和 seed，避免新旧数据混淆。

这一步的目的是削弱 action label 与原始 candidate index 的固定绑定，缓解位置偏置。

### V4: Overfit Sanity 与 Token Boundary 修正

在正式训练前，加入 overfit sanity 流程：

```bash
SANITY_MODE=overfit_train_sample
TRAIN_SAMPLE_LIMIT=128
VAL_DATA=TRAIN_DATA
```

早期 overfit sanity 没有明显学习趋势，说明需要先排查训练链路，而不是直接大规模训练。

后续修正 action token 边界与 target completion 后，overfit sanity 出现明确学习信号：

- `overfit_128_local_choice_v4_hard_lr1e4`: best action acc 约 0.998
- `overfit_128_local_choice_v5_soft_lr1e4`: best action acc 约 0.991

结论：

- LoRA/梯度/score path 本身可以学习；
- 训练集可被模型拟合；
- 后续泛化差不是简单的实现断路，而是目标和数据状态的问题。

### V5: 中等规模训练暴露 Train-Val Gap

代表性运行：

```text
outputs/selectors/llm_action_selector/qwen25_3b_vig_soft_lr1e4_sw005
```

配置约为：

```text
LR=1e-4
SOFT_LOSS_WEIGHT=0.05
TRAIN_SAMPLE_LIMIT=2048
VAL_SAMPLE_LIMIT=1024
EPOCHS=10
```

观察：

- train action acc 随训练后期上升，最后可到约 0.72；
- val action acc 最好约 0.123，明显低于训练集表现；
- 完整 selection eval 512 条：
  - `Recall@5 = 0.3641`
  - `Jaccard@5 = 0.2422`
  - `NDCG@5 = 0.3197`
  - `top1_match = 0.1484`

这说明模型确实能在训练样本上学到某种 action mapping，但没有很好转化为验证集 evidence selection 能力。

后续尝试包括：

- 更低 soft weight / 更尖 soft tau / set loss；
- 较高学习率；
- best checkpoint 改用 selection metric；
- 加强 LR/scheduler 日志。

但整体结论一致：train acc 提升没有稳定带来 val selection 提升。

### V6: Soft/Set Loss 与 LR Ablation

代表性运行：

```text
qwen25_3b_vig_soft_lr1e4_sw002_tau01_set01
qwen25_3b_vig_lr2e4_sw002
```

观察：

- `SET_LOSS_WEIGHT=0.1`、`SOFT_TAU=0.1` 一类配置没有改善泛化，甚至出现 label collapse 风险。
- `LR=2e-4` 可以让训练集进一步拟合，最后 train acc 可到 1.0，但 val acc 和 selection 指标仍不理想。
- `qwen25_3b_vig_lr2e4_sw002` 的 512 条 selection eval：
  - `Jaccard@5 = 0.2442`
  - `NDCG@5 = 0.2724`
  - `top1_match = 0.0566`

结论：

- 单纯提高 LR 主要加快训练集拟合，不解决验证集泛化。
- soft/set loss 如果权重或温度不合适，容易放大噪声或 label prior。
- 当前瓶颈不再是“收敛不够快”，而是“teacher-forced action imitation 的泛化目标不稳”。

### V7: Label-Bias Calibration 的定位

action-token selector 直接用 raw label-token logits 排序：

```text
score(candidate) = logits[last_position, action_label_token]
```

这天然可能带来字母 token 先验，比如某些 action label 在无信息条件下 logit 更高。

因此提出 label-bias calibration：

```text
calibrated_score = raw_score - alpha * bias[action_label]
```

但阶段性判断是：

- 它可以作为诊断或轻量 decode 修正；
- 它可能缓解 label collapse；
- 它不能根治 train-val gap，也不能让模型在错误 prefix 下恢复。

所以 label-bias calibration 被定位为辅线，不是主线。

### V8: 当前版本 Robust-Prefix Preference-Style SFT

当前最新代码把目标从纯 teacher-forced next-action imitation 推向 prefix-robust training。

核心变化：

1. 训练时动态 candidate-order augmentation
   - build 阶段保存 `candidate_text_by_idx`、`candidate_score_by_idx` 等结构化字段；
   - 训练 dataset 每个 epoch 动态重排 remaining candidates；
   - local choice label 随重排重新分配；
   - 防止模型跨 epoch 记住固定 prompt/action mapping。

2. bad-prefix samples
   - 额外构造 `hybrid`、`random_corrupt` prefix；
   - 用高分非 oracle 或随机非 oracle evidence 替换正确 prefix 中的 1-2 个位置；
   - 样本目标不是唯一 hard next action，而是 remaining oracle positives。

3. multi-positive / pairwise loss
   - oracle-prefix 样本仍保留 hard CE；
   - bad-prefix 样本默认不使用 hard CE；
   - 所有 remaining oracle candidates 都作为 positive；
   - pairwise loss 鼓励 positive scores 高于 negatives：

```text
loss_pairwise = mean softplus(-(score_pos - score_neg))
```

4. bad-prefix validation
   - 新增：
     - `bad_prefix_remaining_oracle_hit@1`
     - `bad_prefix_positive_prob`
     - `bad_prefix_val_loss`
   - 用来观测模型在 prefix 选错后是否还能补救。

当前推荐启动形态：

```bash
OUTPUT_DIR=outputs/selectors/llm_action_selector/qwen25_3b_robust_prefix_v1_smoke \
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

## 当前阶段性结论

截至目前，LLM selector 方向经历了三个阶段：

1. 工程可行性阶段：解决 OOM、速度、tokenizer label、日志和 checkpoint 管理问题。
2. 学习链路验证阶段：通过 overfit sanity 证明模型/LoRA/score path 可以拟合训练样本。
3. 泛化问题定位阶段：确认主要问题是 teacher-forced next-action imitation 泛化差，而不是单纯 LR、batch size 或训练不够。

因此当前主线不再是继续调 action CE，而是验证 robust-prefix 目标是否能改善：

- train-val action gap；
- bad-prefix recovery；
- selection `Jaccard@5/NDCG@5`；
- label/position collapse。

若 robust-prefix v1 仍无法超过当前 action selector 和 saved-score utility baseline，则下一步应考虑 DPO/IPO 或 GRPO，但前提是先确认 prefix-robust supervised/preference signal 本身有效。

## 推荐后续观察指标

优先看：

```text
val/bad_prefix/remaining_oracle_hit@1
val/bad_prefix/positive_prob
val/selection/jaccard@5
val/selection/ndcg@5
val/selection/top1_match
```

辅助看：

```text
train/action_accuracy
train/remaining_oracle_hit@1
val/action_accuracy
selected action label distribution
selected candidate index/rank distribution
```

验收判断：

- 如果 bad-prefix hit@1 明显高于随机，同时 selection metrics 上升，说明 robust-prefix 方向值得继续。
- 如果 train 指标继续上升但 val/selection 不动，说明 LLM action-label imitation 仍然在背题，需要转向更直接的 preference/DPO 或 candidate-utility reward。
- 如果出现单 label 或单位置 collapse，则优先检查 label-bias calibration、candidate-order augmentation 是否生效。
