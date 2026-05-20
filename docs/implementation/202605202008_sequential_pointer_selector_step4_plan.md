# Sequential Pointer Selector Step4 实现计划

生成日期: 2026-05-20 20:08 CST

## 当前结论与目标

Step1 cross-encoder pairwise 与 Step3 set-aware listwise 均已 No-Go。最新 Step3 rank-prior ablation 结论是：去掉显式 rank/index prior 后，`recall@5` 只从 `0.3732` 提升到 `0.3826`，`jaccard@5` 只从 `0.2518` 提升到 `0.2588`，仍远低于 `0.50 / 0.35` gate。因此 Step4 不应继续微调 listwise rank prior，而应转向 sequential pointer selector，直接建模 oracle greedy order 与 prefix-dependent selection。

Step4 的目标是在固定 Stage2 sentence-level top15 candidate pool 中，按步骤输出 ordered top5：

```text
state_t  = claim + candidate_pool + selected_prefix + remaining_mask + deployment-time features
action_t = one remaining candidate index
output   = [a_1, a_2, ..., a_K], K <= 5
```

第一版 Step4 只做 supervised teacher-forcing sequential selector，不做 OPD / DAgger，也不做 GRPO。Step5 OPD 需要以 Step4 可用 baseline 为前置。

## 不变约束

必须沿用当前 selector 主线约束：

```text
chunk_mmr_fingerprint = 432dfc970e75
chunking.strategy     = sentence
candidate pool        = saved Stage2 oracle candidate_pool top15
selector output       = ordered top5
oracle objective      = margin
```

部署时 selector 输入不得使用：

```text
gold_label
oracle margin
oracle selected flag
oracle label logprobs
is_correct
```

这些字段只能用于训练标签、过滤、分析和报告。

## 复用边界

优先复用已有模块，避免重新造候选池和指标：

| 现有模块 | Step4 用法 |
|---|---|
| `src/fact_checking/selectors/stage2_oracle.py` | 继续读取、审计 Stage2 oracle rows，保留 fingerprint fail-fast |
| `src/fact_checking/selectors/metrics.py` | 继续产出 ordered selection metrics、controls、trace |
| `src/fact_checking/selectors/listwise.py` | 复用 `build_numeric_features`、tokenize/pooling helper 设计；避免复制已验证的特征逻辑 |
| `src/fact_checking/build/candidates.py` | 新增 `sequential_selector` selection_method 分支，结构参考 `listwise_selector` |
| `scripts/selectors/train_listwise_selector.py` | 训练循环、DDP、metadata、validation 保存方式可作为模板 |
| `scripts/selectors/eval_listwise_selector.py` | selection-only eval、controls、trace 输出方式可作为模板 |

## 计划新增文件

```text
src/fact_checking/selectors/sequential.py
src/fact_checking/selectors/test_sequential.py
scripts/selectors/train_sequential_selector.py
scripts/selectors/eval_sequential_selector.py
scripts/selectors/run_sequential_step4.sh
configs/experiment/b3_sequential_stage2_sentence_1024.yaml
docs/implementation/<timestamp>_sequential_pointer_selector_step4.md
```

计划修改文件：

```text
src/fact_checking/build/candidates.py
src/fact_checking/selectors/__init__.py
```

## 模型设计

### Candidate 表示

第一版采用与 Step3 相同的 pair encoder 路线：

```text
input_i = "Claim: <claim>\nEvidence: <candidate_text>"
h_i     = AutoModel(input_i).pooled_or_mean_embedding
```

第一版不拼接任何数值特征。`hybrid_score`、`sent_idx_norm`、`source_index_norm`、`text_token_len_norm`、`claim_token_overlap`、`number_overlap` 都不作为主线 always-on 特征，只保留为后续 ablation 或 error analysis 辅助。默认不要重新引入显式 `hybrid_rank_norm`、`candidate_idx_norm` 或 `rank_embedding`。Step3 rank-ablation 已证明这些信号容易形成 top1 shortcut。

### 判别特征清单与作用

Step4 的主力判别特征应从浅层统计转为深层语义交互。第一版只实现 semantic backbone 与 prefix-conditioned semantic interaction；fact-checking targeted semantic features、retrieval prior 和 shallow controls 只作为后续 profile 扩展。除训练 target 外，以下特征都必须是部署时可得信息。

| 特征组 | 字段 | 来源 | 推理可用 | 具体作用 |
|---|---|---|---|---|
| Pair 语义表示 | `h_i_pair` | `AutoModel("Claim: ... Evidence: ...")` | 是 | 识别 candidate 与 claim 的深层语义关系，是基础 evidence utility 表示 |
| Set 上下文化表示 | `H_i_ctx` | `TransformerEncoder(z_1, ..., z_N)` | 是 | 在同一 claim 的候选集合内重新编码 candidate，表达候选间互补、竞争和冗余 |
| Prefix 表示 | `P_t` | 已选 `H_j_ctx` 的 attention/mean pooling | 是 | 表示当前已选择 evidence prefix，是 step-wise selection 的状态语义核心 |
| Candidate-prefix 乘积 | `H_i_ctx * P_t` | 深层表示交互 | 是 | 捕捉 candidate 与已选 prefix 的语义匹配/冲突模式 |
| Candidate-prefix 差分 | `abs(H_i_ctx - P_t)` | 深层表示交互 | 是 | 捕捉 candidate 相对 prefix 的新增信息或语义偏离 |
| Candidate-prefix 相似度 | `cos(H_i_ctx, P_t)` | 深层表示交互 | 是 | 衡量 candidate 与当前 prefix 的整体接近程度 |
| 双线性交互 | `bilinear(H_i_ctx, P_t)` | 可学习 interaction head | 是 | 让模型学习非对称的“当前 prefix 下下一条 evidence utility” |
| Claim aspect 表示 | `A_m` | claim decomposition / aspect encoder | 是，若离线缓存 | 表示 claim 的实体、事件、时间、数量、关系等可核查方面 |
| Candidate-aspect attention | `attn(H_i_ctx, A_m)` | aspect-aware attention | 是，若 aspect 已缓存 | 判断 candidate 覆盖哪个 subclaim/aspect，避免只围绕单一方面选证据 |
| Aspect coverage state | `covered_aspects_t` | prefix 与 claim aspects 的 attention 聚合 | 是 | 判断当前 prefix 已覆盖哪些事实核查方面，指导下一步补缺 |
| 辩护视角语义表示 | `D_i` | claim-candidate stance/utility encoder | 是，若离线缓存 | 表示 candidate 对 claim 的 support/refute/qualify/insufficient 语义，不使用 gold label |
| 辩护视角 prefix 平衡 | `balance(D_i, D_prefix)` | 辩护视角表示交互 | 是，若该表示已缓存 | 判断下一条 evidence 是否补充另一侧辩护/反驳信息，避免单侧证据堆叠 |
| Graph-lite 节点表示 | `G_claim`, `G_i` | claim/candidate entity-time-number-relation extraction | 是，若离线缓存 | 表达 fact-checking 中实体、时间、数量、关系节点的结构覆盖 |
| Graph alignment 表示 | `align(G_i, G_claim, G_prefix)` | graph-lite matching / attention | 是，若 graph 已缓存 | 判断 candidate 是否补充未覆盖实体/时间/数量/关系路径 |
| 已选 mask | `selected_mask_t` | 当前 decoding prefix | 是 | 防止重复选择，是 action mask，不拼入 scorer |
| 剩余 mask | `remaining_mask_t` | 当前 decoding prefix | 是 | 限制 action space，只允许从未选 candidates 中选择 |
| 连续 retrieval prior | `hybrid_score` | Stage2 saved candidate score / build candidate pool | 是，但第一版不用 | 仅作为 control / 后续 ablation，不拼入 deep baseline |
| 文本/位置浅层特征 | `sent_idx_norm`, `source_index_norm`, `text_token_len_norm` | candidate metadata/text | 是，但默认不用 | 仅作为 error analysis 或 ablation，避免浅层 shortcut 主导 |
| 词面/数字 overlap | `claim_token_overlap`, `number_overlap` | claim + candidate text | 是，但默认不用 | 仅作为 graph/aspect 特征不可用时的 cheap fallback |

默认主线特征配置：

```text
always_on:
  h_i_pair
  H_i_ctx
  P_t
  H_i_ctx * P_t
  abs(H_i_ctx - P_t)
  cos(H_i_ctx, P_t)
  bilinear(H_i_ctx, P_t)
  selected_mask_t
  remaining_mask_t

ablation_only:
  A_m claim aspect embeddings
  candidate-aspect attention
  covered_aspects_t
  D_i stance/utility embeddings
  stance-prefix diversity/balance
  G_claim / G_i graph-lite embeddings
  graph alignment features
  hybrid_score
  dense_score
  lexical_score
  bm25_log_norm
  sent_idx_norm
  source_index_norm
  text_token_len_norm
  claim_token_overlap
  number_overlap

disabled_by_default:
  hybrid_rank_norm
  candidate_idx_norm
  rank_embedding
```

这些特征的分工是：

1. `h_i_pair` 负责 claim-candidate 的基础深语义判别。
2. `H_i_ctx` 负责 candidate set 内的上下文建模，判断候选之间的竞争、互补和冗余。
3. `P_t` 以及 `H_i_ctx * P_t`、`abs(H_i_ctx - P_t)`、cosine、bilinear 交互负责 Step4 最关键的 prefix-conditioned semantic interaction。
4. fact-checking targeted features 负责把语义空间对齐到任务结构：claim aspects、辩护视角的证据作用、entity-time-number-relation graph。
5. `selected_mask_t` / `remaining_mask_t` 负责 action space 约束，保证 ordered top5 无重复。
6. retrieval prior、浅层 overlap、位置、长度特征只作为 control/ablation，因为它们可能改善局部 recall，也可能引入浅层 shortcut。

### Fact-checking targeted semantic features

针对 fact-checking，应该加入比通用 overlap 更贴任务的语义特征。但这里的目标不是直接复刻 L-defense 或 G-defense，而是参考它们暴露出的任务结构：证据是否能形成辩护/反驳、claim 是否可拆成可核查方面、候选证据是否覆盖实体-时间-数量-关系结构。所有 targeted features 都必须保持轻量、可缓存、可消融，并且不改变 Stage2 candidate pool 边界。

| 模块 | 是否第一版主线 | 设计 | 作用 | 风险 |
|---|---|---|---|---|
| Claim aspect decomposition | 建议作为第一版可选缓存 | 将 claim 分成 `entity / event / time / quantity / relation / qualifier` 等 aspects，并编码为 `A_m` | 让 selector 选择覆盖不同事实核查方面的证据 | 分解错误会传播；需要缓存和审计 |
| Aspect-aware coverage | 建议第一轮 ablation | 计算 candidate 对每个 `A_m` 的 attention，以及 prefix 已覆盖 aspect state | 判断下一条证据是否补足未覆盖 subclaim | 可能过度追求覆盖而牺牲 verifier utility |
| 辩护视角语义特征 | 建议第一轮 ablation | 参考 L-defense 的辩护视角，但不复刻其生成式防御流程；只编码 candidate 对 claim 的 support / refute / qualify / insufficient 证据作用 | 帮助模型区分“看似相关”与“能为某一侧辩护/反驳”的 evidence | 自动生成/抽取可能有幻觉；不能用 gold label |
| 结构一致性 / graph-lite 特征 | 建议第二轮 ablation | 参考 G-defense 的 claim 分解与图增强问题意识，但只从 claim/candidate 抽取 entity-time-number-relation 节点，构造轻量 matching 表示 | 对事实核查中的实体、数字、时间、关系一致性更敏感 | 完整图构建成本高，工程复杂 |
| Full graph reasoning / KG expansion | 暂不进 Step4 第一版 | 引入外部 KG 或多跳图推理 | 可能提升解释性与复杂 claim 覆盖 | 会改变 retrieval boundary，容易和 selector 实验混淆 |

第一版建议落地的 targeted features 是低成本、可缓存的版本：

```text
claim_aspects:
  entity_spans
  event_predicate
  time_expressions
  quantity_expressions
  relation_phrases
  qualifiers / negation / modality

candidate_aspect_alignment:
  aspect_attention_scores
  max_aligned_aspect
  uncovered_aspect_gain_t

stance_utility_semantics:
  support_score
  refute_score
  qualify_score
  insufficient_score
  stance_embedding D_i

graph_lite:
  entity_overlap_nodes
  time_match_nodes
  quantity_match_nodes
  relation_match_edges
  contradiction_edges
```

这些 feature 必须满足两个边界：

1. 只能来自 claim、candidate text、retrieval metadata 或离线无标签解析结果。
2. 不得使用 gold label、oracle selected flag、oracle margin、verifier gold-side logprobs 或人工 verdict。

实现顺序建议：

1. Step4 baseline 先只使用 deep semantic interaction：`h_i_pair`、`H_i_ctx`、`P_t`、乘积/差分/cosine/bilinear。
2. 第一轮 ablation 加 `claim_aspects + aspect-aware coverage`。
3. 第二轮 ablation 加受辩护视角启发的 `stance_utility_semantics`。
4. 第三轮 ablation 加受图增强启发的 `graph_lite`。
5. 只有当 graph-lite 有稳定增益，再考虑完整图构建或外部 KG expansion。

明确禁止作为 selector 推理输入的字段：

```text
gold_label
oracle margin
oracle selected flag
oracle label logprobs
is_correct
selected_indices
```

`selected_indices` 只能作为 teacher-forcing 训练 target 和离线评估 oracle，不得作为模型 feature。

### Set/context 编码

候选表示先经过 projection，再进入 TransformerEncoder。第一版只投影 pair encoder 表示，不拼接 `hybrid_score` 或其他 metadata：

```text
z_i     = projection(h_i_pair)
H_i_ctx = TransformerEncoder(z_1, ..., z_N, attention_mask)
```

`H_i_ctx` 是候选在当前 claim 的 candidate set 中的 contextual representation。N <= 15，计算成本可控。

### Prefix state

每个 step t 需要构造 prefix-dependent state：

```text
P_t
candidate_prefix_interaction_i,t
selected_mask_t / remaining_mask_t
```

第一版 prefix representation：

```text
if t == 0:
    P_t = learned_start_prefix
else:
    P_t = attention_or_mean_pool(H_j_ctx for j in selected_prefix)
```

第一版 candidate-prefix 深层交互特征：

```text
H_i_ctx * P_t
abs(H_i_ctx - P_t)
cos(H_i_ctx, P_t)
bilinear(H_i_ctx, P_t)
```

targeted semantic features 不作为第一版 always-on，而是分阶段打开：

```text
none
aspect
aspect_stance
aspect_stance_graph
```

词面 overlap、位置、长度只保留为 shallow control。若 targeted features 不稳定，优先回退到 deep semantic interaction baseline，而不是把 cheap coverage 当作默认主线。

### Action scorer

每步对 remaining candidates 打分：

```text
logit_i,t = MLP([
    H_i_ctx,
    P_t,
    H_i_ctx * P_t,
    abs(H_i_ctx - P_t),
    cos(H_i_ctx, P_t),
    bilinear(H_i_ctx, P_t),
])
```

然后对已选 candidates 和 padding candidates mask 为 `-inf`，只在 remaining candidates 中 softmax。

推理时：

```text
selected = []
for t in range(top_k):
    logits = model(state_t)
    action = argmax(logits over remaining)
    selected.append(action)
return selected
```

第一版不加 STOP action。当前 Stage2 oracle `selected_indices` 通常 top5，且 pipeline 需要 top5 evidence；若后续发现 `len(selected_indices)<5` 的样本有系统影响，再加 STOP action。

## 训练目标

### Teacher forcing CE

训练时 prefix 使用 oracle prefix：

```text
prefix_t = selected_indices[:t]
target_t = selected_indices[t]
```

loss：

```text
L_seq = sum_t w_t * CE(logits_t over remaining, target_t)
```

位置权重：

```text
w_t = 1 / log2(t + 2)
```

该 loss 直接约束每一步选择 oracle greedy order 的下一条 evidence，避免 Step3 listwise 把顺序压成单次全局 score。

### 辅助 loss

第一版建议只加轻量辅助，不要过早复杂化：

```text
L = L_seq + 0.2 * L_mask
```

`L_mask` 是 selected-vs-nonselected BCE，可复用 Step3 selected-mask 思路，帮助模型先识别 oracle selected set。不要对 non-selected 之间引入任意顺序监督。

### Scheduled sampling 暂不做

Step4 第一版只做 teacher forcing，避免把 Step4 与 Step5 OPD 混在一起。若 teacher-forcing 模型 selection-only order 指标提升但 inference 崩，才进入 Step5 OPD / DAgger。

## 数据与过滤策略

主线训练使用：

```text
train_oracle_results = outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl
val_oracle_results   = outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl
filter_policy        = all
```

过滤实验只作为辅助诊断：

```text
margin_positive
is_correct
high_margin
```

主表必须以 all rows 为准。过滤模型不能替代全量 val gate。

## 训练脚本接口

`scripts/selectors/train_sequential_selector.py` 参数计划：

```text
--train-oracle-results
--val-oracle-results
--output-dir
--model-name /data/models/deberta-v3-base/
--expected-chunk-mmr-fingerprint 432dfc970e75
--max-candidates 15
--top-k 5
--filter-policy all
--max-length 384
--batch-size 2
--epochs 2
--learning-rate 2e-5
--head-learning-rate 1e-4
--weight-decay 0.01
--warmup-ratio 0.06
--gradient-accumulation-steps 1
--seq-loss-weight 1.0
--mask-loss-weight 0.0
--list-hidden-size 256
--list-layers 2
--list-heads 4
--dropout 0.1
--freeze-pair-encoder
--semantic-feature-profile deep
--targeted-feature-profile none
--shallow-feature-profile off
--eval-every 500
--early-stopping-metric oracle_rank_ndcg@5
--early-stopping-patience 4
--ddp-find-unused-parameters
--no-progress
```

第一版实现默认 `semantic_feature_profile=deep`、`targeted_feature_profile=none`、`shallow_feature_profile=off`，不接入任何 retrieval/meta 数值特征。后续如果加入 aspect / stance_utility / graph_lite，应扩展 profile 选择并作为独立 ablation 打开。

## 保存产物

模型目录：

```text
outputs/selectors/stage2_sentence_sequential/deberta_sequential_deep
```

必须保存：

```text
encoder config / weights
tokenizer files
sequential_head.pt
metadata.json
selection_metrics.json
val_trace.jsonl
```

`metadata.json` 至少包含：

```text
selector_type = sequential_pointer
base_model
train_oracle_results
val_oracle_results
chunk_mmr_fingerprint
candidate_pool_policy
filter_policy
top_k
max_candidates
max_length
semantic_feature_profile
targeted_feature_profile
shallow_feature_profile
losses
model_config
seed
best_metric
best_metric_value
global_step
```

## Selection-only eval

`scripts/selectors/eval_sequential_selector.py` 应输出：

```text
selection_metrics.json
selection_trace.jsonl
control_hybrid_trace.jsonl
control_candidate_pool_trace.jsonl
```

复用当前 ordered metrics：

```text
recall@5
precision@5
jaccard@5
top1_match
prefix_match@1 / @3 / @5
ordered_hit@5
oracle_rank_ndcg@5
pairwise_order_acc@5
ordered_exact_match@5
```

必须同时报告 controls：

```text
hybrid_score_top5
candidate_pool_order_top5
same predicted set + hybrid-order
same predicted set + candidate-pool-order
same predicted set + random-order seeds 0-4
```

Sequential 特有 trace 字段：

```text
step_logits_topk
selected_index
selected_score
step_entropy
remaining_indices_before_step
```

这些字段用于判断错误是否集中在 top1、后续 prefix drift，还是候选 set 本身不可学。

## Build pipeline 接入

新增 selection_method：

```text
build.retrieval.selection_method=sequential_selector
```

新增 config：

```yaml
build:
  retrieval:
    selection_method: sequential_selector
    sequential_selector:
      model_dir: outputs/selectors/stage2_sentence_sequential/deberta_sequential_deep
      candidate_pool_size: 15
      max_length: 384
      batch_size: 8
      device: cuda
      dump_trace: true
      strict_fingerprint: true
```

`src/fact_checking/build/candidates.py` 中新增分支应与 listwise 分支同级：

```text
load SequentialSelector once
for each chunk sample:
    build_pointwise_inference_pool(...)
    selector.score/select(...)
    write build_<split>.jsonl
    optionally dump sequential_selector_trace_<split>.jsonl
```

build trace 需要保留每步选择过程，而不是只保存最终 top5。

## Wrapper

新增：

```text
scripts/selectors/run_sequential_step4.sh
```

默认：

```text
MODEL_NAME=/data/models/deberta-v3-base/
OUTPUT_DIR=outputs/selectors/stage2_sentence_sequential/deberta_sequential_deep
TRAIN_ORACLE_RESULTS=outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl
VAL_ORACLE_RESULTS=outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl
FILTER_POLICY=all
SEMANTIC_FEATURE_PROFILE=deep
TARGETED_FEATURE_PROFILE=none
SHALLOW_FEATURE_PROFILE=off
```

注意：Sequential selector 不需要 Step3 的 candidate permutation augmentation 作为默认主线。它的顺序来自 step-wise mask / prefix，而不是 candidate 初始顺序；若要做 permutation robustness，单独作为 ablation。

## 测试计划

新增 `src/fact_checking/selectors/test_sequential.py`，至少覆盖：

1. `sequential_teacher_forcing_loss` 在正确 step logits 下低于错误 logits。
2. 已选 candidate 在下一步被 mask，不能重复选择。
3. padding candidates 被 mask，不能被选择。
4. 推理输出长度为 `top_k`，且无重复 index。
5. 第一版 profile 锁定为 `semantic_feature_profile=deep`、`targeted_feature_profile=none`、`shallow_feature_profile=off`，其他 profile 先 fail-fast。
6. shallow feature profile 不会默认暴露 rank/index/position shortcut。
7. `SequentialSelector` 读取 metadata 后能恢复 semantic/targeted/shallow profile 和 fingerprint 约束。

轻量验证命令：

```bash
PYTHONPATH=src python -m compileall -q src/fact_checking/selectors scripts/selectors/train_sequential_selector.py scripts/selectors/eval_sequential_selector.py src/fact_checking/build/candidates.py
PYTHONPATH=src python -m unittest src/fact_checking/selectors/test_metrics.py src/fact_checking/selectors/test_sequential.py
PYTHONPATH=src python scripts/selectors/train_sequential_selector.py --help
PYTHONPATH=src python scripts/selectors/eval_sequential_selector.py --help
bash -n scripts/selectors/run_sequential_step4.sh
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b3_sequential_stage2_sentence_1024 --cfg job
git diff --check
```

## 第一轮实验矩阵

先只跑少量必要变体，避免再次变成大规模调参：

| Run | 目的 | filter | semantic profile | targeted features | shallow profile | 期望 |
|---|---|---|---|---|---|---|
| `deberta_sequential_deep` | 主线 baseline | all | deep | none | off | set/order 同时超过 Step3 |
| `deberta_sequential_no_maskloss` | 检查 mask BCE 是否干扰 order | all | deep | none | off | NDCG/top1 不降 |
| `deberta_sequential_aspect` | 检查 claim 分解与 aspect coverage 的价值 | all | deep | aspect | off | recall/jaccard 或 prefix_match 上升 |
| `deberta_sequential_stance_utility` | 检查辩护视角证据作用特征 | all | deep | aspect_stance | off | top1/order 或 label-specific failures 改善 |
| `deberta_sequential_graph_lite` | 检查结构一致性特征 | all | deep | aspect_stance_graph | off | 数字/时间/实体错误 bucket 改善 |
| `deberta_sequential_shallow_control` | 诊断浅层特征是否 shortcut | all | shallow_control | none | full_shallow | 若只提升 top1 但 set/order 不稳，不进主线 |
| `deberta_sequential_margin_positive` | 高信号辅助诊断 | margin_positive | deep | none | off | 只看 order upper diagnostic |

主表只接受 all rows。

## Gate 与 stop/go

Selection-only gate：

```text
recall@5 >= 0.50
jaccard@5 >= 0.35
oracle_rank_ndcg@5 > Step3 best all-val
top1_match > Step3 best all-val
prefix_match@3 明显高于 Step3
```

当前 Step3 best all-val 参考：

```text
recall@5           = 0.3826
jaccard@5          = 0.2588
top1_match         = 0.1279
oracle_rank_ndcg@5 = 0.3131
```

进入 full pipeline 的最低条件：

```text
recall@5 >= 0.50
jaccard@5 >= 0.35
oracle_rank_ndcg@5 and top1_match clearly improve over Step3 / hybrid controls
no fingerprint mismatch
no duplicate selected evidence
no prompt truncation increase
```

如果 Step4 teacher-forcing 模型：

```text
selection-only set metrics 仍停在 recall@5≈0.38 / jaccard@5≈0.26
且 top1/order metrics 没有明显提升
```

则不进入 full pipeline，也不直接上 OPD；先做 error analysis，确认是 candidate pool supervision 不可学、pair encoder 不足、还是 prefix inference drift。

如果 Step4：

```text
teacher-forcing val loss 下降
oracle-prefix eval 好
free-running inference 差
```

则说明 exposure bias 是主因，应进入 Step5 OPD / DAgger。

## Error analysis

Step4 报告必须新增 step-wise 诊断：

```text
step1 accuracy / target rank
step2 accuracy conditioned on step1 correct
step2 accuracy conditioned on step1 wrong
per-step entropy
per-step target rank in logits
first wrong step distribution
duplicate prevention check
```

同时保留现有 selector buckets：

```text
selector hits oracle but verifier wrong
selector misses oracle and verifier wrong
selector set overlap high but order wrong
top1 wrong but set overlap high
high hybrid rank false positives
redundant false positives
label-specific failures
low-margin oracle rows
```

## 风险与处理

| 风险 | 处理 |
|---|---|
| exposure bias | Step4 先量化 free-running vs oracle-prefix gap；若明显，进入 Step5 OPD |
| rank shortcut 回潮 | 第一版完全不接入 rank/index/retrieval numeric features；后续 shallow control 必须单独对照 |
| 深层 prefix interaction 无效 | 先检查 oracle-prefix vs free-running gap、target rank、first wrong step；不要直接回退到浅层 overlap |
| targeted features 噪声大 | 逐个 profile 消融，只保留能在 all-row gate 和错误 bucket 同时改善的特征 |
| 训练成本过高 | N<=15，先沿用 DeBERTa-base；支持 `--freeze-pair-encoder` |
| 指标只提升 order 不提升 set | 不进入 full pipeline，先分析 target rank / first wrong step |
| 只在 margin_positive 上好 | 作为诊断，不替代 all-row gate |

## 实施顺序

1. 新增 `sequential.py`，实现 model、teacher-forcing loss、free-running decode、metadata load/save。
2. 新增 `test_sequential.py`，先覆盖 masking、loss、decode 无重复。
3. 新增 `train_sequential_selector.py`，复用 Step3 的 DDP / validation / metadata 保存骨架。
4. 新增 `eval_sequential_selector.py`，输出 ordered metrics、controls、step-wise trace。
5. 接入 `build/candidates.py` 的 `sequential_selector` 分支。
6. 新增 Hydra config 与 bash wrapper。
7. 跑轻量验证。
8. 跑主线训练 `deberta_sequential_deep`。
9. 根据 selection-only gate 决定是否 full pipeline 或进入 Step5 OPD。

## 本计划状态

2026-05-20 已实现第一版 `deep` Sequential Pointer Selector：默认只使用 `h_i_pair`、`H_i_ctx`、`P_t`、乘积/差分/cosine/bilinear 深层交互；targeted/shallow profile 暂锁为 `none/off`，保留接口用于后续扩展。
