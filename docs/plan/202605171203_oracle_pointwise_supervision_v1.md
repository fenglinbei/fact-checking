# Oracle-set Supervision V1: Filtered Data Pool + Pointwise Evidence Utility Model

生成日期: 2026-05-17

## 0. 目标

本计划定义第一版可执行闭环: 使用已生成的 train oracle evidence set 构造过滤后的监督数据，并实现 `6.1 Pointwise evidence utility model`，用最轻量模型验证 Oracle-set supervision 是否可被吸收。

第一版不追求最终 full 6-class selector。它的目标是回答一个更小的问题:

> 在高置信、低污染的 oracle 子集上，模型能否学会给 oracle-selected evidence 更高分，并在 retained buckets 上超过 fixed-MMR 或至少提升 oracle-overlap?

若这个问题失败，后续 DPO、sequential selector、multi-weight MMR 都不应贸然推进。

## 1. 当前输入数据

Oracle train set:

```text
outputs/oracle_evidence/20260517_041502/oracle_results_train.jsonl
outputs/oracle_evidence/20260517_041502/oracle_metrics_train.json
```

生成命令:

```bash
VERIFIER_MODEL=/data/models/Qwen2.5-7B-Instruct/ \
MODEL_BASE_PATH=/data/models/ \
LORA_ADAPTER=outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train/best \
SPLIT=train \
bash scripts/oracle_evidence/run_search.sh
```

运行事实:

| 项目 | 值 |
|---|---:|
| split | train |
| samples | 10065 |
| oracle accuracy | 43.38% |
| oracle macro-F1 | 39.02% |
| search method | greedy |
| target top-K | 5 |
| two-stage pruning | enabled |
| effective candidate cap | `top_k * 3 = 15` |

重要约束:

1. `selected_indices` 是 two-stage 后 pruned candidate pool 内的索引，不是完整 chunk cache 的原始索引。
2. 数据构造必须复现 `run_search.sh` 的候选池处理: chunk cache -> 去重 -> 按 `hybrid_score` 降序截断到最多 15 个候选。
3. Oracle 搜索目标是最大化 gold label token logprob；最终 `is_correct` 是另一次 label generation 后的 argmax/parse 结果。

## 2. 数据过滤策略

### 2.1 为什么不能直接全量使用

当前 verifier 存在明显 true-side false bias。即使 oracle search 已经最大化正确标签 logprob，`true / mostly-true` 的最终 argmax 仍大量落到错误类别。

按 train oracle 结果:

| gold label | N | Oracle Acc | 使用策略 |
|---|---:|---:|---|
| pants-fire | 812 | 79.19% | 高权重使用 |
| false | 1958 | 86.01% | 高权重使用 |
| barely-true | 1611 | 33.21% | 中等权重，需过滤 |
| half-true | 2087 | 46.67% | 高权重使用 |
| mostly-true | 1950 | 25.28% | 不做 hard oracle positive |
| true | 1647 | 2.25% | 不做 hard oracle positive |

因此 V1 主训练池只用于验证可吸收性，不代表最终类别分布。

### 2.2 V1a: High-confidence Absorption Probe

主过滤条件:

```text
oracle_correct == true
gold_label in {pants-fire, false, barely-true, half-true}
final_logprob >= -0.5
n_candidates > 5
```

解释:

- `oracle_correct == true`: 避免把 objective failure 当正例。
- 去掉 `mostly-true / true`: 避免把 verifier false bias 蒸馏进 selector。
- `final_logprob >= -0.5`: 保留 verifier 对 gold label 有较高置信的样本。
- `n_candidates > 5`: 排除没有真实选择空间的样本；当候选数小于等于 K，pointwise selector 学不到选择策略。

预计保留规模需要由脚本实际统计，当前粗略估计约 3k 级别。脚本必须输出过滤前后每类样本数。

### 2.3 类别均衡

过滤后不能按 raw sample mean 训练，否则 `false` 类会主导。训练目标应使用 label-balanced sampling 或 weighted loss。

推荐第一版使用 balanced sampler:

```text
每个 batch 内按 gold_label 均匀采样:
pants-fire : false : barely-true : half-true = 1 : 1 : 1 : 1
```

如果使用 weighted loss:

$$w_c = \frac{N_{\mathrm{filtered}}}{|\mathcal{C}| \cdot N_c}$$

其中 `C={pants-fire,false,barely-true,half-true}`。

报告时必须同时给:

1. retained-label macro 指标。
2. full-val 指标。
3. true / mostly-true 单独指标。

V1a 成功只说明 oracle supervision 可被吸收；不能说明 full 6-class task 已解决。

### 2.4 V1b: True-side Anchor Extension

若 V1a 在 retained buckets 有收益，再做 V1b，加入防退化 anchor。

Anchor 来源按优先级:

1. `mostly-true / true` 中 `oracle_correct == true` 的少量样本，只做低权重正例。
2. fixed-MMR 已判对的 `mostly-true / true` evidence set，作为 conservative anchor。
3. 若 fixed-MMR 逐条预测暂不可用，则 V1b 暂缓，避免构造伪 anchor。

Anchor 不参与高权重 hard imitation。建议权重:

```text
oracle-positive retained buckets: 1.0
mostly-true anchor: 0.25
true anchor: 0.10
```

V1b 的目标不是提升 true 类，而是确认 pointwise selector 不进一步破坏 true-side。

## 3. Pointwise Evidence Utility Model

### 3.1 任务定义

对每个 claim 的候选池 `C={d_i}`，训练模型输出每个 candidate 的 utility score:

$$u_\theta(c,d_i,C) \rightarrow \mathbb{R}$$

监督标签:

$$y_i = \mathbb{1}[d_i \in S_{\mathrm{oracle}}]$$

推理:

$$S_K = \operatorname{TopK}_{d_i \in C} u_\theta(c,d_i,C)$$

V1 使用 pointwise BCE 或 pairwise ranking loss。推荐先用 BCE + class-balanced candidate weights，减少实现复杂度。

### 3.2 特征

V1 不训练大模型，只用 chunk cache 中已有信息构造轻量特征。

每个 candidate 的特征建议:

| 特征 | 说明 |
|---|---|
| `hybrid_score` | 当前检索综合相关性 |
| `dense_score` / `lexical_score` / `bm25_score` | 若候选中存在则使用 |
| `rank_by_hybrid` | two-stage candidate 内排名 |
| `n_candidates` | 当前候选池大小 |
| `candidate_text_len` | 字符数或 token 数 |
| `claim_candidate_cosine` | claim embedding 与 candidate embedding 相似度，若 cache 可取 |
| `mean_sim_to_pool` | 与其他候选平均相似度，表示冗余 |
| `max_sim_to_pool` | 最大近重复程度 |
| `source_count_for_report` | 同 report 候选数量，若 metadata 可取 |
| `position_norm` | candidate 在 pruned pool 中的位置归一化 |

最低可行版本只需要:

```text
hybrid_score
rank_by_hybrid
n_candidates
candidate_text_len
mean_sim_to_pool
max_sim_to_pool
```

模型选择:

1. Logistic Regression: 最低成本 sanity check。
2. LightGBM / XGBoost: 第一版主模型，适合 tabular 特征。
3. MLP: 仅在 tabular baseline 有信号后再加。

V1 主模型建议 LightGBM；若依赖不可用，退化为 sklearn `HistGradientBoostingClassifier` 或 Logistic Regression。

## 4. 代码实现计划

### 4.1 数据构造脚本

新增:

```text
scripts/selectors/build_pointwise_oracle_dataset.py
```

输入:

```text
--oracle-results outputs/oracle_evidence/20260517_041502/oracle_results_train.jsonl
--config configs/experiment/b3_mmr_topk_sweep_1024.yaml
--split train
--output-dir outputs/oracle_pointwise/v1/data
--top-k 5
--two-stage-multiplier 3
--filter-preset v1a
```

核心流程:

1. 加载 Hydra build config。
2. 解析 chunk-MMR cache fingerprint 并加载对应 split cache。
3. 对每个 sample 复现 oracle search 的候选池:
   - `canonicalize_sentence` 去重。
   - 按 `hybrid_score` 降序截断到 `top_k * two_stage_multiplier`。
4. 读取 oracle result，按 `event_id` 对齐。
5. 校验:
   - oracle result 的 `n_candidates` 等于复现后的候选数。
   - `selected_texts` 与复现候选的 `selected_indices` 文本一致。
6. 应用过滤条件。
7. 展开为 pointwise rows:

```json
{
  "event_id": "2635.json",
  "gold_label": "false",
  "candidate_idx": 4,
  "is_oracle_selected": 1,
  "candidate_text": "...",
  "features": {
    "hybrid_score": 0.91,
    "rank_by_hybrid": 4,
    "n_candidates": 15,
    "candidate_text_len": 284,
    "mean_sim_to_pool": 0.37,
    "max_sim_to_pool": 0.82
  },
  "oracle_final_logprob": -0.03,
  "oracle_correct": true,
  "filter_bucket": "v1a"
}
```

输出:

```text
outputs/oracle_pointwise/v1/data/train_pointwise.jsonl
outputs/oracle_pointwise/v1/data/filter_report.json
outputs/oracle_pointwise/v1/data/feature_schema.json
```

### 4.2 训练脚本

新增:

```text
scripts/selectors/train_pointwise_oracle_selector.py
```

输入:

```text
--train-jsonl outputs/oracle_pointwise/v1/data/train_pointwise.jsonl
--output-dir outputs/oracle_pointwise/v1/lightgbm
--model lightgbm
--balance-labels true
--balance-candidates true
```

训练方式:

1. 按 `event_id` 划分 train/dev，避免同一 claim 的 candidates 泄漏。
2. 默认 `dev_ratio=0.1`。
3. loss/采样:
   - claim-level balanced: 每个 claim 内 positive/negative candidate 重新加权。
   - label-level balanced: retained labels 等权。
4. 保存模型和特征统计。

输出:

```text
outputs/oracle_pointwise/v1/lightgbm/model.pkl
outputs/oracle_pointwise/v1/lightgbm/training_metrics.json
outputs/oracle_pointwise/v1/lightgbm/feature_importance.json
outputs/oracle_pointwise/v1/lightgbm/dev_predictions.jsonl
```

训练指标:

| 指标 | 用途 |
|---|---|
| candidate AUROC / AUPRC | 判断是否能区分 oracle-selected candidate |
| claim Recall@5 | 每个 claim 的 oracle selected 被 top-5 命中比例 |
| claim Jaccard@5 | 预测 set 与 oracle set overlap |
| exact-set match | 严格指标，预期很低 |
| retained-label macro overlap | 防止只学 false 类 |

### 4.3 Selector 评估脚本

新增:

```text
scripts/selectors/eval_pointwise_oracle_selector.py
```

输入:

```text
--model-dir outputs/oracle_pointwise/v1/lightgbm
--config configs/experiment/b3_mmr_topk_sweep_1024.yaml
--split val
--oracle-results outputs/oracle_evidence/20260516_135632/oracle_results_val.jsonl
--output-dir outputs/oracle_pointwise/v1/lightgbm/eval_val
```

两层评估:

1. Selection-only evaluation:
   - oracle overlap
   - Jaccard@5
   - Recall@5
   - rank distribution
   - label bucket metrics
2. Verifier evaluation:
   - 用 selector top-5 证据构造 prompt。
   - 复用现有 verifier inference。
   - 与 fixed-MMR λ=0.7 和 oracle upper bound 比较。

V1 可以先完成 selection-only evaluation；只有 overlap 有信号，再跑 verifier evaluation。

输出:

```text
outputs/oracle_pointwise/v1/lightgbm/eval_val/selected_evidence.jsonl
outputs/oracle_pointwise/v1/lightgbm/eval_val/selection_metrics.json
outputs/oracle_pointwise/v1/lightgbm/eval_val/metrics_by_bucket.json
```

## 5. 执行命令草案

### 5.1 构造 V1a 数据

```bash
PYTHONPATH=src python scripts/selectors/build_pointwise_oracle_dataset.py \
  --oracle-results outputs/oracle_evidence/20260517_041502/oracle_results_train.jsonl \
  --config configs/experiment/b3_mmr_topk_sweep_1024.yaml \
  --split train \
  --output-dir outputs/oracle_pointwise/v1/data \
  --top-k 5 \
  --two-stage-multiplier 3 \
  --filter-preset v1a
```

### 5.2 训练轻量 pointwise selector

```bash
PYTHONPATH=src python scripts/selectors/train_pointwise_oracle_selector.py \
  --train-jsonl outputs/oracle_pointwise/v1/data/train_pointwise.jsonl \
  --output-dir outputs/oracle_pointwise/v1/lightgbm \
  --model lightgbm \
  --balance-labels true \
  --balance-candidates true
```

### 5.3 selection-only 验证

```bash
PYTHONPATH=src python scripts/selectors/eval_pointwise_oracle_selector.py \
  --model-dir outputs/oracle_pointwise/v1/lightgbm \
  --config configs/experiment/b3_mmr_topk_sweep_1024.yaml \
  --split val \
  --oracle-results outputs/oracle_evidence/20260516_135632/oracle_results_val.jsonl \
  --output-dir outputs/oracle_pointwise/v1/lightgbm/eval_val
```

### 5.4 verifier 验证

仅当 selection-only 通过 gate 后执行。可以复用现有 verifier 推理逻辑或新增 thin wrapper。

```bash
VERIFIER_MODEL=/data/models/Qwen2.5-7B-Instruct/ \
MODEL_BASE_PATH=/data/models/ \
LORA_ADAPTER=outputs/runs/b3_mmr_topk_sweep_1024/build.retrieval.mmr_lambda-0.7,build.retrieval.top_k-5__b23a0bbe/train/best \
PYTHONPATH=src python scripts/selectors/eval_pointwise_oracle_selector.py \
  --model-dir outputs/oracle_pointwise/v1/lightgbm \
  --config configs/experiment/b3_mmr_topk_sweep_1024.yaml \
  --split val \
  --oracle-results outputs/oracle_evidence/20260516_135632/oracle_results_val.jsonl \
  --output-dir outputs/oracle_pointwise/v1/lightgbm/eval_val_verifier \
  --run-verifier true
```

## 6. 成功标准与停止标准

### 6.1 数据构造 gate

必须满足:

1. `event_id` 对齐率 100%。
2. 复现候选池的 `n_candidates` 与 oracle JSONL 一致。
3. `selected_indices` 对应文本与 `selected_texts` 一致。
4. `filter_report.json` 包含每类过滤前后样本数。

任一失败都不能训练。

### 6.2 吸收性 gate

V1a 通过条件:

| 指标 | 最低要求 |
|---|---:|
| dev candidate AUPRC | 高于 positive-rate baseline 至少 2 倍 |
| dev mean Recall@5 | 高于 fixed hybrid top-5 |
| dev retained-label macro Jaccard@5 | 高于 fixed hybrid top-5 |
| false 类占比 | 不允许单类贡献超过 50% 的 batch 平均 loss |

如果只提升 `false`，但 `pants-fire / half-true / barely-true` 无提升，视为未通过。

### 6.3 Verifier gate

若跑 verifier evaluation，比较:

1. fixed-MMR λ=0.7。
2. pointwise selector。
3. oracle upper bound。

通过条件:

```text
retained-label macro-F1 > fixed-MMR retained-label macro-F1
且 true / mostly-true full-val accuracy 不显著低于 fixed-MMR
```

若 V1a 没有 true-side anchor，则 full-val 指标只作为风险监控，不作为 V1a 主成功指标。

## 7. 风险

### 7.1 没有 true 标签

V1a 主训练池确实排除了 `true / mostly-true`。这是为了验证 oracle supervision 的可吸收性，而不是训练最终 6 类系统。

处理:

1. 报告中明确 V1a 是 retained-bucket probe。
2. full-val 只做 sanity check。
3. V1a 通过后再做 V1b true-side anchor。

### 7.2 Pointwise 模型忽略 set interaction

Pointwise 模型无法显式建模 coverage、redundancy、stance diversity。

处理:

1. 把它作为最小可行 probe。
2. 若 pointwise overlap 有信号，再推进 sequential selector 或 multi-weight MMR。
3. 若 pointwise 无信号，不应直接上 DPO/GRPO。

### 7.3 Oracle set 是 verifier-relative label

Oracle set 最大化当前 LoRA verifier 的 gold-label logprob，不是人工 gold evidence。

处理:

1. 不把 Oracle set 作为永久标签。
2. 在 verifier calibration 后重新抽样 oracle。
3. 训练目标优先解释为 utility imitation，而不是 factual gold imitation。

## 8. 最终产物清单

代码:

```text
scripts/selectors/build_pointwise_oracle_dataset.py
scripts/selectors/train_pointwise_oracle_selector.py
scripts/selectors/eval_pointwise_oracle_selector.py
```

数据与模型:

```text
outputs/oracle_pointwise/v1/data/train_pointwise.jsonl
outputs/oracle_pointwise/v1/data/filter_report.json
outputs/oracle_pointwise/v1/data/feature_schema.json
outputs/oracle_pointwise/v1/lightgbm/model.pkl
outputs/oracle_pointwise/v1/lightgbm/training_metrics.json
outputs/oracle_pointwise/v1/lightgbm/eval_val/selection_metrics.json
```

报告:

```text
docs/analysis/oracle_pointwise_supervision_v1_results.md
```

## 9. 推进顺序

1. 实现 `build_pointwise_oracle_dataset.py`，先只输出数据和过滤报告。
2. 人工检查 20 条样本的候选池复现和 label。
3. 实现 Logistic Regression baseline。
4. 若 baseline 有信号，再接 LightGBM。
5. 做 selection-only val evaluation。
6. 若 overlap 指标通过 gate，再做 verifier evaluation。
7. 若 verifier retained buckets 有收益，再设计 V1b true-side anchor。
