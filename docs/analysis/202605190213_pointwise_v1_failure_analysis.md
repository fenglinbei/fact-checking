# Pointwise Selector V1 下游失败原因复盘

生成日期：2026-05-19

## 1. 结论摘要

`Pointwise selector V1` 的失败不是一个单纯的 verifier 顺序敏感问题，也不能简单解释为“Recall@5 / Jaccard@5 高但 verifier 不会用”。本次检查确认，V1 的 selection-only gate 本身存在明显的评估口径偏乐观：

1. **V1 selection-only 使用的显式 cache 是 `outputs/cache/chunk_mmr/57e1c87dcd33`，其候选基本是 sentence-level；当前 b3 semantic pipeline 对应的 cache 是 `outputs/cache/chunk_mmr/e0b01520364d`。两者候选集合几乎不是同一个空间。**
2. **V1 selection-only 的 `build_candidate_pool()` 会先把 oracle positives 强行放入评估候选池，再补 hybrid negatives；正式 build pipeline 不会这样做。** 因此记录中的 Recall@5 / Jaccard@5 衡量的是“在已注入 oracle positives 的重建候选池里能否排序”，不是“在真实 pipeline 候选池里能否找回 oracle evidence”。
3. **旧 oracle 输出没有保存完整 `candidate_pool` / `candidate_scores` / fingerprint，V1 只能靠文本或子串匹配重建正例。** 对 semantic chunk 来说，这会把一个长 chunk 匹配到其中一句 sentence fragment，进一步抬高 selection-only 指标。
4. 在更接近 pipeline 的 semantic cache + top15 候选池上复算后，V1 的 overlap 不再高于 fixed-MMR：full-val pipeline-style 下，pointwise Jaccard@5≈0.319，fixed-MMR/hybrid≈0.332；true-side 上 pointwise 更差。
5. 即便排除 gate 口径问题，V1 仍有建模和监督问题：旧 oracle 继承 false-side bias；V1a 排除了 `mostly-true/true`；pointwise logreg 学到的是偏向低 hybrid rank/低 relevance 的规则；长 semantic chunks 使 pointwise prompt 更容易被截断。

所以，**实现上确实存在与 cache / candidate pool / chunking 相关的漏洞，但主要漏洞在 selection-only gate 和训练数据构造口径，而不是 vLLM infer 阶段把证据直接喂错。** 下游 verifier 低于 fixed-MMR 是合理结果：V1 并没有在真实下游候选空间里稳定提供更好的 evidence。

## 1.1 2026-05-19 修正状态

本结论已落实到代码和运行规范：

1. V1 `oracle_n_top_hybrid_with_positives` selection-only gate 标记为历史诊断口径，不能作为强 go/no-go gate。
2. 新 selector 数据默认从 oracle result 保存的 `candidate_pool` / `candidate_scores` / `candidate_pool_metadata` 构造，不再靠 `selected_texts` 文本重建正例并注入候选池。
3. `build_pointwise_oracle_dataset.py`、`eval_pointwise_oracle_selector.py`、正式 `run_build()` 都会校验同一个 `chunk_mmr_fingerprint`；显式 cache 或模型元数据不匹配时直接抛异常。
4. selection-only eval 默认改为 pipeline-style pool：`dedup -> hybrid top candidate_pool_size -> selector topK`。
5. `run_reoracle_stage2.sh` 默认 config 已改回 `configs/experiment/b3_mmr_topk_sweep_1024.yaml`，后续 re-oracle 默认使用 b3 semantic chunk cache `e0b01520364d`。

当前已基于已有 train re-oracle 结果另做一个 sentence-cache 试验，目的是只回答“当前 432 候选池上的 oracle 选择模式能否被 pointwise 学到”，不把它与后续 semantic re-oracle 混为一谈。该试验使用 `configs/experiment/b3_pointwise_stage2_sentence_1024.yaml`，其 Chunk-MMR fingerprint 固定为 `432dfc970e75`。

## 2. 检查范围

本次检查使用现有代码和产物，没有重跑 vLLM：

- 时间线：`docs/analysis/202605190028_experiment_progress_timeline.md`
- V1 计划与实现：`docs/plan/202605171203_oracle_pointwise_supervision_v1.md`、`docs/implementation/202605171203_oracle_pointwise_supervision_v1.md`
- pipeline 接入：`docs/implementation/202605171430_pointwise_oracle_pipeline.md`
- oracle 输出契约修正：`docs/implementation/202605171322_oracle_search_output_contract.md`
- 代码：`src/fact_checking/oracle_pointwise.py`、`src/fact_checking/build/candidates.py`、`scripts/selectors/eval_pointwise_oracle_selector.py`
- 配置：`configs/experiment/b3_mmr_topk_sweep_1024.yaml`、`configs/experiment/b3_pointwise_oracle_selector_1024.yaml`
- 产物：`outputs/oracle_pointwise/v1/`、`outputs/oracle_evidence/20260516_135632/`、`outputs/oracle_evidence/20260517_041502/`、`outputs/runs/b3_pointwise_oracle_selector_1024/`、`outputs/runs/b3_mmr_topk_sweep_1024/`

## 3. 先确认下游失败是真实存在的

同一个旧 b3 verifier checkpoint 的 evaluation-only val 对比：

| selector | split | accuracy | macro-F1 |
|---|---:|---:|---:|
| fixed-MMR λ=0.7, top_k=5 | val | 0.2967 | 0.3003 |
| Pointwise V1a | val | 0.2582 | 0.2582 |
| Pointwise V1b | val | 0.2630 | 0.2632 |

完整 build→train→infer 的 test 对比：

| run | split | accuracy | macro-F1 |
|---|---:|---:|---:|
| fixed-MMR λ=0.7, top_k=5 | test | 0.2702 | 0.2769 |
| Pointwise V1a full | test | 0.2230 | 0.2059 |

val 上同一 verifier 的逐样本对比也显示，V1 不是小幅波动：

| bucket | 样本数 |
|---|---:|
| both correct | 201 |
| pointwise only correct | 128 |
| fixed only correct | 177 |
| both wrong | 768 |

Pointwise 主要把预测推向 false-side：val 上 `pants-fire/false` 预测占比从 fixed-MMR 的 32.65% 升到 43.56%，`mostly-true/true` 预测占比从 25.04% 降到 18.84%。`mostly-true` 上 fixed-only correct 有 42 条，而 pointwise-only correct 只有 12 条。

## 4. 关键实现路径

### 4.1 当前正式 pipeline 的配置是 semantic chunking

当前 Hydra 解析结果显示，`b3_pointwise_oracle_selector_1024` 和 fixed-MMR top_k=5 的差异主要是 `selection_method`：

| 项 | fixed-MMR | pointwise |
|---|---|---|
| `build.retrieval.chunking.strategy` | `semantic` | `semantic` |
| `theta` | 0.5 | 0.5 |
| `top_k` | 5 | 5 |
| `candidate_pool_multiplier` | — | 3 |
| `selection_method` | `mmr` | `pointwise_oracle` |
| 当前 chunk fp | `e0b01520364d` | `e0b01520364d` |

代码路径：

- fixed-MMR：`run_build()` → `_select_candidates_from_chunk_sample()` → `maximal_marginal_relevance()` → `_build_training_row()`
- pointwise：`run_build()` → `select_candidates_pointwise_oracle()` → `build_pointwise_inference_pool()` → `score_pointwise_features()` → `_build_training_row()`

因此正式 pipeline 本身并不是换成了 sentence chunking；问题出在 V1 训练/selection-only gate 使用的 cache 和候选池构造。

### 4.2 V1 selection-only 明确使用了旧显式 cache

`outputs/oracle_pointwise/v1/data/filter_report.json` 记录：

```text
cache_path = outputs/cache/chunk_mmr/57e1c87dcd33/train.pkl
pool_mode = oracle_n_top_hybrid_with_positives
positive_text_match = 16286 / 16295 = 99.94%
```

`outputs/oracle_pointwise/v1/logreg/eval_val_all/selection_metrics.json` 也记录 val 使用：

```text
cache_path = outputs/cache/chunk_mmr/57e1c87dcd33/val.pkl
```

但当前 b3 semantic 配置解析出的 chunk cache 是：

```text
outputs/cache/chunk_mmr/e0b01520364d/{train,val,test}.pkl
```

这不是一个无害的 fingerprint 差异；两个 cache 的候选形态完全不同。

## 5. cache / chunking 口径错配

本次直接读取两个 val cache 的候选统计：

| cache | rows | 候选数 median | 候选数 mean | `chunk_sent_indices` median | chunk words median | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `57e1c87dcd33` | 1274 | 64 | 70.18 | 1 | 28 | sentence-level |
| `e0b01520364d` | 1274 | 16 | 15.70 | 4 | 129 | semantic chunking |

两个 cache 按 event_id 对齐后，候选文本集合几乎不重合：

| 对比 | exact text-set match | mean Jaccard | median Jaccard |
|---|---:|---:|---:|
| `57e1...` vs `e0b...` | 9 / 1274 | 0.0495 | 0.0217 |

旧 oracle val metrics 的候选统计是：

```text
min=1, p25=9, median=16, p75=22, max=40, mean=15.70
```

这与 `e0b01520364d` 的 semantic cache 完全一致，而不是 `57e1c87dcd33` 的 sentence-level cache。也就是说：

> Oracle 上界和正式 pointwise pipeline 是 semantic chunk 空间；V1 selection-only 训练/评估却显式用了一个 sentence-level cache，并通过文本/子串匹配把 semantic oracle chunks 投影到 sentence fragments。

一个真实例子：

```text
event_id = 12134.json
oracle selected chunk: 100 words
57e matched positive: 15-word sentence fragment
e0b matched positive: 100-word semantic chunk
```

这会让 selection-only 指标回答一个错误问题：模型是否能找回某些 oracle chunk 内部的句子片段，而不是能否在 pipeline 实际消费的 semantic chunks 中选对证据。

## 6. candidate pool 构造进一步抬高了 Recall/Jaccard

`src/fact_checking/oracle_pointwise.py::build_candidate_pool()` 的训练/selection-only 逻辑是：

1. 在 cache 中匹配 oracle `selected_texts`，得到 positives。
2. `selected_source` 先放入 positives。
3. 再按 hybrid score 补 negatives 到 `oracle_n` 或 fallback pool size。
4. 最后在这个重建 pool 上计算 Recall@K / Jaccard@K。

这与正式 pipeline 不同。正式 `build_pointwise_inference_pool()` 只做：

```text
dedup -> hybrid_score desc -> top candidate_pool_size
```

不会预先注入 oracle positives。

因此 V1 selection-only gate 既有 cache 粒度错配，也有候选池正例注入口径。这个问题在数值上很明显。

### 6.1 原记录的 V1 selection-only 指标

记录在 `outputs/oracle_pointwise/v1/logreg/eval_val_all/selection_metrics.json`：

| 评估口径 | cache | Recall@5 | Jaccard@5 |
|---|---|---:|---:|
| V1 recorded full-val | `57e1...` + positives injected | 0.8495 | 0.7706 |
| hybrid baseline | `57e1...` + positives injected | 0.2622 | 0.2070 |

这个结果看起来非常强，但它不是正式下游的候选空间。

### 6.2 换成当前 semantic cache 后，指标大幅下降

用同一个 V1 logreg，在 `e0b01520364d` 上复算 selection-only：

| 评估口径 | split/filter | Recall@5 | Jaccard@5 |
|---|---|---:|---:|
| V1 model, semantic cache, positives injected | full val | 0.5812 | 0.4593 |
| V1 model, semantic cache, positives injected | retained V1a | 0.5563 | 0.4191 |

这已经不是“高到足以期待下游必然提升”的水平。

### 6.3 更接近正式 pipeline 的 top15 pool 后，Pointwise 不优于 fixed-MMR

本次另外按正式 pointwise inference 口径复算：

```text
semantic cache e0b
dedup -> hybrid top15
不注入 oracle positives
pointwise score top5
```

结果：

| bucket | selector | Recall@5 | Jaccard@5 |
|---|---|---:|---:|
| full val | Pointwise V1 | 0.4157 | 0.3192 |
| full val | fixed-MMR / hybrid top5 | 0.4330 | 0.3315 |
| retained labels | Pointwise V1 | 0.4253 | 0.3290 |
| retained labels | fixed-MMR / hybrid top5 | 0.4136 | 0.3167 |
| true-side | Pointwise V1 | 0.3962 | 0.2991 |
| true-side | fixed-MMR / hybrid top5 | 0.4724 | 0.3616 |

这解释了为什么下游 verifier 会低于 fixed-MMR：在真实 pipeline 候选池上，V1 并没有更稳定地找回 oracle evidence，尤其 true-side 明显更差。

### 6.4 oracle positives 在当前 top15 中并非总是可见

以当前 semantic cache `e0b...` 重新匹配旧 oracle selected texts：

```text
matched positives = 6147 / 6147
positives in current hybrid top15 = 5088 / 6147 = 82.77%
all positives visible in top15 claims = 740 / 1274 = 58.08%
```

也就是说，selection-only 的 positive-injected pool 隐藏了一个正式 pipeline 必须面对的问题：不少 oracle positives 并不在当前 top15 候选池内，或者不能完整进入同一个候选池。V1 gate 没有测这个失败模式。

## 7. 旧 oracle 输出契约本身也不足以支撑严格监督

V1 使用的旧 oracle 输出只保存：

```text
selected_texts / selected_indices / search_steps
```

没有保存完整：

```text
candidate_pool
candidate_scores
candidate_pool_fingerprint
candidate_pool_metadata
```

后续 `docs/implementation/202605171322_oracle_search_output_contract.md` 已经修正了这个问题，并明确 `selected_indices` 应该是 effective candidate pool 内的索引。这个修正本身说明 V1 使用的旧输出不适合做严格 selector supervision。

当前代码已经在 `scripts/oracle_evidence/search_optimal_evidence.py` 中重新计算 `dense/lexical/bm25/hybrid_score`，再按 hybrid 排序截断 two-stage pool。但 V1 训练数据来自旧 run，无法回溯当时每条样本的真实 candidate coordinate。因此 V1 的监督标签只能叫 reconstructed labels，不能叫权威 oracle candidate labels。

## 8. 即使修正 gate，V1 仍有建模问题

### 8.1 旧 oracle 继承 false-side bias

旧 oracle-set 分析已经显示，gold-logprob oracle 对 false-side labels 很强，但 true-side 很差：

| label | old oracle val recall/acc |
|---|---:|
| pants-fire | 88.7% |
| false | 89.6% |
| mostly-true | 21.9% |
| true | 1.2% |

V1a 又主动排除了 `mostly-true / true` 训练样本：

```text
labels_after:
half-true 837
false 1435
barely-true 453
pants-fire 534
```

因此 V1a 学到的是 retained false-side/mid-side oracle imitation，不是 full 6-class selector。V1b 加入 anchor 后也只有：

```text
mostly-true_anchor = 470, weight = 117.5
true_anchor = 37, weight = 3.7
```

`true` anchor 的有效权重太小，无法抵消主训练池的偏置。

### 8.2 logreg 学到的是“低 hybrid / 后排候选更像 oracle”

V1 feature weights：

| feature | weight |
|---|---:|
| `rank_norm` | +0.8128 |
| `hybrid_score` | -0.6197 |
| `rank_by_hybrid` | +0.5742 |
| `same_report_count` | -0.5348 |

这说明模型强烈倾向选择 hybrid 排名靠后的候选。这符合 oracle search 可能选多样/反直觉证据的表象，但在没有 set interaction 和 margin calibration 时，会把低相关 chunk 直接塞给 verifier。

在 semantic val cache 上，Pointwise 与 fixed-MMR 的选中证据统计：

| 指标 | Pointwise V1 | fixed-MMR λ=0.7 |
|---|---:|---:|
| selected mean hybrid | 0.3682 | 0.6962 |
| selected min hybrid | 0.2187 | 0.5263 |
| mean hybrid rank | 9.18 | 1.98 |
| max hybrid rank | 11.12 | 4.11 |
| mean words per evidence | 160.57 | 140.63 |

Pointwise 牺牲了大量 relevance，只换来很有限的去冗余收益。对 verifier 来说，这会表现为证据更长、更散、更不直接。

### 8.3 prompt 截断放大了长 chunk 和排序问题

从 val predictions 的 prompt 反推 evidence 数量：

| selector | mean evidence count | `<5` evidence 的样本数 | 5 条完整 evidence 的样本数 |
|---|---:|---:|---:|
| Pointwise V1 eval-only | 3.71 | 783 / 1274 | 491 |
| fixed-MMR λ=0.7 | 4.10 | 648 / 1274 | 626 |

`_auto_truncate_evidence()` 会从 tail 移除 evidence。Pointwise 选出的 semantic chunks 更长，且按 pointwise score 排序；一旦超出 1024 token budget，后排 evidence 会被删掉。即使 pointwise top5 中包含有用 evidence，也可能因为排序和 chunk 长度无法完整进入 prompt。

## 9. 对用户疑问的逐项回答

### 是否用错了 cache？

**是，至少 selection-only gate 用错了或用了过期/不一致的 cache。**

V1 记录的 gate 使用 `57e1c87dcd33`，实际是 sentence-level；当前 b3 semantic pipeline 解析出的 cache 是 `e0b01520364d`。旧 oracle metrics 的候选统计也匹配 `e0b...`，不匹配 `57e...`。因此 V1 的高 Recall/Jaccard 不能代表下游 pointwise build。

### 是否给错了证据候选？

**selection-only 评估候选池给法有问题。**

`build_candidate_pool()` 先注入 oracle positives，再补 negatives；正式 pipeline 只从 hybrid top15 pool 中打分选择。这个差异足以把 full-val Jaccard 从 pipeline-style 的约 0.319 抬到记录中的 0.771。

### 是否 evidence chunking 策略选错？

**V1 selection-only 实际落在了 sentence fragment 口径，而下游和 fixed-MMR 是 semantic chunk 口径。**

这不是“semantic chunking 本身错”，而是训练/评估 selector 的候选单位没有和正式 pipeline 对齐。若要评估 semantic pipeline，selector 数据集也必须来自同一个 semantic candidate pool，并用 `candidate_uid/source_index/fingerprint` 对齐。

### 是否只是 evidence order 导致掉点？

**不是主因，但是放大因素。**

当前 fixed-MMR 实现最终也会按 `hybrid_score` 降序呈现 selected candidates；pointwise 则按 pointwise score 呈现。更大的问题是 pointwise score 倾向低 hybrid 候选，且 chunks 更长，导致 prompt 截断更多。order 影响主要体现在 tail evidence 被删掉，而不是唯一根因。

## 10. 建议修正与下一步

1. **把 V1 selection-only gate 标记为无效或仅作弱参考。** 它不能支撑“Pointwise selector 已经能高 Recall/Jaccard 找回 oracle set”的结论。
2. **后续 selector 数据必须只从带完整 candidate pool 的 oracle 输出构造。** 要求每条 oracle result 有 `candidate_pool`、`candidate_scores`、`candidate_pool_fingerprint`、`selected_indices_coordinate`；禁止用跨 cache 子串匹配作为主监督。
3. **训练、selection-only eval、build pipeline 必须共享同一个 chunk cache fingerprint。** selector model metadata 中应保存 cache fingerprint、chunking config、candidate_pool_version；build 加载时不一致直接 warning 或 fail。
4. **selection-only eval 应改成 pipeline-style pool。** 即 `dedup -> hybrid top candidate_pool_size -> selector topK`，不得预先注入 positives；并且 baseline 应同时报告 fixed-MMR λ=0.7，而不只是 hybrid_score topK。
5. **不要继续用旧 gold-logprob oracle 训练主 selector。** 应等待 Stage 2 margin re-oracle，用 calibration-aware objective 和完整 candidate pool 重建监督。
6. **若仍做 pointwise，需要 relevance floor 或 MMR fallback。** 例如限制 selected mean hybrid 不低于 fixed-MMR 的某个比例，或只允许 pointwise 在 fixed topM 内重排，避免低 relevance chunk 大量进入 prompt。
7. **优先考虑 sequential / preference selector。** Pointwise independent model 无法显式建模 coverage、互补性、冗余和 prompt budget；这些正是 oracle-set selection 与 verifier 下游效果之间的关键差异。

## 11. 最终判断

`2026-05-17：Pointwise selector V1` 的失败应重新表述为：

> V1 在 reconstructed / positive-injected / cache-mismatched selection-only gate 上通过，但这个 gate 没有代表正式 semantic build pipeline。进入真实下游候选空间后，V1 对 oracle evidence 的找回并不稳定优于 fixed-MMR，且因为旧 oracle false-side bias、true-side 训练缺失、低 relevance 倾向和 prompt 截断，最终 verifier 指标低于 fixed-MMR。

因此，继续在 V1/V1b 上微调权重意义不大。正确路线是使用带完整候选池的新 re-oracle 结果重建监督，并先把 selection-only gate 修成与正式 build 完全一致的评估口径。
