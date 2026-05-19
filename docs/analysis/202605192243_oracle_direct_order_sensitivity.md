# Oracle Direct Verifier Evidence Order Sensitivity

更新时间：2026-05-19 22:43

## 背景

本实验用于验证一个窄问题：

> 在 oracle evidence set 不变的前提下，仅打乱 evidence 在 verifier prompt 中的顺序，会不会显著影响 oracle-direct verifier 的下游分类性能？

这不是 selector 实验，也不是重新训练 verifier。实验只做推理：

```text
Stage2 sentence-level oracle result
-> 固定 selected_indices 对应的 evidence set
-> 改变 selected evidence 的 prompt 顺序
-> 复用 oracle-direct verifier checkpoint 做 val inference
```

运行结果位于：

```text
outputs/runs/b3_oracle_direct_order_sensitivity
```

当前本地同步到的是 infer run 产物；`outputs/oracle_direct_verifier/stage2_sentence_order_sensitivity` 这类 build 中间目录未同步到当前工作区。但 wrapper 与构造脚本的代码口径可确认：实验固定使用 oracle-selected evidence set，只改变 evidence order。

## 2026-05-19 口径修正：0.7111 与 0.6099 不是同一 eval path

此前 `oracle sentence direct verifier` 记录的 val oracle evidence 指标是：

```text
train-time label-token eval, step-600 / best:
  logged eval n = 1280
  unique sample_idx n = 1274
  unique accuracy = 0.711146
  unique macro-F1 = 0.7169
```

本次 order-sensitivity 中的 `oracle_best` 指标是：

```text
pipeline API inference / vLLM guided_choice:
  n = 1274
  accuracy = 0.609890
  macro-F1 = 0.620744
```

这两个数不能直接当作同一口径下的下降。已确认 `best` checkpoint 与 `checkpoint-600` 的 adapter 文件一致，因此不是 checkpoint 选错。差异主要来自 label-token eval 与 pipeline API infer 的 prompt suffix 不一致。

训练 eval 的 `LabelTokenDataset` 会先执行：

```text
sample.prompt.rstrip() + "Label:"
```

而 pipeline API infer 当前执行：

```text
sample.prompt + "Label:"
```

当前 `build_val.jsonl` 中的 prompt 结尾是：

```text
...<|im_end|>
<|im_start|>assistant\n
```

因此两条路径实际给模型的 label 上下文不同：

```text
train-time label-token eval:
...<|im_start|>assistantLabel:

pipeline API inference:
...<|im_start|>assistant
Label:
```

对同一份 val oracle evidence、同一个 step-600/best checkpoint 的预测交叉检查：

| Comparison | n |
|---|---:|
| common `sample_idx` | 1274 |
| prediction disagreement | 388 |
| disagreement rate | 0.304553 |
| both correct | 696 |
| train eval only correct | 210 |
| API infer only correct | 81 |
| both wrong | 287 |

所以，`0.7111 -> 0.6099` 更准确的解释是 **train-time label-token eval 与 pipeline API infer 尚未完全对齐**，不是 oracle evidence set 自身变差。

本页后续 order-sensitivity 结论应按以下方式理解：

1. **相对顺序效应仍成立**：在同一个 API infer 口径下，oracle / hybrid / candidate_pool / random 只改变 evidence order，oracle order 明显更好。
2. **绝对 oracle upper-bound 暂不应从 0.7111 改写为 0.6099**：已在代码中对齐 label-token eval 与 API infer 的 prompt suffix；需要重跑 order-sensitivity 后再更新最终可比数。
3. **当前文档中的 order 表格是 API-infer 口径表格**，可用于判断 API 口径下顺序敏感，但不能替代此前 train-time eval 的 0.7111 upper-bound 结论。

相关代码：

| 文件 | 作用 |
|---|---|
| `scripts/verifier/run_oracle_direct_order_sensitivity.sh` | inference-only order sensitivity wrapper |
| `scripts/oracle_evidence/build_oracle_direct_verifier_data.py` | 从 `candidate_pool[selected_indices]` 构造 verifier-ready prompt，并按指定 order 排列 |

代码路径中的关键约束：

1. `selected_indices` 被解释为每条 oracle row 的 `candidate_pool` 坐标。
2. 先固定取出 `candidate_pool[selected_indices]`，再执行排序。
3. `order=hybrid` 按 `hybrid_score` 降序。
4. `order=candidate_pool` 按 `oracle_candidate_idx` 升序。
5. `order=random` 对每个 `event_id` 使用 `sha1(order_seed:event_id)` 派生随机种子打乱。
6. `expected_chunk_mmr_fingerprint=432dfc970e75` 不匹配会直接抛异常。

因此，在当前 API infer 口径内部，不同 order case 之间的相对指标变化可以归因于 evidence 顺序变化，而不是 evidence 集合变化。但 `oracle_best=0.6099` 与此前 `0.7111` 的绝对差异应归因于 eval/infer prompt suffix 口径不一致。

## 指标总表

所有 case 都是 val split，`n=1274`，预测文件无重复 `sample_idx`，`parse_error_rate=0`。

| Order case | Accuracy | Macro-F1 | Macro-P | Macro-R | vs oracle Acc | vs oracle Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 0.609890 | 0.620744 | 0.661009 | 0.608002 | +0.000000 | +0.000000 |
| hybrid | 0.453689 | 0.465368 | 0.499727 | 0.451126 | -0.156201 | -0.155376 |
| candidate_pool | 0.453689 | 0.465368 | 0.499727 | 0.451126 | -0.156201 | -0.155376 |
| random_seed0 | 0.469388 | 0.477067 | 0.509457 | 0.463833 | -0.140502 | -0.143678 |
| random_seed1 | 0.465463 | 0.473041 | 0.509745 | 0.458837 | -0.144427 | -0.147703 |
| random_seed2 | 0.474097 | 0.481870 | 0.512859 | 0.469208 | -0.135793 | -0.138875 |
| random_seed3 | 0.459969 | 0.469291 | 0.498080 | 0.456798 | -0.149922 | -0.151454 |
| random_seed4 | 0.468603 | 0.480254 | 0.513878 | 0.466544 | -0.141287 | -0.140491 |

随机顺序汇总：

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| Accuracy | 0.467504 | 0.004673 | 0.459969 | 0.474097 |
| Macro-F1 | 0.476304 | 0.004626 | 0.469291 | 0.481870 |

API infer 口径下，结论很直接：与 oracle greedy order 相比，hybrid / candidate_pool 顺序下降约 **15.6 pp accuracy** 和 **15.5 pp macro-F1**；随机顺序平均下降约 **14.2 pp accuracy** 和 **14.4 pp macro-F1**。这已经不是轻微噪声，而是明显的顺序敏感性。

## Paired Prediction 对比

以下表格以 `oracle` 顺序的预测为参照，统计同一批 1274 条样本在其他顺序下的预测翻转。

| Order case | Prediction disagreement | Disagreement rate | Both correct | Oracle only correct | Case only correct | Both wrong |
|---|---:|---:|---:|---:|---:|---:|
| candidate_pool | 470 | 0.368917 | 513 | 264 | 65 | 432 |
| hybrid | 470 | 0.368917 | 513 | 264 | 65 | 432 |
| random_seed0 | 482 | 0.378336 | 516 | 261 | 82 | 415 |
| random_seed1 | 483 | 0.379121 | 515 | 262 | 78 | 419 |
| random_seed2 | 469 | 0.368132 | 528 | 249 | 76 | 421 |
| random_seed3 | 471 | 0.369702 | 516 | 261 | 70 | 427 |
| random_seed4 | 469 | 0.368132 | 527 | 250 | 70 | 427 |

这里的关键信号是：

1. 非 oracle 顺序会让约 **36.8% - 37.9%** 的样本预测标签发生变化。
2. `oracle_only_correct` 明显大于 `case_only_correct`：例如 hybrid / candidate_pool 中，oracle 顺序独有正确 264 条，替代顺序独有正确只有 65 条。
3. 这说明顺序变化不只是随机改变预测，而是系统性破坏了当前 verifier 已学到的 oracle-order prompt 分布。

## Per-class F1

| Order case | pants-fire | false | barely-true | half-true | mostly-true | true |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 0.676056 | 0.655052 | 0.567986 | 0.497778 | 0.638950 | 0.688645 |
| hybrid | 0.537313 | 0.459649 | 0.408759 | 0.355649 | 0.538302 | 0.492537 |
| candidate_pool | 0.537313 | 0.459649 | 0.408759 | 0.355649 | 0.538302 | 0.492537 |
| random_seed0 | 0.507463 | 0.484642 | 0.428835 | 0.405010 | 0.518201 | 0.518248 |
| random_seed1 | 0.502513 | 0.463333 | 0.428305 | 0.422833 | 0.513800 | 0.507463 |
| random_seed2 | 0.517413 | 0.505300 | 0.458716 | 0.376518 | 0.509636 | 0.523636 |
| random_seed3 | 0.507463 | 0.469595 | 0.439394 | 0.383333 | 0.489270 | 0.526690 |
| random_seed4 | 0.544554 | 0.479310 | 0.441606 | 0.390144 | 0.507659 | 0.518248 |

各类标签基本都受影响，不是某一个类别单独拖累。`false`、`true`、`barely-true`、`half-true` 的掉幅尤其明显。

## 对当前主线结论的修正

此前我们把 pointwise full pipeline 低分主要归因于 selector 没有选到 oracle 分布附近。这个判断仍成立，但需要补充一个更强的约束：

> 对当前 oracle-direct verifier 来说，selector 不能只恢复 oracle evidence set 的无序集合，还需要恢复接近 oracle greedy order 的证据顺序。

也就是说，`Recall@5` / `Jaccard@5` 这类 selection-only set 指标即使较高，也可能不足以预测下游 verifier 表现。原因是 verifier 已经显著利用了 evidence 在 prompt 中的先后位置，尤其是 oracle search 贪心加入 evidence 的顺序。

这也解释了一个现象：按道理“同样的 5 条证据”应该包含同样的信息，但 Qwen verifier 并不是集合函数。Prompt 是序列输入，训练时一直看到 oracle greedy order，因此下游模型把位置、局部上下文和 evidence 间的渐进组合模式一起学进去了。把这些证据改成 hybrid/candidate/random 顺序，会制造 train-infer prompt 分布偏移。

## 对 selector 研究的影响

后续 selector 不能只被定义为 “从 top15 pool 中选出 top5 evidence”。更准确的任务应是：

```text
从 Stage2 sentence candidate_pool 中输出一个有序 evidence list：
selected evidence set + oracle-like order
```

因此 selector 评估至少应拆成三层：

1. **候选池覆盖**：oracle selected evidence 是否存在于 hybrid top15 candidate_pool。
2. **集合恢复**：selector top5 与 oracle selected set 的 Recall@5 / Jaccard@5。
3. **有序恢复**：在集合恢复基础上，进一步评估 rank correlation、ordered exact match、或 downstream verifier on selected evidence。

如果只看集合指标，可能会低估 evidence order 对 verifier 的真实影响。

## 建议

1. 当前 order sensitivity 结果应纳入 selector 构建前提：新 selector 输出必须是有序 list，不应在 build 阶段再按 candidate_pool index 或 hybrid score 重排。
2. pointwise / reranker 的监督目标应保留 `oracle_selected_rank`，训练时至少能区分第 1 条、第 2 条等顺序信息。
3. full pipeline 构造 evidence prompt 时应直接使用 selector 输出顺序；除非显式做排序消融，否则不要在后处理阶段重新排序。
4. 如果后续要提升 verifier 鲁棒性，可以尝试 order augmentation，但这会改变当前 oracle-direct verifier 的学习目标；短期更直接的路线是先让 selector 学 oracle-like ordering。
5. 之后汇报 selector 指标时，必须把 `set-level` 与 `order-sensitive downstream` 分开报告。

## 最终结论

已确认：在当前 `outputs/runs/b3_oracle_direct_order_sensitivity` 的 API inference 结果中，evidence 顺序对 oracle-direct verifier 性能有显著影响。

核心量级：

```text
oracle order:
  accuracy  = 0.609890
  macro-F1  = 0.620744

hybrid / candidate_pool order:
  accuracy  = 0.453689
  macro-F1  = 0.465368
  drop      = about -15.6 pp accuracy / -15.5 pp macro-F1

random order mean over seed0..4:
  accuracy  = 0.467504
  macro-F1  = 0.476304
  drop      = about -14.2 pp accuracy / -14.4 pp macro-F1
```

因此，后续方向不能把 oracle evidence set 简化成无序集合。当前 verifier 明显依赖 oracle greedy order；selector 需要同时学习“选哪些 evidence”和“以什么顺序给 verifier”。

同时需要补充：当前 `oracle_best=0.6099` 不能推翻此前 train-time label-token eval 的 `0.7111`。已将 pipeline API infer / vLLM infer / online vLLM eval 的 label prompt 拼接改为与 label-token eval 一致的 `sample.prompt.rstrip() + label_prefix` 口径；若要给出最终可比数，需要基于修正后的代码重跑该实验。
