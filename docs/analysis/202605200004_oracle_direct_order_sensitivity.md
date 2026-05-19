# Oracle Direct Verifier Evidence Order Sensitivity

更新时间：2026-05-20 00:04

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

## 2026-05-19 口径修正：0.7111 与 API gap 不是同一 eval path

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
2. **绝对 oracle upper-bound 暂不应从 0.7111 改写为 0.6099**：已在代码中对齐 label-token eval 与 API infer 的 prompt suffix；重跑后 oracle API 指标升至 0.6327，但仍不能替代 train-time label-token eval 的 0.7111。
3. **当前文档中的 order 表格是 API-infer 口径表格**，可用于判断 API 口径下顺序敏感，但不能替代此前 train-time eval 的 0.7111 upper-bound 结论。

## 2026-05-19 23:36 重跑后追加诊断

修正 prompt suffix 后，`outputs/runs/b3_oracle_direct_order_sensitivity` 已更新。`oracle_best` 从旧 API 口径的 `0.609890 / 0.620744` 提升到：

```text
oracle_best after prompt suffix alignment:
  n = 1274
  accuracy = 0.632653
  macro-F1 = 0.643008
```

但它仍低于 train-time label-token eval 的 `0.711146 / 0.7169`。进一步核查结果：

| Check | Result |
|---|---:|
| API prompt vs `build_val.jsonl` prompt | 1274 / 1274 identical |
| API target vs `build_val.jsonl` target | 1274 / 1274 identical |
| API gold label vs `build_val.jsonl` gold label | 1274 / 1274 identical |
| train unique prompt vs `build_val.jsonl` prompt | 1274 / 1274 identical |
| train unique target vs `build_val.jsonl` target | 1274 / 1274 identical |
| train unique gold label vs `build_val.jsonl` gold label | 1274 / 1274 identical |
| train eval vs API common sample_idx | 1274 |
| train/API prediction disagreement | 345 |
| disagreement rate | 0.270801 |
| both correct | 729 |
| train eval only correct | 177 |
| API infer only correct | 77 |
| both wrong | 291 |

还核查了 Qwen tokenizer 在 label boundary 上的分词。对当前样例：

```text
tokenizer(prompt.rstrip()) + tokenizer("Label:")
==
tokenizer(prompt.rstrip() + "Label:")
```

两者 token ids 完全一致；`" A"` ... `" F"` 也都是单 token。因此，剩余 gap 不是 prompt 文本、target/gold、样本顺序、候选证据或 tokenizer 边界问题。

当前最可能的问题是 **评估路径本身不等价**：

```text
train-time label-token eval:
  torch forward
  input_ids = prompt_ids + label_prefix_ids
  直接取下一 token 在 A-F label token 上的 logits argmax

pipeline API infer:
  vLLM OpenAI completions
  prompt text = prompt.rstrip() + label_prefix
  max_tokens=1
  guided_choice=[" A", ..., " F"]
  通过生成路径返回一个 constrained completion
```

前者是判别式 label scoring；后者是 constrained generation。即使 prompt 和 label tokens 已经对齐，二者也不应再被默认视为严格相同的评估口径。仓库里的 oracle scorer / learned-lambda 工具已经使用过更接近 label scoring 的做法：展开 A-F 六个 scoring prompt，用 vLLM `prompt_logprobs` 抽取各 label token 的 logprob，再取 argmax。后续如果要复现 `0.7111` 口径，应优先使用这种 label-token scoring eval，而不是继续用 `guided_choice` 生成结果作为严格 parity 指标。

2026-05-19 追加实现：pipeline API 推理默认已切换为 `infer.label_decoding.mode=prompt_logprobs`。当前默认行为是对每条样本展开 A-F 六个 scoring prompt，读取最后一个 prompt label token 的 `prompt_logprobs`，再对六个 label logprob 取 argmax。旧的 `guided_choice` 生成路径保留为显式 fallback，可通过 `infer.label_decoding.mode=guided_choice` 使用。

## 2026-05-20 prompt_logprobs 重跑结果

目标服务器上传了新一版 `oracle_best` 结果：

```text
outputs/runs/b3_oracle_direct_order_sensitivity/oracle_best/infer/val/best/393161cb0a90
```

该结果已经确认走到新 scoring path：

| Check | Result |
|---|---:|
| `label_decoding_mode=prompt_logprobs` | 1274 / 1274 |
| 每条 `label_logprobs` 包含 A-F 6 个分数 | 1274 / 1274 |
| parse error | 0 |

但指标仍未复现 train-time label-token eval：

| Eval path | n | accuracy | macro-F1 |
|---|---:|---:|---:|
| train-time label-token eval, unique sample_idx | 1274 | 0.711146 | 0.7169 |
| vLLM prompt_logprobs API, `393161cb0a90` | 1274 | 0.628728 | 0.638976 |
| previous vLLM guided-choice API, `7bc680045c2c` | 1274 | 0.632653 | 0.643008 |

与 train-time prediction 的逐样本对比：

| Comparison | n |
|---|---:|
| common sample_idx | 1274 |
| prediction disagreement | 346 |
| disagreement rate | 0.271586 |
| both correct | 725 |
| train eval only correct | 181 |
| vLLM prompt_logprobs only correct | 76 |
| both wrong | 292 |

补充核查：

1. API prediction 中的 prompt / target / gold label 与 `build_val.jsonl` 完全一致。
2. train-time unique prediction 中的 prompt / target / gold label 与 `build_val.jsonl` 完全一致。
3. 完整 scoring prompt 的分词也一致：

```text
tokenizer(prompt.rstrip()) + tokenizer("Label:") + tokenizer(" A")
==
tokenizer(prompt.rstrip() + "Label: A")
```

A-F 六个 label token 都满足该条件。

因此，当前 gap 已不能再归因于 prompt suffix、target/gold 错位、样本顺序、evidence order、或 tokenizer 拼接边界。更准确的结论是：

> vLLM prompt_logprobs 路径下加载/执行的模型 logits，与训练时 `label_token_trainer` 的 torch-forward logits 仍不一致。

剩余最可疑的两类原因：

1. **vLLM / merged-LoRA 路径与训练时 PEFT torch-forward 不等价**：包括 merged LoRA cache、dtype、vLLM kernel、或 server 复用旧模型。
2. **saved adapter 与训练时内存权重不完全等价**：train-time eval 用的是当时内存里的模型；API infer 用的是保存后的 adapter/merged model。需要单独用 HF/PEFT 从磁盘加载 `best` 或 `checkpoint-600`，跑一次 torch-forward label-token eval 来区分。

下一步最小判别实验：

```text
HF/PEFT torch-forward label-token eval on saved best/checkpoint-600
```

如果该结果接近 `0.7111`，说明 checkpoint 保存没问题，gap 在 vLLM merged inference。若该结果也只有 `~0.63`，说明此前 `0.7111` 是训练时内存模型评估结果，保存后的 checkpoint 没有完全复现该状态。

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

因此，在当前 API infer 口径内部，不同 order case 之间的相对指标变化可以归因于 evidence 顺序变化，而不是 evidence 集合变化。但 `oracle_best=0.6287` 与此前 `0.7111` 的绝对差异应归因于 train-time torch-forward label-token scoring 与 vLLM prompt_logprobs scoring 的模型执行路径仍不一致。

## 指标总表

所有 case 都是 val split，`n=1274`，预测文件无重复 `sample_idx`，`parse_error_rate=0`。

| Order case | Accuracy | Macro-F1 | Macro-P | Macro-R | vs oracle Acc | vs oracle Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 0.632653 | 0.643008 | 0.678933 | 0.627515 | +0.000000 | +0.000000 |
| hybrid | 0.463893 | 0.472120 | 0.508007 | 0.457475 | -0.168760 | -0.170889 |
| candidate_pool | 0.463893 | 0.472120 | 0.508007 | 0.457475 | -0.168760 | -0.170889 |
| random_seed0 | 0.478022 | 0.485622 | 0.518017 | 0.471499 | -0.154631 | -0.157386 |
| random_seed1 | 0.476452 | 0.483863 | 0.519181 | 0.469519 | -0.156201 | -0.159145 |
| random_seed2 | 0.482732 | 0.493705 | 0.526932 | 0.478129 | -0.149922 | -0.149303 |
| random_seed3 | 0.463893 | 0.472080 | 0.502093 | 0.458538 | -0.168760 | -0.170928 |
| random_seed4 | 0.468603 | 0.479239 | 0.507370 | 0.466417 | -0.164050 | -0.163769 |

随机顺序汇总：

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| Accuracy | 0.473940 | 0.006778 | 0.463893 | 0.482732 |
| Macro-F1 | 0.482902 | 0.007149 | 0.472080 | 0.493705 |

API infer 口径下，结论很直接：与 oracle greedy order 相比，hybrid / candidate_pool 顺序下降约 **16.9 pp accuracy** 和 **17.1 pp macro-F1**；随机顺序平均下降约 **15.9 pp accuracy** 和 **16.0 pp macro-F1**。这已经不是轻微噪声，而是明显的顺序敏感性。

## Paired Prediction 对比

以下表格以 `oracle` 顺序的预测为参照，统计同一批 1274 条样本在其他顺序下的预测翻转。

| Order case | Prediction disagreement | Disagreement rate | Both correct | Oracle only correct | Case only correct | Both wrong |
|---|---:|---:|---:|---:|---:|---:|
| candidate_pool | 482 | 0.378336 | 530 | 276 | 61 | 407 |
| hybrid | 482 | 0.378336 | 530 | 276 | 61 | 407 |
| random_seed0 | 488 | 0.383046 | 539 | 267 | 70 | 398 |
| random_seed1 | 482 | 0.378336 | 535 | 271 | 72 | 396 |
| random_seed2 | 494 | 0.387755 | 533 | 273 | 82 | 386 |
| random_seed3 | 492 | 0.386185 | 531 | 275 | 60 | 408 |
| random_seed4 | 487 | 0.382261 | 532 | 274 | 65 | 403 |

这里的关键信号是：

1. 非 oracle 顺序会让约 **37.8% - 38.8%** 的样本预测标签发生变化。
2. `oracle_only_correct` 明显大于 `case_only_correct`：例如 hybrid / candidate_pool 中，oracle 顺序独有正确 276 条，替代顺序独有正确只有 61 条。
3. 这说明顺序变化不只是随机改变预测，而是系统性破坏了当前 verifier 已学到的 oracle-order prompt 分布。

## Per-class F1

| Order case | pants-fire | false | barely-true | half-true | mostly-true | true |
|---|---:|---:|---:|---:|---:|---:|
| oracle | 0.682692 | 0.669027 | 0.592030 | 0.540434 | 0.670968 | 0.702899 |
| hybrid | 0.522613 | 0.469751 | 0.417495 | 0.377649 | 0.556660 | 0.488550 |
| candidate_pool | 0.522613 | 0.469751 | 0.417495 | 0.377649 | 0.556660 | 0.488550 |
| random_seed0 | 0.510000 | 0.487719 | 0.430020 | 0.436229 | 0.522293 | 0.527473 |
| random_seed1 | 0.510000 | 0.470588 | 0.454361 | 0.438662 | 0.512712 | 0.516854 |
| random_seed2 | 0.522613 | 0.488971 | 0.442191 | 0.422182 | 0.528302 | 0.557971 |
| random_seed3 | 0.500000 | 0.472028 | 0.454545 | 0.405157 | 0.486258 | 0.514493 |
| random_seed4 | 0.528846 | 0.480000 | 0.425703 | 0.413793 | 0.510730 | 0.516364 |

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
  accuracy  = 0.632653
  macro-F1  = 0.643008

hybrid / candidate_pool order:
  accuracy  = 0.463893
  macro-F1  = 0.472120
  drop      = about -16.9 pp accuracy / -17.1 pp macro-F1

random order mean over seed0..4:
  accuracy  = 0.473940
  macro-F1  = 0.482902
  drop      = about -15.9 pp accuracy / -16.0 pp macro-F1
```

因此，后续方向不能把 oracle evidence set 简化成无序集合。当前 verifier 明显依赖 oracle greedy order；selector 需要同时学习“选哪些 evidence”和“以什么顺序给 verifier”。

同时需要补充：当前 `oracle_best=0.6287` 不能推翻此前 train-time label-token eval 的 `0.7111`。已将 pipeline API infer / vLLM infer / online vLLM eval 的 label prompt 拼接改为与 label-token eval 一致的 `sample.prompt.rstrip() + label_prefix` 口径，并改用 vLLM `prompt_logprobs` 对 A-F label tokens 打分；重跑后剩余 gap 仍然存在，主因应是 vLLM/merged-LoRA scoring 与训练时 torch-forward scoring 不等价。若要给出最终可比数，需要先跑 saved checkpoint 的 HF/PEFT torch-forward parity eval。
