# Oracle Direct Verifier 结果确认与下一步计划

文档更新时间：2026-05-19 21:13 CST

命名约定：从本文开始，新增或实质更新后重新落盘的分析文档，文件名前缀使用文档更新时间 `YYYYMMDDHHMM`，而不是实验启动时间或首次创建时间。

## 结论摘要

`outputs/oracle_direct_verifier/stage2_sentence` 是强阳性结果：sentence-level oracle evidence supervision 可以被 label-token CE verifier 吸收，并在 oracle evidence 条件下显著提升 verifier 表现。

但该结果必须解释为 gold-conditioned diagnostic / upper-bound，不是可部署 pipeline 指标。该实验直接把 oracle search 选出的 evidence set 喂给 verifier，没有训练或应用 selector。

当前最重要的结论是：

1. verifier 不是主要瓶颈。它能利用 sentence-level oracle evidence。
2. pointwise full pipeline 低分更像 selector gap，而不是 evidence 顺序或 verifier 无法吸收 evidence。
3. 下一阶段应优先把 selector 从当前 pointwise logreg 升级到 pairwise / listwise reranker，并先过 selection-only gate，再跑完整 pipeline。

## 产物与训练来源

主产物：

```text
outputs/oracle_direct_verifier/stage2_sentence
```

训练目录：

```text
outputs/oracle_direct_verifier/stage2_sentence/train/b3_oracle_sentence_direct_verifier_1024_20260519-200709
```

输入 oracle：

```text
train: outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl
val:   outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl
```

训练方式：

```text
base model = /data/models/Qwen2.5-7B-Instruct
method     = LoRA, r=16, alpha=32, dropout=0.05
objective  = label-token weighted CE
max_length = 1024
```

因此这版不是从 Stage1 verifier checkpoint 继续训，而是从原始 Qwen2.5-7B-Instruct 底座重新挂 LoRA 训练。

## 数据构造检查

`build_report.json` 显示构造侧没有明显异常。

| split | rows | skipped | chunk fingerprint | oracle correct rate | truncation rate | mean evidence count |
|---|---:|---:|---|---:|---:|---:|
| train | 10065 | 0 | `432dfc970e75` | 0.6188 | 0.0013 | 4.9389 |
| val | 1274 | 0 | `432dfc970e75` | 0.6593 | 0.0008 | 4.9678 |

候选与 evidence 语义：

```text
chunking.strategy = sentence
candidate pool    = dedup -> hybrid top15
oracle set        = greedy margin top5
order             = oracle greedy order
```

这与当前主线的 sentence-level Stage2 oracle 决策一致。

## 训练结果确认

训练曲线稳定上升，best checkpoint 为 step 600。

| step | accuracy | macro-F1 | true-side macro-F1 | selection score | eval loss |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.4289 | 0.4365 | 0.4563 | 0.6647 | 1.4268 |
| 100 | 0.5406 | 0.5410 | 0.6028 | 0.8424 | 1.1986 |
| 200 | 0.6359 | 0.6365 | 0.6609 | 0.9670 | 0.9487 |
| 350 | 0.6953 | 0.7009 | 0.7503 | 1.0760 | 0.8114 |
| 500 | 0.7039 | 0.7099 | 0.7358 | 1.0778 | 0.7904 |
| **600** | **0.7125** | **0.7183** | **0.7507** | **1.0937** | **0.7864** |

`val_predictions.jsonl` 中有 1280 行 prediction，而 `build_val.jsonl` 为 1274 行。重复的是前 6 个 `sample_idx`，预测一致，推测来自 eval padding / distributed gathering。按唯一 `sample_idx` 去重后：

| 口径 | n | accuracy | macro-F1 |
|---|---:|---:|---:|
| logged metrics | 1280 | 0.7125 | 0.7183 |
| unique sample_idx | 1274 | 0.7111 | 0.7169 |

去重前后差异很小，不改变结论；但后续应修正 eval metric，避免 padding 样本进入正式指标。

## 与 Stage2 Oracle 的配对关系

在同一份 val oracle evidence 上，Stage1 verifier 的 oracle correct rate 为 0.6593；oracle-direct verifier 去重后为 0.7111。

按 `event_id` 配对：

| bucket | count | 含义 |
|---|---:|---|
| both_correct | 752 | Stage1 oracle 与 direct verifier 都预测正确 |
| direct_only | 154 | Stage1 oracle 错，但 direct verifier 正确 |
| oracle_only | 88 | Stage1 oracle 正确，但 direct verifier 错 |
| both_wrong | 280 | 两者都错 |

direct verifier 相比 Stage1 oracle 多净修正 66 条 val 样本，accuracy 提升约 +5.18 pp。

## 与现有 Pipeline 的关系

当前固定 MMR / pointwise pipeline 的代表性指标：

| 实验 | split | accuracy | macro-F1 | 备注 |
|---|---|---:|---:|---|
| fixed-MMR sentence baseline | val | 0.2951 | 0.2981 | `top_k=6`, vLLM infer |
| fixed-MMR sentence baseline | test | 0.2686 | 0.2746 | `top_k=6`, vLLM infer |
| stage2 sentence pointwise full | test | 0.2614 | 0.2515 | 当前 pointwise selector full pipeline |
| oracle sentence direct verifier | val oracle evidence | 0.7111 | 0.7169 | gold-conditioned diagnostic |

这个对比不能直接说明可部署系统已经达到 0.71；它说明只要 evidence 接近 oracle set，verifier 有足够能力吸收监督并做出强判别。

当前 pointwise selector 的 selection-only val 指标仍较弱：

```text
recall@5  = 0.3755
jaccard@5 = 0.2536
```

因此 pointwise full pipeline 低分的主因更可能是 selector 没有把 evidence 选到 oracle 分布附近，而不是 verifier 或 evidence order 本身。

## 当前 Stop / Go

| 方向 | 状态 | 判断 |
|---|---|---|
| sentence-level Stage2 oracle | Go / 主线 | paired 对比中显著强于 semantic-level oracle。 |
| oracle sentence direct verifier | Go / 诊断上界已确认 | val oracle evidence 上 accuracy 0.7111、macro-F1 0.7169。 |
| 当前 pointwise logreg selector | Stop / 弱 baseline | selection overlap 不足，full pipeline 低于 fixed-MMR。 |
| semantic-level oracle | Diagnostic only | semantic paired subset 明显低于 sentence-level。 |
| fixed-MMR | Baseline | 仍是必须保留的部署基线。 |

## 下一步执行方案

1. 修正 eval metric 去重：对 `val_predictions.jsonl` / distributed gather 的输出按唯一 `sample_idx` 去重后再算正式 eval metrics。该修正优先级高，因为它会影响所有后续 checkpoint 选择和报告口径。

2. 做 oracle-direct verifier 的非 oracle evidence 对照：用 step-600 / best checkpoint 分别在 fixed-MMR sentence evidence 和当前 pointwise-selected evidence 上跑 val。目标是拆分两件事：verifier 是否泛化到普通 evidence 分布，以及 selector gap 对最终指标的实际伤害。

3. 建立 selector 升级路线。固定候选池口径为：

```text
dedup -> hybrid top15 -> selector top5
chunk_mmr_fingerprint = 432dfc970e75
```

用 Stage2 oracle 的 `selected_indices` 构造 pairwise / listwise reranker 数据。正例为 oracle-selected candidate，负例为同一 claim 内未选 candidate。

4. 先跑 selection-only gate，再跑完整 pipeline：当前 pointwise selection-only 的 recall@5 只有 0.3755，Jaccard@5 只有 0.2536。新 selector 至少应显著高于这个口径，才值得烧完整 verifier pipeline。

5. 通过 gate 后再组合 full pipeline。推荐组合：

```text
sentence candidate pool top15
-> learned reranker top5
-> oracle-direct-trained verifier
-> val full pipeline
-> test only after val beats fixed-MMR
```

6. 保留 oracle direct verifier 作为 upper-bound probe。后续任何 selector 的最终解释都应同时报告：

```text
fixed-MMR verifier result
selector + oracle-direct verifier result
oracle evidence + oracle-direct verifier upper-bound
```

这样可以区分 selector gap、verifier gap 与 oracle upper-bound gap。
