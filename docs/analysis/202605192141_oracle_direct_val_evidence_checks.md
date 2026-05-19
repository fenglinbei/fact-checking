# Oracle Direct Verifier 非 Oracle Evidence 对照结果

文档更新时间：2026-05-19 21:41 CST

## 结论摘要

`outputs/runs/b3_oracle_direct_verifier_val_evidence_checks` 已完成，用同一个 oracle-direct verifier 分别评估：

```text
fixed-MMR sentence evidence + oracle-direct verifier
pointwise-selected sentence evidence + oracle-direct verifier
```

结果与先前判断一致：

1. oracle-direct verifier 只在 oracle evidence 条件下强；换成非 oracle evidence 后，val 指标回到 0.26-0.27 区间。
2. 当前 pointwise selector 没有缩小 gap，反而略低于 fixed-MMR sentence evidence。
3. best 与 checkpoint-600 指标完全一致，说明不是 checkpoint 选择问题。
4. parse error 为 0、val predictions 无重复、prompt truncation 为 0，说明低分不是解码、重复样本或 prompt 长度问题。

因此当前瓶颈仍然是 selector / evidence distribution gap：verifier 能吃 oracle evidence，但现有可部署 selector 没有把 evidence 选到 oracle 分布附近。

## 运行产物

Run group：

```text
outputs/runs/b3_oracle_direct_verifier_val_evidence_checks
```

四个 run：

| evidence | checkpoint | run dir | build_id |
|---|---|---|---|
| fixed-MMR sentence | best | `oracle_direct_val_fixed_mmr_sentence_best__cf7153a5` | `9b6a995af64a` |
| fixed-MMR sentence | checkpoint-600 | `oracle_direct_val_fixed_mmr_sentence_checkpoint-600__cf7153a5` | `9b6a995af64a` |
| pointwise sentence | best | `oracle_direct_val_pointwise_sentence_best__0516df13` | `0bb9343d53e7` |
| pointwise sentence | checkpoint-600 | `oracle_direct_val_pointwise_sentence_checkpoint-600__0516df13` | `0bb9343d53e7` |

两个 fixed-MMR run 复用同一份 build，两个 pointwise run 复用同一份 build。

## 主指标

| evidence | checkpoint | n | accuracy | macro-F1 | macro-P | macro-R | parse error |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed-MMR sentence | best | 1274 | 0.2716 | 0.2663 | 0.2770 | 0.2635 | 0.0000 |
| fixed-MMR sentence | checkpoint-600 | 1274 | 0.2716 | 0.2663 | 0.2770 | 0.2635 | 0.0000 |
| pointwise sentence | best | 1274 | 0.2637 | 0.2596 | 0.2699 | 0.2569 | 0.0000 |
| pointwise sentence | checkpoint-600 | 1274 | 0.2637 | 0.2596 | 0.2699 | 0.2569 | 0.0000 |

对照：

| 条件 | val accuracy | val macro-F1 | 解释 |
|---|---:|---:|---|
| oracle evidence + oracle-direct verifier | 0.7111 | 0.7169 | gold-conditioned upper-bound |
| Stage2 oracle under Stage1 verifier | 0.6593 | 0.6620 | oracle search 原始 verifier 口径 |
| fixed-MMR sentence + oracle-direct verifier | 0.2716 | 0.2663 | 本次非 oracle evidence 对照 |
| pointwise sentence + oracle-direct verifier | 0.2637 | 0.2596 | 本次非 oracle evidence 对照 |
| fixed-MMR sentence baseline verifier | 0.2951 | 0.2981 | 先前 val baseline |

直接差异：

```text
oracle evidence upper-bound - fixed-MMR evidence = +43.95 pp accuracy, +45.06 pp macro-F1
oracle evidence upper-bound - pointwise evidence = +44.74 pp accuracy, +45.73 pp macro-F1
pointwise evidence - fixed-MMR evidence = -0.78 pp accuracy, -0.66 pp macro-F1
```

这说明当前 pointwise selector 不仅没有接近 oracle evidence upper-bound，也没有稳定超过 fixed-MMR。

## Per-class F1

| evidence | checkpoint | pants-fire | false | barely-true | half-true | mostly-true | true |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed-MMR sentence | best | 0.2514 | 0.2993 | 0.2530 | 0.2136 | 0.3229 | 0.2575 |
| fixed-MMR sentence | checkpoint-600 | 0.2514 | 0.2993 | 0.2530 | 0.2136 | 0.3229 | 0.2575 |
| pointwise sentence | best | 0.2762 | 0.2912 | 0.2390 | 0.2558 | 0.2863 | 0.2094 |
| pointwise sentence | checkpoint-600 | 0.2762 | 0.2912 | 0.2390 | 0.2558 | 0.2863 | 0.2094 |

pointwise 对 `pants-fire` 和 `half-true` 略好，但损失了 `mostly-true` 和 `true`；宏平均仍低于 fixed-MMR。

## Prompt 与预测完整性检查

| evidence | split | n | prompt mean | prompt p95 | prompt max | truncation | mean evidence count |
|---|---|---:|---:|---:|---:|---:|---:|
| fixed-MMR sentence | train | 10065 | 412.61 | 548.00 | 995 | 0.0002 | 4.9406 |
| fixed-MMR sentence | val | 1274 | 411.66 | 542.05 | 958 | 0.0000 | 4.9702 |
| pointwise sentence | train | 10065 | 364.95 | 438.00 | 850 | 0.0000 | 4.9412 |
| pointwise sentence | val | 1274 | 364.16 | 433.00 | 956 | 0.0000 | 4.9702 |

Prediction 文件检查：

| evidence | checkpoint | prediction rows | unique sample_idx | duplicates |
|---|---|---:|---:|---:|
| fixed-MMR sentence | best | 1274 | 1274 | 0 |
| fixed-MMR sentence | checkpoint-600 | 1274 | 1274 | 0 |
| pointwise sentence | best | 1274 | 1274 | 0 |
| pointwise sentence | checkpoint-600 | 1274 | 1274 | 0 |

因此本次低分不能归因于 prompt overflow、truncation、重复样本或 label decoding。

## 与 Selection-only 指标的关系

当前 Stage2 sentence pointwise selector 的 val selection-only 指标：

```text
recall@5  = 0.3755
jaccard@5 = 0.2536
```

该 overlap 水平不足以支撑 oracle-direct verifier。即使 oracle-direct verifier 在 oracle evidence 上达到 0.7111 / 0.7169，只要 selector 只能找回约 37.6% 的 oracle evidence，最终 full pipeline 仍会退回 fixed-MMR 附近。

## 判断

这次对照把问题拆清楚了：

1. **不是 checkpoint 问题**：best 与 checkpoint-600 完全一致。
2. **不是 decoding 问题**：parse error 为 0。
3. **不是 eval duplicate 问题**：四个 prediction 文件都是 1274 unique rows。
4. **不是 prompt budget 问题**：val truncation 为 0。
5. **不是 verifier 完全无效**：oracle evidence 条件下 verifier 上界仍然很高。
6. **是 evidence distribution 问题**：fixed-MMR / pointwise evidence 与 oracle evidence 分布差距太大，oracle-direct verifier 无法在这些 evidence 上保持上界收益。

## 决策更新

| 方向 | 状态 | 原因 |
|---|---|---|
| oracle-direct verifier | 保留为 upper-bound probe | 只在 oracle evidence 条件下强，不应直接替代部署 verifier。 |
| current pointwise selector | 停止作为主线 | 非 oracle evidence eval 低于 fixed-MMR，selection overlap 不足。 |
| fixed-MMR sentence | 保留为部署 baseline | 虽远低于 oracle upper-bound，但仍强于当前 pointwise evidence。 |
| next selector | Go，需升级 | 必须改成 pairwise / listwise / reranker 或 sequential selector，并先过 selection-only gate。 |

## 下一步

1. 暂停继续放大当前 pointwise logreg。
2. 训练新的 selector 前，固定候选池口径：

```text
chunk_mmr_fingerprint = 432dfc970e75
candidate pool = dedup -> hybrid top15
selector output = top5
```

3. 用 Stage2 oracle `selected_indices` 构造 pairwise / listwise reranker 数据。
4. 新 selector 必须先在 val selection-only 上显著超过当前 pointwise：

```text
current recall@5  = 0.3755
current jaccard@5 = 0.2536
```

5. 只有 selection-only gate 通过后，才跑：

```text
learned selector evidence + oracle-direct verifier
```

否则完整 pipeline 大概率继续停留在 0.26-0.28 区间。
