# RAWFC LoRA Selector/LR Matrix Analysis (2026-06-13)

## 实验范围

本轮只分析 RAWFC 矩阵的已落盘结果：Llama3.1-8B，LoRA r16/alpha32/dropout0.05，EBS16，selector = old adaptive5_10 / v0.7 adaptive3_10 / v0.7 adaptive5_10 / v0.7 adaptive5_12，LR = 1e-5 / 5e-6。

所有 8 个训练组合均有 `train/training_complete.json`，并且 val/best 的 `label_token`、`tau0`、`tau0p5`、`tau0p75` metrics/predictions 都齐全。当前矩阵没有 test eval 产物，因此本报告只做 val selection；test 仍应只在最终确认候选上补跑。

## 口径说明

- 排名主指标使用已存 `metrics.json` 里的 `selection_score`。
- 当前 RAWFC `train.resolved.yaml` 的 `early_stopping_metric` 是 `macro_f1_plus_true_side_plus_mae`，`true_side_metric_weight=0.5`，`mae_metric_weight=0.3`。
- 对 RAWFC 3 类标签来说，现有 `_true_side_macro_f1` 仍沿用 LIAR 的 `(mostly-true, true)` 口径；由于 RAWFC 没有 `mostly-true`，这里实际等价于 `0.5 * true_class_f1`。因此 `selection_score` 不是纯 `macro_f1`，它也奖励 true 类 F1 和较低 ordinal MAE。
- bootstrap CI 为本报告从 `val_predictions.jsonl` 以 5,000 次样本重采样计算；paired CI 为 10,000 次 paired bootstrap。CI 仅用于不确定性判断，不替代 test confirmation。

## 去重后的 val 结果

`tau0/tau0p5/tau0p75` 与 plain 的预测完全一致，因此下表只保留每个 selector/LR 的 plain 行。

| Rank | Selector | LR | selection_score | macro_f1 | accuracy | false F1 | half F1 | true F1 | ordinal MAE | selection 95% CI |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | v0.7_adaptive5_12 | 1e-5 | 1.0080 | 0.6121 | 0.610 | 0.634 | 0.545 | 0.657 | 0.455 | [0.9121, 1.1044] |
| 2 | v0.7_adaptive5_10 | 1e-5 | 1.0049 | 0.6162 | 0.615 | 0.640 | 0.571 | 0.637 | 0.470 | [0.9050, 1.0991] |
| 3 | v0.7_adaptive3_10 | 1e-5 | 1.0035 | 0.6119 | 0.610 | 0.650 | 0.543 | 0.642 | 0.460 | [0.8997, 1.0982] |
| 4 | old_adaptive5_10 | 5e-6 | 0.9917 | 0.6059 | 0.605 | 0.677 | 0.519 | 0.622 | 0.465 | [0.8889, 1.0868] |
| 5 | old_adaptive5_10 | 1e-5 | 0.9866 | 0.5965 | 0.595 | 0.640 | 0.507 | 0.642 | 0.470 | [0.8843, 1.0833] |
| 6 | v0.7_adaptive5_12 | 5e-6 | 0.9617 | 0.5843 | 0.585 | 0.630 | 0.508 | 0.615 | 0.510 | [0.8614, 1.0567] |
| 7 | v0.7_adaptive5_10 | 5e-6 | 0.9573 | 0.5857 | 0.585 | 0.635 | 0.526 | 0.596 | 0.515 | [0.8567, 1.0555] |
| 8 | v0.7_adaptive3_10 | 5e-6 | 0.9556 | 0.5753 | 0.575 | 0.613 | 0.489 | 0.624 | 0.505 | [0.8540, 1.0492] |

## Paired Bootstrap 对比

点估计最高的 checkpoint 是 `v0.7_adaptive5_12 + LR=1e-5`，但它与第二、第三名的差距分别只有 0.0031 和 0.0046，均小于预设 tie 阈值 0.005；paired CI 也跨 0，不能当作硬胜出。

| Comparison | Delta selection | Paired 95% CI | Delta macro_f1 | Paired 95% CI |
|---|---:|---|---:|---|
| best - v0.7_adaptive5_10 1e-5 | 0.0031 | [-0.0304, 0.0371] | -0.0040 | [-0.0301, 0.0213] |
| best - v0.7_adaptive3_10 1e-5 | 0.0046 | [-0.0530, 0.0609] | 0.0002 | [-0.0408, 0.0410] |
| best - old_adaptive5_10 5e-6 | 0.0163 | [-0.0644, 0.1014] | 0.0062 | [-0.0498, 0.0652] |
| best - old_adaptive5_10 1e-5 | 0.0214 | [-0.0555, 0.0994] | 0.0156 | [-0.0396, 0.0710] |
| best - v0.7_adaptive5_12 5e-6 | 0.0464 | [-0.0052, 0.1003] | 0.0278 | [-0.0093, 0.0668] |
| best - v0.7_adaptive5_10 5e-6 | 0.0507 | [-0.0023, 0.1077] | 0.0264 | [-0.0097, 0.0653] |
| best - v0.7_adaptive3_10 5e-6 | 0.0525 | [-0.0090, 0.1146] | 0.0368 | [-0.0077, 0.0821] |

## 稳定性与解释

1. `tau` 扫描在这组 RAWFC val 上没有改变预测：plain、tau0、tau0.5、tau0.75 的 metrics 完全相同。后续 RAWFC 调参不应把 tau 当成主要差异来源。
2. v0.7 系列在 LR=1e-5 下整体优于 old adaptive5_10；top3 都是 v0.7 + LR=1e-5。
3. v0.7 系列对 LR 明显敏感：同一 selector 下，LR=5e-6 比 1e-5 低约 0.046-0.052 selection_score。old adaptive5_10 则是 5e-6 略高于 1e-5，但 old 的整体点估计仍低于 v0.7 top3。
4. v0.7 adaptive5_10 与 v0.7 adaptive5_12 在 RAWFC 的 train/val/test prompt 完全一致；两者 evidence_count 统计也一致。因此二者的 val 差异不应解释为 max_k=12 带来真实 selector 改进，更像同数据下训练随机性/非确定性的小幅波动。
5. v0.7 adaptive3_10 使用更短证据预算：val 平均 evidence_count 约 3.21，adaptive5_x 约 5.04，old 约 8.91。它的 selection_score 只比点估计第一低 0.0046，若强调 prompt 成本/证据简洁性，它是很强的备选。

## 建议选择

- 严格按 val 点估计，最高 checkpoint 是 `v0.7_adaptive5_12 + LR=1e-5`，selection_score = 1.0080。
- 按预设 tie-break，不把 <0.005 当硬胜出：推荐把 `v0.7_adaptive5_10 + LR=1e-5` 作为 selector 配方层面的主候选，因为它与 5_12 的输入完全相同、macro_f1/accuracy 反而略高，且避免把 5_12 的小幅 selection 优势误读成证据预算收益。
- 若最终只允许补一个 test confirmation：建议测 `v0.7_adaptive5_10 + LR=1e-5`。若允许补两个，则同时测 `v0.7_adaptive5_12 + LR=1e-5`，用于确认同 prompt、不同训练轨迹的 val 波动是否会延续到 test。

## 输出来源

- 汇总 CSV: `outputs/sentence_trace_method/analysis/rawfc_lora_selector_lr_matrix_val_summary.csv`
- 稳定性 CSV: `outputs/sentence_trace_method/analysis/rawfc_lora_selector_lr_matrix_stability.csv`
- selector 稳定性 CSV: `outputs/sentence_trace_method/analysis/rawfc_lora_selector_lr_matrix_selector_stability.csv`
- paired CI JSON: `outputs/sentence_trace_method/analysis/rawfc_lora_selector_lr_matrix_distinct_paired_ci.json`
