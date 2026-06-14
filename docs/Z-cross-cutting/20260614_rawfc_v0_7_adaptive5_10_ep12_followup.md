# RAWFC v0.7 adaptive5_10 LR=1e-5 ep12 复核

## 结论

ep12 运行本身完成正常，但没有带来可保留的性能提升。与同配置 ep8 相比，ep12 在最终 `eval/val/best/label_token` 上的 `selection_score`、`macro_f1`、`accuracy` 都略低；配对 bootstrap CI 覆盖 0，不能支持显著提升。训练后半段 `eval_ce_loss` 明显反弹，更像是过拟合/置信度变差，而不是 argmax 标签分布稳定变好。

## 运行状态

| run | completed | global_step | training best_score | latest_state |
|---|---:|---:|---:|---|
| ep8 | true | 808 | 1.018619 | 已清理 |
| ep12 | true | 1212 | 1.004961 | 已清理 |

ep12 日志显示 `num_epochs=12`、`max_train_steps=1212`，最后有 `removed completed-run latest state`，未发现 Traceback 或 early stop 异常。

## 训练期 val 曲线

| run | best step | selection_score | macro_f1 | accuracy | ordinal_mae | eval_ce_loss |
|---|---:|---:|---:|---:|---:|---:|
| ep8 | 800 | 1.018619 | 0.625854 | 0.625 | 0.460 | 1.250609 |
| ep12 | 1200 | 1.004961 | 0.611672 | 0.610 | 0.455 | 3.101403 |

ep12 的 CE 最低点在 step 200：`eval_ce_loss=0.945335`。之后整体抬升，step 800 为 `1.516097`，step 1000 为 `2.444496`，step 1100 为 `3.002892`，step 1200 为 `3.101403`。这说明后半段虽然 argmax 指标偶有回升，但概率质量已经明显偏离 gold label。

## 最终 best eval 对比

| run | selection_score | macro_f1 | accuracy | recomputed MAE | endpoint flip rate | pred false/half/true |
|---|---:|---:|---:|---:|---:|---|
| ep8 | 1.004914 | 0.616155 | 0.615 | 0.470 | 0.085 | 59 / 73 / 68 |
| ep12 | 0.996638 | 0.606547 | 0.605 | 0.460 | 0.065 | 64 / 71 / 65 |
| ep12 - ep8 | -0.008276 | -0.009608 | -0.010 | -0.010 | -0.020 | +5 / -2 / -3 |

配对检验输出：

- `outputs/sentence_trace_method/analysis/rawfc_v0_7_adaptive5_10_lr1e5_ep12_vs_ep8_val_paired_ci_trainer_equiv.json`
- delta 方向为 ep12 - ep8。
- `accuracy` delta `-0.0100`，95% CI `[-0.0550, 0.0300]`，McNemar p=`0.8238`。
- `macro_f1` delta `-0.0096`，95% CI `[-0.0534, 0.0326]`，approx-randomization p=`0.7061`。
- trainer-equivalent `selection_score` delta `-0.0083`，95% CI `[-0.0673, 0.0501]`，approx-randomization p=`0.7972`。

## 关于“标签极端化”

从最终 best 的 argmax 分布看，ep12 没有明显变得更极端：`false+true` 从 ep8 的 127/200 变为 ep12 的 129/200，只多 2 条；按端点翻转 `abs(pred_id-gold_id)==2` 复算，反而从 8.5% 降到 6.5%。因此“极端化”如果存在，更可能体现在 logit/概率置信度层面，而不是最终 argmax 标签分布层面。当前 prediction JSONL 不保存 logits，不能直接复算置信度分布；`eval_ce_loss` 的大幅反弹是这里最强证据。

另外，当前 `src/sft/metrics.py` 里的 `extreme_error_rate` 使用 `abs_diff >= 3`，对 3 类标签最大差值只有 2 的 RAWFC/LIAR 设置会恒为 0。因此三分类端点翻转不能直接用该字段判断，需要像本次这样从 prediction JSONL 复算 `abs(pred_id-gold_id)==2`。

## 建议

不建议继续把这个配置拉长 epoch。后续若还要围绕该 selector 调参，优先考虑更早停的训练上限或正则/学习率策略，而不是继续延长；当前最稳妥的报告口径仍是保留 ep8 结果，把 ep12 作为“延长训练未收益且 CE 反弹”的反证实验。
