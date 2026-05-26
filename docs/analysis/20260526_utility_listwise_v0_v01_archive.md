# Utility Listwise v0 / v0.1 Experiment Archive

日期：2026-05-26

## 结论

`utility_listwise` 的 v0 / v0.1 诊断线建议归档为 No-Go，不继续放大训练。

这两次实验回答的是一个很窄的问题：冻结 DeBERTa encoder，仅训练候选级 MLP / set head，能否从 step-0 VIG `delta_margin` 派生出可用的候选排序 scorer。当前答案是否定的。v0.1 通过关闭 rank/position prior 和训练时 candidate-order shuffle，确实消除了 v0 的极端位置坍塌，但 row-level utility ranking 仍接近随机，selector order metrics 仍低于 retrieval control。

因此，后续不建议继续做 frozen-head v0.x 调参。若继续 utility scorer 方向，应转向 verifier-aware / question-decomposition / evidence-structure 建模，而不是继续让小模型从原始 claim-candidate text 直接预测 `delta_margin`。

## 产物路径

| run | path | 状态 |
| --- | --- | --- |
| v0 | `outputs/selectors/utility_listwise/deberta_v0_step0_static` | full run / metrics available |
| v0.1 | `outputs/selectors/utility_listwise/deberta_v0_1_no_rank_shuffle` | synced best-checkpoint snapshot; no final `train_history.jsonl` / `val_history.jsonl` yet |

v0.1 目录含 `selection_metrics.json`、`val_trace.jsonl`、metadata/tokenizer/config。当前 `metadata.json` 显示 best checkpoint 为 step 1500 / epoch 2，总步数配置为 3777。虽然训练可能未完整落盘最终 history，但当前趋势已经足够稳定：row Spearman 与 pairwise acc 未出现可用增长。

## 配置对比

| item | v0 | v0.1 |
| --- | --- | --- |
| selector type | `utility_listwise_v0` | `utility_listwise_v0_1_no_rank_shuffle` |
| base model | `/data/models/deberta-v3-base/` | `/data/models/deberta-v3-base/` |
| train / val examples | 10065 / 1274 | 10065 / 1274 |
| pair encoder | frozen | frozen |
| target source | `vig_step0_delta_margin` | `vig_step0_delta_margin` |
| feature ablation | `none` | `no_rank_prior` |
| rank embedding | true | false |
| train candidate shuffle | none | 1.0 |
| loss weights | pairwise 1.0, soft CE 0.2, BCE 0.2 | pairwise 1.0, soft CE 0.2, BCE 0.2 |
| best metric | `jaccard@5` | `jaccard@5` |

v0.1 的目的不是追求最终最优，而是验证 v0 的失败是否主要来自 rank/index shortcut。结果显示：shortcut 可以被压掉，但 scorer 仍没有学到 utility。

## Main Metrics

同一 full-val 1274 claims 上：

| method | recall@5 | jaccard@5 | top1_match | oracle_rank_ndcg@5 | pairwise_order_acc@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| retrieval / candidate-pool control | 0.3435 | 0.2294 | 0.1028 | 0.2872 | 0.5271 |
| v0 best, step 200 | 0.3597 | 0.2410 | 0.0612 | 0.2585 | 0.4776 |
| v0.1 best, step 1500 | 0.3546 | 0.2379 | 0.0691 | 0.2534 | 0.4862 |
| `ridge_all_step0_static` | 0.5201 | 0.3747 | 0.6923 | 0.7130 | 0.7522 |
| `single_margin_step0_static` | 0.5224 | 0.3761 | 1.0000 | 0.8082 | 0.8473 |

解释：

- v0 / v0.1 都只比 retrieval control 有很小的 set-overlap 增益。
- v0 / v0.1 的 order metrics 均低于 retrieval control。
- saved-score static rankers 远高于 v0.x，说明 `delta_margin` 目标本身有信号，但 frozen text scorer 没有学到该信号。

## Row-Level Utility Metrics

| run | pairwise_accuracy | top1_delta_match | positive_hit@1 | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0 best | 0.5080 | 0.0612 | 0.3108 | 0.0250 | 0.0198 |
| v0.1 best | 0.5103 | 0.0691 | 0.2951 | 0.0398 | 0.0251 |

v0.1 history 中的最佳 row-level 观察值也很弱：

- best row pairwise accuracy: about 0.5122 at step 1100
- best row Spearman: about 0.0306 at step 1400
- best top1_delta_match: about 0.0848 at step 300

这些数值都不足以支持继续训练。一个可用 utility scorer 至少应先表现为 row-level pairwise acc 明显高于随机、Spearman 有稳定正相关；v0.x 没有达到。

## Failure Mode

v0 的主要失败形态是位置坍塌：

| v0 predicted top-1 candidate_idx | count |
| ---: | ---: |
| 12 | 994 |
| 11 | 232 |
| 3 | 35 |
| 0 | 11 |
| 1 | 2 |

v0.1 关闭 rank prior 并加 shuffle 后，top-1 分布变散：

| v0.1 predicted top-1 candidate_idx | count |
| ---: | ---: |
| 14 | 122 |
| 12 | 115 |
| 11 | 111 |
| 10 | 111 |
| 9 | 109 |
| 13 | 106 |

真实 best-delta candidate 在 full val 上本来也较分散：

| true best candidate_idx | count |
| ---: | ---: |
| 0 | 131 |
| 2 | 110 |
| 1 | 109 |
| 3 | 94 |
| 13 | 93 |

因此 v0.1 的价值是诊断性的：它证明 v0 的 `candidate_idx=12` collapse 不是唯一问题。去掉 shortcut 后，模型变成“没有学到明显 scorer 逻辑”，而不是“学到了正确 scorer”。

## Stop Rule

停止 v0.x 的理由：

1. row-level correlation 近似随机，且到 step 1500 仍没有打开。
2. set-overlap 增益太小，`jaccard@5` 仅比 control 高约 0.008-0.012。
3. order metrics 低于 retrieval control，而本任务已经确认 evidence ordering 是一等指标。
4. saved-score static rankers 证明 utility 信号存在，但 v0.x 模型形态无法抽取。
5. 继续训练 frozen encoder / set head 大概率只会在 control 附近震荡。

归档判定：

```text
decision = stop_frozen_text_utility_listwise_v0x
next = shift_to_question_decomposition_and_evidence_structure_or_verifier_aware_scorer
```

## 后续方向

不建议继续：

- 提高 v0.x epoch
- 微调 loss weights
- 继续只改 rank prior / shuffle
- 直接上 DAgger-lite bad-prefix utility labels

建议改为：

1. 设问式多路检索：先改变候选池和证据覆盖，而不是只重排当前 pool。
2. 圆桌式 evidence structure：对候选证据做 stance / cluster / source-diversity profiling，作为 selector 或 classifier 的结构化输入。
3. verifier-aware scorer：若仍做 scorer，应引入 verifier-derived teacher signal，而不是只从 claim-candidate text 预测 `delta_margin`。
4. 强模型结构化 teacher：用闭源模型生成 subquestions、evidence roles、cluster labels、短理由，再蒸馏给小模型；避免只学长解释文本。

## 相关文档

- `docs/analysis/20260526_utility_listwise_v0_report.md`
- `docs/analysis/20260526_selector_roundtable_question_teacher_ideas.md`
- `outputs/selectors/utility_listwise/deberta_v0_step0_static/selection_metrics.json`
- `outputs/selectors/utility_listwise/deberta_v0_1_no_rank_shuffle/selection_metrics.json`
- `outputs/selectors/vig_utility/saved_step_train_to_val/ranker_eval/selection_metrics.json`
