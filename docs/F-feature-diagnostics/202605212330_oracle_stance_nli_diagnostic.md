# Oracle Stance/NLI Diagnostic Implementation

## 目的

在正式把 stance-aware 特征接入 Step4 sequential selector 前，先统计 Stage2 oracle candidate pool 中 support / refute / neutral 信号是否与 oracle selection 有稳定关系。

本实现只做离线诊断与缓存，不改 selector 训练架构。

## 模型

默认模型：

```text
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

默认 NLI 方向：

```text
premise = candidate evidence text
hypothesis = claim
```

字段映射：

```text
support_score = P(entailment | evidence, claim)
neutral_score = P(neutral | evidence, claim)
refute_score  = P(contradiction | evidence, claim)
```

`qualify_proxy_score` 只是 `neutral_score * clipped_hybrid_score`，不是 NLI 模型的真实四分类标签。

## 新增脚本

```text
scripts/selectors/score_oracle_stance_nli.py
scripts/selectors/analyze_oracle_stance_distribution.py
```

打分脚本输出：

```text
candidate_stance_scores.jsonl
manifest.json
```

分析脚本输出：

```text
oracle_vs_pool_stance_distribution.json
stance_by_gold_label.json
stance_by_oracle_step.json
selected_vs_nonselected_probe.json
stance_set_patterns.json
analysis_summary.json
analysis.md
```

如果使用 `--num-shards > 1`，打分脚本会自动写成：

```text
candidate_stance_scores.shard_00000_of_00004.jsonl
manifest.shard_00000_of_00004.json
```

分析脚本的 `--stance-scores` 可以一次接收多个 shard 文件。

## 推荐先跑 val

```bash
PYTHONPATH=src python scripts/selectors/score_oracle_stance_nli.py \
  --oracle-results outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl \
  --split val \
  --model-name MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \
  --output-dir outputs/selectors/stance_nli/deberta_v3_base_mnli_fever_anli_val \
  --batch-size 64 \
  --max-length 384 \
  --device cuda \
  --resume
```

```bash
PYTHONPATH=src python scripts/selectors/analyze_oracle_stance_distribution.py \
  --stance-scores outputs/selectors/stance_nli/deberta_v3_base_mnli_fever_anli_val/candidate_stance_scores.jsonl \
  --output-dir outputs/selectors/stance_nli/deberta_v3_base_mnli_fever_anli_val/analysis
```

## Stop/Go

分析脚本默认阈值：

```text
support/refute selected-vs-pool lift >= 5pp
或 best stance feature separability AUROC >= 0.57
```

若输出 `decision=go_selector_ablation`，再实现 selector 侧的 `deep + stance_nli_scalar` ablation。

若输出 `decision=stop_or_calibrate_nli`，先不要训练 stance-aware selector，优先检查 NLI 模型质量、label id 映射、pair orientation 或换更强 NLI 模型复核。

## 2026-05-21 val 运行结论

运行产物：

```text
outputs/selectors/stance_nli/deberta_v3_base_mnli_fever_anli_val/
```

运行配置确认：

```text
split = val
model = MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
pair_orientation = evidence_claim
id2label = {0: entailment, 1: neutral, 2: contradiction}
n_examples = 1274
n_candidates = 18625
n_selected = 6332
chunk_mmr_fingerprint = 432dfc970e75
```

总体结论：当前 NLI stance 分数不应直接进入下一轮 selector ablation，决策为 `stop_or_calibrate_nli`。

核心证据：

1. Oracle selected 与 candidate pool 的 stance 分布几乎没有差异。pool 中 `support/refute` 总占比为 `20.80%`，oracle selected 为 `20.20%`，selected lift 为 `-0.60pp`，低于 `+5pp` 的 go 阈值。
2. selected-vs-nonselected 的 stance 单变量 separability 接近随机。最佳 stance feature 是 `support_score`，`separability_auc=0.5090`，低于 `0.57` 的 go 阈值。
3. NLI 模型整体高度偏向 neutral：pool neutral 为 `79.20%`，oracle selected neutral 为 `79.80%`。这说明当前句级 candidate 与 claim 的直接 NLI 关系多数被模型判为 insufficient/neutral，无法有效区分 oracle utility。
4. step0 单独看有弱信号，但不足以支撑训练。`support_score` 对 step0-vs-rest 的 AUROC 约 `0.5338`，`hybrid_score` 约 `0.5319`，二者同量级；steps 1-4 的 stance separability 进一步回落到约 `0.52` 以内。
5. label-specific 模式方向部分合理但幅度太小：`true` selected support 相比 pool 约 `+1.87pp`，`mostly-true` 约 `+1.27pp`，`pants-fire` selected refute 约 `+3.14pp`。这些现象可作为校准线索，但不是可直接训练的强特征。

研判：

当前结果更像是 NLI 模型对 LIAR-RAW 句级候选的 claim-evidence 关系不够敏感，而不是 selector 已经缺少一个可直接利用的 stance scalar。若把这版 `support_score/refute_score/neutral_score` 直接拼到 sequential selector，预期只会带来噪声或极弱 top1 变化，难以改善 `recall@5 / jaccard@5`。

下一步建议：

1. 暂缓 `deep + stance_nli_scalar` 训练，不把当前缓存作为主线 selector feature。
2. 如果继续 stance-aware，先做 NLI 校准诊断：抽样人工检查高 support / high refute / high neutral case，比较 `evidence_claim` 与 `claim_evidence` 方向，并用更强或更贴近 fact-checking 的 NLI/stance 模型复核。
3. 主线优先级应回到 claim/aspect coverage 或 h_claim 这类能改变 candidate 语义表示的信息源；stance-aware 只有在校准后达到 selected-vs-pool lift 或 AUROC 阈值时再接入 selector。
