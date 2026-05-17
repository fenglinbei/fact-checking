# Oracle Pointwise Supervision V1 实现与运行说明

生成日期: 2026-05-17

对应计划:

```text
docs/plan/202605171203_oracle_pointwise_supervision_v1.md
```

## 1. 实现范围

本轮已实现第一版轻量闭环:

1. 从 oracle evidence search 输出构造 filtered pointwise supervision rows。
2. 训练一个 NumPy logistic regression pointwise evidence utility model。
3. 在 val oracle set 上做 selection-only evaluation。

本轮没有跑 verifier re-inference，因此结果只说明模型是否能吸收 oracle-selected evidence 的排序模式，不说明最终 fact-checking accuracy / Macro-F1 已提升。

## 2. 新增代码

```text
src/fact_checking/oracle_pointwise.py
scripts/selectors/build_pointwise_oracle_dataset.py
scripts/selectors/train_pointwise_oracle_selector.py
scripts/selectors/eval_pointwise_oracle_selector.py
```

### 2.1 `src/fact_checking/oracle_pointwise.py`

共享工具模块，负责:

- 加载 oracle JSONL。
- 加载 build config 与 Chunk-MMR cache。
- 从 Chunk-MMR cache 计算 dense / lexical / BM25 / hybrid scores。
- 通过 oracle `selected_texts` 在当前候选池中匹配 positives。
- 用 hybrid rank 补 negatives，构造 pointwise candidate rows。
- 计算 row-level AUPRC / AUROC 与 claim-level Recall@K / Jaccard@K。
- 保存 `selected_evidence.jsonl`。

### 2.2 `build_pointwise_oracle_dataset.py`

入口:

```bash
PYTHONPATH=src python scripts/selectors/build_pointwise_oracle_dataset.py \
  --oracle-results outputs/oracle_evidence/20260517_041502/oracle_results_train.jsonl \
  --config configs/experiment/b3_mmr_topk_sweep_1024.yaml \
  --split train \
  --chunk-mmr-cache outputs/cache/chunk_mmr/57e1c87dcd33/train.pkl \
  --output-dir outputs/oracle_pointwise/v1/data \
  --top-k 5 \
  --filter-preset v1a \
  --fallback-pool-size 15
```

输出:

```text
outputs/oracle_pointwise/v1/data/train_pointwise.jsonl
outputs/oracle_pointwise/v1/data/filter_report.json
outputs/oracle_pointwise/v1/data/feature_schema.json
```

V1a 过滤条件:

```text
oracle_correct == true
gold_label in {pants-fire, false, barely-true, half-true}
final_logprob >= -0.5
n_candidates > 5
```

### 2.3 `train_pointwise_oracle_selector.py`

入口:

```bash
PYTHONPATH=src python scripts/selectors/train_pointwise_oracle_selector.py \
  --train-jsonl outputs/oracle_pointwise/v1/data/train_pointwise.jsonl \
  --feature-schema outputs/oracle_pointwise/v1/data/feature_schema.json \
  --output-dir outputs/oracle_pointwise/v1/logreg \
  --model logreg \
  --top-k 5 \
  --epochs 800 \
  --lr 0.05 \
  --patience 80
```

输出:

```text
outputs/oracle_pointwise/v1/logreg/model.npz
outputs/oracle_pointwise/v1/logreg/training_metrics.json
outputs/oracle_pointwise/v1/logreg/feature_importance.json
outputs/oracle_pointwise/v1/logreg/dev_predictions.jsonl
outputs/oracle_pointwise/v1/logreg/dev_selected_evidence.jsonl
```

训练细节:

- 按 `event_id` 分 train/dev，避免同一 claim 的 candidate 泄漏。
- 按 label 做分层切分。
- 使用 candidate-level weighted BCE。
- 权重包含两层均衡:
  - label-level inverse frequency。
  - claim-level positive/negative balance。
- 模型为标准化特征上的 logistic regression，纯 NumPy 实现，不依赖 sklearn/lightgbm。

### 2.4 `eval_pointwise_oracle_selector.py`

V1a retained bucket 评估:

```bash
PYTHONPATH=src python scripts/selectors/eval_pointwise_oracle_selector.py \
  --model-dir outputs/oracle_pointwise/v1/logreg \
  --oracle-results outputs/oracle_evidence/20260516_135632/oracle_results_val.jsonl \
  --config configs/experiment/b3_mmr_topk_sweep_1024.yaml \
  --split val \
  --chunk-mmr-cache outputs/cache/chunk_mmr/57e1c87dcd33/val.pkl \
  --output-dir outputs/oracle_pointwise/v1/logreg/eval_val \
  --filter-preset v1a \
  --top-k 5
```

Full-val oracle-overlap sanity check:

```bash
PYTHONPATH=src python scripts/selectors/eval_pointwise_oracle_selector.py \
  --model-dir outputs/oracle_pointwise/v1/logreg \
  --oracle-results outputs/oracle_evidence/20260516_135632/oracle_results_val.jsonl \
  --config configs/experiment/b3_mmr_topk_sweep_1024.yaml \
  --split val \
  --chunk-mmr-cache outputs/cache/chunk_mmr/57e1c87dcd33/val.pkl \
  --output-dir outputs/oracle_pointwise/v1/logreg/eval_val_all \
  --filter-preset all \
  --top-k 5
```

输出:

```text
outputs/oracle_pointwise/v1/logreg/eval_val/selection_metrics.json
outputs/oracle_pointwise/v1/logreg/eval_val/selected_evidence.jsonl
outputs/oracle_pointwise/v1/logreg/eval_val/candidate_scores.jsonl
```

## 3. 重要实现约束

### 3.1 当前候选池不是 oracle run 的严格原始候选池

`outputs/oracle_evidence/20260517_041502/oracle_results_train.jsonl` 没有保存完整 candidate pool，只保存了 oracle-selected texts / indices / search steps。

本地当前可用的 Chunk-MMR cache 为:

```text
outputs/cache/chunk_mmr/57e1c87dcd33/train.pkl
outputs/cache/chunk_mmr/57e1c87dcd33/val.pkl
```

它与 oracle run 写入 metrics 时记录的候选池统计不完全一致。因此本轮执行采用重建策略:

1. 在当前 Chunk-MMR cache 中按文本匹配 oracle `selected_texts`，作为 positives。
2. 以 oracle 记录中的 `n_candidates` 作为目标池大小。
3. 用当前 cache 中 hybrid score 最高的候选补 negatives。
4. 若 positive 数超过目标池大小，则保留所有 positives。

所以本轮结果是 `reconstructed-pool selection-only probe`，不是严格复现 oracle search 的原始 candidate pool。

### 3.2 如何判断重建质量

数据构造脚本会在 `filter_report.json` 中记录 positive text match rate。

本轮 train V1a:

```text
n_kept_claims = 3259
output_rows = 40470
positive_text_match = 16286 / 16295 = 99.94%
```

这说明 oracle positives 基本能在当前 cache 中找到；但 negatives 仍是重建的，不保证与原始 oracle candidate pool 完全一致。

## 4. 本轮运行结果

### 4.1 数据过滤结果

输出文件:

```text
outputs/oracle_pointwise/v1/data/filter_report.json
```

过滤后 label 分布:

| label | claims |
|---|---:|
| pants-fire | 534 |
| false | 1435 |
| barely-true | 453 |
| half-true | 837 |

总计:

```text
3259 claims
40470 candidate rows
positive rate ≈ 40.24%
```

### 4.2 Train/dev selection-only

输出文件:

```text
outputs/oracle_pointwise/v1/logreg/training_metrics.json
```

Dev rows / claims:

```text
n_val_rows = 4105
n_val_claims = 326
```

Candidate-level:

| metric | value |
|---|---:|
| AUPRC | 0.8971 |
| AUROC | 0.8908 |

Claim-level selection:

| scorer | Recall@5 | Jaccard@5 |
|---|---:|---:|
| pointwise logreg | 0.8288 | 0.7402 |
| hybrid_score baseline | 0.1926 | 0.1261 |

### 4.3 Val V1a retained bucket

输出文件:

```text
outputs/oracle_pointwise/v1/logreg/eval_val/selection_metrics.json
```

Val retained subset:

```text
n_claims = 508
n_rows = 6839
```

Candidate-level:

| scorer | AUPRC | AUROC |
|---|---:|---:|
| pointwise logreg | 0.9028 | 0.9016 |
| hybrid_score | 0.2345 | 0.1445 |

Claim-level:

| scorer | Recall@5 | Jaccard@5 |
|---|---:|---:|
| pointwise logreg | 0.8382 | 0.7502 |
| hybrid_score | 0.1496 | 0.0948 |

Per-label Jaccard@5:

| label | pointwise logreg | hybrid_score |
|---|---:|---:|
| pants-fire | 0.8034 | 0.0903 |
| false | 0.7208 | 0.0998 |
| barely-true | 0.8086 | 0.0861 |
| half-true | 0.7217 | 0.0951 |

### 4.4 Full-val oracle-overlap sanity check

输出文件:

```text
outputs/oracle_pointwise/v1/logreg/eval_val_all/selection_metrics.json
```

Full val:

```text
n_claims = 1274
n_rows = 15441
```

Claim-level:

| scorer | Recall@5 | Jaccard@5 |
|---|---:|---:|
| pointwise logreg | 0.8495 | 0.7706 |
| hybrid_score | 0.2622 | 0.2070 |

True-side oracle-overlap:

| label | pointwise Jaccard@5 | hybrid Jaccard@5 |
|---|---:|---:|
| mostly-true | 0.7271 | 0.1929 |
| true | 0.7507 | 0.2457 |

注意: 这里的 true-side overlap 只是说明模型能找回 oracle-selected texts，并不代表这些 true-side oracle sets 对 verifier 是好监督。此前 oracle 分析已经显示 true / mostly-true oracle sets 容易继承 verifier false bias。

## 5. 当前结论

V1a 的 selection-only gate 已通过:

1. 模型能明显区分 oracle-selected candidate。
2. retained labels 上 Jaccard@5 / Recall@5 明显高于 hybrid-score baseline。
3. 四个 retained labels 都有提升，不是只学到 `false` 类。

但该结论只在重建候选池和 oracle-overlap 层面成立。下一步不能直接宣称下游 verifier 提升。

## 6. 下一步

建议顺序:

1. 修改 oracle search 输出格式，保存完整 `candidate_pool`、`candidate_scores` 和候选池 fingerprint。
2. 用保存完整候选池的新 oracle run 重新构造 pointwise 数据，消除 reconstructed negatives 的不确定性。
3. 将 pointwise selector 输出接入 verifier evaluation，比较:
   - fixed-MMR λ=0.7
   - pointwise selector
   - oracle upper bound
4. 若 retained bucket 的 verifier metrics 仍有收益，再做 V1b true-side anchor。
5. 若 verifier evaluation 无收益，则优先检查:
   - pointwise top-K 是否破坏 evidence order。
   - oracle overlap 是否主要来自 verifier-biased artifacts。
   - true / mostly-true 是否被进一步拉向 false-side。
