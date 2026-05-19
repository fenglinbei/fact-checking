# Pointwise Selector 数据契约修正与 Stage2 Sentence 试验

生成日期：2026-05-19

## 结论

本次修正把 V1 selection-only gate 降级为历史弱参考，并把 selector 数据构造统一到正式 pipeline 口径：

```text
dedup -> hybrid top candidate_pool_size -> selector topK
```

训练、selection-only eval、build pipeline 现在都要求共享同一个 Chunk-MMR fingerprint。cache、oracle result 或 pointwise model metadata 不一致时会直接抛异常。

## 代码修正

| 文件 | 修正 |
|---|---|
| `src/fact_checking/oracle_pointwise.py` | 新增 pipeline-style candidate pool、oracle candidate_pool fingerprint 校验、pointwise model metadata/fingerprint 校验 |
| `scripts/selectors/build_pointwise_oracle_dataset.py` | 默认改为 `pipeline_hybrid_topk`，从 oracle result 保存的 candidate_pool 构造监督 |
| `scripts/selectors/eval_pointwise_oracle_selector.py` | 默认改为 pipeline-style selection-only eval，并校验模型/cache fingerprint |
| `scripts/selectors/train_pointwise_oracle_selector.py` | 训练前要求单一 `chunk_mmr_fingerprint`，并写入 `metadata.json` / `model.npz` |
| `src/fact_checking/build/candidates.py` | build 阶段加载 pointwise model 时校验模型 fingerprint 与当前 build fingerprint |
| `scripts/oracle_evidence/search_optimal_evidence.py` | config 加载优先使用 Hydra compose，避免丢失 experiment defaults |
| `scripts/oracle_evidence/run_reoracle_stage2.sh` | 默认 config 改回 b3 semantic pipeline |

## Re-oracle 默认行为

`run_reoracle_stage2.sh` 当前默认：

```text
CONFIG=configs/experiment/b3_mmr_topk_sweep_1024.yaml
```

这会把后续 re-oracle evidence chunking 拉回 b3 semantic pipeline，预期 Chunk-MMR fingerprint 为：

```text
e0b01520364d
```

既有 `outputs/oracle_evidence/stage2_margin_train_sharded` 是修正前结果，保存的 fingerprint 为：

```text
432dfc970e75
```

因此它只能用于当前 sentence-cache pointwise 试验，不能和后续 semantic re-oracle 混用。

## 当前 Pointwise 试验

使用当前 re-oracle train 结果：

```text
outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl
```

使用候选池/cache：

```text
outputs/cache/chunk_mmr/432dfc970e75/train.pkl
configs/experiment/b3_pointwise_stage2_sentence_1024.yaml
```

构造结果：

```text
outputs/oracle_pointwise/stage2_margin_sentence/data/train_pointwise.jsonl
n_claims = 10065
n_rows = 142309
positive_text_match = 49733 / 49733 = 1.0
pool_mode = pipeline_hybrid_topk
chunk_mmr_fingerprint = 432dfc970e75
```

训练产物：

```text
outputs/oracle_pointwise/stage2_margin_sentence/logreg/model.npz
outputs/oracle_pointwise/stage2_margin_sentence/logreg/metadata.json
```

内部 dev selection-only：

| selector | Recall@5 | Jaccard@5 |
|---|---:|---:|
| pointwise | 0.3893 | 0.2685 |
| hybrid | 0.3587 | 0.2463 |

独立 val re-oracle selection-only：

| selector | Recall@5 | Jaccard@5 |
|---|---:|---:|
| pointwise | 0.3755 | 0.2536 |
| hybrid | 0.3435 | 0.2294 |

这个结果说明 pointwise 在当前 432 sentence-cache 候选池上学到了一部分 oracle 选择模式，但它仍只是 selection-only 信号。最终是否提升 verifier，需要在目标服务器运行完整 pipeline。

## 目标服务器 Pipeline 脚本

已新增：

```text
scripts/selectors/run_stage2_sentence_pointwise_full.sh
```

默认使用：

```text
experiment=b3_pointwise_stage2_sentence_1024
build.retrieval.pointwise_oracle.model_dir=outputs/oracle_pointwise/stage2_margin_sentence/logreg
pipeline.output_subdir=stage2_sentence_pointwise_full
```

运行：

```bash
bash scripts/selectors/run_stage2_sentence_pointwise_full.sh
```
