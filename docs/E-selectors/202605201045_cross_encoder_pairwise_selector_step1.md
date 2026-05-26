# Cross-encoder Pairwise Selector Step1 实施记录

生成日期: 2026-05-20

## 目标

根据 `docs/analysis/202605200216_selector_experiment_plan_and_literature_review.md` 的 Step 1，实现一个可训练、可 selection-only 评估、可接入 build pipeline 的 cross-encoder pairwise evidence selector。

该 selector 面向固定 Stage2 sentence-level candidate pool：

```text
chunk_mmr_fingerprint = 432dfc970e75
candidate pool        = saved Stage2 oracle candidate_pool, top15
selector output       = ordered top5
oracle objective      = margin
```

## 新增与修改文件

```text
src/fact_checking/selectors/stage2_oracle.py
src/fact_checking/selectors/metrics.py
src/fact_checking/selectors/cross_encoder.py
src/fact_checking/selectors/test_metrics.py
scripts/selectors/train_cross_encoder_pairwise.py
scripts/selectors/eval_cross_encoder_selector.py
scripts/selectors/run_cross_encoder_step1.sh
configs/experiment/b3_cross_encoder_stage2_sentence_1024.yaml
src/fact_checking/build/candidates.py
```

## 数据合同

`stage2_oracle.py` 负责读取并审计 `oracle_results_<split>.jsonl`。每条记录必须满足：

```text
candidate_pool exists
selected_indices exists
0 <= selected_indices[i] < len(candidate_pool)
candidate_pool_metadata.chunk_mmr_fingerprint == 432dfc970e75
search_objective == margin
len(candidate_pool) <= 15
len(selected_indices) <= 5
```

fingerprint mismatch 会 fail fast，不做 silent fallback。gold label、margin、oracle logprobs 只用于训练监督和过滤，不作为 selector 推理输入。

## 训练逻辑

入口：

```bash
PYTHONPATH=src python scripts/selectors/train_cross_encoder_pairwise.py \
  --model-name microsoft/deberta-v3-base \
  --train-oracle-results outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl \
  --val-oracle-results outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl \
  --output-dir outputs/selectors/stage2_sentence_cross_encoder/deberta_pairwise
```

默认输入格式为 tokenizer pair：

```text
Claim: <claim>
Evidence: <candidate_text>
```

loss 组成：

```text
pairwise logistic loss
+ 0.3 * selected-mask BCE
+ 0.5 * selected-order pairwise loss
```

负例使用 claim 内全部 non-selected candidates，并对 hybrid top5 / high-hybrid negatives 加权，覆盖 Step1 所需的 hard-negative 优先级。

训练输出：

```text
config.json / model.safetensors or pytorch_model.bin
tokenizer files
metadata.json
selection_metrics.json
val_trace.jsonl
```

## Selection-only 评估

入口：

```bash
PYTHONPATH=src python scripts/selectors/eval_cross_encoder_selector.py \
  --model-dir outputs/selectors/stage2_sentence_cross_encoder/deberta_pairwise \
  --oracle-results outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl \
  --output-dir outputs/selectors/stage2_sentence_cross_encoder/deberta_pairwise/eval_val
```

评估输出：

```text
selection_metrics.json
selection_trace.jsonl
control_hybrid_trace.jsonl
control_candidate_pool_trace.jsonl
```

trace 每条记录包含：

```text
event_id
claim
gold_label
candidate_pool
candidate_scores
oracle_ordered_indices
selector_ordered_indices
selector_scores
set_overlap
recall@5
precision@5
jaccard@5
top1_match
prefix_match@1 / @3 / @5
ordered_hit@5
oracle_rank_ndcg@5
pairwise_order_acc@5
overlap_pair_count
ordered_exact_match@5
fingerprint
```

同时内置 controls：

```text
hybrid_score top5
candidate_pool_order top5
same predicted set + hybrid-order
same predicted set + candidate-pool-order
same predicted set + random-order seeds 0-4
```

## 一键 wrapper

```bash
scripts/selectors/run_cross_encoder_step1.sh
```

常用覆盖：

```bash
MODEL_NAME=answerdotai/ModernBERT-base \
OUTPUT_DIR=outputs/selectors/stage2_sentence_cross_encoder/modernbert_pairwise \
BATCH_SIZE=8 \
EPOCHS=2 \
scripts/selectors/run_cross_encoder_step1.sh
```

## Pipeline 接入

新增 build selection method：

```yaml
build:
  retrieval:
    selection_method: cross_encoder_selector
    cross_encoder_selector:
      model_dir: outputs/selectors/stage2_sentence_cross_encoder/deberta_pairwise
      candidate_pool_size: 15
      max_length: 384
      batch_size: 32
      strict_fingerprint: true
      dump_trace: true
```

实验配置：

```text
configs/experiment/b3_cross_encoder_stage2_sentence_1024.yaml
```

只构建 evidence：

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b3_cross_encoder_stage2_sentence_1024 \
  pipeline.mode=build \
  pipeline.output_subdir=cross_encoder_step1_build
```

用 oracle-direct verifier 做 evaluation-only：

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b3_cross_encoder_stage2_sentence_1024 \
  'pipeline.steps=[build,infer]' \
  pipeline.output_subdir=cross_encoder_step1_oracle_direct_val \
  'train.run_dir="outputs/oracle_direct_verifier/stage2_sentence/train/b3_oracle_sentence_direct_verifier_1024_20260519-200709"' \
  infer.split=val \
  infer.checkpoint=best
```

build trace：

```text
outputs/cache/build/<build_id>/cross_encoder_selector_trace_<split>.jsonl
```

## Go / No-Go

先看 selection-only：

```text
recall@5 >= 0.50
jaccard@5 >= 0.35
oracle_rank_ndcg@5 > current pointwise
oracle_rank_ndcg@5 > hybrid-order control
top1_match > current pointwise
```

若 selection-only 不过关，不进入 full pipeline；优先检查 hard negative 权重、loss 权重、fingerprint audit、是否意外重排、候选池是否仍为 `432dfc970e75`。

## 2026-05-20 三模型运行结果

### 运行产物

三种 Step1 cross-encoder 都已完成训练与 val selection-only eval，结果目录为：

```text
outputs/selectors/stage2_sentence_cross_encoder/deberta_pairwise
outputs/selectors/stage2_sentence_cross_encoder/modernbert_pairwise
outputs/selectors/stage2_sentence_cross_encoder/bge_reranker_base_pairwise
```

统一评估口径：

```text
split                         = val
n_claims                      = 1274
filter_policy                 = all
chunk_mmr_fingerprint          = 432dfc970e75
candidate pool                 = saved Stage2 oracle candidate_pool top15
selector output                = ordered top5
metric source                  = <model_dir>/eval_val/selection_metrics.json
```

### Selection-only 指标

| 模型目录 | base model | recall@5 | jaccard@5 | oracle_rank_ndcg@5 | top1_match | Gate |
|---|---|---:|---:|---:|---:|---|
| `deberta_pairwise` | `microsoft/deberta-v3-base` | 0.3739 | 0.2522 | 0.2646 | 0.0667 | No-Go |
| `modernbert_pairwise` | `answerdotai/ModernBERT-base` | 0.3733 | 0.2536 | 0.2635 | 0.0659 | No-Go |
| `bge_reranker_base_pairwise` | `/data/models/bge-reranker-base` | 0.3736 | 0.2528 | 0.2593 | 0.0667 | No-Go |

对照项：

| 对照 | recall@5 | jaccard@5 | oracle_rank_ndcg@5 | top1_match | 说明 |
|---|---:|---:|---:|---:|---|
| `hybrid_score_top5` | 0.3435 | 0.2294 | 0.2872 | 0.1028 | 三个 cross-encoder eval 文件中的同一 control |
| current pointwise logreg | 0.3755 | 0.2536 | 0.2837 | 0.0950 | 由 `outputs/oracle_pointwise/stage2_margin_sentence/eval_val/selected_evidence.jsonl` 对齐 Stage2 val oracle 后补算 |

注：`modernbert_pairwise/selection_metrics.json` 中训练阶段保存的 best `jaccard@5=0.2546`，略高于单独 `eval_val` 的 `0.2536`，但仍远低于 `0.35` gate，且 NDCG / top1 也没有超过 pointwise 或 hybrid control，因此不影响 No-Go 判断。

### Gate 判定

Step1 预设 gate 与实际结果：

| Gate 条件 | 要求 | 最好结果 | 是否通过 |
|---|---:|---:|---|
| `recall@5 >= 0.50` | 0.5000 | 0.3739 | 否 |
| `jaccard@5 >= 0.35` | 0.3500 | 0.2536 | 否 |
| `oracle_rank_ndcg@5 > current pointwise` | > 0.2837 | 0.2646 | 否 |
| `oracle_rank_ndcg@5 > hybrid-order control` | > 0.2872 | 0.2646 | 否 |
| `top1_match > current pointwise` | > 0.0950 | 0.0667 | 否 |

### 结论

三种 cross-encoder pairwise selector 均未达到 Step1 selection-only gate，不建议进入 full pipeline 或 oracle-direct verifier evaluation-only。当前结果也不像是候选池或 fingerprint 错配导致的误判：三组结果都覆盖 `1274` 条 val claim，fingerprint 均为 `432dfc970e75`，hybrid control 一致。

更合理的 stop/go 判断是：停止 Step1 pointwise cross-encoder 作为主线强 baseline 的继续扩展；若继续推进 selector，应优先转向 Step3 set-aware / listwise 15-candidate reranker，或者先诊断 pairwise 训练目标是否过度压成“逐候选相关性”而没有学到 oracle greedy order。

## 实现验证与实验状态

实现阶段已完成代码级与轻量单测验证：

```text
compileall
unittest: src/fact_checking/selectors/test_metrics.py
Stage2 oracle sample audit
Hydra config compose
CLI --help
git diff --check
```

后续真实训练已按上述三个模型完成；当前验收结论以 `eval_val/selection_metrics.json` 为准。
