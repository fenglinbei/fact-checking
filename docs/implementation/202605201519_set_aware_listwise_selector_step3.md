# Set-aware Listwise Selector Step3 实施记录

生成日期: 2026-05-20 15:19 CST

## 目标

根据 `docs/analysis/202605200216_selector_experiment_plan_and_literature_review.md` 的 Step 3，实现一个可训练、可 selection-only 评估、可接入 build pipeline 的 set-aware / listwise evidence selector。

Step1 cross-encoder pairwise 三模型已 No-Go，因此 Step3 不再只做逐候选独立打分，而是把同一 claim 下的 top15 candidate pool 作为一个整体建模：

```text
chunk_mmr_fingerprint = 432dfc970e75
chunking.strategy     = sentence
candidate pool        = saved Stage2 oracle candidate_pool top15
selector output       = ordered top5
oracle objective      = margin
```

## 新增与修改文件

```text
src/fact_checking/selectors/listwise.py
src/fact_checking/selectors/test_listwise.py
scripts/selectors/train_listwise_selector.py
scripts/selectors/eval_listwise_selector.py
scripts/selectors/run_listwise_step3.sh
configs/experiment/b3_listwise_stage2_sentence_1024.yaml
src/fact_checking/build/candidates.py
```

## 模型结构

实现采用计划中的 set-aware listwise 路线：

```text
for each candidate i:
    h_i = AutoModel(Cross-encoder pair input).pooled_or_mean_embedding
    numeric_i = retrieval / rank / text-overlap features
    x_i = projection([h_i, numeric_i]) + rank_embedding(hybrid_rank_i)

H = TransformerEncoder(x_1, ..., x_N, attention_mask)
score_i = MLP(H_i)
ordered_top5 = argsort(score_i, descending=True)[:5]
```

Pair 输入仍使用：

```text
Claim: <claim>
Evidence: <candidate_text>
```

数值特征为部署时可得特征，不使用 gold label、oracle margin 或 oracle logprobs：

```text
hybrid_score
dense_score
lexical_score
bm25_log_norm
hybrid_rank_norm
candidate_idx_norm
sent_idx_norm
source_index_norm
text_token_len_norm
claim_token_overlap
number_overlap
```

其中 `hybrid_rank` 同时进入 rank embedding。训练时可通过 `--shuffle-probability` 做 candidate permutation augmentation；默认 `0.0`，便于先跑 no-shuffle 主线，再单独做 shuffle ablation。

## Loss

训练 loss 由三部分组成：

```text
0.3 * selected-mask BCE
+ 1.0 * Plackett-Luce / ListMLE order loss
+ 0.5 * selected-order pairwise loss
```

ListMLE 按 Stage2 oracle `selected_indices` 的 greedy order 计算：

```text
L_PL = - sum_t log exp(score[pi_t]) / sum_{j in remaining_t} exp(score[j])
```

`selected-mask BCE` 负责把 oracle selected set 与 non-selected candidates 分开；`selected-order pairwise` 约束 oracle selected 内部相对顺序。

## 训练入口

```bash
PYTHONPATH=src python scripts/selectors/train_listwise_selector.py \
  --model-name microsoft/deberta-v3-base \
  --train-oracle-results outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl \
  --val-oracle-results outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl \
  --output-dir outputs/selectors/stage2_sentence_listwise/deberta_listwise
```

常用参数：

```text
--batch-size 2
--learning-rate 2e-5
--head-learning-rate 1e-4
--list-hidden-size 256
--list-layers 2
--list-heads 4
--shuffle-probability 0.0
--freeze-pair-encoder
```

训练输出：

```text
encoder config / weights
tokenizer files
listwise_head.pt
metadata.json
selection_metrics.json
val_trace.jsonl
```

## Selection-only 评估

```bash
PYTHONPATH=src python scripts/selectors/eval_listwise_selector.py \
  --model-dir outputs/selectors/stage2_sentence_listwise/deberta_listwise \
  --oracle-results outputs/oracle_evidence/stage2_margin_val_20260518_111721/oracle_results_val.jsonl \
  --output-dir outputs/selectors/stage2_sentence_listwise/deberta_listwise/eval_val
```

评估复用 Step1 的 ordered gate 与 controls：

```text
selection_metrics.json
selection_trace.jsonl
control_hybrid_trace.jsonl
control_candidate_pool_trace.jsonl
```

核心 gate 仍为：

```text
recall@5 >= 0.50
jaccard@5 >= 0.35
oracle_rank_ndcg@5 > current pointwise
oracle_rank_ndcg@5 > hybrid-order control
top1_match > current pointwise
```

若 selection-only 不过 gate，不进入 full pipeline。

## 一键 wrapper

```bash
scripts/selectors/run_listwise_step3.sh
```

示例：

```bash
MODEL_NAME=microsoft/deberta-v3-base \
OUTPUT_DIR=outputs/selectors/stage2_sentence_listwise/deberta_listwise \
BATCH_SIZE=2 \
EPOCHS=2 \
scripts/selectors/run_listwise_step3.sh
```

4 卡 DDP 训练：

```bash
NPROC_PER_NODE=4 \
MODEL_NAME=microsoft/deberta-v3-base \
OUTPUT_DIR=outputs/selectors/stage2_sentence_listwise/deberta_listwise \
BATCH_SIZE=2 \
EPOCHS=2 \
scripts/selectors/run_listwise_step3.sh
```

`BATCH_SIZE` 是每张 GPU 的 micro-batch。4 卡时有效 batch 约为：

```text
effective_batch = BATCH_SIZE * NPROC_PER_NODE * gradient_accumulation_steps
```

训练脚本会按 rank 对 train examples 做 padding 后均匀切分，只有 rank0 执行 val selection-only eval 与模型保存。

shuffle augmentation ablation：

```bash
SHUFFLE_PROBABILITY=0.15 \
OUTPUT_DIR=outputs/selectors/stage2_sentence_listwise/deberta_listwise_shuffle015 \
scripts/selectors/run_listwise_step3.sh
```

## Pipeline 接入

新增 build selection method：

```yaml
build:
  retrieval:
    top_k: 5
    selection_method: listwise_selector
    listwise_selector:
      model_dir: outputs/selectors/stage2_sentence_listwise/deberta_listwise
      candidate_pool_size: 15
      max_length: 384
      batch_size: 8
      strict_fingerprint: true
      dump_trace: true
```

实验配置：

```text
configs/experiment/b3_listwise_stage2_sentence_1024.yaml
```

只构建 evidence：

```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b3_listwise_stage2_sentence_1024 \
  pipeline.mode=build \
  pipeline.output_subdir=listwise_step3_build
```

build trace：

```text
outputs/cache/build/<build_id>/listwise_selector_trace_<split>.jsonl
```

## 实现验证

已完成轻量验证：

```text
PYTHONPATH=src python -m compileall -q src/fact_checking/selectors scripts/selectors/train_listwise_selector.py scripts/selectors/eval_listwise_selector.py src/fact_checking/build/candidates.py
PYTHONPATH=src python -m unittest src/fact_checking/selectors/test_metrics.py src/fact_checking/selectors/test_listwise.py
PYTHONPATH=src python scripts/selectors/train_listwise_selector.py --help
PYTHONPATH=src python scripts/selectors/eval_listwise_selector.py --help
bash -n scripts/selectors/run_listwise_step3.sh
```

## 2026-05-20 诊断运行结果

### 运行产物

当前已完成三组 Step3 listwise 诊断：

```text
outputs/selectors/stage2_sentence_listwise/deberta_listwise
outputs/selectors/stage2_sentence_listwise/deberta_listwise_shuffle03
outputs/selectors/stage2_sentence_listwise/deberta_listwise_margin_positive
```

统一候选池口径：

```text
chunk_mmr_fingerprint = 432dfc970e75
candidate pool        = saved Stage2 oracle candidate_pool top15
selector output       = ordered top5
base model            = /data/models/deberta-v3-base/
metric source          = <run_dir>/eval_val/selection_metrics.json
```

`deberta_listwise_margin_positive` 的 metadata 显示：

```text
filter_policy       = margin_positive
shuffle_probability = 0.3
n_val               = 840
```

因此它不是纯 filter 诊断，而是 `margin_positive + shuffle03` 的组合诊断；其 val 指标只覆盖 margin-positive 子集，不能直接与全量 val 的 1274 条样本等量比较。

### 全量 val 指标

| Run | filter | shuffle | n | recall@5 | jaccard@5 | top1_match | oracle_rank_ndcg@5 | pairwise_order_acc@5 | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `deberta_listwise` | all | 0.0 | 1274 | 0.3689 | 0.2484 | 0.1162 | 0.3072 | 0.5145 | No-Go |
| `deberta_listwise_shuffle03` | all | 0.3 | 1274 | 0.3732 | 0.2518 | 0.1279 | 0.3131 | 0.5372 | No-Go |
| `hybrid_score_top5` control | all | - | 1274 | 0.3435 | 0.2294 | 0.1028 | 0.2872 | 0.5271 | - |

`shuffle03` 相比 no-shuffle 有小幅正向作用：

```text
recall@5           +0.0043
jaccard@5          +0.0034
top1_match         +0.0117
oracle_rank_ndcg@5 +0.0059
```

但仍远低于 Step3 gate：

```text
recall@5 >= 0.50
jaccard@5 >= 0.35
```

因此全量 val 下 Step3 仍不进入 full pipeline。

### margin-positive 子集对齐比较

以 `deberta_listwise_margin_positive/eval_val/selection_trace.jsonl` 的 840 条 event_id 为共同子集，重算三个模型的同子集指标：

| Run | train filter | shuffle | subset n | recall@5 | jaccard@5 | top1_match | oracle_rank_ndcg@5 | pairwise_order_acc@5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `deberta_listwise` | all | 0.0 | 840 | 0.3619 | 0.2398 | 0.1369 | 0.3204 | 0.5190 |
| `deberta_listwise_shuffle03` | all | 0.3 | 840 | 0.3645 | 0.2423 | 0.1464 | 0.3211 | 0.5703 |
| `deberta_listwise_margin_positive` | margin_positive | 0.3 | 840 | 0.3576 | 0.2372 | 0.1798 | 0.3478 | 0.5801 |

结论：`margin_positive + shuffle03` 明显增强 order 指标，尤其 `top1_match` 与 `oracle_rank_ndcg@5`；但没有提升 set selection，`recall@5` / `jaccard@5` 反而略低于 all/shuffle03 模型。

这说明 filtering high-signal rows 可以让模型更会排序“已接近 oracle 的集合”，但没有解决“选中 oracle set”的核心问题。

### Rank shortcut 诊断

候选池原始顺序是 hybrid_score descending。若模型过度依赖 rank prior，会大量选择 `candidate_idx=0` 或 hybrid top5。

| Run | pred top1 为 idx0 | oracle top1 为 idx0 | pred top5 中落在 hybrid top5 的比例 | oracle top5 中落在 hybrid top5 的比例 |
|---|---:|---:|---:|---:|
| `deberta_listwise` | 0.5118 | 0.1028 | 0.4819 | 0.3375 |
| `deberta_listwise_shuffle03` | 0.4262 | 0.1028 | 0.4257 | 0.3375 |
| `deberta_listwise_margin_positive` | 0.6369 | 0.1143 | 0.6390 | 0.3395 |

`shuffle03` 确实削弱了 rank shortcut：idx0 top1 比例从 51.2% 降到 42.6%，pred top5 落在 hybrid top5 的比例从 48.2% 降到 42.6%。但偏置仍明显高于 oracle 分布。

`margin_positive + shuffle03` 的 rank shortcut 反而更强，idx0 top1 达到 63.7%。这说明仅过滤 margin-positive 样本不能消除 hybrid-rank shortcut。

### Loss 曲线

`deberta_listwise` 的训练 history 分段均值：

| loss | first100 | last100 | 变化 |
|---|---:|---:|---:|
| total loss | 1.9368 | 1.9097 | -0.0271 |
| mask_loss | 0.6848 | 0.6133 | -0.0715 |
| listmle_loss | 1.4747 | 1.4704 | -0.0043 |
| order_loss | 0.5133 | 0.5105 | -0.0028 |

主要下降来自 `mask_loss`，`ListMLE/order` 基本没有有效下降。这与 selection-only 指标一致：模型学到一点 selected-mask / rank-prior 信息，但没有稳定学到 oracle greedy order 或 oracle set selection。

### Gate 判定

| Gate 条件 | 要求 | 当前最好结果 | 是否通过 |
|---|---:|---:|---|
| `recall@5 >= 0.50` | 0.5000 | 0.3732 (`shuffle03`) | 否 |
| `jaccard@5 >= 0.35` | 0.3500 | 0.2518 (`shuffle03`) | 否 |
| `oracle_rank_ndcg@5 > hybrid-order control` | > 0.2872 | 0.3131 (`shuffle03`, all-val) | 是 |
| `top1_match > hybrid-order control` | > 0.1028 | 0.1279 (`shuffle03`, all-val) | 是 |

Step3 目前表现为：order metrics 有改善，set metrics 没突破。根据预设 gate，当前 Step3 run 判定为 **No-Go**，不建议进入 full pipeline。

### 阶段结论

1. `shuffle_probability=0.3` 是正向诊断，说明原模型确有 candidate-rank shortcut；但它只能小幅改善，不能把 set metrics 拉到 gate。
2. `margin_positive + shuffle03` 提升 top1 / NDCG / pairwise order，但没有提升 recall / Jaccard，说明高 margin 过滤主要改善 order，不解决 set selection。
3. 当前 Step3 架构没有充分学到 oracle set-selection 信号；继续单纯增加 epoch 或扩大同构训练，不太可能从 `recall@5≈0.37` 提到 `0.50`。
4. 若继续压 Step3，应优先做 rank-prior ablation：去掉或弱化 `rank_embedding`、`hybrid_rank_norm`、`candidate_idx_norm`，只保留 `hybrid_score` 作为连续 retrieval prior。
5. 若 rank-prior ablation 仍不过 gate，应转向 Step4 sequential pointer selector，直接建模 oracle greedy order 与 prefix-dependent selection。
