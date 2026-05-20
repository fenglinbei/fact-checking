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

本次未实际训练 Step3 模型；真实 go/no-go 以后续 `eval_val/selection_metrics.json` 为准。
