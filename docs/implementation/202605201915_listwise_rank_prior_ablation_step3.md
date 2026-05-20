# Listwise Step3 Rank-prior Ablation 关门实验实施记录

本实验用于关闭 Step3 set-aware listwise 路线的最后一个低成本疑点：当前 listwise 模型是否因过度依赖 candidate 初始 rank / index，导致没有真正学习 Stage2 oracle set-selection 信号。

## 实验目标

在保持 Stage2 sentence-level margin oracle 候选池不变的前提下，移除显式 rank prior：

```text
chunk_mmr_fingerprint = 432dfc970e75
candidate pool        = saved Stage2 oracle candidate_pool top15
selector output       = ordered top5
filter_policy         = all
shuffle_probability   = 0.3
feature_ablation      = hybrid_score_only_prior
```

`hybrid_score_only_prior` 的含义：

```text
保留:
hybrid_score

清零:
dense_score
lexical_score
bm25_log_norm
hybrid_rank_norm
candidate_idx_norm

关闭:
rank_embedding
```

非检索先验特征仍保留，例如 `sent_idx_norm`、`source_index_norm`、`text_token_len_norm`、`claim_token_overlap`、`number_overlap`。

## 已实现改动

新增/修改文件：

```text
src/fact_checking/selectors/listwise.py
src/fact_checking/selectors/test_listwise.py
scripts/selectors/train_listwise_selector.py
scripts/selectors/run_listwise_step3.sh
scripts/selectors/run_listwise_step3_rank_ablation.sh
configs/experiment/b3_listwise_rank_ablation_stage2_sentence_1024.yaml
```

`train_listwise_selector.py` 新增参数：

```text
--feature-ablation {none,no_rank_prior,hybrid_score_only_prior}
```

模型保存的 `metadata.json` / `model_config` 会记录：

```text
feature_ablation
rank_embedding_enabled
dropped_numeric_feature_names
```

评估和 build pipeline 通过 checkpoint metadata 自动复用同一套 feature ablation，不需要额外在 eval/build 时手动传特征开关。

## 运行命令

默认关门实验：

```bash
scripts/selectors/run_listwise_step3_rank_ablation.sh
```

默认输出：

```text
outputs/selectors/stage2_sentence_listwise/deberta_listwise_rank_ablation
outputs/selectors/stage2_sentence_listwise/deberta_listwise_rank_ablation/eval_val
```

常用覆盖项：

```bash
NPROC_PER_NODE=2 \
BATCH_SIZE=2 \
EPOCHS=2 \
scripts/selectors/run_listwise_step3_rank_ablation.sh
```

如果只想做更窄的 rank/index ablation，而保留 dense/lexical/BM25 component score：

```bash
FEATURE_ABLATION=no_rank_prior \
OUTPUT_DIR=outputs/selectors/stage2_sentence_listwise/deberta_listwise_no_rank_prior \
scripts/selectors/run_listwise_step3_rank_ablation.sh
```

## 判定标准

仍沿用 Step3 selection-only gate：

```text
recall@5 >= 0.50
jaccard@5 >= 0.35
oracle_rank_ndcg@5 > hybrid-order control
top1_match > hybrid-order control
```

若 rank-prior ablation 仍停留在 `recall@5≈0.37` / `jaccard@5≈0.25`，则可判定 Step3 当前架构没有解决 oracle set-selection，下一步应转 Step4 sequential pointer selector，而不是继续堆同构 listwise 训练。

