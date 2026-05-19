# Stage 2 Calibration-aware Re-Oracle 实现说明

## 目标

Stage 2 使用 Stage 1 的 label-token weighted CE verifier 重新搜索 oracle evidence set。搜索目标从旧版：

```text
gold_logprob = log P(y_gold | claim, evidence_set)
```

扩展为 calibration-aware margin：

```text
margin = log P(y_gold | claim, evidence_set)
       - max_{y != y_gold} log P(y | claim, evidence_set)
```

这样 oracle set 不只提高正确标签概率，还要求正确标签超过最强错误标签，避免旧 verifier 下常见的 `true` / `mostly-true` 被 false-side bias 压住。

## 代码改动

| 文件 | 作用 |
|---|---|
| `src/fact_checking/oracle_evidence/scorer.py` | 新增 all-label logprob scoring，并把 LoRARequest 传入 vLLM generate |
| `src/fact_checking/oracle_evidence/search.py` | 新增 `objective=margin` 搜索目标，记录 `gold_logprob`、`best_wrong_logprob`、`margin` |
| `scripts/oracle_evidence/search_optimal_evidence.py` | 新增 `--objective {gold_logprob,margin}`，输出 oracle-results-v3 |
| `scripts/oracle_evidence/run_search.sh` | 新增 `SEARCH_OBJECTIVE` 和 `OUTPUT_DIR` 环境变量 |
| `scripts/oracle_evidence/run_reoracle_stage2.sh` | Stage 2 默认运行脚本 |
| `scripts/oracle_evidence/merge_shards.py` | 合并 sharded oracle JSONL，并重算合并指标 |

2026-05-19 修正：

- `search_optimal_evidence.py` 的 config 加载已优先使用 Hydra compose 展开 experiment defaults，避免只读取当前 YAML 而丢失父配置。
- `run_reoracle_stage2.sh` 默认 `CONFIG` 改为 `configs/experiment/b3_mmr_topk_sweep_1024.yaml`，使后续 re-oracle 默认回到 b3 semantic chunk pipeline，Chunk-MMR fingerprint 应为 `e0b01520364d`。
- 既有 `outputs/oracle_evidence/stage2_margin_train_sharded` 是修正前产物，里面保存的 `candidate_pool_metadata.chunk_mmr_fingerprint` 为 `432dfc970e75`，属于 sentence-cache 结果。它可以用于当前 sentence-cache pointwise 试验，但不应与后续 semantic re-oracle 结果混用。

2026-05-19 粒度决策更新：

- semantic partial run 已确认 fingerprint 为 `e0b01520364d`，不是继续误用 sentence cache。
- paired subset 上 sentence-level oracle 明显强于 semantic-level oracle：sentence accuracy 0.6192 vs semantic accuracy 0.5407，`sentence_only=247`、`semantic_only=112`。
- 因此实验主线转回 sentence-level Stage2 oracle supervision；semantic run 保留为 diagnostic / chunk granularity 对照，不再作为等权主线。

默认仍保持向后兼容：

```bash
SEARCH_OBJECTIVE=gold_logprob bash scripts/oracle_evidence/run_search.sh
```

Stage 2 使用：

```bash
SEARCH_OBJECTIVE=margin bash scripts/oracle_evidence/run_search.sh
```

## 运行方式

默认使用当前 Stage 1 run：

```bash
bash scripts/oracle_evidence/run_reoracle_stage2.sh
```

默认参数：

```text
STAGE1_RUN_DIR=outputs/runs/b3_label_token_ce_1024/label_token_ce_stage1__0ee9b55f
VERIFIER_MODEL=/data/models/Qwen2.5-7B-Instruct/
LORA_ADAPTER=${STAGE1_RUN_DIR}/train/best
CONFIG=configs/experiment/b3_mmr_topk_sweep_1024.yaml
SPLIT=val
TOP_K=5
SEARCH_METHOD=greedy
SEARCH_OBJECTIVE=margin
SAVE_CANDIDATE_POOL=true
SAVE_SEARCH_STEP_SCORES=true
SCORE_BATCH_SIZE=256
NUM_SHARDS=1
SHARD_INDEX=0
RESUME=true
```

当前脚本默认值已更新为：

```text
CONFIG=configs/experiment/b3_mmr_topk_sweep_1024.yaml
```

Stage 1 verifier 仍由 `STAGE1_RUN_DIR` / `LORA_ADAPTER` 指定；`CONFIG` 在这里主要决定 evidence candidate pool 和 prompt/build 配置。

训练集 re-oracle：

```bash
SPLIT=train bash scripts/oracle_evidence/run_reoracle_stage2.sh
```

小样本 smoke test：

```bash
MAX_SAMPLES=32 bash scripts/oracle_evidence/run_reoracle_stage2.sh
```

指定另一个 Stage 1 run：

```bash
STAGE1_RUN_DIR=outputs/runs/b3_label_token_ce_1024/<run_leaf> \
bash scripts/oracle_evidence/run_reoracle_stage2.sh
```

降低 JSONL 体积：

```bash
SAVE_SEARCH_STEP_SCORES=false bash scripts/oracle_evidence/run_reoracle_stage2.sh
```

## Sharding 与断点续跑

长时间 `train` re-oracle 建议用 shard 并行。shard 由 `sha1(event_id) % NUM_SHARDS` 稳定分配，因此同一 `OUTPUT_DIR` 下重跑同一 `SHARD_INDEX` 会跳过已经写入的 `event_id`。

单个 shard：

```bash
SPLIT=train \
NUM_SHARDS=4 \
SHARD_INDEX=0 \
OUTPUT_DIR=outputs/oracle_evidence/stage2_margin_train_sharded \
SAVE_SEARCH_STEP_SCORES=false \
bash scripts/oracle_evidence/run_reoracle_stage2.sh
```

四个 shard 可分别设置 `SHARD_INDEX=0/1/2/3`，每个 shard 会写：

```text
oracle_results_train.shard-00000-of-00004.jsonl
oracle_metrics_train.shard-00000-of-00004.json
```

断点续跑默认开启：

```text
RESUME=true
```

如果同一 shard 中断，使用相同 `OUTPUT_DIR`、`NUM_SHARDS`、`SHARD_INDEX` 重跑即可；脚本会读取已有 shard JSONL，跳过已完成 `event_id`，并继续 append。若最后一行因为异常退出损坏，resume 会丢弃该无效 JSONL 行后继续。

关闭断点跳过：

```bash
RESUME=false bash scripts/oracle_evidence/run_reoracle_stage2.sh
```

合并 shard：

```bash
PYTHONPATH=src python scripts/oracle_evidence/merge_shards.py \
  --input-dir outputs/oracle_evidence/stage2_margin_train_sharded \
  --split train \
  --num-shards 4
```

合并后得到下游监督构建可直接读取的：

```text
oracle_results_train.jsonl
oracle_metrics_train.json
```

默认输出目录：

```text
outputs/oracle_evidence/stage2_margin_<split>_<timestamp>/
```

## 输出字段

每条 `oracle_results_<split>.jsonl` 仍保留旧字段，并新增：

| 字段 | 含义 |
|---|---|
| `search_objective` | `gold_logprob` 或 `margin` |
| `final_objective` | 当前 objective 下最终 set 的分数 |
| `gold_logprob` | 最终 set 的正确标签 logprob |
| `best_wrong_logprob` | 最终 set 的最高错误标签 logprob |
| `margin` | `gold_logprob - best_wrong_logprob` |
| `label_logprobs` | A-F 六个 label token 的 logprob |
| `pred_label` | label-token argmax 预测标签 |
| `prediction_source` | 当前为 `label_logprob_argmax` |

候选池审计字段继续保存：

```text
candidate_pool
candidate_scores
candidate_pool_fingerprint
candidate_pool_metadata
```

如果 `SAVE_SEARCH_STEP_SCORES=true`，每个 search step 还会保存候选 set 的：

```text
search_steps[].candidate_scores[].objective_score
search_steps[].candidate_scores[].gold_logprob
search_steps[].candidate_scores[].best_wrong_logprob
search_steps[].candidate_scores[].margin
search_steps[].candidate_scores[].label_logprobs
```

## 评估重点

Stage 2 是否可进入后续 filtered supervision，主要看：

1. `true` / `mostly-true` 的 oracle accuracy 是否不再低于 fixed-MMR。
2. `margin > 0` 的样本中，`is_correct` 是否显著更稳定。
3. `oracle only correct` 子集是否仍保留足够规模。
4. `candidate_pool_fingerprint` 是否稳定，保证后续 selector 训练可追溯。

## 当前决策：sentence-level 为主线

当前关键 paired 结果：

```text
paired_n = 1720
semantic_acc = 0.5406976744
sentence_acc = 0.6191860465
both_correct = 818
sentence_only = 247
semantic_only = 112
both_wrong = 543
```

解释：

1. 差距来自同一批 `event_id`，不是 shard label 分布差异。
2. semantic 候选池 `median n_candidates = 10`，可搜索空间小于 sentence-level top15。
3. sentence-level evidence 更原子，margin oracle 更容易组合出能触发正确 label-token 的 top5。
4. semantic chunks 更粗，容易混入无关内容，导致 gold-conditioned oracle 上界反而下降。

因此：

| 方向 | 决策 |
|---|---|
| sentence-level Stage2 oracle (`432dfc970e75`) | 主线继续 |
| semantic-level Stage2 oracle (`e0b01520364d`) | 诊断/对照保留 |
| full semantic train oracle | 低优先级，除非需要完整报告对照 |

后续优先使用：

```text
outputs/oracle_evidence/stage2_margin_train_sharded/oracle_results_train.jsonl
```

作为 oracle supervision 源，先做 oracle selected evidence direct verifier，再决定 selector 范式。

## 注意事项

`objective=margin` 会对每个候选 set 评分 A-F 六个 label token，因此 vLLM scoring 成本约为旧 `gold_logprob` objective 的 6 倍。建议先在 `val` 或 `MAX_SAMPLES=32/128` 上 smoke test，再用 sharding + `SAVE_SEARCH_STEP_SCORES=false` 跑完整 `train`。
