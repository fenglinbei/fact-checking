# Stage2 Oracle 粒度对比与主线决策

生成日期：2026-05-19

## 背景

Stage2 margin re-oracle 原计划修复 V1 中的 cache / chunking 混用问题后，回到 b3 semantic pipeline：

```text
chunk_mmr_fingerprint = e0b01520364d
chunking.strategy = semantic
```

但修正前已完成的 `outputs/oracle_evidence/stage2_margin_train_sharded` 实际来自 sentence-level cache：

```text
chunk_mmr_fingerprint = 432dfc970e75
chunking.strategy = sentence
```

因此需要确认：semantic-level oracle 是否真的优于 sentence-level oracle，还是 sentence 粒度本身更适合当前 margin oracle search。

## 已观察结果

### Sentence-level Stage2 train

产物：

```text
outputs/oracle_evidence/stage2_margin_train_sharded
```

指标：

```text
accuracy = 0.6187779434
macro_f1 = 0.6207150963
chunk_mmr_fingerprint = 432dfc970e75
effective candidate pool: dedup -> hybrid top15 -> greedy oracle top5
```

### Semantic-level partial run

当前运行中的 semantic train oracle 已完成局部样本：

```text
n = 1709
accuracy = 0.5394967817
chunk_mmr_fingerprint = e0b01520364d
n_candidates: min=1, median=10, max=15
```

label 分布：

| label | count |
|---|---:|
| half-true | 347 |
| false | 345 |
| mostly-true | 324 |
| true | 274 |
| barely-true | 256 |
| pants-fire | 163 |

该结果已经明显低于 sentence-level train 的 `0.6188`。

## Paired 对比

为排除 shard / label 分布差异，按相同 `event_id` 对齐 semantic partial 与 sentence-level train oracle。

```text
paired_n = 1720
semantic_acc = 0.5406976744
sentence_acc = 0.6191860465
```

转移矩阵：

| bucket | count |
|---|---:|
| both_correct | 818 |
| sentence_only | 247 |
| semantic_only | 112 |
| both_wrong | 543 |

结论：

```text
sentence - semantic = +7.85 pp
sentence_only - semantic_only = +135 claims
```

这不是样本划分造成的偶然差异；在同一批 claim 上，sentence-level oracle 明显更强。

## 判断

当前差异更像 evidence granularity 的真实影响，而不是脚本或 cache 漏洞：

1. semantic partial 已确认 fingerprint 为 `e0b01520364d`，说明没有继续误用 sentence cache。
2. semantic 候选池 `median n_candidates = 10`，很多样本在 oracle 搜索前就没有足够细粒度候选。
3. sentence-level 候选更原子，margin oracle 可以从 hybrid top15 中拼出更精准的 5 条证据。
4. semantic chunk 更长、更粗，单条 evidence 更容易混入无关内容；即使召回语义相关段落，label-token verifier 的 margin 也可能被噪声压低。

## 决策

后续主线转回 sentence-level oracle evidence supervision。

| 方向 | 状态 | 原因 |
|---|---|---|
| sentence-level Stage2 oracle | Go / 主线 | paired acc 0.6192，高于 semantic 0.5407；sentence_only 247 vs semantic_only 112 |
| semantic-level Stage2 oracle | Diagnostic only | 作为粒度对照保留，不再等权推进 |
| semantic full train oracle | 暂停或低优先级 | 除非为了完整报告对照，否则不建议继续消耗大量 GPU |

## 后续建议

1. 使用 `outputs/oracle_evidence/stage2_margin_train_sharded` 作为下一阶段主要 oracle supervision 源。
2. 不再把 semantic re-oracle 作为默认主线；若继续运行，只作为 paired diagnostic 或论文/报告中的 chunk granularity 对照。
3. 下一步优先验证 `oracle sentence evidence direct verifier`：直接用 sentence-level oracle selected evidence 构造 train/val build JSONL，训练 label-token CE verifier，确认 oracle evidence supervision 是否能转化为下游泛化。
4. 若 direct verifier 有收益，再推进更强 selector；当前 pointwise logreg 只学到弱 oracle pattern，不应作为最终 selector 路线。
5. 文档表述中应区分两件事：
   - `run_reoracle_stage2.sh` 的 semantic 默认修复了原先 Hydra/defaults 漏洞；
   - 实验决策上，paired 对比后主线选择 sentence-level supervision。
