# Source Coverage v2 Flash 数据集运行与质量报告

日期：2026-06-11  
工作目录：`/data/liaozijie/fact-checking`  
版本：`source_coverage_v2_flash`

## 结论

本次应以以下产物作为可用 coverage 数据集：

- Coverage sidecar：`outputs/data_quality/source_coverage_flash/{liar_raw,rawfc}/`
- 物化数据集：`data/processed/coverage/source_coverage_v2_flash/{liar_raw,rawfc}/{all,covered,covered_weak}/`

旧目录 `outputs/data_quality/source_coverage/{liar_raw,rawfc}/` 是早期 DeepSeek API 配置错误版本，不建议用于训练或分析。该旧版本中 LLM review 全部失败，最终标签实际是 rule-only / embedding-rule 结果。

本次 `source_coverage_v2_flash` 的质量检查通过：raw 与 sidecar 行数对齐，LLM review 全部 `ok`，物化数据集与 sidecar 分布一致。

## 运行过程

运行入口使用：

```bash
NO_PROGRESS=1 bash scripts/phase11_data_quality/rerun_coverage_flash.sh
```

核心配置来自各 split 的 manifest：

- `coverage_version=source_coverage_v2_flash`
- `embedding_model=/data/models/bge-base-en-v1.5`
- `embedding_device=cuda`
- `embedding_precision=bf16`
- `llm_base_url=https://api.deepseek.com`
- `llm_model=deepseek-v4-flash`
- `llm_thinking=disabled`
- `llm_workers=8`
- `llm_retries=3`
- `llm_retry_backoff=2.0`
- `llm_min_confidence=0.65`
- `llm_boundary_margin=0.025`
- `llm_embedding_threshold=0.75`
- `llm_critical_weak_threshold=0.6`
- `checkpoint_every=25`

运行中曾出现 `http.client.RemoteDisconnected: Remote end closed connection without response`。该问题来自 DeepSeek/网络侧关闭连接；后续脚本补充了网络异常捕获、可重试 HTTP 状态、指数退避和 checkpoint/resume。恢复后继续执行，没有从头重跑已成功的 LLM review。

`liar_raw train` 是通过 resume 完成的：checkpoint 中已有 10065 行，其中部分 LLM review 已成功。最终 sidecar 中全部 review 都为 `ok`。注意该 split 的 manifest 里 `llm.review_count=1672` 仅表示本次 resume 实际处理的剩余 review 数，不是最终 sidecar 的总 review 数；最终 sidecar 里 review_needed 总数是 4094。

## 完整性检查

检查范围：

- `outputs/data_quality/source_coverage_flash/{dataset}/source_coverage_{split}.jsonl`
- `outputs/data_quality/source_coverage_flash/{dataset}/.checkpoints/source_coverage_{split}.jsonl`
- `data/processed/coverage/source_coverage_v2_flash/{dataset}/{policy}/{split}.json`

完整性结果：

- LIAR-RAW / RAWFC 的 train / val / test 均与 raw 数据行数一致。
- sidecar 无缺失 id、无额外 id、无重复 id、无非法 `coverage_label`、无 split mismatch。
- checkpoint 与最终 sidecar 行数和 id 集合一致。
- 物化数据集 `all / covered / covered_weak` 行数与 sidecar 标签分布完全一致。
- 物化样本保留原始 schema，并包含 `coverage_label`、`coverage_score`、`coverage_version` 和嵌套 `coverage` 字段。

## LLM Review 状态

| dataset | split | rows | review_needed | llm_ok | llm_non_ok | applied_overrides | model |
|---|---:|---:|---:|---:|---:|---:|---|
| liar_raw | train | 10065 | 4094 | 4094 | 0 | 1085 | deepseek-v4-flash |
| liar_raw | val | 1274 | 558 | 558 | 0 | 148 | deepseek-v4-flash |
| liar_raw | test | 1251 | 558 | 558 | 0 | 151 | deepseek-v4-flash |
| rawfc | train | 1612 | 446 | 446 | 0 | 178 | deepseek-v4-flash |
| rawfc | val | 200 | 69 | 69 | 0 | 29 | deepseek-v4-flash |
| rawfc | test | 200 | 54 | 54 | 0 | 24 | deepseek-v4-flash |

全部需要复核的样本均为 `llm_judgment.status=ok`，没有残留 `error`、`pending`、`parse_error` 或 `invalid_label`。

## 最终 Coverage 分布

### LIAR-RAW

| split | all | covered | weak_covered | uncovered | covered_weak |
|---|---:|---:|---:|---:|---:|
| train | 10065 | 2037 | 3126 | 4902 | 5163 |
| val | 1274 | 320 | 431 | 523 | 751 |
| test | 1251 | 321 | 432 | 498 | 753 |

### RAWFC

| split | all | covered | weak_covered | uncovered | covered_weak |
|---|---:|---:|---:|---:|---:|
| train | 1612 | 109 | 340 | 1163 | 449 |
| val | 200 | 18 | 45 | 137 | 63 |
| test | 200 | 19 | 39 | 142 | 58 |

## Rule 到最终标签迁移

### LIAR-RAW

| split | rule_label -> final_label | count |
|---|---|---:|
| train | weak_covered -> weak_covered | 2532 |
| train | uncovered -> uncovered | 4571 |
| train | covered -> covered | 1877 |
| train | uncovered -> weak_covered | 426 |
| train | weak_covered -> uncovered | 308 |
| train | covered -> weak_covered | 168 |
| train | weak_covered -> covered | 113 |
| train | uncovered -> covered | 47 |
| train | covered -> uncovered | 23 |
| val | uncovered -> uncovered | 487 |
| val | weak_covered -> weak_covered | 340 |
| val | covered -> covered | 299 |
| val | uncovered -> weak_covered | 63 |
| val | weak_covered -> uncovered | 31 |
| val | covered -> weak_covered | 28 |
| val | weak_covered -> covered | 11 |
| val | uncovered -> covered | 10 |
| val | covered -> uncovered | 5 |
| test | uncovered -> uncovered | 453 |
| test | weak_covered -> weak_covered | 343 |
| test | covered -> covered | 304 |
| test | uncovered -> weak_covered | 62 |
| test | weak_covered -> uncovered | 41 |
| test | covered -> weak_covered | 27 |
| test | weak_covered -> covered | 10 |
| test | uncovered -> covered | 7 |
| test | covered -> uncovered | 4 |

### RAWFC

| split | rule_label -> final_label | count |
|---|---|---:|
| train | uncovered -> uncovered | 1161 |
| train | weak_covered -> weak_covered | 228 |
| train | uncovered -> weak_covered | 111 |
| train | covered -> covered | 45 |
| train | uncovered -> covered | 41 |
| train | weak_covered -> covered | 23 |
| train | weak_covered -> uncovered | 1 |
| train | covered -> uncovered | 1 |
| train | covered -> weak_covered | 1 |
| val | uncovered -> uncovered | 137 |
| val | weak_covered -> weak_covered | 30 |
| val | uncovered -> weak_covered | 15 |
| val | uncovered -> covered | 9 |
| val | weak_covered -> covered | 5 |
| val | covered -> covered | 4 |
| test | uncovered -> uncovered | 141 |
| test | weak_covered -> weak_covered | 27 |
| test | uncovered -> weak_covered | 12 |
| test | uncovered -> covered | 8 |
| test | covered -> covered | 8 |
| test | weak_covered -> covered | 3 |
| test | weak_covered -> uncovered | 1 |

## 异常与边界样本检查

低置信覆盖：

- 没有 `applied=True` 且 confidence `< 0.65` 的样本。

最终 `covered` 但 rule 层仍有 `critical_missing`：

- LIAR-RAW train: 48
- LIAR-RAW val: 11
- LIAR-RAW test: 7
- RAWFC train: 41
- RAWFC val: 9
- RAWFC test: 8

抽样检查显示，这类样本大多是 rule anchor 抽取过严或误抽导致。例如将年份、孤立数字、泛化 metric phrase 当成必须完全命中的 critical anchor；LLM 根据 evidence 内容判断为实际覆盖。该现象不应直接视为错误，但后续若要进一步提高自动规则质量，可以单独优化 anchor 抽取。

最终 `uncovered` 但 rule score 较高且没有 `critical_missing`：

- LIAR-RAW train: 24
- LIAR-RAW val: 5
- LIAR-RAW test: 4
- RAWFC train: 1
- RAWFC val: 0
- RAWFC test: 0

这些样本均经过 LLM review 或位于阈值边界附近，数量很小，可以接受。

## 关键 Case 回归

`liar_raw` 的 `12134.json` 保持为 `uncovered`：

- sidecar final label: `uncovered`
- rule label: `uncovered`
- LLM status: `ok`
- LLM label: `uncovered`
- confidence: `0.95`
- critical missing: `year:1977`, `year:1979`

物化结果：

- 存在于 `data/processed/coverage/source_coverage_v2_flash/liar_raw/all/val.json`
- 不存在于 `covered/val.json`
- 不存在于 `covered_weak/val.json`

这符合最初观察：v0.6/v0.7 candidate pool 没有召回 explain 所需的关键年份和比较证据。

## 可用产物

后续训练或分析应使用：

```text
data/processed/coverage/source_coverage_v2_flash/liar_raw/all/{train,val,test}.json
data/processed/coverage/source_coverage_v2_flash/liar_raw/covered/{train,val,test}.json
data/processed/coverage/source_coverage_v2_flash/liar_raw/covered_weak/{train,val,test}.json

data/processed/coverage/source_coverage_v2_flash/rawfc/all/{train,val,test}.json
data/processed/coverage/source_coverage_v2_flash/rawfc/covered/{train,val,test}.json
data/processed/coverage/source_coverage_v2_flash/rawfc/covered_weak/{train,val,test}.json
```

sidecar 与 manifest 保留在：

```text
outputs/data_quality/source_coverage_flash/liar_raw/
outputs/data_quality/source_coverage_flash/rawfc/
```

## 建议

1. 将 `source_coverage_v2_flash` 作为当前可用版本，用于后续 covered / covered_weak 子集训练实验。
2. 不要使用旧的 `outputs/data_quality/source_coverage/{liar_raw,rawfc}/` 结果。
3. 如果后续要进一步提高自动规则质量，优先优化 metric phrase 和年份/数字 anchor 的抽取，减少“rule uncovered 但 LLM covered”的人工纠正压力。
4. 训练接入时应把 clean subset 训练和 full val/test 评估分开记录，避免只在清洗后的分布上报告效果。
