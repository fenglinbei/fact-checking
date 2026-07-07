# HoVer 数据 schema 与运行计划

日期：2026-07-07

## 数据落地

HoVer 官方数据已下载到：

```text
data/raw/HoVer/
├── hover_train_release_v1.1.json
├── hover_dev_release_v1.1.json
├── hover_test_release_v1.1.json
├── train.json -> hover_train_release_v1.1.json
├── val.json -> hover_dev_release_v1.1.json
  └── test.json -> hover_test_release_v1.1.json   # downloaded but unused in this plan
```

下载来源：

- Official page: https://hover-nlp.github.io/
- GitHub release files: https://github.com/hover-nlp/hover/tree/main/data/hover

验证后计数：

| split | rows | labels |
|---|---:|---|
| train | 18171 | `supported`: 11023, `not_supported`: 7148 |
| val/dev | 4000 | `supported`: 2000, `not_supported`: 2000 |
| test | 4000 | claim-only，无 gold label / supporting facts；本计划不使用 |

注意：官方 test split 只有 `uid` 和 `claim`，没有本地可用 gold label。本计划只采用主报告口径：使用 HoVer dev/validation set，也就是项目内的 `data/raw/HoVer/val.json`，做本地可复现评估。这也符合后续 ProgramFC / GraphCheck 一类工作的常见做法：由于 test labels 不公开，将 development set 当作 test/eval set 报告。

## 新增 schema

代码层新增 `hover2` label schema：

```text
supported     -> A -> id 0
not_supported -> B -> id 1
```

原始 HoVer label 映射：

```text
SUPPORTED     -> supported
NOT_SUPPORTED -> not_supported
missing label -> ""，仅用于 claim-only test rows
```

`load_split(..., dataset="hover", label_schema="hover2")` 当前输出：

- `event_id`: HoVer `uid`
- `claim`: HoVer `claim`
- `label`: `supported` / `not_supported` / `""`
- `reports`: 目前为空。HoVer 原始 release 不内嵌 Wikipedia 页面正文，后续 build 阶段必须接入 HoVer 数据说明要求的 HotpotQA processed Wikipedia corpus。
- `metadata`: `source_dataset`, `has_gold_label`, `supporting_facts`, `num_hops`, `hpqa_id`

## 当前最优运行锚点

本计划以当前 LIAR-RAW learned-marginal fullpool 运行族为迁移锚点。按 val Macro-F1 选主锚点：

| run | surface | val Macro-F1 | val Acc | test Macro-F1 | test Acc | 用途 |
|---|---|---:|---:|---:|---:|---|
| `learned_marginal_proxy_fullpool_minmax9_9` | `label_token` | 0.3755 | 0.3721 | 0.3481 | 0.3477 | 主锚点，val-selected |
| `learned_marginal_proxy_fullpool_minmax5_10` | `label_token` | 0.3659 | 0.3611 | 0.3666 | 0.3597 | 稳健性参照，test 更高 |

主锚点配置特征：

- backbone: `Ministral-3-8B-Instruct-2512`
- selector: `mrec_greedy_transition_v0_2_learned_marginal_proxy_fullpool`
- prompt evidence policy: `minmax`
- evidence count: `min=9`, `max=9`
- prompt style: `mrec_min`
- evidence text mode: `full`
- LoRA suffix: `_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw`

迁移原则：复用方法结构、训练超参和报告口径；不要复用 LIAR-RAW 的 trace/cache 作为 HoVer 输入。

## 运行目标

第一版 HoVer 运行只建立可复现本地评估：

1. 支持 HoVer train/dev 的 gold label 训练与验证。
2. 生成 HoVer dev 上的 label metrics：Accuracy、Macro-F1、per-hop Macro-F1/Accuracy。
3. 若 retrieval/evidence selector 完整接入，再报告 HoVer 风格的 Passage EM/F1、Sentence EM/F1、HoVer Score。

本轮明确不使用官方 claim-only test，不生成官方提交文件。

## 阶段计划

### S0. 数据与 schema smoke

目标：确保 HoVer raw split、`hover2` schema、parser、metrics 都能工作。

已验证命令：

```bash
PYTHONPATH=src /data/liaozijie/conda/accelerate-fc-gemma4/bin/python -m pytest \
  src/fact_checking/data/test_hover_loader.py \
  src/sft/test_hover_labels.py -q
```

继续保留的检查只覆盖本轮使用的 train/val：

```bash
jq 'length' data/raw/HoVer/train.json
jq 'length' data/raw/HoVer/val.json
```

### S1. Wikipedia corpus 接入

目标：准备 HoVer 数据说明要求的 HotpotQA processed Wikipedia corpus。

HoVer 数据说明要求使用 HotpotQA 团队处理过的 Wikipedia，因为其它 Wikipedia 版本内容可能不同。推荐落地为：

```text
data/raw/HoVer/wiki/
outputs/cache/hover/wiki_index/
```

验收：

- 能用 HoVer `supporting_facts` 的 page title 找到对应 Wikipedia page。
- dev split 的 supporting document title 命中率接近 100%；低于 99% 时先处理 title normalization。
- 页面句子索引能与 `supporting_facts: [title, sent_idx]` 对齐。

### S2. Gold-document verifier baseline

目标：先验证 `hover2` verifier 训练链路，不把 retrieval 问题混进来。

构建方式：

- train/val 从 `supporting_facts` 收集 gold page title。
- 从 Wikipedia corpus 取对应页面正文或 gold sentence window。
- 生成 `build_train.jsonl` / `build_val.jsonl`。
- `label_schema=hover2`，prompt label letters 为 `A/B`。

建议 case name：

```text
hover__ministral3_8b__gold_docs_minmax9_9
hover__ministral3_8b__gold_sentences_minmax9_9
```

指标：

- Accuracy / Macro-F1
- 2-hop / 3-hop / 4-hop 分组 Accuracy / Macro-F1
- 与 claim-only baseline 对比，确认模型确实使用 evidence

### S3. Open-domain retrieval baseline

目标：建立 HoVer open-domain setting 下的检索入口。

最小可行方案：

1. BM25 page retrieval：claim -> top-100 pages。
2. sentence candidate extraction：top pages -> candidate sentences。
3. 当前 evidence scoring：dense + lexical + BM25 + MMR。
4. 先输出 top-9 evidence，接入 `minmax9_9` prompt。

建议 case name：

```text
hover__ministral3_8b__bm25_page_mmr_sentence_minmax9_9
```

验收：

- dev Retrieval@100 按 supported claims 计算并按 hop 分组。
- Passage EM/F1、Sentence EM/F1 可复现。
- 若 Retrieval@100 明显低于 HoVer/Baleen 基线，不进入 LoRA 主训练，先修检索。

### S4. MREC v0.2 learned-marginal fullpool 迁移

目标：复用当前最佳方法形状，而不是复用 LIAR-RAW artifact。

需要新建 HoVer 侧来源：

```text
outputs/selectors/atom_anchor/hover_abc_v0_1/
outputs/selectors/atom_anchor/hover_abc_v0_1/04_evidence_map/
outputs/selectors/atom_anchor/hover_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy_fullpool/
```

迁移步骤：

1. HoVer claim atomization：沿用 atom schema，输入改为 HoVer claim。
2. HoVer evidence map：用 supporting facts 监督 candidate sentence 是否命中 gold page/sentence。
3. HoVer learned marginal proxy：基于 HoVer train 构造 proxy pairwise preferences。
4. HoVer fullpool trace：`selection_policy=learned_marginal_proxy`，`candidate_top_n=0`，`max_steps=0`。
5. Prompt evidence：先跑 `minmax9_9`，再跑 `minmax5_10` 稳健性参照。

建议 case names：

```text
hover__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax9_9
hover__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10
```

### S5. LoRA 训练与 eval

主训练超参沿用 LIAR-RAW 主锚点：

```text
SFT_LEARNING_RATE=2e-5
SFT_NUM_TRAIN_EPOCHS=12
SFT_EVAL_STEPS=100
SFT_EARLY_STOPPING_PATIENCE=8
EVAL_SPLITS=val
LABEL_SCHEMA=hover2
```

HoVer 是二分类，不应沿用 LIAR 六分类 class weights。第一版建议：

```text
supported=1.0,not_supported=1.0
```

如果 open-domain retrieval 造成类别预测偏斜，再按 train prior 或 dev calibration 做二分类 logit-adjust sweep。

### S6. 报告口径

只报告 `data/raw/HoVer/val.json`：

| case | setting | val Macro-F1 | val Acc | 2-hop F1 | 3-hop F1 | 4-hop F1 | Evidence metrics |
|---|---|---:|---:|---:|---:|---:|---|
| claim-only | no evidence |  |  |  |  |  | n/a |
| gold-docs minmax9_9 | gold docs |  |  |  |  |  | optional |
| BM25/MMR minmax9_9 | open retrieval |  |  |  |  |  | Psg/Sent EM/F1 |
| MREC fullpool minmax9_9 | open retrieval + selector |  |  |  |  |  | Psg/Sent EM/F1 + HoVer Score |
| MREC fullpool minmax5_10 | sensitivity |  |  |  |  |  | Psg/Sent EM/F1 + HoVer Score |

## 风险与边界

- HoVer raw release 不包含 Wikipedia 正文；没有 corpus 接入时，`reports=[]` 是预期状态，不代表可直接跑完整 build。
- 官方 test 没有本地 gold label；本计划不使用它，也不生成提交文件。
- 当前 `learned_marginal_proxy_fullpool_minmax9_9` 的优势来自 LIAR-RAW val-selected 结果；迁移到 HoVer 后必须重新训练 HoVer selector/proxy，不应直接使用 LIAR-RAW weights 得出结论。
- `minmax9_9` 的 LIAR-RAW test 低于 `minmax5_10`，因此 HoVer 第一轮必须同时保留 `minmax5_10` 作为稳健性参照。
