# v0.5c Prompt × Evidence Diagnostic Plan

日期：2026-05-28  
目标版本：`v0.5c_prompt_evidence_diagnostic`  
状态：计划文档  
定位：eval-only diagnostic，不训练新 selector，不训练新 verifier。

---

## 1. 背景与问题定义

v0.5b 的结果显示，旧 oracle-direct verifier 在当前 map-aware prompt 下没有吃到 evidence map 的收益；最强组合反而是 `v0_5a_base_only_top5`，而 `v0_5a_evidence_map_top5` 与 `fusion_refit_all_features_plus_direct_ce_top5` 的最终分类指标都处在较低水平。

但这个结果不能直接解释为 “evidence map 无效”。当前 v0.5b 同时改变了两个变量：

1. evidence source：从 oracle evidence 切换到 selector evidence；
2. prompt rendering：从旧 verifier 训练时使用的 plain evidence prompt 切换到 map-aware prompt。

因此 v0.5c 的核心目标是拆解 v0.5b 的失败来源：

```text
最终分类低分 = prompt OOD ? + evidence selection gap ? + token/truncation gap ? + checkpoint sensitivity ?
```

v0.5c 不追求立刻提高最终指标，而是生成一个足够清楚的诊断矩阵，用于决定下一步应该走：

- 缩短 / 重写 map prompt；
- 继续做 selector fusion；
- 训练 map-aware verifier；
- 或者把 evidence map 暂时降级为解释层。

---

## 2. 核心诊断问题

v0.5c 要回答四个问题。

### Q1. 旧 verifier 是否对 map-aware prompt 严重 OOD？

最关键的 paired test 是：

```text
oracle evidence + original/plain prompt
vs
oracle evidence + map-aware prompt
```

如果 oracle evidence 在 plain prompt 下接近原 oracle-direct checkpoint 表现，但在 map prompt 下大幅下降，那么 v0.5b 的主因是 prompt distribution shift，而不是 selector evidence 本身。

### Q2. selector evidence 本身是否不足？

在同一种 prompt style 下比较：

```text
oracle evidence
vs
v0.4d fusion evidence
vs
v0.5a base_only evidence
vs
v0.5a evidence_map evidence
vs
original_pool_order evidence
```

如果 selector evidence 在 plain prompt 和 map prompt 下都远低于 oracle evidence，则主因是 evidence gap。此时下一步应该做 selector utility distillation 或 learned map-feature fusion，而不是只改 prompt。

### Q3. map-aware prompt 是否挤占 token budget，导致 evidence 被截断或丢弃？

当前 map prompt 会额外渲染 claim atoms、relation、directness、covered atoms、key spans 等字段。需要确认：

```text
map prompt 是否让 evidence_count 从 5 变成 <5？
map prompt 是否导致重要 evidence 被 tail-pop？
map prompt token_count 是否显著高于 plain prompt？
```

如果分类下降主要发生在 `was_truncated=true` 或 `evidence_count<5` 的样本上，则应优先压缩 prompt，而不是调整 selector。

### Q4. 低分是否是 checkpoint 特异性？

v0.5b 中不同 selector 的最佳 checkpoint 不完全一致。v0.5c 至少需要覆盖：

```text
checkpoint-600：原 oracle-direct validation prior 最强
checkpoint-500：v0.5b 中 map/fusion 组合较强
```

扩展阶段再加入 `checkpoint-550` 与 `best`。

---

## 3. 实验变量

### 3.1 Evidence source 轴

核心矩阵使用 5 个 evidence source：

| evidence_source | 目的 | 备注 |
| --- | --- | --- |
| `oracle_top5` | 上界 / prompt OOD anchor | 用 Stage2 oracle selected evidence；不作为部署 selector |
| `original_pool_order_top5` | 原始检索顺序 control | 判断 selector 是否至少超过 retrieval baseline |
| `fusion_refit_all_features_plus_direct_ce_top5` | 当前 oracle-overlap selector-of-record | v0.4d 最强 overlap-oriented baseline |
| `v0_5a_base_only_top5` | v0.5b 最强 eval-only evidence source | 用 map base score，但不走 full map greedy |
| `v0_5a_evidence_map_top5` | 当前 explanation-oriented selector | 高 atom coverage / directness，但 overlap 下降 |

可选扩展 evidence source：

| evidence_source | 使用条件 |
| --- | --- |
| `qd_union_source_score_top5` | 需要比较 QD retrieval control 时加入 |
| `v0_5a_coverage_only_top5` | 需要定位 coverage-only 是否伤害分类时加入 |
| `oracle_likelihood_top5` | 需要拆 v0.3.1 与 v0.4d fusion 差异时加入 |
| `direct_ce_text_only_top5` | 需要验证 direct CE top1/NDCG 现象时加入 |

### 3.2 Prompt style 轴

核心矩阵使用 2 个 prompt style：

| prompt_style | 定义 | 目的 |
| --- | --- | --- |
| `plain_original` | 旧 verifier 训练 / eval 所用的 claim + evidence block 格式 | prompt anchor |
| `map_full` | v0.5b 当前 evidence-map prompt 格式 | 检查 map-aware rendering 是否 OOD |

建议加入一个轻量扩展：

| prompt_style | 定义 | 使用条件 |
| --- | --- | --- |
| `map_minimal` | 保留 evidence text；只加极简 `relation/directness/atoms` 元数据；不展开完整 claim atoms 和 spans | 如果 `map_full` 明显低于 `plain_original`，用于判断是否只是 prompt 过长 / 格式过重 |

`map_minimal` 不应在第一轮替代 `map_full`，而应作为 second-pass ablation。

### 3.3 Checkpoint 轴

第一轮使用：

| checkpoint | 目的 |
| --- | --- |
| `checkpoint-600` | 原 oracle-direct validation prior 最强或接近最强 |
| `checkpoint-500` | v0.5b 中 map/fusion 组合表现相对较强 |

扩展阶段加入：

```text
best, checkpoint-550, checkpoint-450
```

---

## 4. 核心实验矩阵

第一轮最小完整矩阵：

```text
5 evidence sources × 2 prompt styles × 2 checkpoints = 20 eval jobs
```

| evidence_source | plain_original / checkpoint-600 | map_full / checkpoint-600 | plain_original / checkpoint-500 | map_full / checkpoint-500 |
| --- | ---: | ---: | ---: | ---: |
| `oracle_top5` | 必跑 | 必跑 | 必跑 | 必跑 |
| `original_pool_order_top5` | 必跑 | 必跑 | 必跑 | 必跑 |
| `fusion_refit_all_features_plus_direct_ce_top5` | 必跑 | 必跑 | 必跑 | 必跑 |
| `v0_5a_base_only_top5` | 必跑 | 必跑 | 必跑 | 必跑 |
| `v0_5a_evidence_map_top5` | 必跑 | 必跑 | 必跑 | 必跑 |

第二轮扩展矩阵：

```text
5 evidence sources × 3 prompt styles × 5 checkpoints = 75 eval jobs
```

第二轮只在第一轮发现 prompt gap 或 checkpoint sensitivity 后运行，不建议一开始全量跑。

---

## 5. 数据构建要求

### 5.1 paired 数据必须严格可比

同一个 `event_id + evidence_source` 在不同 prompt style 下必须满足：

```text
selected evidence text 相同
selected evidence order 相同
gold label 相同
checkpoint 相同
max_length 相同
label_format / output_mode 相同
```

唯一允许变化的是 prompt rendering。

### 5.2 oracle_top5 trace 构建

需要构建一个虚拟 selector trace：

```text
selector_name = oracle_top5
selected_candidates = Stage2 oracle selected_texts / selected_indices
selected_keys = oracle selected candidate keys
selection_rank = oracle order
```

注意：`gold_label`、`oracle_step`、oracle margin/logprob 只能进入 trace metadata 或 analysis file，不能进入 prompt。

对于 `oracle_top5 + map_full`，如果 oracle candidates 已经存在 evidence map annotation，则直接复用 map fields；如果缺少 annotation，优先补跑一次 teacher map annotation。若暂时不补跑，则只能标记为 `map_skeleton`，不应与完整 `map_full` 混为同一实验。

### 5.3 plain_original builder

建议新增一个统一 builder，而不是临时改 v0.5b 脚本：

```text
scripts/phase5_selectors/build/build_prompt_evidence_diagnostic_data.py
```

核心参数：

```bash
--selection-trace PATH
--expected-selector-name SELECTOR
--prompt-style plain_original|map_full|map_minimal
--output-dir PATH
--split val
--raw-path data/raw/LIAR-RAW/val.json
--config configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml
--max-evidence-chars 420
--max-span-chars 160
--sample-limit N
```

`plain_original` 应复用现有 `fact_checking.build.prompts.build_user_content()` 的 claim + evidence block 格式。

`map_full` 可复用当前 `build_evidence_map_verifier_data.py` 的渲染逻辑。

`map_minimal` 建议格式：

```text
Claim:
{claim}

Evidence:
[1] relation={relation}; directness={directness}; atoms={A1,A2}
{text}
[2] relation={relation}; directness={directness}; atoms={A3}
{text}
...

Respond with exactly one line: Label: <label>
```

不展开完整 claim atom list，不渲染 spans，减少 token overhead。

---

## 6. 需要记录的指标

### 6.1 分类指标

每个 job 至少记录：

| metric | 说明 |
| --- | --- |
| `accuracy` | 基础 sanity check |
| `macro_f1` | primary metric |
| `true_side_macro_f1` | 检查 mostly-true / true 侧是否塌陷 |
| `per_label_f1` | 定位标签侧 collapse |
| `confusion_matrix` | 分析 false-side / true-side 混淆 |
| `prediction_label_distribution` | 检查是否预测集中到少数类 |

### 6.2 Prompt / truncation 指标

每个 build dataset 需要输出：

| metric | 说明 |
| --- | --- |
| `prompt_token_count.mean/max/p95` | 判断 prompt 是否过长 |
| `target_token_count.mean/max` | sanity check |
| `evidence_count.mean/min` | 实际进入 verifier 的 evidence 数量 |
| `evidence_count_before.mean` | selector 原始 topK 数量 |
| `was_truncated.rate` | 核心 truncation 指标 |
| `evidence_dropped_rate` | `evidence_count < evidence_count_before` 的比例 |
| `plain_vs_map_token_delta` | paired token 增量 |
| `plain_vs_map_evidence_count_delta` | paired evidence 数量损失 |

### 6.3 Evidence quality 指标

从 selection trace 聚合：

| metric | 说明 |
| --- | --- |
| `recall@5` | 与 oracle set overlap |
| `jaccard@5` | 与 oracle set set-overlap |
| `top1_match` | 第一条是否命中 oracle |
| `NDCG@5` | ordered overlap |
| `weighted_atom_coverage@5` | map coverage |
| `direct_or_partial_map_rate@5` | directness |
| `background_only_map_rate@5` | 背景占比 |

这些指标不直接作为 v0.5c primary metric，但用于解释分类结果。

---

## 7. 判定规则

### 7.1 Prompt OOD 判定

如果满足：

```text
macro_f1(oracle_top5, map_full) <= macro_f1(oracle_top5, plain_original) - 0.05
```

则判定存在显著 prompt OOD。

如果 gap 大于 `0.10`，则不建议继续用旧 verifier 直接评估 full map prompt；下一步应优先做：

```text
map_minimal prompt ablation
或
train-side map-aware verifier training
```

### 7.2 Evidence gap 判定

在同一 prompt style 下，如果：

```text
macro_f1(selector, prompt) <= macro_f1(oracle_top5, prompt) - 0.05
```

且该 gap 在 plain 与 map 两种 prompt 下都成立，则主因是 selector evidence gap。

对应下一步：

```text
v0.6 learned map-feature fusion
或
v0.7 set-level verifier utility distillation
```

### 7.3 Rendering gap 判定

如果：

```text
macro_f1(selector, map_full) <= macro_f1(selector, plain_original) - 0.05
```

但：

```text
macro_f1(selector, plain_original)
```

相对可接受，则主因是 rendering gap。此时不应继续改 selector，而应改 prompt。

### 7.4 Truncation gap 判定

如果满足任一条件：

```text
mean_evidence_count(map_full) <= mean_evidence_count(plain_original) - 1.0
was_truncated_rate(map_full) - was_truncated_rate(plain_original) >= 0.20
```

则需要进一步分层：

```text
truncated samples macro_f1
vs
non-truncated samples macro_f1
```

如果分类下降主要来自 truncated samples，则下一步优先做 prompt compression。

### 7.5 Checkpoint sensitivity 判定

如果不同 checkpoint 对同一 evidence/prompt 组合的 macro-F1 差异大于 `0.03`，则 checkpoint sensitivity 明显。此时后续比较必须固定 checkpoint，不能混用各组合的 best checkpoint 做结论。

---

## 8. 推荐实现顺序

### Step 0. Artifact audit

确认以下输入存在：

```text
outputs/selectors/evidence_map_selector/v0_5a_val/selection_trace_val.jsonl
outputs/oracle_direct_verifier/stage2_sentence/train/b3_oracle_sentence_direct_verifier_1024_20260519-200709/checkpoint-600
outputs/oracle_direct_verifier/stage2_sentence/train/b3_oracle_sentence_direct_verifier_1024_20260519-200709/checkpoint-500
data/raw/LIAR-RAW/val.json
configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml
```

确认 trace 中至少包含：

```text
v0_5a_evidence_map_top5
v0_5a_base_only_top5
fusion_refit_all_features_plus_direct_ce_top5
original_pool_order_top5
```

若没有 `original_pool_order_top5`，从 v0.5a trace 的 broader selector sweep 重新生成。

### Step 1. 构建 oracle_top5 trace

新增脚本：

```text
scripts/phase5_selectors/build/build_oracle_top5_trace.py
```

输出：

```text
outputs/selectors/evidence_map_selector/v0_5c_val_prompt_evidence_diagnostic/traces/oracle_top5_trace_val.jsonl
```

如果 oracle evidence 缺 map annotation，记录：

```json
{
  "map_annotation_status": "missing|ok|fallback"
}
```

### Step 2. 构建 paired verifier data

新增或复用统一 builder：

```text
scripts/phase5_selectors/build/build_prompt_evidence_diagnostic_data.py
```

对每个组合输出：

```text
outputs/selectors/evidence_map_selector/v0_5c_val_prompt_evidence_diagnostic/
  verifier_data/{evidence_source}/{prompt_style}/build_val.jsonl
  verifier_data/{evidence_source}/{prompt_style}/train.resolved.yaml
  verifier_data/{evidence_source}/{prompt_style}/build_report.json
```

### Step 3. Smoke test

先跑小样本，不跑完整 eval：

```bash
SPLIT=val \
SAMPLE_LIMIT=64 \
RUN_EVAL=false \
EVIDENCE_SOURCES=oracle_top5,v0_5a_evidence_map_top5 \
PROMPT_STYLES=plain_original,map_full \
bash scripts/phase5_selectors/run/run_prompt_evidence_diagnostic_v0_5c.sh
```

检查：

```text
build_val.jsonl 是否都有 prompt / target / gold_id
evidence_count 是否符合预期
prompt_token_count 是否没有系统性 overflow
plain 与 map 是否 selected evidence 完全一致
```

### Step 4. First-pass full eval

运行 20-job 最小矩阵：

```bash
SPLIT=val \
EVIDENCE_SOURCES=oracle_top5,original_pool_order_top5,fusion_refit_all_features_plus_direct_ce_top5,v0_5a_base_only_top5,v0_5a_evidence_map_top5 \
PROMPT_STYLES=plain_original,map_full \
CHECKPOINTS=checkpoint-600,checkpoint-500 \
OUTPUT_DIR=outputs/selectors/evidence_map_selector/v0_5c_val_prompt_evidence_diagnostic \
bash scripts/phase5_selectors/run/run_prompt_evidence_diagnostic_v0_5c.sh
```

### Step 5. Summarize and pairwise analysis

新增分析脚本：

```text
scripts/phase5_selectors/analyze/summarize_prompt_evidence_diagnostic_v0_5c.py
```

输出：

```text
comparison_table.csv
comparison_table.json
analysis_summary.md
truncation_report.csv
label_shift_report.csv
paired_prompt_delta_by_event.jsonl
case_studies.md
```

`analysis_summary.md` 至少包含四张表：

1. classification comparison by evidence_source × prompt_style × checkpoint；
2. prompt delta table：map_full - plain_original；
3. selector gap table：selector - oracle_top5 under same prompt；
4. truncation / evidence_count table。

---

## 9. 结果解释模板

### 9.1 如果 oracle_top5 + map_full 也低

结论：旧 verifier 不适配 map prompt。

行动：

```text
1. 不再用旧 verifier 直接评估 full map prompt 作为主结论；
2. 跑 map_minimal prompt ablation；
3. 若 map_minimal 仍低，进入 train-side map-aware verifier；
4. v0.5a evidence_map 暂时作为解释层，不作为分类 prompt。
```

### 9.2 如果 oracle_top5 + map_full 正常，但 selector + map_full 低

结论：prompt 可用，selector evidence gap 是主因。

行动：

```text
1. 回到 selector：做 v0.6 learned map-feature fusion；
2. 加入 set-level utility / marginal gain 训练；
3. 保留 map prompt，但不继续手调 v0.5a greedy weights。
```

### 9.3 如果 selector + plain_original 明显好于 selector + map_full

结论：evidence 本身可用，map rendering 伤害旧 verifier。

行动：

```text
1. 分类链路使用 plain_original；
2. map 信息进入 selector feature 或解释输出，不直接进入旧 verifier prompt；
3. 单独做 map-aware verifier training。
```

### 9.4 如果 map_full 主要因 truncation 下降

结论：结构信息有潜在价值，但 prompt 太重。

行动：

```text
1. 启用 map_minimal；
2. 限制 claim atoms 数量，例如 top 3 atoms；
3. 删除 spans 或只保留 top1 span；
4. evidence text 优先完整保留，map metadata 放在短前缀；
5. 重新跑 v0.5c second-pass。
```

### 9.5 如果所有 selector evidence 在 plain/map 下都低于 oracle 很多

结论：下游差距主要来自 selector evidence 与 oracle utility set 的差异。

行动：

```text
1. 不再只优化 prompt；
2. 做 v0.7 set-level verifier utility distillation；
3. 用 verifier margin / logprob 作为 set-level target；
4. 考虑扩大 candidate pool 或改变 retrieval granularity。
```

---

## 10. 建议输出目录结构

```text
outputs/selectors/evidence_map_selector/v0_5c_val_prompt_evidence_diagnostic/
├── run_manifest.json
├── traces/
│   ├── oracle_top5_trace_val.jsonl
│   └── merged_selection_trace_val.jsonl
├── verifier_data/
│   ├── oracle_top5/
│   │   ├── plain_original/
│   │   │   ├── build_val.jsonl
│   │   │   ├── train.resolved.yaml
│   │   │   └── build_report.json
│   │   └── map_full/
│   ├── original_pool_order_top5/
│   ├── fusion_refit_all_features_plus_direct_ce_top5/
│   ├── v0_5a_base_only_top5/
│   └── v0_5a_evidence_map_top5/
├── eval/
│   └── {evidence_source}/{prompt_style}/{checkpoint}/
│       ├── metrics.json
│       └── val_predictions.jsonl
└── analysis/
    ├── comparison_table.csv
    ├── comparison_table.json
    ├── analysis_summary.md
    ├── truncation_report.csv
    ├── label_shift_report.csv
    ├── paired_prompt_delta_by_event.jsonl
    └── case_studies.md
```

---

## 11. Case study 选择

建议固定以下三类 case：

1. 之前已经分析过的代表性 case：
   - `4855.json`
   - `11447.json`
   - `10443.json`
2. prompt gap 最大的 case：
   - `plain_original` 正确但 `map_full` 错误；
   - 或 plain label logprob margin 明显高于 map。
3. evidence gap 最大的 case：
   - `oracle_top5` 正确但所有 selector evidence 错误。

每个 case study 输出：

```text
claim
gold label
predictions by evidence_source × prompt_style
selected evidence text
prompt token count
evidence_count / was_truncated
map fields: atoms, relation, directness, background flag
plain vs map prompt excerpt
```

不要只看 aggregate metric。v0.5c 的价值在于定位失败机制。

---

## 12. 风险与防泄漏约束

### 12.1 不能把 oracle 信息写入 prompt

以下字段只能用于 offline metric / trace / analysis，不能进入 prompt：

```text
gold_label
oracle_selected
selected_indices
oracle_step
oracle margin / logprob
candidate_key / candidate_uid
rank/source/model score
```

### 12.2 oracle_top5 只是诊断 anchor

`oracle_top5` 不是真实 selector，不得作为部署结果；它只用于判断 prompt 与 checkpoint 是否能复现 oracle-direct 上界附近表现。

### 12.3 不混用 best checkpoint 作为主比较

主表必须先固定 checkpoint，再比较 prompt/evidence。可以另做 “best over checkpoints” 表，但不能用它替代 fixed-checkpoint 结论。

### 12.4 不把 v0.5c 解释成最终 held-out 结果

v0.5c 是 val diagnostic。它用于路线决策，不应被描述成最终 test 结果。

---

## 13. Go / No-Go 决策

### Go to prompt ablation

条件：

```text
oracle_top5 + map_full 明显低于 oracle_top5 + plain_original
```

下一步：`map_minimal` / compressed map prompt。

### Go to learned selector fusion

条件：

```text
selector + plain_original 仍明显低于 oracle_top5 + plain_original
但 map features 在 evidence quality 上有稳定增益
```

下一步：v0.6 `fusion_refit_plus_map_features_top5`。

### Go to map-aware verifier training

条件：

```text
map prompt 对旧 verifier OOD
但 map evidence quality 指标明显优于 plain/fusion evidence
```

下一步：train split 生成同构 map data，训练 map-aware verifier。

### Go to set-level utility distillation

条件：

```text
prompt gap 可控，但 selector evidence 无论 plain/map 都远低于 oracle evidence
```

下一步：v0.7 set-level margin / marginal gain distillation。

---

## 14. 最小可交付标准

v0.5c 完成时，至少应交付：

1. `analysis_summary.md`：包含 20-job 核心矩阵结果；
2. `comparison_table.csv/json`：可复现所有 metric；
3. `truncation_report.csv`：说明 map prompt 是否导致 evidence loss；
4. `case_studies.md`：至少 6 个 paired case；
5. 明确路线决策：
   - `PROMPT_OOD`
   - `EVIDENCE_GAP`
   - `TRUNCATION_GAP`
   - `CHECKPOINT_SENSITIVITY`
   - 或组合标签。

v0.5c 的最终一句结论应该是以下形式：

```text
v0.5b 的低分主要来自 {prompt OOD / evidence gap / truncation / checkpoint sensitivity}；
因此下一步应进入 {map_minimal prompt ablation / v0.6 learned map-feature fusion / map-aware verifier training / set-level utility distillation}。
```
