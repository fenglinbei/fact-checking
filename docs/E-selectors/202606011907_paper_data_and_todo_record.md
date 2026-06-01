# Paper Data and Todo Record

日期：2026-06-01

状态：v0.6c AAAI-style ablation notes and paper data record

## 1. Current Method Context

当前主方法是 v0.6c：

```text
evidence-chain graph + rule-step adaptive5_10 selector
```

核心 selector：

```text
v0_6c_rule_step_adaptive5_10
```

关键设计点：

- 输入候选池来自 v0.6b evidence-map selector features。
- v0.6c 构建 evidence-chain graph，并用 rule-step policy 选择 evidence。
- evidence budget 为 adaptive 5-10，即 `min_top_k=5`、`max_top_k=10`。
- rule-step policy 主要包含 anchor core、P1 new atom、P2 strong edge、P3 bridge context。
- full pipeline 使用 selector trace 转 verifier data，再训练 LoRA 或 FullFT verifier。

相关默认路径：

```text
outputs/selectors/evidence_chain_graph/v0_6c_adaptive5_10_train/selection_trace_train.jsonl
outputs/selectors/evidence_chain_graph/v0_6c_adaptive5_10_val/selection_trace_val.jsonl
```

## 2. Existing Result Snapshot

现有 v0.6c / v0.6d full-pipeline 对比：

| selector | train mode | accuracy | macro-F1 |
| --- | --- | ---: | ---: |
| v0.6c | LoRA | 0.3336 | 0.3354 |
| v0.6d | LoRA | 0.3163 | 0.3115 |
| v0.6c | FullFT | 0.3454 | 0.3546 |
| v0.6d | FullFT | 0.3422 | 0.3497 |

当前结论：

- v0.6c 在 LoRA 和 FullFT 下都强于 v0.6d。
- v0.6d sufficiency + contradiction 不适合作为主结论底座。
- 后续 trace-lite prompt ablation 应固定 v0.6c selector、候选池、evidence 数量和 evidence 顺序。
- v0.6d 可作为 robustness check，但不作为主实验结论来源。

## 3. AAAI-Style Ablation Angles

### 3.1 Structural Selector Utility

目标问题：

```text
提升是否来自 evidence-chain graph structure，而不是普通 score top-k 或候选池排序？
```

建议对照：

| variant | controlled question |
| --- | --- |
| Full v0.6c | 完整 rule-step evidence-chain graph |
| w/o graph rule | 改成 candidate pool / hybrid score top-k |
| w/o evidence-chain edge | 保留 atom coverage，但不使用 complements / corroborates / tension / bridge_context |
| w/o atom coverage priority | 去掉 P1 new-atom priority，只按 base score 或 edge relation 补证据 |

### 3.2 Rule-Step Policy Component Ablation

目标问题：

```text
每条手工可解释 rule 是否有独立贡献？
```

建议对照：

| variant | removed / retained component |
| --- | --- |
| Full v0.6c | anchor + P1 + P2 + P3 |
| w/o P1 | 不优先补新 claim atom |
| w/o P2 | 不使用 complements / corroborates / tension strong edges |
| w/o P3 | 不加入 bridge context |
| P1 only | 只做 atom coverage，不使用 typed evidence relation |

这组最适合放入论文 ablation table，用来说明方法不是单纯靠更多 evidence 或更高 retrieval score。

### 3.3 Adaptive Evidence Budget Ablation

目标问题：

```text
v0.6c 的收益是否来自 adaptive evidence budget，而不是多喂 evidence？
```

建议对照：

| variant | min_top_k | max_top_k | purpose |
| --- | ---: | ---: | --- |
| fixed-5 | 5 | 5 | 与常规 top-5 selector 对齐 |
| fixed-10 | 10 | 10 | 控制 evidence 数量上限带来的收益 |
| adaptive 5-10 | 5 | 10 | 当前 v0.6c 主方法 |
| random append to 10 | 5 | 10 | 前 5 固定，后续随机或 score append |

如果 adaptive 5-10 强于 fixed-5，说明补充 evidence 有价值；如果强于 fixed-10 或 random append，说明 adaptive rule 本身有价值。

### 3.4 Verifier Prompt / Trace Use Ablation

目标问题：

```text
selected evidence 固定时，轻量 evidence-map 结构标签是否帮助 verifier 更好利用证据？
```

主对照：

| selector | prompt style | train mode | case name |
| --- | --- | --- | --- |
| v0.6c | plain | LoRA | `v0_6c_rule_step_adaptive5_10` |
| v0.6c | trace_lite | LoRA | `v0_6e_trace_lite_on_v0_6c` |
| v0.6c | plain | FullFT | `v0_6c_rule_step_adaptive5_10_fullft` |
| v0.6c | trace_lite | FullFT | `v0_6e_trace_lite_on_v0_6c_fullft` |

trace-lite 设计约束：

- 只使用 downstream 已有、oracle-free 的 map 字段。
- 不生成 rationale。
- 不暴露 selector rule。
- 不加入 score / oracle / gold 信息。
- 不重新跑 evidence-map API。

## 4. Recommended Main Ablation Matrix

论文主消融表建议如下：

| variant | selector set | order | prompt | budget |
| --- | --- | --- | --- | --- |
| Full v0.6c | rule-step graph | rule-step | plain | adaptive 5-10 |
| w/o adaptive | rule-step graph | rule-step | plain | fixed 5 |
| fixed-10 | rule-step graph | rule-step | plain | fixed 10 |
| w/o P2 | no strong edge rule | rule-step | plain | adaptive |
| w/o P3 | no bridge context | rule-step | plain | adaptive |
| score top-k | hybrid / base score | score order | plain | 10 |
| same-set random order | v0.6c set | random | plain | same |
| trace-lite | v0.6c set | v0.6c order | trace_lite | same |

优先级：

1. adaptive budget ablation
2. P1 / P2 / P3 rule component ablation
3. trace-lite prompt ablation
4. score top-k and same-set random-order controls

## 5. Metrics to Report

主指标：

- macro-F1
- accuracy
- per-class F1
- half-true F1
- mostly-true F1
- true-side F1 / macro-F1-plus-true-side

健康指标：

- prompt truncation rate
- evidence_count mean / p50 / p95
- evidence_count_before mean
- prompt_token_count mean / p95 / max
- selected_index_lengths
- adaptive_stop_reasons
- selection_rules
- fallback_row_rate
- p3_row_rate
- post_min_background_addition_rate

解释性指标：

- mean_atom_isolate_rate
- mean_evidence_isolate_rate
- mean_oracle_evidence_connected_rate
- mean_oracle_pair_edge_rate
- mean_max_chain_atom_coverage

## 6. Suggested Run Commands

默认 plain v0.6c：

```bash
bash scripts/phase5_selectors/run/run_v0_6c_rule_step_adaptive5_10_all_pipelines.sh
```

trace-lite prompt ablation：

```bash
bash scripts/phase5_selectors/run/run_v0_6e_trace_lite_on_v0_6c_all_pipelines.sh
```

fixed-5 budget control：

```bash
MIN_TOP_K=5 MAX_TOP_K=5 \
LORA_CASE_NAME=v0_6c_rule_step_fixed5 \
FULLFT_CASE_NAME=v0_6c_rule_step_fixed5_fullft \
bash scripts/phase5_selectors/run/run_v0_6c_rule_step_adaptive5_10_all_pipelines.sh
```

fixed-10 budget control：

```bash
MIN_TOP_K=10 MAX_TOP_K=10 \
LORA_CASE_NAME=v0_6c_rule_step_fixed10 \
FULLFT_CASE_NAME=v0_6c_rule_step_fixed10_fullft \
bash scripts/phase5_selectors/run/run_v0_6c_rule_step_adaptive5_10_all_pipelines.sh
```

注意：P1 / P2 / P3 component ablation 目前需要新增 selector variant 或开关，避免直接修改 v0.6c 默认行为。

## 7. Interpretation Guide

推荐论文叙述逻辑：

- 如果 Full v0.6c 强于 score top-k：说明 graph-aware selector 比单纯 retrieval score 更有效。
- 如果 adaptive 5-10 强于 fixed-5：说明模型需要额外 evidence。
- 如果 adaptive 5-10 强于 fixed-10：说明不是 evidence 越多越好，而是 adaptive selection 更有效。
- 如果 w/o P2 下降：说明 typed evidence relation 对 fact-checking verifier 有帮助。
- 如果 w/o P3 下降：说明 bridge context 能补充判断所需背景。
- 如果 trace-lite 提升且 truncation 持平：说明轻量 map structure 帮助 verifier evidence use。
- 如果 trace-lite 下降但 truncation 持平：说明即便轻量结构也可能带来 prompt OOD。
- 如果 trace-lite 下降且 truncation 上升：需要增加 `trace_lite_no_atoms` follow-up，判断是否为 token overhead。

## 8. Todo

- [ ] 跑 fixed-5 budget control。
- [ ] 跑 fixed-10 budget control。
- [ ] 实现并跑 w/o P1 selector variant。
- [ ] 实现并跑 w/o P2 selector variant。
- [ ] 实现并跑 w/o P3 selector variant。
- [ ] 跑 score top-k control。
- [ ] 跑 same-set random-order control，多 seed 报均值和方差。
- [ ] 跑 v0.6e trace-lite on v0.6c LoRA。
- [ ] 跑 v0.6e trace-lite on v0.6c FullFT。
- [ ] 汇总 macro-F1 / accuracy / per-class F1 / health metrics。
- [ ] 形成 AAAI paper ablation table 和 narrative。
