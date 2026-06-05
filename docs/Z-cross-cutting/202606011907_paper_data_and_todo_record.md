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

### 2.1 RAWFC Result Snapshot

RAWFC 于 2026-06-02 跑通 v0.6c 适配版：

```text
RAWFC original 3-way labels: false / half / true
selector: v0_6c_rule_step_adaptive5_10
label_schema: rawfc3
prompt style: plain
evidence setting: RAWFC closed raw evidence
```

该结果来自 LIAR-RAW v0.6c 主实验的同样设置：同一 rule-step adaptive evidence-chain graph、同一 plain verifier prompt、同一训练/推理 wrapper 逻辑和基本超参；仅替换为 RAWFC loader、`rawfc3` 标签集合和 RAWFC 闭集 evidence。当前尚未针对 RAWFC 专门调参，因此仍可能有调整空间。

当前 test 结果如下。FullFT label-token logits 与训练期 validation eval 口径一致，可作为 RAWFC 当前主引用；LoRA 和生成式 native inference 作为辅助记录保留。

| dataset | selector | train mode | split | accuracy | macro-P | macro-R | macro-F1 | false F1 | half F1 | true F1 | note |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| RAWFC | v0.6c | LoRA | test | 58.00 | 59.59 | 58.04 | 55.10 | 63.24 | 36.17 | 65.88 | LIAR-RAW 同设置，未做 RAWFC 专门调参 |
| RAWFC | v0.6c | FullFT | test | 60.50 | 63.52 | 60.47 | 60.95 | 62.07 | 56.25 | 64.52 | label-token logits，validation 同口径，未做 RAWFC 专门调参 |

辅助记录：

- LoRA test metrics: `outputs/runs/rawfc_v0_6c_selector_trace_full_pipeline/v0_6c_rawfc3_rule_step_adaptive5_10_best/infer/test/best/7531fe25d0da/api/metrics.json`
- LoRA test confusion matrix: false/half/true rows vs false/half/true/parse_error columns = `[[43,7,16,0],[19,17,31,0],[8,3,56,0]]`。
- FullFT label-token test metrics: `outputs/selector_trace_verifier/rawfc_v0_6c/v0_6c_rawfc3_rule_step_adaptive5_10_fullft/train/eval/test/best/label_token/metrics.json`
- FullFT label-token test confusion matrix: false/half/true rows vs false/half/true/parse_error columns = `[[36,25,5,0],[10,45,12,0],[4,23,40,0]]`。
- FullFT 生成式 native test metrics: `outputs/selector_trace_verifier/rawfc_v0_6c/v0_6c_rawfc3_rule_step_adaptive5_10_fullft/train/eval/test/best/metrics.rawfc3_corrected.json`，accuracy `58.50`、macro-P/R/F1 `58.48 / 58.56 / 58.39`。
- FullFT validation best checkpoint 对应 step-200 eval：accuracy `61.06`、macro-P/R/F1 `62.60 / 61.01 / 61.39`。
- FullFT 生成式 native test 由 `sft.infer` 生成；旧版 native eval 未把 `rawfc3` schema 透传到 metrics，原始 `metrics.json` 会显示 LIAR 六类标签。这里采用同一 `test_predictions.jsonl` 重新计算的 `rawfc3` corrected metrics，代码侧已修复 schema 透传。
- Evidence-map teacher 已真实调用 DeepSeek，不是 mock；train/test 共有 4 条因 `finish_reason=length` 走 `fallback_missing_annotation`，后续可通过更高 `MAX_TOKENS` 或重跑缺失 annotation 进一步清理。

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

### 5.1 Literature Comparison Baselines

推荐论文主表口径：

```text
LIAR-RAW, 6-way veracity classification, raw reports / closed evidence setting, macro-P / macro-R / macro-F1.
```

当前 v0.6c 主结果应写成：

```text
v0.6c: LIAR-RAW macro-P / macro-R / macro-F1 = 39.74 / 34.86 / 35.45
v0.6c: RAWFC macro-P / macro-R / macro-F1 = 63.52 / 60.47 / 60.95
```

注意：`macro_f1` 是 per-class F1 的 macro average，不是由 macro-P 和 macro-R 再调和得到。

RAWFC 数字来自同 LIAR-RAW v0.6c 设置的 FullFT label-token test inference，和训练期 validation eval 口径一致，尚未针对 RAWFC 做专门调参；LoRA test 结果为 `59.59 / 58.04 / 55.10`，FullFT 生成式 native test 结果为 `58.48 / 58.56 / 58.39`，见 2.1 辅助表。

主对比表建议只放 raw-report / near raw-report setting。`best variant` 表示同一论文中按数据集选该方法公开报告的最好变体；如果变体不同，需要在 caption 或脚注中说明。

| method | setting note | LIAR-RAW / LIAR P/R/F1 | RAWFC P/R/F1 | comparison handling |
| --- | --- | ---: | ---: | --- |
| CofCED | dataset paper; raw reports | 29.48 / 29.55 / 28.93 | 52.99 / 50.99 / 51.07 | main baseline |
| FactLLaMAKnow | LIAR-family; LLaMA LoRA + external knowledge | 32.46 / 32.05 / 30.44 | 56.11 / 55.50 / 55.65 | near-comparable, note LIAR naming |
| L-Defense | raw reports + competing wisdom; best variant per dataset | 31.63 / 31.71 / 31.40 | 61.72 / 61.01 / 61.20 | main baseline |
| G-Defense | raw reports + graph-enhanced defense; best variant per dataset | 34.17 / 32.37 / 32.49 | 66.29 / 65.49 / 65.50 | main baseline, note variant/backbone |
| DeReC-qwen | dense retrieval + DeBERTa classifier | 35.94 / 32.24 / 33.13 | 65.58 / 64.56 / 64.60 | main baseline |
| FFRR(d+q) | feedback-trained retrieval + reader | 34.50 / 32.60 / 33.50 | 56.50 / 57.40 / 57.00 | main baseline |
| DelphiAgent GPT-4o | training-free multi-agent fact-checking | 31.33 / 28.36 / 28.36 | 68.05 / 68.03 / 68.04 | report separately or main-with-LLM note |
| **v0.6c (Ours)** | rule-step adaptive evidence-chain graph | 39.74 / 34.86 / 35.45 | 63.52 / 60.47 / 60.95 | our main method; RAWFC uses LIAR-RAW same setting, not RAWFC-tuned |
| KG-CRAFT Llama 3.3 | KG + contrastive questions + strong LLM | 77.38 / 70.67 / 73.87 | 81.63 / 81.53 / 81.58 | include as strong-LLM upper reference |

Non-main higher-score or different-evidence-setting methods:

| method | why not direct main-table comparison | LIAR-RAW / LIAR P/R/F1 | RAWFC P/R/F1 | recommended handling |
| --- | --- | ---: | ---: | --- |
| HiSS | GPT-3.5/text-davinci-003 + web/search; paper table uses LIAR + RAWFC | 46.80 / 31.30 / 37.50 | 53.40 / 54.40 / 53.90 | put in external-search baselines |
| RAFTS | external Wikipedia/document retrieval + contrastive arguments; paper table uses LIAR + RAWFC | 47.10 / 37.90 / 42.00 | 62.80 / 52.60 / 57.30 | put in external-retrieval baselines |
| AFEV / Fact in Fragments | atomic fact extraction + reranking + dynamic demonstrations; table uses LIAR + RAWFC, not strict LIAR-RAW | 48.20 / 40.30 / 43.90 | 63.30 / 57.60 / 60.20 | put in external/LIAR-family baselines |
| RAV | uses author-written explanations / gold evidence on LIAR-RAW and RAWFC | F1 29.33 | F1 67.53 | gold-evidence table only |
| Entailed Opinion / TBE-3 | uses gold evidence / entailed justifications | 55 / 54 / 54 | 88 / 88 / 88 | gold-evidence upper reference |
| LQ-FJS | reports relative gains only in accessible abstract; video/multimodal system | +7.2 F1 over SOTA | +4.5 F1 over SOTA | related work or unverifiable-relative table, not main baseline |

Source pointers for this literature table:

- CofCED: https://aclanthology.org/2022.coling-1.230.pdf
- FactLLaMA: https://arxiv.org/pdf/2309.00240
- HiSS: https://aclanthology.org/2023.ijcnlp-main.64.pdf
- RAFTS: https://aclanthology.org/2024.acl-long.556.pdf
- FFRR: https://aclanthology.org/2024.lrec-main.1209.pdf
- L-Defense: https://openreview.net/pdf?id=WurgtxoLt3
- G-Defense: https://arxiv.org/pdf/2604.06666
- DeReC: https://aclanthology.org/2025.ldk-1.26.pdf
- AFEV: https://arxiv.org/pdf/2506.07446
- RAV: https://aclanthology.org/2025.emnlp-industry.167.pdf
- Entailed Opinion / TBE-3: https://arxiv.org/pdf/2505.15050
- KG-CRAFT: https://aclanthology.org/2026.eacl-long.302.pdf
- LQ-FJS: https://www.sciencedirect.com/science/article/pii/S0169023X25001028

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

## 9. LLM Backbone Migration Status and Results

目标：基于当前 v0.6c 实现，对 verifier backbone 做 9B 以下模型对比。替换 backbone 时需要同时更新 `build.prompt.model_name_or_path` 和 `train.model_name_or_path`，并先确认 label-token CE 的 `Label:` 后标签字母单 token 约束。RAWFC `rawfc3` 使用 `A-C`；LIAR-RAW 6-way 使用 `A-F`。

统一口径：

- dataset / selector: RAWFC `rawfc3` + `v0_6c_rule_step_adaptive5_10`
- output root: `outputs/selector_trace_verifier/rawfc_v0_6c_eval25_backbone`
- 指标来源：label-token logits test eval，`eval/test/best/label_token/metrics.json` 或旧 layout 下的 `train/eval/test/best/label_token/metrics.json`
- 完整 test 定义：`num_samples=200` 且 `parse_error_rate=0.0`
- `qwen25_7b` LoRA 只有旧 API/generative test metric，FullFT 为 eval25 label-token metric；表内保留星号说明。
- `ministral3_8b` 的 FullFT 汇总采用 `outputs/selector_trace_verifier/rawfc_v0_6c_eval25_backbone/v0_6c_rawfc3_rule_step_adaptive5_10_eval25_ministral3_8b_fullft_mm_text_effective`。

### 9.1 Current Status Summary

按 FullFT validation macro-F1 排序。数值单位为百分比；test 单元格格式为 `Acc / Macro-F1`。

| rank by FullFT val F1 | model | size | LoRA val F1@step | LoRA test Acc/F1 | FullFT val F1@step | FullFT test Acc/F1 | best test F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Meta-Llama-3.1-8B-Instruct | 8B | 61.59@150 | 60.50 / 60.77 | 68.32@100 | 67.00 / 67.23 | **67.23 FullFT** |
| 2 | Ministral-3-8B-Instruct-2512 | 8.4B | 68.81@225 | 60.50 / 60.91 | 64.74@25 | 62.50 / 61.41 | **61.41 FullFT** |
| 3 | Qwen3-8B | 8B | 56.87@125 | 56.00 / 55.79 | 63.75@175 | 62.00 / 62.27 | **62.27 FullFT** |
| 4 | Gemma-4-E4B-it | 8B | 59.46@200 | 59.50 / 59.19 | 62.85@175 | 65.50 / 65.78 | **65.78 FullFT** |
| 5 | Qwen3-4B-Instruct-2507 | 4B | 60.69@225 | 60.50 / 59.96 | 62.42@175 | 67.00 / 67.11 | **67.11 FullFT** |
| 6 | Qwen2.5-7B-Instruct | 7B | 51.70@200 | 58.00 / 55.10* | 60.53@175 | 62.50 / 62.94 | **62.94 FullFT** |
| 7 | Phi-4-mini-instruct | 3.8B | 56.68@250 | 57.50 / 56.99 | 60.26@200 | 62.50 / 63.06 | **63.06 FullFT** |
| 8 | DeepSeek-R1-Distill-Qwen-7B | 7B | 49.01@225 | 55.50 / 54.21 | 58.77@375 | 60.00 / 60.04 | **60.04 FullFT** |
| 9 | Qwen2.5-3B-Instruct | 3B | 52.96@150 | 58.00 / 56.98 | 57.50@150 | 58.50 / 57.55 | **57.55 FullFT** |
| 10 | Qwen3-1.7B | 1.7B | 52.48@225 | 52.00 / 51.36 | 55.26@250 | 54.50 / 54.82 | **54.82 FullFT** |
| 11 | Qwen2.5-1.5B-Instruct | 1.5B | 47.95@250 | 49.50 / 47.53 | 54.90@175 | 57.00 / 57.33 | **57.33 FullFT** |

`*` Qwen2.5-7B LoRA 为旧 API/generative metric，不是 label-token logits 口径。

### 9.2 Completed Full-Test Metrics

下表只列 `num_samples=200` 的 test 结果，数值为百分比；所有已完成 full-test 行 `parse_error_rate=0.0`。单元格格式为 `Acc; Macro-P/R/F1; false/half/true F1`。排序与 9.1 一致。

| rank by FullFT val F1 | backbone case | LoRA result | FullFT result | status |
| ---: | --- | --- | --- | --- |
| 1 | `llama31_8b` | 60.50; 61.94 / 60.55 / 60.77; 65.71 / 51.06 / 65.55 | 67.00; 67.71 / 67.02 / 67.23; 69.12 / 57.97 / 74.60 | 完成 |
| 2 | `ministral3_8b` | 60.50; 62.12 / 60.52 / 60.91; 65.12 / 53.69 / 63.93 | 62.50; 62.11 / 62.54 / 61.41; 70.23 / 45.61 / 68.39 | 完成；FullFT 使用 `fullft_mm_text_effective` 目录 |
| 3 | `qwen3_8b` | 56.00; 55.75 / 56.04 / 55.79; 60.43 / 47.24 / 59.70 | 62.00; 62.66 / 62.02 / 62.27; 67.18 / 51.43 / 68.22 | 完成 |
| 4 | `gemma4_e4b` | 59.50; 59.18 / 59.54 / 59.19; 62.41 / 48.00 / 67.16 | 65.50; 66.92 / 65.50 / 65.78; 69.35 / 61.33 / 66.67 | 完成 |
| 5 | `qwen3_4b_2507` | 60.50; 59.98 / 60.56 / 59.96; 66.67 / 47.54 / 65.67 | 67.00; 67.38 / 67.04 / 67.11; 72.06 / 58.39 / 70.87 | 完成 |
| 6 | `qwen25_7b` | 58.00; 59.59 / 58.04 / 55.10; 63.24 / 36.17 / 65.88 | 62.50; 65.18 / 62.48 / 62.94; 64.46 / 58.23 / 66.12 | 完成；LoRA 为旧 API/generative metric |
| 7 | `phi4_mini` | 57.50; 56.81 / 57.54 / 56.99; 62.32 / 43.90 / 64.75 | 62.50; 64.62 / 62.52 / 63.06; 68.75 / 54.30 / 66.12 | 完成 |
| 8 | `dsr1_qwen7b` | 55.50; 55.21 / 55.50 / 54.21; 57.81 / 41.07 / 63.75 | 60.00; 62.59 / 59.94 / 60.04; 57.66 / 57.86 / 64.62 | 完成 |
| 9 | `qwen25_3b` | 58.00; 57.37 / 58.03 / 56.98; 61.31 / 43.86 / 65.77 | 58.50; 57.77 / 58.50 / 57.55; 58.02 / 45.76 / 68.87 | 完成 |
| 10 | `qwen3_17b` | 52.00; 51.16 / 52.03 / 51.36; 55.07 / 38.02 / 60.99 | 54.50; 57.32 / 54.48 / 54.82; 53.66 / 52.17 / 58.62 | 完成 |
| 11 | `qwen25_15b` | 49.50; 47.61 / 49.56 / 47.53; 54.42 / 28.30 / 59.86 | 57.00; 57.85 / 57.02 / 57.33; 63.49 / 47.89 / 60.61 | 完成 |

当前观察：

- 按 FullFT val F1 排序时，`ministral3_8b` 排名第 2，但 test F1 只有 `61.41`，说明该模型 val/test gap 明显，不宜只按 val 排序解释泛化能力。
- 按 test F1 看，FullFT top tier 是 `llama31_8b`、`qwen3_4b_2507` 和 `gemma4_e4b`；其中 `llama31_8b` 与 `qwen3_4b_2507` 仍只差 `0.12` F1。
- `gemma4_e4b` FullFT 补齐后达到 `65.78` test macro-F1，是 transfer 组里仅次于 Llama/Qwen3-4B 的强结果。
- `qwen25_7b` FullFT test F1 为 `62.94`，低于 `gemma4_e4b`、`qwen3_4b_2507` 和 `llama31_8b`，但高于 `qwen3_8b` 和 `dsr1_qwen7b`。

### 9.3 Significance Check for Close Models

方法：对同一 200 条 RAWFC test 样本做 paired bootstrap，固定 `sample_idx` 对齐两个模型预测，有放回重采样 20,000 次。下表的差值均为 `A - B`，单位为 percentage points；CI 为 percentile 95% CI，`p_boot` 为双侧 bootstrap tail probability。该检查用于判断相近模型是否足以支持排序性结论，不替代多 seed 训练稳定性分析。当前表尚未补算 `gemma4_e4b` FullFT 和 `ministral3_8b` mm_text_effective FullFT。

| comparison | mode | Δ Macro-F1, 95% CI, p_boot | Δ Acc, 95% CI, p_boot | A-only / B-only correct | interpretation |
| --- | --- | --- | --- | --- | --- |
| `llama31_8b - qwen3_4b_2507` | FullFT | +0.12 `[-6.96, +7.42]`, p=0.9746 | +0.00 `[-7.00, +7.00]`, p=1.0000 | 26 / 26 | 无显著差异；只能写成 tied top tier。 |
| `phi4_mini - qwen3_8b` | FullFT | +0.78 `[-5.17, +6.56]`, p=0.7899 | +0.50 `[-5.50, +6.50]`, p=0.9309 | 19 / 18 | 无显著差异；Phi 略高但证据不足。 |
| `qwen25_3b - qwen25_15b` | FullFT | +0.22 `[-7.00, +7.43]`, p=0.9585 | +1.50 `[-6.00, +8.50]`, p=0.7523 | 29 / 26 | 无显著差异；3B 与 1.5B 不应做强排序。 |
| `llama31_8b - qwen3_4b_2507` | LoRA | +0.82 `[-6.34, +8.02]`, p=0.8318 | +0.00 `[-7.00, +7.00]`, p=1.0000 | 26 / 26 | 无显著差异；LoRA 下两者也基本持平。 |
| `phi4_mini - qwen25_3b` | LoRA | +0.01 `[-6.36, +6.27]`, p=0.9924 | -0.50 `[-6.50, +5.50]`, p=0.9414 | 19 / 20 | 无显著差异；几乎完全持平。 |

论文表述建议：

- `llama31_8b` 和 `qwen3_4b_2507` 在 FullFT 下应写作 top-tier tie，而不是显著胜出关系。
- 对 `phi4_mini` vs `qwen3_8b`、`qwen25_3b` vs `qwen25_15b` 这类小差值，只报告点估计和 CI；避免写“模型 A 优于模型 B”。
- 当前 test split 只有 200 条，bootstrap CI 很宽；若需要支持 backbone ranking，后续应补多 seed 或更大评测集。

### 9.4 Smoke / In-Progress Records

| backbone case | train mode | status | available metric | notes |
| --- | --- | --- | --- | --- |
| `gemma4_e4b` | LoRA smoke32 | superseded / smoke | n=32 Acc/F1 = 56.25 / 48.23 | 已有正式 LoRA full test，smoke 仅保留为适配记录。 |
| `gemma4_e4b` | FullFT smoke32 | superseded / smoke | N/A | 已有正式 FullFT full test，smoke 仅保留为适配记录。 |
| `ministral3_8b` | LoRA smoke32 | superseded / smoke | n=32 Acc/F1 = 37.50 / 37.37 | 已有正式 LoRA full test，smoke 不可与 200-sample full test 横比。 |
| `ministral3_8b` | FullFT smoke32 | superseded / smoke | N/A | 已有正式 FullFT full test；主表采用 mm_text_effective 目录。 |

### 9.5 Adaptation Difficulty Groups

适配难度分组：

| group | model | local status | difficulty | notes |
| --- | --- | --- | --- | --- |
| A. 基本 drop-in | `/data/models/Qwen2.5-3B-Instruct` | 本地已有 / 已完成 | 低 | Qwen2 架构；label-token 检查通过；LoRA target 可直接复用。 |
| A. 基本 drop-in | `/data/models/Qwen2.5-1.5B-Instruct` | 本地已有 / 已完成 | 低 | Qwen2 架构；适合做小模型下界。 |
| A. 基本 drop-in | `/data/models/Qwen3-1.7B` | 本地已有 / 已完成 | 低 | `Qwen3ForCausalLM`；label-token 检查通过；需要控制 thinking / non-thinking 输出风格。 |
| A. 基本 drop-in | `/data/models/Qwen3-8B` | 本地已有 / 已完成 | 低 | `Qwen3ForCausalLM`；自然的强小模型对照。 |
| A. 基本 drop-in | `/data/models/DeepSeek-R1-Distill-Qwen-7B` | 本地已有 / 已完成 | 低 | 底座是 Qwen2；主要风险是 reasoning-distill 风格可能影响 label-only 输出。 |
| B. 下载后大概率 drop-in | `/data/models/Qwen3-4B-Instruct-2507` | 本地已有 / 已完成 | 低-中 | 4B 非 thinking instruct 版本；已纳入 phase7 backbone migration。 |
| C. 中等适配 | `/data/models/Meta-Llama-3.1-8B-Instruct` | 本地已有 / 已完成 | 中 | `LlamaForCausalLM`；label-token 检查通过；LoRA + FullFT full test 已完成。 |
| C. 中等适配 | `/data/models/Phi-4-mini-instruct` | 本地已有 / 已完成 | 中 | 3.8B/128K；LoRA + FullFT full test 已完成。 |
| D. 高适配成本 | `google/gemma-4-E4B-it` -> `/data/models/gemma-4-E4B-it` | 本地已有 / 已完成 | 中-高 | 4.5B effective / 8B with embeddings；LoRA + FullFT full test 已完成。 |
| D. 高适配成本 | `mistralai/Ministral-3-8B-Instruct-2512` -> `/data/models/Ministral-3-8B-Instruct-2512` | 本地已有 / 已完成 | 高 | 新 Mistral 3 / 多模态路线；LoRA + FullFT full test 已完成，FullFT 主表采用 mm_text_effective。 |

建议执行顺序：

1. 第一批已完成：`Qwen2.5-1.5B-Instruct`、`Qwen2.5-3B-Instruct`、`Qwen3-1.7B`、`Qwen3-8B`、`DeepSeek-R1-Distill-Qwen-7B`。
2. 第二批已完成：`Qwen3-4B-Instruct-2507`、`Meta-Llama-3.1-8B-Instruct`、`Phi-4-mini-instruct`。
3. 第三批已完成：`gemma-4-E4B-it`、`Ministral-3-8B-Instruct-2512`；作为新架构适配实验，保留 smoke 和 text-effective 适配记录。

待做 smoke checks：

- [x] 已完成 A/B/C 组 tokenizer `Label:` 后单 token 检查和正式 full-test 结果汇总。
- [x] `gemma4_e4b` / `ministral3_8b` 已有 label-token CE meta，说明 rawfc3 标签 token 检查可走通。
- [x] 完成 `gemma4_e4b` FullFT 正式训练和 200-sample test eval。
- [x] 完成 `ministral3_8b` LoRA / FullFT 正式训练和 200-sample test eval。
