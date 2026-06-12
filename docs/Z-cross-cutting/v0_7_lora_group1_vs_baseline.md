# v0.7 LoRA 第一组实验与基准对比

生成日期：2026-06-12

## 实验范围

第一组实验使用 selector `v0_7_budgeted_marginal_chain_adaptive3_10`，训练矩阵如下：

- 数据集：`liar_raw`、`rawfc`
- 模型：`llama31_8b`、`qwen3_4b_2507`
- 训练设置：halfbatch、8 epochs、每 100 step eval/save、early stopping patience 8

当前输出目录中能找到的严格同参基准是 Llama3.1-8B 的两个运行：

- `liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw`
- `rawfc__llama31_8b_lora_halfbatch_ep8_eval100_pat8_rawfc`

Qwen3-4B 只有已完成的历史 `_lora` 运行可作为参考，但它们不是同参基准：

- Qwen 历史基准使用 `gradient_accumulation_steps=8`、`num_train_epochs=5`、`eval_steps=50`、`deepspeed_zero2_bsz1_ga8`
- v0.7 Qwen 运行使用 `gradient_accumulation_steps=4`、`num_train_epochs=8`、`eval_steps=100`、`deepspeed_zero2_bsz1_ga4`

因此，下面的 Qwen 对比只作为参考，不作为严格同参结论。

## 运行完成情况

| 数据集 | 模型 | v0.7 运行目录 | 状态 | 结束 step | 最优验证 selection |
|---|---|---|---:|---:|---:|
| LIAR-RAW | Llama3.1-8B | `liar_raw__llama31_8b__v0_7_bm_lora_halfbatch_ep8_eval100_pat8` | 完成 | 4600 | 0.7219 |
| RAWFC | Llama3.1-8B | `rawfc__llama31_8b__v0_7_bm_lora_halfbatch_ep8_eval100_pat8` | 完成 | 808 | 0.8956 |
| LIAR-RAW | Qwen3-4B | `liar_raw__qwen3_4b_2507__v0_7_bm_lora_halfbatch_ep8_eval100_pat8` | 完成 | 1300 | 0.5639 |
| RAWFC | Qwen3-4B | `rawfc__qwen3_4b_2507__v0_7_bm_lora_halfbatch_ep8_eval100_pat8` | 完成 | 808 | 0.8565 |

## 严格同参基准对比

| 数据集 | 模型 | 验证 selection v0.7 | 验证 selection 基准 | 差值 | 最优 step v0.7/基准 | 测试 selection v0.7 | 测试 selection 基准 | 差值 | 测试 macro-F1 差值 | 测试 acc 差值 | 测试 true-side F1 差值 |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| LIAR-RAW | Llama3.1-8B | 0.7219 | 0.6873 | +0.0346 | 3800 / 1700 | 0.6949 | 0.6792 | +0.0156 | +0.0045 | -0.0024 | +0.0273 |
| RAWFC | Llama3.1-8B | 0.8956 | 0.9438 | -0.0482 | 700 / 700 | 0.9870 | 1.0126 | -0.0256 | -0.0257 | -0.0250 | +0.0078 |

解读：

- LIAR-RAW + Llama3.1-8B 上，v0.7 在验证集和测试集的主指标 `selection_score` 都高于基准。测试集收益主要来自 `true_side_macro_f1`，但 accuracy 略低。
- RAWFC + Llama3.1-8B 上，v0.7 在验证集和测试集的 `selection_score` 都低于基准；测试集 macro-F1 和 accuracy 也下降，但 `true_side_macro_f1` 略高。

## Qwen 参考对比

这些对比不是同参基准对比；它们只是把已完成的 Qwen 历史 `_lora` 运行和当前 v0.7 halfbatch/ep8/eval100 运行放在一起参考。

| 数据集 | 模型 | 验证 selection v0.7 | 验证 selection 参考 | 差值 | 测试 selection v0.7 | 测试 selection 参考 | 差值 | 测试 macro-F1 差值 | 测试 acc 差值 | 测试 true-side F1 差值 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LIAR-RAW | Qwen3-4B | 0.5639 | 0.5391 | +0.0248 | 0.4908 | 0.5131 | -0.0223 | -0.0097 | -0.0016 | -0.0374 |
| RAWFC | Qwen3-4B | 0.8565 | 0.8458 | +0.0107 | 0.9731 | 0.9319 | +0.0412 | +0.0300 | +0.0350 | +0.0224 |

解读：

- Qwen LIAR-RAW：v0.7 的验证集 selection 更高，但测试集 selection、macro-F1 和 true-side F1 都低于历史参考运行。
- Qwen RAWFC：v0.7 的验证集 selection 略高，测试集 selection、macro-F1、accuracy 和 true-side F1 都高于历史参考运行。

## Prompt 统计

证据单元长度按进入 prompt 的 sentence-level evidence 文本做 whitespace-token 统计。Prompt token 使用 build row 中已有的 tokenizer 计数字段 `prompt_token_count`。

| 数据集 | 模型 | 运行 | split | n | Prompt token 均值/p95/最大 | 证据单元数 均值/最小/最大 | 证据单元长度 均值/p95/最大 | 截断率 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LIAR-RAW | Llama3.1-8B | v0.7 | train | 10065 | 367.9/498.0/991 | 3.20/1/10 | 35.7/82.0/796 | 0.01% |
| LIAR-RAW | Llama3.1-8B | v0.7 | val | 1274 | 370.8/501.0/872 | 3.21/1/10 | 36.2/85.0/272 | 0.00% |
| LIAR-RAW | Llama3.1-8B | v0.7 | test | 1251 | 369.9/502.5/995 | 3.27/1/10 | 35.2/83.0/231 | 0.00% |
| LIAR-RAW | Llama3.1-8B | 基准 | train | 10065 | 549.4/797.0/1020 | 7.30/1/10 | 36.7/83.0/796 | 0.30% |
| LIAR-RAW | Llama3.1-8B | 基准 | val | 1274 | 564.8/809.7/1004 | 7.58/1/10 | 37.0/86.0/515 | 0.39% |
| LIAR-RAW | Llama3.1-8B | 基准 | test | 1251 | 559.8/802.5/1010 | 7.55/1/10 | 36.5/84.0/262 | 0.32% |
| RAWFC | Llama3.1-8B | v0.7 | train | 1612 | 318.3/432.0/867 | 3.26/1/10 | 35.7/72.0/441 | 0.00% |
| RAWFC | Llama3.1-8B | v0.7 | val | 200 | 311.9/418.0/647 | 3.21/3/10 | 34.6/73.0/283 | 0.00% |
| RAWFC | Llama3.1-8B | v0.7 | test | 200 | 321.8/437.0/695 | 3.33/3/10 | 35.6/76.0/179 | 0.00% |
| RAWFC | Llama3.1-8B | 基准 | train | 1612 | 564.5/777.0/998 | 8.87/1/10 | 36.5/75.0/441 | 0.12% |
| RAWFC | Llama3.1-8B | 基准 | val | 200 | 565.0/764.3/958 | 8.91/4/10 | 36.6/76.0/372 | 0.00% |
| RAWFC | Llama3.1-8B | 基准 | test | 200 | 581.7/788.2/1016 | 9.04/4/10 | 37.4/77.0/474 | 1.00% |
| LIAR-RAW | Qwen3-4B | v0.7 | train | 10065 | 351.0/484.0/991 | 3.20/1/10 | 35.7/82.0/796 | 0.03% |
| LIAR-RAW | Qwen3-4B | v0.7 | val | 1274 | 354.4/493.7/956 | 3.21/1/10 | 36.2/85.0/272 | 0.00% |
| LIAR-RAW | Qwen3-4B | v0.7 | test | 1251 | 353.5/490.5/1014 | 3.27/1/10 | 35.2/83.0/231 | 0.00% |
| LIAR-RAW | Qwen3-4B | 参考 | train | 10065 | 538.4/790.0/1016 | 7.29/1/10 | 36.7/83.0/796 | 0.35% |
| LIAR-RAW | Qwen3-4B | 参考 | val | 1274 | 554.6/808.0/997 | 7.58/1/10 | 37.0/86.0/515 | 0.39% |
| LIAR-RAW | Qwen3-4B | 参考 | test | 1251 | 550.0/798.0/1016 | 7.55/1/10 | 36.5/84.0/262 | 0.32% |
| RAWFC | Qwen3-4B | v0.7 | train | 1612 | 300.3/417.4/851 | 3.26/1/10 | 35.7/72.0/441 | 0.00% |
| RAWFC | Qwen3-4B | v0.7 | val | 200 | 293.4/403.0/663 | 3.21/3/10 | 34.6/73.0/283 | 0.00% |
| RAWFC | Qwen3-4B | v0.7 | test | 200 | 304.0/421.6/680 | 3.33/3/10 | 35.6/76.0/179 | 0.00% |
| RAWFC | Qwen3-4B | 参考 | train | 1612 | 552.2/768.4/1008 | 8.87/1/10 | 36.5/75.0/441 | 0.12% |
| RAWFC | Qwen3-4B | 参考 | val | 200 | 552.5/763.1/976 | 8.91/4/10 | 36.6/76.0/372 | 0.00% |
| RAWFC | Qwen3-4B | 参考 | test | 200 | 570.5/786.5/995 | 9.04/4/10 | 37.4/77.0/474 | 1.00% |

## Prompt 统计差值

差值均为 v0.7 减去基准或参考运行的 split 均值。Qwen 行仍然只作为参考，因为已完成的历史 Qwen 运行使用了不同训练参数。

| 数据集 | 模型 | 对比对象 | split | Prompt token 均值差值 | 证据单元数均值差值 | 证据单元长度均值差值 |
|---|---|---|---:|---:|---:|---:|
| LIAR-RAW | Llama3.1-8B | 基准 | train | -181.5 | -4.09 | -1.0 |
| LIAR-RAW | Llama3.1-8B | 基准 | val | -194.0 | -4.37 | -0.8 |
| LIAR-RAW | Llama3.1-8B | 基准 | test | -189.9 | -4.28 | -1.3 |
| RAWFC | Llama3.1-8B | 基准 | train | -246.2 | -5.62 | -0.8 |
| RAWFC | Llama3.1-8B | 基准 | val | -253.2 | -5.70 | -2.0 |
| RAWFC | Llama3.1-8B | 基准 | test | -259.9 | -5.71 | -1.8 |
| LIAR-RAW | Qwen3-4B | 参考 | train | -187.3 | -4.09 | -1.0 |
| LIAR-RAW | Qwen3-4B | 参考 | val | -200.2 | -4.37 | -0.8 |
| LIAR-RAW | Qwen3-4B | 参考 | test | -196.5 | -4.28 | -1.3 |
| RAWFC | Qwen3-4B | 参考 | train | -252.0 | -5.62 | -0.8 |
| RAWFC | Qwen3-4B | 参考 | val | -259.1 | -5.70 | -2.0 |
| RAWFC | Qwen3-4B | 参考 | test | -266.4 | -5.71 | -1.8 |

Prompt 规模解读：

- v0.7 使用的证据单元明显少于基准或参考 selector：LIAR-RAW 平均少约 4.1 个证据单元，RAWFC 平均少约 5.6 到 5.7 个证据单元。
- 单条证据单元长度基本不变；prompt token 的减少主要来自 sentence-level 证据单元数量减少，而不是证据句本身变短。
- v0.7 基本消除了截断；相比之下，RAWFC 的基准或参考 test prompt 仍有 1.00% 截断率。

## 备注

- 主指标为 `selection_score`；测试集辅助指标为 `macro_f1`、`accuracy` 和 `true_side_macro_f1`。
- 测试指标来自 `eval/test/best/label_token/metrics.json`。
- Prompt 统计从各运行解析后的 `build_{split}.jsonl` 输入重新计算；对于运行目录下没有本地 `build/` 的参考运行，输入路径从 `train.resolved.yaml` 解析。
- LIAR-RAW Llama 基准还有额外的 `label_token_logit_adjust_tau*` 测试 sweep；主表没有纳入这些 sweep，因为当前 v0.7 测试结果只有共享的 `label_token` 输出。
- Selector trace 确认严格基准使用 `sentence_rule_step_adaptive5_10`，v0.7 运行使用 `v0_7_budgeted_marginal_chain_adaptive3_10`。

## 输出来源

- v0.7 Llama LIAR：`outputs/sentence_trace_method/liar_raw__llama31_8b__v0_7_bm_lora_halfbatch_ep8_eval100_pat8`
- Llama LIAR 基准：`outputs/sentence_trace_method/liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw`
- v0.7 Llama RAWFC：`outputs/sentence_trace_method/rawfc__llama31_8b__v0_7_bm_lora_halfbatch_ep8_eval100_pat8`
- Llama RAWFC 基准：`outputs/sentence_trace_method/rawfc__llama31_8b_lora_halfbatch_ep8_eval100_pat8_rawfc`
- v0.7 Qwen LIAR：`outputs/sentence_trace_method/liar_raw__qwen3_4b_2507__v0_7_bm_lora_halfbatch_ep8_eval100_pat8`
- Qwen LIAR 参考：`outputs/sentence_trace_method/liar_raw__qwen3_4b_2507_lora`
- v0.7 Qwen RAWFC：`outputs/sentence_trace_method/rawfc__qwen3_4b_2507__v0_7_bm_lora_halfbatch_ep8_eval100_pat8`
- Qwen RAWFC 参考：`outputs/sentence_trace_method/rawfc__qwen3_4b_2507_lora`
