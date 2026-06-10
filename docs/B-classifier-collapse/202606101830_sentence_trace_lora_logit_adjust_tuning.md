# Sentence Trace LoRA 调参与 Logit 校准记录

日期：2026-06-10

## 结论

当前 LIAR-RAW + Llama-3.1-8B sentence-trace LoRA 的推荐组合为：

- 训练 run：`outputs/sentence_trace_method/liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw`
- checkpoint：`train/best`，对应训练期 step 1700
- 训练策略：LoRA r=16/alpha=32/dropout=0.05，ZeRO-2，micro batch 1，gradient accumulation 4
- 训练上限：8 epochs，实际 early stop 于 step 2500
- eval/save：每 100 update steps 一次
- early stopping patience：8 次 eval
- LIAR class weights：`pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8`
- 推理/复评策略：label-token logits + train-prior logit adjustment，`tau=0.5`

该组合目前作为 LIAR-RAW + Llama sentence-trace LoRA 的最优工作组合。`tau=0.5` 是按 val selection score 选择；`tau=0.75` 在 test 上更高，但 val 已出现过校准迹象，因此不作为默认。

## 背景

原始 LoRA run 为：

`outputs/sentence_trace_method/liar_raw__llama31_8b_lora`

它能稳定完成训练，但 LIAR 六分类存在明显边界塌缩：`barely-true`、`true` 和 `pants-fire` 输出不足，模型倾向预测 `false`、`half-true`、`mostly-true` 等更常见或邻近类别。

第一轮调参目标是缓解小类和极端类塌缩，同时避免 ZeRO-3 checkpoint recompute 问题。因此 LoRA path 切到 ZeRO-2，并减半有效 batch、增加训练上限和 patience。

## 调参过程

### 1. 基线 LoRA

路径：

`outputs/sentence_trace_method/liar_raw__llama31_8b_lora`

关键设置：

- `gradient_accumulation_steps=8`
- `num_train_epochs=5`
- `update_steps_per_epoch=315`
- `max_train_steps=1575`
- 无 LIAR class weights
- 无 label-token logit adjustment

test best 指标：

| setting | acc | macro-F1 | true-side macro-F1 | selection |
|---|---:|---:|---:|---:|
| old LoRA | 0.3030 | 0.2985 | 0.2970 | 0.6742 |

关键 per-class F1：

| label | F1 | recall |
|---|---:|---:|
| pants-fire | 0.4267 | 0.3721 |
| barely-true | 0.1231 | 0.0762 |
| true | 0.2719 | 0.2195 |

### 2. 半 batch + 8 epoch + LIAR class weights

路径：

`outputs/sentence_trace_method/liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw`

关键设置：

- `deepspeed_config=configs/deepspeed_zero2_bsz1_ga4.json`
- `gradient_accumulation_steps=4`
- `num_train_epochs=8`
- `eval_steps=100`
- `save_steps=100`
- `early_stopping_patience=8`
- LIAR class weights：`pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8`

训练实际情况：

- `update_steps_per_epoch=629`
- `max_train_steps=5032`
- best score 出现在训练期 step 1700
- early stop 于 step 2500
- `train/best` 与 `train/checkpoint-1700` adapter hash 一致

未校准 test best 指标：

| setting | acc | macro-F1 | true-side macro-F1 | selection |
|---|---:|---:|---:|---:|
| old LoRA | 0.3030 | 0.2985 | 0.2970 | 0.6742 |
| halfbatch+weights | 0.3046 | 0.3070 | 0.2850 | 0.6792 |

改善集中在 `barely-true` 和 `half-true`，但 `true` 与 `pants-fire` 未改善：

| label | old F1 | new F1 | old recall | new recall |
|---|---:|---:|---:|---:|
| pants-fire | 0.4267 | 0.4118 | 0.3721 | 0.3256 |
| barely-true | 0.1231 | 0.2102 | 0.0762 | 0.1571 |
| true | 0.2719 | 0.2469 | 0.2195 | 0.1951 |

未校准预测分布仍偏中间类：

| label | gold share | pred share |
|---|---:|---:|
| pants-fire | 6.9% | 4.0% |
| false | 19.9% | 27.1% |
| barely-true | 16.8% | 8.3% |
| half-true | 21.0% | 28.5% |
| mostly-true | 19.0% | 22.5% |
| true | 16.4% | 9.5% |

结论：继续增加训练步数不是主问题。step 1700 后 selection/macro 没有稳定提升，最终 step 2500 early stop。问题更像 label prior / 决策边界校准。

### 3. 接入 label-token logit adjustment

已将旧 generative SFT 路径中的 logit adjustment 接到 label-token CE 路径：

- `src/sft/logit_adjust.py`
- `src/sft/label_token_trainer.py`
- `src/sft/label_token_infer.py`
- `scripts/sentence_trace_method/run_lora_label_token_logit_adjust_eval_only.sh`

校准公式：

```text
adjusted_logit = label_logit - tau * log(train_prior)
```

当前 LIAR-RAW train prior 对应的 `log_priors`：

| label | prior | log prior | tau=1 bias |
|---|---:|---:|---:|
| pants-fire | 0.0807 | -2.5173 | +2.5173 |
| false | 0.1945 | -1.6371 | +1.6371 |
| barely-true | 0.1601 | -1.8322 | +1.8322 |
| half-true | 0.2074 | -1.5733 | +1.5733 |
| mostly-true | 0.1937 | -1.6412 | +1.6412 |
| true | 0.1636 | -1.8101 | +1.8101 |

该校准只作用于 eval / infer 决策，不改变训练 loss。`eval_loss`、`eval_ce_loss` 和 `eval_ordinal_loss` 仍基于 raw label logits。

## Eval-only Tau Sweep

命令入口：

```bash
FORCE_EVAL=true TAUS=0.25,0.5,0.75,1.0 \
  bash scripts/sentence_trace_method/run_lora_label_token_logit_adjust_eval_only.sh
```

默认脚本已设为 `TAUS=0.5`：

```bash
bash scripts/sentence_trace_method/run_lora_label_token_logit_adjust_eval_only.sh
```

输出目录：

```text
outputs/sentence_trace_method/liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw/eval/{val,test}/best/label_token_logit_adjust_tau*
```

### Val 结果

| setting | acc | macro-F1 | true-side macro-F1 | selection |
|---|---:|---:|---:|---:|
| base | 0.3108 | 0.3023 | 0.2921 | 0.6780 |
| tau=0.25 | 0.3061 | 0.3073 | 0.3103 | 0.6900 |
| tau=0.5 | 0.3046 | 0.3100 | 0.3089 | 0.6911 |
| tau=0.75 | 0.2936 | 0.3013 | 0.3146 | 0.6832 |
| tau=1.0 | 0.2881 | 0.2918 | 0.3146 | 0.6724 |

按 val selection，`tau=0.5` 最好。

### Test 结果

| setting | acc | macro-F1 | true-side macro-F1 | selection |
|---|---:|---:|---:|---:|
| base | 0.3046 | 0.3070 | 0.2850 | 0.6792 |
| tau=0.25 | 0.3110 | 0.3218 | 0.3297 | 0.7157 |
| tau=0.5 | 0.3102 | 0.3238 | 0.3297 | 0.7172 |
| tau=0.75 | 0.3125 | 0.3241 | 0.3427 | 0.7224 |
| tau=1.0 | 0.3038 | 0.3094 | 0.3431 | 0.7063 |

test 上 `tau=0.75` selection 最高，但它在 val 上 macro-F1 和 accuracy 已下降明显，不作为默认。

### Tau=0.5 的类别影响

相对未校准 base，test 上 `tau=0.5` 的变化：

| label | F1 delta | recall delta |
|---|---:|---:|
| pants-fire | +0.0411 | +0.0930 |
| false | -0.0433 | -0.0964 |
| barely-true | +0.0398 | +0.0714 |
| half-true | -0.0263 | -0.1027 |
| mostly-true | +0.0143 | +0.0546 |
| true | +0.0752 | +0.1073 |

test 预测分布从：

```text
base:    pants-fire=4.0%, false=27.1%, barely-true=8.3%, half-true=28.5%, mostly-true=22.5%, true=9.5%
tau=0.5: pants-fire=5.8%, false=21.0%, barely-true=13.9%, half-true=17.9%, mostly-true=26.9%, true=14.4%
```

`tau=0.5` 仍不完全匹配 gold distribution，但已显著缓解 `pants-fire / barely-true / true` 输出不足，同时没有像 `tau=0.75/1.0` 那样过度牺牲 `false`。

## 当前最优组合记录

当前使用以下组合，作为 LIAR-RAW + Llama sentence-trace LoRA 的最优引用口径：

```text
case: outputs/sentence_trace_method/liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw
checkpoint: best
training: ZeRO-2, micro_batch=1, gradient_accumulation_steps=4
epochs: 8 max, early stopped at step 2500
best checkpoint: step 1700
eval_steps: 100
save_steps: 100
patience: 8
class_weights: pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8
eval/infer calibration: label-token logit adjustment, tau=0.5
```

主结果建议报告：

| split | acc | macro-F1 | true-side macro-F1 | selection |
|---|---:|---:|---:|---:|
| val | 0.3046 | 0.3100 | 0.3089 | 0.6911 |
| test | 0.3102 | 0.3238 | 0.3297 | 0.7172 |

同时保留未校准结果作为 ablation：

| split | acc | macro-F1 | true-side macro-F1 | selection |
|---|---:|---:|---:|---:|
| val base | 0.3108 | 0.3023 | 0.2921 | 0.6780 |
| test base | 0.3046 | 0.3070 | 0.2850 | 0.6792 |

## 后续建议

1. 后续新增 LIAR-RAW Llama LoRA run 时，训练阶段仍可保持当前 halfbatch + class weights 设置。
2. eval / infer 默认使用 `tau=0.5`。
3. 若需要强调极端类召回，可额外报告 `tau=0.75`，但不要把它作为默认，因为 val 已显示过校准。
4. 下一轮如果继续调，应做 per-label bias sweep，而不是只继续增大统一 `tau`。统一 tau 会同时抬高 `pants-fire/barely-true/true`，但也会压低 `false/half-true`，权衡较粗。
