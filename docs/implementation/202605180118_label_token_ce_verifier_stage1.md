# Stage 1 Label-token Weighted CE Verifier 实现说明

## 训练输入

Stage 1 使用 fixed-MMR 标准 build prompt，而不是 oracle prompt：

```yaml
experiment: b3_label_token_ce_1024
build.retrieval.mmr_lambda: 0.70
build.retrieval.top_k: 5
build.prompt.max_length: 1024
build.prompt.output_mode: label_only
build.prompt.label_format: letter
```

训练时将每条样本改写为：

```text
input  = row["prompt"].rstrip() + "Label:"
target = correct single label token: " A" ... " F"
```

模型仍是 `AutoModelForCausalLM`，LoRA 保存格式仍走现有 `sft.data.io.save_model`，因此产物 `train/best` 可以直接给现有 vLLM infer pipeline 使用。

## 代码路径

新增文件：

| 文件 | 作用 |
|---|---|
| `src/sft/label_token_dataset.py` | 将 prebuilt build rows 转成 `prompt + Label:` 的训练样本，并动态 padding |
| `src/sft/label_token_trainer.py` | 在 A-F label token logits 上训练 weighted CE，并保存 `best/final` checkpoint |
| `configs/experiment/b3_label_token_ce_1024.yaml` | Stage 1 默认实验配置 |
| `scripts/verifier/run_label_token_ce_stage1.sh` | 一键 build/train/infer wrapper |

修改文件：

| 文件 | 改动 |
|---|---|
| `src/fact_checking/pipeline/runner.py` | 支持 `train.kind=label_token_ce`，通过 `sft.label_token_trainer` 训练 |

## Loss

`sft.label_token_trainer` 不再对完整 target 序列做 causal LM loss，而是取 `Label:` 之后下一 token 的 logits：

```text
label_logits = logits_at_last_input_position[:, token_ids(" A"..." F")]
loss = WeightedCrossEntropy(label_logits, gold_label_id)
```

默认权重：

```yaml
sft_train.label_token_ce.class_weights:
  pants-fire: 1.0
  false: 1.0
  barely-true: 1.2
  half-true: 1.2
  mostly-true: 2.0
  true: 3.0
```

## Checkpoint 选择

默认使用：

```text
selection_score = macro_f1 + 0.5 * true_side_macro_f1
true_side_macro_f1 = mean(F1(mostly-true), F1(true))
```

每次 eval 会保存：

```text
train/eval/step-*/metrics.json
train/eval/step-*/confusion_matrix.json
train/eval/step-*/confusion_matrix.png
train/eval/step-*/val_predictions.jsonl
```

同时写出：

```text
train/label_token_ce_meta.json
```

其中包含 label prefix、A-F token ids、类别权重和 checkpoint selection metric。

## 运行方式

训练并在 val 上 infer：

```bash
bash scripts/verifier/run_label_token_ce_stage1.sh
```

只训练：

```bash
PIPELINE_MODE=train bash scripts/verifier/run_label_token_ce_stage1.sh
```

切换 infer split：

```bash
PIPELINE_MODE=infer INFER_SPLIT=test bash scripts/verifier/run_label_token_ce_stage1.sh
```

常用覆盖项：

```bash
bash scripts/verifier/run_label_token_ce_stage1.sh \
  train.cuda_visible_devices=0,1,2,3 \
  infer.cuda_visible_devices=0,1,2,3 \
  infer.tensor_parallel_size=4 \
  sft_train.label_token_ce.class_weights.true=4.0
```

默认输出目录形如：

```text
outputs/runs/b3_label_token_ce_1024/label_token_ce_stage1__<run_id>/
```

## 评估注意事项

1. Stage 1 配置关闭了 `sft_train.logit_adjust.enabled`，避免 weighted CE 与 prior correction 双重校正。
2. infer 侧仍使用 `infer.label_decoding.prefix="Label:"` 和 guided label choice。
3. 判断 Stage 1 是否成功时，先看 fixed-MMR prompt 上的 verifier 指标，再进入 re-oracle；不要直接用旧 oracle set 判定效果。
