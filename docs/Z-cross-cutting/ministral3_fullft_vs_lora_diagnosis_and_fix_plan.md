# Ministral-3-8B FullFT 低于 LoRA 的嫌疑分析与修复计划

## 0. 背景与问题定义

当前主方法模型倾向使用 `Ministral-3-8B-Instruct-2512`，训练方式包括 LoRA 与 FullFT。前期实验观察到：在多数其他 backbone 上，FullFT 相比 LoRA 有明显收益；但在 Ministral 上，FullFT 并没有稳定优于 LoRA，部分结果甚至弱于 LoRA。

本文目标不是证明 “Ministral 不适合 FullFT”，而是把当前代码库中可能导致该现象的工程路径、训练配置、模型加载与校准因素拆开，给出可验证的诊断矩阵和修复方案。

核心判断：

```text
当前现象更像是 Ministral 的 FullFT 路径与 LoRA 路径不完全等价，
再叠加 FullFT 超参没有针对 Ministral 重新校准。
```

因此优先任务不是继续大规模扫 LR，而是先确认：

```text
1. LoRA 和 FullFT 是否训练的是同一个模型路径？
2. FullFT checkpoint selection 是否公平？
3. FullFT 是否因小数据全参更新而破坏 label-token 校准？
4. text-only export、FP8 dequantize、multimodal wrapper 是否引入了路径差异？
```

---

## 1. 已观察到的实验现象

根据当前 `docs/Z-cross-cutting/20260612_final_eval_results_liar_raw_rawfc.md` 中 RAWFC 结果，Ministral 相关结果大致如下：

| version | setting | model / train | val F1 | test F1 | 初步解释 |
|---|---|---|---:|---:|---|
| RF-v022 | `ministral3_8b_fullft` | `ministral3_8b / fullFT` | 0.671950 | 0.576594 | text-only FullFT；val 尚可，test 明显弱 |
| RF-v023 | `ministral3_8b_fullft_mm_text_effective` | `Ministral-3-8B-Instruct-2512 / fullFT` | 0.647376 | 0.614100 | 原始 multimodal wrapper + 冻结 vision/projector；test 不弱 |
| RF-v024 | `ministral3_8b_lora` | `Ministral-3-8B-Instruct-2512 / LoRA` | 0.688145 | 0.609140 | 当前 LoRA 主参考 |
| RF-v045 | legacy `ministral3_8b_lora` | `Ministral-3-8B-Instruct-2512 / LoRA` | 0.714623 | 0.604198 | legacy LoRA，val 高但 test 一般 |

注意：`fullft_mm_text_effective` 的 test F1 `0.614100` 实际略高于 RF-v024 LoRA 的 `0.609140`。因此现象不是 “所有 Ministral FullFT 都输 LoRA”，而是：

```text
Ministral FullFT 的 val/test 稳定性不如预期；
text-only FullFT 明显偏弱；
original multimodal wrapper FullFT 可能更接近或略好于 LoRA，但 val 选择不稳。
```

这提示问题更可能在路径、校准、checkpoint selection 或超参，而不是模型能力本身。

---

## 2. 嫌疑一：LoRA 与 FullFT 使用了不同的模型路径

### 2.1 现象

代码中的 phase7 backbone migration 对 `ministral3_8b` 做了特殊处理：Ministral 被视为 multimodal wrapper。FullFT 默认 `BACKBONE_TEXT_ONLY=auto`，会要求使用 text-only export；LoRA 默认则不需要 text-only export，通常直接使用原始 `/data/models/Ministral-3-8B-Instruct-2512`。

也就是说，当前默认比较很可能是：

```text
LoRA:
  原始 multimodal checkpoint
  /data/models/Ministral-3-8B-Instruct-2512

FullFT:
  text-only export
  outputs/cache/backbone_migration/text_backbones/ministral3_8b
```

这不是严格同模型路径比较。

### 2.2 代码证据

`README.md` 明确说明 `gemma4_e4b` 和 `ministral3_8b` 是 multimodal wrappers；FullFT 默认使用 `BACKBONE_TEXT_ONLY=auto`，需要 text-only export，以避免训练时构造 vision/audio/projector 模块。

对应文件：

```text
scripts/phase7_backbone_migration/README.md
```

关键说明包括：

```text
- gemma4_e4b and ministral3_8b are multimodal wrappers.
- For FullFT, BACKBONE_TEXT_ONLY=auto requires a one-time text-only export.
- BACKBONE_TEXT_ONLY=false forces the original multimodal checkpoint.
```

`run_one_backbone.sh` 中的逻辑也确认：`BACKBONE_TEXT_ONLY=auto` 时，只有 `finetune=fullft` 且 backbone 支持 text-only 才会走 text-only。

对应文件：

```text
scripts/phase7_backbone_migration/run_one_backbone.sh
```

关键逻辑：

```bash
text_only_required_for_run() {
  case "${mode}" in
    auto)
      [[ "${finetune}" == "fullft" ]] && backbone_supports_text_only "${backbone}"
      return
      ;;
  esac
}
```

### 2.3 为什么会影响结果

text-only export 是合理的工程路径，但它仍然引入了额外变量：

```text
1. config 从 multimodal root config 改成 text_config；
2. architecture 改成 Ministral3ForCausalLM；
3. 权重 key 从 language_model.* 映射到 CausalLM；
4. tokenizer / chat template / tekken 文件被复制；
5. quantization extras 被保留。
```

因此 FullFT 训练的是导出的 CausalLM，而 LoRA 训练的可能是原始 multimodal wrapper。这种路径差异足以解释 “FullFT 不如 LoRA”。

### 2.4 修复与诊断

先做路径等价实验，而不是先扫 LR。

建议四组：

| ID | Model path | Finetune | 目的 |
|---|---|---|---|
| A0 | original multimodal | LoRA | 当前 LoRA 主参考 |
| A1 | text-only export | LoRA | 判断 text-only export 是否本身伤模型 |
| A2 | text-only export | FullFT | 当前默认 FullFT 路径 |
| A3 | original multimodal + freeze vision/projector | FullFT | 判断原始 wrapper FullFT 是否优于 text-only |

解释：

```text
A0 vs A1:
  如果 A1 明显低于 A0，text-only export / text-only config 是主要嫌疑。

A1 vs A2:
  同一路径下比较 LoRA 和 FullFT，判断 FullFT 是否真的差。

A0 vs A3:
  同样 original multimodal 路径下比较 LoRA 和 FullFT。
```

### 2.5 建议代码修复

当前 `case_name` 只在 `finetune=fullft && BACKBONE=ministral3_8b && model_variant != text_only` 时追加 `_mm_text_effective`。如果强制 LoRA 走 text-only，则 case name 不会自动区分，容易覆盖或混淆输出。

建议在 `scripts/phase7_backbone_migration/run_one_backbone.sh` 中，`model_variant` 确定后加入：

```bash
if [[ "${model_variant}" == "text_only" ]]; then
  case_name="${case_name}_text_only"
fi
```

最终输出目录建议明确区分：

```text
v0_6c_rawfc3_rule_step_adaptive5_10_eval25_ministral3_8b_lora
v0_6c_rawfc3_rule_step_adaptive5_10_eval25_ministral3_8b_lora_text_only
v0_6c_rawfc3_rule_step_adaptive5_10_eval25_ministral3_8b_fullft_text_only
v0_6c_rawfc3_rule_step_adaptive5_10_eval25_ministral3_8b_fullft_mm_text_effective
```

---

## 3. 嫌疑二：FullFT 与 LoRA 的 early stopping / patience 不公平

### 3.1 现象

LoRA 对 Ministral 有特殊训练策略：

```text
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
early_stopping_patience = 16
```

FullFT 对 7B+ 模型也会设置 bsz1/ga8，但默认并没有自动把 Ministral FullFT 的 patience 提到 16。

这会导致：

```text
LoRA 有更长 checkpoint 搜索空间；
FullFT 更早停止，可能错过后期更好 checkpoint；
val/test 小样本下，这种差异会被放大。
```

### 3.2 代码证据

`prepare_backbone_config.py` 中：

```python
LORA_EARLY_STOPPING_PATIENCE = 16
LOW_MICRO_BATCH_LORA_BACKBONES = {"ministral3_8b"}
```

LoRA 分支中，如果 backbone 是 `ministral3_8b`，会设置 bsz1/ga8，并把 `sft_train.early_stopping_patience` 设为 16。

FullFT 分支中会关闭 LoRA，并对 7B+ 设 bsz1/ga8；对于 Gemma4/Ministral 会设置 ZeRO-3 和 freeze policy，但没有专门把 patience 对齐到 16。

### 3.3 修复建议

先对齐 checkpoint 搜索预算：

```text
Ministral FullFT:
  early_stopping_patience = 16
  eval_steps = 25
  save_steps = 50
```

如果 FullFT 与 LoRA 要做公平比较，至少应保证：

```text
1. 相同 eval cadence；
2. 相同 patience；
3. 相同 val selection rule；
4. 都跑 best checkpoint 和 final checkpoint eval；
5. 最终 test 只按 val 选择的 checkpoint 报告。
```

建议在 `prepare_backbone_config.py` 中临时加入：

```python
if args.finetune == "fullft" and backbone == "ministral3_8b":
    _set_path(payload, "sft_train.early_stopping_patience", 16)
    _set_path(payload, "backbone_migration.fullft_patience_policy", "ministral3_eval25_patience16")
```

---

## 4. 嫌疑三：FullFT 超参未针对 Ministral 校准

### 4.1 现象

当前 FullFT 基础配置的学习率是：

```text
learning_rate = 2.0e-6
```

LoRA 默认常用学习率通常是：

```text
learning_rate = 1.0e-5
```

这两个学习率不能直接横向比较，因为 LoRA 只更新 adapter，FullFT 更新全参数。问题在于：Ministral 的 FullFT 可能需要比通用 FullFT 更保守的更新策略，尤其 RAWFC 训练集和 val/test 都比较小。

### 4.2 可能机制

FullFT 对小数据更容易出现两种问题：

```text
1. lr 偏大：快速拟合 train，但破坏 instruction calibration，val/test 不稳。
2. lr 偏小：训练不充分，val 上看似稳定但没有学到有效边界。
```

Ministral 又叠加了 FP8 dequantize、multimodal wrapper/text-only export 和 MistralCommon tokenizer，因此使用其他 backbone 的 FullFT 超参未必合适。

### 4.3 推荐 FullFT 小矩阵

先在路径等价实验中选出更好的模型路径，然后只在该路径上做小矩阵。

| ID | lr | weight_decay | warmup_ratio | max_grad_norm | epoch | patience | 目的 |
|---|---:|---:|---:|---:|---:|---:|---|
| F0 | 2e-6 | 0.0 | 0.03 | 1.0 | 5 | 16 | 当前 FullFT 基准，补齐 patience |
| F1 | 1e-6 | 0.0 | 0.03 | 1.0 | 8 | 16 | 更保守更新 |
| F2 | 1e-6 | 0.01 | 0.05 | 0.5 | 8 | 16 | 正则化 FullFT |
| F3 | 5e-7 | 0.01 | 0.05 | 0.5 | 8 | 16 | 极保守更新 |
| F4 | 3e-6 | 0.01 | 0.05 | 0.5 | 5 | 16 | 判断当前是否欠拟合 |

优先级：

```text
资源有限时先跑 F1、F2。
```

解释标准：

```text
如果 train CE 快速下降但 val F1 不稳：
  降 LR，加 weight_decay，加 warmup，降低 max_grad_norm。

如果 train CE 下降慢，val/test 都不上升：
  尝试 3e-6 或更长 epoch。

如果 val 高但 test 低：
  重点看 checkpoint selection、class prior、logit_adjust、seed variance。
```

### 4.4 建议代码修复：暴露 FullFT 超参覆盖

`prepare_backbone_config.py` 当前没有暴露 LR、epoch、patience、weight decay、warmup、max grad norm 等命令行参数。建议新增：

```python
parser.add_argument("--learning-rate", type=float, default=None)
parser.add_argument("--num-train-epochs", type=float, default=None)
parser.add_argument("--early-stopping-patience", type=int, default=None)
parser.add_argument("--weight-decay", type=float, default=None)
parser.add_argument("--warmup-ratio", type=float, default=None)
parser.add_argument("--max-grad-norm", type=float, default=None)
```

并在 `_prepare_config()` 末尾统一写入：

```python
if args.learning_rate is not None:
    _set_path(payload, "sft_train.learning_rate", float(args.learning_rate))
if args.num_train_epochs is not None:
    _set_path(payload, "sft_train.num_train_epochs", float(args.num_train_epochs))
if args.early_stopping_patience is not None:
    _set_path(payload, "sft_train.early_stopping_patience", int(args.early_stopping_patience))
if args.weight_decay is not None:
    _set_path(payload, "sft_train.weight_decay", float(args.weight_decay))
if args.warmup_ratio is not None:
    _set_path(payload, "sft_train.warmup_ratio", float(args.warmup_ratio))
if args.max_grad_norm is not None:
    _set_path(payload, "sft_train.max_grad_norm", float(args.max_grad_norm))
```

---

## 5. 嫌疑四：FP8 dequantize + FullFT 破坏校准

### 5.1 现象

Ministral 的本地 checkpoint 可能带有 FP8 quantization config。代码会为 HF forward/eval 构造 `FineGrainedFP8Config(..., dequantize=True)`。

对于 LoRA：

```text
base weights 基本冻结；
dequantize 后的基座只作为固定特征；
adapter 学习小幅修正。
```

对于 FullFT：

```text
dequantize 后的全部 text weights 都进入 AdamW 更新；
小数据全参更新更容易破坏原模型 calibration；
尤其 label-token CE 只监督 A/B/C 或 A-F 标签 token，信号很窄。
```

### 5.2 代码证据

`model_loading.py` 中 `_finegrained_fp8_dequantize_config()` 会检测本地 `config.json` 的 `quantization_config.quant_method == fp8`，然后返回：

```python
FineGrainedFP8Config(
    activation_scheme=...,
    weight_block_size=...,
    dequantize=True,
    modules_to_not_convert=...
)
```

`load_causal_lm_compatible_model()` 会在没有显式传入 `quantization_config` 时自动使用这个 dequantize config。

### 5.3 诊断实验

在更好的路径上做：

```text
FullFT-FP8-conservative:
  lr = 5e-7 or 1e-6
  weight_decay = 0.01
  warmup_ratio = 0.05
  max_grad_norm = 0.5
  patience = 16
```

如果这组明显改善，说明 FullFT 主要问题是全参更新过猛或 dequantize 后 calibration drift。

如果仍然不改善，应继续看 label-token / lm_head 问题。

---

## 6. 嫌疑五：label-token CE 更新 lm_head 导致 label prior drift

### 6.1 现象

当前 verifier 不是普通生成式 SFT，而是 label-token CE：在 `Label:` 后取下一 token logits，只在 label token ids 上做 CE。

训练中：

```text
input = prompt + label_prefix
loss = CE(logits[label_token_ids], gold_label)
```

LoRA 默认只更新 attention / MLP projection modules，不直接更新 embedding / lm_head。FullFT 则会更新所有未冻结参数，包括 embedding 和 lm_head。

这可能导致：

```text
FullFT 快速改写 label token prior；
val 上学到某个类别分布；
test 上校准漂移；
macro-F1 不稳定。
```

### 6.2 代码证据

LoRA 默认 target modules：

```python
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

不包括 `lm_head` 或 `embed_tokens`。

训练代码 `_forward_label_logits_for_inputs()` 会取最后一个 non-pad token 的 next-token logits，再 index 到 label token ids。

### 6.3 诊断实验

建议做两组 FullFT freeze 诊断：

```text
FT-freeze-lm-head:
  freeze_module_prefixes:
    - lm_head

FT-freeze-embed-lm-head:
  freeze_module_prefixes:
    - lm_head
    - model.embed_tokens
```

但注意：Ministral 原始 multimodal wrapper 与 text-only export 的参数名前缀可能不同。应先打印或审计：

```text
包含 lm_head 的参数名
包含 embed_tokens 的参数名
包含 language_model 的参数名
```

当前 `freeze_modules_by_prefix()` 会记录冻结参数比例，并对没有匹配到的 prefix 发出 warning。因此这类 freeze 诊断是安全的。

判断标准：

```text
如果 freeze lm_head 后 FullFT 更稳：
  问题主要是 label-token prior / output head calibration drift。

如果 freeze 后性能下降：
  FullFT 需要更新输出层，问题更可能是 LR / regularization。
```

---

## 7. 嫌疑六：tokenizer / prompt_input_ids / checkpoint tokenizer 不一致

### 7.1 现象

Ministral 可能使用 MistralCommon tokenizer。代码已经做了防御：如果使用 MistralCommon tokenizer，build rows 必须保存 `prompt_input_ids`，否则训练或推理会报错。

这降低了静默 tokenizer 错误的概率，但仍建议审计，因为：

```text
LoRA 推理：checkpoint 是 PEFT adapter，tokenizer 从 base model 目录加载。
FullFT 推理：checkpoint 是完整 HF checkpoint，tokenizer 从 checkpoint_dir 加载。
```

如果 FullFT checkpoint 中保存的 tokenizer 文件与训练 build 阶段 tokenizer 有差异，label token ids 或 prompt ids 可能不一致。

### 7.2 代码证据

`LabelTokenDataset` 中：

```python
requires_prompt_input_ids = is_mistral_common_tokenizer(tokenizer)
if sample.prompt_input_ids is None and requires_prompt_input_ids:
    raise ValueError(...)
```

`build_inference_context()` 中：

```python
if is_peft_adapter:
    tokenizer_dir = base_tokenizer_source
else:
    tokenizer_dir = checkpoint_dir
```

这意味着 FullFT 与 LoRA 的 tokenizer source 不一样。

### 7.3 诊断项

每个 Ministral run 都应记录：

```text
tokenizer class
label_prefix
prefix_token_ids
label_token_ids
label_token_texts
prompt_input_ids presence rate
mean prompt_token_count
was_truncated rate
evidence_count mean
```

训练阶段已经保存 `label_token_ce_meta.json`，其中包含 label token ids、class weights、early stopping metric、ordinal loss、logit_adjust 等。应将 LoRA 和 FullFT 的该文件做 diff。

重点检查：

```text
LoRA 与 FullFT 的 label_token_ids 是否完全一致？
FullFT checkpoint tokenizer 与 build-time tokenizer 是否一致？
MistralCommon tokenizer 的 prompt_input_ids 是否 100% 存在？
```

---

## 8. 嫌疑七：checkpoint selection 与 val/test 方差

### 8.1 现象

RAWFC val/test 是 200 条级别，单 seed 的 `+0.02` 或 `-0.02` 可能只是 checkpoint / seed / class prior 波动。当前结果里 Ministral 的 val/test 也显示出明显不一致：

```text
LoRA: val 高，test 中等
FullFT text-only: val 中等，test 低
FullFT mm_text_effective: val 较低，test 反而更好
```

这说明 `val selection` 可能不稳定。

### 8.2 修复建议

对最终候选至少做 3 training seeds。RAWFC 若配置接近，应扩到 5 seeds。

同时所有配置使用同一组 seeds，例如：

```text
13, 21, 42
```

最终报告：

```text
mean ± std
每个 seed raw result
paired bootstrap CI, optional
```

评估时统一：

```text
best checkpoint by val macro-F1 / selection score
final checkpoint test
best checkpoint test
logit_adjust tau sweep: 0, 0.5, 0.75, 1.0
```

---

## 9. 最小推荐实验矩阵

### 9.1 Phase A：路径等价诊断

| ID | BACKBONE_TEXT_ONLY | Finetune | Expected case suffix | 目的 |
|---|---|---|---|---|
| A0 | false/default | LoRA | `lora` | 当前原始 LoRA 参考 |
| A1 | true | LoRA | `lora_text_only` | 判断 text-only export 是否伤 LoRA |
| A2 | true/auto | FullFT | `fullft_text_only` | 当前默认 FullFT 路径 |
| A3 | false | FullFT | `fullft_mm_text_effective` | 判断原始 multimodal wrapper FullFT 是否更好 |

建议命令示意：

```bash
# A1: LoRA text-only
BACKBONE=ministral3_8b \
BACKBONE_TEXT_ONLY=true \
FINETUNE=lora \
MODE=full \
bash scripts/phase7_backbone_migration/run_one_backbone.sh

# A3: FullFT original multimodal wrapper
BACKBONE=ministral3_8b \
BACKBONE_TEXT_ONLY=false \
FINETUNE=fullft \
MODE=full \
bash scripts/phase7_backbone_migration/run_one_backbone.sh
```

解释：

```text
如果 A1 << A0：text-only export 是主要问题。
如果 A2 << A1：FullFT 训练策略是主要问题。
如果 A3 >= A0：原始 multimodal wrapper FullFT 可以修复。
如果 A3 << A0：FullFT 超参 / label calibration 仍需修。
```

### 9.2 Phase B：FullFT 超参修复矩阵

只在 Phase A 中更好的 FullFT 路径上跑。

| ID | lr | wd | warmup | grad clip | epoch | patience | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| F0 | 2e-6 | 0.0 | 0.03 | 1.0 | 5 | 16 | baseline + patience 对齐 |
| F1 | 1e-6 | 0.0 | 0.03 | 1.0 | 8 | 16 | 保守 LR |
| F2 | 1e-6 | 0.01 | 0.05 | 0.5 | 8 | 16 | 推荐首选修复 |
| F3 | 5e-7 | 0.01 | 0.05 | 0.5 | 8 | 16 | 极保守 |
| F4 | 3e-6 | 0.01 | 0.05 | 0.5 | 5 | 16 | 欠拟合诊断 |

优先级：

```text
F1, F2 > F0 > F3 > F4
```

### 9.3 Phase C：label-token 校准诊断

| ID | 设置 | 目的 |
|---|---|---|
| C0 | FullFT + logit_adjust tau=0 | 原始校准 |
| C1 | FullFT + logit_adjust tau=0.5 | 类先验校正 |
| C2 | FullFT + logit_adjust tau=0.75 | 强一点校正 |
| C3 | FullFT + freeze lm_head | 判断 output head drift |
| C4 | FullFT + freeze lm_head + embed_tokens | 判断 embedding/head 更新是否伤校准 |

### 9.4 Phase D：多 seed 复验

对以下候选做 3 seeds：

```text
best LoRA path
best FullFT path
best FullFT + calibration / regularization variant
```

如果均值差距小于 `0.015` macro-F1，RAWFC 建议扩到 5 seeds。

---

## 10. 推荐的代码实现补丁列表

### 10.1 Patch 1：case name 区分 text-only

文件：

```text
scripts/phase7_backbone_migration/run_one_backbone.sh
```

建议在确定 `model_variant` 后加入：

```bash
if [[ "${model_variant}" == "text_only" ]]; then
  case_name="${case_name}_text_only"
fi
```

目的：避免 LoRA text-only 与 LoRA original 输出目录混淆。

### 10.2 Patch 2：FullFT patience 对齐 LoRA

文件：

```text
scripts/phase7_backbone_migration/prepare_backbone_config.py
```

建议加入：

```python
if args.finetune == "fullft" and backbone == "ministral3_8b":
    _set_path(payload, "sft_train.early_stopping_patience", 16)
    _set_path(payload, "backbone_migration.fullft_patience_policy", "ministral3_eval25_patience16")
```

目的：保证 LoRA 与 FullFT checkpoint 搜索空间更公平。

### 10.3 Patch 3：暴露 FullFT 超参覆盖

文件：

```text
scripts/phase7_backbone_migration/prepare_backbone_config.py
```

新增 CLI 参数：

```python
parser.add_argument("--learning-rate", type=float, default=None)
parser.add_argument("--num-train-epochs", type=float, default=None)
parser.add_argument("--early-stopping-patience", type=int, default=None)
parser.add_argument("--weight-decay", type=float, default=None)
parser.add_argument("--warmup-ratio", type=float, default=None)
parser.add_argument("--max-grad-norm", type=float, default=None)
```

写入 config：

```python
if args.learning_rate is not None:
    _set_path(payload, "sft_train.learning_rate", float(args.learning_rate))
if args.num_train_epochs is not None:
    _set_path(payload, "sft_train.num_train_epochs", float(args.num_train_epochs))
if args.early_stopping_patience is not None:
    _set_path(payload, "sft_train.early_stopping_patience", int(args.early_stopping_patience))
if args.weight_decay is not None:
    _set_path(payload, "sft_train.weight_decay", float(args.weight_decay))
if args.warmup_ratio is not None:
    _set_path(payload, "sft_train.warmup_ratio", float(args.warmup_ratio))
if args.max_grad_norm is not None:
    _set_path(payload, "sft_train.max_grad_norm", float(args.max_grad_norm))
```

### 10.4 Patch 4：run_one_backbone.sh 透传超参

文件：

```text
scripts/phase7_backbone_migration/run_one_backbone.sh
```

在 `prepare_config()` 中支持环境变量：

```bash
if [[ -n "${SFT_LEARNING_RATE:-}" ]]; then
  extra_args+=(--learning-rate "${SFT_LEARNING_RATE}")
fi
if [[ -n "${SFT_NUM_TRAIN_EPOCHS:-}" ]]; then
  extra_args+=(--num-train-epochs "${SFT_NUM_TRAIN_EPOCHS}")
fi
if [[ -n "${SFT_EARLY_STOPPING_PATIENCE:-}" ]]; then
  extra_args+=(--early-stopping-patience "${SFT_EARLY_STOPPING_PATIENCE}")
fi
if [[ -n "${SFT_WEIGHT_DECAY:-}" ]]; then
  extra_args+=(--weight-decay "${SFT_WEIGHT_DECAY}")
fi
if [[ -n "${SFT_WARMUP_RATIO:-}" ]]; then
  extra_args+=(--warmup-ratio "${SFT_WARMUP_RATIO}")
fi
if [[ -n "${SFT_MAX_GRAD_NORM:-}" ]]; then
  extra_args+=(--max-grad-norm "${SFT_MAX_GRAD_NORM}")
fi
```

### 10.5 Patch 5：新增 audit 脚本

建议新增：

```text
scripts/phase7_backbone_migration/audit_backbone_run.py
```

输出：

```text
model_name_or_path
model_variant
tokenizer class
label_prefix
prefix_token_ids
label_token_ids
label_token_texts
trainable parameter count
frozen parameter count
prompt_input_ids presence rate
mean prompt_token_count
was_truncated rate
evidence_count mean
best checkpoint step
best val macro_f1 / selection_score
test macro_f1
```

该脚本用于比较：

```text
LoRA original
LoRA text-only
FullFT text-only
FullFT mm_text_effective
```

---

## 11. 推荐排查顺序

### Step 1：确认路径等价

先跑或补跑：

```text
A1: LoRA text-only
A3: FullFT original multimodal wrapper
```

这一步最关键。

### Step 2：统一 patience 和 eval selection

确保：

```text
LoRA / FullFT 都使用 eval_steps=25 或 100 的同一口径；
patience 都是 16；
best checkpoint 由 val macro-F1 / selection_score 选择；
test 只用于最终报告。
```

### Step 3：跑 FullFT 保守超参

优先：

```text
F1: lr=1e-6, wd=0, warmup=0.03, clip=1.0
F2: lr=1e-6, wd=0.01, warmup=0.05, clip=0.5
```

### Step 4：做 label calibration 诊断

跑：

```text
logit_adjust tau = 0, 0.5, 0.75, 1.0
freeze lm_head
freeze lm_head + embed_tokens
```

### Step 5：对候选做 3 seeds

只有通过前四步筛出的候选才需要多 seed。

---

## 12. 最终判断标准

### 情况 A：LoRA text-only 明显下降

结论：

```text
text-only export 是主要问题。
```

修复方向：

```text
主方法继续使用 original multimodal LoRA；
FullFT 应优先使用 BACKBONE_TEXT_ONLY=false 的 mm_text_effective 路径；
text-only export 只作为工程 fallback，不作为主比较。
```

### 情况 B：LoRA text-only 不降，但 FullFT text-only 仍差

结论：

```text
FullFT 训练策略是主要问题。
```

修复方向：

```text
调低 lr；
增加 weight_decay / warmup；
对齐 patience；
尝试 freeze lm_head / embed_tokens；
做 logit_adjust tau sweep。
```

### 情况 C：FullFT mm_text_effective 稳定优于 LoRA

结论：

```text
Ministral FullFT 可修复；主要问题是 text-only 默认路径或 checkpoint selection。
```

修复方向：

```text
把 FullFT 主路径切到 BACKBONE_TEXT_ONLY=false；
文档中说明 freeze vision/projector，仅更新 text LM 和 lm_head。
```

### 情况 D：所有 FullFT 都不如 LoRA

结论：

```text
对当前 RAWFC 小数据 + label-token CE，Ministral LoRA 的隐式正则化更适合。
```

这仍然不是坏结果。可以在论文中表述为：

```text
For the Ministral backbone, parameter-efficient adaptation provides better validation stability than full fine-tuning under small-data label-token supervision.
```

中文：

```text
在 RAWFC 小数据与 label-token CE 监督下，Ministral 使用 LoRA 的隐式正则化比全参微调更稳定。
```

---

## 13. 推荐结论

当前最合理的结论是：

```text
Ministral FullFT 低于 LoRA 不是不可修复的模型问题，
而是路径差异、FullFT 超参、label-token 校准和小数据方差共同造成的高概率现象。
```

优先修复路线：

```text
1. 先做路径等价实验：LoRA text-only vs original；FullFT text-only vs mm_text_effective。
2. 给 Ministral FullFT 对齐 patience=16。
3. 在更优路径上跑 lr=1e-6 / wd=0.01 / warmup=0.05 / clip=0.5 的保守 FullFT。
4. 做 logit_adjust 与 freeze lm_head 诊断。
5. 对最终候选做 3 seeds。
```

若修复后 FullFT 仍不稳定，则应把主方法训练方式定为 LoRA，并把 FullFT 作为 backbone-specific negative result 或 appendix 诊断，而不是强行使用 FullFT。
