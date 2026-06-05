# RAWFC v0.6c Backbone Migration Scripts

This directory contains the backbone migration launchers. The scripts
reuse the existing RAWFC v0.6c eval25 selector traces and call the existing
`phase5_selectors` pipeline. They do not recompute QD, evidence maps, or graph
traces by default.

## Backbones

| Slug | Model path | Size |
| --- | --- | ---: |
| `qwen25_15b` | `/data/models/Qwen2.5-1.5B-Instruct` | 1.5B |
| `qwen3_17b` | `/data/models/Qwen3-1.7B` | 1.7B |
| `qwen25_3b` | `/data/models/Qwen2.5-3B-Instruct` | 3B |
| `qwen3_4b_2507` | `/data/models/Qwen3-4B-Instruct-2507` | 4B |
| `qwen3_8b` | `/data/models/Qwen3-8B` | 8B |
| `dsr1_qwen7b` | `/data/models/DeepSeek-R1-Distill-Qwen-7B` | 7B |
| `llama31_8b` | `/data/models/Meta-Llama-3.1-8B-Instruct` | 8B |
| `phi4_mini` | `/data/models/Phi-4-mini-instruct` | 3.8B |
| `gemma4_e4b` | `/data/models/gemma-4-E4B-it` | 4.5B effective / 8B with embeddings |
| `ministral3_8b` | `/data/models/Ministral-3-8B-Instruct-2512` | 8.4B |

Full fine-tuning uses the base FullFT config. For 7B and larger backbones
(`qwen3_8b`, `dsr1_qwen7b`, `llama31_8b`, `gemma4_e4b`, `ministral3_8b`),
generated configs set:

- `sft_train.per_device_train_batch_size=1`
- `sft_train.gradient_accumulation_steps=8`

Smaller FullFT runs keep the base `2 / 4` batch settings.

Generated configs also set `build.prompt.chat_template.mode=tokenizer_default`
for every backbone, so prompt rendering uses each tokenizer's shipped chat
template. `qwen3_17b` and `qwen3_8b` additionally set
`build.prompt.chat_template.template_kwargs.enable_thinking=false`, so their
tokenizers apply the hard non-thinking switch and emit the empty
`<think></think>` block required by the Qwen3 chat template. `qwen3_4b_2507`
keeps the default template because it is a native non-thinking Instruct model.

Transfer-backbone notes:

- `phi4_mini` uses fused Phi modules for LoRA: `qkv_proj`, `o_proj`,
  `gate_up_proj`, `down_proj`.
- `gemma4_e4b` is expected at `/data/models/gemma-4-E4B-it`; the instruction
  tuned variant is the intended comparison point for this prompt-based setup.
  Gemma 4 requires a Transformers build that recognizes `gemma4`. Its LoRA
  config targets only `model.language_model.layers.*` by regex so PEFT does
  not try to wrap the multimodal tower's `Gemma4ClippableLinear` modules.
  FlashAttention 2 is disabled for this backbone because Gemma4 has an
  attention head dimension above the current flash-attn kernel limit.
  FullFT runs use `configs/deepspeed_zero3.json`
  (`train_micro_batch_size_per_gpu=1`, `gradient_accumulation_steps=8`, no CPU
  offload), `per_device_train_batch_size=1`, and `per_device_eval_batch_size=1`;
  the phase7 wrapper also forwards that file
  through `FULLFT_DEEPSPEED_CONFIG` so `accelerate launch --deepspeed_config_file`
  uses the same micro-batch policy.
- `gemma4_e4b` and `ministral3_8b` are multimodal wrappers. For FullFT, the
  phase7 runner defaults to `BACKBONE_TEXT_ONLY=auto`, which requires a
  one-time text-only export under `outputs/cache/backbone_migration/text_backbones/`.
  This keeps `sft_train.max_length=1024` while avoiding construction of
  vision/audio/projector modules during training.
- Set `BACKBONE_TEXT_ONLY=false` for the Mistral3 original multimodal-checkpoint
  FullFT alignment run. In that mode, generated configs use
  `configs/deepspeed_zero3_bsz1_ga8_lowpeak.json`, keep `bsz1/ga8`, and freeze
  `model.vision_tower` plus `model.multi_modal_projector` so only the text
  language model and `lm_head` are updated.
- `ministral3_8b` stores Mistral-format vLLM extra args in generated configs.
  Keep `RUN_API_INFER=false` until the HF label-token path passes smoke. The
  local FP8 checkpoint is dequantized for HF forward/eval to avoid depending on
  hub-loaded finegrained-fp8 kernels. Generated LoRA configs use
  `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`, and
  `configs/deepspeed_zero2_bsz1_ga8.json`, which preserves the same effective
  global batch as the earlier LoRA matrix while lowering the per-GPU
  micro-batch. `early_stopping_patience` is raised to 16.

Build text-only exports once before Gemma4/Ministral FullFT:

```bash
PYTHONPATH=src /data/liaozijie/conda/accelerate-fc-gemma4/bin/python \
scripts/phase7_backbone_migration/export_text_only_backbone.py \
  --source /data/models/gemma-4-E4B-it \
  --output outputs/cache/backbone_migration/text_backbones/gemma4_e4b \
  --family gemma4

PYTHONPATH=src /data/liaozijie/conda/accelerate-fc-gemma4/bin/python \
scripts/phase7_backbone_migration/export_text_only_backbone.py \
  --source /data/models/Ministral-3-8B-Instruct-2512 \
  --output outputs/cache/backbone_migration/text_backbones/ministral3_8b \
  --family mistral3
```

Set `BACKBONE_TEXT_ONLY=false` to force the original multimodal checkpoint.

## Compatibility Check

```bash
PYTHONPATH=src python scripts/phase7_backbone_migration/check_compat.py --all
```

Transfer group only:

```bash
PYTHONPATH=src python scripts/phase7_backbone_migration/check_compat.py \
  --backbone llama31_8b
PYTHONPATH=src python scripts/phase7_backbone_migration/check_compat.py \
  --backbone phi4_mini
PYTHONPATH=src python scripts/phase7_backbone_migration/check_compat.py \
  --backbone gemma4_e4b
PYTHONPATH=src python scripts/phase7_backbone_migration/check_compat.py \
  --backbone ministral3_8b
```

## Single Backbone

Dry-run LoRA:

```bash
BACKBONE=qwen3_4b_2507 MODE=dry_run FINETUNE=lora \
bash scripts/phase7_backbone_migration/run_one_backbone.sh
```

Smoke LoRA:

```bash
BACKBONE=qwen25_3b MODE=smoke FINETUNE=lora \
bash scripts/phase7_backbone_migration/run_one_backbone.sh
```

Smoke FullFT for a 7B+ model:

```bash
BACKBONE=qwen3_8b MODE=smoke FINETUNE=fullft \
bash scripts/phase7_backbone_migration/run_one_backbone.sh
```

## All A-Group Backbones

Full LoRA matrix:

```bash
MODE=full FINETUNE=lora \
bash scripts/phase7_backbone_migration/run_all_a_backbones.sh
```

Full FullFT matrix:

```bash
MODE=full FINETUNE=fullft \
bash scripts/phase7_backbone_migration/run_all_a_backbones.sh
```

Run a subset:

```bash
BACKBONES=qwen25_3b,qwen3_8b MODE=dry_run FINETUNE=both \
bash scripts/phase7_backbone_migration/run_all_a_backbones.sh
```

## Transfer Backbones

The default transfer group contains all four registered candidates. In the
current `accelerate-fc` environment, `gemma4_e4b` is expected to fail until
Transformers supports `gemma4`; override `BACKBONES` to run only currently
compatible transfer candidates.

Dry-run all four transfer backbones:

```bash
MODE=dry_run FINETUNE=lora \
bash scripts/phase7_backbone_migration/run_all_transfer_backbones.sh
```

Dry-run currently compatible transfer candidates:

```bash
BACKBONES=llama31_8b,phi4_mini,ministral3_8b MODE=dry_run FINETUNE=lora \
bash scripts/phase7_backbone_migration/run_all_transfer_backbones.sh
```

Smoke all four transfer backbones:

```bash
MODE=smoke FINETUNE=lora SAMPLE_LIMIT=32 \
bash scripts/phase7_backbone_migration/run_all_transfer_backbones.sh
```

Full LoRA transfer matrix:

```bash
MODE=full FINETUNE=lora \
bash scripts/phase7_backbone_migration/run_all_transfer_backbones.sh
```

Run one transfer backbone:

```bash
BACKBONE=ministral3_8b MODE=smoke FINETUNE=lora SAMPLE_LIMIT=32 \
bash scripts/phase7_backbone_migration/run_one_backbone.sh
```

## Outputs

Generated resolved configs are written under:

```text
outputs/cache/backbone_migration/configs/
```

Verifier data and training outputs use:

```text
outputs/selector_trace_verifier/rawfc_v0_6c_eval25_backbone/
outputs/runs/rawfc_v0_6c_eval25_backbone/
```

Smoke runs append `_smoke<N>` to the case name so sample-limited artifacts do
not collide with full-run artifacts.
