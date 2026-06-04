# RAWFC v0.6c Backbone Migration Scripts

This directory contains the A-group backbone migration launchers. The scripts
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

Full fine-tuning uses the base FullFT config. For 7B and larger backbones
(`qwen3_8b`, `dsr1_qwen7b`), generated configs set:

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

## Compatibility Check

```bash
PYTHONPATH=src python scripts/phase7_backbone_migration/check_compat.py --all
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
