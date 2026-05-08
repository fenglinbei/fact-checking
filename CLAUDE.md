# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LIAR-RAW fact-checking pipeline: **build → train → infer**. The build phase retrieves evidence sentences for each claim using hybrid scoring (dense + lexical + BM25) with MMR diversity. The train phase fine-tunes a causal LM (Qwen2.5-7B-Instruct by default) with Accelerate/DeepSpeed. The infer phase serves the checkpoint via vLLM OpenAI-compatible API and computes classification metrics over 6 LIAR-RAW labels.

## Common commands

```bash
# Full experiment run
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0_2 pipeline.mode=full

# Equivalent shell wrapper
bash scripts/pipeline/run_exp.sh experiment=b0_2 pipeline.mode=full

# Single phase only
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.mode=build
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.mode=train
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.mode=infer

# Force re-run a phase (bypass cache)
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.force.build=true

# Hydra multi-run (sweep)
PYTHONPATH=src python -m fact_checking.pipeline.run -m experiment=b0,b1 baseline.top_k=5,10

# Connect to existing vLLM server (skip auto-launch)
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.mode=infer \
  infer.server.manage=false infer.base_url=http://127.0.0.1:8000/v1

# Prompt length stats only (no training)
python -m sft.trainer --config <resolved_config> --prompt-length-stats-only

# Install dependencies (Python 3.10+, CUDA 12.4)
pip install -r requirements.txt
accelerate config  # one-time setup for the training environment
```

## Architecture

### Two packages under `src/`

- **`fact_checking`** — pipeline orchestration, build phase, retrieval, data types, inference API client, config helpers
- **`sft`** — SFT training loop, prompting strategies (v1/v2), evaluation, metrics, dataset tokenization/loading

### Pipeline flow (`fact_checking.pipeline.runner`)

The entry point `fact_checking.pipeline.run` is a Hydra app. `PipelineRunner` resolves all configs, computes fingerprints for build/train configs (SHA1, first 12 chars), and writes a `manifest.json` in `outputs/runs/<experiment>/<run_id>/`. The pipeline is **resumable** — completed phases are skipped unless `pipeline.force.<phase>=true` or `pipeline.resume=false`.

- **Build**: runs in-process via `fact_checking.build.candidates.run_build`, writes `build_{train,val,test}.jsonl` to `outputs/cache/build/<build_id>/`
- **Train**: writes a resolved config YAML (`configs/train.resolved.yaml`), then launches `accelerate launch -m sft.trainer` as a subprocess with DeepSpeed
- **Infer**: manages vLLM server lifecycle, calls `/v1/completions`, parses label IDs, saves predictions/confusion matrix under `outputs/runs/<experiment>/<run_id>/infer/`

### Prompting strategy (`sft.prompting`)

Two prompt versions × two output modes = 4 strategies, selected via `baseline.prompt_version` and `baseline.output_mode`:

| prompt_version | output_mode | Class |
|---|---|---|
| v1 | label_only | `LabelOnlyOutputStrategy` |
| v1 | explanation_label | `ExplanationLabelOutputStrategy` |
| v2 | label_only | `ChatLabelOnlyOutputStrategy` |
| v2 | explanation_label | `ChatExplanationLabelOutputStrategy` |

v1 uses plain text prompts; v2 uses `tokenizer.apply_chat_template` with a system message. The strategy classes determine prompt format AND target format (label-only vs explanation+label). `build_output_strategy()` in `src/sft/prompting/output.py` selects the strategy.

### Evidence retrieval & build (`fact_checking.build`)

For each claim, reports are split into sentences, then:
1. Dense similarity (BGE embeddings via `TextEmbedder`)
2. Lexical overlap F1
3. BM25-like score
4. Hybrid score = weighted combination, min-max scaled
5. MMR (Maximal Marginal Relevance) for diversity
6. Dedup by canonicalized text, truncate to top_k

Config is in `configs/build/default.yaml` (`retrieval.*` parameters).

### Data flow

`data/raw/LIAR-RAW/{train,val,test}.json` → build → `outputs/cache/build/<id>/build_{split}.jsonl` → train (SFT) → `outputs/runs/<experiment>/<run_id>/train/best/` → infer (vLLM) → metrics in `outputs/runs/<experiment>/<run_id>/infer/`

### Labels

6-class LIAR-RAW labels (ordinal): `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`. Defined in `src/fact_checking/data/constants.py`.

## Configuration

Hydra with config groups under `configs/`. The pipeline default (`configs/pipeline/default.yaml`) composes `build/default`, `train/default`, `infer/vllm_api`, and `experiment/<name>`. Experiment configs inherit from `b0` via Hydra `defaults` and override baseline/sft_train settings (e.g., `b0_2` adds LoRA, v2 prompts, explanation output). Training supports `accelerate_deepspeed` or `single` backends (`train.backend`).

## Key files

- `src/fact_checking/pipeline/run.py` — Hydra entry point
- `src/fact_checking/pipeline/runner.py` — Pipeline orchestrator (build/train/infer phases, subprocess management, fingerprint caching)
- `src/fact_checking/build/candidates.py` — Build phase: evidence retrieval with hybrid scoring + MMR
- `src/sft/trainer.py` — SFT training loop (Accelerate, DeepSpeed, LoRA, eval)
- `src/sft/prompting/output.py` — Prompt/target construction (v1/v2, label_only/explanation_label)
- `src/sft/prompting/preparation.py` — Prepares samples from build output rows, applies truncation
- `src/fact_checking/infer/api.py` — vLLM API client, server lifecycle management, inference orchestration
- `src/fact_checking/pipeline/artifacts.py` — Manifest I/O, fingerprinting, path helpers
