
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LIAR-RAW fact-checking pipeline: **build → train → infer**. Given a political claim, the system retrieves candidate evidence sentences from associated reports, scores them (dense embedding + lexical overlap + BM25, then MMR for diversity), formats them into a prompt, and fine-tunes Qwen2.5-7B-Instruct to predict a 6-class veracity label (pants-fire / false / barely-true / half-true / mostly-true / true). Inference runs via vLLM OpenAI-compatible API.

## Common Commands

```bash
# Full pipeline run
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0_2 pipeline.mode=full

# Single phase
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.mode=build
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.mode=train
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.mode=infer

# Shell wrapper (same as above, auto-sets PYTHONPATH)
bash scripts/pipeline/run_exp.sh experiment=b0_2 pipeline.mode=full

# Force re-run a phase (ignore cache)
PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0 pipeline.force.build=true

# Hydra multi-run (grid search)
PYTHONPATH=src python -m fact_checking.pipeline.run -m experiment=b0,b1 baseline.top_k=5,10

# Syntax check
PYTHONPATH=src python -m compileall src

# Classifier mode training/inference (alternative to generative)
PYTHONPATH=src python -m sft.classifier_trainer --config <resolved-config.yaml>
```

GPU allocation override:
```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b0_2 \
  train.cuda_visible_devices=0,1,2,3 \
  infer.cuda_visible_devices=4
```

Connect to an already-running vLLM server:
```bash
PYTHONPATH=src python -m fact_checking.pipeline.run \
  experiment=b0_2 pipeline.mode=infer \
  infer.server.manage=false infer.base_url=http://127.0.0.1:8000/v1
```

## Architecture

### Two top-level packages

- **`src/fact_checking/`** — pipeline orchestration, build (evidence retrieval + scoring), inference (vLLM API), data types/constants, utilities.
- **`src/sft/`** — supervised fine-tuning: trainer, dataset/prompt construction, inference/parsing/metrics, runtime adapters (LoRA merging, FlashAttn), logit adjustment.

The pipeline entry point is `src/fact_checking/pipeline/run.py` (Hydra `@hydra.main`). It instantiates `PipelineRunner` which orchestrates three phases as subprocesses:

1. **Build** (`fact_checking.build.candidates.run_build`) — reads raw LIAR-RAW JSON, chunks report text into candidate sentences (5 strategies: `sentence`, `raw`, `ctx_window`, `semantic`, `ctx_semantic`), computes embedding similarity via SentenceTransformer, plus lexical overlap and BM25-like scores, then applies MMR for diversity. Outputs `build_{train,val,test}.jsonl`.

2. **Train** (`sft.trainer` or `sft.classifier_trainer`) — launched via `accelerate launch` with DeepSpeed ZeRO-2/3. Resolves a merged training config, builds prompt/evidence datasets, fine-tunes with LoRA (optional). Writes checkpoint to `outputs/runs/<exp>/<run>/train/`.

3. **Infer** (`fact_checking.infer.api.run_api_inference` or `sft.classifier_infer.run_classifier_inference`) — starts or reuses a vLLM server, sends `/v1/completions` requests, parses label predictions, computes metrics and confusion matrices.

### Configuration system (Hydra + OmegaConf)

Configs live in `configs/` with groups: `pipeline/`, `build/`, `train/`, `infer/`, `experiment/`. The pipeline defaults compose `configs/pipeline/default.yaml` which references `build/default.yaml`, `train/default.yaml`, `infer/vllm_api.yaml`, and an experiment file (e.g., `experiment/b0.yaml`). Experiment files define the `baseline` group with model path, prompt mode, top_k, retrieval settings etc.

Override via dotted keys: `baseline.top_k=5`, `train.cuda_visible_devices=0,1`.

### Resumability and caching

Every phase is resumable. Build outputs are cached by config fingerprint at `outputs/cache/build/<sha1>/`. Training artifacts at `outputs/runs/<experiment>/<run_sha1>/train/`. Each run writes `manifest.json` tracking phase status. Set `pipeline.resume=false` or `pipeline.force.<phase>=true` to override.

### Key data flow

Raw JSON → `SentenceRecord`/`SampleRecord` → build scoring + MMR → `CandidateSentence` → prompt construction → `PreparedSample` (prompt + target + gold label) → SFT training → vLLM inference → metrics.

The 6 LIAR-RAW labels are mapped to single-letter tokens (A-F) to avoid multi-token bias in generative SFT (`src/fact_checking/data/constants.py`).

### Two inference modes

- **generative** (default): standard CausalLM generation, parses label from generated text.
- **classifier**: uses sequence classification head, set via `train.kind=classifier` and `infer.kind=classifier`.

### Experiment variants

`configs/experiment/` includes b0–b4 variants plus sweep configs for MMR lambda, top-k, and learned-lambda experiments. `b0_2` is the main full-run variant.

## Behavioral Guidelines

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

### 5. Language Requirements

Use Chinese for the final answer/explanation or when presenting a plan; there are no requirements regarding the intermediate steps.
