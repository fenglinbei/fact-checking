# Repository Guidelines

## Project Structure & Module Organization

This repository implements a LIAR-RAW fact-checking pipeline organized as `build -> train -> infer`. Source code lives under `src/`: `fact_checking/` contains pipeline orchestration, build, retrieval, inference, data, and utilities; `sft/` contains fine-tuning, evaluation, parsing, runtime adapter, and dataset code. Hydra configs are in `configs/`, with experiments under `configs/experiment/`. Shell entry points are in `scripts/pipeline/`. Generated artifacts are under `outputs/cache/` and `outputs/runs/`; raw LIAR-RAW data belongs at `data/raw/LIAR-RAW/` and is git-ignored.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate`: create and enter a local Python environment.
- `python -m pip install -U pip setuptools wheel && pip install -r requirements.txt`: install CUDA 12.4 PyTorch wheels and project dependencies.
- `PYTHONPATH=src python -m fact_checking.pipeline.run experiment=b0_2 pipeline.mode=full`: run the full build/train/infer flow.
- `bash scripts/pipeline/run_exp.sh experiment=b0 pipeline.mode=build`: run a single pipeline phase through the wrapper.
- `PYTHONPATH=src python -m compileall src`: quick syntax/import-path hygiene check.
- `PYTHONPATH=src pytest`: run pytest-style tests if added; no coverage gate is currently configured.

## Coding Style & Naming Conventions

Use Python 3.10+ with four-space indentation, type annotations where helpful, and small functions that match existing module boundaries. Name Python files and functions in `snake_case`; keep experiment configs lowercase, such as `mmr_lambda_sweep_1024.yaml`. Hydra overrides use dotted keys, for example `build.retrieval.mmr_lambda=0.2`. No formatter is configured in `pyproject.toml`, so preserve nearby style and avoid unrelated reformatting.

## Testing Guidelines

There is no dedicated `tests/` suite yet. For new behavior, prefer focused pytest tests named `test_*.py` near the relevant package or in a future `tests/` directory. For pipeline changes, include a compile check plus the smallest meaningful phase run, such as `pipeline.mode=build` or a targeted inference rerun.

## Commit & Pull Request Guidelines

Recent history uses short, direct subjects such as `fix`, `topk sweep`, and `Update gitignore for experiment artifacts`. Keep commits concise and scoped when possible, for example `fix adapter validation`. PRs should summarize the change, list commands run, note GPU/runtime assumptions, and identify affected configs or output paths. Do not commit raw data, secrets, `.env` files, or model checkpoint binaries.

## Agent-Specific Instructions

Follow `CLAUDE.md` for local agent behavior: state assumptions, keep edits surgical, and verify with concrete commands. When touching train/eval/infer logic, compare the full code path before changing one side in isolation.
