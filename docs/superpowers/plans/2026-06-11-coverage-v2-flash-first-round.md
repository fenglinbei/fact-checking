# Coverage v2 Flash First Round Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a cost-conscious first-round experiment that tests whether training on `source_coverage_v2_flash/liar_raw/covered_weak` improves over the existing best LIAR-RAW Llama LoRA run on the same full LIAR-RAW eval set.

**Architecture:** Use the existing sentence-trace build and label-token LoRA pipeline without adding a broad hyperparameter sweep. Build a new training config with `covered_weak/train` but keep full LIAR-RAW val/test candidates as the primary eval set, then train one primary LoRA run and perform a small eval-only tau sweep. A second training run is only allowed if predefined diagnostics show the old class weights are likely harming the filtered-train run.

**Tech Stack:** Bash, Python, `scripts/phase5_selectors/build/build_trace_verifier_data.py`, `scripts/sentence_trace_method/prepare_lora_config.py`, `accelerate`, DeepSpeed ZeRO-2, `sft.label_token_trainer`, `sft.label_token_infer`.

---

## Scope And Decision Rules

Primary question:

- Does `covered_weak/train` improve the model on the same full LIAR-RAW val/test evaluation distribution used by the previous best run?

Non-goals for the first round:

- Do not sweep LoRA rank, alpha, dropout, learning rate, scheduler, prompt format, selector, top-k, or model family.
- Do not use `covered_weak/val` as the primary selection set, because that would change the eval distribution relative to the old run.
- Do not run both Llama and Qwen in the first round.
- Do not report the cleaned test subset as the main result.

Primary eval set:

- Train: `data/processed/coverage/source_coverage_v2_flash/liar_raw/covered_weak/train.json`
- Val: full LIAR-RAW val, preferably `data/raw/LIAR-RAW/val.json`
- Test: full LIAR-RAW test, preferably `data/raw/LIAR-RAW/test.json`

Primary comparison baseline:

- Existing best run: `outputs/sentence_trace_method/liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw`
- Baseline checkpoint: `train/best`
- Baseline default calibration from prior tuning: `tau=0.5`

First-round new run:

- Build root: `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval`
- LoRA root: `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw`
- Model: `/data/models/Meta-Llama-3.1-8B-Instruct`
- LoRA: `r=16`, `alpha=32`, `dropout=0.05`, `bias=none`
- DeepSpeed: `configs/deepspeed_zero2_bsz1_ga4.json`
- Train batch: micro batch 1, gradient accumulation 4, 4 local processes, effective global batch 16
- Epoch cap: 8
- Eval/save steps: 50
- Early stopping patience: 8
- Class weights: `pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8`
- Calibration: eval-only tau sweep, not training-loss adjustment

Success threshold:

- Choose checkpoint and tau using full val only.
- Treat the filtered-train run as a useful improvement if full test macro-F1 or selection score improves by at least 0.005 absolute and full val does not regress by more than 0.005 selection score.
- Treat changes smaller than 0.005 as inconclusive unless per-class results show a clear recovery of previously collapsed classes without damaging `false` and `half-true`.
- If full test improves only on `covered_weak/test` but not on full test, report it as a clean-subset improvement, not as a main-pipeline win.

## Files And Artifacts

No source-code changes are required for the first round. The plan uses direct commands against existing scripts.

Artifacts created:

- `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/build/build_train.jsonl`
- `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/build/build_val.jsonl`
- `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/build/build_test.jsonl`
- `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/train.resolved.yaml`
- `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/train.resolved.yaml`
- `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/train/best/`
- `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/eval/{val,test}/best/label_token_logit_adjust_tau*/metrics.json`

Suggested final report:

- Create: `docs/B-classifier-collapse/20260611_coverage_v2_flash_coveredweak_first_round.md`
- Include: command log, row-count checks, baseline table, new-run table, tau selection table, final decision, and whether a second run was triggered.

### Task 1: Freeze Baseline And Eval Contract

**Files:**
- Read: `docs/B-classifier-collapse/202606101830_sentence_trace_lora_logit_adjust_tuning.md`
- Read: `docs/Z-cross-cutting/202606110040_source_coverage_v2_flash_quality_report.md`
- Read: `outputs/sentence_trace_method/liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw/train.resolved.yaml`

- [ ] **Step 1: Confirm baseline config points at full val/test candidates**

Run:

```bash
PYTHONPATH=src python -c "from fact_checking.config import load_yaml; cfg=load_yaml('outputs/sentence_trace_method/liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw/train.resolved.yaml'); print(cfg['data']['val_candidates']); print(cfg['data']['test_candidates']); print(cfg['sft_train']['label_token_ce']['class_weights'])"
```

Expected:

- `val_candidates` and `test_candidates` point to the existing full LIAR-RAW sentence-trace build files under `outputs/sentence_trace_method/...`.
- Class weights match `pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8`.

- [ ] **Step 2: Record baseline metrics with the same eval-only sweep used for the new run**

Run:

```bash
CASE_ROOT=outputs/sentence_trace_method/liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw \
SPLITS=val,test \
CHECKPOINTS=best \
TAUS=0,0.5,0.75 \
LOGIT_ADJUST_MODE=on \
FORCE_EVAL=false \
bash scripts/sentence_trace_method/run_lora_label_token_logit_adjust_eval_only.sh
```

Expected:

- Existing metrics are reused if present, or new metrics are written under `eval/{val,test}/best/label_token_logit_adjust_tau*/metrics.json`.
- The prior-tuning default `tau=0.5` remains the baseline reference unless this command reveals a reproducibility problem.

### Task 2: Build Filtered-Train, Full-Eval Candidate Files

**Files:**
- Create: `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/train.resolved.yaml`
- Create: `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/build/build_{train,val,test}.jsonl`
- Use traces: `outputs/sentence_trace_method/_sources/liar_raw/sentence_rule_step_adaptive5_10/{train,val,test}/selection_trace_{train,val,test}.jsonl`

- [ ] **Step 1: Build the mixed train/eval case**

Run:

```bash
PYTHONPATH=src python scripts/phase5_selectors/build/build_trace_verifier_data.py \
  --config scripts/sentence_trace_method/configs/liar_raw__llama31_8b.yaml \
  --train-raw data/processed/coverage/source_coverage_v2_flash/liar_raw/covered_weak/train.json \
  --val-raw data/raw/LIAR-RAW/val.json \
  --test-raw data/raw/LIAR-RAW/test.json \
  --dataset liar_raw \
  --label-schema liar6 \
  --output-dir outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval \
  --selection-mode trace \
  --trace-prompt-style plain \
  --expected-selector-name sentence_rule_step_adaptive5_10 \
  --expected-chunk-mmr-fingerprint 432dfc970e75 \
  --top-k 10 \
  --prompt-model-name-or-path /data/models/Meta-Llama-3.1-8B-Instruct \
  --train-model-name-or-path /data/models/Meta-Llama-3.1-8B-Instruct \
  --train-trace outputs/sentence_trace_method/_sources/liar_raw/sentence_rule_step_adaptive5_10/train/selection_trace_train.jsonl \
  --val-trace outputs/sentence_trace_method/_sources/liar_raw/sentence_rule_step_adaptive5_10/val/selection_trace_val.jsonl \
  --test-trace outputs/sentence_trace_method/_sources/liar_raw/sentence_rule_step_adaptive5_10/test/selection_trace_test.jsonl
```

Expected:

- Train rows: `5163`
- Val rows: `1274`
- Test rows: `1251`
- Train skipped rows are expected because full train traces include examples removed from `covered_weak/train`.

- [ ] **Step 2: Verify row counts**

Run:

```bash
wc -l \
  outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/build/build_train.jsonl \
  outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/build/build_val.jsonl \
  outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/build/build_test.jsonl
```

Expected:

```text
5163 outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/build/build_train.jsonl
1274 outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/build/build_val.jsonl
1251 outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/build/build_test.jsonl
7688 total
```

### Task 3: Prepare The Primary LoRA Config

**Files:**
- Read: `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/train.resolved.yaml`
- Create: `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/train.resolved.yaml`

- [ ] **Step 1: Generate the LoRA config**

Run:

```bash
PYTHONPATH=src python scripts/sentence_trace_method/prepare_lora_config.py \
  --source-config outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/train.resolved.yaml \
  --output-root outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw \
  --experiment-name liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw \
  --swanlab-project fact-checking-sentence-trace-method-lora-coverage \
  --r 16 \
  --alpha 32 \
  --dropout 0.05 \
  --bias none \
  --deepspeed-config configs/deepspeed_zero2_bsz1_ga4.json \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 8 \
  --eval-steps 50 \
  --save-steps 50 \
  --early-stopping-patience 8 \
  --class-weight pants-fire=1.2 \
  --class-weight false=1.0 \
  --class-weight barely-true=1.5 \
  --class-weight half-true=1.0 \
  --class-weight mostly-true=1.0 \
  --class-weight true=1.8
```

Expected:

- The command prints the new `train.resolved.yaml` path.
- The LoRA config copies the mixed build files into the LoRA root.

- [ ] **Step 2: Verify the config contract**

Run:

```bash
PYTHONPATH=src python -c "from fact_checking.config import load_yaml; p='outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/train.resolved.yaml'; cfg=load_yaml(p); print(cfg['data']['train_candidates']); print(cfg['data']['val_candidates']); print(cfg['data']['test_candidates']); print(cfg['train']['deepspeed_config']); print(cfg['sft_train']['gradient_accumulation_steps'], cfg['sft_train']['eval_steps'], cfg['sft_train']['save_steps'], cfg['sft_train']['early_stopping_patience']); print(cfg['sft_train']['label_token_ce']['class_weights']); print(cfg['sft_train']['logit_adjust'])"
```

Expected:

- `train_candidates` points to the LoRA root copied `build_train.jsonl` with 5163 rows.
- `val_candidates` and `test_candidates` point to full val/test copied build files.
- DeepSpeed config is `configs/deepspeed_zero2_bsz1_ga4.json`.
- Printed step settings are `4 50 50 8`.
- `logit_adjust.enabled` remains `False`; calibration is applied only during eval-only sweeps.

### Task 4: Train One Primary Filtered-Train Run

**Files:**
- Read: `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/train.resolved.yaml`
- Create: `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/train/`

- [ ] **Step 1: Launch training**

Run:

```bash
PYTHONPATH=src \
SAVE_LATEST_TRAIN_STATE=true \
RESUME_LATEST_TRAIN_STATE=true \
accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file configs/deepspeed_zero2_bsz1_ga4.json \
  -m sft.label_token_trainer \
  --config outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/train.resolved.yaml
```

Expected:

- Training writes checkpoints and `train/best`.
- Training writes `train/training_complete.json` with `"completed": true`, unless it is intentionally interrupted and resumed.
- Best checkpoint is selected against the full val build, not against `covered_weak/val`.

- [ ] **Step 2: Verify completion**

Run:

```bash
test -d outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/train/best
test -f outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/train/training_complete.json
cat outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/train/training_complete.json
```

Expected:

- `train/best` exists.
- `training_complete.json` exists and shows successful completion.

### Task 5: Run Small Eval-Only Tau Sweep

**Files:**
- Create: `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/eval/{val,test}/best/label_token_logit_adjust_tau*/metrics.json`

- [ ] **Step 1: Evaluate new run on full val/test**

Run:

```bash
CASE_ROOT=outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw \
SPLITS=val,test \
CHECKPOINTS=best \
TAUS=0,0.5,0.75 \
LOGIT_ADJUST_MODE=on \
FORCE_EVAL=false \
bash scripts/sentence_trace_method/run_lora_label_token_logit_adjust_eval_only.sh
```

Expected:

- `tau=0` gives the no-prior-adjustment reference.
- `tau=0.5` tests the previous best calibration strength.
- `tau=0.75` checks whether the filtered training set benefits from stronger tail-class correction.
- No tau beyond `0.75` is part of the first round because prior tuning already showed `tau=1.0` was too aggressive on val.

- [ ] **Step 2: Select tau using full val only**

Run:

```bash
PYTHONPATH=src python -c "import json, pathlib; root=pathlib.Path('outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/eval/val/best'); rows=[]; [rows.append((p.parent.name, json.loads(p.read_text())['selection_score'], json.loads(p.read_text())['macro_f1'], json.loads(p.read_text())['accuracy'])) for p in sorted(root.glob('label_token_logit_adjust_tau*/metrics.json'))]; [print('%s selection=%.4f macro_f1=%.4f acc=%.4f' % r) for r in sorted(rows, key=lambda x: x[1], reverse=True)]"
```

Expected:

- The selected tau is the one with highest full-val `selection_score`.
- If selection scores tie within `0.002`, choose the lower tau to avoid over-calibration.

### Task 6: Compare Against Old Best On The Same Eval Set

**Files:**
- Read: baseline metrics under `outputs/sentence_trace_method/liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw/eval/{val,test}/best/label_token_logit_adjust_tau*/metrics.json`
- Read: new metrics under `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/eval/{val,test}/best/label_token_logit_adjust_tau*/metrics.json`
- Create: `docs/B-classifier-collapse/20260611_coverage_v2_flash_coveredweak_first_round.md`

- [ ] **Step 1: Build the comparison table**

For the report, include these rows:

```text
old_best tau=0
old_best tau=0.5
old_best tau=0.75
covered_weak_train tau=0
covered_weak_train tau=0.5
covered_weak_train tau=0.75
covered_weak_train selected_tau
```

For each row, report:

```text
split, accuracy, macro_f1, true_side_macro_f1, selection_score, predicted_label_distribution
```

- [ ] **Step 2: Apply the first-round decision rule**

Use this exact decision table:

```text
If covered_weak selected_tau improves full-test selection_score or macro_f1 by >= 0.005 and full-val selection_score does not drop by > 0.005:
  Mark the filtered-train setup as a first-round win.

If covered_weak selected_tau is within +/- 0.005 on both full-val and full-test:
  Mark the result as neutral; do not tune further unless per-class results show clear practical value.

If covered_weak selected_tau drops full-val selection_score by > 0.005 and full-test macro_f1 also drops:
  Do not promote the filtered-train setup. Move to Task 7 only if confusion/prediction distribution shows old class weights are over-correcting.
```

- [ ] **Step 3: Write the report**

Create `docs/B-classifier-collapse/20260611_coverage_v2_flash_coveredweak_first_round.md` with this structure:

```markdown
# Coverage v2 Flash covered_weak First-Round LoRA Result

## Question

Does training on `source_coverage_v2_flash/liar_raw/covered_weak/train.json` improve the existing LIAR-RAW Llama sentence-trace LoRA result when evaluated on the same full LIAR-RAW val/test set?

## Setup

- Baseline:
- New run:
- Train data:
- Eval data:
- Fixed hyperparameters:
- Tau candidates:

## Build Checks

- train rows:
- val rows:
- test rows:
- skipped train rows:

## Full-Val Selection

| run | tau | acc | macro-F1 | true-side macro-F1 | selection |
| --- | --- | ---: | ---: | ---: | ---: |

## Full-Test Result

| run | tau | acc | macro-F1 | true-side macro-F1 | selection |
| --- | --- | ---: | ---: | ---: | ---: |

## Per-Class Effects

Summarize whether `pants-fire`, `barely-true`, and `true` recover without collapsing `false` or `half-true`.

## Decision

State one of: first-round win, neutral, or regression.

## Next Step

State whether Task 7 is triggered.
```

### Task 7: Optional Single Follow-Up Training Run

Only run this task if Task 6 shows a regression and the prediction distribution suggests class weights are over-correcting tail labels.

**Trigger condition:**

- Full-val selection score drops by more than `0.005`, and
- Full-test macro-F1 does not improve, and
- The selected-tau predictions visibly overproduce `pants-fire`, `barely-true`, or `true` relative to the full-test gold distribution.

**Files:**
- Create: `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_now/train.resolved.yaml`
- Create: `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_now/train/`

- [ ] **Step 1: Prepare a no-class-weight follow-up config**

Run:

```bash
PYTHONPATH=src python scripts/sentence_trace_method/prepare_lora_config.py \
  --source-config outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval/train.resolved.yaml \
  --output-root outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_now \
  --experiment-name liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_now \
  --swanlab-project fact-checking-sentence-trace-method-lora-coverage \
  --r 16 \
  --alpha 32 \
  --dropout 0.05 \
  --bias none \
  --deepspeed-config configs/deepspeed_zero2_bsz1_ga4.json \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 8 \
  --eval-steps 50 \
  --save-steps 50 \
  --early-stopping-patience 8
```

Expected:

- Because no `--class-weight` arguments are passed, class weights remain the base config values of `1.0` for all six labels.

- [ ] **Step 2: Train and eval the no-class-weight follow-up**

Run:

```bash
PYTHONPATH=src \
SAVE_LATEST_TRAIN_STATE=true \
RESUME_LATEST_TRAIN_STATE=true \
accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file configs/deepspeed_zero2_bsz1_ga4.json \
  -m sft.label_token_trainer \
  --config outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_now/train.resolved.yaml
```

Then run:

```bash
CASE_ROOT=outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_now \
SPLITS=val,test \
CHECKPOINTS=best \
TAUS=0,0.5,0.75 \
LOGIT_ADJUST_MODE=on \
FORCE_EVAL=false \
bash scripts/sentence_trace_method/run_lora_label_token_logit_adjust_eval_only.sh
```

Expected:

- This is the only allowed second training run in the first round.
- If this follow-up also fails the Task 6 threshold, stop the first round and report that filtered training does not improve the old best under low-budget tuning.

## Self-Review Checklist

- The plan uses one primary training run and one optional diagnostic follow-up only.
- The primary eval set stays full LIAR-RAW val/test for comparability with the old best.
- Tau sweep is eval-only and limited to `0`, `0.5`, and `0.75`.
- The plan avoids broad hyperparameter tuning.
- The report distinguishes main full-eval results from any later clean-subset diagnostic results.
