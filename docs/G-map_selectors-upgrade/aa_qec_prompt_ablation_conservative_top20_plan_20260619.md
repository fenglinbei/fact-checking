# AA-QEC Prompt Ablation and Conservative Top20 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the next AA-QEC LIAR-RAW experiments as a narrow prompt ablation first, then only advance Stage3 top20 through conservative selector variants that preserve the stronger Stage2 selected-scope behavior.

**Architecture:** Keep `v0_7_atom_facts_abc` as the frozen source artifact and treat prompt style, selected-scope reconstruction, and top20 supplementation as separate axes. The first executable batch changes only `TRACE_PROMPT_STYLE` for C3/C4; the conservative top20 variants are gated by build diagnostics before full training.

**Tech Stack:** Bash wrappers under `scripts/sentence_trace_method/`, AA-QEC selector logic in `src/fact_checking/selectors/atom_anchored_qec.py`, prompt construction in `scripts/phase5_selectors/build/build_trace_verifier_data.py`, pytest coverage in `scripts/sentence_trace_method/test_experiment_matrix_scripts.py` and `src/fact_checking/selectors/test_atom_anchored_qec.py`.

---

## Current Decision Context

The current LIAR-RAW ABC prompt results show that `qec_map` is slightly ahead of `qec_min` at selected checkpoints, but not enough to treat it as a confirmed win:

| Run | Checkpoint | Test macro-F1 | Test selection | Note |
| --- | ---: | ---: | ---: | --- |
| `v0_7_atom_facts_abc + qec_map` | 2400 | 0.358709 | 0.768565 | current best prompt-side checkpoint |
| `v0_7_atom_facts_abc + qec_min` | 1600 | 0.357839 | 0.776487 | close to qec_map |
| `C3 + qec_min` | 800 | 0.357041 | 0.774178 | strongest Stage2 checkpoint-level comparison |
| `C4 + qec_min` | 1700 | 0.347842 | 0.769782 | lower macro-F1, still useful as no-secondary control |

The next experiments should answer two focused questions:

1. Does `qec_map` improve the best Stage2 selected-scope AA-QEC variants when the selector stays fixed?
2. Can top20 help only as a conservative supplement, without the broad F2/F3 behavior that diluted Stage3 quality?

## Scope

In scope for the next batch:

- LIAR-RAW only.
- Source artifacts fixed to `v0_7_atom_facts_abc`:
  - `selector_name=v0_7_atom_facts_abc_budgeted_marginal_chain_adaptive5_10`
  - `source_root=outputs/selectors/evidence_chain_graph/liar_raw_v0_7_atom_facts_abc_budgeted_marginal_adaptive5_10`
  - `chunk_mmr_fingerprint=d4cbf7c18126`
  - `allow_multi_sentence_candidates=true`
- Prompt ablation on existing Stage2 C3/C4 selected-scope traces:
  - C3: primary + secondary + fallback, min5/max10.
  - C4: primary + fallback, no secondary, min5/max10.
- Conservative Stage3 design and build-gated variants only after C3/C4 `qec_map` is available.

Out of scope for this batch:

- RAWFC.
- Broad Stage3 F2/F3 reruns without a conservative policy change.
- New prompt styles such as `qec_map_no_relation` or `qec_map_shuffled`; the current code path supports `qec_min` and `qec_map` only.
- Using test split for checkpoint or tau selection. Test remains confirmation only.

## Experiment Matrix

### Batch P: Prompt Ablation on Stage2 Selected Scope

| ID | Selector | Candidate scope | Prompt | Purpose |
| --- | --- | --- | --- | --- |
| P1 | `aa_qec_constrained_atom_facts_abc_primary_secondary_fallback_qd_prefer_selected_min5_10` | selected | `qec_map` | Compare directly against C3 `qec_min`; isolates prompt map visibility. |
| P2 | `aa_qec_constrained_atom_facts_abc_primary_fallback_no_secondary_qd_prefer_selected_min5_10` | selected | `qec_map` | Compare directly against C4 `qec_min`; tests map prompt without secondary evidence. |

Default training recipe:

- `ministral3_8b`
- LoRA `r=16, alpha=32, dropout=0.05`
- effective batch size 16 with `SFT_GRADIENT_ACCUMULATION_STEPS=4`
- `SFT_LEARNING_RATE=2e-5`
- `SFT_NUM_TRAIN_EPOCHS=12`
- `SFT_EVAL_STEPS=100`
- `SFT_SAVE_STEPS=100`
- `SFT_EARLY_STOPPING_PATIENCE=8`
- `LIAR_CLASS_WEIGHTS=pants-fire=1.2,false=1.0,barely-true=1.5,half-true=1.0,mostly-true=1.0,true=1.8`
- `REQUIRE_PROMPT_INPUT_IDS=true`
- `EVAL_SPLITS=val,test`
- `RUN_TAU_EVAL=auto`, `TAUS=0.75`

Acceptance rule:

- Use val metrics to decide whether the prompt ablation is worth checkpoint-level test eval.
- A prompt win is only credible if `qec_map` improves or matches C3/C4 val macro-F1 while keeping selection score and prompt truncation in the same range.
- To claim a new test-side macro-F1 best, the final selected checkpoint must exceed 0.358709 and pass a paired check against the current `v0_7_atom_facts_abc + qec_map` checkpoint-2400 comparison target.

### Batch G: Conservative Top20 Stage3 Variants

Only start this batch after Batch P has a clear C3/C4 prompt result.

| ID | Selector policy | Candidate scope | Prompt | Purpose |
| --- | --- | --- | --- | --- |
| G1 | selected primary/fallback anchors, top20 fallback replacement only | top20-conservative | best of P/C | Tests whether top20 can improve weak fallback fills without changing primary evidence. |
| G2 | selected primary anchors, top20 QD-filtered top-up for uncovered atoms | top20-conservative | best of P/C | Allows at most one high-QD outside-selected candidate per missing atom. |
| G3 | selected C3 chain with top20 replacement by quality margin | top20-conservative | best of P/C | Replaces selected secondary/fallback only when a top20 candidate has a clear QD/quality margin. |

Conservative constraints:

- Preserve selected-scope primary evidence when present.
- Never add broad secondary evidence from top20 by default.
- Use top20 outside-selected candidates only when they fill an uncovered atom, replace a low-quality fallback, or satisfy a strict QD cue.
- Keep min5/max10 unless a dynamic-chain diagnostic is explicitly approved later.
- Keep `duplicate_evidence_rate.mean == 0`.
- Require `chain_diagnostics` on every trace row.

Build gate rule:

- Row counts must match source rows for `train,val,test`.
- Top20 variants should show at least some outside-selected use, but outside-selected rate is descriptive, not a quality target.
- Atom coverage must be split-specific and compared against a selected-scope baseline, not a fixed 0.846 for all splits.
- `qd_cue_rate.mean >= 0.95` is hard for `train,val`; `test` is a warning because test should not block build selection after val passes.
- Prompt truncation is derived from `prompt_stats.json` when available, otherwise from `build/build_{split}.jsonl` fields such as `was_truncated`, `evidence_text_truncated`, `prompt_token_count`, and `evidence_count`.
- Missing baseline prompt stats should produce a warning, not a hard failure, if case build rows are present.

## Files To Create Or Modify

- Create `scripts/sentence_trace_method/run_aa_qec_stage2_c3_c4_qec_map_full_ministral3.sh`
  - Full wrapper for P1/P2.
  - Defaults to `MODE=full`, `TRACE_PROMPT_STYLE=qec_map`, `AA_QEC_STAGE2_CASES=C3,C4`, `EVAL_SPLITS=val,test`, `RUN_TAU_EVAL=auto`.
- Modify `scripts/sentence_trace_method/test_experiment_matrix_scripts.py`
  - Add dry-run coverage for the P1/P2 wrapper.
  - Add gate/report regression coverage for split-specific atom thresholds, build-row prompt fallback, and test-only qd warnings.
- Modify `scripts/sentence_trace_method/check_aa_qec_stage3_build_gate.py`
  - Fix false failures from missing `prompt_stats`.
  - Add split-specific baseline atom coverage floors.
  - Add warning support in both console output and JSON report.
  - Default prompt checks to `train,val,test`.
- Modify `scripts/sentence_trace_method/run_aa_qec_stage3_liar_raw_atom_facts_abc_ministral3.sh`
  - Pass `--prompt-splits train,val,test` to the gate.
- Modify `scripts/sentence_trace_method/run_aa_qec_stage3_liar_raw_atom_facts_abc_f1_f3_full_ministral3.sh`
  - Pass `--prompt-splits train,val,test` to the precheck gate.
- Modify `src/fact_checking/selectors/atom_anchored_qec.py`
  - Add conservative top20 selector policies only after Batch P result review.
- Modify `src/fact_checking/selectors/test_atom_anchored_qec.py`
  - Add conservative top20 unit tests only after selector implementation starts.

## Task 1: Fix Stage3 Gate And Report

- [ ] Add a failing pytest case that builds a synthetic Stage3 F2 trace where:
  - F2 uses top20 indices outside the original selected set.
  - Atom coverage matches a selected-scope baseline below 0.846 on train.
  - `prompt_stats.json` is absent.
  - `build/build_{split}.jsonl` contains non-truncated prompt fields.
  - Test split qd cue is below 0.95 and must become a warning, not a hard failure.
- [ ] Run the new test and confirm it fails on the current script.
- [ ] Implement the minimal gate/report changes:
  - `warnings` top-level report list.
  - split-specific atom floor from `aa_qec_constrained_atom_facts_abc_primary_secondary_fallback_qd_prefer_selected_min5_10` when available.
  - `--qd-hard-splits` defaulting to `train,val`.
  - prompt-quality fallback from build rows.
- [ ] Re-run the new test and the Stage3 wrapper dry-run tests.
- [ ] Run the gate on existing F2/F3 artifacts and inspect the regenerated report.

## Task 2: Add C3/C4 qec_map Prompt Ablation Wrapper

- [ ] Add `scripts/sentence_trace_method/run_aa_qec_stage2_c3_c4_qec_map_full_ministral3.sh`.
- [ ] Make it call the existing Stage2 C3/C4 full wrapper with:
  - `TRACE_PROMPT_STYLE=qec_map`
  - `AA_QEC_STAGE2_CASES=C3,C4`
  - `MODE=full`
  - `EVAL_SPLITS=val,test`
  - `RUN_TAU_EVAL=auto`
  - `FORCE_AA_QEC_BUILD=false`
  - `FORCE_STAGE=false`
- [ ] Add a dry-run test that asserts:
  - C3/C4 are present.
  - C1/C2 are absent.
  - `TRACE_PROMPT_STYLE=qec_map` is present.
  - `EVAL_SPLITS=val,test` is present.
  - `RUN_TAU_EVAL=auto` is present.
  - RAWFC is absent.
- [ ] Verify with:

```bash
PYTHONPATH=.:src /data/liaozijie/conda/accelerate-fc/bin/python -m pytest \
  scripts/sentence_trace_method/test_experiment_matrix_scripts.py::test_aa_qec_stage2_c3_c4_qec_map_full_wrapper_targets_only_c3_c4 -v

bash -n scripts/sentence_trace_method/run_aa_qec_stage2_c3_c4_qec_map_full_ministral3.sh

DRY_RUN=true bash scripts/sentence_trace_method/run_aa_qec_stage2_c3_c4_qec_map_full_ministral3.sh
```

## Task 3: Run Prompt Ablation

- [ ] Start the wrapper:

```bash
bash scripts/sentence_trace_method/run_aa_qec_stage2_c3_c4_qec_map_full_ministral3.sh
```

- [ ] After completion, confirm:
  - `training_complete.json` exists for both qec_map C3 and C4 runs.
  - `eval/val/best/label_token/metrics.json` exists.
  - `eval/test/best/label_token/metrics.json` exists.
  - tau eval artifacts exist when `RUN_TAU_EVAL=auto` applies.
- [ ] If val best is competitive, run macro-F1 Top3 checkpoint test eval for P1/P2 using the same checkpoint-selection pattern as the existing C3/C4 Top3 wrapper.
- [ ] Compare P1/P2 against:
  - C3 `qec_min` checkpoint-800.
  - C4 `qec_min` checkpoint-1700.
  - `v0_7_atom_facts_abc + qec_map` checkpoint-2400.
  - `v0_7_atom_facts_abc + qec_min` checkpoint-1600.

## Task 4: Implement Conservative Top20 Only If Prompt Ablation Justifies It

- [ ] Pick the prompt style from Batch P:
  - use `qec_map` only if it improves selected-scope val behavior or remains test-competitive without worse truncation.
  - otherwise keep `qec_min`.
- [ ] Add one selector policy first, G1:
  - selected primary and selected fallback stay preferred.
  - top20 outside-selected candidates can replace fallback only when selected fallback lacks QD cue or does not cover an uncovered atom.
  - no top20 secondary expansion.
- [ ] Add a unit test where a top20 candidate outside selected replaces a weak fallback and the selected primary remains unchanged.
- [ ] Run build gate for G1 only.
- [ ] Add G2/G3 only if G1 build diagnostics pass and improve prompt quality or coverage without higher truncation.

## Reporting Template

Use this compact table after Batch P completes:

| ID | Prompt | Val macro-F1 | Val selection | Test macro-F1 | Test selection | Best checkpoint | Tau used | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| C3 baseline | qec_min |  |  | 0.357041 | 0.774178 | 800 | off unless artifact says otherwise | selected-scope baseline |
| P1 | qec_map |  |  |  |  |  |  | prompt ablation |
| C4 baseline | qec_min |  |  | 0.347842 | 0.769782 | 1700 | off unless artifact says otherwise | no-secondary baseline |
| P2 | qec_map |  |  |  |  |  |  | prompt ablation |

Decision after Batch P:

- If P1 beats or matches C3 on val and test, make C3+best prompt the default selected-scope AA-QEC comparison.
- If P2 catches up to C3 or clearly beats C4, keep no-secondary as a viable conservative branch.
- If neither P1 nor P2 improves, stop prompt ablation and only consider G1 if the goal is interpretability or evidence-source diagnosis rather than immediate leaderboard improvement.
- If P1/P2 prompt truncation is materially higher, do not launch conservative top20 with `qec_map` until prompt length is reduced.

## Verification Commands

```bash
PYTHONPATH=.:src /data/liaozijie/conda/accelerate-fc/bin/python -m pytest \
  scripts/sentence_trace_method/test_experiment_matrix_scripts.py \
  src/fact_checking/selectors/test_atom_anchored_qec.py -v

bash -n scripts/sentence_trace_method/run_aa_qec_stage3_liar_raw_atom_facts_abc_ministral3.sh
bash -n scripts/sentence_trace_method/run_aa_qec_stage3_liar_raw_atom_facts_abc_f1_f3_full_ministral3.sh

DRY_RUN=true MODE=build bash scripts/sentence_trace_method/run_aa_qec_stage3_liar_raw_atom_facts_abc_ministral3.sh
DRY_RUN=true bash scripts/sentence_trace_method/run_aa_qec_stage3_liar_raw_atom_facts_abc_f1_f3_full_ministral3.sh
```

