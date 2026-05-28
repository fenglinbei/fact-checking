# v0.5b Map-Aware Verifier Checkpoint Diagnostic

## Goal

v0.5b is an eval-only classification diagnostic for the evidence-map selector line. It asks:

1. whether map-aware prompts improve final LIAR-RAW label classification;
2. which existing oracle-direct verifier checkpoint is most stable on map-aware evidence;
3. whether the answer differs from the original oracle-evidence validation checkpoint choice.

No new selector training, DeepSeek annotation, retrieval, or oracle search is run in v0.5b.

## Inputs

- v0.5a selection trace:
  `outputs/selectors/evidence_map_selector/v0_5a_val/selection_trace_val.jsonl`
- existing oracle-direct verifier checkpoints:
  `outputs/oracle_direct_verifier/stage2_sentence/train/b3_oracle_sentence_direct_verifier_1024_20260519-200709/`
- default base experiment config:
  `configs/experiment/b3_oracle_sentence_direct_verifier_1024.yaml`

The wrapper rebuilds verifier JSONL prompts from the v0.5a trace, one directory per selector variant, then evaluates existing checkpoints with the HF/PEFT torch-forward label-token path.

## Default Selectors

The default selector comparison is deliberately small:

- `v0_5a_evidence_map_top5`
- `v0_5a_base_only_top5`
- `fusion_refit_all_features_plus_direct_ce_top5`

This tests the main question: full map greedy versus score-only map prompt versus the strongest prior fusion selector rendered in the same map-aware format.

For a broader sweep, set:

```bash
SELECTORS=v0_5a_evidence_map_top5,v0_5a_base_only_top5,v0_5a_coverage_only_top5,fusion_refit_all_features_plus_direct_ce_top5,oracle_likelihood_top5,direct_ce_text_only_top5,original_pool_order_top5,qd_union_source_score_top5
```

## Checkpoint Prior

Original oracle-direct verifier validation metrics favor the late checkpoint:

| checkpoint | original val accuracy | original val macro-F1 | selection score |
| --- | ---: | ---: | ---: |
| `checkpoint-600` | 0.7125 | 0.7183 | 1.0937 |
| `checkpoint-550` | 0.7039 | 0.7092 | 1.0813 |
| `checkpoint-500` | 0.7039 | 0.7099 | 1.0778 |
| `checkpoint-450` | 0.7039 | 0.7108 | 1.0747 |
| `checkpoint-350` | 0.6953 | 0.7009 | 1.0760 |

Therefore the default v0.5b checkpoint sweep is:

```bash
CHECKPOINTS=best,checkpoint-600,checkpoint-550,checkpoint-500,checkpoint-450
```

`best` is expected to match the best saved selection-score checkpoint, but v0.5b keeps it explicit to catch any checkpoint-copy mismatch.

## Run

Smoke the command shape first:

```bash
DRY_RUN=true bash scripts/phase5_selectors/run/run_evidence_map_verifier_v0_5b.sh
```

On the remote GPU server, run:

```bash
CUDA_VISIBLE_DEVICES=0 \
PROMPT_MODEL_NAME_OR_PATH=/data/models/Qwen2.5-7B-Instruct \
TRAIN_MODEL_NAME_OR_PATH=/data/models/Qwen2.5-7B-Instruct \
BASE_MODEL_NAME_OR_PATH=/data/models/Qwen2.5-7B-Instruct \
bash scripts/phase5_selectors/run/run_evidence_map_verifier_v0_5b.sh
```

If the model root differs, replace the three model paths. `MODEL_BASE_PATH` is also supported for rewriting `/data/models/...` config paths, but explicit `*_MODEL_NAME_OR_PATH` is clearer when moving between servers.

## Outputs

Default output root:

`outputs/selectors/evidence_map_selector/v0_5b_val_map_verifier/`

Important files:

- `verifier_data/<selector>/build_val.jsonl`
- `verifier_data/<selector>/train.resolved.yaml`
- `eval/<selector>/<checkpoint>/metrics.json`
- `eval/<selector>/<checkpoint>/val_predictions.jsonl`
- `checkpoint_comparison.json`
- `analysis_summary.md`

## Decision Rule

Primary metric: `macro_f1`.

Use `accuracy` as a sanity check and `true_side_macro_f1` to see whether the map prompt helps mostly-true/true claims without collapsing false-side labels.

Recommended interpretation:

- If `v0_5a_evidence_map_top5` beats `fusion_refit_all_features_plus_direct_ce_top5` by at least `+0.01` macro-F1, continue the map-aware verifier direction.
- If score-only/fusion evidence wins but map prompt still improves over the old non-map verifier prompt, keep map rendering and retune selector weights.
- If every map-aware prompt is below the original oracle-direct verifier by a large margin, treat v0.5a as explanation-only and move to train-side map generation before judging final utility.

Because current v0.5a artifacts are val-only, this remains a val diagnostic rather than held-out evidence.

## Full-Val Readout

Measured output:

`outputs/selectors/evidence_map_selector/v0_5b_val_map_verifier/`

Best row:

| selector | checkpoint | accuracy | macro-F1 | true-side macro-F1 | selection score |
| --- | --- | ---: | ---: | ---: | ---: |
| `v0_5a_base_only_top5` | `best` / `checkpoint-600` | 0.2943 | 0.2842 | 0.3295 | 0.4489 |

Selector comparison at `best`:

| selector | accuracy | macro-F1 | true-side macro-F1 |
| --- | ---: | ---: | ---: |
| `v0_5a_base_only_top5` | 0.2943 | 0.2842 | 0.3295 |
| `v0_5a_evidence_map_top5` | 0.2732 | 0.2630 | 0.3134 |
| `fusion_refit_all_features_plus_direct_ce_top5` | 0.2716 | 0.2618 | 0.3272 |

Checkpoint choice:

- `best` and `checkpoint-600` are identical on the best selector row, so the checkpoint recommendation remains `best` / `checkpoint-600`.
- Earlier checkpoints do not rescue the map-aware prompt distribution; the spread is small relative to the oracle-direct prior.

Interpretation:

- v0.5b is **not a classification Go** for directly applying the old oracle-direct verifier to map-aware prompts.
- The best map-aware result, 0.2943 accuracy / 0.2842 macro-F1, is only slightly above the historical fixed-MMR + oracle-direct verifier region and far below oracle evidence + oracle-direct verifier.
- `v0_5a_evidence_map_top5` underperforms `base_only`, despite better explainability metrics in v0.5a. This means the old verifier is not benefiting from atom/relation/directness metadata as currently rendered.
- The likely failure mode is verifier prompt-distribution mismatch plus remaining evidence-distribution gap: the verifier was trained on plain numbered evidence, not structured evidence maps with relation/directness annotations.

Recommended next step:

1. Build train-side map artifacts before judging the map prompt itself.
2. Train a small map-aware verifier LoRA using the same rendered prompt format.
3. Keep `v0_5a_base_only_top5` as the strongest eval-only evidence selector for this prompt family.
4. Treat `v0_5a_evidence_map_top5` as explanation/rationale data, not as the current classifier-facing top5 selector.
