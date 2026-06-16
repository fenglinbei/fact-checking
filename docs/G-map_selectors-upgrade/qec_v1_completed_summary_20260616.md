# QEC v1 Completed Summary - 2026-06-16

## Scope

This summary covers only the QEC v1 matrix defined in `qec_min_qec_map_method_and_experiments.md`:

- LIAR-RAW: B0 reused plain baseline, M1 `qec_min`, M2 `qec_map`.
- RAWFC: B3 reused plain baseline, M4 `qec_min`, M5 `qec_map`.
- Main selection split is `val`.
- LIAR-RAW main metric path is `eval/val/best/label_token_logit_adjust_tau0p75/metrics.json`.
- RAWFC main metric path is `eval/val/best/label_token/metrics.json`.

## Completion Status

All six matrix entries have `train/training_complete.json`, the expected main `val/best` metrics exist, and `train/latest_state/` is absent for all six entries after successful completion.

No QEC v1 `eval/test/.../metrics.json` files were found in this scan. Therefore the first-round val matrix is complete; test evaluation is still not present if test is required for a later final report.

| ID | Dataset | Prompt | Training/eval policy | Run policy | Completed | Global step | Step metrics | Best step by selection | Main metric | Test metric |
|---|---|---|---|---|---:|---:|---:|---:|---|---|
| B0 | LIAR-RAW | plain | existing ep12/eval100/pat8, tau0p75 eval | reused | true | 1800 | 18 | 1000 | yes | no |
| M1 | LIAR-RAW | qec_min | ebs16/lr2e-5/ep12/eval100/pat8, tau0p75 eval | new | true | 2200 | 22 | 1400 | yes | no |
| M2 | LIAR-RAW | qec_map | ebs16/lr2e-5/ep12/eval100/pat8, tau0p75 eval | new | true | 2100 | 21 | 1300 | yes | no |
| B3 | RAWFC | plain | existing baseline original policy | reused | true | 1212 | 12 | 1000 | yes | no |
| M4 | RAWFC | qec_min | ebs16/lr1e-5/ep10/eval50/pat8 | new | true | 700 | 14 | 300 | yes | no |
| M5 | RAWFC | qec_map | ebs16/lr1e-5/ep10/eval50/pat8 | new | true | 700 | 14 | 300 | yes | no |

## Main Val Metrics

| ID | Dataset | Prompt | Main metric path | Accuracy | Macro-F1 | True-side Macro-F1 | Selection | CE | baseline_reused | baseline_training_policy |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| B0 | LIAR-RAW | plain | `eval/val/best/label_token_logit_adjust_tau0p75/metrics.json` | 0.3579 | 0.3638 | 0.3530 | 0.7758 | 1.5261 | true | existing_plain_run |
| M1 | LIAR-RAW | qec_min | `eval/val/best/label_token_logit_adjust_tau0p75/metrics.json` | 0.3509 | 0.3618 | 0.3458 | 0.7689 | 1.6889 | false | new_qec_run |
| M2 | LIAR-RAW | qec_map | `eval/val/best/label_token_logit_adjust_tau0p75/metrics.json` | 0.3556 | 0.3650 | 0.3466 | 0.7731 | 1.7200 | false | new_qec_run |
| B3 | RAWFC | plain | `eval/val/best/label_token/metrics.json` | 0.6700 | 0.6707 | 0.3279 | 1.0746 | 2.8636 | true | existing_plain_run |
| M4 | RAWFC | qec_min | `eval/val/best/label_token/metrics.json` | 0.6950 | 0.6939 | 0.3390 | 1.1079 | 0.8374 | false | new_qec_run |
| M5 | RAWFC | qec_map | `eval/val/best/label_token/metrics.json` | 0.6700 | 0.6705 | 0.3248 | 1.0744 | 0.8574 | false | new_qec_run |

## Main Deltas

Use these deltas only as first-round val evidence. B0 and B3 are reused baselines, and B3 is not an ep10/eval50 same-batch rerun.

| Comparison | Accuracy delta | Macro-F1 delta | True-side Macro-F1 delta | Selection delta | CE delta |
|---|---:|---:|---:|---:|---:|
| M1 - B0 | -0.0071 | -0.0020 | -0.0071 | -0.0069 | +0.1628 |
| M2 - B0 | -0.0024 | +0.0011 | -0.0063 | -0.0027 | +0.1940 |
| M2 - M1 | +0.0047 | +0.0032 | +0.0008 | +0.0042 | +0.0311 |
| M4 - B3 | +0.0250 | +0.0232 | +0.0111 | +0.0332 | -2.0261 |
| M5 - B3 | +0.0000 | -0.0002 | -0.0031 | -0.0003 | -2.0062 |
| M5 - M4 | -0.0250 | -0.0234 | -0.0142 | -0.0335 | +0.0200 |

## Paired Significance

Method:

- Alignment: prediction records aligned by `sample_idx`.
- Accuracy test: two-sided exact McNemar on paired correctness disagreements.
- Metric intervals: paired percentile bootstrap over sample indices.
- Metric p-values: two-sided paired approximate randomization by sample-level prediction swaps.
- Resampling: 20,000 bootstrap samples and 20,000 randomization samples, seed `20260616`.
- Delta direction: `new - old`.
- Selection score formula: `macro_f1_plus_true_side_plus_mae` with `true_side_metric_weight=0.5` and `mae_metric_weight=0.3`; RAWFC uses equivalent id-level `true_side_weight=0.25` because the trainer averages missing `mostly-true` as 0 with `true`.

Readout: no comparison reaches conventional significance on the val split. All main 95% bootstrap CIs include 0, and McNemar / randomization p-values are well above 0.05. The observed differences should be treated as directional first-round evidence only.

| Comparison | n | Acc delta | Acc 95% CI | McNemar p | Macro-F1 delta | Macro-F1 95% CI | Macro-F1 rand p | Selection delta | Selection 95% CI | Selection rand p | Discordant new-only / old-only |
|---|---:|---:|---|---:|---:|---|---:|---:|---|---:|---:|
| M1 - B0 | 1274 | -0.0071 | [-0.0338, +0.0196] | 0.6448 | -0.0020 | [-0.0282, +0.0249] | 0.8847 | -0.0069 | [-0.0510, +0.0376] | 0.7672 | 146 / 155 |
| M2 - B0 | 1274 | -0.0024 | [-0.0283, +0.0235] | 0.9040 | +0.0011 | [-0.0239, +0.0266] | 0.9284 | -0.0027 | [-0.0448, +0.0391] | 0.8978 | 136 / 139 |
| M2 - M1 | 1274 | +0.0047 | [-0.0173, +0.0267] | 0.7289 | +0.0032 | [-0.0179, +0.0241] | 0.7736 | +0.0042 | [-0.0324, +0.0410] | 0.8308 | 107 / 101 |
| M4 - B3 | 200 | +0.0250 | [-0.0350, +0.0850] | 0.4996 | +0.0232 | [-0.0346, +0.0818] | 0.4604 | +0.0332 | [-0.0488, +0.1170] | 0.4256 | 20 / 15 |
| M5 - B3 | 200 | +0.0000 | [-0.0550, +0.0550] | 1.0000 | -0.0002 | [-0.0555, +0.0554] | 0.9767 | -0.0003 | [-0.0815, +0.0814] | 0.9947 | 16 / 16 |
| M5 - M4 | 200 | -0.0250 | [-0.0700, +0.0200] | 0.3833 | -0.0234 | [-0.0685, +0.0218] | 0.3525 | -0.0335 | [-0.0998, +0.0335] | 0.3234 | 8 / 13 |

Significance outputs:

- LIAR-RAW calibrated: `outputs/sentence_trace_method/analysis/qec_v1_paired_significance_liar_val_tau0p75_calibrated_20260616.json`
- RAWFC calibrated: `outputs/sentence_trace_method/analysis/qec_v1_paired_significance_rawfc_val_label_token_calibrated_20260616.json`
- LIAR-RAW default script setting: `outputs/sentence_trace_method/analysis/qec_v1_paired_significance_liar_val_tau0p75_20260616.json`
- RAWFC default script setting: `outputs/sentence_trace_method/analysis/qec_v1_paired_significance_rawfc_val_label_token_20260616.json`

## Build And Smoke Checks

QEC build rows preserve the same claim/label order, `evidence_count_before`, and `selector_trace` as the plain build for each dataset/split. The `candidates` text differs by design because QEC prepends the `Check:` view to verifier evidence.

| Dataset | Split | Plain rows | qec_min rows | qec_map rows | qec_min selector_trace same | qec_map selector_trace same | qec_min evidence_count_after diff | qec_map evidence_count_after diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LIAR-RAW | train | 10065 | 10065 | 10065 | 10065/10065 | 10065/10065 | 22 | 82 |
| LIAR-RAW | val | 1274 | 1274 | 1274 | 1274/1274 | 1274/1274 | 4 | 16 |
| LIAR-RAW | test | 1251 | 1251 | 1251 | 1251/1251 | 1251/1251 | 5 | 17 |
| RAWFC | train | 1612 | 1612 | 1612 | 1612/1612 | 1612/1612 | 2 | 11 |
| RAWFC | val | 200 | 200 | 200 | 200/200 | 200/200 | 0 | 0 |
| RAWFC | test | 200 | 200 | 200 | 200/200 | 200/200 | 0 | 2 |

Prompt-length and truncation checks:

| Build | Val prompt mean | Val prompt p95 | Val truncation | Test truncation |
|---|---:|---:|---:|---:|
| LIAR plain | 441.1 | 595.7 | 0.0000 | 0.0016 |
| LIAR qec_min | 552.8 | 734.3 | 0.0031 | 0.0048 |
| LIAR qec_map | 633.1 | 813.3 | 0.0126 | 0.0144 |
| RAWFC plain | 371.4 | 507.6 | 0.0000 | 0.0000 |
| RAWFC qec_min | 491.4 | 666.7 | 0.0000 | 0.0000 |
| RAWFC qec_map | 579.6 | 755.2 | 0.0000 | 0.0100 |

## Readout

- LIAR-RAW: under the agreed main `tau0p75` policy, M2 is slightly better than M1, but both are very close to reused B0. M2 has slightly higher Macro-F1 than B0 but lower accuracy/selection; this is not enough for a strong win claim.
- RAWFC: M4 `qec_min` is the best first-round val result. M5 `qec_map` is effectively at reused B3 level and below M4 by about 2.5 accuracy points / 2.34 Macro-F1 points.
- qec_map consistently makes prompts longer. On RAWFC val it does not truncate evidence, so the RAWFC M5 underperformance is not explained by val evidence truncation. On LIAR val there is mild truncation for qec_map, so LIAR conclusions should keep the prompt-length caveat.
- Since B0/B3 are reused baselines, especially B3 historical reused baseline, small deltas should not be overinterpreted without a same-policy plain sanity rerun.

## Source Run Roots

- B0: `outputs/sentence_trace_method/liar_raw__ministral3_8b__v0_7_bm_adaptive5_10_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw`
- M1: `outputs/sentence_trace_method/liar_raw__ministral3_8b__v0_7_bm_adaptive5_10__qec_min_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw`
- M2: `outputs/sentence_trace_method/liar_raw__ministral3_8b__v0_7_bm_adaptive5_10__qec_map_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw`
- B3: `outputs/sentence_trace_method/rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_ebs16_lr1em5_ep12_eval100_pat8_rawfc`
- M4: `outputs/sentence_trace_method/rawfc__ministral3_8b__v0_7_bm_adaptive5_10__qec_min_lora_ebs16_lr1em5_ep10_eval50_pat8_rawfc`
- M5: `outputs/sentence_trace_method/rawfc__ministral3_8b__v0_7_bm_adaptive5_10__qec_map_lora_ebs16_lr1em5_ep10_eval50_pat8_rawfc`
