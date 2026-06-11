# Coverage v2 Flash covered_weak First-Round LoRA Result

## Question

Does the coverage v2 Flash `covered_weak` LoRA run beat the old prior-selected LoRA baseline on the same full-val and full-test eval set?

Decision scope is limited to eval-only `tau0`, `tau0p5`, and `tau0p75`. The old run's prior selected/default tau is `tau0p5` from validation selection. The new run selects `tau0` by full-val `selection_score`; new test `tau0p5` is reported only as a post-hoc/oracle result.

Primary rule from the plan:

- First-round win if the new selected tau improves full-test `selection_score` or `macro_f1` by at least `0.005`, and full-val `selection_score` does not drop by more than `0.005`.
- Neutral if both full-val and full-test are within `+/-0.005`.
- Regression branch if full-val drops by more than `0.005` and full-test macro-F1 also drops; Task 7 only if the distribution suggests class weights are over-correcting.

## Setup

- Old baseline root: `outputs/sentence_trace_method/liar_raw__llama31_8b_lora_halfbatch_ep8_eval100_pat8_liarw`
- New run root: `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw`
- Eval splits: full validation set with 1,274 rows and full test set with 1,251 rows.
- Metrics read from `eval/{val,test}/best/label_token_logit_adjust_{tau}/metrics.json`.
- Predicted label distributions were computed from `val_predictions.jsonl` and `test_predictions.jsonl`; the metrics JSON files do not store those distributions.
- Parse error rate is `0.0` for every reported old/new tau and split.

## Build Checks

- Build rows: train `5163`, val `1274`, test `1251`.
- Skipped train rows: `missing_raw_sample=4902`.
- Training best checkpoint before the eval-only tau sweep: step `2300`.
- Training best full-val `selection_score`: `0.7098113986267629`.
- The step-2300 training eval artifact records full-val `macro_f1=0.31991165791828396`, `true_side_macro_f1=0.32287640449388116`, and `selection_score=0.7098113986267629` before applying eval-only tau variants.

## Full-Val Selection

Full-val selection picks new `tau0`, not `tau0p5`: `tau0` has the best new full-val `selection_score` at `0.693627`. The old prior selected/default tau remains `tau0p5`, with full-val `selection_score=0.691147`.

| Case | Tau | N | Accuracy | Macro-F1 | True-side F1 | Selection | Eval loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| old | tau0 | 1274 | 0.310832 | 0.302258 | 0.292065 | 0.678024 | 1.579702 |
| old | tau0p5 | 1274 | 0.304553 | 0.309998 | 0.308859 | 0.691147 | 1.579702 |
| old | tau0p75 | 1274 | 0.293564 | 0.301323 | 0.314551 | 0.683245 | 1.579702 |
| new | tau0 | 1274 | 0.311617 | 0.309596 | 0.313022 | 0.693627 | 1.569038 |
| new | tau0p5 | 1274 | 0.302198 | 0.308326 | 0.314485 | 0.690027 | 1.569038 |
| new | tau0p75 | 1274 | 0.297488 | 0.302884 | 0.314449 | 0.683154 | 1.569038 |

Against the old prior-selected `tau0p5`, the new selected `tau0` full-val `selection_score` is higher by `+0.002479`, so the validation side does not violate the plan's maximum-drop guardrail. However, the val gain is small and below the `0.005` win threshold by itself.

## Full-Test Result

The selected new `tau0` does not beat the old prior-selected `tau0p5` on full test. It is slightly higher on accuracy, but lower on macro-F1, true-side macro-F1, and selection score.

| Case | Tau | N | Accuracy | Macro-F1 | True-side F1 | Selection | Eval loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| old | tau0 | 1251 | 0.304556 | 0.306960 | 0.284995 | 0.679242 | 1.582587 |
| old | tau0p5 | 1251 | 0.310152 | 0.323757 | 0.329735 | 0.717161 | 1.582587 |
| old | tau0p75 | 1251 | 0.312550 | 0.324137 | 0.342651 | 0.722417 | 1.582587 |
| new | tau0 | 1251 | 0.311751 | 0.314026 | 0.321554 | 0.702716 | 1.586295 |
| new | tau0p5 | 1251 | 0.315747 | 0.325089 | 0.342427 | 0.721675 | 1.586295 |
| new | tau0p75 | 1251 | 0.316547 | 0.321587 | 0.343755 | 0.717493 | 1.586295 |

Key comparisons:

| Comparison | Selection delta | Macro-F1 delta | Accuracy delta | True-side F1 delta | Interpretation |
|---|---:|---:|---:|---:|---|
| new selected `tau0` vs old selected `tau0p5` | -0.014445 | -0.009731 | +0.001599 | -0.008181 | Fails first-round win; outside neutral band on test selection and macro-F1. |
| new post-hoc `tau0p5` vs old selected `tau0p5` | +0.004513 | +0.001332 | +0.005596 | +0.012692 | Oracle-only neutral on the primary threshold; not selected by full-val. |
| new selected `tau0` vs old `tau0` | +0.023475 | +0.007066 | +0.007194 | +0.036559 | Improvement only against old `tau0`, not against the old prior-selected baseline. |

New `tau0p5` is the higher new test result, but because full-val selected `tau0`, it should be treated as post-hoc/oracle. It is also only `+0.004513` over old selected `tau0p5` on test `selection_score`, just below the `0.005` first-round threshold.

## Per-Class Effects

Full-test predicted distribution, using old prior-selected `tau0p5`, new selected `tau0`, and new post-hoc `tau0p5`:

| Case | Tau | pants-fire | false | barely-true | half-true | mostly-true | true |
|---|---:|---:|---:|---:|---:|---:|---:|
| gold | test | 86 | 249 | 210 | 263 | 238 | 205 |
| old | tau0p5 | 73 | 263 | 174 | 224 | 337 | 180 |
| new selected | tau0 | 54 | 349 | 85 | 293 | 272 | 198 |
| new post-hoc | tau0p5 | 85 | 253 | 126 | 219 | 281 | 287 |

Selected new `tau0` shifts predictions away from `pants-fire` and `barely-true` and toward `false` and `half-true`. That helps the false/half-true guardrail, but it hurts two weak classes enough to pull down full-test macro-F1.

| Label | Gold N | Old `tau0p5` pred / F1 | New selected `tau0` pred / F1 | F1 delta | Effect |
|---|---:|---:|---:|---:|---|
| pants-fire | 86 | 73 / 0.452830 | 54 / 0.414286 | -0.038545 | Worse recall and fewer predictions; weak-class recovery regresses. |
| barely-true | 210 | 174 / 0.250000 | 85 / 0.176271 | -0.073729 | Largest focused-class loss; predictions collapse relative to old selected tau. |
| true | 205 | 180 / 0.322078 | 198 / 0.317618 | -0.004460 | Roughly neutral F1, with higher recall but lower precision. |
| false | 249 | 263 / 0.296875 | 349 / 0.341137 | +0.044262 | Guardrail improves, mostly through higher recall and many more false predictions. |
| half-true | 263 | 224 / 0.283368 | 293 / 0.309353 | +0.025985 | Guardrail improves, again with more predictions and higher recall. |

This does not look like class weights over-correcting toward rare or extreme classes. The selected new run underpredicts `pants-fire` and especially `barely-true`; its main selected-tau shift is into `false` and `half-true`.

## Paired Significance Checks

Significance checks were added for the two full-test comparisons that matter for the first-round decision and the calibration follow-up. The reproducible output is `outputs/sentence_trace_method/liar_raw__llama31_8b_covv2flash_cwtrain_fulleval_lora_halfbatch_ep8_eval50_pat8_liarw/analysis/paired_significance_test.json`.

Method:

- Accuracy uses a two-sided exact McNemar test over paired correctness disagreements.
- Macro-F1 and `selection_score` use paired bootstrap 95% CIs over sample indices plus a two-sided paired approximate-randomization p-value from sample-level prediction swaps.
- Bootstrap and randomization both use `10000` resamples with seed `20260611`.
- `selection_score` is recomputed with the run config formula `macro_f1 + 0.5 * true_side_macro_f1 + 0.3 * (1 - ordinal_mae_norm)`, matching the metrics JSON files.

| Comparison | Metric | Delta | 95% paired bootstrap CI | Paired p-value | Interpretation |
|---|---:|---:|---:|---:|---|
| new selected `tau0` vs old selected `tau0p5` | accuracy | +0.001599 | [-0.017586, +0.020803] | McNemar p=0.934525 | No detectable accuracy difference. |
| new selected `tau0` vs old selected `tau0p5` | macro-F1 | -0.009731 | [-0.029408, +0.009998] | randomization p=0.338766 | Negative point estimate, but CI crosses zero. |
| new selected `tau0` vs old selected `tau0p5` | selection | -0.014445 | [-0.044339, +0.015631] | randomization p=0.357364 | Fails the practical win rule, but not statistically significant at 0.05. |
| new post-hoc `tau0p5` vs old selected `tau0p5` | accuracy | +0.005596 | [-0.011990, +0.023981] | McNemar p=0.597485 | Small positive point estimate, not significant. |
| new post-hoc `tau0p5` vs old selected `tau0p5` | macro-F1 | +0.001332 | [-0.016205, +0.019151] | randomization p=0.882912 | Indistinguishable from noise. |
| new post-hoc `tau0p5` vs old selected `tau0p5` | selection | +0.004513 | [-0.026332, +0.036037] | randomization p=0.779022 | The post-hoc lift is not statistically reliable. |

The significance layer does not change the first-round decision. It does soften the interpretation: the selected new run is worse under the pre-registered practical threshold, but this run alone does not establish a statistically significant degradation. The post-hoc `tau0p5` signal also remains too small and uncertain to count as evidence of improvement.

## Decision

Decision: no first-round win against the old prior-selected baseline.

The new selected `tau0` satisfies the full-val guardrail because validation selection is `+0.002479` versus old selected `tau0p5`. It fails the required full-test improvement condition: test `selection_score` is `-0.014445` and test `macro_f1` is `-0.009731` versus old selected `tau0p5`.

This is also not neutral under the plan because full-test selection and macro-F1 are outside the `+/-0.005` band. It is not the specific regression branch that requires full-val to drop by more than `0.005`; validation did not drop. The practical decision is still negative for first-round adoption.

The post-hoc new `tau0p5` result is worth noting but not selecting: it is only `+0.004513` on test `selection_score` and `+0.001332` on test `macro_f1` versus old selected `tau0p5`, and it was not chosen by the new full-val selection rule.

## Next Step

Do not advance the new selected run as a first-round win.

Task 7 is not indicated by the current decision rule unless a separate distribution audit is requested. The selected distribution does not show class-weight over-correction into rare/extreme labels; it shows underprediction of `pants-fire` and `barely-true` with a shift into `false` and `half-true`.

If continuing, the most useful next check is selection calibration: why full-val selects `tau0` while full-test would prefer a more adjusted post-hoc tau. Do not count the new `tau0p5` test result as selected performance.
