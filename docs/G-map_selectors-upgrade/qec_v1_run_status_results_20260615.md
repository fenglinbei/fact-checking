# QEC v1 run status and results snapshot

Scan time: 2026-06-15, Asia/Shanghai.

Scope: first-version QEC matrix only: B0 / M1 / M2 / B3 / M4 / M5.

Metric policy:

- LIAR-RAW main metric: `eval/val/best/label_token_logit_adjust_tau0p75/metrics.json`.
- RAWFC main metric: `eval/val/best/label_token/metrics.json`.
- B0/B3 are reused plain baselines. B3 is a historical reused baseline and is not an ep10/eval50 same-batch rerun.
- Incomplete runs list best-so-far step metrics only as a progress readout; they are not final comparable metrics.

## Completed Main-Metric Runs

| ID | Dataset | Prompt | Policy | Status | Step | Acc | Macro-F1 | Selection score | CE loss | Baseline reused |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| B0 | LIAR-RAW | plain | ep12/eval100/pat8, tau0p75 | complete | 1800 | 0.3579 | 0.3638 | 0.7758 | 1.5261 | true |
| B3 | RAWFC | plain | existing baseline ep12/eval100/pat8 | complete | 1212 | 0.6700 | 0.6707 | 1.0746 | 2.8636 | true |
| M4 | RAWFC | qec_min | ebs16/lr1e-5/ep10/eval50/pat8 | complete | 700 | 0.6950 | 0.6939 | 1.1079 | 0.8374 | false |
| M5 | RAWFC | qec_map | ebs16/lr1e-5/ep10/eval50/pat8 | complete | 700 | 0.6700 | 0.6705 | 1.0744 | 0.8574 | false |

## Incomplete Or Not Started

| ID | Dataset | Prompt | Policy | Status | Progress | Best-so-far metric source | Acc | Macro-F1 | Selection score | CE loss | Notes |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|
| M1 | LIAR-RAW | qec_min | ebs16/lr2e-5/ep12/eval100/pat8, tau0p75 | incomplete_or_paused | 1300/7548 | `eval/step-1200/metrics.json` | 0.3564 | 0.3512 | 0.7632 | 1.5263 | `training_complete.json` missing; `latest_state/trainer_state.json` has `completed=false`; no final tau0p75 val metric yet. |
| M2 | LIAR-RAW | qec_map | ebs16/lr2e-5/ep12/eval100/pat8, tau0p75 | not_started |  |  |  |  |  |  | Build/run directory not present. |

## Build Row Check

| ID | Build rows |
|---|---|
| B0 | train=10065, val=1274, test=1251 |
| M1 | train=10065, val=1274, test=1251 |
| M2 | missing |
| B3 | train=1612, val=200, test=200 |
| M4 | train=1612, val=200, test=200 |
| M5 | train=1612, val=200, test=200 |

## Metric Paths

| ID | Metric path used |
|---|---|
| B0 | `outputs/sentence_trace_method/liar_raw__ministral3_8b__v0_7_bm_adaptive5_10_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/eval/val/best/label_token_logit_adjust_tau0p75/metrics.json` |
| B3 | `outputs/sentence_trace_method/rawfc__ministral3_8b__v0_7_bm_adaptive5_10_lora_ebs16_lr1em5_ep12_eval100_pat8_rawfc/eval/val/best/label_token/metrics.json` |
| M4 | `outputs/sentence_trace_method/rawfc__ministral3_8b__v0_7_bm_adaptive5_10__qec_min_lora_ebs16_lr1em5_ep10_eval50_pat8_rawfc/eval/val/best/label_token/metrics.json` |
| M5 | `outputs/sentence_trace_method/rawfc__ministral3_8b__v0_7_bm_adaptive5_10__qec_map_lora_ebs16_lr1em5_ep10_eval50_pat8_rawfc/eval/val/best/label_token/metrics.json` |
| M1 best-so-far | `outputs/sentence_trace_method/liar_raw__ministral3_8b__v0_7_bm_adaptive5_10__qec_min_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/eval/step-1200/metrics.json` |

## Operational Notes

- RAWFC first-pass QEC result: M4 is ahead of reused B3 on val under the agreed main metric path; M5 is effectively aligned with B3 and below M4.
- LIAR-RAW QEC is not ready for final comparison: M1 is incomplete and M2 has not started.
- Process scan during this snapshot did not show an active `label_token_trainer` / QEC matrix process; M1 appears paused rather than actively training.
