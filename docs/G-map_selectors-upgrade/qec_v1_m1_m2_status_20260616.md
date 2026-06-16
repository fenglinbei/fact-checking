# QEC v1 M1/M2 status snapshot

Scan time: 2026-06-16, Asia/Shanghai.

Scope: LIAR-RAW QEC runs only.

Main metric policy: `eval/val/best/label_token_logit_adjust_tau0p75/metrics.json`.

## Completion Status

| ID | Prompt | Training status | Global step | Best training step | Main val metric exists | Test metric exists |
|---|---|---|---:|---:|---|---|
| M1 | qec_min | complete | 2200 | 1400 | yes | no |
| M2 | qec_map | complete | 2100 | 1300 | yes | no |

## Main Val Metrics

| ID | Prompt | Acc | Macro-F1 | True-side macro-F1 | Selection score | CE loss |
|---|---|---:|---:|---:|---:|---:|
| M1 | qec_min | 0.3509 | 0.3618 | 0.3458 | 0.7689 | 1.6889 |
| M2 | qec_map | 0.3556 | 0.3650 | 0.3466 | 0.7731 | 1.7201 |

## Plain Label-Token Val Metrics

| ID | Prompt | Acc | Macro-F1 | True-side macro-F1 | Selection score | CE loss |
|---|---|---:|---:|---:|---:|---:|
| M1 | qec_min | 0.3603 | 0.3650 | 0.3540 | 0.7779 | 1.6889 |
| M2 | qec_map | 0.3666 | 0.3751 | 0.3366 | 0.7805 | 1.7201 |

## Build Check

| ID | Prompt | Train rows | Val rows | Test rows | Val truncation | Val token mean | Val token p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| M1 | qec_min | 10065 | 1274 | 1251 | 0.0031 | 552.8 | 734.3 |
| M2 | qec_map | 10065 | 1274 | 1251 | 0.0126 | 633.1 | 813.3 |

## Metric Paths

| ID | Metric path |
|---|---|
| M1 main | `outputs/sentence_trace_method/liar_raw__ministral3_8b__v0_7_bm_adaptive5_10__qec_min_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/eval/val/best/label_token_logit_adjust_tau0p75/metrics.json` |
| M2 main | `outputs/sentence_trace_method/liar_raw__ministral3_8b__v0_7_bm_adaptive5_10__qec_map_lora_ebs16_lr2em5_ep12_eval100_pat8_liarw/eval/val/best/label_token_logit_adjust_tau0p75/metrics.json` |

## Notes

- Both M1 and M2 have `training_complete.json` with `completed=true`; `latest_state` was removed after completion.
- Only val metrics are present under the agreed first-round policy. No test metrics were found for M1/M2.
- Only `tau0p75` logit-adjust final metrics are present; `tau0` and `tau0p5` final folders were not found for M1/M2.
- A process scan did not show active M1/M2 or `run_qec_v1` processes at scan time.
