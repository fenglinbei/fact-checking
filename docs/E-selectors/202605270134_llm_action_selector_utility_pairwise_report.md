# LLM Action Selector Utility Pairwise Smoke Report

Run directory: `outputs/selectors/llm_action_selector/qwen25_3b_utility_pairwise_v1_smoke`

## Verdict

This run is a no-go for Phase 3 acceptance. The trained selector improves set overlap over the retrieval control, but it does not learn a reliable ordered selector policy. On the full post-train val eval, `oracle_rank_ndcg@5` is below the retrieval/control ordering and slightly below the previous robust-prefix smoke.

The most important diagnostic is that `target_mode=utility` did not actually change the hard target in this data: for both train and val action samples, `target_idx == oracle_next_idx` for 100% of rows. Therefore the hard CE term is still oracle next-action imitation; only the multi-positive, pairwise, set, and soft losses provide additional utility signal.

## Configuration Check

- `target_mode=utility`
- no bad-prefix data
- train/val target mode in metadata: `utility`
- train samples: 10,129
- val samples used during action eval: 512
- full post-train selector eval: 512 claims
- best selection checkpoint: step 800, selected by `oracle_rank_ndcg@5`
- best action checkpoint: step 500, selected by `val_action_accuracy`

## Main Metrics

Full post-train val eval:

| model/control | recall@5 | jaccard@5 | top1_match | oracle_rank_ndcg@5 | pairwise_order_acc@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| utility-pairwise final | 0.3797 | 0.2567 | 0.0957 | 0.2776 | 0.4962 |
| robust-prefix final | 0.3910 | 0.2659 | 0.0664 | 0.2803 | 0.5149 |
| hybrid/candidate-pool control | 0.3465 | 0.2321 | 0.0938 | 0.2874 | 0.5495 |

The utility-pairwise run gains set overlap over the control (`jaccard@5 +0.0246`) but loses order quality (`oracle_rank_ndcg@5 -0.0098`, `pairwise_order_acc@5 -0.0533`). It also does not beat robust-prefix on the main selection metrics.

Saved-score utility ranker reference remains far ahead:

| reference | recall@5 | jaccard@5 | top1_match | oracle_rank_ndcg@5 | pairwise_order_acc@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| single_margin_step0_static, train-to-val | 0.5224 | 0.3761 | 1.0000 | 0.8082 | 0.8473 |
| ridge_all_step0_static, train-to-val | 0.5201 | 0.3747 | 0.6923 | 0.7130 | 0.7522 |

## Training Dynamics

The model does optimize the local training objective:

- train `target_accuracy`: 0.0781 at step 5 to 0.2813 at step 1590
- train `positive_hit@1`: 0.1406 to 0.5000
- train `oracle_remaining_hit@1`: 0.2031 to 0.6719

But the val action metrics saturate much lower:

- best `val_target_accuracy`: 0.1426 at step 500
- best `val_positive_hit@1`: 0.3379 at step 900
- best `val_oracle_remaining_hit@1`: 0.3457 at step 700
- final `val_target_accuracy`: 0.1172
- final `val_positive_hit@1`: 0.2930

During-train selection eval on the 128-claim sample also shows drift rather than stable selector learning:

- initial step 0: `oracle_rank_ndcg@5=0.3146`, `jaccard@5=0.2398`
- best saved step 800: `oracle_rank_ndcg@5=0.2889`, `jaccard@5=0.2637`
- final step 1590: `oracle_rank_ndcg@5=0.2651`, `jaccard@5=0.2454`

This is a set-overlap improvement at the cost of ranking quality. The initial base model was not selected as a checkpoint even though it had higher `oracle_rank_ndcg@5` on the during-train eval sample, so the checkpoint policy should be treated carefully when comparing this run.

## Data Diagnostics

Utility target construction did not create a new hard target:

| split | samples | `target_idx != oracle_next_idx` | mean positive set size | oracle in positive set |
| --- | ---: | ---: | ---: | ---: |
| train | 10,129 | 0.0000 | 3.18 | 1.0000 |
| val | 5,086 | 0.0000 | 3.25 | 1.0000 |

This means Phase 3 v1 changed the auxiliary ranking supervision, but not the main action target. The run is still anchored to oracle-prefix next-action imitation.

## Selector Behavior

The final rollout remains dominated by label/position preferences. On full val eval, selected actions concentrate heavily on a few local-choice labels:

- utility-pairwise: `I` 820, `A` 717, `J` 300, `F` 233 out of 2,546 steps
- robust-prefix: `G` 612, `J` 524, `I` 265, `A` 195 out of 2,546 steps

The utility run's action scores are also very flat:

- mean score range across choices: 0.273
- median top-1 margin: 0.0625
- tied-top rate: 32.4%
- oracle is top-scored in only 10.8% of action steps

This supports the current interpretation: the action-token path is still mostly learning small raw label-logit preferences, not a robust candidate-utility policy.

## Recommendation

Do not continue this branch by increasing hard CE or simply training longer. The current implementation is useful as a diagnostic, but the result does not satisfy Phase 3 acceptance.

Recommended next direction:

1. Add a `utility_only` or `ranking_only` ablation with `HARD_LOSS_WEIGHT=0`, stronger pairwise/set weights, and checkpoint selection by `jaccard@5` plus `oracle_rank_ndcg@5`.
2. Change the hard target only if it can genuinely differ from oracle next action; with current VIG rows it does not.
3. Add an explicit label-bias diagnostic or calibration eval for action-token logits, but treat it as diagnostic only.
4. Prefer a stronger utility scorer/listwise selector path for the next serious run, using the saved-score ranker gap as the target signal.
5. Move to DAgger-lite/bad-prefix utility labels only after the oracle-prefix utility scorer can beat robust-prefix and retrieval controls on both set and order metrics.

