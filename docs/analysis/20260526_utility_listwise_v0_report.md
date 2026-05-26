# Utility Listwise v0 Step0 Static Report

Run directory: `outputs/selectors/utility_listwise/deberta_v0_step0_static`

## Verdict

This v0 run is useful as a sanity baseline, but it is a no-go as the next selector direction.

It passes the weak runtime/sanity bar: the training ran on full train/val, wrote metrics and traces, and the best saved eval beats the retrieval/candidate-pool control slightly on set overlap. But it fails the substantive utility-scorer bar: row-level delta ranking is almost random, order metrics are worse than the retrieval control, and the result is far behind the saved-score static rankers.

The clearest failure mode is positional collapse. In `val_trace.jsonl`, the model chooses `candidate_idx=12` as top-1 for 994 / 1274 claims, while the true best-delta candidate is broadly distributed across the pool. This means the frozen-encoder + set-head v0 is mostly learning a position template, not verifier utility.

## Run Configuration

- base model: `/data/models/deberta-v3-base/`
- train examples: 10,065
- val examples: 1,274
- step filter: `step=0`
- group key: `event_id`
- max candidates: 15
- pair encoder: frozen
- losses: pairwise delta `1.0`, soft CE `0.2`, positive BCE `0.2`
- early-stop metric: `jaccard@5`
- best checkpoint by metric: step 200, epoch 1
- total train steps: 3,777

Note: the synced directory currently contains metrics, tokenizer/config files, and traces, but not `listwise_head.pt` or model weight files. It is enough for analysis, but not enough to reload this selector checkpoint locally.

## Main Metrics

Best saved eval from `selection_metrics.json`:

| method | n | recall@5 | jaccard@5 | top1_match | oracle_rank_ndcg@5 | pairwise_order_acc@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| utility_listwise_v0 best | 1274 | 0.3597 | 0.2410 | 0.0612 | 0.2585 | 0.4776 |
| hybrid/candidate-pool control | 1274 | 0.3435 | 0.2294 | 0.1028 | 0.2872 | 0.5271 |
| single_margin_step0_static | 1274 | 0.5224 | 0.3761 | 1.0000 | 0.8082 | 0.8473 |
| ridge_all_step0_static | 1274 | 0.5201 | 0.3747 | 0.6923 | 0.7130 | 0.7522 |

Delta against retrieval control:

- `recall@5`: +0.0162
- `jaccard@5`: +0.0116
- `top1_match`: -0.0416
- `oracle_rank_ndcg@5`: -0.0288
- `pairwise_order_acc@5`: -0.0495

So the model finds a few more oracle-set items, but orders them worse and identifies the best utility candidate less often than the control.

The comparison to prior LLM action-selector smokes is not strictly aligned because those evals used 512 claims rather than the full 1274-claim val set. Still, v0 is not directionally better: robust-prefix smoke reported `jaccard@5=0.2659`, `oracle_rank_ndcg@5=0.2803`; utility-pairwise smoke reported `jaccard@5=0.2567`, `oracle_rank_ndcg@5=0.2776`.

## Training Dynamics

The best set metric appears very early:

| checkpoint | jaccard@5 | recall@5 | oracle_rank_ndcg@5 | row_pairwise_acc | row_spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| step 100 | 0.2351 | 0.3524 | 0.2718 | 0.5082 | 0.0216 |
| step 200, best jaccard | 0.2410 | 0.3597 | 0.2585 | 0.5080 | 0.0198 |
| step 3777, final eval | 0.2310 | 0.3466 | 0.2630 | 0.5138 | 0.0382 |

Training longer did not unlock utility learning. Row-level pairwise accuracy only moved from about 0.508 to 0.514, and Spearman stayed near zero. The loss decreased a little, but the validation ranking signal remained weak.

## Row-Level Utility Learning

Best saved row metrics:

- `pairwise_accuracy`: 0.5080
- `top1_delta_match`: 0.0612
- `positive_hit@1`: 0.3108
- `pearson`: 0.0250
- `spearman`: 0.0198

This is the main no-go evidence. A useful utility scorer should first show nontrivial agreement with `delta_margin`; here it does not.

The VIG target itself is not degenerate. On the same val split, saved-score probes show much stronger signal:

- `single_margin` row Spearman against delta: 0.4250
- `single_gold_logprob` row Spearman against delta: 0.3713
- `ridge_all_step0_static` selector `jaccard@5`: 0.3747
- `single_margin_step0_static` selector `jaccard@5`: 0.3761

So the target has signal, but this frozen DeBERTa v0 did not extract it.

## Selector Behavior

The predicted top positions are heavily collapsed:

| predicted top-1 candidate_idx | count |
| ---: | ---: |
| 12 | 994 |
| 11 | 232 |
| 3 | 35 |
| 0 | 11 |
| 1 | 2 |

By contrast, the true best-delta candidate is much more evenly spread:

| true best candidate_idx | count |
| ---: | ---: |
| 0 | 131 |
| 2 | 110 |
| 1 | 109 |
| 3 | 94 |
| 13 | 93 |

This points to a model/feature shortcut rather than learned candidate semantics. The current model config includes both `candidate_idx_norm` and rank embedding (`use_rank_embedding=true`), so the first diagnostic should remove these rank/position priors before scaling the model.

Label-level split also shows the improvement is not robust:

| label | jaccard delta vs control | ndcg delta vs control |
| --- | ---: | ---: |
| false | +0.0427 | +0.0250 |
| barely-true | +0.0215 | -0.0138 |
| pants-fire | +0.0171 | -0.0136 |
| half-true | +0.0046 | +0.0035 |
| mostly-true | -0.0096 | -0.1089 |
| true | -0.0121 | -0.0700 |

## Recommendation

Do not move this v0 checkpoint into the pipeline, and do not spend the next run only increasing epochs or hardening the current loss. The bottleneck is not lack of training time; it is that the frozen encoder/set head is not learning delta utility and is falling into position shortcuts.

Recommended next queue:

1. Run a v0.1 no-rank-prior diagnostic. Expose/use `feature_ablation=no_rank_prior`, disable rank embedding, and ideally add train-time candidate-order shuffle. Keep the encoder frozen. If row Spearman and pairwise accuracy remain near random, the frozen text representation is insufficient.
2. If v0.1 still fails, move to v1 with a trainable pair encoder: LoRA or unfreeze top DeBERTa layers, lower encoder LR around `2e-5`, keep head LR around `1e-4`, and checkpoint by `jaccard@5` while requiring `oracle_rank_ndcg@5` not to drop below the retrieval control.
3. Add an auxiliary scalar distillation target, not only pairwise/listwise labels. Predict normalized `delta_margin` or a proxy such as `single_margin`, with Huber/MSE plus the current pairwise and soft CE losses. The goal is to force the scorer to learn a calibrated utility axis before asking it to produce a top-5 list.
4. Keep saved-score static rankers as the teacher/ceiling, but separate deployability: `single_margin_step0_static` is a strong diagnostic because it uses verifier-derived saved scores; a learned scorer must close part of that gap without access to true per-candidate verifier outputs at inference time.
5. Delay DAgger-lite or bad-prefix expansion until the oracle-prefix scorer clears a minimum gate: row pairwise accuracy >= 0.60, row Spearman >= 0.20, full-val `jaccard@5` at least 0.03 above control, and `oracle_rank_ndcg@5` no worse than control.

The immediate next implementation should be small: add the no-rank-prior/order-shuffle ablation to the v0 wrapper and rerun on the same full val split. If that ablation removes the `candidate_idx=12` collapse but still stays random on row utility, then the next real experiment should be trainable DeBERTa or a stronger cross-encoder utility distiller, not more frozen-head tuning.
