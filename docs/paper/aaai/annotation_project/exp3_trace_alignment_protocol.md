# EviTrace Trace Alignment Human Evaluation

Status: **frozen protocol, before formal annotation**

Protocol version: `exp3-trace-alignment-v1`

Sample seed: `20260724`

This protocol adds a small-scale, exploratory human evaluation of the final
evidence organization produced by EviTrace. It complements, and does not
replace, the existing Atomization and Evidence Map audits.

## 1. Evaluation questions

The study has three layers:

1. **Main preference (120 claims).** Given two ordered evidence sequences with
   the same number of evidence units, which sequence better enables an
   independent fact-checker to reach an accurate and well-supported verdict?
2. **Order-only preference (80 claims).** Given the same evidence set in two
   different orders, which ordering better supports the same goal?
3. **Transition audit (100 claims).** Is a proposed atom-state transition
   justified by the newly selected evidence, and how much marginal information
   does that evidence add?

Two non-author annotators independently label every formal task. There is no
author adjudication and no derived pseudo-gold label. Subjective disagreements
remain in the released raw annotations.

We use *reasoning trace* only for the external, inspectable sequence of evidence
selections and atom-state transitions. It is not a model's latent
chain-of-thought and is not a causal explanation of the verifier prediction.

## 2. Frozen source artifacts

Formal test tasks are constructed only from:

- verifier-visible build:
  `outputs/sentence_trace_method/liar_raw__ministral3_8b__atom_anchor_v0_2_learned_marginal_proxy_fullpool_minmax5_10/build/build_test.jsonl`;
- EviTrace full-pool trace:
  `outputs/selectors/atom_anchor/liar_raw_abc_v0_1/05_mrec_v0_2_learned_marginal_proxy_fullpool/selection_trace_test.jsonl`;
- S4 source-score ordering:
  `outputs/selectors/selector_mechanism_ablation_chunking/liar_raw_abc_selector_mech_s4_atom_union_source_score_ordered_test/selection_trace_test.jsonl`.

The exporter must stop before sampling unless all 1,251 events align, every
candidate UID is unique within an event, and the stored S4 order exactly matches
`rank_atom_union_source_score_candidates()`. The sole predetermined exclusion
is the atom-parse failure `7845.json`.

For event \(i\), the comparison length \(K_i\) is the build artifact's final
`evidence_count`, after the 1,024-token prompt guard. Neither
`evidence_count_before` nor the full 20-step trace may determine \(K_i\).
The paper must distinguish the selected-prefix mean
\(K_{\mathrm{sel}}=6.206\) from the verifier-visible mean
\(K_{\mathrm{vis}}=5.958\).

EviTrace candidates are recovered by UID from the clean full-pool candidate
text. The control is the first \(K_i\) candidates in the recomputed S4 order
over the same Atom-Union pool. Evidence count is matched; token budget is not.
Tokenizer counts for both sides are retained in the private key.

## 3. Sampling

Formal cohorts use seed `20260724` and have disjoint claims.

- **Main, 120:** six LIAR-RAW labels crossed with single- versus multi-atom
  claims, ten claims in each of the 12 cells.
- **Order-only, 80:** 40 single-atom and 40 multi-atom claims. Within each
  complexity stratum, label quotas follow largest-remainder allocation from
  the eligible pool. Claims whose EviTrace order already equals the S4 order
  are removed before sampling and the removed count is reported.
- **Transition, 100:** at most one visible step per claim: 40 `OPEN`, 20
  `CONTRAST`, 20 `BRIDGE`, 10 `CORROBORATE`, and 10 `FALLBACK`. `OPEN` and
  `CONTRAST` must change state; the other three operations must be
  self-transitions.

A separate LIAR-RAW validation pilot contains 20 main, 10 order-only, and 15
transition tasks. Pilot annotations are excluded from every paper statistic.
Pilot observations may improve instructions or interface mechanics, but the
decision to run the formal study cannot depend on the observed preference
direction.

## 4. Blinding and annotation contract

The public preference record contains only:

- `blind_task_id`;
- authoritative English claim and auxiliary cached Chinese claim;
- `sequence_a_html` and `sequence_b_html`.

The preference form records a five-level `overall_preference`, optional
`data_issue`, and optional `notes`. It does not ask for confidence or an
earliest-sufficient prefix. Evidence is rendered as neutral text with a common
step number, source ID, and source domain. Atom cues beginning with `Check:`,
atom text, scores, state, map fields, method names, and method-specific styling
are forbidden.

The public transition record contains the focal atom, a state legend, earlier
evidence for the same atom, the current evidence, and the proposed
before-to-after state. Relation, directness, confidence, gold label, verifier
prediction, event identity, and method identity are private. The form records:

- `transition_validity`: `valid`, `partially_valid`, or `invalid`;
- `marginal_contribution`: `clear`, `limited`, or `none`.

English is authoritative. Each canonical claim and evidence text has one cached
Chinese translation reused across all tasks and both annotators.

The mapping between methods and A/B is committed only in private key files.
EviTrace is side A exactly 60 times in main and 40 times in order-only. Each
annotator receives an independently randomized task order. Private files and
their hashes are not distributed to annotators.

## 5. Project sequencing

Each formal annotator has a separate `ONLY` preference project and a separate
`ONLY` transition project, all with `maximum_annotations=1`. Community Label
Studio project names are an operational boundary rather than an access-control
guarantee, so the launch report also freezes project IDs, user IDs, task
fingerprints, and task counts.

Every four-stage chain is scoped to an explicit deployment `revision` (default
`v1`). The revision is part of the project title, marker, task metadata, task
order derivation, validation, gate lookup, and launch report. If the pilot leads
to any change in XML, task projection, translations, instructions, or exclusion
rules, the operator must increment the revision and rerun that revision's pilot.
An older revision remains an audit artifact and cannot satisfy a newer
revision's formal-stage gate.

Only preference projects may be opened initially. Transition tasks expose
states and remain unpublished until:

1. both preference projects are complete;
2. the data-issue exclusion rule is frozen;
3. preference task data and analysis code are frozen.

The private side mapping is unblinded only after both annotators finish, data
issues are resolved under the frozen rule, and analysis code is frozen.

## 6. Frozen analysis

After unblinding, the five preference levels map to EviTrace scores
\(+2,+1,0,-1,-2\). Main results include EviTrace wins, control wins, ties,
non-tie conditional win rate, and a design-weighted result over the 12 sampling
cells.

Uncertainty uses 10,000 stratified, claim-clustered bootstrap replicates and a
claim-level label-swap randomization test. Agreement includes exact agreement,
linearly weighted Cohen's \(\kappa\), and Cohen's \(\kappa\) after collapsing
to EviTrace/tie/control.

The complete primary sample is retained regardless of token imbalance.
Predetermined sensitivity analyses are:

- the subset with
  \(\lvert T_{\mathrm{EviTrace}}-T_{\mathrm{S4}}\rvert\leq64\);
- a secondary logistic model with token difference and annotator fixed effect,
  using claim-clustered standard errors.

If complete or quasi-complete separation prevents a stable logistic estimate,
only the predetermined token-balanced subset is reported.

Order-only uses exactly the same text and evidence set and is therefore the
only comparison that can isolate ordering. Transition validity and marginal
contribution are reported separately for change steps and self-transitions and
stratified by operation. Balanced transition samples are not presented as the
natural operation distribution.

## 7. Interpretation gates

- Main preference significant, order-only not significant: “improves evidence
  selection and overall organization.”
- Both comparisons stably favor EviTrace: “improves decision-oriented
  ordering.”
- Transition `valid` has a 95% confidence-interval lower bound above 50%:
  “human-aligned transitions.” Otherwise the result is described as mixed or
  uncertain.

No outcome licenses a claim of improved human fact-checking accuracy or a
causal explanation of verifier behavior.

## 8. Completion and release

The live SQLite database is backed up before any project mutation. Launch uses
a read-only dry run followed by a transaction, SQLite `quick_check`, exact task
counts, and task-fingerprint verification.

The formal study is complete only when both annotators contribute 200
preference and 100 transition annotations: 600 annotations total. The analyzer
may write `complete=true` only if annotation counts, side mappings, project
identities, task fingerprints, source artifact hashes, and all output hashes
match the frozen manifests.
