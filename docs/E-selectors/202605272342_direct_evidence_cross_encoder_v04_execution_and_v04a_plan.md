# v0.4 Direct Evidence Cross-Encoder Execution Order and v0.4a Plan

## Summary

v0.4 的目标不是继续扩大已有 selector imitation 路线，而是先验证一个更窄的问题：只看 `claim + evidence_text`，强 cross-encoder reranker 是否能识别 direct evidence。若 text-only signal 不成立，则不应继续把 rank/provenance、anchor、stance bucket 或更复杂 selector 后处理叠上去。

当前 reference 是 v0.3.1 `pointwise_all_features`：

| run | AUROC | recall@5 | jaccard@5 | top1 | NDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pointwise_all_features` | 0.6339 | 0.3642 | 0.2445 | 0.1028 | 0.2996 |

## Execution Order

1. **v0.4a off-the-shelf scoring**：使用 `Qwen/Qwen3-Reranker-8B` 通过 `sentence_transformers.CrossEncoder` 对 `claim + evidence_text` 做 text-only direct evidence scoring。不训练，不调用 DeepSeek，不使用 rank/provenance 作为模型输入。
2. **v0.4b BGE smoke fine-tune**：若 v0.4a 显示 text-only signal 有意义，再用较轻的 BGE reranker 跑训练/评估 smoke，验证 dataset、loss、hard negative sampling 和 held-out eval 合同。
3. **v0.4c Qwen3 LoRA**：在远程 4 卡 48G L20 上训练 Qwen3-Reranker-8B LoRA，目标是 directness / oracle utility 的 pairwise 和 soft target，而不是单纯 `oracle_selected` imitation。
4. **v0.4d late fusion / light MMR**：只有当 text-only scorer 有独立收益时，才加入很轻的 retrieval late fusion 或 source-diverse penalty。若收益依赖 rank/provenance，则判为旧路径复发。

## v0.4a Design

输入：

```text
outputs/selectors/count_amplified_stance_bucket_selector/v0_2_val/candidate_stance_buckets_v02_n7_val.jsonl
```

输出：

```text
outputs/selectors/direct_evidence_cross_encoder/v0_4a_val/
```

真实 backend 只有：

```text
sentence_transformers.CrossEncoder
BASE_MODEL=Qwen/Qwen3-Reranker-8B
```

若 CrossEncoder 无法加载或无法运行 Qwen3-Reranker-8B，直接 fail fast，不做 transformers yes/no logit fallback。

模型输入只允许：

```text
query = claim
passage = evidence_text
```

禁止进入 scorer input 的字段包括：

```text
oracle_selected, oracle_step, gold_label, event_id, candidate_key,
candidate_uid, baseline_rank, qd_pool_rank, union_pool_rank,
from_baseline, from_qd, source/group/report metadata
```

这些字段只能用于 eval、hard-negative diagnostics 和 trace。

## Run Commands

Mock smoke：

```bash
MOCK_SCORES=1 SAMPLE_LIMIT=5 NUM_SHARDS=1 \
bash scripts/phase5_selectors/run/run_direct_evidence_cross_encoder_v0_4a.sh
```

Single-GPU real smoke：

```bash
CUDA_VISIBLE_DEVICES=0 SAMPLE_LIMIT=20 NUM_SHARDS=1 BATCH_SIZE=1 \
bash scripts/phase5_selectors/run/run_direct_evidence_cross_encoder_v0_4a.sh
```

Full remote 4-card run：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_SHARDS=4 BATCH_SIZE=4 \
bash scripts/phase5_selectors/run/run_direct_evidence_cross_encoder_v0_4a.sh
```

If OOM occurs, first lower `BATCH_SIZE=1`; do not switch backend.

## Evaluation

v0.4a writes:

```text
direct_ce_scored_candidates_val.jsonl
direct_ce_score_manifest.json
eval/selection_trace_val.jsonl
eval/selector_metrics.json
eval/direct_ce_diagnostics.json
eval/analysis_summary.md
eval/manifest.json
```

Compared selectors:

- `original_pool_order_top5`
- `qd_union_source_score_top5`
- `count_amplified_stance_bucket_top5`
- `v0_3_1_pointwise_all_features_top5`
- `direct_ce_text_only_top5`
- `direct_ce_light_source_diverse_top5`

Go criteria:

- candidate AUROC > `0.56`
- same-source hard-negative pairwise accuracy > `0.57`
- `direct_ce_text_only_top5` jaccard@5 >= `0.250`
- continue to v0.4b/c only if text-only signal beats v0.3.1 or shows clear hard-negative lift.

No-Go interpretation:

- If candidate AUROC is near random and jaccard does not beat controls, do not train Qwen3 LoRA; the bottleneck is likely supervision or oracle target, not model capacity.
- If candidate AUROC is high but top5 overlap does not improve, inspect false positives among high-retrieval non-oracle candidates before adding fusion.
- If gains appear only after adding rank/provenance metadata, treat it as the same shortcut failure observed in earlier selector lines.

## v0.4a.1 Fix

The synced v0.4a run showed collapsed CrossEncoder scores: almost every event had identical candidate scores, so the result should be interpreted as a scoring-interface failure rather than evidence that Qwen3-Reranker-8B lacks direct-evidence signal.

v0.4a.1 adds:

- `direct_ce_raw_score` plus normalized `direct_ce_score` in scored candidates.
- `PROMPT_MODE=direct_evidence_custom/default_query` switch.
- canary scoring before real rows; obvious positive evidence must score above unrelated evidence.
- score sanity gate after each shard and merge: global score std, unique score count, and event-level all-tie rate.
- tie-aware AUPRC; all-tied scores now produce positive-rate AP instead of an inflated value.
- fresh default output dirs under `v0_4a_1_${SPLIT}_${PROMPT_MODE}` and `RESUME=0`.

Recommended repair sweep:

```bash
PROMPT_MODE=default_query OUTPUT_DIR=outputs/selectors/direct_evidence_cross_encoder/v0_4a_1_val_default_query \
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_SHARDS=4 BATCH_SIZE=2 RESUME=0 \
bash scripts/phase5_selectors/run/run_direct_evidence_cross_encoder_v0_4a_1.sh

PROMPT_MODE=direct_evidence_custom OUTPUT_DIR=outputs/selectors/direct_evidence_cross_encoder/v0_4a_1_val_direct_evidence_custom \
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_SHARDS=4 BATCH_SIZE=2 RESUME=0 \
bash scripts/phase5_selectors/run/run_direct_evidence_cross_encoder_v0_4a_1.sh
```

If either prompt mode fails the canary or score sanity gate, stop at v0.4a.1 and inspect the CrossEncoder/Qwen3 interface before moving to v0.4b/c. Do not bypass the gate for full runs.
