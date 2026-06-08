# Phase 9: Dense-Only Reruns

This phase contains wrappers for rerunning the final experiments after the
retrieval decision changed from dense/lexical/BM25-like hybrid scoring to
dense-only scoring.

The common dense-only setting is:

```bash
ALPHA_DENSE=1.0
ALPHA_LEXICAL=0.0
ALPHA_BM25=0.0
```

The existing code still uses field names such as `hybrid_score`. Under these
wrappers those fields mean min-max normalized dense relevance.

## 1. Build Dense-Only Traces

RAWFC:

```bash
DATASET=rawfc \
RUN_CACHE_BUILD=false \
RUN_QD=true \
RUN_EVIDENCE_MAP=true \
RUN_GRAPH_BUILD=true \
bash scripts/phase9_dense_only/build_dense_only_traces.sh
```

LIAR-RAW:

```bash
DATASET=liar_raw \
RUN_CACHE_BUILD=false \
RUN_QD=true \
RUN_EVIDENCE_MAP=true \
RUN_GRAPH_BUILD=true \
bash scripts/phase9_dense_only/build_dense_only_traces.sh
```

Outputs are split-specific directories:

```text
outputs/selectors/evidence_chain_graph/rawfc_dense_v0_6c_adaptive5_10_{train,val,test}/selection_trace_*.jsonl
outputs/selectors/evidence_chain_graph/liar_raw_dense_v0_6c_adaptive5_10_{train,val,test}/selection_trace_*.jsonl
```

## 2. Run RAWFC Backbones

```bash
MODE=full \
FINETUNE=fullft \
BACKBONES=qwen3_4b_2507,llama31_8b \
bash scripts/phase9_dense_only/run_rawfc_dense_only_backbones.sh
```

## 3. Run LIAR-RAW Backbones

```bash
MODE=full \
BACKBONES=qwen3_4b_2507,llama31_8b \
bash scripts/phase9_dense_only/run_liar_raw_dense_only_backbones.sh
```

## 4. Full-Pool Oracle Search

After a verifier checkpoint exists:

```bash
DATASET=rawfc \
BACKBONE=qwen3_4b_2507 \
VERIFIER_MODEL=outputs/selector_trace_verifier/rawfc_dense_v0_6c_eval25_backbone/<case>/train/best \
SPLITS="val test" \
bash scripts/phase9_dense_only/run_dense_only_oracle.sh
```

The oracle script forces dense-only retrieval in `CONFIG_OVERRIDES` and uses
the full deduplicated evidence pool by default (`TWO_STAGE=false`,
`MAX_CANDIDATE_POOL_SIZE=0`).

## 5. Candidate-Pool Recall

Use the phase9 wrapper against each full-pool oracle output:

```bash
ORACLE_DIR=outputs/oracle_evidence/<run> \
RUN_NAME=<run> \
SPLITS="val test" \
bash scripts/phase9_dense_only/run_candidate_pool_recall.sh
```
