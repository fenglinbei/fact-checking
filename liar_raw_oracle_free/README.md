# LIAR-RAW Oracle-Free Stage A / B

This project implements an **oracle-free** pipeline for LIAR-RAW using the JSON format you described.

- **Stage A**: frozen dense retrieval over report sentences, optionally fused with simple lexical overlap and a local BM25-like score, then diversified with MMR.
- **Stage B**: cross-encoder with **latent evidence attention** and an **ordinal classification head**. It trains only from the **claim-level 6-way label** and does **not** use `is_evidence`.

The code is intentionally split into multiple modules instead of a single file.

## Directory layout

```text
liar_raw_oracle_free/
├── configs/
│   ├── stage_a.yaml
│   └── stage_b.yaml
├── scripts/
│   ├── run_stage_a.sh
│   ├── train_stage_b.sh
│   └── predict_stage_b.sh
├── src/liar_raw/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── io.py
│   │   └── types.py
│   ├── models/
│   │   ├── latent_evidence.py
│   │   ├── ordinal.py
│   │   └── sparsemax.py
│   ├── retrieval/
│   │   ├── build_stage_a.py
│   │   ├── embedder.py
│   │   ├── mmr.py
│   │   └── text_utils.py
│   └── training/
│       ├── metrics.py
│       ├── predict_stage_b.py
│       ├── stage_b_data.py
│       └── train_stage_b.py
├── pyproject.toml
└── requirements.txt
```

## Expected raw data format

Each split is a JSON list. Each item looks like this:

```json
{
  "event_id": "11972.json",
  "claim": "Building a wall on the U.S.-Mexico border will take literally years.",
  "label": "true",
  "explain": "...",
  "reports": [
    {
      "report_id": 4815065,
      "link": "https://...",
      "content": "...",
      "domain": "https://...",
      "tokenized": [
        {"sent": "...", "is_evidence": 0},
        {"sent": "...", "is_evidence": 0}
      ]
    }
  ]
}
```

Notes:

1. If `reports[*].tokenized` exists, the loader uses `tokenized[*].sent` as the sentence inventory.
2. If `tokenized` is missing, the loader falls back to a simple sentence splitter over `content`.
3. `is_evidence` is ignored in this pipeline.

## How Stage A works

For each claim:

1. collect all report sentences
2. embed the claim with a frozen query encoder
3. embed each sentence with a frozen passage encoder
4. score each sentence with a weighted hybrid score:

```text
hybrid = 0.70 * dense + 0.20 * lexical_overlap + 0.10 * bm25_like
```

5. run MMR to avoid returning many near-duplicate sentences
6. save top-k candidates as JSONL

Output file example (`stage_a_train.jsonl`):

```json
{
  "event_id": "11972.json",
  "claim": "Building a wall on the U.S.-Mexico border will take literally years.",
  "label": "true",
  "explain": "...",
  "candidates": [
    {
      "report_id": 123,
      "sent_idx": 0,
      "text": "Engineering experts agree the wall would take years.",
      "dense_score": 0.81,
      "lexical_score": 0.42,
      "bm25_score": 0.68,
      "hybrid_score": 0.76,
      "link": "https://...",
      "domain": "https://..."
    }
  ]
}
```

## How Stage B works

Stage B reads Stage A candidates and trains a claim-level model.

For each claim, it takes the top-k candidate sentences and builds `claim + sentence` pairs.

The model predicts, **for each sentence**:

- a latent attention weight (how important it is)
- a latent support probability
- a latent refute probability

Then it aggregates them into claim-level features:

- support score
- refute score
- support-refute margin
- total evidence strength
- attention-weighted sentence representation

These are passed to:

- a 6-way classification head
- a CORAL ordinal head

The losses are:

```text
L = cross_entropy
  + λ1 * coral_loss
  + λ2 * margin_regression
  + λ3 * support/refute overlap penalty
  + λ4 * attention entropy penalty
```

This is still **oracle-free** because the training signal is the claim-level label only.

## Setup

```bash
cd liar_raw_oracle_free
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

## Step 1: edit configs

Update these paths in `configs/stage_a.yaml`:

```yaml
data:
  train_path: /path/to/liar_raw/train.json
  val_path: /path/to/liar_raw/val.json
  test_path: /path/to/liar_raw/test.json
```

After Stage A finishes, make sure `configs/stage_b.yaml` points to the generated JSONL files.

## Step 2: run Stage A

```bash
bash scripts/run_stage_a.sh
```

Or manually:

```bash
PYTHONPATH=src python -m liar_raw.retrieval.build_stage_a --config configs/stage_a.yaml
```

## Step 3: train Stage B

```bash
bash scripts/train_stage_b.sh
```

Or manually:

```bash
PYTHONPATH=src python -m liar_raw.training.train_stage_b --config configs/stage_b.yaml
```

Best checkpoint is saved to:

```text
outputs/stage_b/best_model.pt
```

## Step 4: predict and export evidence

```bash
bash scripts/predict_stage_b.sh
```

Or manually:

```bash
PYTHONPATH=src python -m liar_raw.training.predict_stage_b \
  --config configs/stage_b.yaml \
  --checkpoint outputs/stage_b/best_model.pt \
  --split test
```

The exported JSONL contains:

- predicted label
- class probabilities
- top support evidence sentences
- top refute evidence sentences

## Practical defaults

Good first settings:

- Stage A embedder: `BAAI/bge-base-en-v1.5`
- Stage A top-k: `24`
- Stage B backbone: `microsoft/deberta-v3-base`
- Stage B batch size: `4`
- max length: `256`

If memory is tight:

- reduce Stage A batch size from `64` to `16`
- reduce Stage B top-k from `24` to `16`
- reduce Stage B max length from `256` to `192`
- keep only the last `1` encoder layer unfrozen

## Important caveats

1. This code is intentionally **oracle-free**, so it does not use `is_evidence` anywhere.
2. Stage A here is **frozen retrieval**, not a trained retriever.
3. Because support/refute is latent, the extracted evidence is a model interpretation, not gold supervision.
4. The default Stage B implementation optimizes claim-level macro-F1, not sentence-level evidence F1.

## Recommended next upgrade

The strongest next step is to add a **small manually audited sentence-level dev set** and use it only for model selection and evidence evaluation. That lets you keep training oracle-free while still checking whether the latent support/refute sentences are actually reasonable.
