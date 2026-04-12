from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from liar_raw.config import load_yaml
from liar_raw.data.io import iter_sentences, load_split
from liar_raw.retrieval.embedder import EmbedderConfig, TextEmbedder
from liar_raw.retrieval.mmr import maximal_marginal_relevance
from liar_raw.retrieval.text_utils import bm25_like_score, lexical_overlap_f1



def minmax_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < 1e-8:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)



def build_candidates_for_sample(
    sample,
    embedder: TextEmbedder,
    top_k: int,
    alpha_dense: float,
    alpha_lexical: float,
    alpha_bm25: float,
    mmr_lambda: float,
) -> dict:
    sentences = list(iter_sentences(sample))
    if not sentences:
        return {
            "event_id": sample.event_id,
            "claim": sample.claim,
            "label": sample.label,
            "explain": sample.explain,
            "candidates": [],
        }

    sent_texts = [s.text for s in sentences]
    sent_emb = embedder.encode(sent_texts, is_query=False)
    claim_emb = embedder.encode([sample.claim], is_query=True)[0]
    dense_scores = sent_emb @ claim_emb

    lexical_scores = np.asarray([lexical_overlap_f1(sample.claim, s) for s in sent_texts], dtype=np.float32)
    bm25_scores = np.asarray([bm25_like_score(sample.claim, s) for s in sent_texts], dtype=np.float32)

    dense_scaled = minmax_scale(dense_scores)
    lexical_scaled = minmax_scale(lexical_scores)
    bm25_scaled = minmax_scale(bm25_scores)

    hybrid_scores = (
        alpha_dense * dense_scaled + alpha_lexical * lexical_scaled + alpha_bm25 * bm25_scaled
    )

    keep_indices = maximal_marginal_relevance(
        query_scores=hybrid_scores,
        sentence_vectors=sent_emb,
        top_k=min(top_k, len(sentences)),
        lambda_weight=mmr_lambda,
    )

    candidates = []
    for idx in keep_indices:
        sent = sentences[idx]
        candidates.append(
            {
                "report_id": sent.report_id,
                "sent_idx": sent.sent_idx,
                "text": sent.text,
                "dense_score": float(dense_scores[idx]),
                "lexical_score": float(lexical_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "hybrid_score": float(hybrid_scores[idx]),
                "link": sent.link,
                "domain": sent.domain,
            }
        )
    candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return {
        "event_id": sample.event_id,
        "claim": sample.claim,
        "label": sample.label,
        "explain": sample.explain,
        "candidates": candidates,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Build oracle-free Stage A retrieval candidates.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test", None])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    retrieval_cfg = cfg["retrieval"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    embedder = TextEmbedder(
        EmbedderConfig(
            model_name=retrieval_cfg["embedder_model"],
            device=retrieval_cfg.get("device", "cuda"),
            max_length=int(retrieval_cfg.get("max_length", 256)),
            batch_size=int(retrieval_cfg.get("batch_size", 64)),
        )
    )

    split_names = [args.split] if args.split else ["train", "val", "test"]
    for split_name in split_names:
        input_path = data_cfg[f"{split_name}_path"]
        samples = load_split(input_path)
        output_path = output_dir / f"stage_a_{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as writer:
            for sample in tqdm(samples, desc=f"Stage A [{split_name}]"):
                row = build_candidates_for_sample(
                    sample=sample,
                    embedder=embedder,
                    top_k=int(retrieval_cfg.get("top_k", 24)),
                    alpha_dense=float(retrieval_cfg.get("alpha_dense", 0.70)),
                    alpha_lexical=float(retrieval_cfg.get("alpha_lexical", 0.20)),
                    alpha_bm25=float(retrieval_cfg.get("alpha_bm25", 0.10)),
                    mmr_lambda=float(retrieval_cfg.get("mmr_lambda", 0.70)),
                )
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
