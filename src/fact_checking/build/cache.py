"""Cache and serialization infrastructure for the build pipeline."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from fact_checking.build.chunking import ChunkingStrategy, build_chunking_strategy
from fact_checking.data.io import iter_sentences, load_split
from fact_checking.retrieval.embedder import EmbedderConfig, TextEmbedder

BUILD_LOGIC_VERSION = "chunk-first-mmr-v1"
CHUNK_MMR_CACHE_VERSION = "chunk-text-embedding-v1"


def sentence_reader_config(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = dict(cfg.get("data", {}) or {})
    retrieval_cfg = dict(cfg.get("retrieval", {}) or {})
    selection_method = str(retrieval_cfg.get("selection_method", "mmr")).strip().lower()
    source = str(data_cfg.get("sentence_source") or data_cfg.get("source") or "").strip().lower()
    if not source:
        source = "tokenized" if selection_method in {"raw_top_evidence", "raw_label_topk", "raw_evidence"} else "content"
    return {
        "sentence_source": source,
        "sentence_min_char_len": int(data_cfg.get("sentence_min_char_len", data_cfg.get("min_char_len", 10))),
    }


def sentence_reader_fingerprint_payload(cfg: dict[str, Any]) -> dict[str, Any] | None:
    reader = sentence_reader_config(cfg)
    if reader == {"sentence_source": "content", "sentence_min_char_len": 10}:
        return None
    return reader


def premmr_config_fingerprint(cfg: dict[str, Any]) -> str:
    from fact_checking.pipeline.artifacts import fingerprint

    retrieval_cfg = dict(cfg.get("retrieval", {}) or {})
    retrieval = {
        key: retrieval_cfg.get(key)
        for key in ("embedder_model", "device", "max_length", "precision")
        if key in retrieval_cfg
    }
    payload = {
        "data": cfg.get("data", {}),
        "retrieval": retrieval,
    }
    sentence_reader = sentence_reader_fingerprint_payload(cfg)
    if sentence_reader is not None:
        payload["sentence_reader"] = sentence_reader
    return fingerprint(payload)


def chunk_mmr_config_fingerprint(cfg: dict[str, Any]) -> str:
    from fact_checking.pipeline.artifacts import fingerprint

    retrieval_cfg = dict(cfg.get("retrieval", {}) or {})
    retrieval = {
        key: retrieval_cfg.get(key)
        for key in ("embedder_model", "device", "max_length", "precision")
        if key in retrieval_cfg
    }
    payload = {
        "version": CHUNK_MMR_CACHE_VERSION,
        "data": cfg.get("data", {}),
        "retrieval": retrieval,
        "chunking": retrieval_cfg.get("chunking", {}),
    }
    sentence_reader = sentence_reader_fingerprint_payload(cfg)
    if sentence_reader is not None:
        payload["sentence_reader"] = sentence_reader
    return fingerprint(payload)


def save_pickle_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def sentence_to_dict(sent) -> dict[str, Any]:
    return {
        "event_id": sent.event_id,
        "report_id": sent.report_id,
        "sent_idx": sent.sent_idx,
        "text": sent.text,
        "link": sent.link,
        "domain": sent.domain,
        "raw": sent.raw,
    }


def dict_to_sentence(d: dict[str, Any], *, event_id_fallback: str | None = None):
    from fact_checking.data.io import SentenceRecord

    return SentenceRecord(
        event_id=d.get("event_id") or event_id_fallback or "",
        report_id=d["report_id"],
        sent_idx=d["sent_idx"],
        text=d["text"],
        link=d.get("link"),
        domain=d.get("domain"),
        raw=d.get("raw", {}),
    )


def normalize_model_name(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def can_reuse_chunk_embeddings(strategy: ChunkingStrategy, retrieval_cfg: dict[str, Any]) -> bool:
    embedder_cfg = getattr(strategy, "_embedder_cfg", None)
    if embedder_cfg is None:
        return False
    retrieval_model = retrieval_cfg.get("embedder_model")
    if not retrieval_model:
        return False
    return (
        normalize_model_name(getattr(embedder_cfg, "model_name", ""))
        == normalize_model_name(retrieval_model)
        and int(getattr(embedder_cfg, "max_length", 256)) == int(retrieval_cfg.get("max_length", 256))
    )


def visible_gpu_for_worker(gpu_id: int, run_summary: dict[str, Any]) -> str:
    visible = str(run_summary.get("cuda_visible_devices") or "").strip()
    if not visible:
        return str(gpu_id)
    devices = [part.strip() for part in visible.split(",") if part.strip()]
    if gpu_id >= len(devices):
        raise ValueError(
            f"Requested worker gpu_id={gpu_id}, but cuda_visible_devices={visible!r} "
            f"only exposes {len(devices)} device(s)."
        )
    return devices[gpu_id]


def chunk_embeddings_by_content(
    sents: list,
    sent_emb: np.ndarray,
) -> dict[str, dict[int, np.ndarray]]:
    if sent_emb.ndim != 2 or len(sent_emb) < len(sents):
        return {}

    by_content: dict[str, dict[int, np.ndarray]] = {}
    for row_idx, sent in enumerate(sents):
        content = sent.raw.get("content", "") if isinstance(sent.raw, dict) else ""
        if not content:
            continue
        by_content.setdefault(str(content), {})[int(sent.sent_idx)] = sent_emb[row_idx]
    return by_content


def compute_pre_mmr_batch(
    samples: list,
    embedder: TextEmbedder,
    sentence_source: str = "content",
    sentence_min_char_len: int = 10,
) -> list:
    """Batch-embed all sentences and claims; return PreMMRSample list."""
    from fact_checking.build.candidates import PreMMRSample

    all_sent_texts: list[str] = []
    sample_boundaries: list[tuple[int, int]] = []
    claims: list[str] = []
    per_sample: list[tuple[list, list[str]]] = []

    for sample in samples:
        sents = list(
            iter_sentences(
                sample,
                min_char_len=sentence_min_char_len,
                source=sentence_source,
            )
        )
        if not sents:
            sample_boundaries.append((0, 0))
            claims.append(sample.claim)
            per_sample.append(([], []))
            continue
        start = len(all_sent_texts)
        sent_texts = [s.text for s in sents]
        all_sent_texts.extend(sent_texts)
        end = len(all_sent_texts)
        sample_boundaries.append((start, end))
        claims.append(sample.claim)
        per_sample.append((sents, sent_texts))

    if all_sent_texts:
        all_sent_emb = embedder.encode(all_sent_texts, is_query=False)
    else:
        all_sent_emb = np.zeros((0,), dtype=np.float32)
    all_claim_emb = embedder.encode(claims, is_query=True)

    results: list[PreMMRSample] = []
    for i in range(len(samples)):
        sample = samples[i]
        sents, _sent_texts = per_sample[i]
        start, end = sample_boundaries[i]
        sent_emb = all_sent_emb[start:end].copy() if end > start else np.zeros((0,), dtype=np.float32)
        claim_emb = all_claim_emb[i].copy()
        results.append(PreMMRSample(
            event_id=sample.event_id,
            claim=sample.claim,
            label=sample.label,
            explain=sample.explain,
            sentences=[sentence_to_dict(s) for s in sents],
            sent_emb=sent_emb,
            claim_emb=claim_emb,
        ))
    return results


def premmr_worker(
    gpu_id: int,
    run_summary: dict[str, Any],
    data_cfg: dict[str, Any],
    split_name: str,
    output_path: Path,
) -> None:
    """GPU worker: embed its chunk of the split, write PreMMRSample pickle."""
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu_for_worker(gpu_id, run_summary)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    samples = load_split(data_cfg[f"{split_name}_path"])
    num_gpus = run_summary["num_gpus"]
    chunk_size = (len(samples) + num_gpus - 1) // num_gpus
    samples_chunk = samples[gpu_id * chunk_size : (gpu_id + 1) * chunk_size]
    if not samples_chunk:
        save_pickle_atomic(output_path, [])
        return

    embedder = TextEmbedder(
        EmbedderConfig(
            model_name=run_summary["embedder_model"],
            device="cuda",
            max_length=run_summary["max_length"],
            batch_size=run_summary["batch_size"],
            precision=run_summary["precision"],
        )
    )

    prefetch_size = run_summary["prefetch_size"]
    sentence_source = str(run_summary.get("sentence_source", "content"))
    sentence_min_char_len = int(run_summary.get("sentence_min_char_len", 10))
    results: list = []
    if prefetch_size > 1:
        for start in tqdm(range(0, len(samples_chunk), prefetch_size),
                          desc=f"PreMMR [{split_name}] GPU {gpu_id}",
                          unit="batch"):
            batch = samples_chunk[start : start + prefetch_size]
            results.extend(
                compute_pre_mmr_batch(
                    batch,
                    embedder,
                    sentence_source=sentence_source,
                    sentence_min_char_len=sentence_min_char_len,
                )
            )
    else:
        for sample in tqdm(samples_chunk,
                           desc=f"PreMMR [{split_name}] GPU {gpu_id}"):
            results.extend(
                compute_pre_mmr_batch(
                    [sample],
                    embedder,
                    sentence_source=sentence_source,
                    sentence_min_char_len=sentence_min_char_len,
                )
            )

    save_pickle_atomic(output_path, results)


def compute_pre_mmr_split(
    split_name: str,
    data_cfg: dict[str, Any],
    retrieval_cfg: dict[str, Any],
    run_summary: dict[str, Any],
    cache_dir: Path,
    num_gpus: int,
) -> Path:
    """Ensure pre-MMR cache exists for one split; return path to cached pickle."""
    cache_path = cache_dir / f"{split_name}.pkl"
    if cache_path.exists():
        return cache_path

    if num_gpus > 1:
        ctx = multiprocessing.get_context("fork")
        chunk_paths: list[Path] = []
        workers: list[multiprocessing.Process] = []
        for gpu_id in range(num_gpus):
            chunk_path = cache_dir / f"{split_name}_gpu{gpu_id}.pkl"
            chunk_paths.append(chunk_path)
            p = ctx.Process(
                target=premmr_worker,
                args=(gpu_id, run_summary, data_cfg, split_name, chunk_path),
            )
            p.start()
            workers.append(p)
        for p in workers:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"PreMMR worker failed with exit code {p.exitcode}")

        all_results: list = []
        for chunk_path in chunk_paths:
            all_results.extend(load_pickle(chunk_path))
            chunk_path.unlink()
        save_pickle_atomic(cache_path, all_results)
    else:
        embedder = TextEmbedder(
            EmbedderConfig(
                model_name=run_summary["embedder_model"],
                device=run_summary.get("device", "cuda"),
                max_length=run_summary["max_length"],
                batch_size=run_summary["batch_size"],
                precision=run_summary["precision"],
            )
        )
        samples = load_split(data_cfg[f"{split_name}_path"])
        prefetch_size = run_summary["prefetch_size"]
        sentence_source = str(run_summary.get("sentence_source", "content"))
        sentence_min_char_len = int(run_summary.get("sentence_min_char_len", 10))
        results: list = []
        if prefetch_size > 1:
            for start in tqdm(range(0, len(samples), prefetch_size),
                              desc=f"PreMMR [{split_name}]",
                              unit="batch"):
                batch = samples[start : start + prefetch_size]
                results.extend(
                    compute_pre_mmr_batch(
                        batch,
                        embedder,
                        sentence_source=sentence_source,
                        sentence_min_char_len=sentence_min_char_len,
                    )
                )
        else:
            for sample in tqdm(samples, desc=f"PreMMR [{split_name}]"):
                results.extend(
                    compute_pre_mmr_batch(
                        [sample],
                        embedder,
                        sentence_source=sentence_source,
                        sentence_min_char_len=sentence_min_char_len,
                    )
                )
        save_pickle_atomic(cache_path, results)

    return cache_path


def chunk_mmr_worker(
    gpu_id: int,
    run_summary: dict[str, Any],
    retrieval_cfg: dict[str, Any],
    pre_samples: list,
    output_path: Path,
) -> None:
    """GPU worker: build chunk candidates, re-embed chunk texts, write ChunkMMRSamples."""
    from fact_checking.build.candidates import _compute_chunk_mmr_batch

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu_for_worker(gpu_id, run_summary)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if not pre_samples:
        save_pickle_atomic(output_path, [])
        return

    embedder = TextEmbedder(
        EmbedderConfig(
            model_name=run_summary["embedder_model"],
            device="cuda",
            max_length=run_summary["max_length"],
            batch_size=run_summary["batch_size"],
            precision=run_summary["precision"],
        )
    )
    strategy = build_chunking_strategy(retrieval_cfg.get("chunking"), retrieval_cfg)
    reuse_chunk_embeddings = can_reuse_chunk_embeddings(strategy, retrieval_cfg)
    prefetch_size = run_summary["prefetch_size"]

    results: list = []
    if prefetch_size > 1:
        for start in tqdm(range(0, len(pre_samples), prefetch_size),
                          desc=f"ChunkMMR GPU {gpu_id}",
                          unit="batch"):
            batch = pre_samples[start : start + prefetch_size]
            results.extend(_compute_chunk_mmr_batch(batch, embedder, strategy, reuse_chunk_embeddings))
    else:
        for pre in tqdm(pre_samples, desc=f"ChunkMMR GPU {gpu_id}"):
            results.extend(_compute_chunk_mmr_batch([pre], embedder, strategy, reuse_chunk_embeddings))

    save_pickle_atomic(output_path, results)


def compute_chunk_mmr_split(
    split_name: str,
    retrieval_cfg: dict[str, Any],
    run_summary: dict[str, Any],
    pre_mmr_path: Path,
    cache_dir: Path,
    num_gpus: int,
) -> Path:
    """Ensure chunk-level MMR cache exists for one split; return cached pickle path."""
    from fact_checking.build.candidates import _compute_chunk_mmr_batch

    cache_path = cache_dir / f"{split_name}.pkl"
    if cache_path.exists():
        return cache_path

    pre_samples = load_pickle(pre_mmr_path)
    if num_gpus > 1:
        ctx = multiprocessing.get_context("fork")
        chunk_size = (len(pre_samples) + num_gpus - 1) // num_gpus
        chunk_paths: list[Path] = []
        workers: list[multiprocessing.Process] = []
        for gpu_id in range(num_gpus):
            chunk_path = cache_dir / f"{split_name}_gpu{gpu_id}.pkl"
            chunk_paths.append(chunk_path)
            pre_chunk = pre_samples[gpu_id * chunk_size : (gpu_id + 1) * chunk_size]
            p = ctx.Process(
                target=chunk_mmr_worker,
                args=(gpu_id, run_summary, retrieval_cfg, pre_chunk, chunk_path),
            )
            p.start()
            workers.append(p)
        for p in workers:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"ChunkMMR worker failed with exit code {p.exitcode}")

        all_results: list = []
        for chunk_path in chunk_paths:
            all_results.extend(load_pickle(chunk_path))
            chunk_path.unlink()
        save_pickle_atomic(cache_path, all_results)
    else:
        embedder = TextEmbedder(
            EmbedderConfig(
                model_name=run_summary["embedder_model"],
                device=run_summary.get("device", "cuda"),
                max_length=run_summary["max_length"],
                batch_size=run_summary["batch_size"],
                precision=run_summary["precision"],
            )
        )
        strategy = build_chunking_strategy(retrieval_cfg.get("chunking"), retrieval_cfg)
        reuse_chunk_embeddings = can_reuse_chunk_embeddings(strategy, retrieval_cfg)
        prefetch_size = run_summary["prefetch_size"]
        results: list = []
        if prefetch_size > 1:
            for start in tqdm(range(0, len(pre_samples), prefetch_size),
                              desc=f"ChunkMMR [{split_name}]",
                              unit="batch"):
                batch = pre_samples[start : start + prefetch_size]
                results.extend(_compute_chunk_mmr_batch(batch, embedder, strategy, reuse_chunk_embeddings))
        else:
            for pre in tqdm(pre_samples, desc=f"ChunkMMR [{split_name}]"):
                results.extend(_compute_chunk_mmr_batch([pre], embedder, strategy, reuse_chunk_embeddings))
        save_pickle_atomic(cache_path, results)

    return cache_path
