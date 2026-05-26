"""Build candidate evidence files for fact-checking experiments."""

from typing import Any

__all__ = [
    "BuildResult",
    "ChunkMMRSample",
    "PreMMRSample",
    "build_chunking_strategy",
    "build_training_row",
    "chunk_mmr_config_fingerprint",
    "compute_chunk_mmr_split",
    "compute_pre_mmr_split",
    "generate_prompt_stats",
    "load_pickle",
    "load_prompt_tokenizer",
    "premmr_config_fingerprint",
    "rows_to_prepared_samples",
    "run_build",
    "save_pickle_atomic",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from fact_checking.build import candidates, chunking, cache, prompts, stats

        for module in (candidates, chunking, cache, prompts, stats):
            if hasattr(module, name):
                return getattr(module, name)
    raise AttributeError(name)
