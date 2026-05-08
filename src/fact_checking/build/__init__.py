"""Build candidate evidence files for fact-checking experiments."""

from typing import Any

__all__ = ["BuildResult", "build_chunking_strategy", "run_build"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from fact_checking.build import candidates, chunking

        if hasattr(candidates, name):
            return getattr(candidates, name)
        if hasattr(chunking, name):
            return getattr(chunking, name)
    raise AttributeError(name)
