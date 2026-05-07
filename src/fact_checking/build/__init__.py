"""Build candidate evidence files for fact-checking experiments."""

from typing import Any

__all__ = ["BuildResult", "run_build"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from fact_checking.build import candidates

        return getattr(candidates, name)
    raise AttributeError(name)
