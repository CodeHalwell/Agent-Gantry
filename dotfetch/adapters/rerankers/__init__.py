"""
Reranker adapters for dotfetch.

The base :class:`RerankerAdapter` protocol is imported eagerly; concrete
implementations that pull in optional dependencies (cohere, sentence-transformers
cross-encoders, typesafe-sdk) are lazy-loaded via ``__getattr__`` so importing this module
does not require optional installs.
"""

from __future__ import annotations

from dotfetch.adapters.rerankers.base import RerankerAdapter

__all__ = [
    "CohereReranker",
    "CrossEncoderReranker",
    "RerankerAdapter",
    "SystemOneReranker",
]


def __getattr__(name: str) -> type:
    if name == "CohereReranker":
        from dotfetch.adapters.rerankers.cohere import CohereReranker

        return CohereReranker
    if name == "CrossEncoderReranker":
        from dotfetch.adapters.rerankers.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker
    if name == "SystemOneReranker":
        from dotfetch.adapters.rerankers.system_one import SystemOneReranker

        return SystemOneReranker
    if name == "JevReranker":  # renamed in 0.23.0
        from dotfetch._deprecation import warn_renamed
        from dotfetch.adapters.rerankers.system_one import SystemOneReranker

        warn_renamed(f"{__name__}.JevReranker", f"{__name__}.SystemOneReranker")
        return SystemOneReranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
