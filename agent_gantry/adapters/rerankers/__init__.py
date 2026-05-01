"""
Reranker adapters for Agent-Gantry.

The base :class:`RerankerAdapter` protocol is imported eagerly; concrete
implementations that pull in optional dependencies (cohere, sentence-transformers
cross-encoders) are lazy-loaded via ``__getattr__`` so importing this module
does not require optional installs.
"""

from agent_gantry.adapters.rerankers.base import RerankerAdapter

__all__ = [
    "CohereReranker",
    "CrossEncoderReranker",
    "RerankerAdapter",
]


def __getattr__(name: str) -> type:
    if name == "CohereReranker":
        from agent_gantry.adapters.rerankers.cohere import CohereReranker

        return CohereReranker
    if name == "CrossEncoderReranker":
        from agent_gantry.adapters.rerankers.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
