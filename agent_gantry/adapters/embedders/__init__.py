"""
Embedding adapters for Agent-Gantry.
"""

from agent_gantry.adapters.embedders.base import EmbeddingAdapter
from agent_gantry.adapters.embedders.simple import SimpleEmbedder

__all__ = [
    "EmbeddingAdapter",
    "NomicEmbedder",
    "SimpleEmbedder",
]


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "NomicEmbedder":
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        return NomicEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
