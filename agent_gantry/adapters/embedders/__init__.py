"""
Embedding adapters for Agent-Gantry.

The base protocol :class:`EmbeddingAdapter` and the dependency-free
:class:`SimpleEmbedder` are imported eagerly. Heavier adapters that pull in
optional third-party dependencies (sentence-transformers, OpenAI SDK,
LlamaIndex, etc.) are lazy-loaded via ``__getattr__`` so importing this
module never forces optional installs.

All names listed in ``__all__`` are accessible from this module — including
the lazy ones — so a single ``from agent_gantry.adapters.embedders import X``
works regardless of whether ``X`` lives behind an optional dep:

.. code-block:: python

    from agent_gantry.adapters.embedders import (
        SimpleEmbedder,
        SentenceTransformersEmbedder,  # requires sentence-transformers
        NomicEmbedder,                  # requires sentence-transformers
        OpenAIEmbedder,                 # requires openai
        AzureOpenAIEmbedder,            # requires openai
    )
"""

from agent_gantry.adapters.embedders.base import EmbeddingAdapter
from agent_gantry.adapters.embedders.simple import SimpleEmbedder

__all__ = [
    "AzureOpenAIEmbedder",
    "EmbeddingAdapter",
    "NomicEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformersEmbedder",
    "SimpleEmbedder",
]


def __getattr__(name: str) -> type:
    """Lazy import for optional dependencies."""
    if name == "NomicEmbedder":
        from agent_gantry.adapters.embedders.nomic import NomicEmbedder

        return NomicEmbedder
    if name == "SentenceTransformersEmbedder":
        from agent_gantry.adapters.embedders.sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        return SentenceTransformersEmbedder
    if name == "OpenAIEmbedder":
        from agent_gantry.adapters.embedders.openai import OpenAIEmbedder

        return OpenAIEmbedder
    if name == "AzureOpenAIEmbedder":
        from agent_gantry.adapters.embedders.openai import AzureOpenAIEmbedder

        return AzureOpenAIEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
