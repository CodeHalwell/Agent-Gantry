"""
Vector store adapters for dotfetch.
"""

from __future__ import annotations

from dotfetch.adapters.vector_stores.base import VectorStoreAdapter
from dotfetch.adapters.vector_stores.memory import InMemoryVectorStore

__all__ = [
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "LanceDBVectorStore",
    "PGVectorStore",
    "QdrantVectorStore",
    "VectorStoreAdapter",
]


def __getattr__(name: str) -> type:
    """Lazy import for optional dependencies."""
    if name == "LanceDBVectorStore":
        from dotfetch.adapters.vector_stores.lancedb import LanceDBVectorStore

        return LanceDBVectorStore
    if name in ("QdrantVectorStore", "ChromaVectorStore", "PGVectorStore"):
        module = __import__("dotfetch.adapters.vector_stores.remote", fromlist=[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
