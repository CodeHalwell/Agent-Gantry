"""
Vector store adapters for Agent-Gantry.
"""

from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore

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
        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        return LanceDBVectorStore
    if name in ("QdrantVectorStore", "ChromaVectorStore", "PGVectorStore"):
        module = __import__(
            "agent_gantry.adapters.vector_stores.remote",
            fromlist=[name]
        )
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
