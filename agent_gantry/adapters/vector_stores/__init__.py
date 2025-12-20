"""
Vector store adapters for Agent-Gantry.
"""

from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore

__all__ = [
    "InMemoryVectorStore",
    "LanceDBVectorStore",
    "VectorStoreAdapter",
]


def __getattr__(name: str) -> type:
    """Lazy import for optional dependencies."""
    if name == "LanceDBVectorStore":
        from agent_gantry.adapters.vector_stores.lancedb import LanceDBVectorStore

        return LanceDBVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
