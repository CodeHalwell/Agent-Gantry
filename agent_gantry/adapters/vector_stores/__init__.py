"""
Vector store adapters for Agent-Gantry.
"""

from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore

__all__ = [
    "InMemoryVectorStore",
    "VectorStoreAdapter",
]
