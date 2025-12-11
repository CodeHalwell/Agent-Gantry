"""
Adapter protocols and implementations for Agent-Gantry.

Contains adapters for vector stores, embedders, rerankers, and executors.
"""

from agent_gantry.adapters.embedders.base import EmbeddingAdapter
from agent_gantry.adapters.executors.base import ExecutorAdapter
from agent_gantry.adapters.rerankers.base import RerankerAdapter
from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter

__all__ = [
    "EmbeddingAdapter",
    "ExecutorAdapter",
    "RerankerAdapter",
    "VectorStoreAdapter",
]
