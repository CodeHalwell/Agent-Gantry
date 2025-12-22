"""
Adapter protocols and implementations for Agent-Gantry.

Contains adapters for vector stores, embedders, rerankers, executors, and tool specs.
"""

from agent_gantry.adapters.embedders.base import EmbeddingAdapter
from agent_gantry.adapters.executors.base import ExecutorAdapter
from agent_gantry.adapters.rerankers.base import RerankerAdapter
from agent_gantry.adapters.tool_spec import (
    DialectRegistry,
    ToolCallPayload,
    ToolSpecAdapter,
    get_adapter,
)
from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter

__all__ = [
    "DialectRegistry",
    "EmbeddingAdapter",
    "ExecutorAdapter",
    "RerankerAdapter",
    "ToolCallPayload",
    "ToolSpecAdapter",
    "VectorStoreAdapter",
    "get_adapter",
]
