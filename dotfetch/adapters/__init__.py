"""
Adapter protocols and implementations for dotfetch.

Contains adapters for vector stores, embedders, rerankers, executors, and tool specs.
"""

from dotfetch.adapters.embedders.base import EmbeddingAdapter
from dotfetch.adapters.executors.base import ExecutorAdapter
from dotfetch.adapters.rerankers.base import RerankerAdapter
from dotfetch.adapters.tool_spec import (
    DialectRegistry,
    ToolCallPayload,
    ToolSpecAdapter,
    get_adapter,
)
from dotfetch.adapters.vector_stores.base import VectorStoreAdapter

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
