"""
Stubbed production vector store adapters.

These adapters mimic Qdrant, Chroma, and PGVector behaviours while reusing the
in-memory implementation for storage. They allow config-driven switching without
introducing heavy external dependencies.
"""

from __future__ import annotations

from typing import Any

from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
from agent_gantry.schema.tool import ToolDefinition


class _ProxyVectorStore:
    """Proxy vector store that delegates to the in-memory implementation."""

    def __init__(self, url: str | None = None, api_key: str | None = None) -> None:
        self._delegate = InMemoryVectorStore()
        self._url = url
        self._api_key = api_key
        self._initialized = False

    async def initialize(self) -> None:
        await self._delegate.initialize()
        self._initialized = True

    async def add_tools(
        self,
        tools: list[ToolDefinition],
        embeddings: list[list[float]],
        upsert: bool = True,
    ) -> int:
        return await self._delegate.add_tools(tools, embeddings, upsert=upsert)

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[ToolDefinition, float]]:
        return await self._delegate.search(
            query_vector=query_vector,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
        )

    async def get_by_name(
        self, name: str, namespace: str = "default"
    ) -> ToolDefinition | None:
        return await self._delegate.get_by_name(name, namespace)

    async def delete(self, name: str, namespace: str = "default") -> bool:
        return await self._delegate.delete(name, namespace)

    async def list_all(
        self,
        namespace: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[ToolDefinition]:
        return await self._delegate.list_all(namespace=namespace, limit=limit, offset=offset)

    async def count(self, namespace: str | None = None) -> int:
        return await self._delegate.count(namespace=namespace)

    async def health_check(self) -> bool:
        """Treat missing endpoints as unhealthy to surface misconfiguration early."""
        has_endpoint = bool(self._url or self._api_key)
        return self._initialized and has_endpoint


class QdrantVectorStore(_ProxyVectorStore):
    """Qdrant adapter backed by the in-memory store."""


class ChromaVectorStore(_ProxyVectorStore):
    """Chroma adapter backed by the in-memory store."""


class PGVectorStore(_ProxyVectorStore):
    """PGVector adapter backed by the in-memory store."""
