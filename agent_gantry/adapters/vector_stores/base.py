"""
Base vector store adapter protocol.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Protocol

from agent_gantry.schema.tool import ToolDefinition


class VectorStoreAdapter(Protocol):
    """
    Vector DB abstraction for tools.

    Implementations: QdrantAdapter, ChromaAdapter, PGVectorAdapter,
                     PineconeAdapter, WeaviateAdapter, InMemoryAdapter.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Idempotent setup of collections / indexes."""
        ...

    @abstractmethod
    async def add_tools(
        self,
        tools: list[ToolDefinition],
        embeddings: list[list[float]],
        upsert: bool = True,
    ) -> int:
        """
        Add tools with their embeddings to the store.

        Args:
            tools: List of tool definitions
            embeddings: List of embedding vectors
            upsert: Whether to update existing tools

        Returns:
            Number of tools added/updated
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[ToolDefinition, float]]:
        """
        Search for tools similar to the query vector.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filters: Optional filters to apply
            score_threshold: Minimum score threshold

        Returns:
            List of (tool, score) tuples
        """
        ...

    @abstractmethod
    async def get_by_name(
        self, name: str, namespace: str = "default"
    ) -> ToolDefinition | None:
        """
        Get a tool by name.

        Args:
            name: Tool name
            namespace: Tool namespace

        Returns:
            The tool if found
        """
        ...

    @abstractmethod
    async def delete(self, name: str, namespace: str = "default") -> bool:
        """
        Delete a tool.

        Args:
            name: Tool name
            namespace: Tool namespace

        Returns:
            True if deleted
        """
        ...

    @abstractmethod
    async def list_all(
        self,
        namespace: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[ToolDefinition]:
        """
        List all tools.

        Args:
            namespace: Filter by namespace
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of tools
        """
        ...

    @abstractmethod
    async def count(self, namespace: str | None = None) -> int:
        """
        Count tools.

        Args:
            namespace: Filter by namespace

        Returns:
            Number of tools
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check health of the vector store.

        Returns:
            True if healthy
        """
        ...
