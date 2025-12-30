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
        include_embeddings: bool = False,
    ) -> list[tuple[ToolDefinition, float]] | list[tuple[ToolDefinition, float, list[float]]]:
        """
        Search for tools similar to the query vector.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filters: Optional filters to apply
            score_threshold: Minimum score threshold
            include_embeddings: If True, return embeddings along with tools

        Returns:
            List of (tool, score) tuples if include_embeddings=False
            List of (tool, score, embedding) tuples if include_embeddings=True
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

    async def get_stored_fingerprints(self) -> dict[str, str]:
        """
        Get all stored tool fingerprints for change detection.

        This method enables smart sync by returning fingerprints of all tools
        currently in the vector store. Vector stores that don't support
        fingerprinting can use the default implementation.

        Returns:
            Dictionary mapping tool_id (namespace.name) to fingerprint hash.
            Default implementation returns empty dict (no fingerprinting support).
        """
        return {}

    async def get_metadata(self, key: str) -> str | None:
        """
        Get a metadata value by key.

        This method supports storing vector store metadata such as embedder_id
        and dimension for consistency checking. Vector stores that don't support
        metadata storage can use the default implementation.

        Args:
            key: The metadata key to retrieve

        Returns:
            The metadata value if found, None otherwise.
            Default implementation returns None (no metadata support).
        """
        return None

    async def set_metadata(self, key: str, value: str) -> None:
        """
        Set a metadata value.

        This method supports storing vector store metadata such as embedder_id
        and dimension for consistency checking. Vector stores that don't support
        metadata storage can use the default implementation (no-op).

        Args:
            key: The metadata key
            value: The value to store
        """
        pass

    async def update_sync_metadata(self, embedder_id: str, dimension: int) -> None:
        """
        Update sync metadata after a successful sync operation.

        This is a convenience method that updates both embedder_id and dimension
        metadata. Vector stores that don't support metadata storage can use the
        default implementation (no-op).

        Args:
            embedder_id: Unique identifier for the embedder configuration
            dimension: Vector dimension
        """
        pass
