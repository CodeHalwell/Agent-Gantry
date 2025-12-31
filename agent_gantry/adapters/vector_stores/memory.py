"""
In-memory vector store implementation.
"""

from __future__ import annotations

import math
from typing import Any

from agent_gantry.schema.tool import ToolDefinition


class InMemoryVectorStore:
    """
    Simple in-memory vector store for development and testing.

    Uses cosine similarity for search.
    """

    def __init__(self, dimension: int = 0) -> None:
        """
        Initialize the in-memory store.

        Args:
            dimension: Vector dimension (optional, auto-detected from first embedding)
        """
        self._tools: dict[str, ToolDefinition] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._fingerprints: dict[str, str] = {}
        self._metadata: dict[str, str] = {}
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """
        Return the vector dimension.

        Returns the configured dimension, or auto-detects from first stored embedding.
        """
        if self._dimension > 0:
            return self._dimension
        # Auto-detect from first embedding
        if self._embeddings:
            first_embedding = next(iter(self._embeddings.values()))
            return len(first_embedding)
        return 0

    async def initialize(self) -> None:
        """Initialize the store (no-op for in-memory)."""
        pass

    async def add_tools(
        self,
        tools: list[ToolDefinition],
        embeddings: list[list[float]],
        upsert: bool = True,
    ) -> int:
        """Add tools with their embeddings."""
        count = 0
        for tool, embedding in zip(tools, embeddings):
            key = f"{tool.namespace}.{tool.name}"
            if key not in self._tools or upsert:
                self._tools[key] = tool
                self._embeddings[key] = embedding
                # Store fingerprint for change detection
                self._fingerprints[key] = tool.content_hash
                count += 1
        return count

    async def search(
        self,
        query_vector: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        include_embeddings: bool = False,
    ) -> list[tuple[ToolDefinition, float]] | list[tuple[ToolDefinition, float, list[float]]]:
        """Search for similar tools using cosine similarity."""
        results: list[tuple[ToolDefinition, float, list[float]]] = []

        for key, tool in self._tools.items():
            embedding = self._embeddings.get(key)
            if embedding is None:
                continue

            # Apply filters
            if filters:
                if "namespace" in filters:
                    ns_filter = filters["namespace"]
                    if isinstance(ns_filter, (list, tuple, set)):
                        if tool.namespace not in ns_filter:
                            continue
                    elif tool.namespace != ns_filter:
                        continue
                if "tags" in filters:
                    if not any(tag in tool.tags for tag in filters["tags"]):
                        continue

            # Calculate cosine similarity
            score = self._cosine_similarity(query_vector, embedding)

            if score_threshold is None or score >= score_threshold:
                results.append((tool, score, embedding))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Return with or without embeddings based on parameter
        limited = results[:limit]
        if include_embeddings:
            return limited
        return [(tool, score) for tool, score, _ in limited]

    async def get_by_name(
        self, name: str, namespace: str = "default"
    ) -> ToolDefinition | None:
        """Get a tool by name."""
        key = f"{namespace}.{name}"
        return self._tools.get(key)

    async def delete(self, name: str, namespace: str = "default") -> bool:
        """Delete a tool."""
        key = f"{namespace}.{name}"
        if key in self._tools:
            del self._tools[key]
            self._embeddings.pop(key, None)
            self._fingerprints.pop(key, None)
            return True
        return False

    async def list_all(
        self,
        namespace: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[ToolDefinition]:
        """List all tools."""
        tools = list(self._tools.values())
        if namespace:
            tools = [t for t in tools if t.namespace == namespace]
        return tools[offset : offset + limit]

    async def count(self, namespace: str | None = None) -> int:
        """Count tools."""
        if namespace:
            return sum(1 for t in self._tools.values() if t.namespace == namespace)
        return len(self._tools)

    async def health_check(self) -> bool:
        """Check health (always healthy for in-memory)."""
        return True

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def get_stored_fingerprints(self) -> dict[str, str]:
        """
        Get all stored tool fingerprints for change detection.

        Returns:
            Dictionary mapping tool_id (namespace.name) to fingerprint hash.
        """
        return dict(self._fingerprints)

    async def get_metadata(self, key: str) -> str | None:
        """
        Get a metadata value by key.

        Args:
            key: The metadata key to retrieve

        Returns:
            The metadata value if found, None otherwise.
        """
        return self._metadata.get(key)

    async def set_metadata(self, key: str, value: str) -> None:
        """
        Set a metadata value.

        Args:
            key: The metadata key
            value: The value to store
        """
        self._metadata[key] = value

    async def update_sync_metadata(self, embedder_id: str, dimension: int) -> None:
        """
        Update sync metadata after a successful sync operation.

        Args:
            embedder_id: Unique identifier for the embedder configuration
            dimension: Vector dimension
        """
        self._metadata["embedder_id"] = embedder_id
        self._metadata["dimension"] = str(dimension)
