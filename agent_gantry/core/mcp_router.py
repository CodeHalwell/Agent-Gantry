"""
MCP Server semantic router for Agent-Gantry.

Intelligent MCP server selection using semantic search and context.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.schema.mcp import MCPServerDefinition


@dataclass
class MCPServerScore:
    """Scored MCP server result."""

    server: MCPServerDefinition
    score: float


@dataclass
class MCPRoutingResult:
    """MCP server routing outcome with timing metadata."""

    servers: list[MCPServerScore]
    query_embedding_time_ms: float
    search_time_ms: float
    total_time_ms: float


class MCPRouter:
    """
    Semantic router for intelligent MCP server selection.

    Similar to SemanticRouter but for MCP servers instead of tools.
    Uses vector similarity search to find the most relevant servers
    for a given query.
    """

    def __init__(
        self,
        vector_store: VectorStoreAdapter,
        embedder: EmbeddingAdapter,
    ) -> None:
        """
        Initialize the MCP router.

        Args:
            vector_store: Vector store for server embeddings
            embedder: Embedding model for queries
        """
        self._vector_store = vector_store
        self._embedder = embedder

    async def route(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float | None = None,
        namespaces: list[str] | None = None,
    ) -> MCPRoutingResult:
        """
        Route a query to the most relevant MCP servers.

        Args:
            query: Natural language query
            limit: Maximum number of servers to return
            score_threshold: Minimum similarity score
            namespaces: Filter by server namespaces

        Returns:
            Routing result with scored servers and timings
        """
        start_time = perf_counter()

        # Embed the query
        embed_start = perf_counter()
        query_embedding = await self._embedder.embed_text(query)
        query_embedding_time_ms = (perf_counter() - embed_start) * 1000

        # Prepare filters
        filters: dict[str, list[str]] | None = None
        if namespaces:
            filters = {"namespace": namespaces}

        # Search for relevant servers
        search_start = perf_counter()
        # Note: We're reusing the same vector store but with a different collection/tag
        # The vector store implementation should support storing both tools and MCP servers
        _ = await self._vector_store.search(
            query_vector=query_embedding,
            limit=limit * 2,  # Get extra candidates for filtering
            filters=filters,
            score_threshold=score_threshold,
        )
        search_time_ms = (perf_counter() - search_start) * 1000

        # Convert results to MCPServerScore
        # Note: candidates contain tools, but we'll need to handle MCP servers separately
        # For now, we'll assume the vector store can distinguish between tools and servers
        scored_servers: list[MCPServerScore] = []

        # This is a placeholder - actual implementation will require
        # the vector store to support multiple entity types
        # For now, we'll return empty list and handle this in integration
        # When vector store integration is complete, we'll process candidates like:
        # for candidate in candidates[:limit]:
        #     entity, score = candidate[0], candidate[1]
        #     if isinstance(entity, MCPServerDefinition):
        #         scored_servers.append(MCPServerScore(server=entity, score=score))

        total_time_ms = (perf_counter() - start_time) * 1000

        return MCPRoutingResult(
            servers=scored_servers,
            query_embedding_time_ms=query_embedding_time_ms,
            search_time_ms=search_time_ms,
            total_time_ms=total_time_ms,
        )

    async def filter_by_capabilities(
        self,
        servers: list[MCPServerDefinition],
        required_capabilities: list[str],
    ) -> list[MCPServerDefinition]:
        """
        Filter servers by required capabilities.

        Args:
            servers: List of server definitions
            required_capabilities: Capabilities that must be present

        Returns:
            Filtered list of servers
        """
        if not required_capabilities:
            return servers

        return [
            server
            for server in servers
            if all(cap in server.capabilities for cap in required_capabilities)
        ]

    async def filter_by_health(
        self,
        servers: list[MCPServerDefinition],
        exclude_unavailable: bool = True,
    ) -> list[MCPServerDefinition]:
        """
        Filter servers by health status.

        Args:
            servers: List of server definitions
            exclude_unavailable: Whether to exclude unavailable servers

        Returns:
            Filtered list of servers
        """
        if not exclude_unavailable:
            return servers

        return [server for server in servers if server.health.available]
