"""
MCP Server semantic router for Agent-Gantry.

Intelligent MCP server selection using semantic search and context.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from agent_gantry.adapters.embedders.base import embed_query

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.core.mcp_registry import MCPRegistry
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
        registry: MCPRegistry | None = None,
    ) -> None:
        """
        Initialize the MCP router.

        Args:
            vector_store: Vector store for server embeddings
            embedder: Embedding model for queries
            registry: MCP registry for looking up server definitions
        """
        self._vector_store = vector_store
        self._embedder = embedder
        self._registry = registry

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
        query_embedding = await embed_query(self._embedder, query)
        query_embedding_time_ms = (perf_counter() - embed_start) * 1000

        # Search for relevant servers
        search_start = perf_counter()
        # Search the vector store for MCP server pseudo-tools
        # These are stored with namespace "__mcp_servers__" to distinguish from real tools
        mcp_namespace_filter: dict[str, list[str]] = {"namespace": ["__mcp_servers__"]}

        candidates = await self._vector_store.search(
            query_vector=query_embedding,
            limit=limit * 2,  # Get extra candidates for filtering
            filters=mcp_namespace_filter,
            score_threshold=score_threshold,
        )
        search_time_ms = (perf_counter() - search_start) * 1000

        # Map pseudo-tools back to registered servers. A stale pseudo-tool
        # (an older naming scheme, a deregistered server) either resolves to
        # nothing or to a server already seen, and is skipped either way.
        scored_servers: list[MCPServerScore] = []
        seen: set[tuple[str, str]] = set()
        for pseudo_tool, score, *_ in candidates:
            if len(scored_servers) >= limit:
                break
            if (
                pseudo_tool.metadata.get("entity_type") != "mcp_server"
                or pseudo_tool.namespace != "__mcp_servers__"
            ):
                continue
            server_name = pseudo_tool.metadata.get("server_name")
            server_namespace = pseudo_tool.metadata.get("server_namespace", "default")
            if not server_name or (namespaces and server_namespace not in namespaces):
                continue
            key = (server_namespace, server_name)
            if key in seen:
                continue
            seen.add(key)
            server = await self._get_server_from_registry(server_name, server_namespace)
            if server:
                scored_servers.append(MCPServerScore(server=server, score=score))

        total_time_ms = (perf_counter() - start_time) * 1000

        return MCPRoutingResult(
            servers=scored_servers,
            query_embedding_time_ms=query_embedding_time_ms,
            search_time_ms=search_time_ms,
            total_time_ms=total_time_ms,
        )

    async def _get_server_from_registry(
        self,
        server_name: str,
        server_namespace: str = "default",
    ) -> MCPServerDefinition | None:
        """
        Get server definition from registry.

        Args:
            server_name: Name of the server
            server_namespace: Namespace of the server

        Returns:
            Server definition if found
        """
        if self._registry is None:
            return None
        return self._registry.get_server(server_name, server_namespace)
