"""
MCP Server semantic router for Agent-Gantry.

Intelligent MCP server selection using semantic search and context.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from agent_gantry.adapters.embedders.base import embed_query
from agent_gantry.core.mcp_manager import CATALOGUE_ENTITY, PSEUDO_NAMESPACE, SERVER_ENTITY
from agent_gantry.core.router import UNAVAILABLE_SERVER_PENALTY, mmr_order

#: How many times a search may be widened when one server's catalogue pieces
#: crowd the other servers out of the window.
_MAX_WIDENINGS = 2

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
        diversity_factor: float = 0.0,
    ) -> MCPRoutingResult:
        """
        Route a query to the most relevant MCP servers.

        Args:
            query: Natural language query
            limit: Maximum number of servers to return
            score_threshold: Minimum similarity score
            namespaces: Filter by server namespaces
            diversity_factor: ``0`` ranks by relevance alone; above it, a
                server much like one already picked gives way to a different
                one (maximal marginal relevance).

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
        mcp_namespace_filter: dict[str, list[str]] = {"namespace": [PSEUDO_NAMESPACE]}

        # A server is indexed as its own entry and, once its tools have been
        # listed, as pieces of that catalogue. Results come back best first,
        # so the first row seen for a server is the best of them, and that is
        # its score. Several of one server's pieces can outrank every other
        # server, so the window is widened until ``limit`` distinct servers
        # are in it or the store has no more rows.
        diverse = diversity_factor > 0
        pool = limit * 4 if diverse else limit  # diversity chooses among more
        fetch = pool * 4  # extra candidates for filtering and for pieces
        scored_servers: list[MCPServerScore] = []
        vectors: dict[tuple[str, str], Sequence[float]] = {}
        for _ in range(_MAX_WIDENINGS + 1):
            candidates = await self._vector_store.search(
                query_vector=query_embedding,
                limit=fetch,
                filters=mcp_namespace_filter,
                score_threshold=score_threshold,
                include_embeddings=diverse,
            )
            # Map pseudo-tools back to registered servers. A stale pseudo-tool
            # (an older naming scheme, a deregistered server) either resolves
            # to nothing or to a server already seen, and is skipped either way.
            scored_servers = []
            vectors = {}
            seen: set[tuple[str, str]] = set()
            for pseudo_tool, score, *rest in candidates:
                if (
                    pseudo_tool.metadata.get("entity_type") not in (SERVER_ENTITY, CATALOGUE_ENTITY)
                    or pseudo_tool.namespace != PSEUDO_NAMESPACE
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
                    # A server whose last connection attempt failed is still
                    # offered, but below a healthy one of similar relevance:
                    # it used to route first however often it had failed.
                    penalty = 0.0 if server.health.available else UNAVAILABLE_SERVER_PENALTY
                    scored_servers.append(
                        MCPServerScore(server=server, score=max(0.0, score - penalty))
                    )
                    if rest:
                        # The vector of the row that matched best: the
                        # server's own entry, or a piece of its catalogue.
                        vectors[key] = rest[0]
            if len(scored_servers) >= pool or len(candidates) < fetch:
                break
            fetch *= 4
        scored_servers.sort(key=lambda entry: entry.score, reverse=True)
        scored_servers = scored_servers[:pool]
        # Each server kept must have its vector; ``vectors`` may also hold
        # those of servers the cut above dropped.
        if (
            diverse
            and len(scored_servers) > 1
            and all((e.server.namespace, e.server.name) in vectors for e in scored_servers)
        ):
            order = mmr_order(
                [entry.score for entry in scored_servers],
                [vectors[(entry.server.namespace, entry.server.name)] for entry in scored_servers],
                diversity_factor,
                limit,
            )
            scored_servers = [scored_servers[index] for index in order]
        scored_servers = scored_servers[:limit]
        search_time_ms = (perf_counter() - search_start) * 1000

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
