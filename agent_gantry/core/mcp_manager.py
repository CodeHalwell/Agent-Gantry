"""
MCP server lifecycle manager for Agent-Gantry.

Handles MCP server registration and sync for dynamic (semantic) server
selection. :class:`~agent_gantry.core.gantry.AgentGantry` delegates
``register_mcp_server`` / ``sync_mcp_servers`` here; on-demand tool discovery
stays on the facade because it wires execution handlers into the tool
registry.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agent_gantry.schema.mcp import MCPServerDefinition
from agent_gantry.schema.tool import ToolDefinition
from agent_gantry.utils.fingerprint import compute_tool_fingerprint

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.core.mcp_registry import MCPRegistry
    from agent_gantry.core.mcp_router import MCPRouter

logger = logging.getLogger(__name__)


class MCPManager:
    """
    Manages MCP server lifecycle: registration, sync, discovery, and serving.

    Extracted from AgentGantry to keep the facade thin.
    """

    def __init__(
        self,
        vector_store: VectorStoreAdapter,
        embedder: EmbeddingAdapter,
        registry: MCPRegistry,
        router: MCPRouter,
        get_embedder_id: callable,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._registry = registry
        self._router = router
        self._get_embedder_id = get_embedder_id
        self._synced = False

    @property
    def synced(self) -> bool:
        return self._synced

    async def ensure_synced(self) -> None:
        """Ensure MCP servers are synced to the vector store."""
        if not self._synced:
            await self.sync_servers()

    def register_server(
        self,
        name: str,
        command: list[str] | None = None,
        *,
        description: str,
        namespace: str = "default",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        tags: list[str] | None = None,
        examples: list[str] | None = None,
        capabilities: list[str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        transport: str | None = None,
    ) -> None:
        """Register an MCP server for dynamic semantic selection.

        Give either ``command`` (local stdio server) or ``url`` (remote
        Streamable HTTP / SSE server); see
        :class:`~agent_gantry.schema.mcp.MCPServerDefinition`.
        """
        server_def = MCPServerDefinition(
            name=name,
            namespace=namespace,
            description=description,
            command=command or [],
            args=args or [],
            env=env or {},
            tags=tags or [],
            examples=examples or [],
            capabilities=capabilities or [],
            url=url,
            headers=headers or {},
            transport=transport,  # type: ignore[arg-type]
        )

        # Re-registering under the same name may point somewhere else now
        # (a new url, headers or command). A client cached from the previous
        # definition would keep discovering and executing against the old
        # endpoint, so it is dropped here; the next lookup builds one from
        # the definition just registered.
        previous = self._registry.get_server(name, namespace)
        if previous is not None and previous.to_config() != server_def.to_config():
            self._registry.forget_client(name, namespace)

        self._registry.register_server(server_def)
        self._registry.add_pending(server_def)
        logger.info(f"Registered MCP server: {server_def.qualified_name}")

    async def sync_servers(self, batch_size: int = 100, force: bool = False) -> int:
        """
        Sync MCP server registrations to vector store with fingerprint detection.

        Args:
            batch_size: Number of servers per batch
            force: If True, re-embed all servers

        Returns:
            Number of servers synced
        """
        all_servers = self._registry.list_servers()
        # The pending entries *this* sync answers for. A register_mcp_server()
        # landing while the awaits below are in flight appends to the buffer
        # but is not in this snapshot, so it must survive the drain.
        pending_snapshot = self._registry.get_pending()
        if not all_servers:
            self._synced = True
            return 0

        # Build pseudo-tools for embedding
        pseudo_tools_map: dict[str, ToolDefinition] = {}
        for server in all_servers:
            pseudo_name = f"mcp_server_{server.namespace}_{server.name}".replace("-", "_")
            pseudo_tool = ToolDefinition(
                name=pseudo_name,
                namespace="__mcp_servers__",
                description=server.to_searchable_text(),
                parameters_schema={"type": "object", "properties": {}},
                metadata={
                    "entity_type": "mcp_server",
                    "server_name": server.name,
                    "server_namespace": server.namespace,
                    "server_tags": server.tags,
                    "server_capabilities": server.capabilities,
                    "server_command": server.command,
                },
            )
            pseudo_tools_map[f"{server.namespace}.{server.name}"] = pseudo_tool

        # Compute fingerprints
        current_fingerprints = {
            f"{server.namespace}.{server.name}": compute_tool_fingerprint(pseudo_tool)
            for server in all_servers
            for pseudo_tool in [pseudo_tools_map[f"{server.namespace}.{server.name}"]]
        }

        embedder_id = self._get_embedder_id()
        needs_full_resync = force

        stored_fingerprints = await self._vector_store.get_stored_fingerprints()
        stored_embedder = await self._vector_store.get_metadata("embedder_id")
        stored_dim = await self._vector_store.get_metadata("dimension")

        if stored_embedder and stored_embedder != embedder_id:
            logger.info(
                f"Embedder changed from '{stored_embedder}' to '{embedder_id}'. "
                "Full re-sync required for MCP servers."
            )
            needs_full_resync = True
        elif stored_dim and int(stored_dim) != self._vector_store.dimension:
            logger.info(
                f"Dimension changed from {stored_dim} to {self._vector_store.dimension}. "
                "Full re-sync required for MCP servers."
            )
            needs_full_resync = True

        if needs_full_resync:
            servers_to_sync = all_servers
        else:
            servers_to_sync = []
            for server in all_servers:
                server_id = f"{server.namespace}.{server.name}"
                pseudo_name = f"mcp_server_{server.namespace}_{server.name}".replace("-", "_")
                pseudo_tool_id = f"__mcp_servers__.{pseudo_name}"

                current_fp = current_fingerprints[server_id]
                stored_fp = stored_fingerprints.get(pseudo_tool_id, "")

                if current_fp != stored_fp:
                    servers_to_sync.append(server)
                    if stored_fp:
                        logger.debug(f"MCP server '{server_id}' changed, will re-embed")
                    else:
                        logger.debug(f"MCP server '{server_id}' is new, will embed")

        self._registry.drain_pending(pending_snapshot)

        if not servers_to_sync:
            logger.debug(f"All {len(all_servers)} MCP servers up-to-date, skipping sync")
            self._synced = True
            return 0

        logger.info(f"Syncing {len(servers_to_sync)}/{len(all_servers)} MCP servers...")

        total_synced = 0
        for i in range(0, len(servers_to_sync), batch_size):
            batch = servers_to_sync[i : i + batch_size]
            texts = [s.to_searchable_text() for s in batch]
            embeddings = await self._embedder.embed_batch(texts)
            pseudo_tools = [pseudo_tools_map[f"{s.namespace}.{s.name}"] for s in batch]
            count = await self._vector_store.add_tools(pseudo_tools, embeddings, upsert=True)
            total_synced += count

        await self._vector_store.update_sync_metadata(
            embedder_id=embedder_id,
            dimension=self._vector_store.dimension,
        )

        self._synced = True
        logger.info(f"Synced {total_synced} MCP servers")
        return total_synced

    async def retrieve_servers(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float | None = None,
        namespaces: list[str] | None = None,
    ) -> list[MCPServerDefinition]:
        """Retrieve relevant MCP servers based on a query."""
        await self.ensure_synced()

        result = await self._router.route(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
        )
        return [scored.server for scored in result.servers]
