"""
MCP server lifecycle manager for Agent-Gantry.

Handles MCP server registration, sync, discovery, and serving.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from agent_gantry.schema.mcp import MCPServerDefinition
from agent_gantry.schema.tool import ToolDefinition
from agent_gantry.utils.fingerprint import compute_tool_fingerprint

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.core.mcp_registry import MCPRegistry
    from agent_gantry.core.mcp_router import MCPRouter
    from agent_gantry.schema.config import MCPServerConfig

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
        command: list[str],
        *,
        description: str,
        namespace: str = "default",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        tags: list[str] | None = None,
        examples: list[str] | None = None,
        capabilities: list[str] | None = None,
    ) -> None:
        """Register an MCP server for dynamic semantic selection."""
        server_def = MCPServerDefinition(
            name=name,
            namespace=namespace,
            description=description,
            command=command,
            args=args or [],
            env=env or {},
            tags=tags or [],
            examples=examples or [],
            capabilities=capabilities or [],
        )

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

        self._registry.clear_pending()

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

    async def add_server(self, config: MCPServerConfig) -> int:
        """
        Add an MCP server by immediately discovering all its tools.

        Returns:
            Number of tools discovered
        """
        from agent_gantry.adapters.executors.mcp_client import MCPClient

        client = MCPClient(config)
        tools = await client.list_tools()
        return len(tools), tools

    async def discover_tools_from_server(
        self,
        server_name: str,
        namespace: str = "default",
        timeout: float = 30.0,
    ) -> int:
        """
        Dynamically discover tools from a previously registered MCP server.

        Returns:
            Number of tools discovered

        Raises:
            ValueError: If server not registered
            TimeoutError: If connection times out
        """
        client = self._registry.get_client(server_name, namespace)
        if not client:
            raise ValueError(
                f"MCP server '{namespace}.{server_name}' not found. "
                f"Register it first with register_mcp_server()."
            )

        try:
            tools = await asyncio.wait_for(client.list_tools(), timeout=timeout)

            self._registry.update_health(
                server_name,
                namespace,
                available=True,
                last_success=datetime.now(timezone.utc),
                consecutive_failures=0,
            )

            logger.info(
                f"Discovered {len(tools)} tools from MCP server: {namespace}.{server_name}"
            )
            return len(tools), tools

        except asyncio.TimeoutError:
            server = self._registry.get_server(server_name, namespace)
            consecutive_failures = server.health.consecutive_failures + 1 if server else 1

            self._registry.update_health(
                server_name,
                namespace,
                available=False,
                last_failure=datetime.now(timezone.utc),
                consecutive_failures=consecutive_failures,
            )

            logger.error(
                f"Timeout discovering tools from MCP server {namespace}.{server_name} "
                f"(timeout: {timeout}s)"
            )
            raise TimeoutError(
                f"MCP server {namespace}.{server_name} did not respond within {timeout}s"
            )

        except Exception as e:
            server = self._registry.get_server(server_name, namespace)
            consecutive_failures = server.health.consecutive_failures + 1 if server else 1

            self._registry.update_health(
                server_name,
                namespace,
                available=False,
                last_failure=datetime.now(timezone.utc),
                consecutive_failures=consecutive_failures,
            )

            logger.error(
                f"Failed to discover tools from MCP server {namespace}.{server_name}: {e}"
            )
            raise
