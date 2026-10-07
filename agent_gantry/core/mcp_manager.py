"""
MCP server lifecycle manager for Agent-Gantry.

Handles MCP server registration and sync for dynamic (semantic) server
selection. :class:`~agent_gantry.core.gantry.AgentGantry` delegates
``register_mcp_server`` / ``sync_mcp_servers`` here; on-demand tool discovery
stays on the facade because it wires execution handlers into the tool
registry.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING

from agent_gantry.schema.mcp import PSEUDO_NAMESPACE, MCPServerDefinition
from agent_gantry.schema.tool import ToolDefinition

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.core.mcp_registry import MCPRegistry
    from agent_gantry.core.sync_manager import SyncManager

logger = logging.getLogger(__name__)

#: Namespace of the pseudo-tools that stand in for MCP servers in the store.


def _pseudo_tool_name(server: MCPServerDefinition) -> str:
    """Name of the pseudo-tool ``server`` is embedded as.

    Sanitised to ``ToolDefinition``'s name pattern; the digest keeps two servers
    whose sanitised names coincide (``b_c.a`` and ``b.c_a``) apart.
    """
    qualified = f"{server.namespace}.{server.name}"
    stem = re.sub(r"[^a-z0-9_]+", "_", qualified.lower()).strip("_")[:100]
    digest = hashlib.sha256(qualified.encode()).hexdigest()[:8]
    return f"mcp_server_{stem}_{digest}"


def _pseudo_tool(server: MCPServerDefinition) -> ToolDefinition:
    """The ``ToolDefinition`` that stands in for ``server`` in the vector store."""
    return ToolDefinition(
        name=_pseudo_tool_name(server),
        namespace=PSEUDO_NAMESPACE,
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


class MCPManager:
    """Registers MCP servers and syncs them into the vector store as pseudo-tools."""

    def __init__(
        self,
        vector_store: VectorStoreAdapter,
        embedder: EmbeddingAdapter,
        registry: MCPRegistry,
        sync_manager: SyncManager,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._registry = registry
        self._sync_manager = sync_manager

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

        # A re-registration may point somewhere else now; a client cached from
        # the previous definition would keep using the old endpoint.
        previous = self._registry.get_server(name, namespace)
        if previous is not None and previous.to_config() != server_def.to_config():
            self._registry.forget_client(name, namespace)

        self._registry.register_server(server_def)
        self._registry.add_pending(server_def)
        logger.info(f"Registered MCP server: {server_def.qualified_name}")

    async def sync_servers(self, batch_size: int = 100, force: bool = False) -> int:
        """Embed new or changed servers into the vector store.

        Change detection is the tool sync's: fingerprints, plus a full re-sync
        when the embedder or its dimension changed.

        Args:
            batch_size: Number of servers per batch
            force: If True, re-embed all servers

        Returns:
            Number of servers synced
        """
        all_servers = self._registry.list_servers()
        # Pending entries this sync answers for; a registration landing while
        # the awaits below run is not in the snapshot and survives the drain.
        pending_snapshot = self._registry.get_pending()
        if not all_servers:
            return 0

        pseudo_tools = [_pseudo_tool(server) for server in all_servers]
        to_sync = await self._sync_manager.detect_changes(pseudo_tools, force)
        self._registry.drain_pending(pending_snapshot)
        total_synced = 0
        if to_sync:
            logger.info(f"Syncing {len(to_sync)}/{len(all_servers)} MCP servers...")
            for i in range(0, len(to_sync), batch_size):
                batch = to_sync[i : i + batch_size]
                # A pseudo-tool's description is the server's searchable text.
                embeddings = await self._embedder.embed_batch([tool.description for tool in batch])
                total_synced += await self._vector_store.add_tools(batch, embeddings, upsert=True)
            await self._sync_manager.update_metadata()
            logger.info(f"Synced {total_synced} MCP servers")
        else:
            logger.debug(f"All {len(all_servers)} MCP servers up-to-date, skipping sync")
        # After the upsert, so a renamed server is never briefly absent.
        await self._prune_pseudo_tools({tool.name for tool in pseudo_tools})
        return total_synced

    async def _prune_pseudo_tools(self, wanted: set[str]) -> int:
        """Delete pseudo-tool rows that no registered server owns.

        Sync only ever upserts, so a persistent store kept the row of a
        server removed from the registry, and after the digest-based rename
        it kept every row under the old name beside the new one. The router
        fetches a bounded candidate window, so such duplicates could crowd
        other servers out. As with ``AgentGantry.prune_stale_tools``, a store
        shared between gantries is pruned to *this* gantry's servers; a
        gantry with no servers never gets here.
        """
        stored: list[ToolDefinition] = []
        offset = 0
        while True:
            page = list(
                await self._vector_store.list_all(
                    namespace=PSEUDO_NAMESPACE, limit=1000, offset=offset
                )
            )
            stored.extend(page)
            if len(page) < 1000:
                break
            offset += 1000
        removed = 0
        for tool in stored:
            if tool.name in wanted:
                continue
            if await self._vector_store.delete(tool.name, PSEUDO_NAMESPACE):
                removed += 1
        if removed:
            logger.info(f"Pruned {removed} MCP server pseudo-tool(s) no registered server owns")
        return removed
