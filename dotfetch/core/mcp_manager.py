"""
MCP server lifecycle manager for dotfetch.

Handles MCP server registration and sync for dynamic (semantic) server
selection. :class:`~dotfetch.core.fetcher.DotFetch` delegates
``register_mcp_server`` / ``sync_mcp_servers`` here; on-demand tool discovery
stays on the facade because it wires execution handlers into the tool
registry.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from dotfetch.core.chunks import CHUNK_CHARS
from dotfetch.schema.mcp import MCPServerDefinition
from dotfetch.schema.tool import MCP_SERVER_NAMESPACE, ToolDefinition

if TYPE_CHECKING:
    from dotfetch.adapters.embedders.base import EmbeddingAdapter
    from dotfetch.adapters.vector_stores.base import VectorStoreAdapter
    from dotfetch.core.mcp_registry import MCPRegistry
    from dotfetch.core.sync_manager import SyncManager

logger = logging.getLogger(__name__)

#: Namespace of the pseudo-tools that stand in for MCP servers in the store.
PSEUDO_NAMESPACE = MCP_SERVER_NAMESPACE

#: ``metadata["entity_type"]`` of a server's own entry, and of a piece of its
#: tool catalogue. The router credits both to the server.
SERVER_ENTITY = "mcp_server"
CATALOGUE_ENTITY = "mcp_server_tools"


def _pseudo_tool_name(server: MCPServerDefinition) -> str:
    """Name of the pseudo-tool ``server`` is embedded as.

    Sanitised to ``ToolDefinition``'s name pattern; the digest keeps two servers
    whose sanitised names coincide (``b_c.a`` and ``b.c_a``) apart.
    """
    qualified = f"{server.namespace}.{server.name}"
    stem = re.sub(r"[^a-z0-9_]+", "_", qualified.lower()).strip("_")[:100]
    digest = hashlib.sha256(qualified.encode()).hexdigest()[:8]
    return f"mcp_server_{stem}_{digest}"


#: ``ToolDefinition.description``'s ceiling, which the pseudo-tool's text must fit.
_DESCRIPTION_MAX = 2000
_EXTENDED_DESCRIPTION_MAX = 10000


def _pseudo_tool(server: MCPServerDefinition) -> ToolDefinition:
    """The ``ToolDefinition`` that stands in for ``server`` in the vector store."""
    text = server.to_searchable_text()
    return ToolDefinition(
        name=_pseudo_tool_name(server),
        namespace=PSEUDO_NAMESPACE,
        # A server may carry up to 10,000 characters of extended description,
        # and a tool's description holds 2,000: without the cut, a server with
        # a long one could not be indexed at all (it failed validation here).
        description=text[:_DESCRIPTION_MAX],
        extended_description=(
            text[:_EXTENDED_DESCRIPTION_MAX] if len(text) > _DESCRIPTION_MAX else None
        ),
        parameters_schema={"type": "object", "properties": {}},
        metadata={
            "entity_type": SERVER_ENTITY,
            "server_name": server.name,
            "server_namespace": server.namespace,
            "server_tags": server.tags,
            "server_capabilities": server.capabilities,
            "server_command": server.command,
        },
    )


#: Most pieces one server's tool catalogue is indexed in.
MAX_CATALOGUE_CHUNKS = 48


def catalogue_chunks(server: MCPServerDefinition, tools: Sequence[ToolDefinition]) -> list[ToolDefinition]:
    """Index entries that say what ``server`` does, from the tools it lists.

    A server's hand-written description is all a query has to match, and
    nothing checks it against the server's tools: one registered as
    "Observability platform." does not match "why is my pod crashlooping"
    however good the embedder is. Writing the tools into that one description
    does not help either. Measured, a single vector over nineteen tools is too
    diffuse to match any one of them.

    So the catalogue is indexed in pieces beside the server's own entry, a
    handful of tools to a piece, and the router scores a server by the best of
    its own vector and its pieces'. Each tool contributes its name, as words,
    and the first sentence of its description. A server with too many tools
    for that falls back to names alone, which fit about four times as many.

    Args:
        server: The server the tools belong to.
        tools: Its tools, as discovery lists them.

    Returns:
        At most :data:`MAX_CATALOGUE_CHUNKS` pseudo-tools; none for a server
        with no tools.
    """

    def pack(lines: list[str]) -> list[str]:
        pieces: list[str] = []
        current = ""
        for line in lines:
            if current and len(current) + len(line) + 2 > CHUNK_CHARS:
                pieces.append(current)
                current = ""
            current = f"{current}; {line}" if current else line
        if current:
            pieces.append(current)
        return pieces

    names = [tool.name.replace("_", " ") for tool in tools]
    described = [
        f"{name}: {tool.description.split('. ')[0].rstrip('.')[:120]}"
        for name, tool in zip(names, tools)
    ]
    pieces = pack(described)
    if len(pieces) > MAX_CATALOGUE_CHUNKS:
        pieces = pack(names)
    digest = hashlib.sha256(f"{server.namespace}.{server.name}".encode()).hexdigest()[:16]
    return [
        ToolDefinition(
            name=f"mcp_tools_{digest}_{index:02d}",
            namespace=PSEUDO_NAMESPACE,
            description=f"{server.name} tools: {piece}"[:_DESCRIPTION_MAX],
            parameters_schema={"type": "object", "properties": {}},
            metadata={
                "entity_type": CATALOGUE_ENTITY,
                "server_name": server.name,
                "server_namespace": server.namespace,
                "chunk": index,
            },
        )
        for index, piece in enumerate(pieces[:MAX_CATALOGUE_CHUNKS])
    ]


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
        # What each server listed when last asked, keyed by (namespace, name).
        self._catalogues: dict[tuple[str, str], list[ToolDefinition]] = {}

    def set_catalogue(
        self, name: str, namespace: str, tools: Sequence[ToolDefinition] | None
    ) -> None:
        """Record the tools ``namespace.name`` lists, to be indexed at the next sync.

        ``None`` or an empty listing forgets them, and the next sync removes
        their index entries.
        """
        if tools:
            self._catalogues[(namespace, name)] = list(tools)
        else:
            self._catalogues.pop((namespace, name), None)

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
        :class:`~dotfetch.schema.mcp.MCPServerDefinition`.
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
            # A listing made under the previous endpoint may be another
            # server's by now.
            self._catalogues.pop((namespace, name), None)

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
        for server in all_servers:
            listed = self._catalogues.get((server.namespace, server.name))
            if listed:
                pseudo_tools.extend(catalogue_chunks(server, listed))
        to_sync = await self._sync_manager.detect_changes(pseudo_tools, force)
        self._registry.drain_pending(pending_snapshot)
        total_synced = 0
        if to_sync:
            logger.info(f"Syncing {len(to_sync)} MCP server entries...")
            for i in range(0, len(to_sync), batch_size):
                batch = to_sync[i : i + batch_size]
                # A pseudo-tool's description is the server's searchable text.
                embeddings = await self._embedder.embed_batch([tool.description for tool in batch])
                await self._vector_store.add_tools(batch, embeddings, upsert=True)
                # Servers, not rows: the pieces of a server's tool catalogue
                # are synced with it and are not servers themselves.
                total_synced += sum(
                    1 for tool in batch if tool.metadata.get("entity_type") == SERVER_ENTITY
                )
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
        other servers out. As with ``DotFetch.prune_stale_tools``, a store
        shared between instances is pruned to *this* fetcher's servers; a
        fetcher with no servers never gets here.
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
