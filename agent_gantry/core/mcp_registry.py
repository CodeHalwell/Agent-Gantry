"""
MCP Server registry for Agent-Gantry.

Manages MCP server registration, health tracking, and lifecycle.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from agent_gantry.adapters.executors.mcp_client import MCPClient
from agent_gantry.schema.mcp import MCPServerDefinition

logger = logging.getLogger(__name__)


class MCPRegistry:
    """
    Registry for managing MCP server definitions and clients.

    Handles:
    - MCP server registration and metadata
    - Client lifecycle management
    - Server health tracking
    - Lazy client instantiation
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._servers: dict[str, MCPServerDefinition] = {}
        self._clients: dict[str, MCPClient] = {}
        # Clients dropped from the cache that could not be closed at the time,
        # because no event loop was running to schedule the close on. Held so
        # ``close_all_clients`` can still reach them; otherwise the HTTP
        # connection or stdio subprocess would survive to process teardown.
        self._retired: list[MCPClient] = []
        self._pending: list[MCPServerDefinition] = []

    def register_server(
        self,
        server: MCPServerDefinition,
    ) -> None:
        """
        Register an MCP server definition.

        Args:
            server: The server definition to register
        """
        key = server.qualified_name
        self._servers[key] = server
        logger.debug(f"Registered MCP server: {key}")

    def add_pending(self, server: MCPServerDefinition) -> None:
        """
        Add a server to the pending list.

        Args:
            server: The server definition to add
        """
        self._pending.append(server)

    def get_server(self, name: str, namespace: str = "default") -> MCPServerDefinition | None:
        """
        Get a server by name and namespace.

        Args:
            name: Server name
            namespace: Server namespace

        Returns:
            The server definition if found
        """
        key = f"{namespace}.{name}"
        return self._servers.get(key)

    def get_client(self, name: str, namespace: str = "default") -> MCPClient | None:
        """
        Get or create an MCP client for a server.

        Lazy instantiation - client is created on first access.

        Args:
            name: Server name
            namespace: Server namespace

        Returns:
            MCPClient instance or None if server not found
        """
        key = f"{namespace}.{name}"
        server = self._servers.get(key)

        if not server:
            return None

        # Return existing client if available
        if key in self._clients:
            return self._clients[key]

        # Create new client from server definition
        from agent_gantry.schema.config import MCPServerConfig

        config = MCPServerConfig(**server.to_config())
        client = MCPClient(config)
        self._clients[key] = client
        logger.debug(f"Created MCP client for: {key}")

        return client

    def forget_client(self, name: str, namespace: str = "default") -> bool:
        """Drop a cached client, closing its connection, keeping the server.

        Used when a server is re-registered with a different endpoint: the
        cached client still points at the old one.

        Returns:
            True if a cached client was dropped.
        """
        client = self._clients.pop(f"{namespace}.{name}", None)
        if client is None:
            return False
        from agent_gantry.adapters.executors.mcp_client import _schedule_client_close

        if not _schedule_client_close(client):
            self._retired.append(client)
        logger.debug(f"Dropped cached MCP client for: {namespace}.{name}")
        return True

    def list_servers(self, namespace: str | None = None) -> list[MCPServerDefinition]:
        """
        List all registered servers.

        Args:
            namespace: Filter by namespace

        Returns:
            List of server definitions
        """
        servers = list(self._servers.values())
        if namespace:
            servers = [s for s in servers if s.namespace == namespace]
        return servers

    def delete_server(self, name: str, namespace: str = "default") -> bool:
        """
        Delete a server from the registry.

        Args:
            name: Server name
            namespace: Server namespace

        Returns:
            True if the server was deleted
        """
        key = f"{namespace}.{name}"
        if key in self._servers:
            del self._servers[key]
            # Also remove client if it exists (best-effort close of its
            # persistent connection)
            client = self._clients.pop(key, None)
            if client is not None:
                from agent_gantry.adapters.executors.mcp_client import _schedule_client_close

                if not _schedule_client_close(client):
                    self._retired.append(client)
            logger.debug(f"Deleted MCP server: {key}")
            return True
        return False

    async def close_all_clients(self) -> None:
        """Close all cached clients' persistent connections.

        Closes run concurrently so shutdown is bounded by the slowest single
        client (each close can wait up to 5s on a stuck owner task), not the
        sum across servers.
        """
        import asyncio

        # Retired clients included: one dropped outside a running loop had no
        # way to be closed then, and is unreachable through the cache now.
        clients = list(self._clients.values()) + self._retired
        results = await asyncio.gather(
            *(client.close() for client in clients), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                logger.debug("Error closing MCP client", exc_info=result)
        self._clients.clear()
        self._retired.clear()

    def get_pending(self) -> list[MCPServerDefinition]:
        """
        Get servers pending sync to vector store.

        Returns:
            List of pending server definitions
        """
        return self._pending.copy()

    def clear_pending(self) -> None:
        """Clear the pending servers list."""
        self._pending = []

    def drain_pending(self, covered: Sequence[MCPServerDefinition]) -> None:
        """Drop only the pending entries ``covered`` accounts for.

        A blanket :meth:`clear_pending` discarded a ``register_mcp_server()``
        that landed while a sync was awaiting, so the server was never
        embedded and ``_synced`` stayed ``True`` — the same late-registration
        loss the tool-side buffer had. Identity, not equality: two
        registrations of the same definition are distinct buffer entries.
        """
        done = {id(server) for server in covered}
        self._pending = [server for server in self._pending if id(server) not in done]

    def update_health(
        self,
        name: str,
        namespace: str = "default",
        **health_updates: Any,
    ) -> bool:
        """
        Update health metrics for a server.

        Args:
            name: Server name
            namespace: Server namespace
            **health_updates: Health fields to update

        Returns:
            True if server was found and updated
        """
        server = self.get_server(name, namespace)
        if not server:
            return False

        for key, value in health_updates.items():
            if hasattr(server.health, key):
                setattr(server.health, key, value)

        return True

    @property
    def server_count(self) -> int:
        """Return the number of registered servers."""
        return len(self._servers)

    @property
    def active_client_count(self) -> int:
        """Return the number of active clients."""
        return len(self._clients)
