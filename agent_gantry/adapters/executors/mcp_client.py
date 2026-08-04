"""
MCP Client adapter for Agent-Gantry.

Connects to MCP servers and converts their tools to ToolDefinition.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_gantry.schema.config import MCPServerConfig
from agent_gantry.schema.tool import ToolDefinition, ToolSource


class MCPClient:
    """
    Client for connecting to MCP servers.

    Handles:
    - Connection via stdio (subprocess)
    - MCP handshake (initialize/initialized)
    - Tool discovery (tools/list)
    - Tool execution (tools/call) over a persistent session
    - Conversion of MCP tools to ToolDefinition

    ``call_tool`` keeps one long-lived connection per client: spawning the
    server subprocess and re-running the initialize handshake per call adds
    hundreds of milliseconds (or seconds, for ``npx``-launched servers) to
    every tool execution. The connection is owned by a dedicated background
    task so the anyio cancel scopes of ``stdio_client``/``ClientSession`` are
    entered and exited in the same task — holding them open across arbitrary
    caller tasks is unsafe. Call :meth:`close` to shut the connection down.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        """
        Initialize MCP client.

        Args:
            config: Configuration for the MCP server to connect to
        """
        self.config = config
        self._session: ClientSession | None = None
        self._connected = False
        # Persistent-session machinery (see class docstring)
        self._owner_task: asyncio.Task[None] | None = None
        self._close_event: asyncio.Event | None = None
        self._connect_lock: asyncio.Lock | None = None
        self._loop_id: int | None = None

    @asynccontextmanager
    async def connect(self) -> Any:
        """
        Connect to the MCP server.

        Yields:
            ClientSession for interacting with the server
        """
        server_params = StdioServerParameters(
            command=self.config.command[0],
            args=self.config.command[1:] + self.config.args,
            env=self.config.env or None,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                self._session = session
                self._connected = True
                try:
                    yield session
                finally:
                    self._session = None
                    self._connected = False

    async def _ensure_session(self) -> ClientSession:
        """
        Return a live persistent session, connecting if needed.

        The connection is opened and later closed by a dedicated owner task
        (via :meth:`connect`), so the underlying anyio cancel scopes stay in
        one task. Other tasks may safely *use* the returned session.
        """
        loop_id = id(asyncio.get_running_loop())

        if self._connected and self._session is not None and self._loop_id == loop_id:
            return self._session

        if self._loop_id is not None and self._loop_id != loop_id:
            # Session belongs to a different (likely dead) event loop. Its
            # owner task can't be awaited from here — signal it to close and
            # abandon it, then reconnect on the current loop.
            #
            # Concurrency scope: this reset runs without awaits, so tasks on
            # a single loop can never interleave inside it. Simultaneous
            # first-use from two OS threads (two live loops sharing one
            # MCPClient) is not supported — same bound as the per-loop lock
            # pattern documented in core/rate_limiter.py.
            if self._close_event is not None:
                self._close_event.set()
            self._owner_task = None
            self._close_event = None
            self._connect_lock = None
            self._loop_id = None

        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()

        async with self._connect_lock:
            if self._connected and self._session is not None:
                return self._session

            ready = asyncio.Event()
            close_event = asyncio.Event()
            startup_error: list[BaseException] = []

            async def owner() -> None:
                try:
                    async with self.connect() as session:
                        # connect() sets _session/_connected for the real
                        # implementation; set them explicitly so patched/mock
                        # connect() implementations work too.
                        self._session = session
                        self._connected = True
                        ready.set()
                        await close_event.wait()
                except BaseException as e:
                    startup_error.append(e)
                    ready.set()
                finally:
                    self._session = None
                    self._connected = False

            self._close_event = close_event
            self._loop_id = loop_id
            self._owner_task = asyncio.create_task(owner())
            await ready.wait()

            if startup_error:
                self._owner_task = None
                self._close_event = None
                self._loop_id = None
                raise RuntimeError(
                    f"Failed to connect to MCP server '{self.config.name}': {startup_error[0]}"
                ) from startup_error[0]

            assert self._session is not None
            return self._session

    async def _invalidate_session(self) -> None:
        """Drop the persistent connection so the next call reconnects."""
        same_loop = self._loop_id is None or self._loop_id == id(asyncio.get_running_loop())
        event = self._close_event
        task = self._owner_task
        self._owner_task = None
        self._close_event = None
        self._loop_id = None
        if event is not None:
            if same_loop:
                event.set()
            else:
                # The event belongs to another (likely dead) loop. set() wakes
                # its waiters through that loop and can raise "Event loop is
                # closed" — signal best-effort, then abandon like
                # _ensure_session does.
                try:
                    event.set()
                except RuntimeError:
                    pass
        if task is not None and same_loop:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except Exception:
                logger.debug("Error while closing MCP session", exc_info=True)

    async def close(self) -> None:
        """Close the persistent connection (if any). Safe to call repeatedly."""
        await self._invalidate_session()

    async def list_tools(self) -> list[ToolDefinition]:
        """
        List all tools from the MCP server.

        Reuses the persistent session when one is live; otherwise opens a
        short-lived discovery connection.

        Returns:
            List of ToolDefinition objects
        """
        # Reuse the persistent session only when it belongs to the current
        # event loop — a session created on another loop cannot be used here.
        if (
            self._connected
            and self._session is not None
            and self._loop_id == id(asyncio.get_running_loop())
        ):
            session = self._session
            try:
                result = await session.list_tools()
            except Exception:
                await self._invalidate_session()
                raise
            return [self._convert_tool(tool) for tool in result.tools]

        async with self.connect() as session:
            result = await session.list_tools()
            tools = []
            for tool in result.tools:
                tool_def = self._convert_tool(tool)
                tools.append(tool_def)
            return tools

    def _convert_tool(self, mcp_tool: Any) -> ToolDefinition:
        """
        Convert MCP tool to ToolDefinition.

        Args:
            mcp_tool: MCP tool object

        Returns:
            ToolDefinition object
        """
        # Extract tool information
        name = mcp_tool.name
        description = mcp_tool.description or f"Tool: {name}"

        # Convert input schema to parameters_schema. mcp 2.x renamed the
        # attribute inputSchema -> input_schema (the old spelling remains a
        # construction alias only), so read both — checking just the 1.x name
        # would silently replace every v2 tool's schema with the empty default.
        parameters_schema = (
            getattr(mcp_tool, "input_schema", None)
            or getattr(mcp_tool, "inputSchema", None)
            or {"type": "object", "properties": {}, "required": []}
        )

        # Create ToolDefinition with MCP source
        return ToolDefinition(
            name=name,
            description=description,
            parameters_schema=parameters_schema,
            namespace=self.config.namespace,
            source=ToolSource.MCP_SERVER,
            source_uri=f"mcp://{self.config.name}",
            metadata={
                "mcp_server": self.config.name,
                "mcp_command": " ".join(self.config.command),
            },
        )

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool on the MCP server over the persistent session.

        The first call connects (subprocess spawn + initialize handshake);
        subsequent calls reuse the connection. On a transport error the
        session is dropped so the next call reconnects.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool

        Returns:
            Tool execution result
        """
        session = await self._ensure_session()
        try:
            result = await session.call_tool(tool_name, arguments)
        except Exception:
            await self._invalidate_session()
            raise
        # The MCP protocol reports tool failures in-band (isError on the
        # result) rather than as transport errors, so the call above resolves
        # normally. Surface them as exceptions so the execution engine records
        # a failure (retries, health, telemetry) instead of a success. The
        # session stays valid — this is a tool error, not a broken connection.
        # Attribute is spelled isError on mcp 1.x and is_error on 2.x.
        if getattr(result, "isError", False) or getattr(result, "is_error", False):
            raise RuntimeError(self._extract_error_text(result))
        return result

    @staticmethod
    def _extract_error_text(result: Any) -> str:
        """Pull a readable message out of an error CallToolResult."""
        texts = [
            text
            for item in getattr(result, "content", None) or []
            if isinstance(text := getattr(item, "text", None), str)
        ]
        return "; ".join(texts) if texts else f"MCP tool call failed: {result!r}"


class MCPClientPool:
    """
    Pool of MCP clients for managing multiple server connections.
    """

    def __init__(self) -> None:
        """Initialize the client pool."""
        self._clients: dict[str, MCPClient] = {}

    def add_server(self, config: MCPServerConfig) -> MCPClient:
        """
        Add an MCP server to the pool.

        Args:
            config: Configuration for the MCP server

        Returns:
            MCPClient instance
        """
        client = MCPClient(config)
        self._clients[config.name] = client
        return client

    def get_client(self, name: str) -> MCPClient | None:
        """
        Get an MCP client by name.

        Args:
            name: Server name

        Returns:
            MCPClient instance or None
        """
        return self._clients.get(name)

    async def list_all_tools(self) -> list[ToolDefinition]:
        """
        List tools from all connected servers.

        Returns:
            List of all ToolDefinition objects from all servers
        """
        all_tools = []
        for client in self._clients.values():
            try:
                tools = await client.list_tools()
                all_tools.extend(tools)
            except Exception as e:
                # Log error but continue with other servers
                logger.error(f"Error listing tools from {client.config.name}: {e}")
        return all_tools

    def remove_server(self, name: str) -> bool:
        """
        Remove an MCP server from the pool.

        Best-effort closes the client's persistent connection (scheduled on
        the running loop when there is one).

        Args:
            name: Server name

        Returns:
            True if server was removed
        """
        client = self._clients.pop(name, None)
        if client is None:
            return False
        _schedule_client_close(client)
        return True

    async def close_all(self) -> None:
        """Close all pooled clients' persistent connections."""
        for client in list(self._clients.values()):
            try:
                await client.close()
            except Exception:
                logger.debug("Error closing MCP client", exc_info=True)


def _schedule_client_close(client: MCPClient) -> None:
    """Schedule ``client.close()`` on the running loop, if any."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(client.close())
    # Keep a reference until done so the task isn't garbage-collected early.
    task.add_done_callback(lambda _t: None)
