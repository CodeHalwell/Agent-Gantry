"""
MCP Client adapter for Agent-Gantry.

Connects to MCP servers and converts their tools to ToolDefinition.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters  # type: ignore[import-not-found]
from mcp.client.stdio import stdio_client  # type: ignore[import-not-found]

from agent_gantry.schema.config import MCPServerConfig
from agent_gantry.schema.tool import ToolDefinition, ToolSource


class MCPClient:
    """
    Client for connecting to MCP servers.

    Handles:
    - Connection via stdio (subprocess)
    - MCP handshake (initialize/initialized)
    - Tool discovery (tools/list)
    - Tool execution (tools/call)
    - Conversion of MCP tools to ToolDefinition
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

    async def list_tools(self) -> list[ToolDefinition]:
        """
        List all tools from the MCP server.

        Returns:
            List of ToolDefinition objects
        """
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

        # Convert input schema to parameters_schema
        parameters_schema = getattr(mcp_tool, 'inputSchema', {
            "type": "object",
            "properties": {},
            "required": []
        })

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
        Execute a tool on the MCP server.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool

        Returns:
            Tool execution result
        """
        async with self.connect() as session:
            result = await session.call_tool(tool_name, arguments)
            return result


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
                print(f"Error listing tools from {client.config.name}: {e}")
        return all_tools

    def remove_server(self, name: str) -> bool:
        """
        Remove an MCP server from the pool.

        Args:
            name: Server name

        Returns:
            True if server was removed
        """
        if name in self._clients:
            del self._clients[name]
            return True
        return False
