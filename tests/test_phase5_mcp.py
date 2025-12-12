"""
Tests for Phase 5: MCP Integration.

Tests MCP client, server, and protocol compliance.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.executors.mcp_client import MCPClient, MCPClientPool
from agent_gantry.schema.config import MCPServerConfig
from agent_gantry.schema.tool import ToolDefinition, ToolSource
from agent_gantry.servers.mcp_server import MCPServer, create_mcp_server


class TestMCPClient:
    """Tests for MCP client functionality."""

    @pytest.fixture
    def mcp_config(self) -> MCPServerConfig:
        """Create a sample MCP server config."""
        return MCPServerConfig(
            name="test-server",
            command=["python", "-m", "test_mcp_server"],
            args=["--port", "8080"],
            env={"API_KEY": "test-key"},
            namespace="mcp_test",
        )

    @pytest.fixture
    def mcp_client(self, mcp_config: MCPServerConfig) -> MCPClient:
        """Create an MCP client instance."""
        return MCPClient(mcp_config)

    def test_client_initialization(
        self, mcp_client: MCPClient, mcp_config: MCPServerConfig
    ) -> None:
        """Test MCP client initialization."""
        assert mcp_client.config == mcp_config
        assert mcp_client._session is None
        assert mcp_client._connected is False

    @pytest.mark.asyncio
    async def test_convert_tool(
        self, mcp_client: MCPClient, mcp_config: MCPServerConfig
    ) -> None:
        """Test converting MCP tool to ToolDefinition."""
        # Mock MCP tool
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
            },
            "required": ["param1"],
        }

        tool_def = mcp_client._convert_tool(mock_tool)

        assert isinstance(tool_def, ToolDefinition)
        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"
        assert tool_def.namespace == mcp_config.namespace
        assert tool_def.source == ToolSource.MCP_SERVER
        assert tool_def.source_uri == f"mcp://{mcp_config.name}"
        assert tool_def.metadata["mcp_server"] == mcp_config.name
        assert "test_mcp_server" in tool_def.metadata["mcp_command"]

    @pytest.mark.asyncio
    async def test_list_tools_with_mock(self, mcp_client: MCPClient) -> None:
        """Test listing tools from MCP server with mocked connection."""
        # Mock the connection and session
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "This is the first test tool for MCP"
        mock_tool1.inputSchema = {"type": "object", "properties": {}}

        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "This is the second test tool for MCP"
        mock_tool2.inputSchema = {"type": "object", "properties": {}}

        mock_result = MagicMock()
        mock_result.tools = [mock_tool1, mock_tool2]

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_result)
        mock_session.initialize = AsyncMock()

        # Create a proper async context manager mock
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_connect():
            yield mock_session

        # Patch the connect method
        with patch.object(mcp_client, "connect", side_effect=mock_connect):
            tools = await mcp_client.list_tools()

            assert len(tools) == 2
            assert tools[0].name == "tool1"
            assert tools[1].name == "tool2"
            assert all(isinstance(t, ToolDefinition) for t in tools)
            assert all(t.source == ToolSource.MCP_SERVER for t in tools)


class TestMCPClientPool:
    """Tests for MCP client pool."""

    @pytest.fixture
    def pool(self) -> MCPClientPool:
        """Create an MCP client pool."""
        return MCPClientPool()

    @pytest.fixture
    def config1(self) -> MCPServerConfig:
        """Create first server config."""
        return MCPServerConfig(
            name="server1",
            command=["python", "-m", "server1"],
            namespace="ns1",
        )

    @pytest.fixture
    def config2(self) -> MCPServerConfig:
        """Create second server config."""
        return MCPServerConfig(
            name="server2",
            command=["python", "-m", "server2"],
            namespace="ns2",
        )

    def test_add_server(
        self, pool: MCPClientPool, config1: MCPServerConfig
    ) -> None:
        """Test adding server to pool."""
        client = pool.add_server(config1)
        assert isinstance(client, MCPClient)
        assert client.config == config1
        assert pool.get_client("server1") == client

    def test_get_client(
        self, pool: MCPClientPool, config1: MCPServerConfig
    ) -> None:
        """Test getting client from pool."""
        pool.add_server(config1)
        client = pool.get_client("server1")
        assert client is not None
        assert client.config.name == "server1"

        # Non-existent server
        assert pool.get_client("nonexistent") is None

    def test_remove_server(
        self, pool: MCPClientPool, config1: MCPServerConfig
    ) -> None:
        """Test removing server from pool."""
        pool.add_server(config1)
        assert pool.remove_server("server1") is True
        assert pool.get_client("server1") is None
        assert pool.remove_server("server1") is False

    @pytest.mark.asyncio
    async def test_list_all_tools(
        self,
        pool: MCPClientPool,
        config1: MCPServerConfig,
        config2: MCPServerConfig,
    ) -> None:
        """Test listing tools from all servers."""
        client1 = pool.add_server(config1)
        client2 = pool.add_server(config2)

        # Mock list_tools for both clients
        mock_tools1 = [
            ToolDefinition(
                name="tool1",
                description="First tool from server one for testing",
                parameters_schema={"type": "object"},
            )
        ]
        mock_tools2 = [
            ToolDefinition(
                name="tool2",
                description="Second tool from server two for testing",
                parameters_schema={"type": "object"},
            )
        ]

        client1.list_tools = AsyncMock(return_value=mock_tools1)
        client2.list_tools = AsyncMock(return_value=mock_tools2)

        all_tools = await pool.list_all_tools()
        assert len(all_tools) == 2
        assert all_tools[0].name == "tool1"
        assert all_tools[1].name == "tool2"


class TestMCPServer:
    """Tests for MCP server functionality."""

    @pytest.fixture
    async def gantry(self) -> AgentGantry:
        """Create a gantry instance with sample tools."""
        gantry = AgentGantry()

        @gantry.register
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        @gantry.register
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny, 72°F"

        await gantry.sync()
        return gantry

    @pytest.fixture
    def mcp_server_dynamic(self, gantry: AgentGantry) -> MCPServer:
        """Create MCP server in dynamic mode."""
        return create_mcp_server(gantry, mode="dynamic", name="test-server")

    @pytest.fixture
    def mcp_server_static(self, gantry: AgentGantry) -> MCPServer:
        """Create MCP server in static mode."""
        return create_mcp_server(gantry, mode="static", name="test-server")

    def test_server_initialization(
        self, mcp_server_dynamic: MCPServer, gantry: AgentGantry
    ) -> None:
        """Test MCP server initialization."""
        assert mcp_server_dynamic.gantry == gantry
        assert mcp_server_dynamic.mode == "dynamic"
        assert mcp_server_dynamic.name == "test-server"
        assert mcp_server_dynamic.server is not None

    @pytest.mark.asyncio
    async def test_dynamic_mode_tools(
        self, mcp_server_dynamic: MCPServer
    ) -> None:
        """Test that dynamic mode exposes meta-tools."""
        # Dynamic mode should provide meta-tools, not direct tools
        # We verify this by checking the _handle methods exist
        assert hasattr(mcp_server_dynamic, "_handle_find_relevant_tools")
        assert hasattr(mcp_server_dynamic, "_handle_execute_tool")

        # Verify the server is in dynamic mode
        assert mcp_server_dynamic.mode == "dynamic"
        
        # Verify tools are still accessible through gantry
        tools = await mcp_server_dynamic.gantry.list_tools()
        assert len(tools) > 0

    @pytest.mark.asyncio
    async def test_static_mode_tools(
        self, mcp_server_static: MCPServer
    ) -> None:
        """Test that static mode exposes all tools."""
        # In static mode, tools should be accessible through the gantry
        tools = await mcp_server_static.gantry.list_tools()
        assert len(tools) >= 2  # At least our registered tools

        tool_names = [t.name for t in tools]
        assert "add_numbers" in tool_names
        assert "get_weather" in tool_names

        # Verify the server is in static mode
        assert mcp_server_static.mode == "static"

    @pytest.mark.asyncio
    async def test_find_relevant_tools(
        self, mcp_server_dynamic: MCPServer
    ) -> None:
        """Test find_relevant_tools meta-tool."""
        result = await mcp_server_dynamic._handle_find_relevant_tools(
            {"query": "add two numbers", "limit": 5}
        )

        assert isinstance(result, list)
        assert len(result) > 0

        # Check that result contains tool information
        first_result = result[0]
        assert first_result["type"] == "text"
        assert "Tool:" in first_result["text"]
        assert "Description:" in first_result["text"]
        assert "Parameters:" in first_result["text"]

    @pytest.mark.asyncio
    async def test_execute_tool(self, mcp_server_dynamic: MCPServer) -> None:
        """Test execute_tool meta-tool."""
        result = await mcp_server_dynamic._handle_execute_tool(
            {"tool_name": "add_numbers", "arguments": {"a": 5, "b": 3}}
        )

        assert isinstance(result, list)
        assert len(result) > 0

        first_result = result[0]
        assert first_result["type"] == "text"
        assert "8" in first_result["text"]

    @pytest.mark.asyncio
    async def test_execute_tool_error(
        self, mcp_server_dynamic: MCPServer
    ) -> None:
        """Test execute_tool with non-existent tool."""
        result = await mcp_server_dynamic._handle_execute_tool(
            {"tool_name": "nonexistent_tool", "arguments": {}}
        )

        assert isinstance(result, list)
        assert len(result) > 0

        first_result = result[0]
        assert first_result["type"] == "text"
        assert "Error" in first_result["text"]


class TestAgentGantryMCPIntegration:
    """Tests for MCP integration in AgentGantry."""

    @pytest.fixture
    async def gantry(self) -> AgentGantry:
        """Create a gantry instance."""
        gantry = AgentGantry()

        @gantry.register
        def sample_tool(x: int) -> int:
            """A sample tool."""
            return x * 2

        await gantry.sync()
        return gantry

    @pytest.mark.asyncio
    async def test_add_mcp_server(self, gantry: AgentGantry) -> None:
        """Test adding an MCP server to AgentGantry."""
        config = MCPServerConfig(
            name="test-server",
            command=["python", "-m", "test_mcp_server"],
            namespace="mcp_test",
        )

        # Mock the MCPClient to avoid actual connection
        with patch(
            "agent_gantry.adapters.executors.mcp_client.MCPClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_tools = [
                ToolDefinition(
                    name="external_tool",
                    description="External tool from MCP",
                    parameters_schema={"type": "object"},
                )
            ]
            mock_client.list_tools = AsyncMock(return_value=mock_tools)
            mock_client_class.return_value = mock_client

            count = await gantry.add_mcp_server(config)

            assert count == 1
            # Verify the tool was added
            tool = await gantry.get_tool("external_tool")
            assert tool is not None
            assert tool.name == "external_tool"

    @pytest.mark.asyncio
    async def test_serve_mcp_dynamic(self, gantry: AgentGantry) -> None:
        """Test serving as MCP server in dynamic mode."""
        # Mock the server to avoid actually starting it
        with patch(
            "agent_gantry.servers.mcp_server.create_mcp_server"
        ) as mock_create_server:
            mock_server = AsyncMock()
            mock_server.run_stdio = AsyncMock()
            mock_create_server.return_value = mock_server

            # This would normally block, so we'll just verify it's called correctly
            await gantry.serve_mcp(transport="stdio", mode="dynamic")

            mock_create_server.assert_called_once_with(
                gantry, mode="dynamic", name="agent-gantry"
            )
            mock_server.run_stdio.assert_called_once()

    @pytest.mark.asyncio
    async def test_serve_mcp_invalid_transport(
        self, gantry: AgentGantry
    ) -> None:
        """Test that invalid transport raises error."""
        with pytest.raises(ValueError, match="Unsupported transport"):
            await gantry.serve_mcp(transport="invalid")


class TestMCPProtocolCompliance:
    """Tests for MCP protocol compliance."""

    @pytest.mark.asyncio
    async def test_tool_schema_format(self) -> None:
        """Test that tool schemas comply with MCP format."""
        gantry = AgentGantry()

        @gantry.register
        def test_tool(param1: str, param2: int) -> str:
            """Test tool with parameters for MCP schema validation."""
            return f"{param1}: {param2}"

        await gantry.sync()

        # Get the tool from gantry
        tool = await gantry.get_tool("test_tool")
        assert tool is not None

        # Verify the tool has proper MCP-compatible schema
        assert tool.name == "test_tool"
        assert len(tool.description) >= 10  # Meets minimum length
        assert tool.parameters_schema is not None

        # Verify schema structure
        schema = tool.parameters_schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "param1" in schema["properties"]
        assert "param2" in schema["properties"]

        # Test conversion to MCP Tool format
        server = create_mcp_server(gantry, mode="static")
        mcp_tool = server._convert_tool(tool)
        assert mcp_tool.name == "test_tool"
        assert hasattr(mcp_tool, "description")
        assert hasattr(mcp_tool, "inputSchema")

    @pytest.mark.asyncio
    async def test_meta_tool_discovery_flow(self) -> None:
        """Test the complete meta-tool discovery and execution flow."""
        gantry = AgentGantry()

        @gantry.register
        def calculate_sum(a: int, b: int) -> int:
            """Calculate the sum of two numbers."""
            return a + b

        await gantry.sync()

        server = create_mcp_server(gantry, mode="dynamic")

        # Step 1: Discover tools using find_relevant_tools
        discovery_result = await server._handle_find_relevant_tools(
            {"query": "calculate sum of numbers", "limit": 5}
        )

        assert len(discovery_result) > 0
        result_text = discovery_result[0]["text"]
        assert "calculate_sum" in result_text

        # Step 2: Execute the discovered tool
        execution_result = await server._handle_execute_tool(
            {"tool_name": "calculate_sum", "arguments": {"a": 10, "b": 20}}
        )

        assert len(execution_result) > 0
        result_text = execution_result[0]["text"]
        assert "30" in result_text

    @pytest.mark.asyncio
    async def test_context_window_minimization(self) -> None:
        """Test that dynamic mode minimizes context window usage."""
        gantry = AgentGantry()

        # Register many tools
        for i in range(20):
            # Use closure to capture the loop variable correctly
            def make_tool(idx):
                @gantry.register(name=f"tool_{idx}")
                def tool_fn(x: int) -> int:
                    """Tool for testing context window minimization with many tools."""
                    return x + idx
                return tool_fn

            make_tool(i)

        await gantry.sync()

        # Verify all tools were registered
        all_tools = await gantry.list_tools()
        assert len(all_tools) >= 20

        # Dynamic mode exposes meta-tools for discovery
        dynamic_server = create_mcp_server(gantry, mode="dynamic")
        assert dynamic_server.mode == "dynamic"

        # In dynamic mode, clients would first call find_relevant_tools
        # to discover a small subset, minimizing context window usage
        result = await dynamic_server._handle_find_relevant_tools(
            {"query": "tool for number 5", "limit": 3}
        )
        # Should return a small subset, not all 20+ tools
        assert len(result) <= 3

        # Static mode would expose all tools directly
        static_server = create_mcp_server(gantry, mode="static")
        static_tools = await static_server.gantry.list_tools()
        assert len(static_tools) >= 20  # All registered tools
