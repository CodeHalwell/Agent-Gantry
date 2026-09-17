"""
Tests for Phase 5: MCP Integration.

Tests MCP client, server, and protocol compliance.
"""

from __future__ import annotations

import asyncio
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
    async def test_convert_tool(self, mcp_client: MCPClient, mcp_config: MCPServerConfig) -> None:
        """Test converting MCP tool to ToolDefinition."""
        # Mock MCP tool
        # spec-limited mock: a real mcp 1.x Tool has no `input_schema`
        # attribute (that's the 2.x spelling), and an unspecced MagicMock
        # would auto-create a truthy mock for it.
        mock_tool = MagicMock(spec=["name", "description", "inputSchema"])
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
        mock_tool1 = MagicMock(spec=["name", "description", "inputSchema"])
        mock_tool1.name = "tool1"
        mock_tool1.description = "This is the first test tool for MCP"
        mock_tool1.inputSchema = {"type": "object", "properties": {}}

        mock_tool2 = MagicMock(spec=["name", "description", "inputSchema"])
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

    def test_add_server(self, pool: MCPClientPool, config1: MCPServerConfig) -> None:
        """Test adding server to pool."""
        client = pool.add_server(config1)
        assert isinstance(client, MCPClient)
        assert client.config == config1
        assert pool.get_client("server1") == client

    def test_get_client(self, pool: MCPClientPool, config1: MCPServerConfig) -> None:
        """Test getting client from pool."""
        pool.add_server(config1)
        client = pool.get_client("server1")
        assert client is not None
        assert client.config.name == "server1"

        # Non-existent server
        assert pool.get_client("nonexistent") is None

    def test_remove_server(self, pool: MCPClientPool, config1: MCPServerConfig) -> None:
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
    async def test_dynamic_mode_tools(self, mcp_server_dynamic: MCPServer) -> None:
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
    async def test_static_mode_tools(self, mcp_server_static: MCPServer) -> None:
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
    async def test_find_relevant_tools(self, mcp_server_dynamic: MCPServer) -> None:
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
    async def test_execute_tool_error(self, mcp_server_dynamic: MCPServer) -> None:
        """Failed executions raise so the MCP layer marks the result isError.

        Returning error text instead would make MCP clients record the
        failure as a successful call.
        """
        with pytest.raises(RuntimeError, match="Error"):
            await mcp_server_dynamic._handle_execute_tool(
                {"tool_name": "nonexistent_tool", "arguments": {}}
            )


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
        with patch("agent_gantry.adapters.executors.mcp_client.MCPClient") as mock_client_class:
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
        with patch("agent_gantry.servers.mcp_server.create_mcp_server") as mock_create_server:
            mock_server = AsyncMock()
            mock_server.run_stdio = AsyncMock()
            mock_create_server.return_value = mock_server

            # This would normally block, so we'll just verify it's called correctly
            await gantry.serve_mcp(transport="stdio", mode="dynamic")

            mock_create_server.assert_called_once_with(
                gantry, mode="dynamic", name="agent-gantry", expose=None
            )
            mock_server.run_stdio.assert_called_once()

    @pytest.mark.asyncio
    async def test_serve_mcp_http_transports(self, gantry: AgentGantry) -> None:
        """The HTTP transports route to the server's runners with host/port/path."""
        with patch("agent_gantry.servers.mcp_server.create_mcp_server") as mock_create_server:
            mock_server = AsyncMock()
            mock_create_server.return_value = mock_server

            await gantry.serve_mcp(
                transport="http", mode="hybrid", expose=["add_numbers"], port=9999, path="/x"
            )
            mock_create_server.assert_called_once_with(
                gantry, mode="hybrid", name="agent-gantry", expose=["add_numbers"]
            )
            mock_server.run_http.assert_awaited_once_with(host="127.0.0.1", port=9999, path="/x")

            await gantry.serve_mcp(transport="sse", host="0.0.0.0", allowed_hosts=["a:1"])
            mock_server.run_sse.assert_awaited_once_with(
                host="0.0.0.0", port=8000, allowed_hosts=["a:1"]
            )

    @pytest.mark.asyncio
    async def test_serve_mcp_invalid_transport(self, gantry: AgentGantry) -> None:
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
        # mcp 1.x exposes inputSchema; 2.x renamed the attribute input_schema
        assert hasattr(mcp_tool, "inputSchema") or hasattr(mcp_tool, "input_schema")

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


@pytest.mark.asyncio
async def test_a_client_dropped_outside_a_loop_is_still_closed_at_shutdown() -> None:
    """``forget_client`` schedules the close on the running loop, but from a
    synchronous thread there is none, so the close was skipped and the client
    was gone from the cache -- leaving its HTTP connection or stdio
    subprocess alive with nothing able to reach it."""
    from agent_gantry.core.mcp_registry import MCPRegistry
    from agent_gantry.schema.mcp import MCPServerDefinition

    class _FakeClient:
        def __init__(self, name: str) -> None:
            self.config = type("C", (), {"name": name})()
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    registry = MCPRegistry()
    registry.register_server(
        MCPServerDefinition(
            name="srv",
            description="A stand-in server for the client-retirement check.",
            command=["echo", "hi"],
            namespace="default",
        )
    )
    client = _FakeClient("srv")
    registry._clients["default.srv"] = client

    # Dropped from a thread with no running loop, as a synchronous
    # re-registration would.
    await asyncio.to_thread(registry.forget_client, "srv")
    assert client.closed is False, "nothing can close it from there"
    assert client not in registry._clients.values(), "and it leaves the cache"

    # ...but shutdown still reaches it
    await registry.close_all_clients()
    assert client.closed is True
    assert registry._retired == []


@pytest.mark.asyncio
async def test_a_client_added_during_shutdown_is_not_silently_dropped() -> None:
    """``close_all_clients`` cleared its bookkeeping *after* the gather, so a
    client cached while that await was in flight was neither closed by it nor
    reachable through the cache afterwards. ``MCPClientPool.close_all`` clears
    first for this reason; the registry twin cleared last."""
    from agent_gantry.core.mcp_registry import MCPRegistry

    registry = MCPRegistry()

    class _SlowClient:
        def __init__(self, name: str) -> None:
            self.config = type("C", (), {"name": name})()
            self.closed = False

        async def close(self) -> None:
            await asyncio.sleep(0.05)
            self.closed = True

    first = _SlowClient("first")
    registry._clients["default.first"] = first

    async def _register_during_shutdown() -> _SlowClient:
        await asyncio.sleep(0.01)  # while the gather above is in flight
        late = _SlowClient("late")
        registry._clients["default.late"] = late
        return late

    latecomer = asyncio.create_task(_register_during_shutdown())
    await registry.close_all_clients()
    late = await latecomer

    assert first.closed is True
    # The latecomer was not part of that batch, but it must still be reachable
    # so a later shutdown can close it -- not wiped by this one.
    assert late in registry._clients.values() or late.closed
    await registry.close_all_clients()
    assert late.closed is True


@pytest.mark.asyncio
async def test_a_server_registered_mid_sync_is_not_dropped_from_the_buffer() -> None:
    """``sync_servers`` cleared the whole pending buffer, so a
    ``register_mcp_server()`` landing while it awaited was discarded: the
    server was never embedded and ``_synced`` stayed True, so nothing tried
    again. The tool-side buffer was fixed for exactly this; the MCP path kept
    the blanket clear."""
    from agent_gantry.core.mcp_registry import MCPRegistry
    from agent_gantry.schema.mcp import MCPServerDefinition

    def _definition(name: str) -> MCPServerDefinition:
        return MCPServerDefinition(
            name=name,
            description=f"Server {name} for the mid-sync registration check.",
            command=["echo", name],
            namespace="default",
        )

    registry = MCPRegistry()
    first = _definition("early")
    registry.register_server(first)
    registry.add_pending(first)

    # What a sync in flight would have snapshotted...
    snapshot = registry.get_pending()

    # ...and a registration that lands while it is awaiting.
    late = _definition("late")
    registry.register_server(late)
    registry.add_pending(late)

    registry.drain_pending(snapshot)
    remaining = [server.name for server in registry.get_pending()]
    assert remaining == ["late"], remaining


@pytest.mark.asyncio
async def test_a_pooled_client_dropped_outside_a_loop_is_still_closed() -> None:
    """The same lifecycle hole as above, on ``MCPClientPool``: it dropped the
    client whether or not the close could be scheduled, so a ``remove_server``
    from a synchronous thread stranded a live connection that ``close_all``
    could never reach."""

    class _FakeClient:
        def __init__(self, name: str) -> None:
            self.config = type("C", (), {"name": name})()
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    pool = MCPClientPool()
    client = _FakeClient("srv")
    pool._clients["srv"] = client

    assert await asyncio.to_thread(pool.remove_server, "srv") is True
    assert client.closed is False, "nothing can close it from there"
    assert client not in pool._clients.values(), "and it leaves the pool"

    await pool.close_all()
    assert client.closed is True
    assert pool._retired == []


@pytest.mark.asyncio
async def test_an_empty_discovery_is_authoritative() -> None:
    """An empty ``tools/list`` is the server's complete catalogue, so it
    removes the tools it had. Refusing to prune on empty looks safer but has
    no way out: every later empty answer takes the same branch, so the tools
    stay retrievable forever and every execution is dispatched to a server
    that no longer exposes them. Discovery *failures* raise instead."""

    class _FakeClient:
        def __init__(self, config: MCPServerConfig) -> None:
            self.config = config

    config = MCPServerConfig(name="files", command=["fake"], namespace="mcp")
    client = _FakeClient(config)
    tools = [
        ToolDefinition(
            name=f"read_file_{i}",
            namespace="mcp",
            description=f"Read a file from the MCP server, variant {i}.",
            parameters_schema={"type": "object", "properties": {}},
            metadata={"mcp_server": "files"},
        )
        for i in range(3)
    ]

    gantry = AgentGantry()
    try:
        gantry._register_mcp_tool_handlers(client, tools, resolve_client=lambda: client)
        gantry._pending_tools.extend(tools)
        await gantry.sync()
        assert len(gantry._registry.list_tools("mcp")) == 3

        # A server that still lists some tools prunes only the rest.
        await gantry._remove_stale_mcp_tools(client, tools[:1])
        assert [t.name for t in gantry._registry.list_tools("mcp")] == ["read_file_0"]

        # ...and one that lists none prunes them all, from the store too.
        await gantry._remove_stale_mcp_tools(client, [])
        assert gantry._registry.list_tools("mcp") == []
        assert await gantry._vector_store.list_all(namespace="mcp") == []
    finally:
        await gantry.close()
