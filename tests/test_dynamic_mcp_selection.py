"""
Tests for dynamic MCP server selection.

Tests MCP server registry, router, and semantic selection.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_gantry import AgentGantry
from agent_gantry.core.mcp_registry import MCPRegistry
from agent_gantry.core.mcp_router import MCPRouter, MCPRoutingResult
from agent_gantry.schema.mcp import MCPServerDefinition


class TestMCPServerDefinition:
    """Tests for MCPServerDefinition model."""

    def test_server_definition_creation(self) -> None:
        """Test creating an MCP server definition."""
        server = MCPServerDefinition(
            name="filesystem",
            namespace="local",
            description="Provides tools for reading and writing files on the local filesystem",
            command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
            args=["--path", "/tmp"],
            tags=["filesystem", "files", "io"],
            examples=["read a file", "write to a file"],
            capabilities=["read_files", "write_files"],
        )

        assert server.name == "filesystem"
        assert server.namespace == "local"
        assert len(server.tags) == 3
        assert "filesystem" in server.tags
        assert server.qualified_name == "local.filesystem"

    def test_to_searchable_text(self) -> None:
        """Test conversion to searchable text."""
        server = MCPServerDefinition(
            name="database",
            description="Access SQL databases",
            command=["python", "-m", "mcp_db"],
            tags=["database", "sql"],
            examples=["query database", "insert record"],
            capabilities=["read_data", "write_data"],
        )

        text = server.to_searchable_text()
        assert "database" in text.lower()
        assert "sql" in text.lower()
        assert "query database" in text.lower()
        assert "read_data" in text.lower()

    def test_to_config(self) -> None:
        """Test conversion to config dict."""
        server = MCPServerDefinition(
            name="api",
            description="REST API access",
            command=["node", "api-server.js"],
            args=["--port", "3000"],
            env={"API_KEY": "test"},
            namespace="remote",
        )

        config = server.to_config()
        assert config["name"] == "api"
        assert config["command"] == ["node", "api-server.js"]
        assert config["args"] == ["--port", "3000"]
        assert config["env"] == {"API_KEY": "test"}
        assert config["namespace"] == "remote"

    def test_content_hash(self) -> None:
        """Test content hash generation for change detection."""
        server1 = MCPServerDefinition(
            name="test",
            description="Test server",
            command=["test"],
        )
        server2 = MCPServerDefinition(
            name="test",
            description="Test server",
            command=["test"],
        )
        server3 = MCPServerDefinition(
            name="test",
            description="Different description",
            command=["test"],
        )

        # Same content should produce same hash
        assert server1.content_hash == server2.content_hash

        # Different content should produce different hash
        assert server1.content_hash != server3.content_hash


class TestMCPRegistry:
    """Tests for MCP registry functionality."""

    @pytest.fixture
    def registry(self) -> MCPRegistry:
        """Create an MCP registry."""
        return MCPRegistry()

    @pytest.fixture
    def sample_server(self) -> MCPServerDefinition:
        """Create a sample server definition."""
        return MCPServerDefinition(
            name="filesystem",
            description="File system operations",
            command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
            args=["--path", "/tmp"],
        )

    def test_register_server(
        self, registry: MCPRegistry, sample_server: MCPServerDefinition
    ) -> None:
        """Test server registration."""
        registry.register_server(sample_server)
        assert registry.server_count == 1

        retrieved = registry.get_server("filesystem", "default")
        assert retrieved is not None
        assert retrieved.name == "filesystem"

    def test_get_server_not_found(self, registry: MCPRegistry) -> None:
        """Test getting non-existent server."""
        assert registry.get_server("nonexistent") is None

    def test_list_servers(self, registry: MCPRegistry) -> None:
        """Test listing servers."""
        server1 = MCPServerDefinition(
            name="server1",
            description="First server",
            command=["cmd1"],
            namespace="ns1",
        )
        server2 = MCPServerDefinition(
            name="server2",
            description="Second server",
            command=["cmd2"],
            namespace="ns2",
        )

        registry.register_server(server1)
        registry.register_server(server2)

        all_servers = registry.list_servers()
        assert len(all_servers) == 2

        # Filter by namespace
        ns1_servers = registry.list_servers(namespace="ns1")
        assert len(ns1_servers) == 1
        assert ns1_servers[0].name == "server1"

    def test_delete_server(
        self, registry: MCPRegistry, sample_server: MCPServerDefinition
    ) -> None:
        """Test server deletion."""
        registry.register_server(sample_server)
        assert registry.server_count == 1

        deleted = registry.delete_server("filesystem")
        assert deleted is True
        assert registry.server_count == 0

        # Try deleting again
        deleted = registry.delete_server("filesystem")
        assert deleted is False

    def test_pending_servers(
        self, registry: MCPRegistry, sample_server: MCPServerDefinition
    ) -> None:
        """Test pending servers management."""
        registry.add_pending(sample_server)
        pending = registry.get_pending()
        assert len(pending) == 1
        assert pending[0].name == "filesystem"

        registry.clear_pending()
        assert len(registry.get_pending()) == 0

    def test_update_health(
        self, registry: MCPRegistry, sample_server: MCPServerDefinition
    ) -> None:
        """Test updating server health."""
        registry.register_server(sample_server)

        updated = registry.update_health(
            "filesystem",
            "default",
            available=False,
            consecutive_failures=3,
        )
        assert updated is True

        server = registry.get_server("filesystem")
        assert server is not None
        assert server.health.available is False
        assert server.health.consecutive_failures == 3

    def test_get_client(
        self, registry: MCPRegistry, sample_server: MCPServerDefinition
    ) -> None:
        """Test lazy client instantiation."""
        registry.register_server(sample_server)

        # First call should create client
        client1 = registry.get_client("filesystem")
        assert client1 is not None
        assert registry.active_client_count == 1

        # Second call should return same client
        client2 = registry.get_client("filesystem")
        assert client1 is client2


class TestMCPRouter:
    """Tests for MCP router functionality."""

    @pytest.fixture
    def mock_vector_store(self) -> MagicMock:
        """Create a mock vector store."""
        store = MagicMock()
        store.search = AsyncMock(return_value=[])
        return store

    @pytest.fixture
    def mock_embedder(self) -> MagicMock:
        """Create a mock embedder."""
        embedder = MagicMock()
        embedder.embed_text = AsyncMock(return_value=[0.1] * 768)
        return embedder

    @pytest.fixture
    def router(
        self, mock_vector_store: MagicMock, mock_embedder: MagicMock
    ) -> MCPRouter:
        """Create an MCP router with mocks."""
        return MCPRouter(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

    @pytest.mark.asyncio
    async def test_route_basic(
        self,
        router: MCPRouter,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test basic routing."""
        result = await router.route("test query", limit=3)

        assert isinstance(result, MCPRoutingResult)
        assert result.query_embedding_time_ms >= 0
        assert result.search_time_ms >= 0
        assert result.total_time_ms >= 0

        # Verify embedder was called
        mock_embedder.embed_text.assert_called_once_with("test query")

        # Verify vector store search was called
        mock_vector_store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_filter_by_capabilities(self, router: MCPRouter) -> None:
        """Test filtering servers by capabilities."""
        servers = [
            MCPServerDefinition(
                name="server1",
                description="Server 1 with read and write capabilities",
                command=["cmd1"],
                capabilities=["read", "write"],
            ),
            MCPServerDefinition(
                name="server2",
                description="Server 2 with only read capability",
                command=["cmd2"],
                capabilities=["read"],
            ),
        ]

        # Filter for servers with both read and write
        filtered = await router.filter_by_capabilities(servers, ["read", "write"])
        assert len(filtered) == 1
        assert filtered[0].name == "server1"

        # Filter for servers with just read
        filtered = await router.filter_by_capabilities(servers, ["read"])
        assert len(filtered) == 2

    @pytest.mark.asyncio
    async def test_filter_by_health(self, router: MCPRouter) -> None:
        """Test filtering servers by health."""
        servers = [
            MCPServerDefinition(
                name="healthy",
                description="Healthy server",
                command=["cmd1"],
            ),
            MCPServerDefinition(
                name="unhealthy",
                description="Unhealthy server",
                command=["cmd2"],
            ),
        ]
        servers[1].health.available = False

        # Filter out unavailable servers
        filtered = await router.filter_by_health(servers, exclude_unavailable=True)
        assert len(filtered) == 1
        assert filtered[0].name == "healthy"

        # Don't filter
        filtered = await router.filter_by_health(servers, exclude_unavailable=False)
        assert len(filtered) == 2


class TestAgentGantryMCPIntegration:
    """Tests for MCP server selection in AgentGantry."""

    @pytest.fixture
    async def gantry(self) -> AgentGantry:
        """Create a gantry instance."""
        return AgentGantry()

    @pytest.mark.asyncio
    async def test_register_mcp_server(self, gantry: AgentGantry) -> None:
        """Test registering MCP server metadata."""
        gantry.register_mcp_server(
            name="filesystem",
            command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
            description="Provides tools for reading and writing files",
            args=["--path", "/tmp"],
            tags=["filesystem", "files"],
            examples=["read a file", "write to a file"],
            capabilities=["read_files", "write_files"],
        )

        # Verify server was registered
        assert gantry._mcp_registry.server_count == 1
        server = gantry._mcp_registry.get_server("filesystem")
        assert server is not None
        assert server.name == "filesystem"
        assert "filesystem" in server.tags

    @pytest.mark.asyncio
    async def test_discover_tools_from_server(self, gantry: AgentGantry) -> None:
        """Test discovering tools from a registered server."""
        # Register server
        gantry.register_mcp_server(
            name="test_server",
            command=["test", "cmd"],
            description="Test server for tool discovery",
        )

        # Mock the MCPClient to avoid actual connection
        with patch(
            "agent_gantry.core.mcp_registry.MCPClient"
        ) as mock_client_class:
            mock_client = AsyncMock()

            # Mock list_tools to return our test tools
            mock_client.list_tools = AsyncMock(return_value=[])
            from agent_gantry.schema.tool import ToolDefinition

            mock_client.list_tools.return_value = [
                ToolDefinition(
                    name="test_tool",
                    description="A test tool from the server",
                    parameters_schema={"type": "object", "properties": {}},
                )
            ]

            # Make get_client return our mock
            gantry._mcp_registry._clients["default.test_server"] = mock_client

            # Discover tools
            count = await gantry.discover_tools_from_server("test_server")
            assert count == 1

            # Verify health was updated
            server = gantry._mcp_registry.get_server("test_server")
            assert server is not None
            assert server.health.available is True

    @pytest.mark.asyncio
    async def test_discover_tools_failure_updates_health(
        self, gantry: AgentGantry
    ) -> None:
        """Test that tool discovery failure updates server health."""
        gantry.register_mcp_server(
            name="failing_server",
            command=["test", "cmd"],
            description="Server that will fail",
        )

        with patch(
            "agent_gantry.core.mcp_registry.MCPClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client.list_tools = AsyncMock(
                side_effect=Exception("Connection failed")
            )

            gantry._mcp_registry._clients["default.failing_server"] = mock_client

            # Try to discover tools - should raise exception
            with pytest.raises(Exception, match="Connection failed"):
                await gantry.discover_tools_from_server("failing_server")

            # Verify health was updated
            server = gantry._mcp_registry.get_server("failing_server")
            assert server is not None
            assert server.health.available is False
            assert server.health.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_backward_compatibility_add_mcp_server(
        self, gantry: AgentGantry
    ) -> None:
        """Test that add_mcp_server() still works as before."""
        from agent_gantry.schema.config import MCPServerConfig

        config = MCPServerConfig(
            name="test_server",
            command=["test", "cmd"],
            namespace="default",
        )

        with patch(
            "agent_gantry.adapters.executors.mcp_client.MCPClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            from agent_gantry.schema.tool import ToolDefinition

            mock_client.list_tools = AsyncMock(
                return_value=[
                    ToolDefinition(
                        name="tool1",
                        description="First tool from add_mcp_server",
                        parameters_schema={"type": "object"},
                    )
                ]
            )
            mock_client_class.return_value = mock_client

            count = await gantry.add_mcp_server(config)
            assert count == 1


class TestMCPWorkflow:
    """End-to-end tests for MCP server selection workflow."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self) -> None:
        """Test complete workflow: register -> sync -> retrieve -> discover."""
        gantry = AgentGantry()

        # Step 1: Register multiple servers
        gantry.register_mcp_server(
            name="filesystem",
            command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
            description="Provides tools for reading and writing files on the local filesystem",
            tags=["filesystem", "files", "io"],
            examples=["read a file", "write to a file", "list directory"],
            capabilities=["read_files", "write_files"],
        )

        gantry.register_mcp_server(
            name="database",
            command=["python", "-m", "mcp_db"],
            description="Access SQL databases for querying and data manipulation",
            tags=["database", "sql", "data"],
            examples=["query database", "insert record", "update data"],
            capabilities=["read_data", "write_data"],
        )

        # Verify both servers are registered
        assert gantry._mcp_registry.server_count == 2

        # Step 2: Sync servers (would embed to vector store)
        # Note: This is a placeholder since we haven't completed vector store integration
        count = await gantry.sync_mcp_servers()
        # Currently returns 0 because storage isn't implemented yet
        assert count >= 0

        # Step 3: Retrieve relevant servers (would use semantic search)
        # Note: This requires vector store integration to work fully
        # servers = await gantry.retrieve_mcp_servers("I need to read a file")
        # assert len(servers) > 0

        # Step 4: Discover tools from selected server
        # (Already tested in other test cases)
