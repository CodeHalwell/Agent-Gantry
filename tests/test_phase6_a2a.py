"""
Tests for Phase 6: A2A Integration.

Tests A2A client, server, agent card, and protocol compliance.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_gantry import AgentGantry
from agent_gantry.providers.a2a_client import A2AClient
from agent_gantry.schema.a2a import AgentCard, AgentSkill
from agent_gantry.schema.config import A2AAgentConfig
from agent_gantry.schema.tool import ToolDefinition, ToolSource
from agent_gantry.servers.a2a_server import create_a2a_server, generate_agent_card


class TestAgentCard:
    """Tests for Agent Card generation."""

    @pytest.mark.asyncio
    async def test_generate_agent_card(self, gantry: AgentGantry) -> None:
        """Test agent card generation from gantry instance."""
        # Register a few tools
        @gantry.register
        def test_tool(x: int) -> int:
            """A test tool for agent card generation."""
            return x * 2

        await gantry.sync()

        # Generate agent card
        card = generate_agent_card(gantry, "http://localhost:8080")

        assert isinstance(card, AgentCard)
        assert card.name == "AgentGantry"
        assert "tool" in card.description.lower()
        assert card.url == "http://localhost:8080"
        assert card.version == "1.0.0"
        assert len(card.skills) == 2  # tool_discovery and tool_execution

        # Check skills
        skill_ids = {skill.id for skill in card.skills}
        assert "tool_discovery" in skill_ids
        assert "tool_execution" in skill_ids

    def test_agent_skill_model(self) -> None:
        """Test AgentSkill model."""
        skill = AgentSkill(
            id="test_skill",
            name="Test Skill",
            description="A skill for testing",
            input_modes=["text", "json"],
            output_modes=["text"],
        )

        assert skill.id == "test_skill"
        assert skill.name == "Test Skill"
        assert "json" in skill.input_modes

    def test_agent_card_serialization(self) -> None:
        """Test agent card can be serialized to dict."""
        card = AgentCard(
            name="TestAgent",
            description="Test agent",
            url="http://test.example.com",
            skills=[
                AgentSkill(
                    id="skill1",
                    name="Skill 1",
                    description="First skill",
                )
            ],
        )

        data = card.model_dump()
        assert data["name"] == "TestAgent"
        assert len(data["skills"]) == 1
        assert data["skills"][0]["id"] == "skill1"


class TestA2AClient:
    """Tests for A2A client functionality."""

    @pytest.fixture
    def a2a_config(self) -> A2AAgentConfig:
        """Create a sample A2A agent config."""
        return A2AAgentConfig(
            name="test-agent",
            url="http://test-agent.example.com",
            namespace="a2a_test",
        )

    @pytest.fixture
    def a2a_client(self, a2a_config: A2AAgentConfig) -> A2AClient:
        """Create an A2A client instance."""
        return A2AClient(a2a_config)

    def test_client_initialization(
        self, a2a_client: A2AClient, a2a_config: A2AAgentConfig
    ) -> None:
        """Test A2A client initialization."""
        assert a2a_client.config == a2a_config
        assert a2a_client._agent_card is None
        assert a2a_client._base_url == "http://test-agent.example.com"

    @pytest.mark.asyncio
    async def test_discover_agent(self, a2a_client: A2AClient) -> None:
        """Test discovering A2A agent with mocked HTTP."""
        mock_agent_card = {
            "name": "TestAgent",
            "description": "A test agent",
            "url": "http://test-agent.example.com",
            "version": "1.0.0",
            "skills": [
                {
                    "id": "skill1",
                    "name": "Skill 1",
                    "description": "First skill",
                    "input_modes": ["text"],
                    "output_modes": ["text"],
                }
            ],
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_agent_card
            mock_response.raise_for_status = MagicMock()

            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            card = await a2a_client.discover()

            assert isinstance(card, AgentCard)
            assert card.name == "TestAgent"
            assert len(card.skills) == 1
            assert card.skills[0].id == "skill1"
            assert a2a_client._agent_card is not None

    @pytest.mark.asyncio
    async def test_skill_to_tool_conversion(self, a2a_client: A2AClient) -> None:
        """Test converting agent skill to ToolDefinition."""
        skill = AgentSkill(
            id="analyze_data",
            name="Data Analysis",
            description="Analyze data and provide insights",
            input_modes=["text", "json"],
            output_modes=["text", "json"],
        )

        tool = a2a_client._skill_to_tool(skill)

        assert isinstance(tool, ToolDefinition)
        assert tool.name == "a2a_test_agent_analyze_data"
        assert tool.description == skill.description
        assert tool.namespace == a2a_client.config.namespace
        assert tool.source == ToolSource.A2A_AGENT
        assert tool.source_uri == "a2a://test-agent"
        assert tool.metadata["a2a_agent"] == "test-agent"
        assert tool.metadata["skill_id"] == "analyze_data"
        assert "query" in tool.parameters_schema["properties"]

    @pytest.mark.asyncio
    async def test_list_tools_requires_discovery(self, a2a_client: A2AClient) -> None:
        """Test that list_tools discovers agent if not already done."""
        mock_agent_card = {
            "name": "TestAgent",
            "description": "A test agent",
            "url": "http://test-agent.example.com",
            "version": "1.0.0",
            "skills": [
                {
                    "id": "skill1",
                    "name": "Skill 1",
                    "description": "First skill for testing",
                    "input_modes": ["text"],
                    "output_modes": ["text"],
                },
                {
                    "id": "skill2",
                    "name": "Skill 2",
                    "description": "Second skill for testing",
                    "input_modes": ["text"],
                    "output_modes": ["text"],
                },
            ],
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_agent_card
            mock_response.raise_for_status = MagicMock()

            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            tools = await a2a_client.list_tools()

            assert len(tools) == 2
            assert all(isinstance(t, ToolDefinition) for t in tools)
            assert all(t.source == ToolSource.A2A_AGENT for t in tools)

    @pytest.mark.asyncio
    async def test_send_task(self, a2a_client: A2AClient) -> None:
        """Test sending task to A2A agent."""
        mock_response_data = {"result": {"status": "success", "output": "Task completed"}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status = MagicMock()

            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            result = await a2a_client.send_task(
                skill_id="test_skill",
                query="Analyze this data",
                metadata={"trace_id": "test-123"},
            )

            assert result["status"] == "success"
            assert result["output"] == "Task completed"


class TestA2AServer:
    """Tests for A2A server functionality."""

    @pytest.mark.asyncio
    async def test_create_a2a_server(self, gantry: AgentGantry) -> None:
        """Test creating A2A server FastAPI app."""
        try:
            app = create_a2a_server(gantry, base_url="http://localhost:8080")
            assert app is not None
            assert app.title == "AgentGantry A2A Server"
        except ImportError:
            pytest.skip("FastAPI not installed")

    @pytest.mark.asyncio
    async def test_agent_card_endpoint(self, gantry: AgentGantry) -> None:
        """Test /.well-known/agent.json endpoint."""
        try:
            from fastapi.testclient import TestClient

            app = create_a2a_server(gantry, base_url="http://localhost:8080")
            client = TestClient(app)

            response = client.get("/.well-known/agent.json")
            assert response.status_code == 200
            data = response.json()

            assert data["name"] == "AgentGantry"
            assert data["url"] == "http://localhost:8080"
            assert "skills" in data
            assert len(data["skills"]) == 2

        except ImportError:
            pytest.skip("FastAPI not installed")

    @pytest.mark.asyncio
    async def test_tasks_send_endpoint_tool_discovery(self, gantry: AgentGantry) -> None:
        """Test tasks/send endpoint with tool_discovery skill."""
        try:
            from fastapi.testclient import TestClient

            # Register a test tool
            @gantry.register
            def calculate_sum(a: int, b: int) -> int:
                """Calculate the sum of two numbers for testing."""
                return a + b

            await gantry.sync()

            app = create_a2a_server(gantry, base_url="http://localhost:8080")
            client = TestClient(app)

            # Send JSON-RPC request
            request_data = {
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "params": {
                    "skill_id": "tool_discovery",
                    "messages": [
                        {
                            "role": "user",
                            "parts": [{"type": "text", "text": "calculate sum"}],
                        }
                    ],
                },
                "id": 1,
            }

            response = client.post("/tasks/send", json=request_data)
            assert response.status_code == 200

            data = response.json()
            assert data["jsonrpc"] == "2.0"
            assert "result" in data
            assert data["result"]["status"] == "success"

            result = data["result"]["result"]
            assert "tools_found" in result
            assert result["tools_found"] > 0

        except ImportError:
            pytest.skip("FastAPI not installed")


class TestA2AIntegration:
    """Tests for end-to-end A2A integration."""

    @pytest.mark.asyncio
    async def test_add_a2a_agent_to_gantry(self, gantry: AgentGantry) -> None:
        """Test adding A2A agent to AgentGantry."""
        config = A2AAgentConfig(
            name="test-agent",
            url="http://test-agent.example.com",
            namespace="external",
        )

        mock_agent_card = {
            "name": "ExternalAgent",
            "description": "An external A2A agent",
            "url": "http://test-agent.example.com",
            "version": "1.0.0",
            "skills": [
                {
                    "id": "translation",
                    "name": "Translation",
                    "description": "Translate text between languages in testing",
                    "input_modes": ["text"],
                    "output_modes": ["text"],
                }
            ],
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_agent_card
            mock_response.raise_for_status = MagicMock()

            mock_instance = AsyncMock()
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_instance

            count = await gantry.add_a2a_agent(config)

            assert count == 1

            # Check that tool was added
            tools = await gantry.list_tools(namespace="external")
            assert len(tools) == 1
            assert tools[0].source == ToolSource.A2A_AGENT
            assert "translation" in tools[0].name

    @pytest.mark.asyncio
    async def test_a2a_authentication_config(self) -> None:
        """Test A2A agent card with authentication configuration."""
        card = AgentCard(
            name="SecureAgent",
            description="Agent with authentication",
            url="http://secure-agent.example.com",
            skills=[
                AgentSkill(
                    id="secure_skill",
                    name="Secure Skill",
                    description="A skill requiring authentication",
                )
            ],
            authentication={"type": "bearer", "required": True},
        )

        assert card.authentication is not None
        assert card.authentication["type"] == "bearer"
        assert card.authentication["required"] is True

        # Test serialization
        data = card.model_dump()
        assert "authentication" in data
        assert data["authentication"]["type"] == "bearer"
