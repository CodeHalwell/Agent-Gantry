"""
Tests for Anthropic Skills API integration.

Tests the Skills API features including:
- Skill registration
- Skill registry management
- Skills client functionality
- Tool execution with skills
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from agent_gantry import AgentGantry
from agent_gantry.integrations.anthropic_skills import (
    Skill,
    SkillRegistry,
    SkillsClient,
    create_skills_client,
)

# Create mock anthropic module
mock_anthropic = Mock()
mock_anthropic.AsyncAnthropic = MagicMock
sys.modules['anthropic'] = mock_anthropic


class TestSkill:
    """Tests for Skill dataclass."""

    def test_skill_creation(self):
        """Test creating a skill."""
        skill = Skill(
            name="test_skill",
            description="A test skill",
            instructions="Test instructions",
            tools=["tool1", "tool2"],
        )

        assert skill.name == "test_skill"
        assert skill.description == "A test skill"
        assert skill.instructions == "Test instructions"
        assert skill.tools == ["tool1", "tool2"]
        assert skill.examples == []
        assert skill.metadata == {}

    def test_skill_to_anthropic_schema(self):
        """Test converting skill to Anthropic schema."""
        skill = Skill(
            name="customer_support",
            description="Handle customer inquiries",
            instructions="Use these tools to help customers",
            tools=["get_order", "process_refund"],
            examples=[{"input": "test", "output": "result"}],
            metadata={"category": "support"},
        )

        schema = skill.to_anthropic_schema()

        assert schema["type"] == "skill"
        assert schema["name"] == "customer_support"
        assert schema["description"] == "Handle customer inquiries"
        assert schema["instructions"] == "Use these tools to help customers"
        assert schema["tools"] == ["get_order", "process_refund"]
        assert len(schema["examples"]) == 1
        assert schema["metadata"]["category"] == "support"

    def test_skill_minimal_schema(self):
        """Test skill with minimal fields."""
        skill = Skill(
            name="minimal",
            description="Minimal skill",
            instructions="Do something",
        )

        schema = skill.to_anthropic_schema()

        assert "type" in schema
        assert "name" in schema
        assert "description" in schema
        assert "instructions" in schema
        assert "tools" not in schema  # Empty list shouldn't be included
        assert "examples" not in schema
        assert "metadata" not in schema


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = SkillRegistry()
        assert len(registry.list_skills()) == 0

    def test_register_skill(self):
        """Test registering a skill."""
        registry = SkillRegistry()

        skill = registry.register(
            name="test",
            description="Test skill",
            instructions="Test instructions",
        )

        assert skill.name == "test"
        assert len(registry.list_skills()) == 1

    def test_register_duplicate_skill(self):
        """Test that registering duplicate skill raises error."""
        registry = SkillRegistry()

        registry.register(
            name="test",
            description="Test skill",
            instructions="Test instructions",
        )

        with pytest.raises(ValueError, match="already registered"):
            registry.register(
                name="test",
                description="Another skill",
                instructions="Different instructions",
            )

    def test_get_skill(self):
        """Test getting a skill by name."""
        registry = SkillRegistry()

        registry.register(
            name="skill1",
            description="First skill",
            instructions="Instructions 1",
        )

        skill = registry.get("skill1")
        assert skill is not None
        assert skill.name == "skill1"

        missing = registry.get("nonexistent")
        assert missing is None

    def test_list_skills(self):
        """Test listing all skills."""
        registry = SkillRegistry()

        registry.register("skill1", "Desc 1", "Inst 1")
        registry.register("skill2", "Desc 2", "Inst 2")
        registry.register("skill3", "Desc 3", "Inst 3")

        skills = registry.list_skills()
        assert len(skills) == 3
        assert all(isinstance(s, Skill) for s in skills)

    def test_to_anthropic_schema(self):
        """Test converting registry to Anthropic schema."""
        registry = SkillRegistry()

        registry.register("skill1", "Desc 1", "Inst 1", tools=["tool1"])
        registry.register("skill2", "Desc 2", "Inst 2", tools=["tool2"])

        schemas = registry.to_anthropic_schema()
        assert len(schemas) == 2
        assert all("type" in s and s["type"] == "skill" for s in schemas)

    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = SkillRegistry()

        registry.register("skill1", "Desc 1", "Inst 1")
        registry.register("skill2", "Desc 2", "Inst 2")
        assert len(registry.list_skills()) == 2

        registry.clear()
        assert len(registry.list_skills()) == 0


class TestSkillsClient:
    """Tests for SkillsClient."""

    def test_client_initialization(self):
        """Test client initialization."""
        client = SkillsClient(api_key="test-key")
        assert client is not None
        assert isinstance(client.skills, SkillRegistry)

    def test_client_with_gantry(self):
        """Test client initialization with AgentGantry."""
        gantry = MagicMock(spec=AgentGantry)
        client = SkillsClient(api_key="test-key", gantry=gantry)
        assert client._gantry == gantry

    def test_client_with_custom_registry(self):
        """Test client with custom skill registry."""
        registry = SkillRegistry()
        registry.register("test", "Test skill", "Instructions")

        client = SkillsClient(api_key="test-key", skill_registry=registry)
        assert len(client.skills.list_skills()) == 1

    def test_skills_property(self):
        """Test accessing skills registry."""
        client = SkillsClient(api_key="test-key")
        assert isinstance(client.skills, SkillRegistry)

        # Can register through client.skills
        client.skills.register("test", "Test", "Instructions")
        assert len(client.skills.list_skills()) == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls(self):
        """Test executing tool calls from response."""
        gantry = MagicMock(spec=AgentGantry)
        mock_result = MagicMock()
        mock_result.status = "success"
        mock_result.result = "Tool executed"
        gantry.execute = AsyncMock(return_value=mock_result)

        client = SkillsClient(api_key="test-key", gantry=gantry)

        # Mock response with tool use
        response = MagicMock()
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_123"
        tool_block.name = "test_tool"
        tool_block.input = {"arg": "value"}
        response.content = [tool_block]

        tool_results = await client.execute_tool_calls(response)

        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_123"
        assert "Tool executed" in tool_results[0]["content"]

    @pytest.mark.asyncio
    async def test_execute_tool_calls_without_gantry(self):
        """Test that execute_tool_calls requires gantry."""
        client = SkillsClient(api_key="test-key")
        response = MagicMock()

        with pytest.raises(ValueError, match="AgentGantry instance required"):
            await client.execute_tool_calls(response)

    def test_register_skill_from_gantry_tools(self):
        """Test registering skill from Agent-Gantry tools."""
        client = SkillsClient(api_key="test-key")

        skill = client.register_skill_from_gantry_tools(
            skill_name="math",
            description="Math operations",
            instructions="Use these tools for math",
            tool_names=["add", "subtract", "multiply"],
            examples=[{"input": "2+2", "output": "4"}],
        )

        assert skill.name == "math"
        assert skill.description == "Math operations"
        assert skill.tools == ["add", "subtract", "multiply"]
        assert len(skill.examples) == 1
        assert len(client.skills.list_skills()) == 1


class TestCreateSkillsClient:
    """Tests for create_skills_client convenience function."""

    @pytest.mark.asyncio
    async def test_create_client_basic(self):
        """Test creating client with basic parameters."""
        client = await create_skills_client(api_key="test-key")
        assert isinstance(client, SkillsClient)
        assert isinstance(client.skills, SkillRegistry)

    @pytest.mark.asyncio
    async def test_create_client_with_gantry(self):
        """Test creating client with AgentGantry."""
        gantry = MagicMock(spec=AgentGantry)
        client = await create_skills_client(api_key="test-key", gantry=gantry)
        assert client._gantry == gantry


class TestSkillsIntegration:
    """Integration tests for Skills API."""

    @pytest.mark.asyncio
    async def test_skill_workflow(self):
        """Test complete skill registration and usage workflow."""
        # Create client with registry
        client = await create_skills_client(api_key="test-key")

        # Register a skill
        client.skills.register(
            name="customer_support",
            description="Handle customer support",
            instructions="Use tools to help customers",
            tools=["get_order", "process_refund"],
            examples=[
                {
                    "input": "I need a refund",
                    "steps": ["Check order", "Process refund"],
                }
            ],
        )

        # Verify skill is registered
        assert len(client.skills.list_skills()) == 1
        skill = client.skills.get("customer_support")
        assert skill is not None
        assert skill.tools == ["get_order", "process_refund"]

        # Convert to Anthropic schema
        schemas = client.skills.to_anthropic_schema()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "customer_support"
        assert schemas[0]["type"] == "skill"

    @pytest.mark.asyncio
    async def test_multiple_skills_management(self):
        """Test managing multiple skills."""
        client = await create_skills_client(api_key="test-key")

        # Register multiple skills
        client.skills.register("skill1", "Desc 1", "Inst 1", tools=["tool1"])
        client.skills.register("skill2", "Desc 2", "Inst 2", tools=["tool2"])
        client.skills.register("skill3", "Desc 3", "Inst 3", tools=["tool3"])

        # Verify all registered
        skills = client.skills.list_skills()
        assert len(skills) == 3

        # Get specific skills
        assert client.skills.get("skill1") is not None
        assert client.skills.get("skill2") is not None
        assert client.skills.get("skill3") is not None
        assert client.skills.get("nonexistent") is None

        # Clear and verify
        client.skills.clear()
        assert len(client.skills.list_skills()) == 0
