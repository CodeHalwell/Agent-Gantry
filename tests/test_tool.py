"""
Tests for tool definition models.
"""

from __future__ import annotations

import pytest

from agent_gantry.schema.tool import (
    SchemaDialect,
    ToolCapability,
    ToolCost,
    ToolDefinition,
    ToolHealth,
    ToolSource,
)


class TestToolDefinition:
    """Tests for ToolDefinition model."""

    def test_create_minimal_tool(self) -> None:
        """Test creating a tool with minimal required fields."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool for testing purposes.",
            parameters_schema={"type": "object", "properties": {}},
        )
        assert tool.name == "test_tool"
        assert tool.version == "1.0.0"
        assert tool.namespace == "default"

    def test_create_full_tool(self) -> None:
        """Test creating a tool with all fields."""
        tool = ToolDefinition(
            name="full_tool",
            version="2.0.0",
            namespace="custom",
            description="A fully specified test tool.",
            parameters_schema={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "required": ["arg"],
            },
            tags=["test", "example"],
            capabilities=[ToolCapability.READ_DATA],
            requires_confirmation=True,
            source=ToolSource.MANUAL,
        )
        assert tool.name == "full_tool"
        assert tool.version == "2.0.0"
        assert tool.namespace == "custom"
        assert tool.requires_confirmation is True

    def test_qualified_name(self) -> None:
        """Test the qualified name property."""
        tool = ToolDefinition(
            name="my_tool",
            namespace="my_namespace",
            version="1.2.3",
            description="Test tool description.",
            parameters_schema={"type": "object", "properties": {}},
        )
        assert tool.qualified_name == "my_namespace.my_tool:1.2.3"

    def test_content_hash_consistency(self) -> None:
        """Test that content hash is consistent."""
        tool1 = ToolDefinition(
            name="test_tool",
            description="A test tool description.",
            parameters_schema={"type": "object", "properties": {}},
        )
        tool2 = ToolDefinition(
            name="test_tool",
            description="A test tool description.",
            parameters_schema={"type": "object", "properties": {}},
        )
        assert tool1.content_hash == tool2.content_hash

    def test_content_hash_changes_on_modification(self) -> None:
        """Test that content hash changes when tool is modified."""
        tool1 = ToolDefinition(
            name="test_tool",
            description="Original description for the test.",
            parameters_schema={"type": "object", "properties": {}},
        )
        tool2 = ToolDefinition(
            name="test_tool",
            description="Modified description for the test.",
            parameters_schema={"type": "object", "properties": {}},
        )
        assert tool1.content_hash != tool2.content_hash

    def test_reserved_name_validation(self) -> None:
        """Test that reserved names are rejected."""
        with pytest.raises(ValueError, match="reserved"):
            ToolDefinition(
                name="register",
                description="This should fail because 'register' is reserved.",
                parameters_schema={"type": "object", "properties": {}},
            )

    def test_to_dialect_openai(self) -> None:
        """Test dialect conversion to OpenAI schema format."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool for OpenAI.",
            parameters_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
        )
        schema = tool.to_dialect("openai")
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool for OpenAI."

    def test_to_dialect_anthropic(self) -> None:
        """Test dialect conversion to Anthropic schema format."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool for Anthropic.",
            parameters_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
        )
        schema = tool.to_dialect("anthropic")
        assert schema["name"] == "test_tool"
        assert schema["description"] == "A test tool for Anthropic."
        assert "input_schema" in schema

    def test_to_dialect_auto(self) -> None:
        """Test dialect conversion with AUTO defaults to OpenAI."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool for dialect.",
            parameters_schema={"type": "object", "properties": {}},
        )
        schema = tool.to_dialect(SchemaDialect.AUTO)
        assert schema["type"] == "function"


class TestToolSource:
    """Tests for the ToolSource enum."""

    def test_framework_member_exists(self) -> None:
        """FRAMEWORK is a valid source for tools imported from other agent
        frameworks (LangChain/CrewAI/LlamaIndex/...) via
        agent_gantry.integrations.importers."""
        assert ToolSource.FRAMEWORK == "framework"
        assert ToolSource("framework") is ToolSource.FRAMEWORK

    def test_all_expected_members_present(self) -> None:
        """Adding FRAMEWORK must not disturb any pre-existing member."""
        assert {member.value for member in ToolSource} == {
            "python_function",
            "mcp_server",
            "openapi",
            "a2a_agent",
            "manual",
            "framework",
        }

    def test_tool_definition_with_framework_source(self) -> None:
        """A ToolDefinition can be constructed with source=ToolSource.FRAMEWORK."""
        tool = ToolDefinition(
            name="imported_tool",
            description="A tool imported from an external agent framework.",
            parameters_schema={"type": "object", "properties": {}},
            source=ToolSource.FRAMEWORK,
            source_uri="langchain://imported_tool",
        )
        assert tool.source == ToolSource.FRAMEWORK
        assert tool.source_uri == "langchain://imported_tool"

    def test_default_source_is_python_function(self) -> None:
        """Existing behavior is unaffected: default source is still PYTHON_FUNCTION."""
        tool = ToolDefinition(
            name="default_source_tool",
            description="A tool with no explicit source.",
            parameters_schema={"type": "object", "properties": {}},
        )
        assert tool.source == ToolSource.PYTHON_FUNCTION


class TestToolCost:
    """Tests for ToolCost model."""

    def test_default_values(self) -> None:
        """Test default cost values."""
        cost = ToolCost()
        assert cost.estimated_latency_ms == 100
        assert cost.monetary_cost is None
        assert cost.rate_limit is None
        assert cost.context_tokens == 0


class TestToolHealth:
    """Tests for ToolHealth model."""

    def test_default_values(self) -> None:
        """Test default health values."""
        health = ToolHealth()
        assert health.success_rate == 1.0
        assert health.total_calls == 0
        assert health.circuit_breaker_open is False
