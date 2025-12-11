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

    def test_to_openai_schema(self) -> None:
        """Test conversion to OpenAI schema format."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool for OpenAI.",
            parameters_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
        )
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool for OpenAI."

    def test_to_anthropic_schema(self) -> None:
        """Test conversion to Anthropic schema format."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool for Anthropic.",
            parameters_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
        )
        schema = tool.to_anthropic_schema()
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
