"""
Tests for the AgentGantry main facade.
"""

from __future__ import annotations

import pytest

from agent_gantry import AgentGantry


class TestAgentGantry:
    """Tests for AgentGantry class."""

    def test_create_instance(self, gantry: AgentGantry) -> None:
        """Test creating an AgentGantry instance."""
        assert gantry is not None
        assert gantry.tool_count == 0

    def test_register_decorator(self, gantry: AgentGantry) -> None:
        """Test registering a tool with the decorator."""

        @gantry.register
        def my_tool(x: int) -> str:
            """A simple test tool that multiplies a number."""
            return str(x * 2)

        assert gantry.tool_count == 1

    def test_register_decorator_with_options(self, gantry: AgentGantry) -> None:
        """Test registering a tool with decorator options."""

        @gantry.register(name="custom_name", tags=["test"])
        def another_tool(y: float) -> float:
            """A test tool with custom options."""
            return y * 3.0

        assert gantry.tool_count == 1

    def test_multiple_registrations(self, gantry: AgentGantry) -> None:
        """Test registering multiple tools."""

        @gantry.register
        def tool_one(a: str) -> str:
            """First test tool that processes strings."""
            return a.upper()

        @gantry.register
        def tool_two(b: int) -> int:
            """Second test tool that processes integers."""
            return b + 1

        assert gantry.tool_count == 2


class TestAgentGantryHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check(self, gantry: AgentGantry) -> None:
        """Test the health check method."""
        health = await gantry.health_check()
        assert isinstance(health, dict)
        assert "vector_store" in health
        assert "embedder" in health
        assert "telemetry" in health
