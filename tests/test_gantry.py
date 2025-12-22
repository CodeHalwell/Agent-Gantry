"""
Tests for the AgentGantry main facade.
"""

from __future__ import annotations

import pytest

from agent_gantry import AgentGantry, create_default_gantry


class TestCreateDefaultGantry:
    """Tests for create_default_gantry factory function."""

    def test_create_default_gantry_returns_instance(self) -> None:
        """Test that create_default_gantry returns an AgentGantry instance."""
        gantry = create_default_gantry()
        assert isinstance(gantry, AgentGantry)
        assert gantry is not None
        assert gantry.tool_count == 0

    def test_create_default_gantry_with_dimension(self) -> None:
        """Test create_default_gantry with custom dimension parameter."""
        gantry = create_default_gantry(dimension=512)
        assert isinstance(gantry, AgentGantry)
        assert gantry is not None

    def test_create_default_gantry_multiple_instances(self) -> None:
        """Test that create_default_gantry creates independent instances."""
        gantry1 = create_default_gantry()
        gantry2 = create_default_gantry()
        assert gantry1 is not gantry2

        # Register tool on one instance
        @gantry1.register
        def test_tool(x: int) -> int:
            """A test tool."""
            return x * 2

        # Should not affect the other instance
        assert gantry1.tool_count == 1
        assert gantry2.tool_count == 0

    def test_create_default_gantry_can_register_tools(self) -> None:
        """Test that gantry from create_default_gantry can register tools."""
        gantry = create_default_gantry()

        @gantry.register(tags=["math"])
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert gantry.tool_count == 1


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


class TestAgentGantryModuleImport:
    """Tests for from_modules and collect_tools_from_modules functionality."""

    @pytest.mark.asyncio
    async def test_from_modules_basic(self) -> None:
        """Test from_modules with single module."""
        gantry = await AgentGantry.from_modules(["tests.test_modules.module_a"])
        assert gantry.tool_count == 2
        tools = gantry._registry.list_tools()
        tool_names = {t.name for t in tools}
        assert "tool_a1" in tool_names
        assert "tool_a2" in tool_names

    @pytest.mark.asyncio
    async def test_from_modules_multiple(self) -> None:
        """Test from_modules with multiple modules."""
        gantry = await AgentGantry.from_modules(
            ["tests.test_modules.module_a", "tests.test_modules.module_b"]
        )
        assert gantry.tool_count == 4
        tools = gantry._registry.list_tools()
        tool_names = {t.name for t in tools}
        assert "tool_a1" in tool_names
        assert "tool_a2" in tool_names
        assert "tool_b1" in tool_names
        assert "tool_b2" in tool_names

    @pytest.mark.asyncio
    async def test_from_modules_custom_attr(self) -> None:
        """Test from_modules with custom attribute name."""
        gantry = await AgentGantry.from_modules(
            ["tests.test_modules.module_custom_attr"],
            attr="my_custom_tools",
        )
        assert gantry.tool_count == 1
        tools = gantry._registry.list_tools()
        assert tools[0].name == "custom_tool"

    @pytest.mark.asyncio
    async def test_from_modules_missing_attribute(self) -> None:
        """Test from_modules raises error when attribute doesn't exist."""
        with pytest.raises(ValueError, match="does not expose an AgentGantry instance"):
            await AgentGantry.from_modules(["tests.test_modules.module_no_tools"])

    @pytest.mark.asyncio
    async def test_from_modules_wrong_type(self) -> None:
        """Test from_modules raises error when attribute is not AgentGantry."""
        with pytest.raises(ValueError, match="does not expose an AgentGantry instance"):
            await AgentGantry.from_modules(
                ["tests.test_modules.module_no_tools"],
                attr="some_variable",
            )

    @pytest.mark.asyncio
    async def test_collect_tools_from_modules_basic(self, gantry: AgentGantry) -> None:
        """Test collect_tools_from_modules adds tools to existing gantry."""
        # Register a tool first
        @gantry.register
        def existing_tool(x: int) -> int:
            """An existing tool."""
            return x

        assert gantry.tool_count == 1

        # Now collect tools from module
        count = await gantry.collect_tools_from_modules(["tests.test_modules.module_a"])
        assert count == 2
        assert gantry.tool_count == 3

    @pytest.mark.asyncio
    async def test_collect_tools_from_modules_duplicate_handling(
        self, gantry: AgentGantry
    ) -> None:
        """Test that duplicate tools within same batch are skipped."""
        # Import from module_a and module_c_duplicate in one call
        # module_c_duplicate has tool_a1 which duplicates module_a's tool_a1
        count = await gantry.collect_tools_from_modules([
            "tests.test_modules.module_a",
            "tests.test_modules.module_c_duplicate"
        ])
        # Should import 2 from module_a, and skip the duplicate from module_c_duplicate
        assert count == 2
        assert gantry.tool_count == 2

    @pytest.mark.asyncio
    async def test_collect_tools_from_modules_handlers(
        self, gantry: AgentGantry
    ) -> None:
        """Test that tool handlers are properly registered."""
        await gantry.collect_tools_from_modules(["tests.test_modules.module_a"])

        # Check that handlers are registered
        assert "tool_a1" in gantry._tool_handlers
        assert "tool_a2" in gantry._tool_handlers

        # Verify handlers are callable
        handler1 = gantry._tool_handlers["tool_a1"]
        assert callable(handler1)
        result = handler1(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_collect_tools_from_modules_custom_attr(
        self, gantry: AgentGantry
    ) -> None:
        """Test collect_tools_from_modules with custom attribute name."""
        count = await gantry.collect_tools_from_modules(
            ["tests.test_modules.module_custom_attr"],
            module_attr="my_custom_tools",
        )
        assert count == 1
        assert gantry.tool_count == 1

    @pytest.mark.asyncio
    async def test_collect_tools_from_modules_error_handling(
        self, gantry: AgentGantry
    ) -> None:
        """Test collect_tools_from_modules error handling."""
        with pytest.raises(ValueError, match="does not expose an AgentGantry instance"):
            await gantry.collect_tools_from_modules(["tests.test_modules.module_no_tools"])
