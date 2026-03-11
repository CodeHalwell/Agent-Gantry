"""
Tests for Microsoft Agent Framework integration.

Covers the AgentFrameworkAdapter (tool spec), the GantryToolBridge,
the framework_adapters entry, and the updated example.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

from agent_gantry.adapters.tool_spec import DialectRegistry, ToolCallPayload, get_adapter
from agent_gantry.adapters.tool_spec.providers import AgentFrameworkAdapter
from agent_gantry.schema.tool import SchemaDialect, ToolDefinition


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_tool() -> ToolDefinition:
    """Create a sample tool definition for testing."""
    return ToolDefinition(
        name="get_weather",
        description="Get the current weather for a specified city.",
        parameters_schema={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get weather for",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["city"],
        },
        tags=["weather", "api"],
    )


@pytest.fixture
def single_param_tool() -> ToolDefinition:
    return ToolDefinition(
        name="lookup_user",
        description="Look up a user by their unique identifier.",
        parameters_schema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The user ID to look up",
                },
            },
            "required": ["user_id"],
        },
    )


@pytest.fixture
def no_param_tool() -> ToolDefinition:
    return ToolDefinition(
        name="get_system_status",
        description="Retrieve the current system health status.",
        parameters_schema={
            "type": "object",
            "properties": {},
        },
    )


@pytest.fixture
def adapter() -> AgentFrameworkAdapter:
    return AgentFrameworkAdapter()


# ---------------------------------------------------------------------------
# AgentFrameworkAdapter — schema conversion
# ---------------------------------------------------------------------------


class TestAgentFrameworkAdapter:
    """Tests for the Microsoft Agent Framework tool spec adapter."""

    def test_dialect_name(self, adapter: AgentFrameworkAdapter) -> None:
        assert adapter.dialect_name == "agent_framework"

    def test_to_provider_schema_basic(
        self, adapter: AgentFrameworkAdapter, sample_tool: ToolDefinition
    ) -> None:
        schema = adapter.to_provider_schema(sample_tool)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert schema["function"]["description"] == sample_tool.description
        assert schema["function"]["parameters"] == sample_tool.parameters_schema
        assert "metadata" not in schema["function"]

    def test_to_provider_schema_with_metadata(
        self, adapter: AgentFrameworkAdapter, sample_tool: ToolDefinition
    ) -> None:
        schema = adapter.to_provider_schema(sample_tool, include_metadata=True)
        assert "metadata" in schema["function"]
        meta = schema["function"]["metadata"]
        assert meta["namespace"] == "default"
        assert meta["version"] == "1.0.0"
        assert meta["source"] == "python_function"

    def test_from_provider_payload_openai_style(
        self, adapter: AgentFrameworkAdapter
    ) -> None:
        payload = adapter.from_provider_payload(
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                },
            }
        )
        assert payload.tool_name == "get_weather"
        assert payload.tool_call_id == "call_abc"
        assert payload.arguments == {"city": "London"}

    def test_from_provider_payload_simplified(
        self, adapter: AgentFrameworkAdapter
    ) -> None:
        payload = adapter.from_provider_payload(
            {
                "name": "get_weather",
                "arguments": {"city": "London"},
                "call_id": "call_xyz",
            }
        )
        assert payload.tool_name == "get_weather"
        assert payload.tool_call_id == "call_xyz"
        assert payload.arguments == {"city": "London"}

    def test_from_provider_payload_invalid_json_args(
        self, adapter: AgentFrameworkAdapter
    ) -> None:
        payload = adapter.from_provider_payload(
            {
                "name": "test",
                "arguments": "not valid json{",
            }
        )
        assert payload.arguments == {}

    def test_to_tool_call(self, adapter: AgentFrameworkAdapter) -> None:
        payload = ToolCallPayload(
            tool_name="get_weather",
            tool_call_id="call_123",
            arguments={"city": "Paris"},
        )
        call = adapter.to_tool_call(payload, timeout_ms=5000, retry_count=2)
        assert call.tool_name == "get_weather"
        assert call.arguments == {"city": "Paris"}
        assert call.timeout_ms == 5000
        assert call.retry_count == 2
        assert call.trace_id == "call_123"

    def test_format_tool_result_string(
        self, adapter: AgentFrameworkAdapter
    ) -> None:
        result = adapter.format_tool_result("get_weather", "Sunny, 22C", "call_1")
        assert result["role"] == "tool"
        assert result["content"] == "Sunny, 22C"
        assert result["name"] == "get_weather"
        assert result["tool_call_id"] == "call_1"

    def test_format_tool_result_dict(
        self, adapter: AgentFrameworkAdapter
    ) -> None:
        result = adapter.format_tool_result(
            "get_weather", {"temp": 22, "unit": "C"}, None
        )
        assert result["role"] == "tool"
        assert json.loads(result["content"]) == {"temp": 22, "unit": "C"}
        assert "tool_call_id" not in result

    def test_format_tool_result_no_call_id(
        self, adapter: AgentFrameworkAdapter
    ) -> None:
        result = adapter.format_tool_result("test", "ok")
        assert "tool_call_id" not in result


# ---------------------------------------------------------------------------
# Dialect Registry integration
# ---------------------------------------------------------------------------


class TestDialectRegistryIntegration:
    """Verify AgentFrameworkAdapter is registered in the default registry."""

    def test_agent_framework_registered(self) -> None:
        # Reset singleton to ensure fresh registration
        DialectRegistry._instance = None
        registry = DialectRegistry.default()
        assert registry.has("agent_framework")

    def test_get_adapter_by_name(self) -> None:
        DialectRegistry._instance = None
        adapter = get_adapter("agent_framework")
        assert adapter.dialect_name == "agent_framework"

    def test_list_dialects_includes_agent_framework(self) -> None:
        DialectRegistry._instance = None
        dialects = DialectRegistry.default().list_dialects()
        assert "agent_framework" in dialects


# ---------------------------------------------------------------------------
# SchemaDialect enum
# ---------------------------------------------------------------------------


class TestSchemaDialectEnum:
    def test_agent_framework_dialect_exists(self) -> None:
        assert SchemaDialect.AGENT_FRAMEWORK.value == "agent_framework"

    def test_tool_definition_to_dialect(self, sample_tool: ToolDefinition) -> None:
        schema = sample_tool.to_dialect("agent_framework")
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# GantryToolBridge
# ---------------------------------------------------------------------------


class TestGantryToolBridge:
    """Tests for the GantryToolBridge."""

    @pytest.mark.asyncio
    async def test_bridge_get_tools_returns_callables(self) -> None:
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def get_user_profile(user_id: str) -> dict[str, str]:
            """Fetch a user's profile from the CRM system."""
            return {"user_id": user_id, "plan": "pro"}

        await gantry.sync()

        bridge = GantryToolBridge(gantry, score_threshold=0.0)
        tools = await bridge.get_tools("user profile", limit=5, score_threshold=0.0)

        assert len(tools) >= 1
        tool = tools[0]
        assert callable(tool)
        assert tool.__name__ == "get_user_profile"
        assert "profile" in (tool.__doc__ or "").lower()

    @pytest.mark.asyncio
    async def test_bridge_tool_execution(self) -> None:
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def get_user_profile(user_id: str) -> dict[str, str]:
            """Fetch a user's profile from the CRM system."""
            return {"user_id": user_id, "plan": "pro"}

        await gantry.sync()

        bridge = GantryToolBridge(gantry, score_threshold=0.0)
        tools = await bridge.get_tools("user profile", limit=5, score_threshold=0.0)

        # Execute the wrapped tool
        result = await tools[0](user_id="abc123")
        parsed = json.loads(result)
        assert parsed["user_id"] == "abc123"
        assert parsed["plan"] == "pro"

    @pytest.mark.asyncio
    async def test_bridge_multi_param_tool(self) -> None:
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def search_products(query: str, category: str) -> str:
            """Search for products by query and category in the catalog."""
            return f"Found 5 products matching '{query}' in '{category}'"

        await gantry.sync()

        bridge = GantryToolBridge(gantry, score_threshold=0.0)
        tools = await bridge.get_tools("search products", limit=5, score_threshold=0.0)

        assert len(tools) >= 1
        tool = tools[0]
        assert tool.__name__ == "search_products"

        # Check the wrapper has proper parameter annotations
        sig = inspect.signature(tool)
        param_names = [p for p in sig.parameters if p != "kwargs"]
        assert "query" in param_names
        assert "category" in param_names

    @pytest.mark.asyncio
    async def test_bridge_no_param_tool(self) -> None:
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def get_system_status() -> str:
            """Retrieve the current system health status information."""
            return "all systems operational"

        await gantry.sync()

        bridge = GantryToolBridge(gantry, score_threshold=0.0)
        tools = await bridge.get_tools("system status", limit=5, score_threshold=0.0)

        assert len(tools) >= 1
        result = await tools[0]()
        assert "operational" in result

    @pytest.mark.asyncio
    async def test_bridge_caching(self) -> None:
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def get_user_profile(user_id: str) -> dict[str, str]:
            """Fetch a user's profile from the CRM system."""
            return {"user_id": user_id, "plan": "pro"}

        await gantry.sync()

        bridge = GantryToolBridge(gantry, score_threshold=0.0)
        tools1 = await bridge.get_tools("user profile", limit=5, score_threshold=0.0)
        tools2 = await bridge.get_tools("user profile", limit=5, score_threshold=0.0)

        # Same callable object should be returned due to caching
        assert tools1[0] is tools2[0]

    @pytest.mark.asyncio
    async def test_bridge_clear_cache(self) -> None:
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def get_user_profile(user_id: str) -> dict[str, str]:
            """Fetch a user's profile from the CRM system."""
            return {"user_id": user_id, "plan": "pro"}

        await gantry.sync()

        bridge = GantryToolBridge(gantry, score_threshold=0.0)
        tools1 = await bridge.get_tools("user profile", limit=5, score_threshold=0.0)
        bridge.clear_cache()
        tools2 = await bridge.get_tools("user profile", limit=5, score_threshold=0.0)

        # Different callable objects after cache clear
        assert tools1[0] is not tools2[0]

    @pytest.mark.asyncio
    async def test_bridge_wrap_tools_directly(self) -> None:
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def get_user_profile(user_id: str) -> dict[str, str]:
            """Fetch a user's profile from the CRM system."""
            return {"user_id": user_id, "plan": "pro"}

        await gantry.sync()

        bridge = GantryToolBridge(gantry)
        tool_defs = gantry.export_tools()
        wrapped = bridge.wrap_tools(tool_defs)

        assert len(wrapped) == 1
        assert wrapped[0].__name__ == "get_user_profile"

    @pytest.mark.asyncio
    async def test_bridge_wrap_single(self) -> None:
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def get_user_profile(user_id: str) -> dict[str, str]:
            """Fetch a user's profile from the CRM system."""
            return {"user_id": user_id, "plan": "pro"}

        await gantry.sync()

        bridge = GantryToolBridge(gantry)
        tool_def = gantry.export_tools()[0]
        wrapped = bridge.wrap_single(tool_def)

        assert callable(wrapped)
        assert wrapped.__name__ == "get_user_profile"

    @pytest.mark.asyncio
    async def test_bridge_get_tools_with_scores(self) -> None:
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def get_user_profile(user_id: str) -> dict[str, str]:
            """Fetch a user's profile from the CRM system."""
            return {"user_id": user_id, "plan": "pro"}

        await gantry.sync()

        bridge = GantryToolBridge(gantry, score_threshold=0.0)
        results = await bridge.get_tools_with_scores(
            "user profile", limit=5, score_threshold=0.0
        )

        assert len(results) >= 1
        tool, score = results[0]
        assert callable(tool)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Framework adapters entry
# ---------------------------------------------------------------------------


class TestFrameworkAdaptersEntry:
    """Verify agent_framework is a supported framework in fetch_framework_tools."""

    @pytest.mark.asyncio
    async def test_fetch_framework_tools_agent_framework(self) -> None:
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.framework_adapters import fetch_framework_tools

        gantry = AgentGantry()

        @gantry.register
        def get_user_profile(user_id: str) -> dict[str, str]:
            """Fetch a user's profile from the CRM system."""
            return {"user_id": user_id, "plan": "pro"}

        await gantry.sync()

        tools = await fetch_framework_tools(
            gantry,
            "user profile",
            framework="agent_framework",
            limit=5,
            score_threshold=0.0,
        )

        assert len(tools) >= 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "get_user_profile"

    @pytest.mark.asyncio
    async def test_unsupported_framework_raises(self) -> None:
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.framework_adapters import fetch_framework_tools

        gantry = AgentGantry()
        with pytest.raises(ValueError, match="Unsupported framework"):
            await fetch_framework_tools(
                gantry, "test", framework="nonexistent"  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Additional edge-case tests requested by reviewers
# ---------------------------------------------------------------------------


class TestGantryToolBridgeEdgeCases:
    """Edge-case and regression tests for the GantryToolBridge."""

    @pytest.mark.asyncio
    async def test_positional_args_single_param(self) -> None:
        """Calling a single-param wrapper positionally should work."""
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def get_user_profile(user_id: str) -> dict[str, str]:
            """Fetch a user's profile from the CRM system."""
            return {"user_id": user_id, "plan": "pro"}

        await gantry.sync()

        bridge = GantryToolBridge(gantry, score_threshold=0.0)
        tools = await bridge.get_tools("user profile", limit=5, score_threshold=0.0)

        # Positional call
        result = await tools[0]("abc123")
        parsed = json.loads(result)
        assert parsed["user_id"] == "abc123"

    @pytest.mark.asyncio
    async def test_positional_args_multi_param(self) -> None:
        """Calling a multi-param wrapper positionally should map by order."""
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def search_products(query: str, category: str) -> str:
            """Search for products by query and category in the catalog."""
            return f"Found 5 products matching '{query}' in '{category}'"

        await gantry.sync()

        bridge = GantryToolBridge(gantry, score_threshold=0.0)
        tools = await bridge.get_tools("search products", limit=5, score_threshold=0.0)

        result = await tools[0]("shoes", "footwear")
        assert "shoes" in result
        assert "footwear" in result

    @pytest.mark.asyncio
    async def test_positional_args_too_many_raises(self) -> None:
        """Passing more positional args than parameters should raise TypeError."""
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def get_user_profile(user_id: str) -> dict[str, str]:
            """Fetch a user's profile from the CRM system."""
            return {"user_id": user_id, "plan": "pro"}

        await gantry.sync()

        bridge = GantryToolBridge(gantry, score_threshold=0.0)
        tools = await bridge.get_tools("user profile", limit=5, score_threshold=0.0)

        with pytest.raises(TypeError, match="takes at most 1"):
            await tools[0]("abc", "extra_arg")

    @pytest.mark.asyncio
    async def test_no_var_keyword_in_signature(self) -> None:
        """Wrapper signatures should not expose **kwargs to AF introspection."""
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def get_user_profile(user_id: str) -> dict[str, str]:
            """Fetch a user's profile from the CRM system."""
            return {"user_id": user_id, "plan": "pro"}

        await gantry.sync()

        bridge = GantryToolBridge(gantry, score_threshold=0.0)
        tools = await bridge.get_tools("user profile", limit=5, score_threshold=0.0)

        sig = inspect.signature(tools[0])
        var_keyword_params = [
            p for p in sig.parameters.values()
            if p.kind == inspect.Parameter.VAR_KEYWORD
        ]
        assert len(var_keyword_params) == 0, (
            "Wrapper should not expose **kwargs in its signature"
        )

    @pytest.mark.asyncio
    async def test_wrap_tools_cache_bypass(self) -> None:
        """wrap_tools with cache=False should create fresh wrappers."""
        from agent_gantry import AgentGantry
        from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

        gantry = AgentGantry()

        @gantry.register
        def get_user_profile(user_id: str) -> dict[str, str]:
            """Fetch a user's profile from the CRM system."""
            return {"user_id": user_id, "plan": "pro"}

        await gantry.sync()

        bridge = GantryToolBridge(gantry)
        tool_defs = gantry.export_tools()
        wrapped1 = bridge.wrap_tools(tool_defs, cache=True)
        wrapped2 = bridge.wrap_tools(tool_defs, cache=False)

        # With cache=False, should get a new wrapper object
        assert wrapped1[0] is not wrapped2[0]

    def test_adapter_logs_warning_on_malformed_json(self, caplog: Any) -> None:
        """AgentFrameworkAdapter should log a warning for malformed arguments."""
        import logging as _logging

        adapter = AgentFrameworkAdapter()
        with caplog.at_level(_logging.WARNING):
            payload = adapter.from_provider_payload(
                {"name": "test", "arguments": "not valid json{"}
            )
        assert payload.arguments == {}
        assert payload.tool_name == "test"
        assert "malformed JSON" in caplog.text
