"""
Tests for the LangChain bridge.

These tests stub out ``langchain_core`` and ``langchain`` so the suite runs
even when the framework is not installed, mirroring the existing pattern
used for the Microsoft Agent Framework bridge tests.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from agent_gantry import AgentGantry
from agent_gantry.core.security import (
    ConfirmationRequiredError,
    PermissionDeniedError,
    SecurityPolicy,
)
from agent_gantry.schema.tool import ToolCapability, ToolDefinition

# ---------------------------------------------------------------------------
# langchain_core stubs
# ---------------------------------------------------------------------------


class _StubStructuredTool:
    """Minimal stand-in for langchain_core.tools.StructuredTool.

    Captures the kwargs passed to ``from_function`` so tests can assert on
    name/description/args_schema and invoke the wrapped coroutine.
    """

    def __init__(
        self,
        *,
        func: Any,
        coroutine: Any,
        name: str,
        description: str,
        args_schema: Any,
    ) -> None:
        self.func = func
        self.coroutine = coroutine
        self.name = name
        self.description = description
        self.args_schema = args_schema

    @classmethod
    def from_function(
        cls,
        *,
        func: Any,
        coroutine: Any,
        name: str,
        description: str,
        args_schema: Any,
    ) -> _StubStructuredTool:
        return cls(
            func=func,
            coroutine=coroutine,
            name=name,
            description=description,
            args_schema=args_schema,
        )


@pytest.fixture(autouse=True)
def stub_langchain_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install langchain_core / langchain_core.tools stubs for every test.

    The bridge imports ``langchain_core`` for the install check and
    ``langchain_core.tools.StructuredTool`` for the actual wrapping. Both are
    replaced with stub modules so we can run without the real package.
    """
    langchain_core = ModuleType("langchain_core")
    tools_mod = ModuleType("langchain_core.tools")
    tools_mod.StructuredTool = _StubStructuredTool
    langchain_core.tools = tools_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", tools_mod)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _populate_gantry(gantry: AgentGantry) -> ToolDefinition:
    """Register a single benign tool and return its definition."""
    weather_def = ToolDefinition(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {
                    "type": "string",
                    "description": "Temperature units",
                    "default": "celsius",
                },
            },
            "required": ["city"],
        },
    )
    await gantry.add_tool(weather_def)
    return weather_def


async def _populate_destructive_gantry(gantry: AgentGantry) -> ToolDefinition:
    """Register a destructive tool that requires approval by capability."""
    delete_def = ToolDefinition(
        name="delete_account",
        description="Permanently delete a user account.",
        parameters_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User to delete"},
            },
            "required": ["user_id"],
        },
        capabilities=[ToolCapability.DELETE_DATA],
    )
    await gantry.add_tool(delete_def)
    return delete_def


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrap_tool_builds_structured_tool_with_metadata() -> None:
    from agent_gantry.integrations.langchain_bridge import GantryToolBridge

    gantry = AgentGantry()
    weather_def = await _populate_gantry(gantry)
    bridge = GantryToolBridge(gantry)

    tool = bridge.wrap_tool(weather_def)

    assert isinstance(tool, _StubStructuredTool)
    assert tool.name == "get_weather"
    assert tool.description == weather_def.description
    # args_schema must be a Pydantic model with the expected fields
    fields = tool.args_schema.model_fields
    assert "city" in fields
    assert "units" in fields
    # Required field surfaced as required
    assert fields["city"].is_required()
    # Optional with default surfaces with the default value
    assert not fields["units"].is_required()


@pytest.mark.asyncio
async def test_wrap_tool_caches_repeated_calls() -> None:
    from agent_gantry.integrations.langchain_bridge import GantryToolBridge

    gantry = AgentGantry()
    weather_def = await _populate_gantry(gantry)
    bridge = GantryToolBridge(gantry)

    first = bridge.wrap_tool(weather_def)
    second = bridge.wrap_tool(weather_def)
    assert first is second

    bridge.clear_cache()
    third = bridge.wrap_tool(weather_def)
    assert third is not first


@pytest.mark.asyncio
async def test_get_tools_returns_top_k_langchain_tools() -> None:
    from agent_gantry.integrations.langchain_bridge import GantryToolBridge

    gantry = AgentGantry()
    await _populate_gantry(gantry)
    await gantry.add_tool(
        ToolDefinition(
            name="send_email",
            description="Send an email to a recipient.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required": ["to", "body"],
            },
        )
    )

    bridge = GantryToolBridge(gantry)
    tools = await bridge.get_tools(
        "what is the weather in Paris", limit=1, score_threshold=0.0
    )

    assert len(tools) == 1
    assert tools[0].name == "get_weather"


@pytest.mark.asyncio
async def test_wrapped_tool_executes_via_gantry(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_gantry.integrations.langchain_bridge import GantryToolBridge

    gantry = AgentGantry()
    weather_def = await _populate_gantry(gantry)

    captured: dict[str, Any] = {}

    class _FakeStatus:
        value = "success"

    class _FakeResult:
        status = _FakeStatus()
        result = "Sunny, 22C"
        error = None

    async def fake_execute(self: AgentGantry, call: Any) -> Any:
        captured["call"] = call
        return _FakeResult()

    monkeypatch.setattr(AgentGantry, "execute", fake_execute, raising=False)

    bridge = GantryToolBridge(gantry)
    tool = bridge.wrap_tool(weather_def)

    output = await tool.coroutine(city="Paris")

    assert output == "Sunny, 22C"
    assert captured["call"].tool_name == "get_weather"
    assert captured["call"].arguments == {"city": "Paris"}


@pytest.mark.asyncio
async def test_wrapped_tool_raises_on_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_gantry.integrations.langchain_bridge import GantryToolBridge

    gantry = AgentGantry()
    weather_def = await _populate_gantry(gantry)

    class _FailStatus:
        value = "error"

    class _FailResult:
        status = _FailStatus()
        result = None
        error = "boom"

    async def fake_execute(self: AgentGantry, call: Any) -> Any:
        return _FailResult()

    monkeypatch.setattr(AgentGantry, "execute", fake_execute, raising=False)

    bridge = GantryToolBridge(gantry)
    tool = bridge.wrap_tool(weather_def)

    with pytest.raises(RuntimeError, match="get_weather failed: boom"):
        await tool.coroutine(city="Paris")


@pytest.mark.asyncio
async def test_security_policy_denial_blocks_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_gantry.integrations.langchain_bridge import GantryToolBridge

    gantry = AgentGantry()

    delete_def = ToolDefinition(
        name="delete_records",
        description="Delete records.",
        parameters_schema={
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name"},
            },
            "required": ["table"],
        },
    )
    await gantry.add_tool(delete_def)

    async def should_not_execute(self: AgentGantry, call: Any) -> Any:
        raise AssertionError("execute should not run when policy denies")

    monkeypatch.setattr(AgentGantry, "execute", should_not_execute, raising=False)

    policy = SecurityPolicy(require_confirmation=["delete_*"])
    bridge = GantryToolBridge(gantry, security_policy=policy)
    tool = bridge.wrap_tool(delete_def)

    with pytest.raises(ConfirmationRequiredError):
        await tool.coroutine(table="users")


@pytest.mark.asyncio
async def test_security_policy_with_approval_callback_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_gantry.integrations.langchain_bridge import GantryToolBridge

    gantry = AgentGantry()
    delete_def = await _populate_destructive_gantry(gantry)

    seen: list[tuple[str, dict[str, Any]]] = []

    class _Status:
        value = "success"

    class _OK:
        status = _Status()
        result = "deleted"
        error = None

    async def fake_execute(self: AgentGantry, call: Any) -> Any:
        return _OK()

    monkeypatch.setattr(AgentGantry, "execute", fake_execute, raising=False)

    async def approve(td: ToolDefinition, args: dict[str, Any]) -> bool:
        seen.append((td.name, args))
        return True

    bridge = GantryToolBridge(gantry, approval_callback=approve)
    tool = bridge.wrap_tool(delete_def)

    output = await tool.coroutine(user_id="abc")

    assert output == "deleted"
    assert seen == [("delete_account", {"user_id": "abc"})]


@pytest.mark.asyncio
async def test_capability_approval_callback_denial_blocks_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_gantry.integrations.langchain_bridge import GantryToolBridge

    gantry = AgentGantry()
    delete_def = await _populate_destructive_gantry(gantry)

    async def should_not_execute(self: AgentGantry, call: Any) -> Any:
        raise AssertionError("execute should not run when approval is denied")

    monkeypatch.setattr(AgentGantry, "execute", should_not_execute, raising=False)

    def deny(td: ToolDefinition, args: dict[str, Any]) -> bool:
        return False

    bridge = GantryToolBridge(gantry, approval_callback=deny)
    tool = bridge.wrap_tool(delete_def)

    with pytest.raises(PermissionDeniedError):
        await tool.coroutine(user_id="abc")


@pytest.mark.asyncio
async def test_capability_approval_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``capability_approval=False``, destructive caps don't gate execution."""
    from agent_gantry.integrations.langchain_bridge import GantryToolBridge

    gantry = AgentGantry()
    delete_def = await _populate_destructive_gantry(gantry)

    class _Status:
        value = "success"

    class _OK:
        status = _Status()
        result = "deleted"
        error = None

    async def fake_execute(self: AgentGantry, call: Any) -> Any:
        return _OK()

    monkeypatch.setattr(AgentGantry, "execute", fake_execute, raising=False)

    bridge = GantryToolBridge(gantry, capability_approval=False)
    tool = bridge.wrap_tool(delete_def)

    output = await tool.coroutine(user_id="abc")
    assert output == "deleted"


@pytest.mark.asyncio
async def test_build_agent_passes_tools_to_create_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``build_agent`` should call ``langchain.agents.create_agent`` with retrieved tools."""
    from agent_gantry.integrations.langchain_bridge import GantryToolBridge

    # Stub langchain.agents.create_agent
    langchain_pkg = ModuleType("langchain")
    agents_mod = ModuleType("langchain.agents")

    captured: dict[str, Any] = {}

    def fake_create_agent(*, model: Any, tools: list[Any], **kwargs: Any) -> str:
        captured["model"] = model
        captured["tools"] = tools
        captured["kwargs"] = kwargs
        return "fake-agent"

    agents_mod.create_agent = fake_create_agent
    langchain_pkg.agents = agents_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain", langchain_pkg)
    monkeypatch.setitem(sys.modules, "langchain.agents", agents_mod)

    gantry = AgentGantry()
    await _populate_gantry(gantry)

    bridge = GantryToolBridge(gantry)
    agent = await bridge.build_agent(
        "openai:gpt-4o",
        "what is the weather in Paris",
        limit=1,
        score_threshold=0.0,
        prompt="You are a helpful assistant.",
    )

    assert agent == "fake-agent"
    assert captured["model"] == "openai:gpt-4o"
    assert len(captured["tools"]) == 1
    assert captured["tools"][0].name == "get_weather"
    assert captured["kwargs"] == {"prompt": "You are a helpful assistant."}


@pytest.mark.asyncio
async def test_get_tools_with_scores_returns_pairs() -> None:
    from agent_gantry.integrations.langchain_bridge import GantryToolBridge

    gantry = AgentGantry()
    await _populate_gantry(gantry)

    bridge = GantryToolBridge(gantry)
    pairs = await bridge.get_tools_with_scores(
        "what is the weather in Paris", limit=1, score_threshold=0.0
    )

    assert len(pairs) == 1
    tool, score = pairs[0]
    assert tool.name == "get_weather"
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
