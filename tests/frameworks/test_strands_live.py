"""Tests for the deep, per-turn Strands Agents dynamic-tool hook.

Exercises :class:`GantryStrandsToolHook` against the *real* installed
``strands`` package. Strands fires a ``BeforeModelCallEvent`` immediately
before every model call and only reads ``agent.tool_registry`` for the tool
specs sent to the model *afterward* — so a hook that swaps the registry during
that event changes what the model would see on the very call about to happen.
We never invoke a real model here (no API keys / AWS credentials required):
constructing a ``strands.Agent`` does not require model credentials (the
default Bedrock model client is only touched on an actual inference call), and
the hook is driven directly by building a real ``BeforeModelCallEvent`` and
awaiting the hook's callback, mirroring how ``test_pydantic_ai_live.py``
drives ``AbstractToolset`` with a hand-built ``RunContext``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("strands")

from strands import Agent
from strands.hooks import BeforeModelCallEvent

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.strands import StrandsAdapter


@pytest.fixture
async def gantry():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["weather", "forecast"])
    def get_weather(city: str) -> str:
        "Get the current weather forecast for a city."
        return f"weather:{city}:sunny"

    @g.register(tags=["email", "communication"])
    def send_email(to: str, body: str = "") -> str:
        "Send an email message to a recipient."
        return f"sent:{to}"

    @g.register(tags=["math", "arithmetic"])
    def add(a: int, b: int) -> int:
        "Add two integers together."
        return a + b

    await g.sync()
    return g


def _agent_with_user_text(text: str) -> Agent:
    """A real ``strands.Agent`` whose message history ends with a user turn.

    ``callback_handler=None`` suppresses Strands' default console printer;
    no model call is made anywhere in this module.
    """
    return Agent(
        tools=[],
        callback_handler=None,
        messages=[{"role": "user", "content": [{"text": text}]}],
    )


async def _fire_before_model_call(hook, agent: Agent) -> None:
    event = BeforeModelCallEvent(agent=agent, invocation_state={}, projected_input_tokens=None)
    await hook._on_before_model_call(event)


async def test_hook_is_real_hook_provider(gantry):
    from strands.hooks import HookProvider

    hook = StrandsAdapter(gantry).tool_hook(limit=2)
    assert isinstance(hook, HookProvider)


async def test_agent_construction_wires_the_hook(gantry):
    """``StrandsAdapter(gantry).agent(...)`` builds a real Agent with no static tools."""
    agent = StrandsAdapter(gantry).agent(limit=2, callback_handler=None)

    assert isinstance(agent, Agent)
    # No statically registered tools; the hook injects them per turn.
    assert list(agent.tool_registry.registry) == []
    assert agent.hooks.has_callbacks()


async def test_before_model_call_injects_tools_for_weather_turn(gantry):
    hook = StrandsAdapter(gantry).tool_hook(limit=2)
    agent = _agent_with_user_text("what is the weather forecast for the city")

    await _fire_before_model_call(hook, agent)

    names = set(agent.tool_registry.registry)
    assert "get_weather" in names

    specs = {spec["name"]: spec for spec in agent.tool_registry.get_all_tool_specs()}
    assert "get_weather" in specs
    assert specs["get_weather"]["description"] == "Get the current weather forecast for a city."
    schema = specs["get_weather"]["inputSchema"]["json"]
    assert "city" in schema["properties"]
    assert "city" in schema["required"]


async def test_before_model_call_reselects_across_turns(gantry):
    """Each model call re-derives the query from the latest message and swaps tools."""
    hook = StrandsAdapter(gantry).tool_hook(limit=1)

    weather_agent = _agent_with_user_text("weather forecast for the city")
    await _fire_before_model_call(hook, weather_agent)
    assert set(weather_agent.tool_registry.registry) == {"get_weather"}

    # Same hook, new turn: append an email message and fire again on the SAME
    # agent object — the stale weather tool must be retracted.
    weather_agent.messages.append(
        {"role": "user", "content": [{"text": "send an email to a recipient"}]}
    )
    await _fire_before_model_call(hook, weather_agent)

    names = set(weather_agent.tool_registry.registry)
    assert names == {"send_email"}, f"expected only send_email, got {names}"


async def test_tool_call_through_swapped_registry_executes_via_gantry(gantry):
    """A tool injected by the hook actually routes execution through gantry.execute."""
    hook = StrandsAdapter(gantry).tool_hook(limit=1)
    agent = _agent_with_user_text("send an email to a recipient")

    await _fire_before_model_call(hook, agent)

    tool = agent.tool_registry.registry["send_email"]
    result = await tool(to="boss@x.com")
    assert result == "sent:boss@x.com"


async def test_empty_query_leaves_registry_untouched(gantry):
    hook = StrandsAdapter(gantry).tool_hook(limit=2)
    agent = Agent(tools=[], callback_handler=None, messages=[])

    await _fire_before_model_call(hook, agent)

    assert list(agent.tool_registry.registry) == []


async def test_empty_query_retracts_stale_tools(gantry):
    """A turn with no query signal retracts the previous turn's tools.

    Regression test: an early return on an empty query used to leave the
    previous turn's tools registered in Strands' stateful tool_registry
    instead of retracting them — inconsistent with the other stateful live
    providers (e.g. a plugin-refresh provider clears its plugin on a blank query).
    """
    hook = StrandsAdapter(gantry).tool_hook(limit=1)
    agent = _agent_with_user_text("weather forecast for the city")
    await _fire_before_model_call(hook, agent)
    assert set(agent.tool_registry.registry) == {"get_weather"}

    agent.messages.clear()
    await _fire_before_model_call(hook, agent)
    assert list(agent.tool_registry.registry) == []


async def test_stream_absorbs_tool_failure_into_error_tool_result(gantry):
    """Documents a deliberate deviation that lives in Strands' own code, not ours.

    ``DecoratedFunctionTool.__call__`` (proven below to still raise directly)
    is a bypass for manual/direct calls, not what Strands' own agent event
    loop uses for a model-issued tool call -- that's ``.stream()``, which
    wraps the call in a bare ``except Exception`` and converts it into an
    error ``ToolResult`` (``status="error"``) instead of raising. That is
    Strands' own native tool-execution contract (every function-based tool
    behaves this way, not just Gantry's), not overridden by this adapter --
    see "Error-handling policy" in ``integrations/frameworks/README.md``.
    """
    from agent_gantry.integrations.frameworks.base import ToolExecutionError
    from agent_gantry.strands import StrandsAdapter

    @gantry.register(tags=["danger"])
    def explode() -> str:
        "Always raises."
        raise RuntimeError("boom")

    await gantry.sync()

    tools = await StrandsAdapter(gantry).select("always raises", limit=1)
    tool = tools[0]

    # __call__ bypasses Strands' own tool-use dispatch -- still raises directly.
    with pytest.raises(ToolExecutionError):
        await tool()

    # .stream() is what the real Strands agent loop calls for a model-issued
    # tool call -- it absorbs the error into a ToolResult instead of raising.
    tool_use = {"toolUseId": "t1", "name": "explode", "input": {}}
    events = [event async for event in tool.stream(tool_use, {})]
    result_event = events[-1]

    assert result_event.tool_result["status"] == "error"
    assert "boom" in result_event.tool_result["content"][0]["text"]
    assert isinstance(result_event.exception, ToolExecutionError)
