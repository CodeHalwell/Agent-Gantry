"""Tests for the deep, per-turn Pydantic AI (pydantic-ai) dynamic-tool provider.

Exercises :class:`GantryToolset` against the *real* installed ``pydantic_ai``
:class:`~pydantic_ai.toolsets.AbstractToolset` contract: it must be a usable
toolset whose ``get_tools`` re-selects from Gantry each run (so the available
tool definitions change run-to-run as the query changes) and whose ``call_tool``
routes execution through Gantry and returns the tool's result.

A minimal-but-real ``RunContext`` is constructed (deps/model/usage plus the
run's ``prompt``/``messages``) so the context-derived per-turn query path is
exercised end to end; an explicit ``set_query`` override is also covered.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.usage import RunUsage

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.frameworks.pydantic_ai_live import GantryToolset
from agent_gantry.pydantic_ai import PydanticAIAdapter


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


def _ctx(prompt: str) -> RunContext:
    """Build a minimal real ``RunContext`` carrying the run's user prompt."""
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=prompt,
        messages=[ModelRequest(parts=[UserPromptPart(content=prompt)])],
    )


async def test_toolset_is_real_abstract_toolset(gantry):
    ts = PydanticAIAdapter(gantry).toolset(limit=5)
    assert isinstance(ts, AbstractToolset)
    assert isinstance(ts, GantryToolset)
    # Required abstract member.
    assert ts.id == "agent-gantry"


async def test_get_tools_reselects_per_turn_from_context(gantry):
    ts = PydanticAIAdapter(gantry).toolset(limit=2)

    # Run 1: a weather-focused prompt surfaces the weather tool def.
    weather_tools = await ts.get_tools(_ctx("what is the weather forecast for the city"))
    assert "get_weather" in weather_tools

    weather_tool = weather_tools["get_weather"]
    assert isinstance(weather_tool, ToolsetTool)
    assert isinstance(weather_tool.tool_def, ToolDefinition)
    assert weather_tool.tool_def.description == "Get the current weather forecast for a city."
    schema = weather_tool.tool_def.parameters_json_schema
    assert "city" in schema["properties"]
    assert "city" in schema["required"]

    # Run 2: switching the prompt re-selects and now surfaces the email tool def
    # (per-turn re-selection driven purely by the run context).
    email_tools = await ts.get_tools(_ctx("send an email message to a recipient"))
    assert "send_email" in email_tools
    assert isinstance(email_tools["send_email"], ToolsetTool)


async def test_get_tools_honours_explicit_query_override(gantry):
    ts = PydanticAIAdapter(gantry).toolset(limit=2)
    ts.set_query("send an email message to a recipient")

    # The explicit override wins even though the context prompt is about math.
    tools = await ts.get_tools(_ctx("add two integers"))
    assert "send_email" in tools


async def test_call_tool_executes_through_gantry(gantry):
    ts = PydanticAIAdapter(gantry).toolset(limit=3)
    ctx = _ctx("what is the weather forecast for the city")
    tools = await ts.get_tools(ctx)

    result = await ts.call_tool("get_weather", {"city": "Paris"}, ctx, tools["get_weather"])

    # The result reflects the actual gantry execution.
    assert result == "weather:Paris:sunny"


async def test_call_tool_without_prior_get_tools_reselects(gantry):
    ts = PydanticAIAdapter(gantry).toolset(limit=3)
    ctx = _ctx("send an email message to a recipient")

    # No prior get_tools; call_tool runs a fresh selection from the context.
    result = await ts.call_tool("send_email", {"to": "boss@x.com"}, ctx, None)
    assert result == "sent:boss@x.com"
