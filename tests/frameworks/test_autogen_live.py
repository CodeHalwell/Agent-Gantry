"""Tests for the deep, per-turn AutoGen (autogen-core) dynamic-tool provider.

Exercises :class:`GantryWorkbench` against the *real* installed ``autogen_core``
``Workbench`` contract: it must be a usable ``Workbench`` whose ``list_tools``
re-selects from Gantry each call (so the available tools change turn-to-turn as
the query changes) and whose ``call_tool`` routes execution through Gantry and
returns a native ``ToolResult``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("autogen_core")

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.autogen import AutoGenAdapter
from agent_gantry.integrations.frameworks.autogen_live import GantryWorkbench


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


async def test_workbench_is_real_autogen_workbench(gantry):
    from autogen_core.tools import Workbench

    wb = AutoGenAdapter(gantry).workbench(limit=5)
    assert isinstance(wb, Workbench)
    assert isinstance(wb, GantryWorkbench)


async def test_list_tools_reselects_per_turn(gantry):
    from autogen_core.tools import ToolResult  # noqa: F401 - sanity that types load

    wb = AutoGenAdapter(gantry).workbench(limit=2)

    # Turn 1: a weather-focused query surfaces the weather tool.
    wb.set_query("what is the weather forecast for the city")
    weather_schemas = await wb.list_tools()
    weather_names = {s["name"] for s in weather_schemas}
    assert "get_weather" in weather_names

    weather_schema = next(s for s in weather_schemas if s["name"] == "get_weather")
    assert weather_schema["description"] == "Get the current weather forecast for a city."
    assert "city" in weather_schema["parameters"]["properties"]
    assert "city" in weather_schema["parameters"]["required"]

    # Turn 2: switching the query re-selects and now surfaces the email tool.
    wb.query = "send an email message to a recipient"
    email_schemas = await wb.list_tools()
    email_names = {s["name"] for s in email_schemas}
    assert "send_email" in email_names


async def test_call_tool_returns_tool_result_from_gantry(gantry):
    from autogen_core.tools import ToolResult

    wb = AutoGenAdapter(gantry).workbench(limit=3)
    wb.set_query("what is the weather forecast for the city")
    await wb.list_tools()

    result = await wb.call_tool("get_weather", {"city": "Paris"})

    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert result.name == "get_weather"
    # Content reflects the actual gantry execution.
    assert result.to_text() == "weather:Paris:sunny"


async def test_call_tool_unknown_returns_error_result(gantry):
    from autogen_core.tools import ToolResult

    wb = AutoGenAdapter(gantry).workbench(limit=2)
    wb.set_query("send an email message to a recipient")

    result = await wb.call_tool("does_not_exist", {})
    assert isinstance(result, ToolResult)
    assert result.is_error is True


async def test_lifecycle_methods_are_safe(gantry):
    wb = AutoGenAdapter(gantry).workbench(query="weather", limit=1)
    await wb.start()
    state = await wb.save_state()
    assert state["query"] == "weather"

    wb2 = AutoGenAdapter(gantry).workbench(limit=1)
    await wb2.load_state(state)
    assert wb2.query == "weather"

    await wb.reset()
    await wb.stop()
