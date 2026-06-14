"""Tests for the framework-agnostic multi-turn :class:`ToolRefresher`.

These use a *real* gantry with the hash-based :class:`SimpleEmbedder` and five
tools across clearly distinct domains, then simulate a single agent run as a
growing list of message dicts whose user intent pivots across turns
(weather -> email -> currency). The refresher should track those pivots and
surface a different, on-topic top tool each turn.

SimpleEmbedder is a toy hash-based embedder, so we keep tool descriptions
unambiguously distinct and assert the expected tool is *within* the returned
top-k (limit=3) rather than insisting it is strictly rank 1.
"""

from __future__ import annotations

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.frameworks.base import ToolSpec
from agent_gantry.integrations.refresh import ToolRefresher


@pytest.fixture
async def gantry():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=128))

    @g.register(tags=["weather"])
    def get_weather(city: str) -> str:
        "Get the current weather forecast and temperature for a city."
        return f"weather:{city}"

    @g.register(tags=["email"])
    def send_email(to: str, body: str = "") -> str:
        "Send an email message to a recipient mailbox address."
        return f"email:{to}"

    @g.register(tags=["currency"])
    def convert_currency(amount: float, source: str, target: str) -> str:
        "Convert a money amount between currencies using exchange rates."
        return f"currency:{amount}{source}->{target}"

    @g.register(tags=["todo"])
    def add_todo(task: str) -> str:
        "Add a task item to the personal todo list reminder."
        return f"todo:{task}"

    @g.register(tags=["sql"])
    def run_sql_query(query: str) -> str:
        "Execute a SQL database query and return matching table rows."
        return f"sql:{query}"

    await g.sync()
    return g


def _names(schemas: list[dict]) -> set[str]:
    """Pull tool names out of OpenAI-dialect schema dicts."""
    out: set[str] = set()
    for s in schemas:
        fn = s.get("function", s)
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            out.add(name)
    return out


async def test_refresh_pivots_with_changing_intent(gantry):
    """As the conversation pivots, refresh() surfaces the matching tool."""
    refresher = ToolRefresher(gantry, limit=3, dialect="openai")

    # Turn 1: weather.
    messages = [
        {"role": "user", "content": "What is the weather forecast in Paris today?"}
    ]
    turn1 = await refresher.refresh(messages)
    assert "get_weather" in _names(turn1)

    # Turn 2: pivot to sending an email.
    messages += [
        {"role": "assistant", "content": "It is sunny in Paris."},
        {"role": "user", "content": "Now send an email message to my boss."},
    ]
    turn2 = await refresher.refresh(messages)
    assert "send_email" in _names(turn2)

    # Turn 3: pivot to currency conversion.
    messages += [
        {"role": "assistant", "content": "Email sent."},
        {
            "role": "user",
            "content": "Convert 100 dollars to euros using the exchange rate.",
        },
    ]
    turn3 = await refresher.refresh(messages)
    assert "convert_currency" in _names(turn3)

    # The selection genuinely changed across turns (direction-changing).
    assert _names(turn1) != _names(turn2)
    assert _names(turn2) != _names(turn3)


async def test_tools_used_accumulates_across_turns(gantry):
    """tools_used accumulates as tool-role messages are appended."""
    refresher = ToolRefresher(gantry, limit=3, track_used=True)

    messages = [{"role": "user", "content": "What is the weather in Paris?"}]
    await refresher.refresh(messages)
    assert refresher.tools_used == []

    # Append a tool-role result for the weather tool.
    messages += [
        {"role": "tool", "name": "get_weather", "content": "sunny, 21C"},
        {"role": "user", "content": "Now send an email to my boss."},
    ]
    await refresher.refresh(messages)
    assert refresher.tools_used == ["get_weather"]

    # Append another tool result for the email tool.
    messages += [
        {"role": "tool", "name": "send_email", "content": "delivered"},
        {"role": "user", "content": "Convert 100 dollars to euros."},
    ]
    await refresher.refresh(messages)
    assert refresher.tools_used == ["get_weather", "send_email"]


async def test_track_used_disabled_keeps_empty(gantry):
    """With track_used=False, tools_used stays empty despite tool messages."""
    refresher = ToolRefresher(gantry, limit=3, track_used=False)
    messages = [
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "tool", "name": "get_weather", "content": "sunny"},
    ]
    await refresher.refresh(messages)
    assert refresher.tools_used == []


async def test_refresh_returns_dialect_schemas(gantry):
    """refresh() returns a list of dialect schema dicts."""
    refresher = ToolRefresher(gantry, limit=3, dialect="openai")
    messages = [{"role": "user", "content": "Run a SQL database query for rows."}]
    schemas = await refresher.refresh(messages)

    assert isinstance(schemas, list)
    assert schemas
    assert all(isinstance(s, dict) for s in schemas)
    assert "run_sql_query" in _names(schemas)


async def test_refresh_specs_returns_toolspecs(gantry):
    """refresh_specs() returns ToolSpec objects and sets last_selection."""
    refresher = ToolRefresher(gantry, limit=3)
    messages = [{"role": "user", "content": "Add a task to my todo list reminder."}]
    specs = await refresher.refresh_specs(messages)

    assert isinstance(specs, list)
    assert specs
    assert all(isinstance(s, ToolSpec) for s in specs)
    assert "add_todo" in {s.name for s in specs}

    # last_selection mirrors the most recent specs.
    assert {s.name for s in refresher.last_selection} == {s.name for s in specs}
