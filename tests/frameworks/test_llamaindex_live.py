"""Tests for the LlamaIndex *live* per-turn tool provider.

These exercise the real ``llama-index-core`` package (skipped if absent). They
prove the deep integration genuinely re-selects tools on each retriever call —
a weather query surfaces the weather tool, an email query surfaces the email
tool — and that the returned objects are real ``FunctionTool`` instances whose
invocation routes through the gantry.
"""

from __future__ import annotations

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.tools import FunctionTool  # noqa: E402

from agent_gantry import AgentGantry  # noqa: E402
from agent_gantry.adapters.embedders.simple import SimpleEmbedder  # noqa: E402
from agent_gantry.integrations.frameworks.llamaindex_live import (  # noqa: E402
    GantryToolRetriever,
)
from agent_gantry.llamaindex import LlamaIndexAdapter  # noqa: E402


@pytest.fixture
async def gantry() -> AgentGantry:
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["weather"])
    def get_weather(city: str) -> str:
        "Get the current weather forecast for a city."
        return f"weather:{city}"

    @g.register(tags=["email"])
    def send_email(to: str, body: str = "") -> str:
        "Send an email message to a recipient address."
        return f"sent:{to}"

    @g.register(tags=["currency"])
    def convert_currency(amount: float, to_code: str) -> str:
        "Convert a monetary amount into another currency."
        return f"converted:{amount}:{to_code}"

    await g.sync()
    return g


async def test_factory_builds_object_retriever(gantry: AgentGantry) -> None:
    retriever = LlamaIndexAdapter(gantry).tool_retriever(limit=5)
    assert isinstance(retriever, GantryToolRetriever)
    # It is a genuine ObjectRetriever subtype so FunctionAgent accepts it.
    from llama_index.core.objects import ObjectRetriever

    assert isinstance(retriever, ObjectRetriever)


async def test_per_turn_reselection_weather_then_email(
    gantry: AgentGantry,
) -> None:
    """Same retriever, different queries -> different tools (per-turn deep)."""
    retriever = LlamaIndexAdapter(gantry).tool_retriever(limit=1)

    weather_tools = await retriever.aretrieve("what's the weather in Paris?")
    weather_names = {t.metadata.name for t in weather_tools}
    assert "get_weather" in weather_names, weather_names

    email_tools = await retriever.aretrieve("send an email to my boss")
    email_names = {t.metadata.name for t in email_tools}
    assert "send_email" in email_names, email_names

    # Proves it re-selected: the email turn did not return the weather tool.
    assert "get_weather" not in email_names


async def test_returned_objects_are_real_function_tools(
    gantry: AgentGantry,
) -> None:
    retriever = LlamaIndexAdapter(gantry).tool_retriever(limit=5)
    tools = await retriever.aretrieve("what's the weather in Tokyo?")
    assert tools
    for t in tools:
        assert isinstance(t, FunctionTool)


async def test_invocation_routes_through_gantry(gantry: AgentGantry) -> None:
    retriever = LlamaIndexAdapter(gantry).tool_retriever(limit=1)
    tools = await retriever.aretrieve("what's the weather in Berlin?")
    by_name = {t.metadata.name: t for t in tools}
    assert "get_weather" in by_name, list(by_name)

    tool = by_name["get_weather"]
    # async_fn path routes through gantry.execute and returns the tool result.
    out = await tool.async_fn(city="Berlin")
    result = getattr(out, "raw_output", out)
    assert result == "weather:Berlin", result


async def test_sync_retrieve_matches_async(gantry: AgentGantry) -> None:
    retriever = LlamaIndexAdapter(gantry).tool_retriever(limit=1)
    tools = retriever.retrieve("convert 100 USD to EUR")
    names = {t.metadata.name for t in tools}
    assert "convert_currency" in names, names
