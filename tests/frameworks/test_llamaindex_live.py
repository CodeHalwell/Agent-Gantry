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
from agent_gantry.schema.tool import ToolDefinition  # noqa: E402


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


async def test_scalar_coercion_matches_what_the_executor_accepts(
    gantry: AgentGantry,
) -> None:
    """LlamaIndex validates against ``fn_schema`` but forwards the caller's
    *original* values, so a model answering ``"100"`` for a ``number``
    parameter passed the tool's own validation and was then rejected by the
    executor, which holds the caller to the advertised schema. CrewAI forwards
    the validated values and never had this gap (PR #381 review).

    Exercised against the real package because that forwarding behaviour is
    the whole finding — a fake would just pin our assumption about it.
    """
    tools = await LlamaIndexAdapter(gantry).select("convert 100 USD to EUR", limit=1)
    by_name = {t.metadata.name: t for t in tools}
    assert "convert_currency" in by_name, list(by_name)
    tool = by_name["convert_currency"]

    out = tool.call(amount="100", to_code="EUR")
    result = getattr(out, "raw_output", out)
    assert result == "converted:100.0:EUR", result

    out_async = await tool.acall(amount="100", to_code="EUR")
    assert getattr(out_async, "raw_output", out_async) == "converted:100.0:EUR"


async def test_well_typed_values_are_passed_through_unchanged(
    gantry: AgentGantry,
) -> None:
    tools = await LlamaIndexAdapter(gantry).select("convert 100 USD to EUR", limit=1)
    tool = {t.metadata.name: t for t in tools}["convert_currency"]
    out = tool.call(amount=100.0, to_code="EUR")
    assert getattr(out, "raw_output", out) == "converted:100.0:EUR"


async def test_coercion_does_not_inject_defaults_into_typed_map_values() -> None:
    """``model_dump()`` recursively materializes every omitted optional field,
    including inside a typed map's values, where an optional child with no
    schema default becomes ``None``. Executor normalization walks named
    ``properties`` but not schema-valued map entries, so that injected null
    survived to be rejected — turning a call the caller made correctly into an
    error (PR #381 review).

    Exercised through the real adapter for the same reason its sibling above
    is: the dump happens inside ``_coerced``, and asserting on ``model_dump``
    directly would pin the mechanism rather than the path that uses it.

    The schema is written out rather than derived from a Pydantic model,
    because the shape that breaks needs an optional child that is *not*
    nullable. A Python ``b: str | None = None`` emits ``anyOf: [string,
    null]``, which makes the injected null legal and the bug invisible —
    an imported or hand-authored schema is where a non-nullable optional
    field actually occurs.
    """
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    def record_entries(entries: dict) -> str:
        return ",".join(
            f"{k}:{v.get('a')}:{v.get('b')}" for k, v in sorted(entries.items())
        )

    await g.add_tool(
        ToolDefinition(
            name="record_entries",
            description="Record a mapping of named ledger entries.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "entries": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "string"},
                                "b": {"type": "string"},
                            },
                            "required": ["a"],
                        },
                    }
                },
                "required": ["entries"],
            },
            tags=["ledger"],
        ),
        handler=record_entries,
    )
    await g.sync()

    tools = await LlamaIndexAdapter(g).select("record ledger entries", limit=1)
    tool = {t.metadata.name: t for t in tools}["record_entries"]

    # ``b`` is omitted by the caller and must stay omitted rather than being
    # materialized as a null the executor then refuses.
    out = tool.call(entries={"k": {"a": "x"}})
    assert getattr(out, "raw_output", out) == "k:x:None"

    # Supplying it explicitly still works, so the exclusion did not start
    # dropping real values.
    out_both = tool.call(entries={"k": {"a": "x", "b": "y"}})
    assert getattr(out_both, "raw_output", out_both) == "k:x:y"
