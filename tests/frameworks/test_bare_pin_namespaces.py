"""Bare-name pins resolve to the in-scope registration under a namespace filter.

With the same tool name in two namespaces, ``required=["ping"]`` and
``namespaces=["b"]`` must pin ``b.ping`` everywhere: in selection, in each
per-turn provider's catalogue, and in its retrieval-failure fallback.
"""

from __future__ import annotations

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.frameworks.base import GantryToolset, _resolve_pins


@pytest.fixture
async def gantry():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(namespace="a")
    def ping() -> str:
        """Ping service A."""
        return "a"

    @g.register(namespace="b", name="ping")
    def ping_b() -> str:
        """Ping service B."""
        return "b"

    await g.sync()
    try:
        yield g
    finally:
        await g.close()


def test_resolve_pins_prefers_the_in_scope_registration(gantry):
    [pin] = _resolve_pins(gantry, [], required=["ping"], always_include=None, namespaces=["b"])
    assert pin._namespace == "b"
    [pin] = _resolve_pins(gantry, [], required=["ping"], always_include=None)
    assert pin._namespace == "a"  # unfiltered: first registration, as before
    [pin] = _resolve_pins(gantry, [], required=["a.ping"], always_include=None, namespaces=["b"])
    assert pin._namespace == "a"  # qualified pins are unaffected


async def test_blank_query_selection_pins_the_in_scope_tool(gantry):
    specs = await GantryToolset(gantry).select_or_empty("", namespaces=["b"], required=["ping"])
    assert [(s._namespace, s.name) for s in specs] == [("b", "ping")]


async def test_langgraph_catalogue_and_fallback_use_the_in_scope_pin(gantry):
    pytest.importorskip("langchain.agents")
    from agent_gantry.integrations.frameworks.langgraph_live import _catalogue

    specs, pinned = await _catalogue(gantry, namespaces=["b"], required=["ping"])
    assert [(s._namespace, s.name) for s in specs] == [("b", "ping")]
    assert pinned == {specs[0].native_name}


async def test_adk_fallback_keeps_same_named_pins_distinct(gantry):
    pytest.importorskip("google.adk")
    from types import SimpleNamespace

    from agent_gantry.integrations.frameworks.google_adk_live import _Selector

    selector = _Selector(
        gantry,
        limit=1,
        score_threshold=0.0,
        namespaces=None,
        required=["a.ping", "b.ping"],
        always_include=None,
        sticky=True,
        max_offered=3,
        query_from_results=True,
    )

    async def down(query):
        raise RuntimeError("vector store down")

    gantry.retrieve = down
    context = SimpleNamespace(invocation_id="inv-1", session=None, user_content="ping")
    specs = await selector.select(context)
    assert sorted((s._namespace, s.name) for s in specs) == [("a", "ping"), ("b", "ping")]
    assert len({s.native_name for s in specs}) == 2
