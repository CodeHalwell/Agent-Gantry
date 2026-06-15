"""Tests for the smolagents framework adapter.

Smolagents is not installed in the test environment, so a minimal fake
``smolagents`` module is injected into ``sys.modules``. The fake ``Tool`` base
class provides the ``__init__`` (setting ``is_initialized``) that the adapter's
dynamically built subclass relies on, letting us assert the adapter wired
name / description / inputs / output_type / forward correctly.
"""

from __future__ import annotations

import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.smolagents import SmolagentsAdapter


@pytest.fixture
async def gantry() -> AgentGantry:
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["email"])
    def send_email(to: str, body: str = "") -> str:
        "Send an email message to a recipient."
        return f"sent:{to}"

    @g.register(tags=["math"])
    def add(a: int, b: int) -> int:
        "Add two integers together."
        return a + b

    await g.sync()
    return g


@pytest.fixture
def fake_smolagents(monkeypatch: pytest.MonkeyPatch) -> type:
    """Inject a fake ``smolagents`` module exposing a minimal ``Tool`` base."""

    class Tool:
        def __init__(self) -> None:
            self.is_initialized = True

    module = types.ModuleType("smolagents")
    module.Tool = Tool
    monkeypatch.setitem(sys.modules, "smolagents", module)
    return Tool


async def _first_spec(gantry: AgentGantry):
    from agent_gantry.integrations.frameworks.base import GantryToolset

    specs = await GantryToolset(gantry).select("send an email", limit=5)
    by_name = {s.name: s for s in specs}
    assert "send_email" in by_name, f"send_email not selected: {list(by_name)}"
    return by_name["send_email"]


async def test_spec_to_smolagents_captures_metadata(
    gantry: AgentGantry, fake_smolagents: type
) -> None:
    spec = await _first_spec(gantry)
    tool = SmolagentsAdapter.convert(spec)

    assert tool.name == "send_email"
    assert tool.description == "Send an email message to a recipient."
    assert tool.output_type == "string"
    assert tool.is_initialized is True


async def test_spec_to_smolagents_inputs_shape(
    gantry: AgentGantry, fake_smolagents: type
) -> None:
    spec = await _first_spec(gantry)
    tool = SmolagentsAdapter.convert(spec)

    assert isinstance(tool.inputs, dict)
    assert tool.inputs, "inputs should not be empty"
    for value in tool.inputs.values():
        assert "type" in value
        assert "description" in value
    assert "to" in tool.inputs


def test_spec_to_smolagents_forward_executes(fake_smolagents: type) -> None:
    # ``forward`` routes through the synchronous ``ToolSpec.invoke``, which must
    # run outside a live event loop — so this test is intentionally non-async and
    # builds its own gantry rather than using the async ``gantry`` fixture.
    import asyncio

    async def _build_spec():
        g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

        @g.register(tags=["email"])
        def send_email(to: str, body: str = "") -> str:
            "Send an email message to a recipient."
            return f"sent:{to}"

        await g.sync()
        return await _first_spec(g)

    spec = asyncio.run(_build_spec())
    tool = SmolagentsAdapter.convert(spec)

    assert tool.forward(to="boss@x.com") == "sent:boss@x.com"


async def test_for_smolagents_maps_all_specs(
    gantry: AgentGantry, fake_smolagents: type
) -> None:
    tools = await SmolagentsAdapter(gantry).select("send an email", limit=5)

    assert len(tools) >= 1
    names = {t.name for t in tools}
    assert "send_email" in names
    for t in tools:
        assert t.output_type == "string"
        assert isinstance(t.inputs, dict)


def test_spec_to_smolagents_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the fake module installed, a helpful ImportError is raised."""
    monkeypatch.setitem(sys.modules, "smolagents", None)

    class _DummySpec:
        name = "x"
        description = "d"
        parameters = {"type": "object", "properties": {}}

        def invoke(self, **kwargs: object) -> object:  # pragma: no cover
            return None

    with pytest.raises(ImportError, match="pip install smolagents"):
        SmolagentsAdapter.convert(_DummySpec())
