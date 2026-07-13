"""Tests for the LlamaIndex framework adapter.

LlamaIndex is not installed in the test environment, so a minimal fake
``llama_index.core.tools`` module is injected into ``sys.modules``. The fake
``FunctionTool`` simply records the kwargs passed to ``from_defaults`` so we can
assert the adapter wired name/description/callables correctly.
"""

from __future__ import annotations

import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.llamaindex import LlamaIndexAdapter


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
def fake_llama_index(monkeypatch: pytest.MonkeyPatch) -> type:
    """Inject a fake ``llama_index.core.tools`` module exposing FunctionTool."""

    class FunctionTool:
        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def from_defaults(cls, **kwargs: object) -> FunctionTool:
            return cls(**kwargs)

    pkg = types.ModuleType("llama_index")
    core = types.ModuleType("llama_index.core")
    tools = types.ModuleType("llama_index.core.tools")
    tools.FunctionTool = FunctionTool
    core.tools = tools
    pkg.core = core

    monkeypatch.setitem(sys.modules, "llama_index", pkg)
    monkeypatch.setitem(sys.modules, "llama_index.core", core)
    monkeypatch.setitem(sys.modules, "llama_index.core.tools", tools)

    return FunctionTool


async def _first_spec(gantry: AgentGantry):
    from agent_gantry.integrations.frameworks.base import GantryToolset

    specs = await GantryToolset(gantry).select("send an email", limit=5)
    by_name = {s.name: s for s in specs}
    assert "send_email" in by_name, f"send_email not selected: {list(by_name)}"
    return by_name["send_email"]


async def test_spec_to_llamaindex_captures_metadata(
    gantry: AgentGantry, fake_llama_index: type
) -> None:
    spec = await _first_spec(gantry)
    tool = LlamaIndexAdapter.convert(spec)

    assert tool.name == "send_email"
    assert tool.description == "Send an email message to a recipient."
    assert callable(tool.fn)
    assert callable(tool.async_fn)


async def test_spec_to_llamaindex_async_fn_executes(
    gantry: AgentGantry, fake_llama_index: type
) -> None:
    spec = await _first_spec(gantry)
    tool = LlamaIndexAdapter.convert(spec)

    result = await tool.async_fn(to="boss@x.com")
    assert result == "sent:boss@x.com"


async def test_for_llamaindex_maps_all_specs(gantry: AgentGantry, fake_llama_index: type) -> None:
    tools = await LlamaIndexAdapter(gantry).select("send an email", limit=5)

    assert len(tools) >= 1
    names = {t.name for t in tools}
    assert "send_email" in names
    for t in tools:
        assert callable(t.fn)
        assert callable(t.async_fn)


def test_spec_to_llamaindex_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the fake module installed, a helpful ImportError is raised."""
    for mod in ("llama_index", "llama_index.core", "llama_index.core.tools"):
        monkeypatch.setitem(sys.modules, mod, None)

    class _DummySpec:
        name = "x"
        description = "d"

        def invoke(self, **kwargs: object) -> object:  # pragma: no cover
            return None

        def callable_for_signature(self):  # pragma: no cover
            async def _fn(**kwargs: object) -> object:
                return None

            return _fn

    with pytest.raises(ImportError, match="pip install llama-index-core"):
        LlamaIndexAdapter.convert(_DummySpec())
