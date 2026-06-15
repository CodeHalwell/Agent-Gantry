"""Tests for the Haystack native tool adapter.

Haystack (``haystack-ai``) is not installed in this environment, so minimal
fake ``haystack`` and ``haystack.tools`` modules are injected into
``sys.modules`` (cleaned up by ``monkeypatch.setitem``) to resolve the
adapter's lazy import. The stub ``Tool`` simply records the constructor
arguments the adapter passes.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder


class _FakeTool:
    """Stand-in for ``haystack.tools.Tool``.

    Stores ``name`` / ``description`` / ``parameters`` / ``function`` exactly as
    Haystack 2.x's ``Tool`` would, so the adapter's output can be inspected.
    """

    def __init__(self, name, description, parameters, function):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function


@pytest.fixture
def fake_haystack(monkeypatch):
    haystack = types.ModuleType("haystack")
    tools = types.ModuleType("haystack.tools")
    tools.Tool = _FakeTool
    haystack.tools = tools
    monkeypatch.setitem(sys.modules, "haystack", haystack)
    monkeypatch.setitem(sys.modules, "haystack.tools", tools)
    return tools


@pytest.fixture
async def gantry():
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


async def test_spec_to_haystack_builds_native_tool(fake_haystack, gantry):
    from agent_gantry.haystack import HaystackAdapter
    from agent_gantry.integrations.frameworks.base import GantryToolset

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    spec = specs[0]
    tool = HaystackAdapter.convert(spec)

    assert tool.name == spec.name == "send_email"
    assert (
        tool.description
        == spec.description
        == "Send an email message to a recipient."
    )
    assert tool.parameters == spec.parameters
    assert isinstance(tool.parameters, dict)

    # ``function`` routes through the sync ``spec.invoke`` (which spins its own
    # event loop), so call it off the test's running loop via a worker thread.
    result = await asyncio.to_thread(tool.function, to="boss@x.com")
    assert result == "sent:boss@x.com"


async def test_for_haystack_returns_tool_list(fake_haystack, gantry):
    from agent_gantry.haystack import HaystackAdapter

    tools = await HaystackAdapter(gantry).select("send an email", limit=2)

    assert isinstance(tools, list)
    assert len(tools) >= 1
    names = {t.name for t in tools}
    assert "send_email" in names


async def test_missing_haystack_raises_helpful_error(monkeypatch, gantry):
    # Ensure the lazy import fails even if a real package is somehow present.
    monkeypatch.setitem(sys.modules, "haystack", None)
    monkeypatch.setitem(sys.modules, "haystack.tools", None)

    from agent_gantry.haystack import HaystackAdapter
    from agent_gantry.integrations.frameworks.base import GantryToolset

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    with pytest.raises(ImportError, match="pip install haystack-ai"):
        HaystackAdapter.convert(specs[0])
