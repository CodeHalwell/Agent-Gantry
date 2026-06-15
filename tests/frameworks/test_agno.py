"""Tests for the Agno (formerly Phidata) native tool adapter.

Agno is not installed in this environment, so minimal fake ``agno``,
``agno.tools`` and ``agno.tools.function`` modules are injected into
``sys.modules`` (cleaned up by ``monkeypatch.setitem``) to resolve the
adapter's lazy import.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder


class _FakeFunction:
    """Stand-in for ``agno.tools.function.Function``.

    Stores the kwargs passed to ``__init__`` so tests can assert how the
    adapter built the tool.
    """

    def __init__(self, name=None, description=None, parameters=None, entrypoint=None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.entrypoint = entrypoint


@pytest.fixture
def fake_agno(monkeypatch):
    agno = types.ModuleType("agno")
    agno_tools = types.ModuleType("agno.tools")
    agno_tools_function = types.ModuleType("agno.tools.function")
    agno_tools_function.Function = _FakeFunction
    agno.tools = agno_tools
    agno_tools.function = agno_tools_function

    monkeypatch.setitem(sys.modules, "agno", agno)
    monkeypatch.setitem(sys.modules, "agno.tools", agno_tools)
    monkeypatch.setitem(sys.modules, "agno.tools.function", agno_tools_function)
    return agno


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


async def test_spec_to_agno_builds_native_tool(fake_agno, gantry):
    from agent_gantry.agno import AgnoAdapter
    from agent_gantry.integrations.frameworks.base import GantryToolset

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    spec = specs[0]
    fn = AgnoAdapter.convert(spec)

    assert fn.name == spec.name == "send_email"
    assert fn.description == spec.description == "Send an email message to a recipient."
    assert fn.parameters == spec.parameters
    assert fn.parameters["properties"]["to"]["type"] == "string"

    # entrypoint is a sync wrapper (spec.invoke), so Agno calls it from
    # synchronous code with the tool arguments as keyword args. Run it off the
    # event loop here to mimic that, routing through gantry.execute.
    result = await asyncio.to_thread(fn.entrypoint, to="boss@x.com")
    assert result == "sent:boss@x.com"


async def test_for_agno_returns_function_list(fake_agno, gantry):
    from agent_gantry.agno import AgnoAdapter

    tools = await AgnoAdapter(gantry).select("send an email", limit=2)

    assert isinstance(tools, list)
    assert len(tools) >= 1
    names = {t.name for t in tools}
    assert "send_email" in names


async def test_missing_agno_raises_helpful_error(monkeypatch, gantry):
    # Ensure the lazy import fails even if a real package is somehow present.
    monkeypatch.setitem(sys.modules, "agno.tools.function", None)

    from agent_gantry.agno import AgnoAdapter
    from agent_gantry.integrations.frameworks.base import GantryToolset

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    with pytest.raises(ImportError, match="pip install agno"):
        AgnoAdapter.convert(specs[0])
