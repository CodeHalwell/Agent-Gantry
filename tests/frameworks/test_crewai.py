"""Tests for the CrewAI native tool adapter.

CrewAI is not installed in this environment, so minimal fake ``crewai`` and
``crewai.tools`` modules are injected into ``sys.modules`` (cleaned up by
``monkeypatch.setitem``) to resolve the adapter's lazy import. The stub
``BaseTool`` is a plain (non-Pydantic) subclass-friendly class.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder


class _FakeBaseTool:
    """Stand-in for ``crewai.tools.BaseTool``.

    Plain Python (not Pydantic): instances construct with no required args and
    expose class-level ``name`` / ``description`` attributes. ``_run`` is meant
    to be overridden by the adapter's dynamically-built subclass.
    """

    name: str = ""
    description: str = ""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _run(self, **kwargs):  # pragma: no cover - overridden by adapter
        raise NotImplementedError


@pytest.fixture
def fake_crewai(monkeypatch):
    crewai = types.ModuleType("crewai")
    tools = types.ModuleType("crewai.tools")
    tools.BaseTool = _FakeBaseTool
    crewai.tools = tools
    monkeypatch.setitem(sys.modules, "crewai", crewai)
    monkeypatch.setitem(sys.modules, "crewai.tools", tools)
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


async def test_spec_to_crewai_builds_native_tool(fake_crewai, gantry):
    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.integrations.frameworks.crewai import spec_to_crewai

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    spec = specs[0]
    tool = spec_to_crewai(spec)

    assert tool.name == spec.name == "send_email"
    assert (
        tool.description
        == spec.description
        == "Send an email message to a recipient."
    )

    # ``_run`` routes through the sync ``spec.invoke`` (which spins its own event
    # loop), so call it off the test's running loop via a worker thread.
    result = await asyncio.to_thread(tool._run, to="boss@x.com")
    assert result == "sent:boss@x.com"


async def test_for_crewai_returns_tool_list(fake_crewai, gantry):
    from agent_gantry.integrations.frameworks.crewai import for_crewai

    tools = await for_crewai(gantry, "send an email", limit=2)

    assert isinstance(tools, list)
    assert len(tools) >= 1
    names = {t.name for t in tools}
    assert "send_email" in names


async def test_missing_crewai_raises_helpful_error(monkeypatch, gantry):
    # Ensure the lazy import fails even if a real package is somehow present.
    monkeypatch.setitem(sys.modules, "crewai", None)
    monkeypatch.setitem(sys.modules, "crewai.tools", None)

    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.integrations.frameworks.crewai import spec_to_crewai

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    with pytest.raises(ImportError, match="pip install crewai"):
        spec_to_crewai(specs[0])
