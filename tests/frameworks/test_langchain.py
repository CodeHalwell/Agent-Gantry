"""Tests for the LangChain native tool adapter.

LangChain is not installed in this environment, so a minimal fake
``langchain_core.tools`` module is injected into ``sys.modules`` (cleaned up by
``monkeypatch.setitem``) to resolve the adapter's lazy import.
"""

from __future__ import annotations

import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder


class _FakeStructuredTool:
    """Stand-in for ``langchain_core.tools.StructuredTool``.

    Stores the kwargs passed to ``from_function`` on the instance so tests can
    assert how the adapter built the tool.
    """

    def __init__(self, **kwargs):
        self.func = kwargs.get("func")
        self.coroutine = kwargs.get("coroutine")
        self.name = kwargs.get("name")
        self.description = kwargs.get("description")
        self.args_schema = kwargs.get("args_schema")

    @classmethod
    def from_function(cls, **kwargs):
        return cls(**kwargs)


@pytest.fixture
def fake_langchain(monkeypatch):
    core = types.ModuleType("langchain_core")
    tools = types.ModuleType("langchain_core.tools")
    tools.StructuredTool = _FakeStructuredTool
    core.tools = tools
    monkeypatch.setitem(sys.modules, "langchain_core", core)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", tools)
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


async def test_spec_to_langchain_builds_native_tool(fake_langchain, gantry):
    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.integrations.frameworks.langchain import spec_to_langchain

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    spec = specs[0]
    tool = spec_to_langchain(spec)

    assert tool.name == spec.name == "send_email"
    assert tool.description == spec.description == "Send an email message to a recipient."
    assert tool.args_schema == spec.parameters
    assert tool.args_schema["properties"]["to"]["type"] == "string"

    # The captured coroutine routes through gantry.execute and returns the result.
    assert await tool.coroutine(to="boss@x.com") == "sent:boss@x.com"


async def test_for_langchain_returns_tool_list(fake_langchain, gantry):
    from agent_gantry.integrations.frameworks.langchain import for_langchain

    tools = await for_langchain(gantry, "send an email", limit=2)

    assert isinstance(tools, list)
    assert len(tools) >= 1
    names = {t.name for t in tools}
    assert "send_email" in names


async def test_missing_langchain_raises_helpful_error(monkeypatch, gantry):
    # Ensure the lazy import fails even if a real package is somehow present.
    # Null BOTH the parent package and the submodule the adapter imports —
    # nulling only the parent doesn't force failure when `langchain_core.tools`
    # is already cached in sys.modules (e.g. langchain-core is installed).
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", None)

    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.integrations.frameworks.langchain import spec_to_langchain

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    with pytest.raises(ImportError, match="pip install langchain-core"):
        spec_to_langchain(specs[0])
