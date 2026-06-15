"""Tests for the LangGraph native tool adapter.

LangGraph consumes LangChain ``BaseTool`` objects, so the adapter reuses the
LangChain wrappers. As with the LangChain tests, a fake ``langchain_core.tools``
module is injected so the lazy import resolves without the real package.
"""

from __future__ import annotations

import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder


class _FakeStructuredTool:
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


def test_spec_to_langgraph_is_langchain_wrapper():
    from agent_gantry.integrations.frameworks.langchain import _spec_to_langchain
    from agent_gantry.integrations.frameworks.langgraph import _spec_to_langgraph

    assert _spec_to_langgraph is _spec_to_langchain


async def test_spec_to_langgraph_builds_native_tool(fake_langchain, gantry):
    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.langgraph import LangGraphAdapter

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    spec = specs[0]
    tool = LangGraphAdapter.convert(spec)

    assert tool.name == spec.name == "send_email"
    assert tool.description == spec.description
    assert tool.args_schema == spec.parameters

    assert await tool.coroutine(to="boss@x.com") == "sent:boss@x.com"


async def test_for_langgraph_returns_tool_list(fake_langchain, gantry):
    from agent_gantry.langgraph import LangGraphAdapter

    tools = await LangGraphAdapter(gantry).select("send an email", limit=2)

    assert isinstance(tools, list)
    assert len(tools) >= 1
    assert "send_email" in {t.name for t in tools}
