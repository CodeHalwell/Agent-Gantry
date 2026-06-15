"""Tests for the OpenAI Agents SDK native tool adapter.

The OpenAI Agents SDK is not installed in this environment, so a minimal fake
``agents`` module is injected into ``sys.modules`` (cleaned up by
``monkeypatch.setitem``) to resolve the adapter's lazy import.
"""

from __future__ import annotations

import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder


class _FakeFunctionTool:
    """Stand-in for ``agents.FunctionTool``.

    Stores the kwargs passed to ``__init__`` so tests can assert how the
    adapter built the tool.
    """

    def __init__(self, name=None, description=None, params_json_schema=None, on_invoke_tool=None):
        self.name = name
        self.description = description
        self.params_json_schema = params_json_schema
        self.on_invoke_tool = on_invoke_tool


@pytest.fixture
def fake_agents(monkeypatch):
    agents = types.ModuleType("agents")
    agents.FunctionTool = _FakeFunctionTool
    monkeypatch.setitem(sys.modules, "agents", agents)
    return agents


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


async def test_spec_to_openai_agents_builds_native_tool(fake_agents, gantry):
    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.integrations.frameworks.openai_agents import OpenAIAgentsAdapter

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    spec = specs[0]
    tool = OpenAIAgentsAdapter.convert(spec)

    assert tool.name == spec.name == "send_email"
    assert tool.description == spec.description == "Send an email message to a recipient."

    # Strict tools must declare additionalProperties: False.
    assert tool.params_json_schema["additionalProperties"] is False
    assert tool.params_json_schema["properties"]["to"]["type"] == "string"

    # on_invoke_tool parses a JSON string and routes through gantry.execute.
    result = await tool.on_invoke_tool(None, '{"to": "boss@x.com"}')
    assert isinstance(result, str)
    assert "sent:boss@x.com" in result

    # on_invoke_tool also accepts an already-parsed dict of arguments.
    result_dict = await tool.on_invoke_tool(None, {"to": "boss@x.com"})
    assert isinstance(result_dict, str)
    assert "sent:boss@x.com" in result_dict


async def test_for_openai_agents_returns_tool_list(fake_agents, gantry):
    from agent_gantry.integrations.frameworks.openai_agents import OpenAIAgentsAdapter

    tools = await OpenAIAgentsAdapter(gantry).select("send an email", limit=2)

    assert isinstance(tools, list)
    assert len(tools) >= 1
    names = {t.name for t in tools}
    assert "send_email" in names
    assert all(t.params_json_schema["additionalProperties"] is False for t in tools)


async def test_missing_openai_agents_raises_helpful_error(monkeypatch, gantry):
    # Ensure the lazy import fails even if a real package is somehow present.
    monkeypatch.setitem(sys.modules, "agents", None)

    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.integrations.frameworks.openai_agents import OpenAIAgentsAdapter

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    with pytest.raises(ImportError, match="pip install openai-agents"):
        OpenAIAgentsAdapter.convert(specs[0])
