"""Tests for the Pydantic AI native tool adapter.

Pydantic AI is not installed in this environment, so minimal fake ``pydantic_ai``
and ``pydantic_ai.tools`` modules are injected into ``sys.modules`` (cleaned up
by ``monkeypatch.setitem``) to resolve the adapter's lazy import. Two stub
``Tool`` variants are used: one exposing ``from_schema`` (the preferred path) and
one without it (the fallback path).
"""

from __future__ import annotations

import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder


class _FakeToolFromSchema:
    """Stand-in for ``pydantic_ai.tools.Tool`` with a ``from_schema`` builder.

    Stores the kwargs passed to ``from_schema`` on the instance so tests can
    assert how the adapter built the tool.
    """

    def __init__(self, **kwargs):
        self.function = kwargs.get("function")
        self.name = kwargs.get("name")
        self.description = kwargs.get("description")
        self.json_schema = kwargs.get("json_schema")

    @classmethod
    def from_schema(cls, **kwargs):
        return cls(**kwargs)


class _FakeToolNoSchema:
    """Stand-in for an older ``Tool`` without ``from_schema``.

    Only supports the function-based constructor used by the fallback path.
    """

    def __init__(self, function, *, name=None, description=None, takes_ctx=False):
        self.function = function
        self.name = name
        self.description = description
        self.takes_ctx = takes_ctx


def _inject_pydantic_ai(monkeypatch, tool_cls):
    pkg = types.ModuleType("pydantic_ai")
    tools = types.ModuleType("pydantic_ai.tools")
    tools.Tool = tool_cls
    pkg.tools = tools
    monkeypatch.setitem(sys.modules, "pydantic_ai", pkg)
    monkeypatch.setitem(sys.modules, "pydantic_ai.tools", tools)
    return tools


@pytest.fixture
def fake_pydantic_ai(monkeypatch):
    return _inject_pydantic_ai(monkeypatch, _FakeToolFromSchema)


@pytest.fixture
def fake_pydantic_ai_no_schema(monkeypatch):
    return _inject_pydantic_ai(monkeypatch, _FakeToolNoSchema)


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


async def test_spec_to_pydantic_ai_uses_from_schema(fake_pydantic_ai, gantry):
    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.pydantic_ai import PydanticAIAdapter

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    spec = specs[0]
    tool = PydanticAIAdapter.convert(spec)

    assert isinstance(tool, _FakeToolFromSchema)
    assert tool.name == spec.name == "send_email"
    assert tool.description == spec.description == "Send an email message to a recipient."
    assert tool.json_schema == spec.parameters
    assert tool.json_schema["properties"]["to"]["type"] == "string"

    # The captured function routes through gantry.execute and returns the result.
    assert await tool.function(to="boss@x.com") == "sent:boss@x.com"


async def test_spec_to_pydantic_ai_falls_back_without_from_schema(
    fake_pydantic_ai_no_schema, gantry
):
    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.pydantic_ai import PydanticAIAdapter

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    spec = specs[0]
    tool = PydanticAIAdapter.convert(spec)

    assert isinstance(tool, _FakeToolNoSchema)
    assert tool.name == spec.name == "send_email"
    assert tool.description == spec.description == "Send an email message to a recipient."
    assert tool.takes_ctx is False

    assert await tool.function(to="boss@x.com") == "sent:boss@x.com"


async def test_for_pydantic_ai_returns_tool_list(fake_pydantic_ai, gantry):
    from agent_gantry.pydantic_ai import PydanticAIAdapter

    tools = await PydanticAIAdapter(gantry).select("send an email", limit=2)

    assert isinstance(tools, list)
    assert len(tools) >= 1
    names = {t.name for t in tools}
    assert "send_email" in names


async def test_missing_pydantic_ai_raises_helpful_error(monkeypatch, gantry):
    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.pydantic_ai import PydanticAIAdapter

    # Ensure the lazy import fails even if a real package is somehow present.
    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    monkeypatch.setitem(sys.modules, "pydantic_ai.tools", None)

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    with pytest.raises(ImportError, match="pip install pydantic-ai"):
        PydanticAIAdapter.convert(specs[0])
