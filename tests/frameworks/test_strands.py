"""Tests for the AWS Strands Agents native tool adapter.

Strands Agents is not installed in this environment, so a minimal fake
``strands`` module (with a stand-in ``@tool`` decorator) is injected into
``sys.modules`` (cleaned up by ``monkeypatch.setitem``) to resolve the
adapter's lazy import. The fake decorator mirrors the real
``strands.tools.decorator.tool``'s ``name``/``description``/``inputSchema``
override contract closely enough to assert how the adapter builds tools,
without pulling in Strands' Pydantic-model-building machinery.
"""

from __future__ import annotations

import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.frameworks.base import ToolExecutionError


class _FakeDecoratedFunctionTool:
    """Stand-in for ``strands.tools.decorator.DecoratedFunctionTool``.

    Stores the resolved ``tool_spec`` so tests can assert how the adapter
    built the tool, and stays directly callable (mirroring
    ``DecoratedFunctionTool.__call__``, which just calls the wrapped function).
    """

    def __init__(self, tool_name, tool_spec, func):
        self.tool_name = tool_name
        self.tool_spec = tool_spec
        self._tool_func = func
        self._is_dynamic = False

    def __call__(self, *args, **kwargs):
        return self._tool_func(*args, **kwargs)

    @property
    def is_dynamic(self):
        return self._is_dynamic

    def mark_dynamic(self):
        self._is_dynamic = True


def _fake_tool(func=None, *, name=None, description=None, inputSchema=None, context=False):  # noqa: N803 - mirrors strands.tool's real kwarg name
    """Stand-in for ``strands.tool`` / ``strands.tools.decorator.tool``."""

    def decorator(f):
        tool_name = name or f.__name__
        spec = {
            "name": tool_name,
            "description": description if description is not None else (f.__doc__ or tool_name),
            "inputSchema": inputSchema or {"json": {"type": "object", "properties": {}}},
        }
        return _FakeDecoratedFunctionTool(tool_name, spec, f)

    if func is not None:
        return decorator(func)
    return decorator


@pytest.fixture
def fake_strands(monkeypatch):
    strands = types.ModuleType("strands")
    strands.tool = _fake_tool
    monkeypatch.setitem(sys.modules, "strands", strands)
    return strands


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


async def test_spec_to_strands_builds_native_tool(fake_strands, gantry):
    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.strands import StrandsAdapter

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    spec = specs[0]
    tool = StrandsAdapter.convert(spec)

    assert tool.tool_name == spec.name == "send_email"
    assert (
        tool.tool_spec["description"] == spec.description == "Send an email message to a recipient."
    )
    # Gantry's own JSON-Schema parameters are passed through verbatim under
    # the Bedrock-style {"json": ...} envelope Strands' ToolSpec expects.
    assert tool.tool_spec["inputSchema"] == {"json": spec.parameters}
    assert tool.tool_spec["inputSchema"]["json"]["properties"]["to"]["type"] == "string"

    # The wrapped function carries a real signature/annotations derived from
    # the JSON schema (not a bare **kwargs), and routes through gantry.execute.
    import inspect

    params = inspect.signature(tool._tool_func).parameters
    assert "to" in params and "body" in params

    result = await tool(to="boss@x.com")
    assert result == "sent:boss@x.com"


async def test_spec_to_strands_drops_none_optional_args(fake_strands, gantry):
    """Optional params left at their None default are dropped so the tool's own default applies."""
    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.strands import StrandsAdapter

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    tool = StrandsAdapter.convert(specs[0])

    # `body` defaults to None in the generated signature; omitting it should
    # not fail even though the underlying tool types it as `str`.
    result = await tool(to="boss@x.com")
    assert result == "sent:boss@x.com"


async def test_for_strands_returns_tool_list(fake_strands, gantry):
    from agent_gantry.strands import StrandsAdapter

    tools = await StrandsAdapter(gantry).select("send an email", limit=2)

    assert isinstance(tools, list)
    assert len(tools) >= 1
    names = {t.tool_name for t in tools}
    assert "send_email" in names


async def test_select_routes_through_gantry_execute_and_raises_on_failure(fake_strands, gantry):
    """A tool call that fails Gantry execution surfaces as ToolExecutionError, not a silent result."""
    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.strands import StrandsAdapter

    @gantry.register(tags=["danger"])
    def explode() -> str:
        "Always raises."
        raise RuntimeError("boom")

    await gantry.sync()

    specs = await GantryToolset(gantry).select("always raises", limit=1)
    tool = StrandsAdapter.convert(specs[0])

    with pytest.raises(ToolExecutionError):
        await tool()


async def test_missing_strands_raises_helpful_error(monkeypatch, gantry):
    # Ensure the lazy import fails even if a real package is somehow present.
    monkeypatch.setitem(sys.modules, "strands", None)

    from agent_gantry.integrations.frameworks.base import GantryToolset
    from agent_gantry.strands import StrandsAdapter

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    with pytest.raises(ImportError, match="pip install strands-agents"):
        StrandsAdapter.convert(specs[0])


async def test_tool_hook_and_agent_builders_require_strands(monkeypatch, gantry):
    """The live per-turn helpers also raise a clean ImportError without strands-agents."""
    monkeypatch.setitem(sys.modules, "strands", None)
    monkeypatch.setitem(sys.modules, "strands.hooks", None)

    from agent_gantry.strands import StrandsAdapter

    adapter = StrandsAdapter(gantry)
    # Building the hook object itself never imports strands (lazy inside
    # register_hooks), but building a full agent does.
    hook = adapter.tool_hook(limit=2)
    assert hook is not None

    with pytest.raises(ImportError, match="pip install strands-agents"):
        adapter.agent()
