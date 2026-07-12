"""Tests for the DSPy native tool adapter.

DSPy is not installed in this environment, so a minimal fake ``dspy`` package
(with a stand-in ``Tool``, ``ReAct``, and
``dspy.adapters.types.tool.convert_input_schema_to_tool_args``) is injected
into ``sys.modules`` (cleaned up by ``monkeypatch.setitem``) to resolve the
adapter's lazy imports. The fakes mirror the real ``dspy.Tool``'s
``name``/``desc``/``args``/``arg_types``/``arg_desc`` constructor contract and
DSPy's own JSON-Schema-to-``Tool``-args converter closely enough to assert how
the adapter builds tools, without pulling in DSPy's Pydantic/LM machinery.
"""

from __future__ import annotations

import sys
import types

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.integrations.frameworks.base import ToolExecutionError


class _FakeTool:
    """Stand-in for ``dspy.Tool``: stores exactly the kwargs it's given."""

    def __init__(self, func, name=None, desc=None, args=None, arg_types=None, arg_desc=None):
        self.func = func
        self.name = name
        self.desc = desc
        self.args = args
        self.arg_types = arg_types
        self.arg_desc = arg_desc

    def __call__(self, **kwargs):
        return self.func(**kwargs)

    async def acall(self, **kwargs):
        result = self.func(**kwargs)
        return result


class _FakeReAct:
    """Stand-in for ``dspy.ReAct``: just remembers what it was built with."""

    def __init__(self, signature, tools=None, max_iters=20, **kwargs):
        self.signature = signature
        self.tools = list(tools or [])
        self.max_iters = max_iters
        self.kwargs = kwargs


def _fake_convert_input_schema_to_tool_args(schema):
    """Stand-in for ``dspy.adapters.types.tool.convert_input_schema_to_tool_args``."""
    schema = schema or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    args = dict(properties)
    arg_types = {name: prop.get("type", "Any") for name, prop in properties.items()}
    arg_desc = {}
    for name, prop in properties.items():
        desc = prop.get("description", "No description provided.")
        if name in required:
            desc += " (Required)"
        arg_desc[name] = desc
    return args, arg_types, arg_desc


@pytest.fixture
def fake_dspy(monkeypatch):
    dspy_module = types.ModuleType("dspy")
    dspy_module.Tool = _FakeTool
    dspy_module.ReAct = _FakeReAct
    adapters_module = types.ModuleType("dspy.adapters")
    types_module = types.ModuleType("dspy.adapters.types")
    tool_module = types.ModuleType("dspy.adapters.types.tool")
    tool_module.convert_input_schema_to_tool_args = _fake_convert_input_schema_to_tool_args

    monkeypatch.setitem(sys.modules, "dspy", dspy_module)
    monkeypatch.setitem(sys.modules, "dspy.adapters", adapters_module)
    monkeypatch.setitem(sys.modules, "dspy.adapters.types", types_module)
    monkeypatch.setitem(sys.modules, "dspy.adapters.types.tool", tool_module)
    return dspy_module


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


async def test_spec_to_dspy_builds_native_tool(fake_dspy, gantry):
    from agent_gantry.dspy import DSPyAdapter
    from agent_gantry.integrations.frameworks.base import GantryToolset

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    spec = specs[0]
    tool = DSPyAdapter.convert(spec)

    assert tool.name == spec.name == "send_email"
    assert tool.desc == spec.description == "Send an email message to a recipient."
    # Gantry's own JSON-Schema parameters are passed through DSPy's own
    # convert_input_schema_to_tool_args() converter, not re-inferred.
    assert "to" in tool.args and "body" in tool.args
    assert tool.arg_desc["to"].endswith("(Required)")
    assert not tool.arg_desc["body"].endswith("(Required)")

    # The wrapped function carries a real signature derived from the JSON
    # schema (not a bare **kwargs), and is SYNCHRONOUS (see module docstring).
    import inspect

    params = inspect.signature(tool.func).parameters
    assert "to" in params and "body" in params
    assert not inspect.iscoroutinefunction(tool.func)

    # Calling it synchronously (dspy.Tool.__call__ / ReAct.forward's path)
    # works with no event loop running and no DSPy configuration needed.
    result = tool(to="boss@x.com")
    assert result == "sent:boss@x.com"


async def test_dspy_tool_executes_synchronously_inside_running_loop(fake_dspy, gantry):
    """The wrapped function is sync-callable even from inside a running loop.

    This test function is itself async (pytest-asyncio runs it on a live event
    loop), so calling ``tool(...)`` — a plain, non-awaited call, exactly what
    ``dspy.Tool.__call__``/``ReAct.forward`` does — exercises the loop-aware
    branch of ``ToolSpec.invoke``'s sync bridge (worker-thread offload) rather
    than the no-loop ``asyncio.run`` branch.
    """
    from agent_gantry.dspy import DSPyAdapter
    from agent_gantry.integrations.frameworks.base import GantryToolset

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    tool = DSPyAdapter.convert(specs[0])

    result = tool(to="boss@x.com")
    assert result == "sent:boss@x.com"


async def test_spec_to_dspy_drops_none_optional_args(fake_dspy, gantry):
    """Optional params left at their None default are dropped so the tool's own default applies."""
    from agent_gantry.dspy import DSPyAdapter
    from agent_gantry.integrations.frameworks.base import GantryToolset

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    tool = DSPyAdapter.convert(specs[0])

    # `body` defaults to None in the generated signature; omitting it should
    # not fail even though the underlying tool types it as `str`.
    result = tool(to="boss@x.com")
    assert result == "sent:boss@x.com"


async def test_for_dspy_returns_tool_list(fake_dspy, gantry):
    from agent_gantry.dspy import DSPyAdapter

    tools = await DSPyAdapter(gantry).select("send an email", limit=2)

    assert isinstance(tools, list)
    assert len(tools) >= 1
    names = {t.name for t in tools}
    assert "send_email" in names


async def test_select_routes_through_gantry_execute_and_raises_on_failure(fake_dspy, gantry):
    """A tool call that fails Gantry execution surfaces as ToolExecutionError, not a silent result."""
    from agent_gantry.dspy import DSPyAdapter
    from agent_gantry.integrations.frameworks.base import GantryToolset

    @gantry.register(tags=["danger"])
    def explode() -> str:
        "Always raises."
        raise RuntimeError("boom")

    await gantry.sync()

    specs = await GantryToolset(gantry).select("always raises", limit=1)
    tool = DSPyAdapter.convert(specs[0])

    with pytest.raises(ToolExecutionError):
        tool()


async def test_missing_dspy_raises_helpful_error(monkeypatch, gantry):
    # Ensure the lazy import fails even if a real package is somehow present.
    monkeypatch.setitem(sys.modules, "dspy", None)

    from agent_gantry.dspy import DSPyAdapter
    from agent_gantry.integrations.frameworks.base import GantryToolset

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    with pytest.raises(ImportError, match="pip install dspy"):
        DSPyAdapter.convert(specs[0])


async def test_agent_builder_rebuilds_dspy_react_with_reselected_tools(fake_dspy, gantry):
    """``agent_builder(...)`` returns a per-call builder; ``.build(query)`` makes a fresh ReAct."""
    from agent_gantry.dspy import DSPyAdapter
    from agent_gantry.integrations.frameworks.dspy import GantryLiveDSPyReAct

    builder = DSPyAdapter(gantry).agent_builder("question -> answer", max_iters=3, limit=1)
    assert isinstance(builder, GantryLiveDSPyReAct)

    react = await builder.build("send an email to a recipient")
    assert isinstance(react, _FakeReAct)
    assert react.signature == "question -> answer"
    assert react.max_iters == 3
    names = {t.name for t in react.tools}
    assert "send_email" in names

    # A second call with a different query re-selects for THAT query.
    react2 = await builder.build("add two integers")
    names2 = {t.name for t in react2.tools}
    assert "add" in names2


async def test_agent_builder_build_requires_dspy(monkeypatch, gantry):
    monkeypatch.setitem(sys.modules, "dspy", None)

    from agent_gantry.dspy import DSPyAdapter

    builder = DSPyAdapter(gantry).agent_builder("question -> answer")
    with pytest.raises(ImportError, match="pip install dspy"):
        await builder.build("send an email")
