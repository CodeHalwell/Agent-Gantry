"""Tests for the DSPy native tool adapter against the *real* installed ``dspy``.

Exercises :class:`DSPyAdapter` (tool conversion + the per-call
``agent_builder``) against the actual ``dspy`` package, skipped everywhere it
isn't installed. No API keys are required: ``dspy.ReAct`` construction never
touches an LM (only an actual reasoning call does), and
``dspy.utils.dummies.DummyLM`` — a scripted stand-in ``BaseLM`` DSPy ships for
its own unit tests — drives one full, offline ``dspy.ReAct`` run end to end,
proving the adapter's tools work correctly under DSPy's default synchronous
``react(question=...)`` call path (see the design rationale in
``agent_gantry/integrations/frameworks/dspy.py``'s module docstring).
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("dspy")

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.utils.dummies import DummyLM

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.dspy import DSPyAdapter
from agent_gantry.integrations.frameworks.base import GantryToolset, ToolExecutionError
from agent_gantry.integrations.frameworks.dspy import GantryLiveDSPyReAct


@pytest.fixture
async def gantry():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=64))

    @g.register(tags=["email", "communication"])
    def send_email(to: str, body: str = "") -> str:
        "Send an email message to a recipient."
        return f"sent:{to}"

    @g.register(tags=["math", "arithmetic"])
    def add(a: int, b: int) -> int:
        "Add two integers together."
        return a + b

    await g.sync()
    return g


async def test_spec_to_dspy_builds_real_tool(gantry):
    specs = await GantryToolset(gantry).select("send an email", limit=1)
    spec = specs[0]
    tool = DSPyAdapter.convert(spec)

    assert isinstance(tool, dspy.Tool)
    assert tool.name == spec.name == "send_email"
    assert tool.desc == spec.description == "Send an email message to a recipient."
    assert "to" in tool.args and "body" in tool.args
    assert tool.arg_desc["to"].endswith("(Required)")

    # Real signature, not a bare **kwargs.
    params = inspect.signature(tool.func).parameters
    assert "to" in params and "body" in params
    assert not inspect.iscoroutinefunction(tool.func)


async def test_dspy_tool_call_routes_through_gantry_execute(gantry):
    """``dspy.Tool.__call__`` — the sync path ``ReAct.forward`` uses — works with no LM/loop caveats."""
    specs = await GantryToolset(gantry).select("send an email", limit=1)
    tool = DSPyAdapter.convert(specs[0])

    result = tool(to="boss@x.com")
    assert result == "sent:boss@x.com"


async def test_dspy_tool_acall_also_works(gantry):
    """``dspy.Tool.acall`` — the async path ``ReAct.aforward`` uses — also works for our sync-bridged tool."""
    specs = await GantryToolset(gantry).select("send an email", limit=1)
    tool = DSPyAdapter.convert(specs[0])

    result = await tool.acall(to="boss@x.com")
    assert result == "sent:boss@x.com"


async def test_missing_optional_arg_uses_tool_default(gantry):
    specs = await GantryToolset(gantry).select("send an email", limit=1)
    tool = DSPyAdapter.convert(specs[0])

    result = tool(to="boss@x.com")
    assert result == "sent:boss@x.com"


async def test_tool_failure_surfaces_as_tool_execution_error(gantry):
    """``dspy.Tool.__call__``/``.acall`` never swallow the error -- see
    ``test_react_records_tool_failure_as_an_error_result`` below for the
    *different* behaviour of ``dspy.ReAct`` itself, one layer up.
    """

    @gantry.register(tags=["danger"])
    def explode() -> str:
        "Always raises."
        raise RuntimeError("boom")

    await gantry.sync()

    specs = await GantryToolset(gantry).select("always raises", limit=1)
    tool = DSPyAdapter.convert(specs[0])

    with pytest.raises(ToolExecutionError):
        tool()

    with pytest.raises(ToolExecutionError):
        await tool.acall()


def _calls(*calls):
    """One DummyLM turn issuing ``calls`` (``(name, args)`` pairs) through ``tool_calls``."""
    return {
        "next_thought": "next",
        "tool_calls": {"tool_calls": [{"name": n, "args": a} for n, a in calls]},
    }


def _results(pred):
    """Every non-submit ``ToolCallResult`` in a ``dspy.ReAct`` prediction's history."""
    return [
        result
        for event in pred.history.messages
        if event.get("tool_calls") is not None
        for result in event["tool_calls"].tool_call_results.tool_call_results
        if result.name != "submit"
    ]


async def test_react_records_tool_failure_as_an_error_result(gantry):
    """``dspy.ReAct`` turns a raising tool into an ``is_error`` result, not an exception.

    DSPy adds its own short traceback; the text it ends with is Gantry's.
    """

    @gantry.register(tags=["danger"])
    def explode() -> str:
        "Always raises."
        raise RuntimeError("boom")

    await gantry.sync()

    tools = await DSPyAdapter(gantry).select("always raises", limit=1)
    react = dspy.ReAct("question -> answer", tools=tools, max_iters=2)

    adapter = ChatAdapter()
    lm = DummyLM(
        [_calls(("explode", {})), _calls(("submit", {"answer": "could not complete"}))],
        adapter=adapter,
    )
    # `dspy.context(...)` (not `dspy.configure(...)`): only one async task per
    # process may ever call `dspy.configure`.
    with dspy.context(lm=lm, adapter=adapter):
        pred = react(question="please explode")

    assert pred.answer == "could not complete"
    [result] = _results(pred)
    assert result.is_error
    assert result.value.startswith("Execution error in explode:")
    assert result.value.splitlines()[-1] == "agent_gantry.dspy.DSPyToolError: RuntimeError: boom"


async def test_agent_builder_builds_real_react_no_lm_required(gantry):
    """Constructing ``dspy.ReAct`` never touches the LM -- only calling it does."""
    builder = DSPyAdapter(gantry).agent_builder("question -> answer", max_iters=3, limit=1)
    assert isinstance(builder, GantryLiveDSPyReAct)

    react = await builder.build("send an email to a recipient")
    assert isinstance(react, dspy.ReAct)
    assert react.max_iters == 3
    assert "send_email" in react.tools

    react2 = await builder.build("add two integers together")
    assert "add" in react2.tools


async def test_react_end_to_end_with_dummy_lm_sync_call(gantry):
    """Full offline ``dspy.ReAct`` run through the SYNCHRONOUS ``react(...)`` entry point.

    Proves the sync-bridged tools work with zero DSPy configuration (no
    ``allow_tool_async_sync_conversion``). No network access or API key.
    """
    adapter = ChatAdapter()
    lm = DummyLM(
        [
            _calls(("send_email", {"to": "boss@x.com"})),
            _calls(("submit", {"answer": "Email sent to boss@x.com."})),
        ],
        adapter=adapter,
    )
    with dspy.context(lm=lm, adapter=adapter):
        tools = await DSPyAdapter(gantry).select("send an email", limit=1)
        react = dspy.ReAct("question -> answer", tools=tools, max_iters=3)

        pred = react(question="Please email my boss.")

    assert pred.answer == "Email sent to boss@x.com."
    assert pred.termination_reason == "submit"
    [result] = _results(pred)
    assert (result.value, result.is_error) == ("sent:boss@x.com", False)
