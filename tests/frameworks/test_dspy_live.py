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
    @gantry.register(tags=["danger"])
    def explode() -> str:
        "Always raises."
        raise RuntimeError("boom")

    await gantry.sync()

    specs = await GantryToolset(gantry).select("always raises", limit=1)
    tool = DSPyAdapter.convert(specs[0])

    with pytest.raises(ToolExecutionError):
        tool()


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
    """Full offline ``dspy.ReAct`` run: DummyLM scripts tool-call then finish.

    Uses the SYNCHRONOUS ``react(question=...)`` call (not ``await
    react.acall(...)``) -- the default DSPy entry point -- to prove the
    adapter's sync-bridged tools work under it with zero DSPy configuration
    (no ``allow_tool_async_sync_conversion``). No network access or API key.
    """
    adapter = ChatAdapter()
    lm = DummyLM(
        [
            {
                "next_thought": "I should send the email",
                "next_tool_name": "send_email",
                "next_tool_args": {"to": "boss@x.com"},
            },
            {"next_thought": "Done", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "The email was sent successfully.", "answer": "Email sent to boss@x.com."},
        ],
        adapter=adapter,
    )
    dspy.configure(lm=lm, adapter=adapter)
    try:
        tools = await DSPyAdapter(gantry).select("send an email", limit=1)
        react = dspy.ReAct("question -> answer", tools=tools, max_iters=3)

        pred = react(question="Please email my boss.")

        assert pred.answer == "Email sent to boss@x.com."
        assert pred.trajectory["observation_0"] == "sent:boss@x.com"
    finally:
        dspy.configure(lm=None, adapter=None)
