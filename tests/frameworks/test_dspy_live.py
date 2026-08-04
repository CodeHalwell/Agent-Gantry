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
    ``test_react_forward_absorbs_tool_failure_into_trajectory`` below for the
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


async def test_react_forward_absorbs_tool_failure_into_trajectory(gantry):
    """Documents a deliberate deviation that lives in DSPy's own code, not ours.

    ``dspy.Tool.__call__``/``.acall`` (proven above) never swallow
    ``ToolExecutionError`` -- but ``dspy.ReAct.forward``/``aforward``
    (DSPy's *own* agentic driver, installed dspy 3.2.1 and 3.3.0) wrap each tool call
    in a bare ``except Exception`` and fold it into the trajectory as an
    ``"Execution error in <tool>: ..."`` observation string instead of
    raising -- the same absorption pattern as a standard ReAct loop feeding
    a tool failure back to the model as an observation. Since
    ``DSPyAdapter.agent_builder``/``.live()`` hand the user exactly this
    ``dspy.ReAct``, this is the failure behaviour most DSPy users actually
    see -- see "Error-handling policy" in
    ``integrations/frameworks/README.md``.
    """

    @gantry.register(tags=["danger"])
    def explode() -> str:
        "Always raises."
        raise RuntimeError("boom")

    await gantry.sync()

    tools = await DSPyAdapter(gantry).select("always raises", limit=1)
    react = dspy.ReAct("question -> answer", tools=tools, max_iters=1)

    adapter = ChatAdapter()
    lm = DummyLM(
        [
            {"next_thought": "call it", "next_tool_name": "explode", "next_tool_args": {}},
            {"reasoning": "it failed", "answer": "could not complete"},
        ],
        adapter=adapter,
    )
    # `dspy.context(...)` (not `dspy.configure(...)`): only one async task per
    # process may ever call `dspy.configure`, so a second test using it here
    # would break whichever of the two tests in this file runs second.
    # `dspy.context` is the safe, callable-from-any-task, block-scoped form.
    with dspy.context(lm=lm, adapter=adapter):
        pred = react(question="please explode")

    # No exception escaped `react(...)`; the failure is folded into the
    # trajectory as an observation instead.
    assert pred.answer == "could not complete"
    observation = pred.trajectory["observation_0"]
    assert observation.startswith("Execution error in explode:")
    assert "boom" in observation


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

    Uses ``dspy.context(...)`` rather than ``dspy.configure(...)``: only one
    async task per process may ever call ``dspy.configure`` (DSPy raises for
    every subsequent caller from a different task), so with more than one
    test in this module needing a scripted LM, ``dspy.context`` -- callable
    from any task, scoped to the ``with`` block -- is the only safe choice.
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
    with dspy.context(lm=lm, adapter=adapter):
        tools = await DSPyAdapter(gantry).select("send an email", limit=1)
        react = dspy.ReAct("question -> answer", tools=tools, max_iters=3)

        pred = react(question="Please email my boss.")

        assert pred.answer == "Email sent to boss@x.com."
        assert pred.trajectory["observation_0"] == "sent:boss@x.com"
