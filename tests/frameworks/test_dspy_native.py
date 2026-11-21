"""DSPy first-class behaviour through a real ``dspy.ReAct`` (``dspy>=3.5``).

Error results, ``error_formatter``, the approver, run context and the per-call
builder, all driven offline by DSPy's own scripted ``DummyLM``.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("dspy")

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.utils.dummies import DummyLM

from agent_gantry import AgentGantry, current_tool_invocation
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.core.security import SecurityPolicy
from agent_gantry.dspy import DSPyAdapter, DSPyToolError, current_dspy_module
from agent_gantry.integrations.frameworks.base import ToolExecutionError


async def _build(**kwargs: Any) -> AgentGantry:
    g = AgentGantry(embedder=SimpleEmbedder(dimension=256), **kwargs)
    g.sent = []  # type: ignore[attr-defined]

    @g.register(examples=["what's the weather in London", "weather forecast for Paris"])
    def get_weather(city: str) -> str:
        """Get the current weather forecast for a city."""
        return f"sunny in {city}"

    @g.register
    def divide(a: int, b: int) -> float:
        """Divide integer a by integer b."""
        return a / b

    @g.register(requires_confirmation=True)
    def send_email(to: str) -> str:
        """Send an email message to a recipient."""
        g.sent.append(to)  # type: ignore[attr-defined]
        return f"sent:{to}"

    @g.register
    def whoami() -> str:
        """Say which DSPy module and framework are running this call."""
        invocation = current_tool_invocation()
        return f"{type(current_dspy_module()).__name__}/{getattr(invocation, 'framework', None)}"

    @g.register
    def delete_record(record_id: str) -> str:
        """Delete a stored record by its identifier."""
        g.sent.append(f"deleted:{record_id}")  # type: ignore[attr-defined]
        return f"deleted:{record_id}"

    await g.sync()
    return g


@pytest.fixture
async def gantry():
    g = await _build()
    try:
        yield g
    finally:
        await g.close()


def _script(*calls: tuple[str, dict[str, Any]]) -> DummyLM:
    """A DummyLM that makes ``calls`` in one turn, then submits ``answer="done"``."""
    turns = [
        {
            "next_thought": "act",
            "tool_calls": {"tool_calls": [{"name": name, "args": args} for name, args in calls]},
        },
        {
            "next_thought": "finish",
            "tool_calls": {"tool_calls": [{"name": "submit", "args": {"answer": "done"}}]},
        },
    ]
    return DummyLM(turns, adapter=ChatAdapter())


def _results(pred: Any) -> list[Any]:
    return [
        result
        for event in pred.history.messages
        if event.get("tool_calls") is not None
        for result in event["tool_calls"].tool_call_results.tool_call_results
        if result.name != "submit"
    ]


async def _run(adapter: DSPyAdapter, query: str, *calls: tuple[str, dict[str, Any]]) -> Any:
    tools = await adapter.select(query, limit=5)
    react = dspy.ReAct("question -> answer", tools=tools, max_iters=2)
    lm = _script(*calls)
    with dspy.context(lm=lm, adapter=lm.adapter):
        return await react.acall(question=query)


def _only(pred: Any) -> Any:
    [result] = _results(pred)
    return result


async def test_failure_is_an_error_result_with_clean_text(gantry):
    pred = await _run(DSPyAdapter(gantry), "divide integer a by b", ("divide", {"a": 1, "b": 0}))
    result = _only(pred)
    assert result.is_error
    assert result.value.startswith("Execution error in divide:")
    last_line = result.value.splitlines()[-1]
    assert last_line == "agent_gantry.dspy.DSPyToolError: ZeroDivisionError: division by zero"


async def test_error_formatter_rewrites_and_hides_the_raw_message(gantry):
    adapter = DSPyAdapter(gantry, error_formatter=lambda exc: "[redacted]")
    pred = await _run(adapter, "divide integer a by b", ("divide", {"a": 1, "b": 0}))
    result = _only(pred)
    assert result.value.endswith("DSPyToolError: [redacted]")
    # DSPy's traceback shows no chained Gantry error, so nothing unredacted leaks.
    assert "division by zero" not in result.value
    assert "ToolExecutionError" not in result.value


async def test_direct_call_raises_a_tool_execution_error(gantry):
    [tool] = [t for t in await DSPyAdapter(gantry).select("divide", limit=5) if t.name == "divide"]
    with pytest.raises(DSPyToolError) as info:
        tool(a=1, b=0)
    assert isinstance(info.value, ToolExecutionError)
    assert info.value.error_type == "ZeroDivisionError"


async def test_error_mode_raise_raises_the_plain_error(gantry):
    adapter = DSPyAdapter(gantry, error_mode="raise")
    [tool] = [t for t in await adapter.select("divide", limit=5) if t.name == "divide"]
    with pytest.raises(ToolExecutionError) as info:
        tool(a=1, b=0)
    assert not isinstance(info.value, DSPyToolError)


@pytest.mark.parametrize("decision", [True, False])
async def test_approver_decides_a_gated_call_once(gantry, decision):
    asked = []

    def approve(request):
        asked.append(request)
        return decision

    pred = await _run(
        DSPyAdapter(gantry, approve=approve),
        "send an email message to bob",
        ("send_email", {"to": "bob@x"}),
    )
    [request] = asked
    assert (request.tool_name, request.arguments) == ("send_email", {"to": "bob@x"})
    assert request.description == "Send an email message to a recipient."
    assert isinstance(request.module, dspy.ReAct)
    result = _only(pred)
    if decision:
        assert gantry.sent == ["bob@x"]
        assert (result.value, result.is_error) == ("sent:bob@x", False)
    else:
        assert gantry.sent == []
        assert result.is_error and "declined" in result.value


async def test_without_an_approver_a_gated_call_is_not_run(gantry):
    pred = await _run(
        DSPyAdapter(gantry), "send an email message to bob", ("send_email", {"to": "bob@x"})
    )
    assert gantry.sent == []
    result = _only(pred)
    assert result.is_error and "requires human approval" in result.value


async def test_security_policy_pattern_also_asks_the_approver():
    g = await _build(security_policy=SecurityPolicy(require_confirmation=["delete_*"]))
    try:
        asked = []
        adapter = DSPyAdapter(g, approve=lambda request: asked.append(request.tool_name) or True)
        await _run(adapter, "delete a stored record", ("delete_record", {"record_id": "r1"}))
        assert asked == ["delete_record"]
        assert g.sent == ["deleted:r1"]
    finally:
        await g.close()


async def test_ungated_calls_never_ask_the_approver(gantry):
    asked = []
    adapter = DSPyAdapter(gantry, approve=lambda request: asked.append(request) or True)
    pred = await _run(adapter, "weather forecast for Paris", ("get_weather", {"city": "Paris"}))
    assert asked == []
    assert _only(pred).value == "sunny in Paris"


async def test_an_approver_that_raises_blocks_the_call(gantry, caplog):
    def approve(request):
        raise RuntimeError("approval service down")

    with caplog.at_level("WARNING"):
        pred = await _run(
            DSPyAdapter(gantry, approve=approve),
            "send an email message to bob",
            ("send_email", {"to": "bob@x"}),
        )
    assert gantry.sent == []
    assert _only(pred).is_error
    assert "approver" in caplog.text


async def test_error_mode_raise_never_asks_the_approver(gantry):
    asked = []
    adapter = DSPyAdapter(gantry, error_mode="raise", approve=lambda r: asked.append(r) or True)
    [tool] = [t for t in await adapter.select("send email", limit=5) if t.name == "send_email"]
    with pytest.raises(ToolExecutionError):
        tool(to="bob@x")
    assert asked == [] and gantry.sent == []


def test_approve_must_be_callable(gantry):
    with pytest.raises(ValueError, match="approve"):
        DSPyAdapter(gantry, approve="console")  # type: ignore[arg-type]


@pytest.mark.parametrize("entry", ["sync", "async"])
async def test_run_context_reaches_gantry_handlers(gantry, entry):
    tools = await DSPyAdapter(gantry).select("which DSPy module is running", limit=5)
    react = dspy.ReAct("question -> answer", tools=tools, max_iters=2)
    lm = _script(("whoami", {}))
    with dspy.context(lm=lm, adapter=lm.adapter):
        if entry == "sync":
            pred = react(question="who")
        else:
            pred = await react.acall(question="who")
    assert _only(pred).value == "ReAct/dspy"


async def test_direct_call_has_no_dspy_module(gantry):
    [tool] = [t for t in await DSPyAdapter(gantry).select("which", limit=5) if t.name == "whoami"]
    assert tool() == "NoneType/None"


async def test_agent_builder_uses_the_adapter_settings(gantry):
    asked = []
    adapter = DSPyAdapter(
        gantry,
        error_formatter=lambda exc: "[redacted]",
        approve=lambda request: asked.append(request.tool_name) or True,
    )
    builder = adapter.agent_builder("question -> answer", max_iters=2, limit=5)
    react = await builder.build("send an email message and divide integers")
    lm = _script(("send_email", {"to": "bob@x"}), ("divide", {"a": 1, "b": 0}))
    with dspy.context(lm=lm, adapter=lm.adapter):
        pred = await react.acall(question="go")
    sent, divided = _results(pred)
    assert asked == ["send_email"] and sent.value == "sent:bob@x"
    assert divided.value.endswith("DSPyToolError: [redacted]")


async def test_agent_builder_offers_pins_when_retrieval_fails(gantry, caplog):
    builder = DSPyAdapter(gantry).agent_builder(
        "question -> answer", limit=1, always_include=["get_weather"]
    )

    async def down(*args, **kwargs):
        raise RuntimeError("vector store down")

    gantry.retrieve = down
    with caplog.at_level("WARNING"):
        react = await builder.build("weather in Leeds")
    assert sorted(react.tools) == ["get_weather", "submit"]
    assert "semantic retrieval failed" in caplog.text
