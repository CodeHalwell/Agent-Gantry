"""Strands Agents first-class behaviour, through a real ``Agent`` run.

A scripted ``Model`` streams tool-use events, so Strands' own event loop,
tool execution, interrupts and hooks run offline.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("strands.interrupt")

from strands import Agent  # noqa: E402
from strands.models.model import Model  # noqa: E402

from agent_gantry import AgentGantry, current_tool_invocation  # noqa: E402
from agent_gantry.adapters.embedders.simple import SimpleEmbedder  # noqa: E402
from agent_gantry.core.security import SecurityPolicy  # noqa: E402
from agent_gantry.integrations.frameworks.base import ToolExecutionError  # noqa: E402
from agent_gantry.strands import (  # noqa: E402
    APPROVAL_INTERRUPT,
    StrandsAdapter,
    current_tool_context,
)


class ScriptedModel(Model):
    """Streams scripted tool calls, then a final answer; records tools offered."""

    def __init__(self, *steps: list[tuple[str, dict[str, Any]]]) -> None:
        self.steps = list(steps)
        self.offered: list[list[str]] = []

    def update_config(self, **kwargs: Any) -> None:
        pass

    def get_config(self) -> dict[str, Any]:
        return {}

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def stream(self, messages: Any, tool_specs: Any = None, *args: Any, **kwargs: Any) -> Any:
        self.offered.append(sorted(spec["name"] for spec in tool_specs or []))
        step = self.steps.pop(0) if self.steps else None
        yield {"messageStart": {"role": "assistant"}}
        if step:
            for i, (name, args_) in enumerate(step):
                call_id = f"call_{len(self.offered)}_{i}"
                yield {
                    "contentBlockStart": {
                        "start": {"toolUse": {"toolUseId": call_id, "name": name}}
                    }
                }
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args_)}}}}
                yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": "done"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


def _results(agent: Any) -> list[dict[str, Any]]:
    return [
        block["toolResult"]
        for message in agent.messages
        for block in message["content"]
        if "toolResult" in block
    ]


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
        """Return the user passed to the agent call."""
        ctx = current_tool_context()
        invocation = current_tool_invocation()
        return (
            f"user={ctx.invocation_state['user']};call={invocation.tool_call_id};"
            f"agent={invocation.metadata.get('agent_name')};fw={invocation.framework}"
        )

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


async def _agent(gantry, query, *steps, **adapter_kwargs):
    tools = await StrandsAdapter(gantry, **adapter_kwargs).select(query, limit=6)
    model = ScriptedModel(*steps)
    return Agent(name="clerk", model=model, tools=tools, callback_handler=None), model


async def test_tool_failure_is_an_error_tool_result(gantry):
    agent, model = await _agent(gantry, "divide", [("divide", {"a": 1, "b": 0})])
    await agent.invoke_async("divide 1 by 0")
    [result] = _results(agent)
    assert result["status"] == "error"
    assert result["content"][0]["text"] == "ZeroDivisionError: division by zero"
    assert len(model.offered) == 2  # the model got another turn


async def test_error_formatter_rewrites_the_text(gantry):
    agent, _ = await _agent(
        gantry,
        "divide",
        [("divide", {"a": 1, "b": 0})],
        error_formatter=lambda exc: f"{exc.error_type}: [redacted]",
    )
    await agent.invoke_async("divide")
    assert _results(agent)[0]["content"][0]["text"] == "ZeroDivisionError: [redacted]"


async def test_error_mode_raise_leaves_it_to_strands(gantry):
    agent, _ = await _agent(gantry, "divide", [("divide", {"a": 1, "b": 0})], error_mode="raise")
    await agent.invoke_async("divide")
    [result] = _results(agent)
    assert result["status"] == "error"
    assert "ToolExecutionError" in result["content"][0]["text"]  # Strands' own wording


@pytest.mark.parametrize("response", ["yes", True, {"approved": True}, "no", False])
async def test_gated_call_interrupts_and_resumes(gantry, response):
    agent, _ = await _agent(gantry, "send an email message", [("send_email", {"to": "bob@x"})])
    first = await agent.invoke_async("email bob")
    assert first.stop_reason == "interrupt"
    [interrupt] = first.interrupts
    assert interrupt.name == APPROVAL_INTERRUPT
    assert interrupt.reason["tool"] == "send_email"
    assert interrupt.reason["arguments"] == {"to": "bob@x"}
    assert gantry.sent == []

    await agent.invoke_async(
        [{"interruptResponse": {"interruptId": interrupt.id, "response": response}}]
    )
    approved = response in ("yes", True) or response == {"approved": True}
    assert gantry.sent == (["bob@x"] if approved else [])  # exactly once when approved
    [result] = _results(agent)
    assert (result["status"] == "success") is approved
    if not approved:
        assert "rejected" in result["content"][0]["text"]


async def test_security_policy_pattern_also_interrupts():
    g = await _build(security_policy=SecurityPolicy(require_confirmation=["delete_*"]))
    try:
        agent, _ = await _agent(
            g, "delete a stored record", [("delete_record", {"record_id": "r1"})]
        )
        first = await agent.invoke_async("delete r1")
        [interrupt] = first.interrupts
        await agent.invoke_async(
            [{"interruptResponse": {"interruptId": interrupt.id, "response": "yes"}}]
        )
        assert g.sent == ["deleted:r1"]
    finally:
        await g.close()


async def test_direct_call_of_a_gated_tool_is_reported_not_run(gantry):
    [tool] = await StrandsAdapter(gantry).select("send an email message", limit=1)
    result = await tool(to="bob@x")
    assert result["status"] == "error" and "approval" in result["content"][0]["text"]
    assert gantry.sent == []
    [raising] = await StrandsAdapter(gantry, error_mode="raise").select("send an email", limit=1)
    with pytest.raises(ToolExecutionError):
        await raising(to="bob@x")


async def test_tool_context_reaches_gantry_handlers(gantry):
    agent, _ = await _agent(gantry, "who am I", [("whoami", {})])
    await agent.invoke_async("who am I", user="dan")
    assert _results(agent)[0]["content"][0]["text"] == (
        "user=dan;call=call_1_0;agent=clerk;fw=strands"
    )


async def test_per_turn_hook_tools_use_the_adapter_settings(gantry):
    adapter = StrandsAdapter(gantry, error_formatter=lambda exc: "[redacted]")
    model = ScriptedModel([("divide", {"a": 1, "b": 0})])
    agent = adapter.agent(model=model, limit=1, callback_handler=None)
    await agent.invoke_async("divide 10 by 2")
    assert _results(agent)[0]["content"][0]["text"] == "[redacted]"


async def test_per_turn_hook_tools_interrupt_for_approval(gantry):
    model = ScriptedModel([("send_email", {"to": "bob@x"})])
    agent = StrandsAdapter(gantry).agent(model=model, limit=1, callback_handler=None)
    first = await agent.invoke_async("send an email message to bob")
    assert first.stop_reason == "interrupt"
    [interrupt] = first.interrupts
    await agent.invoke_async(
        [{"interruptResponse": {"interruptId": interrupt.id, "response": "yes"}}]
    )
    assert gantry.sent == ["bob@x"]


async def test_a_tool_argument_named_tool_context_is_kept():
    g = AgentGantry(embedder=SimpleEmbedder(dimension=256))

    @g.register
    def annotate(tool_context: str) -> str:
        """Annotate the given tool context string."""
        ctx = current_tool_context()
        return f"note:{tool_context}:{getattr(ctx, 'agent', None) is not None}"

    await g.sync()
    try:
        [tool] = await StrandsAdapter(g).select("annotate the tool context string", limit=1)
        model = ScriptedModel([("annotate", {"tool_context": "hello"})])
        agent = Agent(name="clerk", model=model, tools=[tool], callback_handler=None)
        await agent.invoke_async("annotate hello")
        assert _results(agent)[0]["content"][0]["text"] == "note:hello:True"
    finally:
        await g.close()
