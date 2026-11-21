"""Agno first-class behaviour, through a real ``Agent`` run.

A scripted ``Model`` issues the tool calls, so Agno's own run loop, tool
execution, confirmation pause / ``continue_run`` and context injection run
offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("agno.run.requirement")

from agno.agent import Agent  # noqa: E402
from agno.models.base import Model  # noqa: E402
from agno.models.response import ModelResponse  # noqa: E402

from agent_gantry import AgentGantry, current_tool_invocation  # noqa: E402
from agent_gantry.adapters.embedders.simple import SimpleEmbedder  # noqa: E402
from agent_gantry.agno import AgnoAdapter, AgnoToolError, current_run_context  # noqa: E402
from agent_gantry.core.security import SecurityPolicy  # noqa: E402
from agent_gantry.integrations.frameworks.base import ToolExecutionError  # noqa: E402


@dataclass
class ScriptedModel(Model):
    """Issues scripted tool calls, then a final answer; records the tools offered."""

    id: str = "scripted"
    name: str = "scripted"
    provider: str = "scripted"
    steps: list[list[tuple[str, dict[str, Any]]]] = field(default_factory=list)
    offered: list[list[str]] = field(default_factory=list)

    def _reply(self, tools: Any) -> ModelResponse:
        self.offered.append(
            sorted((tool.get("function") or tool).get("name", "?") for tool in tools or [])
        )
        step = self.steps.pop(0) if self.steps else None
        if not step:
            return ModelResponse(role="assistant", content="done")
        calls = [
            {
                "id": f"call_{len(self.offered)}_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args) in enumerate(step)
        ]
        return ModelResponse(role="assistant", content="", tool_calls=calls)

    def invoke(self, *args: Any, tools: Any = None, **kwargs: Any) -> ModelResponse:
        return self._reply(tools)

    async def ainvoke(self, *args: Any, tools: Any = None, **kwargs: Any) -> ModelResponse:
        return self._reply(tools)

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


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
        """Return the user and dependencies passed to the agent run."""
        ctx = current_run_context()
        invocation = current_tool_invocation()
        return (
            f"user={ctx.user_id};deps={invocation.deps};call={invocation.tool_call_id};"
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
    tools = await AgnoAdapter(gantry, **adapter_kwargs).select(query, limit=6)
    model = ScriptedModel(steps=list(steps))
    return Agent(name="clerk", model=model, tools=tools), model


def _tool(run: Any) -> Any:
    [tool] = run.tools
    return tool


async def test_tool_failure_is_a_failed_tool_result(gantry):
    agent, model = await _agent(gantry, "divide", [("divide", {"a": 1, "b": 0})])
    run = await agent.arun("divide 1 by 0")
    tool = _tool(run)
    assert tool.tool_call_error is True
    assert tool.result == "ZeroDivisionError: division by zero"
    assert len(model.offered) == 2  # the model got another turn
    assert run.content == "done"


async def test_error_formatter_rewrites_the_text(gantry):
    agent, _ = await _agent(
        gantry,
        "divide",
        [("divide", {"a": 1, "b": 0})],
        error_formatter=lambda exc: f"{exc.error_type}: [redacted]",
    )
    run = await agent.arun("divide")
    assert _tool(run).result == "ZeroDivisionError: [redacted]"


async def test_failures_are_tool_execution_errors(gantry):
    [tool] = await AgnoAdapter(gantry).select("divide", limit=1)
    with pytest.raises(AgnoToolError) as info:
        tool.entrypoint(a=1, b=0)
    assert isinstance(info.value, ToolExecutionError)
    assert info.value.tool_name == "divide"
    assert str(info.value) == "ZeroDivisionError: division by zero"


async def test_error_mode_raise_leaves_the_plain_error(gantry):
    agent, _ = await _agent(gantry, "divide", [("divide", {"a": 1, "b": 0})], error_mode="raise")
    run = await agent.arun("divide")
    tool = _tool(run)
    assert tool.tool_call_error is True
    assert "failed (status=failure)" in tool.result  # ToolExecutionError's own wording


@pytest.mark.parametrize("approve", [True, False])
async def test_gated_call_pauses_and_continues(gantry, approve):
    agent, _ = await _agent(gantry, "send an email message", [("send_email", {"to": "bob@x"})])
    run = await agent.arun("email bob")
    assert run.is_paused
    [requirement] = run.active_requirements
    assert requirement.tool_execution.tool_name == "send_email"
    assert requirement.tool_execution.tool_args == {"to": "bob@x"}
    assert gantry.sent == []

    requirement.confirm() if approve else requirement.reject()
    resumed = await agent.acontinue_run(run_response=run)
    assert gantry.sent == (["bob@x"] if approve else [])  # exactly once when approved
    assert resumed.content == "done"
    assert (_tool(resumed).result == "sent:bob@x") is approve


async def test_security_policy_pattern_also_pauses():
    g = await _build(security_policy=SecurityPolicy(require_confirmation=["delete_*"]))
    try:
        agent, _ = await _agent(
            g, "delete a stored record", [("delete_record", {"record_id": "r1"})]
        )
        run = await agent.arun("delete r1")
        [requirement] = run.active_requirements
        requirement.confirm()
        await agent.acontinue_run(run_response=run)
        assert g.sent == ["deleted:r1"]
    finally:
        await g.close()


async def test_sync_run_pauses_and_continues_too(gantry):
    agent, _ = await _agent(gantry, "send an email message", [("send_email", {"to": "bob@x"})])
    run = agent.run("email bob")
    [requirement] = run.active_requirements
    requirement.confirm()
    agent.continue_run(run_response=run)
    assert gantry.sent == ["bob@x"]


async def test_direct_call_of_a_gated_tool_is_reported_not_run(gantry):
    [tool] = await AgnoAdapter(gantry).select("send an email message", limit=1)
    assert tool.requires_confirmation is True
    with pytest.raises(AgnoToolError, match="approval"):
        tool.entrypoint(to="bob@x")
    assert gantry.sent == []


async def test_error_mode_raise_has_no_pause_and_never_runs_it(gantry):
    agent, _ = await _agent(
        gantry, "send an email message", [("send_email", {"to": "bob@x"})], error_mode="raise"
    )
    run = await agent.arun("email bob")
    assert not run.is_paused
    assert _tool(run).tool_call_error is True
    assert gantry.sent == []


async def test_run_context_reaches_gantry_handlers(gantry):
    agent, _ = await _agent(gantry, "who am I", [("whoami", {})])
    run = await agent.arun("who am I", user_id="dan", dependencies={"db": "prod"})
    assert _tool(run).result == ("user=dan;deps={'db': 'prod'};call=call_1_0;agent=clerk;fw=agno")


async def test_injected_parameters_stay_out_of_the_schema(gantry):
    [tool] = await AgnoAdapter(gantry).select("weather forecast for Paris", limit=1)
    agent = Agent(model=ScriptedModel(steps=[[("get_weather", {"city": "Paris"})]]), tools=[tool])
    run = await agent.arun("weather in Paris")
    assert _tool(run).result == "sunny in Paris"
    assert set(tool.parameters["properties"]) == {"city"}


async def test_per_run_agent_reselects_for_each_run(gantry):
    model = ScriptedModel()
    agent = AgnoAdapter(gantry).agent(model=model, limit=1)
    await agent.arun("what's the weather forecast for Paris")
    await agent.arun("divide integer a by integer b")
    assert model.offered == [["get_weather"], ["divide"]]


@pytest.mark.parametrize("approve", [True, False])
async def test_per_run_agent_resumes_an_approved_call(gantry, approve):
    model = ScriptedModel(steps=[[("send_email", {"to": "bob@x"})]])
    agent = AgnoAdapter(gantry).agent(model=model, limit=1)
    run = await agent.arun("send an email message to bob")
    [requirement] = run.active_requirements
    requirement.confirm() if approve else requirement.reject()
    await agent.acontinue_run(run_response=run)
    assert gantry.sent == (["bob@x"] if approve else [])
    assert model.offered == [["send_email"], ["send_email"]]


async def test_paused_run_outlives_many_later_runs(gantry, monkeypatch):
    from agent_gantry.integrations.frameworks import agno as agno_module

    monkeypatch.setattr(agno_module, "_MAX_REMEMBERED_RUNS", 2)
    model = ScriptedModel(steps=[[("send_email", {"to": "bob@x"})]])
    agent = AgnoAdapter(gantry).agent(model=model, limit=1)
    run = await agent.arun("send an email message to bob")
    for _ in range(5):
        await agent.arun("what's the weather forecast for Paris")
    [requirement] = run.active_requirements
    requirement.confirm()
    await agent.acontinue_run(run_response=run)
    assert gantry.sent == ["bob@x"]
    assert model.offered[-1] == ["send_email"]


async def test_finished_runs_are_forgotten(gantry):
    agent = AgnoAdapter(gantry).agent(model=ScriptedModel(), limit=1)
    selector = agent.tools.__self__
    await agent.arun("what's the weather forecast for Paris")
    assert not selector._tools and not selector._queries


async def test_another_agent_object_resumes_from_the_shared_db(gantry):
    from agno.db.in_memory import InMemoryDb

    db = InMemoryDb()
    first = AgnoAdapter(gantry).agent(
        model=ScriptedModel(steps=[[("send_email", {"to": "bob@x"})]]), limit=1, db=db
    )
    run = await first.arun("send an email message to bob", session_id="s1")
    [requirement] = run.active_requirements
    requirement.confirm()

    model = ScriptedModel()
    second = AgnoAdapter(gantry).agent(model=model, limit=1, db=db)
    await second.acontinue_run(run_id=run.run_id, session_id="s1", requirements=run.requirements)
    assert gantry.sent == ["bob@x"]
    assert model.offered == [["send_email"]]


async def test_per_run_agent_uses_the_adapter_settings(gantry):
    model = ScriptedModel(steps=[[("divide", {"a": 1, "b": 0})]])
    agent = AgnoAdapter(gantry, error_formatter=lambda exc: "[redacted]").agent(
        model=model, limit=1
    )
    run = await agent.arun("divide integer a by integer b")
    assert _tool(run).result == "[redacted]"


async def test_per_run_agent_keeps_user_pre_hooks(gantry):
    seen = []
    agent = AgnoAdapter(gantry).agent(
        model=ScriptedModel(), limit=1, pre_hooks=[lambda run_input: seen.append(run_input)]
    )
    await agent.arun("weather in Paris")
    assert len(seen) == 1


async def test_per_run_agent_keeps_user_post_hooks(gantry):
    seen = []
    agent = AgnoAdapter(gantry).agent(
        model=ScriptedModel(), limit=1, post_hooks=[lambda run_output: seen.append(run_output)]
    )
    await agent.arun("weather in Paris")
    assert len(seen) == 1


async def test_per_run_agent_offers_pins_when_retrieval_fails(gantry, caplog):
    model = ScriptedModel()
    agent = AgnoAdapter(gantry).agent(model=model, limit=1, always_include=["get_weather"])

    async def down(*args, **kwargs):
        raise RuntimeError("vector store down")

    gantry.retrieve = down
    with caplog.at_level("WARNING"):
        await agent.arun("weather in Leeds")
    assert model.offered == [["get_weather"]]
    assert "semantic retrieval failed" in caplog.text


async def test_per_run_agent_rejects_static_tools(gantry):
    with pytest.raises(TypeError, match="tools"):
        AgnoAdapter(gantry).agent(model=ScriptedModel(), tools=[])


async def test_agent_builder_uses_the_adapter_settings(gantry):
    builder = AgnoAdapter(gantry, error_formatter=lambda exc: "[redacted]").agent_builder(
        limit=1, model=ScriptedModel(steps=[[("divide", {"a": 1, "b": 0})]])
    )
    agent = await builder.build("divide integer a by integer b")
    run = await agent.arun("divide")
    assert _tool(run).result == "[redacted]"
