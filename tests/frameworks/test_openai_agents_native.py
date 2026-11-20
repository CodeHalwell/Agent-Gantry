"""OpenAI Agents SDK integration tests against the real installed ``agents`` package.

Every test drives a real ``Runner.run`` with the SDK's own ``ScriptedModel`` (no
network, no API key), so they exercise the SDK's tool runner, approval
interruptions, run state and per-turn ``is_enabled`` evaluation rather than
stand-ins. Covers: failures reaching the model as tool output, ``error_mode``,
``error_formatter``, native ``needs_approval`` (tool flag and ``SecurityPolicy``
pattern) with approve/reject, the custom-policy fallback, ``ToolContext`` in
Gantry handlers, trace propagation, and :meth:`OpenAIAgentsAdapter.dynamic_tools`
(mid-run re-selection, run isolation, sticky cap, privacy opt-out, retrieval
failure, no ``agent.tools`` mutation).
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytest.importorskip("agents.testing")
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

from agents import Agent, Runner  # noqa: E402
from agents.testing import ScriptedModel, assistant_message, function_call  # noqa: E402

from agent_gantry import AgentGantry, current_tool_invocation  # noqa: E402
from agent_gantry.adapters.embedders.simple import SimpleEmbedder  # noqa: E402
from agent_gantry.core.security import (  # noqa: E402
    ConfirmationRequiredError,
    SecurityPolicy,
)
from agent_gantry.integrations.frameworks.base import (  # noqa: E402
    ToolConfirmationRequiredError,
    ToolExecutionError,
)
from agent_gantry.openai_agents import OpenAIAgentsAdapter  # noqa: E402


async def _build(**kwargs) -> AgentGantry:
    g = AgentGantry(embedder=SimpleEmbedder(dimension=256), **kwargs)
    g.sent = []  # type: ignore[attr-defined]

    @g.register
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
    def lookup_customer(name: str) -> str:
        """Look up a customer record by name."""
        return "Customer found. Next step: send an email message to jo@example.com"

    @g.register
    def whoami() -> str:
        """Return the name of the current user from the run context."""
        from agent_gantry.openai_agents import current_run_context

        ctx = current_run_context()
        invocation = current_tool_invocation()
        return f"user={ctx.context['user']};call={invocation.tool_call_id}"

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
    yield g
    await g.close()


def _script(*steps):
    """A ``ScriptedModel`` playing ``steps`` (lists of ``(name, args)``), then "done"."""
    turns = [
        [function_call(name, args, call_id=f"call_{i}_{j}") for j, (name, args) in enumerate(step)]
        for i, step in enumerate(steps)
    ]
    return ScriptedModel([*turns, [assistant_message("done")]])


def _outputs(result) -> list[str]:
    return [i.output for i in result.new_items if type(i).__name__ == "ToolCallOutputItem"]


async def _run(tools, *steps, prompt="go", **run_kwargs):
    model = _script(*steps)
    result = await Runner.run(Agent(name="a", model=model, tools=tools), prompt, **run_kwargs)
    return result, model


# --------------------------------------------------------------------------- #
# Failures and error modes
# --------------------------------------------------------------------------- #


async def test_tool_failure_reaches_the_model_as_output(gantry):
    tools = await OpenAIAgentsAdapter(gantry).select("divide integers", limit=6)
    result, _ = await _run(tools, [("divide", {"a": 1, "b": 0})])
    assert _outputs(result) == ["Error: ZeroDivisionError: division by zero"]
    assert result.final_output == "done"


async def test_argument_errors_are_shown_so_the_model_can_correct_them(gantry):
    tools = await OpenAIAgentsAdapter(gantry).select("divide integers", limit=6)
    result, _ = await _run(tools, [("divide", {"a": "x", "b": 2})], [("divide", {"a": 4, "b": 2})])
    assert _outputs(result) == ["Error: Parameter 'a' must be an integer", "2.0"]


async def test_error_mode_raise_ends_the_run(gantry):
    tools = await OpenAIAgentsAdapter(gantry, error_mode="raise").select("divide", limit=6)
    with pytest.raises(Exception) as info:
        await _run(tools, [("divide", {"a": 1, "b": 0})])
    chain = [info.value]
    while chain[-1].__cause__ is not None:
        chain.append(chain[-1].__cause__)
    assert any(isinstance(e, ToolExecutionError) for e in chain)


async def test_error_mode_raise_with_a_gated_tool_raises_and_never_runs(gantry):
    tools = await OpenAIAgentsAdapter(gantry, error_mode="raise").select("send email", limit=6)
    with pytest.raises(Exception) as info:
        await _run(tools, [("send_email", {"to": "bob@x"})])
    chain = [info.value]
    while chain[-1].__cause__ is not None:
        chain.append(chain[-1].__cause__)
    assert any(isinstance(e, ToolConfirmationRequiredError) for e in chain)
    assert gantry.sent == []


@pytest.mark.parametrize("helper", ["run", "session"])
async def test_legacy_helpers_honour_the_adapter_error_mode(gantry, helper):
    adapter = OpenAIAgentsAdapter(gantry, error_mode="raise")
    agent = Agent(name="a", model=_script([("divide", {"a": 1, "b": 0})]), tools=[])
    with pytest.raises(Exception) as info:
        if helper == "run":
            await adapter.run(agent, "divide 1 by 0 with the divide tool", limit=6)
        else:
            await adapter.session(agent, limit=6).run("divide 1 by 0 with the divide tool")
    chain = [info.value]
    while chain[-1].__cause__ is not None:
        chain.append(chain[-1].__cause__)
    assert any(isinstance(e, ToolExecutionError) for e in chain)


async def test_error_formatter_rewrites_the_output(gantry):
    adapter = OpenAIAgentsAdapter(
        gantry, error_formatter=lambda exc: f"{exc.error_type}: [redacted]"
    )
    result, _ = await _run(await adapter.select("divide", limit=6), [("divide", {"a": 1, "b": 0})])
    assert _outputs(result) == ["Error: ZeroDivisionError: [redacted]"]


# --------------------------------------------------------------------------- #
# Human approval
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("approve", [True, False])
async def test_confirmation_gate_uses_native_approval(gantry, approve):
    tools = await OpenAIAgentsAdapter(gantry).select("send an email message", limit=6)
    model = _script([("send_email", {"to": "bob@x"})])
    agent = Agent(name="a", model=model, tools=tools)
    first = await Runner.run(agent, "email bob")
    assert [item.name for item in first.interruptions] == ["send_email"]
    assert gantry.sent == []

    state = first.to_state()
    if approve:
        state.approve(first.interruptions[0])
    else:
        state.reject(first.interruptions[0])
    second = await Runner.run(agent, state)

    assert gantry.sent == (["bob@x"] if approve else [])  # exactly once when approved
    outputs = _outputs(second)
    assert outputs and (outputs[-1] == "sent:bob@x") is approve


async def test_security_policy_pattern_also_needs_approval():
    g = await _build(security_policy=SecurityPolicy(require_confirmation=["delete_*"]))
    try:
        tools = await OpenAIAgentsAdapter(g).select("delete a stored record", limit=6)
        model = _script([("delete_record", {"record_id": "r1"})])
        agent = Agent(name="a", model=model, tools=tools)
        first = await Runner.run(agent, "delete r1")
        assert [item.name for item in first.interruptions] == ["delete_record"]
        state = first.to_state()
        state.approve(first.interruptions[0])
        await Runner.run(agent, state)
        assert g.sent == ["deleted:r1"]
    finally:
        await g.close()


async def test_custom_policy_gate_is_reported_not_run():
    class Strict(SecurityPolicy):
        def check_permission(self, tool_name, arguments, *, confirmation_approved=False, **kw):
            if tool_name == "get_weather" and not confirmation_approved:
                raise ConfirmationRequiredError("weather needs sign-off")
            return super().check_permission(
                tool_name, arguments, confirmation_approved=confirmation_approved, **kw
            )

    g = await _build(security_policy=Strict(require_confirmation=[]))
    try:
        tools = await OpenAIAgentsAdapter(g).select("weather forecast", limit=6)
        result, _ = await _run(tools, [("get_weather", {"city": "Leeds"})])
        assert "requires human approval and was not run" in _outputs(result)[0]
    finally:
        await g.close()


def test_requires_confirmation_mirrors_the_up_front_gates():
    g = AgentGantry(
        embedder=SimpleEmbedder(dimension=64),
        security_policy=SecurityPolicy(require_confirmation=["drop_*"]),
    )

    @g.register(requires_confirmation=True)
    def flagged(x: str) -> str:
        """A tool that sets its own confirmation flag."""
        return x

    @g.register
    def drop_table(name: str) -> str:
        """Drop a database table by its name."""
        return name

    @g.register
    def harmless(x: str) -> str:
        """A tool that needs no confirmation at all."""
        return x

    assert g.requires_confirmation("flagged")
    assert g.requires_confirmation("drop_table")
    assert not g.requires_confirmation("harmless")


# --------------------------------------------------------------------------- #
# Run context and tracing
# --------------------------------------------------------------------------- #


async def test_tool_context_reaches_gantry_handlers(gantry):
    tools = await OpenAIAgentsAdapter(gantry).select(
        "current user", limit=6, always_include=["whoami"]
    )
    result, _ = await _run(tools, [("whoami", {})], context={"user": "dan"})
    assert _outputs(result) == ["user=dan;call=call_0_0"]
    assert current_tool_invocation() is None


async def test_sdk_trace_becomes_the_calls_trace_parent(gantry):
    from agents import set_tracing_disabled
    from agents.tracing import trace

    events = []
    gantry.on_tool_call(events.append)
    tools = await OpenAIAgentsAdapter(gantry).select("weather forecast", limit=6)
    set_tracing_disabled(False)
    try:
        with trace("gantry-test") as t:
            await _run(tools, [("get_weather", {"city": "York"})])
    finally:
        set_tracing_disabled(True)
    assert events[-1].call.trace_id == t.trace_id
    assert events[-1].call.parent_span_id


# --------------------------------------------------------------------------- #
# dynamic_tools: per-turn selection via is_enabled
# --------------------------------------------------------------------------- #


def _offered(model) -> list[list[str]]:
    return [[t.name for t in call.tools] for call in model.calls]


async def test_dynamic_tools_reselect_mid_run_from_tool_output(gantry):
    tools = await OpenAIAgentsAdapter(gantry).dynamic_tools(limit=1)
    agent_tools_before = [t.name for t in tools]
    result, model = await _run(
        tools, [("lookup_customer", {"name": "Jo"})], prompt="look up the customer record for Jo"
    )
    offered = _offered(model)
    assert offered[0] == ["lookup_customer"]
    assert set(offered[1]) == {"lookup_customer", "send_email"}  # surfaced by the output
    assert [t.name for t in tools] == agent_tools_before  # the tool list itself is untouched


async def test_dynamic_tools_runs_are_isolated(gantry):
    tools = await OpenAIAgentsAdapter(gantry).dynamic_tools(limit=1)
    first, second = (
        ScriptedModel([[assistant_message("1")]]),
        ScriptedModel([[assistant_message("2")]]),
    )
    await asyncio.gather(
        Runner.run(Agent(name="a", model=first, tools=tools), "divide integer a by integer b"),
        Runner.run(Agent(name="b", model=second, tools=tools), "weather forecast for a city"),
    )
    assert _offered(first) == [["divide"]]
    assert _offered(second) == [["get_weather"]]


async def test_dynamic_tools_follow_up_turn_uses_the_new_message(gantry):
    tools = await OpenAIAgentsAdapter(gantry).dynamic_tools(limit=1)
    first, _ = await _run(tools, prompt="weather forecast for a city")
    model = ScriptedModel([[assistant_message("ok")]])
    history = first.to_input_list() + [
        {"role": "user", "content": "now divide integer a by integer b"}
    ]
    await Runner.run(Agent(name="a", model=model, tools=tools), history)
    assert _offered(model) == [["divide"]]


async def test_dynamic_tools_sticky_cap(gantry):
    tools = await OpenAIAgentsAdapter(gantry).dynamic_tools(limit=1, max_offered=1)
    _, model = await _run(
        tools, [("lookup_customer", {"name": "Jo"})], prompt="look up the customer record for Jo"
    )
    assert _offered(model)[1] == ["send_email"]  # carried-over lookup dropped by the cap


async def test_dynamic_tools_query_from_results_false(gantry):
    queries = []
    original = gantry.retrieve

    async def spy(query):
        queries.append(query.context.query)
        return await original(query)

    gantry.retrieve = spy
    tools = await OpenAIAgentsAdapter(gantry).dynamic_tools(limit=1, query_from_results=False)
    _, model = await _run(
        tools, [("lookup_customer", {"name": "Jo"})], prompt="look up the customer record for Jo"
    )
    assert all("jo@example.com" not in q for q in queries)
    assert _offered(model)[1] == ["lookup_customer"]


async def test_dynamic_tools_error_output_never_feeds_retrieval(gantry):
    queries = []
    original = gantry.retrieve

    async def spy(query):
        queries.append(query.context.query)
        return await original(query)

    gantry.retrieve = spy
    tools = await OpenAIAgentsAdapter(gantry).dynamic_tools(limit=1)
    await _run(tools, [("divide", {"a": 1, "b": 0})], prompt="divide 1 by 0")
    assert len(queries) >= 2  # reselected after the failed call
    assert all("ZeroDivisionError" not in q for q in queries)


@pytest.mark.parametrize("sticky", [True, False])
async def test_dynamic_tools_retrieval_failure_keeps_earlier_tools(gantry, caplog, sticky):
    original = gantry.retrieve
    calls = {"n": 0}

    async def flaky(query):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("vector store down")
        return await original(query)

    gantry.retrieve = flaky
    tools = await OpenAIAgentsAdapter(gantry).dynamic_tools(limit=1, sticky=sticky)
    with caplog.at_level("WARNING"):
        _, model = await _run(
            tools,
            [("lookup_customer", {"name": "Jo"})],
            prompt="look up the customer record for Jo",
        )
    assert _offered(model)[1] == ["lookup_customer"]
    assert "semantic retrieval failed" in caplog.text
    assert calls["n"] == 2  # one failed attempt per turn, shared by every tool's gate


async def test_dynamic_tools_pins_outside_the_namespace_filter_are_offered(gantry):
    adapter = OpenAIAgentsAdapter(gantry)
    tools = await adapter.dynamic_tools(limit=1, namespaces=["elsewhere"], required=["get_weather"])
    assert [t.name for t in tools] == ["get_weather"]
    _, model = await _run(tools, [("get_weather", {"city": "Leeds"})], prompt="weather in Leeds")
    assert _offered(model)[0] == ["get_weather"]


async def test_dynamic_tools_pins_survive_a_first_turn_retrieval_failure(gantry):
    async def down(query):
        raise RuntimeError("vector store down")

    tools = await OpenAIAgentsAdapter(gantry).dynamic_tools(limit=1, always_include=["get_weather"])
    gantry.retrieve = down
    _, model = await _run(tools, [("get_weather", {"city": "Leeds"})], prompt="weather in Leeds")
    assert _offered(model)[0] == ["get_weather"]


async def test_dynamic_tools_validate_arguments(gantry):
    with pytest.raises(ValueError, match="max_offered"):
        await OpenAIAgentsAdapter(gantry).dynamic_tools(max_offered=0)
