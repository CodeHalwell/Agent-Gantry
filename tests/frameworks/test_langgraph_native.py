"""LangChain / LangGraph first-class behaviour, through a real ``create_agent`` run.

A scripted fake chat model issues the tool calls, so the real LangGraph
``ToolNode``, middleware, ``interrupt()`` and checkpointer all run offline.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("langchain.agents")
pytest.importorskip("langgraph.checkpoint.memory")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool as lc_tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent_gantry import AgentGantry, current_tool_invocation  # noqa: E402
from agent_gantry.adapters.embedders.simple import SimpleEmbedder  # noqa: E402
from agent_gantry.core.security import SecurityPolicy  # noqa: E402
from agent_gantry.integrations.frameworks.base import (  # noqa: E402
    ToolConfirmationRequiredError,
    ToolExecutionError,
)
from agent_gantry.langchain import LangChainAdapter, current_tool_runtime  # noqa: E402
from agent_gantry.langgraph import LangGraphAdapter  # noqa: E402


class ScriptedModel(GenericFakeChatModel):
    """Replies from a script and records the tool names offered each turn."""

    offered: list[list[str]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:  # type: ignore[override]
        self.offered.append(sorted(getattr(t, "name", str(t)) for t in tools))
        return self


def _model(*steps: list[tuple[str, dict[str, Any]]]) -> ScriptedModel:
    replies = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": name, "args": args, "id": f"call_{i}_{j}"}
                for j, (name, args) in enumerate(step)
            ],
        )
        for i, step in enumerate(steps)
    ]
    model = ScriptedModel(messages=iter([*replies, AIMessage(content="done")]))
    model.offered = []
    return model


def _tool_messages(result: dict[str, Any]) -> list[ToolMessage]:
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


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

    @g.register(examples=["look up the customer record", "find a customer"])
    def lookup_customer(name: str) -> str:
        """Look up a customer record by name."""
        return "Customer found. Next step: send an email message to jo@example.com"

    @g.register
    def whoami() -> str:
        """Return the name of the current user from the run context."""
        runtime = current_tool_runtime()
        invocation = current_tool_invocation()
        return (
            f"user={runtime.context['user']};call={runtime.tool_call_id};"
            f"thread={invocation.metadata.get('thread_id')};fw={invocation.framework}"
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


async def _static_agent(gantry, query, *steps, checkpointer=None, **adapter_kwargs):
    tools = await LangChainAdapter(gantry, **adapter_kwargs).select(query, limit=6)
    model = _model(*steps)
    return create_agent(model=model, tools=tools, checkpointer=checkpointer), model


THREAD = {"configurable": {"thread_id": "t1"}}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


async def test_tool_failure_reaches_the_model_as_an_error_tool_message(gantry):
    agent, _ = await _static_agent(gantry, "divide", [("divide", {"a": 1, "b": 0})])
    result = await agent.ainvoke({"messages": [("user", "divide 1 by 0")]})
    [message] = _tool_messages(result)
    assert message.status == "error"
    assert "ZeroDivisionError" in message.content
    assert result["messages"][-1].content == "done"  # the run carried on


async def test_bad_arguments_are_shown_so_the_model_can_correct_them(gantry):
    agent, _ = await _static_agent(gantry, "divide", [("divide", {"a": "x", "b": 1})])
    result = await agent.ainvoke({"messages": [("user", "divide")]})
    [message] = _tool_messages(result)
    assert message.status == "error" and message.content


async def test_error_formatter_rewrites_the_text(gantry):
    agent, _ = await _static_agent(
        gantry,
        "divide",
        [("divide", {"a": 1, "b": 0})],
        error_formatter=lambda exc: f"{exc.error_type}: [redacted]",
    )
    result = await agent.ainvoke({"messages": [("user", "divide")]})
    assert _tool_messages(result)[0].content == "ZeroDivisionError: [redacted]"


def _chain(exc: BaseException) -> list[BaseException]:
    chain = [exc]
    while chain[-1].__cause__ is not None:
        chain.append(chain[-1].__cause__)
    return chain


async def test_error_mode_raise_propagates(gantry):
    agent, _ = await _static_agent(
        gantry, "divide", [("divide", {"a": 1, "b": 0})], error_mode="raise"
    )
    with pytest.raises(Exception) as info:
        await agent.ainvoke({"messages": [("user", "divide")]})
    assert any(isinstance(e, ToolExecutionError) for e in _chain(info.value))


# --------------------------------------------------------------------------- #
# Human approval
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("decision", [True, {"approved": True}, False, "reject", "yes"])
async def test_confirmation_gate_interrupts_and_resumes(gantry, decision):
    agent, _ = await _static_agent(
        gantry,
        "send an email message",
        [("send_email", {"to": "bob@x"})],
        checkpointer=InMemorySaver(),
    )
    first = await agent.ainvoke({"messages": [("user", "email bob")]}, THREAD)
    [pending] = first["__interrupt__"]
    assert pending.value["tool"] == "send_email"
    assert pending.value["arguments"] == {"to": "bob@x"}
    assert pending.value["tool_call_id"] == "call_0_0"
    assert gantry.sent == []

    second = await agent.ainvoke(Command(resume=decision), THREAD)
    approved = decision in (True, {"approved": True})
    assert gantry.sent == (["bob@x"] if approved else [])  # exactly once when approved
    [message] = _tool_messages(second)
    assert (message.content == "sent:bob@x") is approved
    if not approved:
        assert message.status == "error" and "rejected" in message.content


async def test_security_policy_pattern_also_interrupts():
    g = await _build(security_policy=SecurityPolicy(require_confirmation=["delete_*"]))
    try:
        agent, _ = await _static_agent(
            g,
            "delete a stored record",
            [("delete_record", {"record_id": "r1"})],
            checkpointer=InMemorySaver(),
        )
        first = await agent.ainvoke({"messages": [("user", "delete r1")]}, THREAD)
        assert first["__interrupt__"][0].value["tool"] == "delete_record"
        await agent.ainvoke(Command(resume=True), THREAD)
        assert g.sent == ["deleted:r1"]
    finally:
        await g.close()


async def test_without_a_checkpointer_the_gated_call_is_reported_not_run(gantry):
    agent, _ = await _static_agent(
        gantry, "send an email message", [("send_email", {"to": "bob@x"})]
    )
    result = await agent.ainvoke({"messages": [("user", "email bob")]})
    [message] = _tool_messages(result)
    assert message.status == "error" and "approval" in message.content
    assert gantry.sent == []


async def test_error_mode_raise_with_a_gated_tool_raises_and_never_runs(gantry):
    agent, _ = await _static_agent(
        gantry,
        "send an email message",
        [("send_email", {"to": "bob@x"})],
        checkpointer=InMemorySaver(),
        error_mode="raise",
    )
    with pytest.raises(Exception) as info:
        await agent.ainvoke({"messages": [("user", "email bob")]}, THREAD)
    assert any(isinstance(e, ToolConfirmationRequiredError) for e in _chain(info.value))
    assert gantry.sent == []


# --------------------------------------------------------------------------- #
# Run context
# --------------------------------------------------------------------------- #


async def test_runtime_context_reaches_gantry_handlers(gantry):
    tools = await LangGraphAdapter(gantry).select("who am I signed in as", limit=6)
    agent = create_agent(
        model=_model([("whoami", {})]),
        tools=tools,
        checkpointer=InMemorySaver(),
        context_schema=dict,
    )
    result = await agent.ainvoke(
        {"messages": [("user", "who am I")]}, THREAD, context={"user": "dan"}
    )
    assert _tool_messages(result)[0].content == ("user=dan;call=call_0_0;thread=t1;fw=langgraph")


def test_sync_invoke_path_uses_the_same_behaviour(gantry):
    async def tools():
        return await LangChainAdapter(gantry).select("divide", limit=6)

    agent = create_agent(model=_model([("divide", {"a": 1, "b": 0})]), tools=asyncio.run(tools()))
    result = agent.invoke({"messages": [("user", "divide")]})
    assert _tool_messages(result)[0].status == "error"


# --------------------------------------------------------------------------- #
# Per-turn selection (middleware)
# --------------------------------------------------------------------------- #


async def _dynamic(gantry, *steps, prompt, middleware_kwargs=None, tools=(), **run):
    middleware = await LangGraphAdapter(gantry).middleware(**(middleware_kwargs or {}))
    model = _model(*steps)
    agent = create_agent(model=model, tools=list(tools), middleware=[middleware], **run)
    result = await agent.ainvoke({"messages": [("user", prompt)]}, THREAD if run else None)
    return result, model


async def test_middleware_reselects_mid_run_from_tool_output(gantry):
    _, model = await _dynamic(
        gantry,
        [("lookup_customer", {"name": "Jo"})],
        prompt="look up the customer record for Jo",
        middleware_kwargs={"limit": 1, "sticky": False},
    )
    assert model.offered[0] == ["lookup_customer"]
    assert model.offered[1] == ["send_email"]  # chosen from the lookup's output


async def test_middleware_sticky_keeps_earlier_tools(gantry):
    _, model = await _dynamic(
        gantry,
        [("lookup_customer", {"name": "Jo"})],
        prompt="look up the customer record for Jo",
        middleware_kwargs={"limit": 1},
    )
    assert model.offered[1] == ["lookup_customer", "send_email"]


async def test_middleware_sticky_cap(gantry):
    _, model = await _dynamic(
        gantry,
        [("lookup_customer", {"name": "Jo"})],
        prompt="look up the customer record for Jo",
        middleware_kwargs={"limit": 1, "max_offered": 1},
    )
    assert model.offered[1] == ["send_email"]


async def test_middleware_failed_output_never_feeds_retrieval(gantry):
    queries = []
    original = gantry.retrieve

    async def spy(query):
        queries.append(query.context.query)
        return await original(query)

    gantry.retrieve = spy
    await _dynamic(
        gantry,
        [("divide", {"a": 1, "b": 0})],
        prompt="divide 1 by 0",
        middleware_kwargs={"limit": 1},
    )
    assert queries and all("ZeroDivisionError" not in q for q in queries)


def test_failed_dict_tool_messages_never_drive_the_query():
    from agent_gantry.integrations.frameworks.langgraph_live import _query_from_state

    state = {
        "messages": [
            {"role": "user", "content": "look up the customer record"},
            {"role": "tool", "status": "error", "content": "Traceback: boom"},
        ]
    }
    assert _query_from_state(state) == "look up the customer record"


async def test_middleware_query_from_results_false(gantry):
    _, model = await _dynamic(
        gantry,
        [("lookup_customer", {"name": "Jo"})],
        prompt="look up the customer record for Jo",
        middleware_kwargs={"limit": 1, "sticky": False, "query_from_results": False},
    )
    assert model.offered[1] == ["lookup_customer"]


async def test_middleware_retrieval_failure_keeps_earlier_tools_and_pins(gantry, caplog):
    original = gantry.retrieve
    calls = {"n": 0}

    async def flaky(query):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("vector store down")
        return await original(query)

    gantry.retrieve = flaky
    with caplog.at_level("WARNING"):
        _, model = await _dynamic(
            gantry,
            [("lookup_customer", {"name": "Jo"})],
            prompt="look up the customer record for Jo",
            middleware_kwargs={"limit": 1, "sticky": False, "always_include": ["get_weather"]},
        )
    assert model.offered[1] == ["get_weather", "lookup_customer"]
    assert "semantic retrieval failed" in caplog.text


async def test_middleware_first_turn_failure_still_offers_pins(gantry):
    async def down(query):
        raise RuntimeError("vector store down")

    middleware = await LangGraphAdapter(gantry).middleware(limit=1, always_include=["get_weather"])
    gantry.retrieve = down
    model = _model([("get_weather", {"city": "Leeds"})])
    agent = create_agent(model=model, middleware=[middleware])
    result = await agent.ainvoke({"messages": [("user", "weather in Leeds")]})
    assert model.offered[0] == ["get_weather"]
    assert _tool_messages(result)[0].content == "sunny in Leeds"


async def test_middleware_pins_outside_the_namespace_filter(gantry):
    middleware = await LangGraphAdapter(gantry).middleware(
        limit=1, namespaces=["elsewhere"], required=["get_weather"]
    )
    assert [t.name for t in middleware.tools] == ["get_weather"]


async def test_middleware_keeps_the_callers_own_tools(gantry):
    @lc_tool
    def ping() -> str:
        """Reply with pong."""
        return "pong"

    _, model = await _dynamic(
        gantry,
        [("ping", {})],
        prompt="what's the weather in London",
        middleware_kwargs={"limit": 1, "sticky": False},
        tools=[ping],
    )
    assert all("ping" in offered for offered in model.offered)
    assert model.offered[0] == ["get_weather", "ping"]


async def test_middleware_runs_are_isolated(gantry):
    middleware = await LangGraphAdapter(gantry).middleware(limit=1)

    async def run(prompt, thread):
        model = _model()
        agent = create_agent(model=model, middleware=[middleware], checkpointer=InMemorySaver())
        await agent.ainvoke(
            {"messages": [("user", prompt)]}, {"configurable": {"thread_id": thread}}
        )
        return model.offered

    weather, crm = await asyncio.gather(
        run("what's the weather in London", "a"), run("look up the customer record", "b")
    )
    assert weather == [["get_weather"]]
    assert crm == [["lookup_customer"]]


async def test_areact_agent_builds_a_per_turn_agent(gantry):
    model = _model([("get_weather", {"city": "Paris"})])
    agent = await LangGraphAdapter(gantry).areact_agent(model, limit=1)
    result = await agent.ainvoke({"messages": [("user", "what's the weather in Paris")]})
    assert model.offered[0] == ["get_weather"]
    assert _tool_messages(result)[0].content == "sunny in Paris"


async def test_middleware_validates_arguments(gantry):
    with pytest.raises(ValueError, match="max_offered"):
        await LangGraphAdapter(gantry).middleware(limit=3, max_offered=1)
    with pytest.raises(ValueError, match="error_mode"):
        LangGraphAdapter(gantry, error_mode="explode")  # type: ignore[arg-type]
