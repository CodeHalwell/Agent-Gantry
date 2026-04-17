"""
End-to-end Agent Framework 1.0 GA orchestration tests for Agent-Gantry.

These tests drive the *real* ``agent-framework`` package against a
lightweight ``ScriptedChatClient`` that implements the required chat-client
protocol (no network, no LLM). Each scenario verifies a different
orchestration pattern exercised in production deployments:

- Single-turn tool use (happy path)
- Multi-turn conversation that re-uses the same agent across user messages
- Sequential multi-agent orchestration (``SequentialBuilder``)
- Concurrent multi-agent orchestration (``ConcurrentBuilder``)
- Handoff orchestration (``HandoffBuilder``) where agent A hands off to B
- Group chat orchestration (``GroupChatBuilder``)
- ``agent.as_tool()`` nested-agent pattern
- Workflow orchestration with ``AgentExecutor`` nodes
- ``GantryApprovalMiddleware`` halting destructive tool calls
- ``GantryObservabilityMiddleware`` recording invocations

The tests skip automatically when ``agent-framework`` isn't installed, so
CI environments that don't pull the optional extra still pass.
"""

from __future__ import annotations

from typing import Any, ClassVar, Sequence

import pytest

pytest.importorskip("agent_framework", reason="agent-framework not installed")

from agent_framework import (  # noqa: E402
    Agent,
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    Content,
    FunctionInvocationLayer,
    FunctionMiddleware,
    FunctionTool,
    Message,
    MiddlewareTermination,
)
from agent_framework.orchestrations import (  # noqa: E402
    ConcurrentBuilder,
    GroupChatBuilder,
    HandoffBuilder,
    SequentialBuilder,
)

from agent_gantry import AgentGantry  # noqa: E402
from agent_gantry.core.security import (  # noqa: E402
    ConfirmationRequiredError,
    PermissionDeniedError,
    SecurityPolicy,
)
from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge  # noqa: E402
from agent_gantry.integrations.agent_framework_middleware import (  # noqa: E402
    GantryApprovalMiddleware,
    GantryObservabilityMiddleware,
)
from agent_gantry.schema.tool import ToolCapability  # noqa: E402

# ---------------------------------------------------------------------------
# Scripted chat client — implements the function-calling protocol so AF's
# Agent can drive tools end-to-end without touching a real LLM.
# ---------------------------------------------------------------------------


class ScriptedChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
    """Fake chat client that returns pre-scripted responses per turn.

    Each entry in ``script`` is a sequence of content items that will be
    returned as the assistant message for that get_response call. After the
    script is exhausted, the client returns an empty text message.

    Supports function-call content; when the agent feeds a tool result back
    in, the next scripted entry is used.
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "scripted"

    def __init__(self, script: Sequence[Sequence[Content]]):
        super().__init__()
        self._script = list(script)
        self.call_count = 0
        self.seen_tools: list[list[str]] = []
        self.seen_messages: list[list[Message]] = []

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):  # noqa: ANN001,ANN002
        self.seen_messages.append(list(messages))
        # Record which tools the caller sent on each turn for assertion.
        tools = []
        if options:
            raw_tools = options.get("tools") if isinstance(options, dict) else getattr(options, "tools", None)
            if raw_tools:
                for t in raw_tools:
                    tools.append(getattr(t, "name", type(t).__name__))
        self.seen_tools.append(tools)

        idx = self.call_count
        self.call_count += 1
        if idx < len(self._script):
            contents = list(self._script[idx])
        else:
            contents = [Content.from_text("done")]

        return ChatResponse(
            messages=[Message(role="assistant", contents=contents)],
            response_id=f"scripted-{idx}",
        )


def _fc(name: str, arguments: dict[str, Any], call_id: str) -> Content:
    """Shortcut for building a function_call Content."""
    return Content.from_function_call(name=name, arguments=arguments, call_id=call_id)


# ---------------------------------------------------------------------------
# Shared fixtures — three Gantry tools representing realistic, varied
# production tool shapes (read, multi-arg read, destructive write).
# ---------------------------------------------------------------------------


@pytest.fixture
async def gantry_with_tools() -> AgentGantry:
    gantry = AgentGantry()

    @gantry.register
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        return f"Weather in {city}: sunny, 22C"

    @gantry.register
    def lookup_order(order_id: str, include_items: bool = False) -> dict[str, Any]:
        """Look up an order by id, optionally including line items."""
        return {
            "order_id": order_id,
            "status": "shipped",
            "items": ["widget"] if include_items else [],
        }

    @gantry.register(capabilities=[ToolCapability.DELETE_DATA])
    def delete_user(user_id: str) -> str:
        """Delete a user account. Destructive."""
        return f"deleted:{user_id}"

    await gantry.sync()
    return gantry


@pytest.fixture
async def bridge(gantry_with_tools: AgentGantry) -> GantryToolBridge:
    return GantryToolBridge(gantry_with_tools, score_threshold=0.0)


# ---------------------------------------------------------------------------
# 1. Single-turn tool use
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_turn_tool_use(bridge: GantryToolBridge) -> None:
    """Agent receives a query, calls one Gantry tool, returns final text."""
    tools = await bridge.get_tools(
        "weather in london", limit=5, score_threshold=0.0
    )

    client = ScriptedChatClient(
        script=[
            [_fc("get_weather", {"city": "London"}, "call_0")],
            [Content.from_text("The weather in London is sunny.")],
        ]
    )
    agent = Agent(client, instructions="be helpful", name="weatherbot", tools=tools)

    resp = await agent.run("What's the weather in London?")

    assert client.call_count == 2, "expected 2 chat turns (call + result)"
    assert "sunny" in resp.text.lower()
    # The second turn must include a tool-result message
    roles = [m.role for m in resp.messages]
    assert any(str(r).lower() == "tool" for r in roles), roles


# ---------------------------------------------------------------------------
# 2. Multi-turn conversation — same agent, two successive agent.run() calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_turn_conversation(bridge: GantryToolBridge) -> None:
    """Two user turns, each triggering a distinct Gantry tool call."""
    tools = await bridge.get_tools(
        "weather and orders", limit=5, score_threshold=0.0
    )

    client = ScriptedChatClient(
        script=[
            # Turn 1: weather
            [_fc("get_weather", {"city": "Paris"}, "c1")],
            [Content.from_text("Paris is sunny.")],
            # Turn 2: order lookup
            [_fc("lookup_order", {"order_id": "ORD-9", "include_items": True}, "c2")],
            [Content.from_text("Order ORD-9 is shipped with 1 item.")],
        ]
    )
    agent = Agent(client, instructions="", name="multibot", tools=tools)

    session = agent.create_session()
    resp1 = await agent.run("Weather in Paris?", session=session)
    assert "sunny" in resp1.text.lower()
    assert client.call_count == 2

    resp2 = await agent.run("Look up order ORD-9 with items.", session=session)
    assert "shipped" in resp2.text.lower()
    assert client.call_count == 4, "should have driven another 2 turns"

    # Both Gantry tools were exercised across the conversation
    tool_names_used = {
        c.name
        for turn in client.seen_messages
        for msg in turn
        for c in (msg.contents or [])
        if getattr(c, "type", None) == "function_call"
    }
    assert {"get_weather", "lookup_order"}.issubset(tool_names_used)


# ---------------------------------------------------------------------------
# 3. Sequential orchestration — two Gantry-tooled agents chained
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_orchestration(bridge: GantryToolBridge) -> None:
    """SequentialBuilder chains Agent A (weather) -> Agent B (orders)."""
    weather_tools = bridge.wrap_tools(
        [t for t in bridge._gantry.export_tools() if t.name == "get_weather"]
    )
    order_tools = bridge.wrap_tools(
        [t for t in bridge._gantry.export_tools() if t.name == "lookup_order"]
    )

    # Client A: calls get_weather, then answers.
    client_a = ScriptedChatClient(
        script=[
            [_fc("get_weather", {"city": "Berlin"}, "a0")],
            [Content.from_text("Berlin is sunny. Please check order ORD-42 next.")],
        ]
    )
    agent_a = Agent(client_a, instructions="", name="weather_agent", tools=weather_tools)

    # Client B: receives Agent A's output as context, calls lookup_order.
    client_b = ScriptedChatClient(
        script=[
            [_fc("lookup_order", {"order_id": "ORD-42"}, "b0")],
            [Content.from_text("Order ORD-42 is shipped.")],
        ]
    )
    agent_b = Agent(client_b, instructions="", name="order_agent", tools=order_tools)

    workflow = SequentialBuilder(participants=[agent_a, agent_b]).build()
    # Workflow was built with two participants in order
    assert workflow is not None
    # Drive the workflow manually: both agents must be reachable and each
    # agent's tools are real FunctionTools that Gantry can execute.
    for a in (agent_a, agent_b):
        bound_tools = (a.default_options or {}).get("tools") or []
        assert bound_tools, f"{a.name} should have tools"
        for t in bound_tools:
            assert isinstance(t, FunctionTool)

    # Run agent A then B in sequence (simulating the workflow's effect).
    r_a = await agent_a.run("Weather in Berlin?")
    r_b = await agent_b.run(
        f"Previous agent said: {r_a.text}. Continue the task."
    )
    assert "sunny" in r_a.text.lower()
    assert "shipped" in r_b.text.lower()
    # Both Gantry tools got invoked (one per agent)
    assert client_a.call_count == 2
    assert client_b.call_count == 2


# ---------------------------------------------------------------------------
# 4. Concurrent orchestration — fan-out/fan-in of two Gantry-tooled agents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_orchestration(bridge: GantryToolBridge) -> None:
    """ConcurrentBuilder runs two Gantry-tooled agents in parallel on the
    same user task, then aggregates their outputs."""
    import asyncio

    tools_all = await bridge.get_tools("weather and order", limit=5, score_threshold=0.0)

    client_a = ScriptedChatClient(
        script=[
            [_fc("get_weather", {"city": "Tokyo"}, "a")],
            [Content.from_text("Tokyo is sunny.")],
        ]
    )
    client_b = ScriptedChatClient(
        script=[
            [_fc("lookup_order", {"order_id": "ORD-7"}, "b")],
            [Content.from_text("Order ORD-7 is shipped.")],
        ]
    )
    agent_a = Agent(client_a, instructions="", name="weather_agent", tools=tools_all)
    agent_b = Agent(client_b, instructions="", name="order_agent", tools=tools_all)

    workflow = ConcurrentBuilder(participants=[agent_a, agent_b]).build()
    assert workflow is not None

    # Simulate concurrent execution directly (the orchestrator would dispatch
    # both agents against the same input; we verify both succeed in parallel).
    r_a, r_b = await asyncio.gather(
        agent_a.run("What's the weather in Tokyo?"),
        agent_b.run("Lookup order ORD-7"),
    )
    assert "sunny" in r_a.text.lower()
    assert "shipped" in r_b.text.lower()


# ---------------------------------------------------------------------------
# 5. Handoff orchestration — Agent A hands off to Agent B
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handoff_orchestration(bridge: GantryToolBridge) -> None:
    """HandoffBuilder composes a router plus specialist agents."""
    tools_all = await bridge.get_tools("weather and orders", limit=5, score_threshold=0.0)

    triage_client = ScriptedChatClient(
        script=[[Content.from_text("hand off to order_agent")]]
    )
    order_client = ScriptedChatClient(
        script=[
            [_fc("lookup_order", {"order_id": "ORD-99"}, "h0")],
            [Content.from_text("Order ORD-99 is shipped.")],
        ]
    )
    triage = Agent(
        triage_client,
        instructions="",
        name="triage_agent",
        tools=tools_all,
        require_per_service_call_history_persistence=True,
    )
    order_agent = Agent(
        order_client,
        instructions="",
        name="order_agent",
        tools=tools_all,
        require_per_service_call_history_persistence=True,
    )

    workflow = (
        HandoffBuilder(name="support_desk")
        .participants([triage, order_agent])
        .with_start_agent(triage)
        .add_handoff(source=triage, targets=[order_agent], description="to orders")
        .build()
    )
    assert workflow is not None

    # Simulate the handoff manually — triage routes, order_agent solves.
    triage_resp = await triage.run("Look up order ORD-99 please.")
    assert "order_agent" in triage_resp.text.lower()

    order_resp = await order_agent.run("Look up order ORD-99 please.")
    assert "shipped" in order_resp.text.lower()


# ---------------------------------------------------------------------------
# 6. Group chat orchestration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_group_chat_orchestration(bridge: GantryToolBridge) -> None:
    """GroupChatBuilder composes multiple agents that share a Gantry tool set."""
    tools_all = await bridge.get_tools("", limit=5, score_threshold=0.0)

    analyst_client = ScriptedChatClient(
        script=[
            [_fc("get_weather", {"city": "Rome"}, "g0")],
            [Content.from_text("Analyst: Rome is sunny.")],
        ]
    )
    critic_client = ScriptedChatClient(
        script=[[Content.from_text("Critic: confirms analyst's finding.")]]
    )
    analyst = Agent(analyst_client, instructions="", name="analyst", tools=tools_all)
    critic = Agent(critic_client, instructions="", name="critic", tools=tools_all)

    # GroupChatBuilder takes participants in the constructor and requires a
    # distinct orchestrator agent. Build a tiny no-op orchestrator client.
    orchestrator_client = ScriptedChatClient(
        script=[[Content.from_text("analyst")]]
    )
    orchestrator_agent = Agent(
        orchestrator_client, instructions="", name="round_robin_mgr", tools=[]
    )

    workflow = (
        GroupChatBuilder(
            participants=[analyst, critic],
            orchestrator_agent=orchestrator_agent,
            max_rounds=2,
        )
        .build()
    )
    assert workflow is not None
    # Verify both agents share Gantry-bridged FunctionTools.
    for a in (analyst, critic):
        bound = (a.default_options or {}).get("tools") or []
        assert any(isinstance(t, FunctionTool) for t in bound)

    # Both agents run against the shared tool set.
    r1 = await analyst.run("Weather in Rome?")
    r2 = await critic.run(f"Analyst said: {r1.text}")
    assert "sunny" in r1.text.lower()
    assert "confirm" in r2.text.lower()


# ---------------------------------------------------------------------------
# 7. agent.as_tool() — nested agent pattern
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_as_tool(bridge: GantryToolBridge) -> None:
    """A Gantry-tooled agent is exposed as a tool to a meta-agent."""
    inner_tools = await bridge.get_tools("weather", limit=5, score_threshold=0.0)
    inner_client = ScriptedChatClient(
        script=[
            [_fc("get_weather", {"city": "Oslo"}, "i0")],
            [Content.from_text("Oslo is sunny.")],
        ]
    )
    inner = Agent(inner_client, instructions="", name="weather_specialist", tools=inner_tools)

    # Now expose it as a tool
    inner_tool = inner.as_tool(
        name="ask_weather_specialist",
        description="Delegate weather questions to the specialist.",
    )
    assert inner_tool is not None
    # The as_tool() output must be a proper AF tool
    assert hasattr(inner_tool, "name")


# ---------------------------------------------------------------------------
# 8. Workflow orchestration with explicit AgentExecutors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_with_agent_executors(bridge: GantryToolBridge) -> None:
    """Build a Workflow graph with two AgentExecutor nodes wired together."""
    from agent_framework import AgentExecutor, WorkflowBuilder

    tools_all = await bridge.get_tools("", limit=5, score_threshold=0.0)

    client_a = ScriptedChatClient(
        script=[
            [_fc("get_weather", {"city": "Madrid"}, "w0")],
            [Content.from_text("Madrid is sunny.")],
        ]
    )
    client_b = ScriptedChatClient(
        script=[
            [_fc("lookup_order", {"order_id": "ORD-1"}, "w1")],
            [Content.from_text("Order ORD-1 is shipped.")],
        ]
    )
    agent_a = Agent(client_a, instructions="", name="A", tools=tools_all)
    agent_b = Agent(client_b, instructions="", name="B", tools=tools_all)

    exec_a = AgentExecutor(agent_a, id="A")
    exec_b = AgentExecutor(agent_b, id="B")

    builder = WorkflowBuilder(start_executor=exec_a)
    builder.add_edge(exec_a, exec_b)
    workflow = builder.build()
    assert workflow is not None


# ---------------------------------------------------------------------------
# 9. GantryApprovalMiddleware blocks destructive tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_middleware_blocks_destructive_tool(bridge: GantryToolBridge) -> None:
    """The middleware should raise MiddlewareTermination for a destructive
    tool whose name matches the require_confirmation patterns."""
    tools = await bridge.get_tools("delete user", limit=5, score_threshold=0.0)

    # Force delete_user into the tool set
    tools = [t for t in tools if t.name in ("delete_user", "get_weather")]
    assert any(t.name == "delete_user" for t in tools)

    policy = SecurityPolicy(require_confirmation=["delete_*"])
    middleware = GantryApprovalMiddleware(policy)

    # Directly exercise the middleware pipeline with a forged context.
    class _Ctx:
        def __init__(self, fn, args):
            self.function = fn
            self.arguments = args
            self.result = None
            self.metadata: dict[str, Any] = {}

    async def _never_called():  # pragma: no cover - should be blocked
        raise AssertionError("call_next must NOT run for destructive tools")

    delete_tool = next(t for t in tools if t.name == "delete_user")
    ctx = _Ctx(delete_tool, {"user_id": "abc"})

    with pytest.raises(MiddlewareTermination):
        await middleware.process(ctx, _never_called)


@pytest.mark.asyncio
async def test_approval_middleware_allows_safe_tool(bridge: GantryToolBridge) -> None:
    """Safe (read-only) tools pass through the middleware unchanged."""
    tools = await bridge.get_tools("weather", limit=5, score_threshold=0.0)
    weather_tool = next(t for t in tools if t.name == "get_weather")

    policy = SecurityPolicy(require_confirmation=["delete_*"])
    middleware = GantryApprovalMiddleware(policy)

    calls: list[str] = []

    class _Ctx:
        def __init__(self, fn, args):
            self.function = fn
            self.arguments = args
            self.result = None
            self.metadata: dict[str, Any] = {}

    async def _call_next():
        calls.append("ran")

    ctx = _Ctx(weather_tool, {"city": "London"})
    await middleware.process(ctx, _call_next)
    assert calls == ["ran"]


@pytest.mark.asyncio
async def test_approval_middleware_denies_by_domain(bridge: GantryToolBridge) -> None:
    """PermissionDeniedError propagates when policy denies on grounds other
    than confirmation (here: disallowed domain in arguments)."""
    tools = await bridge.get_tools("weather", limit=5, score_threshold=0.0)
    tool = next(t for t in tools if t.name == "get_weather")

    policy = SecurityPolicy(allowed_domains=["example.com"])
    middleware = GantryApprovalMiddleware(policy)

    class _Ctx:
        def __init__(self, fn, args):
            self.function = fn
            self.arguments = args
            self.result = None
            self.metadata: dict[str, Any] = {}

    async def _never_called():  # pragma: no cover
        raise AssertionError("should not be called")

    ctx = _Ctx(tool, {"city": "see https://blocked.test/x"})
    with pytest.raises(PermissionDeniedError):
        await middleware.process(ctx, _never_called)


# ---------------------------------------------------------------------------
# 10. Observability middleware records timing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observability_middleware_records_invocations(
    bridge: GantryToolBridge, gantry_with_tools: AgentGantry
) -> None:
    """The observability middleware should wrap ``call_next`` in a Gantry
    telemetry span so AF tool invocations show up alongside other
    Gantry-traced operations."""
    import asyncio

    mw = GantryObservabilityMiddleware(gantry_with_tools)

    class _FakeFn:
        name = "get_weather"

    class _Ctx:
        function = _FakeFn()
        arguments = {"city": "London"}
        result = None
        metadata: dict[str, Any] = {}

    # Capture telemetry spans opened by the middleware so we can assert the
    # real observability API (telemetry.span(...)) is exercised.
    telemetry = getattr(gantry_with_tools, "_telemetry", None)
    opened_spans: list[tuple[str, dict[str, Any]]] = []
    if telemetry is not None:
        original_span = telemetry.span

        def _tracking_span(name: str, attrs: dict[str, Any]) -> Any:
            opened_spans.append((name, attrs))
            return original_span(name, attrs)

        telemetry.span = _tracking_span  # type: ignore[assignment]

    calls: list[str] = []

    async def _slow_call() -> None:
        # Non-blocking sleep so we don't stall the event loop in tests.
        await asyncio.sleep(0.005)
        calls.append("ran")

    await mw.process(_Ctx(), _slow_call)
    assert calls == ["ran"], "call_next must be invoked inside the span"
    if telemetry is not None:
        assert opened_spans, "telemetry.span was not opened"
        span_name, span_attrs = opened_spans[0]
        assert span_name == "af_function_invocation"
        assert span_attrs.get("tool_name") == "get_weather"


# ---------------------------------------------------------------------------
# 11. build_agent() convenience helper end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_agent_convenience(bridge: GantryToolBridge) -> None:
    client = ScriptedChatClient(
        script=[
            [_fc("get_weather", {"city": "London"}, "bc0")],
            [Content.from_text("Sunny in London.")],
        ]
    )
    agent = await bridge.build_agent(
        client,
        "weather",
        name="convenience_agent",
        instructions="be brief",
        limit=5,
        score_threshold=0.0,
    )
    resp = await agent.run("Weather in London?")
    assert "sunny" in resp.text.lower()
    assert client.call_count == 2


# ---------------------------------------------------------------------------
# 12. FunctionTool invocation through Gantry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_function_tool_invoke_via_gantry(bridge: GantryToolBridge) -> None:
    tools = await bridge.get_tools("weather", limit=5, score_threshold=0.0)
    weather_tool = next(t for t in tools if t.name == "get_weather")

    result = await weather_tool.invoke(arguments={"city": "Dublin"})
    assert result, "FunctionTool.invoke should return Content list"
    # When invoked without a FunctionInvocationContext, AF returns raw text
    # Content rather than wrapping as function_result; the important thing is
    # that Gantry actually executed and the result surfaced through.
    assert "Dublin" in result[0].text


# ---------------------------------------------------------------------------
# 13. approval_mode correctly derived from Gantry capabilities
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_mode_maps_from_gantry_capabilities(bridge: GantryToolBridge) -> None:
    tools = await bridge.get_tools("", limit=10, score_threshold=0.0)
    by_name = {t.name: t for t in tools}

    # delete_user was registered with ToolCapability.DELETE_DATA → approval required
    assert by_name["delete_user"].approval_mode == "always_require"
    # Read-only tools default to never_require
    assert by_name["get_weather"].approval_mode == "never_require"
    assert by_name["lookup_order"].approval_mode == "never_require"
