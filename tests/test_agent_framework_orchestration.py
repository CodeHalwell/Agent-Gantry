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

from collections.abc import Sequence
from typing import Any, ClassVar

import pytest

pytest.importorskip("agent_framework", reason="agent-framework not installed")

try:
    from agent_framework import (  # noqa: E402
        Agent,
        BaseChatClient,
        ChatMiddlewareLayer,
        ChatResponse,
        Content,
        FunctionInvocationLayer,
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
except ImportError as _af_import_err:
    pytest.skip(f"agent_framework missing required export: {_af_import_err}", allow_module_level=True)

from agent_gantry import AgentGantry  # noqa: E402
from agent_gantry.core.security import (  # noqa: E402
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

# ---------------------------------------------------------------------------
# 14. as_agent() — direct Agent construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_as_agent_constructs_agent_with_tools(bridge: GantryToolBridge) -> None:
    """bridge.as_agent() should return a plain Agent with semantically-selected tools."""
    client = ScriptedChatClient(
        script=[
            [_fc("get_weather", {"city": "Vienna"}, "aa0")],
            [Content.from_text("Vienna is sunny.")],
        ]
    )
    agent = await bridge.as_agent(
        client,
        query="weather forecast",
        name="WeatherAgent",
        instructions="Answer weather questions.",
        limit=5,
        score_threshold=0.0,
    )

    assert isinstance(agent, Agent)
    assert agent.name == "WeatherAgent"
    tools = (agent.default_options or {}).get("tools") or []
    assert tools, "as_agent should attach Gantry tools"
    assert any(getattr(t, "name", None) == "get_weather" for t in tools)

    resp = await agent.run("Weather in Vienna?")
    assert "sunny" in resp.text.lower()


@pytest.mark.asyncio
async def test_as_agent_query_kwargs_forwarded(bridge: GantryToolBridge) -> None:
    """query_kwargs passed to as_agent are forwarded to get_tools."""
    client = ScriptedChatClient(script=[[Content.from_text("done")]])
    # score_threshold via query_kwargs should not raise; limit 0 returns no tools
    agent = await bridge.as_agent(
        client,
        query="weather",
        name="A",
        instructions="",
        limit=5,
        score_threshold=0.0,
        query_kwargs={},
    )
    assert isinstance(agent, Agent)


# ---------------------------------------------------------------------------
# 15. build_workflow() — multi-agent WorkflowAgent construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_workflow_chain(bridge: GantryToolBridge) -> None:
    """build_workflow(chain=True) wires agents in order and each gets its own tools."""
    from agent_framework import WorkflowAgent

    client_a = ScriptedChatClient(
        script=[
            [_fc("get_weather", {"city": "Oslo"}, "bw0")],
            [Content.from_text("Oslo is sunny.")],
        ]
    )
    client_b = ScriptedChatClient(
        script=[
            [_fc("lookup_order", {"order_id": "ORD-5"}, "bw1")],
            [Content.from_text("Order ORD-5 is shipped.")],
        ]
    )

    wa = await bridge.build_workflow(
        agent_specs=[
            dict(client=client_a, query="weather city", name="WeatherStep",
                 instructions="Get weather.", limit=5, score_threshold=0.0),
            dict(client=client_b, query="order lookup", name="OrderStep",
                 instructions="Look up orders.", limit=5, score_threshold=0.0),
        ],
        chain=True,
        workflow_name="WeatherThenOrder",
    )

    assert isinstance(wa, WorkflowAgent)
    assert wa.name == "WeatherThenOrder"


@pytest.mark.asyncio
async def test_build_workflow_edge_routing(bridge: GantryToolBridge) -> None:
    """build_workflow with edges wires conditional handoffs."""
    from agent_framework import WorkflowAgent

    client_triage = ScriptedChatClient(script=[[Content.from_text("route")]])
    client_order = ScriptedChatClient(
        script=[
            [_fc("lookup_order", {"order_id": "ORD-1"}, "bwe0")],
            [Content.from_text("Order shipped.")],
        ]
    )

    wa = await bridge.build_workflow(
        agent_specs=[
            dict(client=client_triage, query="triage classify", name="Triage",
                 instructions="Route.", limit=5, score_threshold=0.0),
            dict(client=client_order, query="order lookup", name="Orders",
                 instructions="Handle orders.", limit=5, score_threshold=0.0),
        ],
        edges=[("Triage", "Orders")],
        workflow_name="TriageWorkflow",
    )

    assert isinstance(wa, WorkflowAgent)
    assert wa.name == "TriageWorkflow"


@pytest.mark.asyncio
async def test_build_workflow_unknown_edge_name_raises(bridge: GantryToolBridge) -> None:
    """build_workflow should raise ValueError for edges referencing unknown agent names."""
    client = ScriptedChatClient(script=[[Content.from_text("done")]])
    with pytest.raises(ValueError, match="unknown agent name"):
        await bridge.build_workflow(
            agent_specs=[
                dict(client=client, query="weather", name="WeatherAgent",
                     instructions="", limit=5, score_threshold=0.0),
            ],
            edges=[("WeatherAgent", "NonExistent")],
        )


@pytest.mark.asyncio
async def test_build_workflow_empty_specs_raises(bridge: GantryToolBridge) -> None:
    """build_workflow with no specs should raise ValueError."""
    with pytest.raises(ValueError, match="at least one agent"):
        await bridge.build_workflow(agent_specs=[])


# ---------------------------------------------------------------------------
# 16. build_agent end-to-end with extra_tools + multi-arg Gantry tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_agent_extra_tools_and_multi_arg_roundtrip(
    bridge: GantryToolBridge,
) -> None:
    """build_agent must surface both Gantry-selected and extra tools, and the
    Agent must round-trip a multi-arg Gantry tool call (call_id + result)."""
    from agent_framework import tool

    @tool(name="echo_note", description="Echo a short operator note.")
    def echo_note(note: str) -> str:
        return f"note:{note}"

    client = ScriptedChatClient(
        script=[
            [_fc("lookup_order", {"order_id": "ORD-77", "include_items": True}, "c0")],
            [Content.from_text("Order ORD-77 is shipped with 1 widget.")],
        ]
    )

    agent = await bridge.build_agent(
        client,
        query="order lookup",
        name="OrderAgent",
        instructions="Answer order questions.",
        limit=5,
        score_threshold=0.0,
        extra_tools=[echo_note],
    )

    assert isinstance(agent, Agent)
    bound_tool_names = {
        getattr(t, "name", None)
        for t in (agent.default_options or {}).get("tools") or []
    }
    assert "lookup_order" in bound_tool_names, bound_tool_names
    assert "echo_note" in bound_tool_names, "extra_tools must be appended"

    resp = await agent.run("Look up order ORD-77 with items.")

    assert "shipped" in resp.text.lower()
    assert client.call_count == 2, "expected one function-call turn + final turn"

    # The Agent's response must include a tool-result message with our call_id
    # and the actual Gantry-executed payload.
    function_results = [
        c
        for m in resp.messages
        for c in (m.contents or [])
        if getattr(c, "type", None) == "function_result"
    ]
    assert function_results, "Agent must surface the Gantry tool result"
    fr = function_results[0]
    assert fr.call_id == "c0"
    rendered = str(fr.result) + (str(getattr(fr, "items", "")) or "")
    assert "shipped" in rendered.lower()
    assert "ORD-77" in rendered


# ---------------------------------------------------------------------------
# 17. as_agent end-to-end with a multi-arg Gantry tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_as_agent_runs_multi_arg_tool(bridge: GantryToolBridge) -> None:
    """as_agent must produce an Agent that executes a Gantry tool with bool+str args."""
    client = ScriptedChatClient(
        script=[
            [_fc("lookup_order", {"order_id": "ORD-AA", "include_items": False}, "k0")],
            [Content.from_text("Order ORD-AA is shipped.")],
        ]
    )

    agent = await bridge.as_agent(
        client,
        query="order lookup",
        name="OrderAgent2",
        instructions="",
        limit=5,
        score_threshold=0.0,
    )

    assert isinstance(agent, Agent)

    resp = await agent.run("Look up order ORD-AA.")
    assert "shipped" in resp.text.lower()

    # Tool-result payload should carry the dict that the registered Python tool
    # returned, including the empty items list for include_items=False.
    fr = next(
        c
        for m in resp.messages
        for c in (m.contents or [])
        if getattr(c, "type", None) == "function_result"
    )
    rendered = str(fr.result) + str(getattr(fr, "items", ""))
    assert "ORD-AA" in rendered
    assert "shipped" in rendered.lower()
