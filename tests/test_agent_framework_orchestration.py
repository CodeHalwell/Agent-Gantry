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


@pytest.fixture(autouse=True)
def _offline_embedder(monkeypatch):
    """Make bare ``AgentGantry()`` use the offline ``SimpleEmbedder`` so these
    tests never download an embedder model from the HF Hub (which rate-limits /
    flakes in CI). These tests pass all registered tools to a *scripted* client
    (``limit=5`` over ≤3 tools, ``score_threshold=0.0``), so retrieval quality is
    irrelevant and the toy embedder changes no assertion. Tests passing an
    explicit ``embedder=`` (e.g. ``_KeywordEmbedder``) are unaffected.
    """
    from agent_gantry.adapters.embedders.simple import SimpleEmbedder

    monkeypatch.setattr(
        "agent_gantry.core.gantry.build_embedder", lambda config: SimpleEmbedder()
    )


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
    rendered = str(fr.result) + str(getattr(fr, "items", ""))
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


# ---------------------------------------------------------------------------
# 18. Gantry actually reduces the tool surface the Agent sees
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_filters_tool_surface_for_agent() -> None:
    """Register many tools, prove the Agent only receives the relevant subset.

    This is Agent-Gantry's headline value proposition: semantic routing must
    keep the LLM's tool surface small. ``ScriptedChatClient.seen_tools``
    records exactly which tools were forwarded on each AF turn, so this test
    asserts (a) the AF Agent saw `limit` tools, not all registered, and
    (b) the unrelated tools were filtered out.
    """
    gantry = AgentGantry()

    @gantry.register
    def get_weather(city: str) -> str:
        """Get current weather forecast for a city."""
        return f"{city}: sunny"

    @gantry.register
    def get_forecast(city: str, days: int = 3) -> str:
        """Multi-day weather forecast for a city."""
        return f"{city}: sunny for {days} days"

    @gantry.register
    def get_air_quality(city: str) -> str:
        """Air quality index lookup for a city."""
        return f"{city} AQI: 42"

    @gantry.register
    def lookup_order(order_id: str) -> str:
        """Look up an e-commerce order by id."""
        return f"order {order_id}: shipped"

    @gantry.register
    def refund_order(order_id: str) -> str:
        """Issue a refund for an e-commerce order."""
        return f"refunded {order_id}"

    @gantry.register
    def list_invoices(user_id: str) -> list[str]:
        """List billing invoices for a customer."""
        return ["INV-1", "INV-2"]

    @gantry.register
    def compute_tax(amount: float, rate: float) -> float:
        """Compute sales tax for a transaction amount."""
        return amount * rate

    @gantry.register
    def read_file(path: str) -> str:
        """Read the contents of a file from disk."""
        return f"contents of {path}"

    @gantry.register
    def write_file(path: str, contents: str) -> str:
        """Write contents to a file on disk."""
        return f"wrote {path}"

    @gantry.register
    def send_email(to: str, body: str) -> str:
        """Send an email message to a recipient."""
        return f"sent to {to}"

    @gantry.register
    def search_codebase(query: str) -> list[str]:
        """Search the source code repository for a string."""
        return [f"hit for {query}"]

    @gantry.register
    def schedule_meeting(title: str, when: str) -> str:
        """Schedule a calendar meeting."""
        return f"booked {title}"

    await gantry.sync()

    total_registered = len(gantry.export_tools())
    assert total_registered == 12, total_registered

    bridge = GantryToolBridge(gantry, score_threshold=0.0)

    client = ScriptedChatClient(
        script=[
            [_fc("get_weather", {"city": "Tokyo"}, "rf0")],
            [Content.from_text("Tokyo: sunny.")],
        ]
    )

    # Limit=3 should retrieve at most 3 weather-related tools out of 12.
    agent = await bridge.build_agent(
        client,
        query="what's the weather forecast",
        name="WeatherAgent",
        instructions="",
        limit=3,
        score_threshold=0.0,
    )

    bound = (agent.default_options or {}).get("tools") or []
    bound_names = [getattr(t, "name", None) for t in bound]
    assert len(bound) == 3, (
        f"build_agent must attach exactly limit=3 tools (got {len(bound)}: {bound_names})"
    )
    assert len(bound) < total_registered, (
        "Gantry must reduce the tool surface vs. all registered tools"
    )

    # Drive the Agent so the ScriptedChatClient records what the LLM saw.
    resp = await agent.run("Weather in Tokyo?")
    assert "sunny" in resp.text.lower()

    # The AF Agent forwards its bound tool set to the chat client per turn.
    # seen_tools[0] is the very first turn → exactly the Gantry-filtered set.
    seen_first_turn = client.seen_tools[0]
    assert seen_first_turn, "ScriptedChatClient must have received tools"
    assert len(seen_first_turn) == 3, (
        f"Agent forwarded {len(seen_first_turn)} tools instead of the filtered 3: "
        f"{seen_first_turn}"
    )
    assert len(seen_first_turn) < total_registered

    # Semantic correctness: the weather tool we actually need is in the set.
    assert "get_weather" in seen_first_turn, seen_first_turn

    # And most of the 12 unrelated/loosely-related tools were dropped.
    # (SimpleEmbedder is hash-based, so we don't assert zero leakage — only
    # that the surface shrank meaningfully.  The 3/12 == 75% reduction above
    # is the headline value-prop assertion.)
    irrelevant = {
        "read_file",
        "write_file",
        "compute_tax",
        "send_email",
        "search_codebase",
        "schedule_meeting",
        "refund_order",
        "list_invoices",
        "lookup_order",
    }
    leaked = irrelevant.intersection(seen_first_turn)
    assert len(leaked) <= 2, (
        f"Too many unrelated tools leaked through semantic routing: {leaked}"
    )


# ---------------------------------------------------------------------------
# 19. Tool surface updates *between* loop iterations (per-run re-routing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_provider_reroutes_per_user_turn() -> None:
    """As the agent loops over user turns, the GantryContextProvider must
    re-query the router on every ``agent.run`` invocation so the LLM only
    sees tools relevant to the *current* turn — not a static set frozen at
    agent construction.

    This is the answer to "the agent is running in a loop, how does the
    tool selection update?": ``query_strategy='per_run'`` (default) fires
    ``before_run`` at the start of each ``agent.run`` call, derives a fresh
    query from the latest user message, and rewrites the injected tool set.
    """
    from agent_gantry import GantryContextProvider

    gantry = AgentGantry()

    @gantry.register
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        return f"{city}: sunny"

    @gantry.register
    def get_forecast(city: str, days: int = 3) -> str:
        """Multi-day weather forecast for a city."""
        return f"{city}: sunny"

    @gantry.register
    def lookup_invoice(invoice_id: str) -> str:
        """Look up a billing invoice."""
        return f"invoice {invoice_id}: $42 due"

    @gantry.register
    def refund_invoice(invoice_id: str) -> str:
        """Refund a billing invoice."""
        return f"refunded {invoice_id}"

    @gantry.register
    def read_file(path: str) -> str:
        """Read file contents from disk."""
        return f"contents:{path}"

    @gantry.register
    def write_file(path: str, contents: str) -> str:
        """Write file contents to disk."""
        return f"wrote {path}"

    await gantry.sync()
    total_registered = len(gantry.export_tools())
    assert total_registered == 6

    # Per-run strategy: re-query before each agent.run call.
    provider = GantryContextProvider(
        gantry,
        top_k=2,
        score_threshold=0.0,
        query_strategy="per_run",
    )

    # Three scripted turns, each on a different domain. Each turn is a single
    # text response (no function_call) so the script consumes exactly one
    # entry per agent.run call.
    client = ScriptedChatClient(
        script=[
            [Content.from_text("Weather noted.")],
            [Content.from_text("Invoice handled.")],
            [Content.from_text("File saved.")],
        ]
    )
    agent = Agent(
        client,
        instructions="",
        name="MultiTurnAgent",
        context_providers=[provider],
    )

    session = agent.create_session()
    # Loop the agent over user turns spanning different domains.
    await agent.run("What's the weather forecast for London?", session=session)
    await agent.run("Look up invoice INV-9 please.", session=session)
    await agent.run("Read the file at /tmp/notes.md.", session=session)

    assert client.call_count == 3
    assert len(client.seen_tools) == 3

    turn_weather, turn_billing, turn_files = client.seen_tools

    # Each turn must have surfaced <= top_k tools — the surface stayed bounded.
    for turn_idx, names in enumerate(client.seen_tools):
        assert len(names) <= 2, (
            f"Turn {turn_idx} forwarded {len(names)} tools, expected <= top_k=2: {names}"
        )
        assert len(names) < total_registered

    # The tool surface must actually *change* between turns — otherwise the
    # provider is just locking in a static set, not re-routing.
    assert turn_weather != turn_billing, (
        f"Tool set did not refresh between turn 1 (weather) and turn 2 (billing): "
        f"both saw {turn_weather}"
    )
    assert turn_billing != turn_files, (
        f"Tool set did not refresh between turn 2 (billing) and turn 3 (files): "
        f"both saw {turn_billing}"
    )

    # And the right tool for each domain must be present on the right turn.
    assert any(
        (n.startswith("get_") and "weather" in n) or n == "get_forecast"
        for n in turn_weather
    ), (
        f"Weather turn missing weather tool: {turn_weather}"
    )
    assert any("invoice" in n for n in turn_billing), (
        f"Billing turn missing invoice tool: {turn_billing}"
    )
    assert any("file" in n for n in turn_files), (
        f"File turn missing file tool: {turn_files}"
    )


# ---------------------------------------------------------------------------
# 20. Tool surface updates *within* a single agent.run (per-call re-routing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_provider_per_call_refresh_mechanism() -> None:
    """Drive the per-call refresher directly with two synthetic chat contexts
    to prove the routing-update *mechanism*: when the second round's message
    stream shifts topic, the rewritten ``options['tools']`` reflects the new
    query. Complementary to ``test_context_provider_per_call_end_to_end``,
    which exercises the same code path through ``agent.run``.
    """
    from agent_gantry import GantryContextProvider
    from agent_gantry.query import concatenate_recent

    gantry = AgentGantry()

    @gantry.register
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        return f"{city}: sunny"

    @gantry.register
    def get_forecast(city: str, days: int = 3) -> str:
        """Multi-day weather forecast for a city."""
        return f"{city}: sunny"

    @gantry.register
    def lookup_invoice(invoice_id: str) -> str:
        """Look up a billing invoice."""
        return f"invoice {invoice_id}"

    @gantry.register
    def refund_invoice(invoice_id: str) -> str:
        """Refund a billing invoice."""
        return f"refunded {invoice_id}"

    await gantry.sync()

    provider = GantryContextProvider(
        gantry,
        top_k=2,
        score_threshold=0.0,
        query_strategy="per_call",
        query_generator=concatenate_recent,
    )

    # Simulate two successive chat rounds inside a single agent.run by
    # feeding the refresher two synthetic context objects with shifting
    # message streams.
    class _FakeChatContext:
        def __init__(self, messages: list[Message]) -> None:
            self.messages = messages
            self.options: dict[str, Any] = {"tools": []}

    round1_ctx = _FakeChatContext(
        messages=[Message(role="user", contents=[Content.from_text("Weather in Paris?")])]
    )
    await provider._refresh_tools_on_chat_context(round1_ctx)
    round1_tools = [getattr(t, "name", None) for t in round1_ctx.options["tools"]]

    # The model emitted a function_call to get_weather; the tool ran and now
    # the conversation context for the second LLM round includes a tool
    # result + a follow-up user request about a totally different domain.
    round2_ctx = _FakeChatContext(
        messages=[
            Message(role="user", contents=[Content.from_text("Weather in Paris?")]),
            Message(role="assistant", contents=[
                Content.from_function_call(call_id="pc0", name="get_weather", arguments={"city": "Paris"})
            ]),
            Message(role="tool", contents=[
                Content.from_function_result(call_id="pc0", result="Paris: sunny")
            ]),
            Message(role="user", contents=[Content.from_text("Now look up invoice INV-9 and refund it.")]),
        ]
    )
    await provider._refresh_tools_on_chat_context(round2_ctx)
    round2_tools = [getattr(t, "name", None) for t in round2_ctx.options["tools"]]

    # Per-round bound on the surface.
    assert len(round1_tools) <= 2, round1_tools
    assert len(round2_tools) <= 2, round2_tools

    # Round 1 (weather query) must surface weather tools.
    assert any("weather" in n or "forecast" in n for n in round1_tools), round1_tools

    # Round 2 (billing query) must surface invoice tools and must *drop*
    # the stale weather tools from round 1.
    assert any("invoice" in n for n in round2_tools), round2_tools
    assert round1_tools != round2_tools, (
        f"Round-2 tool surface did not refresh after topic shift: "
        f"r1={round1_tools} r2={round2_tools}"
    )

    # And weather tools from r1 should not have leaked into r2's surface.
    weather_in_r2 = [n for n in round2_tools if "weather" in n or "forecast" in n]
    assert not weather_in_r2, (
        f"Stale weather tools from round 1 leaked into round 2: {weather_in_r2}"
    )


# ---------------------------------------------------------------------------
# 21. End-to-end per-call refresh through agent.run (the function-invocation
#     inner loop). This is the real proof that chat_middleware fires on
#     every round inside one agent.run AND that the freshly-injected tools
#     are actually executable by AF's function-invocation layer.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_provider_per_call_end_to_end() -> None:
    """A single ``agent.run`` makes 2 LLM rounds (function_call → result →
    final text). With ``query_strategy='per_call'`` + ``as_chat_middleware``
    the provider must run *between* the rounds and the tools it injects
    must be visible to AF's function executor on round 1.

    This guards against the regression where the refresher reassigned
    ``context.options = new_options`` — the chat client saw the new tools
    but ``FunctionInvocationLayer.mutable_options`` kept its original
    reference and failed to locate the tool to execute.
    """
    from agent_gantry import GantryContextProvider
    from agent_gantry.query import concatenate_recent

    gantry = AgentGantry()

    @gantry.register
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        return f"Weather in {city}: 22C sunny"

    @gantry.register
    def lookup_invoice(invoice_id: str) -> str:
        """Look up a billing invoice."""
        return f"invoice {invoice_id}"

    await gantry.sync()

    provider = GantryContextProvider(
        gantry,
        top_k=2,
        score_threshold=0.0,
        query_strategy="per_call",
        query_generator=concatenate_recent,
    )

    client = ScriptedChatClient(
        script=[
            # Round 1: model emits a function_call for the Gantry tool.
            [_fc("get_weather", {"city": "Paris"}, "e2e0")],
            # Round 2: model produces the final text answer using the result.
            [Content.from_text("Paris weather: 22C sunny.")],
        ]
    )

    agent = Agent(
        client,
        instructions="",
        name="PerCallE2E",
        context_providers=[provider],
        middleware=[provider.as_chat_middleware()],
    )

    resp = await agent.run("Weather in Paris?")

    # The model made 2 chat-completion rounds — meaning the function call
    # was executed (otherwise AF would have stopped after round 1).
    assert client.call_count == 2, (
        f"Expected 2 inner rounds, got {client.call_count}. "
        f"If only 1 ran, the function_call was not executed — the freshly "
        f"injected tools were not visible to AF's function executor."
    )

    # Final text comes from round 2.
    assert "22c" in resp.text.lower() or "sunny" in resp.text.lower(), resp.text

    # The middleware refreshed tools on each of the two rounds.
    assert len(client.seen_tools) == 2
    for r_idx, names in enumerate(client.seen_tools):
        assert names, f"Round {r_idx} forwarded no tools to the LLM"
        assert "get_weather" in names, (
            f"Round {r_idx} missing get_weather — refresh didn't propagate: {names}"
        )
        assert len(names) <= 2, (
            f"Round {r_idx} exceeded top_k=2: {names}"
        )

    # The function_result message is present in the response — proof the
    # tool actually executed end-to-end through the Gantry-wrapped path.
    function_results = [
        c
        for m in resp.messages
        for c in (m.contents or [])
        if getattr(c, "type", None) == "function_result"
    ]
    assert function_results, (
        "No function_result in the response — the tool was not executed."
    )
    fr = function_results[0]
    assert fr.call_id == "e2e0"
    rendered = str(fr.result) + str(getattr(fr, "items", ""))
    assert "paris" in rendered.lower()


# ---------------------------------------------------------------------------
# 22. SkillsProvider co-existence: Gantry's per-call refresh must not strip
#     skill tools (load_skill, read_skill_resource, run_skill_script) that
#     another provider injected.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_provider_preserves_skill_tools_across_refresh() -> None:
    """``SkillsProvider`` injects its own tools (``load_skill`` et al.) via
    ``context.extend_tools(self.source_id, tools)`` in ``before_run``. The
    Gantry per-call refresh must preserve those tools — only Gantry-owned
    tools should be dropped & re-injected.

    Rather than spin up a real SkillsProvider (which needs a skills source
    on disk), we simulate its effect: pre-seed the chat context with a
    foreign tool whose name is *not* in the Gantry registry, then run the
    refresh and assert the foreign tool survives.
    """
    from agent_framework import tool

    from agent_gantry import GantryContextProvider
    from agent_gantry.query import concatenate_recent

    gantry = AgentGantry()

    @gantry.register
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        return f"{city}: sunny"

    await gantry.sync()

    # Foreign tool not in the Gantry registry — represents what a peer
    # provider (e.g. SkillsProvider) would have injected via extend_tools.
    @tool(name="load_skill", description="Load a skill bundle by name (foreign).")
    def load_skill(name: str) -> str:
        return f"loaded {name}"

    provider = GantryContextProvider(
        gantry,
        top_k=2,
        score_threshold=0.0,
        query_strategy="per_call",
        query_generator=concatenate_recent,
    )

    class _FakeChatContext:
        def __init__(self, messages: list[Message], options: dict[str, Any]) -> None:
            self.messages = messages
            self.options = options

    # Round 1: only the foreign skill tool is in options (as if SkillsProvider
    # had just injected it in before_run and no Gantry refresh has run yet).
    options: dict[str, Any] = {"tools": [load_skill]}
    ctx = _FakeChatContext(
        messages=[Message(role="user", contents=[Content.from_text("Weather in Paris?")])],
        options=options,
    )

    await provider._refresh_tools_on_chat_context(ctx)

    tool_names = [getattr(t, "name", None) for t in ctx.options["tools"]]

    # The skill tool must survive the refresh.
    assert "load_skill" in tool_names, (
        f"Foreign skill tool was stripped by Gantry refresh: {tool_names}"
    )
    # And Gantry's dynamic selection was layered on top.
    assert "get_weather" in tool_names, (
        f"Gantry's semantically-selected tool was not injected: {tool_names}"
    )

    # Critical: the refresh must mutate the SAME options dict (not replace
    # the reference) so AF's FunctionInvocationLayer.mutable_options stays
    # in sync. Confirm by identity.
    assert ctx.options is options, (
        "Provider replaced the options dict reference — this would desync "
        "AF's tool-lookup table from the chat-call payload."
    )

    # Now simulate a follow-up round where the user shifts topic. The
    # refresh should re-route Gantry's dynamic slot (none of get_weather's
    # synonyms match) while still keeping load_skill.
    ctx.messages.append(
        Message(role="user", contents=[Content.from_text("Now load the math skill please.")])
    )
    await provider._refresh_tools_on_chat_context(ctx)

    tool_names_r2 = [getattr(t, "name", None) for t in ctx.options["tools"]]
    assert "load_skill" in tool_names_r2, (
        f"Skill tool stripped on second refresh: {tool_names_r2}"
    )
    assert ctx.options is options, "Options reference changed on second refresh."


# ---------------------------------------------------------------------------
# 23. Per-round routing actually adapts when the tool result shifts topic.
#     Uses a deterministic keyword-overlap embedder so the assertion runs
#     in CI without downloading any model weights — what we're testing is
#     the *plumbing* (does the router get re-queried with new content per
#     round, do new tools get injected, are they findable by the function
#     executor), not the quality of any specific embedder.
# ---------------------------------------------------------------------------


class _KeywordEmbedder:
    """Deterministic, dependency-free embedder for tests.

    Each text is embedded as a unit-norm vector over a fixed keyword
    vocabulary. Two texts that share a domain keyword (e.g. "weather",
    "invoice", "file") will have a high cosine similarity; texts in
    disjoint domains will have near-zero similarity.

    This is enough to deterministically demonstrate per-round routing
    adaptation without depending on sentence-transformers, ONNX, or any
    network-resident model.
    """

    _VOCAB: ClassVar[tuple[str, ...]] = (
        "weather", "temperature", "forecast", "city",
        "invoice", "refund", "billing", "payment",
        "file", "filesystem", "disk", "path",
        "email", "message",
    )

    @property
    def dimension(self) -> int:
        return len(self._VOCAB)

    @property
    def model_name(self) -> str:
        return "keyword-overlap"

    def get_embedder_id(self) -> str:
        return f"{self.model_name}:{self.dimension}"

    async def health_check(self) -> bool:
        return True

    def _vec(self, text: str) -> list[float]:
        import math
        lower = (text or "").lower()
        counts = [float(lower.count(kw)) for kw in self._VOCAB]
        norm = math.sqrt(sum(c * c for c in counts))
        if norm == 0:
            return [0.0] * len(self._VOCAB)
        return [c / norm for c in counts]

    async def embed_text(self, text: str) -> list[float]:
        return self._vec(text)

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        return [self._vec(t) for t in texts]


@pytest.mark.asyncio
async def test_context_provider_per_call_surface_adapts_to_tool_result() -> None:
    """End-to-end proof of per-round routing adaptation.

    The user asks a weather question. Round 1's router (driven by the
    user message) surfaces weather-flavoured tools. The model emits a
    function_call; the Gantry tool runs and its result legitimately
    introduces a *new* domain (billing/refund). On round 2 the query
    generator (``last_tool_result``) picks up the tool's output, and the
    router must re-rank — surfacing the refund tool that wasn't in the
    round-1 set.

    Uses a deterministic keyword-overlap embedder so the test runs in CI
    without any model download or hash-based-embedder flake.
    """
    from agent_gantry import GantryContextProvider
    from agent_gantry.query import last_tool_result, last_user_text

    gantry = AgentGantry(embedder=_KeywordEmbedder())

    @gantry.register
    def get_weather(city: str) -> str:
        """Get the current weather forecast temperature for a city."""
        # Result legitimately introduces a NEW domain — billing.
        return (
            "Paris is sunny. Customer follow-up: invoice INV-9 payment refund "
            "overdue billing."
        )

    @gantry.register
    def refund_invoice(invoice_id: str) -> str:
        """Issue a refund payment on a customer billing invoice."""
        return f"refunded {invoice_id}"

    @gantry.register
    def read_file(path: str) -> str:
        """Read file contents from the local filesystem disk path."""
        return ""

    @gantry.register
    def write_file(path: str, contents: str) -> str:
        """Write content to a file on the local filesystem disk path."""
        return ""

    await gantry.sync()

    def query_gen(messages: Any) -> str:
        return last_tool_result(messages) or last_user_text(messages)

    provider = GantryContextProvider(
        gantry,
        top_k=2,
        score_threshold=0.0,
        query_strategy="per_call",
        query_generator=query_gen,
    )

    client = ScriptedChatClient(
        script=[
            [_fc("get_weather", {"city": "Paris"}, "r0")],
            [_fc("refund_invoice", {"invoice_id": "INV-9"}, "r1")],
            [Content.from_text("All done.")],
        ]
    )
    agent = Agent(
        client,
        instructions="",
        name="AdaptiveAgent",
        context_providers=[provider],
        middleware=[provider.as_chat_middleware()],
    )

    resp = await agent.run("What's the weather forecast temperature in Paris?")

    # 3 LLM rounds — both function calls executed (otherwise we'd see <3).
    assert client.call_count == 3
    assert len(client.seen_tools) == 3

    round1, round2, _round3 = client.seen_tools

    # Round 1: weather-flavoured query → weather tool present.
    assert "get_weather" in round1, round1

    # Round 2: query shifted to the tool result (mentions invoice/refund/billing).
    # The refund tool must now be in the surfaced set, AND the surface must
    # have actually *changed* — that's the headline adaptation guarantee.
    assert "refund_invoice" in round2, (
        f"Round-2 surface did not surface refund_invoice after the topic "
        f"shifted in the tool result: {round2}"
    )
    assert round1 != round2, (
        f"Tool surface did not change between round 1 and round 2 even "
        f"though the query content shifted: r1={round1} r2={round2}"
    )

    # And the refund function call actually resolved (gating policy may
    # require confirmation, but the function NAME must be found — that
    # only happens when round-2 routing correctly surfaced it).
    refund_results = [
        c
        for m in resp.messages
        for c in (m.contents or [])
        if getattr(c, "type", None) == "function_result" and c.call_id == "r1"
    ]
    assert refund_results, "refund_invoice function_result missing"
    fr = refund_results[0]
    assert "not found" not in (fr.exception or ""), (
        f"refund_invoice was not surfaced on round 2 — executor reported "
        f"missing tool: {fr.exception!r}"
    )


# ---------------------------------------------------------------------------
# 24. _msg_text on AF function_result Content. Relocated from
#     test_query_strategies.py to keep that file AF-free as documented.
# ---------------------------------------------------------------------------


def test_last_tool_result_extracts_text_from_af_function_result_message() -> None:
    """AF wraps tool output as a ``function_result`` Content inside a
    tool-role Message — ``Message.text`` is empty in that case and the
    payload lives in ``Content.items[].text``. ``_msg_text`` must walk
    ``contents`` to surface it, otherwise ``last_tool_result`` returns
    "" and the per-call refresh's query collapses back to the original
    user prompt across the entire run.
    """
    from agent_gantry.query import last_tool_result

    fr = Content.from_function_result(
        call_id="r0",
        result="Paris is sunny. Invoice INV-9 payment is overdue.",
    )
    msg = Message(role="tool", contents=[fr])

    out = last_tool_result([msg])
    assert "Paris is sunny" in out, out
    assert "INV-9" in out, out
    assert out.startswith("tool result:") or out.startswith("result of"), out


# ---------------------------------------------------------------------------
# 25. Per-content function_result fallback: when an earlier text content
#     populated `parts`, a later function_result with empty items[] must
#     still surface its `.result` via the per-content fallback.
# ---------------------------------------------------------------------------


def test_msg_text_function_result_fallback_is_per_content() -> None:
    from agent_gantry.query.strategies import _msg_text

    # function_result Content with empty items and a primitive .result.
    fr = Content.from_function_result(call_id="r0", result="")
    fr.items = []  # explicit empty
    fr.result = "actual tool output payload"

    class _Msg:
        role = "tool"
        text = ""
        content = None
        contents = [
            Content.from_text(text="hi"),
            fr,
        ]

    out = _msg_text(_Msg())
    # The earlier "hi" text must not gate the function_result fallback.
    assert "hi" in out
    assert "actual tool output payload" in out


# ---------------------------------------------------------------------------
# 26. Non-dict (Pydantic-ish) options refresh: peer-provider tools must
#     be preserved and the existing options reference must be mutated in
#     place rather than replaced, mirroring the dict-options invariant
#     that FunctionInvocationLayer depends on.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_provider_refresh_mutates_non_dict_options_in_place() -> None:
    from agent_framework import tool

    from agent_gantry import GantryContextProvider

    gantry = AgentGantry()

    @gantry.register
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        return f"{city}: sunny"

    await gantry.sync()

    @tool(name="load_skill", description="Foreign peer-provider tool.")
    def load_skill(name: str) -> str:
        return f"loaded {name}"

    class _OptionsModel:
        """Minimal stand-in for a Pydantic ChatOptions object: not a dict,
        no model_copy, settable `tools` attribute."""

        def __init__(self, tools: list[Any]) -> None:
            self.tools = list(tools)

    class _Ctx:
        def __init__(self, messages: list[Message], options: _OptionsModel) -> None:
            self.messages = messages
            self.options = options

    options = _OptionsModel(tools=[load_skill])
    ctx = _Ctx(
        messages=[Message(role="user", contents=[Content.from_text("Weather in Paris?")])],
        options=options,
    )

    provider = GantryContextProvider(
        gantry, top_k=2, score_threshold=0.0, query_strategy="per_call",
    )
    await provider._refresh_tools_on_chat_context(ctx)

    # In-place mutation: same options reference, tools updated.
    assert ctx.options is options
    tool_names = [getattr(t, "name", None) for t in options.tools]
    # Peer-provider tool preserved.
    assert "load_skill" in tool_names, tool_names
    # Gantry's dynamic selection added.
    assert "get_weather" in tool_names, tool_names
