"""Pydantic AI adapter tests, run against the real installed ``pydantic_ai``.

Every test drives a real ``Agent`` with ``FunctionModel`` (a scripted model, no
network or API key), so they exercise Pydantic AI's own tool manager, retry
handling, deferred-tool approval flow, ``ToolSearch`` capability and toolset
lifecycle (``for_run``) rather than stand-ins. Covers:

- static ``Tool`` conversion (schema verbatim, ``RunContext`` passthrough);
- outcome mapping: ``ModelRetry`` for argument errors, ``ToolFailed`` for
  terminal failures, ``ApprovalRequired`` for Gantry's confirmation gate (flag
  or ``SecurityPolicy``), and the opt-out ``error_mode="raise"``;
- ``deps``/``run_id``/``tool_call_id`` visible to Gantry handlers, and
  OpenTelemetry trace propagation;
- the per-turn toolset: mid-run re-selection from tool results, per-run
  caching, stickiness, per-run isolation, ids, wrapper compatibility;
- the deferred catalogue + Gantry-backed ``ToolSearch`` strategy;
- the bundled ``GantryCapability`` (per-turn, search, skills).

The one test that must work without Pydantic AI (the helpful ImportError) lives
at the bottom and does not need the package.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass

import pytest

from agent_gantry import AgentGantry, current_tool_invocation
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.core.security import SecurityPolicy
from agent_gantry.integrations.frameworks.base import (
    GantryToolset,
    ToolExecutionError,
    ToolPermissionDeniedError,
)


def _pydantic_ai():
    return pytest.importorskip("pydantic_ai")


@dataclass
class Deps:
    user: str


async def _build_gantry(**kwargs) -> AgentGantry:
    g = AgentGantry(embedder=SimpleEmbedder(dimension=256), **kwargs)

    @g.register(tags=["weather"])
    def get_weather(city: str) -> str:
        """Get the current weather forecast for a city."""
        return f"sunny in {city}"

    @g.register(tags=["math"])
    def divide(a: int, b: int) -> float:
        """Divide integer a by integer b."""
        return a / b

    @g.register(tags=["email"], requires_confirmation=True)
    def send_email(to: str, body: str = "") -> str:
        """Send an email message to a recipient."""
        g.sent.append(to)  # type: ignore[attr-defined]
        return f"sent:{to}"

    @g.register(tags=["crm"])
    def lookup_customer(name: str) -> str:
        """Look up a customer record by name."""
        return "Customer found. Next step: send an email message to jo@example.com"

    @g.register(tags=["identity"])
    def whoami() -> str:
        """Return the name of the current user from the run context."""
        from agent_gantry.pydantic_ai import current_run_context

        ctx = current_run_context()
        invocation = current_tool_invocation()
        return (
            f"user={ctx.deps.user};tool_call_id={invocation.tool_call_id};"
            f"run_id={invocation.run_id is not None}"
        )

    @g.register(tags=["records"])
    def delete_record(record_id: str) -> str:
        """Delete a stored record by its identifier."""
        g.deleted.append(record_id)  # type: ignore[attr-defined]
        return f"deleted:{record_id}"

    g.sent = []  # type: ignore[attr-defined]
    g.deleted = []  # type: ignore[attr-defined]
    await g.sync()
    return g


@pytest.fixture
async def gantry():
    g = await _build_gantry()
    yield g
    await g.close()


def _script(*steps):
    """A ``FunctionModel`` that plays ``steps`` in order, then echoes the last part.

    Each step is a list of ``(tool_name, args)`` calls. After the steps run out
    the model answers with the content of the newest request's first part, so a
    test can assert on what the model was shown.
    """
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    calls = list(steps)
    seen: list[list[str]] = []

    def fn(messages, info):
        seen.append(sorted(t.name for t in info.function_tools))
        if calls:
            step = calls.pop(0)
            return ModelResponse(parts=[ToolCallPart(name, args) for name, args in step])
        part = messages[-1].parts[0]
        return ModelResponse(parts=[TextPart(str(getattr(part, "content", part)))])

    model = FunctionModel(fn)
    model.seen = seen  # type: ignore[attr-defined]
    return model


def _adapter(gantry, **kwargs):
    from agent_gantry.pydantic_ai import PydanticAIAdapter

    return PydanticAIAdapter(gantry, **kwargs)


# --------------------------------------------------------------------------- #
# Static tools
# --------------------------------------------------------------------------- #


async def test_convert_uses_gantry_schema_verbatim(gantry):
    _pydantic_ai()
    from pydantic_ai import Tool

    spec = (await GantryToolset(gantry).select("send an email message", limit=1))[0]
    tool = _adapter(gantry).convert(spec)

    assert isinstance(tool, Tool)
    assert tool.name == "send_email"
    assert tool.description == "Send an email message to a recipient."
    assert tool.takes_ctx is True
    schema = tool.function_schema.json_schema
    assert schema["properties"]["to"]["type"] == "string"
    assert schema["required"] == ["to"]
    # A copy: mutating the emitted schema never touches the registry's.
    schema["properties"]["to"]["type"] = "integer"
    assert spec.parameters["properties"]["to"]["type"] == "string"


async def test_static_tools_run_in_a_real_agent(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent

    tools = await _adapter(gantry).select("weather forecast for a city", limit=2)
    result = await Agent(_script([("get_weather", {"city": "Leeds"})]), tools=tools).run("weather")
    assert result.output == "sunny in Leeds"


async def test_tool_failure_is_reported_to_the_model_not_raised(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent

    tools = await _adapter(gantry).select("divide integers", limit=6)
    result = await Agent(_script([("divide", {"a": 1, "b": 0})]), tools=tools).run("1/0")
    # ToolFailed's text, or (before ToolFailed existed) the tool's error result.
    assert result.output.endswith("ZeroDivisionError: division by zero")


async def test_argument_errors_become_model_retry(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent

    model = _script([("divide", {"a": "x", "b": 2})], [("divide", {"a": 4, "b": 2})])
    tools = await _adapter(gantry).select("divide integers", limit=6)
    result = await Agent(model, tools=tools).run("divide")

    # A retry is a ``RetryPromptPart`` up to pydantic-ai 2.x mid-line and a
    # ``ToolReturnPart(outcome="retried")`` after; both feed the error back.
    retries = [
        part
        for message in result.all_messages()
        for part in getattr(message, "parts", [])
        if getattr(part, "part_kind", None) == "retry-prompt"
        or getattr(part, "outcome", None) == "retried"
    ]
    assert retries and "must be an integer" in str(retries[0].content)
    assert result.output == "2.0"


async def test_error_mode_raise_restores_the_cross_framework_contract(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent

    tools = await _adapter(gantry, error_mode="raise").select("divide integers", limit=6)
    with pytest.raises(ToolExecutionError, match="division by zero"):
        await Agent(_script([("divide", {"a": 1, "b": 0})]), tools=tools).run("1/0")

    toolset = _adapter(gantry).toolset(limit=6, error_mode="raise")
    with pytest.raises(ToolExecutionError, match="division by zero"):
        await Agent(_script([("divide", {"a": 1, "b": 0})]), toolsets=[toolset]).run("divide")


async def test_invalid_error_mode_is_rejected_early(gantry):
    with pytest.raises(ValueError, match="error_mode"):
        _adapter(gantry, error_mode="swallow")
    with pytest.raises(ValueError, match="error_mode"):
        _adapter(gantry).toolset(error_mode="swallow")


async def test_permission_denied_maps_to_tool_failed(gantry):
    pai = _pydantic_ai()
    from agent_gantry.integrations.frameworks.pydantic_ai import to_pydantic_ai_exception

    spec = (await GantryToolset(gantry).select("weather", limit=1))[0]
    mapped = to_pydantic_ai_exception(
        spec, ToolPermissionDeniedError("get_weather", "permission_denied", "blocked")
    )
    if hasattr(pai.exceptions, "ToolFailed"):
        assert isinstance(mapped, pai.exceptions.ToolFailed)
        assert "Permission denied: blocked" in str(mapped)
    else:
        assert mapped is None  # returned to the model as the tool's result


# --------------------------------------------------------------------------- #
# Human-in-the-loop approval
# --------------------------------------------------------------------------- #


async def _approval_round_trip(agent, approve: bool):
    from pydantic_ai import DeferredToolRequests, DeferredToolResults

    first = await agent.run("send an email to bob")
    assert isinstance(first.output, DeferredToolRequests)
    assert [call.tool_name for call in first.output.approvals] == ["send_email"]
    call_id = first.output.approvals[0].tool_call_id
    qualified = first.output.metadata[call_id]["agent_gantry"]["qualified_name"]
    assert qualified.startswith("default.send_email")
    results = DeferredToolResults(approvals={call_id: approve})
    return await agent.run(message_history=first.all_messages(), deferred_tool_results=results)


@pytest.mark.parametrize("approve", [True, False])
@pytest.mark.parametrize("path", ["static", "toolset"])
async def test_confirmation_gate_uses_native_approval(gantry, path, approve):
    _pydantic_ai()
    from pydantic_ai import Agent, DeferredToolRequests

    adapter = _adapter(gantry)
    kwargs = (
        {"tools": await adapter.select("send an email message", limit=6)}
        if path == "static"
        else {"toolsets": [adapter.toolset(limit=6)]}
    )
    agent = Agent(
        _script([("send_email", {"to": "bob@x"})]),
        output_type=[str, DeferredToolRequests],
        **kwargs,
    )
    second = await _approval_round_trip(agent, approve)

    if approve:
        assert second.output == "sent:bob@x"
        assert gantry.sent == ["bob@x"]  # executed exactly once, after approval
    else:
        assert gantry.sent == []
        assert "denied" in second.output.lower()


async def test_security_policy_confirmation_also_pauses_for_approval():
    _pydantic_ai()
    from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults

    g = await _build_gantry(security_policy=SecurityPolicy(require_confirmation=["delete_*"]))
    try:
        agent = Agent(
            _script([("delete_record", {"record_id": "r1"})]),
            output_type=[str, DeferredToolRequests],
            tools=await _adapter(g).select("delete a stored record", limit=6),
        )
        first = await agent.run("delete r1")
        assert isinstance(first.output, DeferredToolRequests)
        assert g.deleted == []
        call_id = first.output.approvals[0].tool_call_id
        second = await agent.run(
            message_history=first.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={call_id: True}),
        )
        assert second.output == "deleted:r1"
        assert g.deleted == ["r1"]
    finally:
        await g.close()


# --------------------------------------------------------------------------- #
# Run context and tracing
# --------------------------------------------------------------------------- #


async def test_run_context_reaches_gantry_handlers(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent

    toolset = _adapter(gantry).toolset(always_include=["whoami"])
    agent = Agent(_script([("whoami", {})]), deps_type=Deps, toolsets=[toolset])
    result = await agent.run("who am I", deps=Deps("dan"))

    assert result.output.startswith("user=dan;tool_call_id=")
    assert "tool_call_id=None" not in result.output
    assert result.output.endswith("run_id=True")
    assert current_tool_invocation() is None  # scoped to the call


async def test_active_otel_span_becomes_the_calls_trace_parent(gantry, monkeypatch):
    _pydantic_ai()
    # Other test modules set this at import to keep framework SDKs quiet; the
    # SDK reads it when a TracerProvider is built.
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    trace = pytest.importorskip("opentelemetry.trace")
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from agent_gantry.integrations.frameworks.pydantic_ai import execute_for_pydantic_ai

    events = []
    gantry.on_tool_call(events.append)
    spec = (await GantryToolset(gantry).select("weather", limit=1))[0]
    tracer = sdk.TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("tool") as span:
        await execute_for_pydantic_ai(spec, {"city": "York"}, None)
    ids = span.get_span_context()
    assert events[-1].call.trace_id == format(ids.trace_id, "032x")
    assert events[-1].call.parent_span_id == format(ids.span_id, "016x")
    assert trace is not None


# --------------------------------------------------------------------------- #
# Per-turn toolset
# --------------------------------------------------------------------------- #


def _count_retrievals(gantry) -> list[str]:
    queries: list[str] = []
    original = gantry.retrieve

    async def spy(query):
        queries.append(query.context.query)
        return await original(query)

    gantry.retrieve = spy
    return queries


async def test_toolset_reselects_mid_run_from_tool_results(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent

    queries = _count_retrievals(gantry)
    model = _script([("lookup_customer", {"name": "Jo"})])
    toolset = _adapter(gantry).toolset(limit=1)
    await Agent(model, toolsets=[toolset]).run("look up the customer record for Jo")

    assert model.seen[0] == ["lookup_customer"]
    # The lookup's result ("...send an email message...") drives step 2.
    assert "send_email" in model.seen[1]
    assert "lookup_customer" in model.seen[1]  # sticky within the run
    # One retrieval per distinct query, not one per step.
    assert len(queries) == 2
    assert "send an email message" in queries[1]


async def test_toolset_without_sticky_replaces_the_selection(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent

    model = _script([("lookup_customer", {"name": "Jo"})])
    toolset = _adapter(gantry).toolset(limit=1, sticky=False)
    await Agent(model, toolsets=[toolset]).run("look up the customer record for Jo")
    assert model.seen[1] == ["send_email"]


async def test_toolset_state_is_isolated_per_run(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent

    toolset = _adapter(gantry).toolset(limit=1)
    weather = _script()
    email = _script()
    await asyncio.gather(
        Agent(weather, toolsets=[toolset]).run("weather forecast for a city"),
        Agent(email, toolsets=[toolset]).run("send an email message to a recipient"),
    )
    assert weather.seen == [["get_weather"]]
    assert email.seen == [["send_email"]]
    # The shared instance itself holds no run state.
    assert toolset._cache is None and toolset._offered is None


async def test_follow_up_turn_selects_from_the_new_prompt(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent

    toolset = _adapter(gantry).toolset(limit=1)
    first = _script()
    result = await Agent(first, toolsets=[toolset]).run("weather forecast for a city")
    second = _script()
    await Agent(second, toolsets=[toolset]).run(
        "now divide integer a by integer b", message_history=result.all_messages()
    )
    assert second.seen == [["divide"]]


async def test_toolset_ids_let_several_gantry_toolsets_coexist(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent

    adapter = _adapter(gantry)
    a = adapter.toolset(limit=1, id="primary")
    b = adapter.toolset(limit=1, id="secondary")
    assert (a.id, b.id) == ("primary", "secondary")
    assert adapter.toolset().id == "agent-gantry"

    model = _script([("p_get_weather", {"city": "Hull"})])
    result = await Agent(model, toolsets=[a.prefixed("p"), b.prefixed("s")]).run(
        "weather forecast for a city"
    )
    assert model.seen[0] == ["p_get_weather", "s_get_weather"]
    assert result.output == "sunny in Hull"


async def test_retrieval_failure_keeps_earlier_tools_and_warns(gantry, caplog):
    _pydantic_ai()
    from pydantic_ai import Agent

    original = gantry.retrieve
    calls = {"n": 0}

    async def flaky(query):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("vector store down")
        return await original(query)

    gantry.retrieve = flaky
    model = _script([("lookup_customer", {"name": "Jo"})])
    toolset = _adapter(gantry).toolset(limit=1)
    with caplog.at_level("WARNING"):
        await Agent(model, toolsets=[toolset]).run("look up the customer record for Jo")
    assert model.seen[1] == ["lookup_customer"]
    assert "semantic retrieval failed" in caplog.text


async def test_toolset_rejects_negative_max_retries(gantry):
    _pydantic_ai()
    with pytest.raises(ValueError, match="max_retries"):
        _adapter(gantry).toolset(max_retries=-1)


# --------------------------------------------------------------------------- #
# Tool search
# --------------------------------------------------------------------------- #


async def test_deferred_toolset_defers_everything_but_pins(gantry):
    _pydantic_ai()
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import RunContext
    from pydantic_ai.usage import RunUsage

    toolset = _adapter(gantry).deferred_toolset(always_include=["get_weather"])
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    tools = await toolset.get_tools(ctx)

    assert set(tools) == {
        "get_weather",
        "divide",
        "send_email",
        "lookup_customer",
        "whoami",
        "delete_record",
    }
    assert tools["get_weather"].tool_def.defer_loading is False
    assert all(t.tool_def.defer_loading for n, t in tools.items() if n != "get_weather")
    meta = tools["divide"].tool_def.metadata["agent_gantry"]
    assert meta["qualified_name"].startswith("default.divide")


async def test_search_strategy_ranks_gantry_tools_semantically(gantry):
    _pydantic_ai()
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import RunContext, ToolDefinition
    from pydantic_ai.usage import RunUsage

    adapter = _adapter(gantry)
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    corpus = [t.tool_def for t in (await adapter.deferred_toolset().get_tools(ctx)).values()]
    foreign = ToolDefinition(name="translate_text", description="Translate text into French.")
    queries = _count_retrievals(gantry)

    names = await adapter.search_strategy(limit=1)(
        ctx, ["weather forecast", "translate into french"], [*corpus, foreign]
    )

    assert names[0] == "get_weather"
    assert "translate_text" in names  # non-Gantry tools fall back to keywords
    assert len(queries) == 2  # one Gantry retrieval per search query


async def test_search_capability_end_to_end(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent

    model = _script(
        [("search_tools", {"queries": ["weather forecast for a city"]})],
        [("get_weather", {"city": "Leeds"})],
    )
    agent = Agent(model, capabilities=[_adapter(gantry).capability(mode="search", limit=1)])
    result = await agent.run("what's the weather in Leeds?")

    assert result.output == "sunny in Leeds"
    first_turn_visible = [
        name for name in model.seen[0] if name in ("search_tools", "get_weather", "divide")
    ]
    assert "search_tools" in first_turn_visible


# --------------------------------------------------------------------------- #
# Capability
# --------------------------------------------------------------------------- #


async def test_per_turn_capability_runs_tools(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import AbstractCapability

    from agent_gantry.pydantic_ai import GantryCapability

    capability = _adapter(gantry).capability(limit=1, id="gantry-tools")
    assert isinstance(capability, GantryCapability)
    assert isinstance(capability, AbstractCapability)
    model = _script([("get_weather", {"city": "Bath"})])
    result = await Agent(model, capabilities=[capability]).run("weather forecast for a city")
    assert result.output == "sunny in Bath"


async def test_capability_injects_relevant_skills(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from agent_gantry.schema.skill import Skill

    await gantry.add_skill(
        Skill(
            name="weather-etiquette",
            description="How to report a weather forecast to the user.",
            content="Always give the temperature in Celsius.",
        )
    )
    seen = {}

    def fn(messages, info):
        seen["instructions"] = messages[-1].instructions or ""
        return ModelResponse(parts=[TextPart("ok")])

    capability = _adapter(gantry).capability(skills=1)
    await Agent(FunctionModel(fn), capabilities=[capability]).run("report the weather forecast")
    assert "Always give the temperature in Celsius." in seen["instructions"]


async def test_capability_validates_arguments(gantry):
    _pydantic_ai()
    with pytest.raises(ValueError, match="mode"):
        _adapter(gantry).capability(mode="pull")
    with pytest.raises(ValueError, match="skills"):
        _adapter(gantry).capability(skills=-1)


async def test_live_returns_the_per_turn_toolset(gantry):
    _pydantic_ai()
    from agent_gantry.pydantic_ai import GantryToolset

    toolset = _adapter(gantry).live(limit=2, id="live")
    assert isinstance(toolset, GantryToolset)
    assert toolset.id == "live"


# --------------------------------------------------------------------------- #
# Without Pydantic AI
# --------------------------------------------------------------------------- #


async def test_missing_pydantic_ai_raises_helpful_error(monkeypatch, gantry):
    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    monkeypatch.setitem(sys.modules, "pydantic_ai.tools", None)

    specs = await GantryToolset(gantry).select("send an email", limit=1)
    with pytest.raises(ImportError, match=r"agent-gantry\[pydantic-ai\]"):
        _adapter(gantry).convert(specs[0])


# --------------------------------------------------------------------------- #
# Review follow-ups (PR #558)
# --------------------------------------------------------------------------- #


def _ctx():
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import RunContext
    from pydantic_ai.usage import RunUsage

    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


async def test_deferred_catalogue_loads_lazy_modules_before_snapshot():
    _pydantic_ai()
    g = AgentGantry(embedder=SimpleEmbedder(dimension=128), modules=["tests.test_modules.module_a"])
    try:
        adapter = _adapter(g)
        tools = await adapter.deferred_toolset().get_tools(_ctx())  # no sync() beforehand
        assert {"tool_a1", "tool_a2"} <= set(tools)

        corpus = [t.tool_def for t in tools.values()]
        names = await adapter.search_strategy(limit=2)(_ctx(), ["first tool"], corpus)
        assert "tool_a1" in names
    finally:
        await g.close()


async def test_search_budget_is_applied_after_the_eligibility_filter():
    _pydantic_ai()
    g = AgentGantry(embedder=SimpleEmbedder(dimension=256))

    @g.register
    def get_weather(city: str) -> str:
        """Get the current weather forecast for a city."""
        return city

    @g.register
    def get_weather_alerts(city: str) -> str:
        """Get severe weather alerts in the forecast for a city."""
        return city

    @g.register
    def divide(a: int, b: int) -> float:
        """Divide integer a by integer b."""
        return a / b

    await g.sync()
    try:
        adapter = _adapter(g)
        # get_weather is pinned visible, so it is not in the search corpus,
        # yet it is the registry's best match for the query.
        tools = await adapter.deferred_toolset(always_include=["get_weather"]).get_tools(_ctx())
        corpus = [t.tool_def for t in tools.values() if t.tool_def.defer_loading]
        assert "get_weather" not in {td.name for td in corpus}

        names = await adapter.search_strategy(limit=1)(
            _ctx(), ["current weather forecast for a city"], corpus
        )
        assert names == ["get_weather_alerts"]
    finally:
        await g.close()


async def test_query_from_results_false_keeps_tool_output_out_of_retrieval(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent

    queries = _count_retrievals(gantry)
    model = _script([("lookup_customer", {"name": "Jo"})])
    toolset = _adapter(gantry).toolset(limit=1, query_from_results=False)
    await Agent(model, toolsets=[toolset]).run("look up the customer record for Jo")

    assert all("jo@example.com" not in q for q in queries)
    assert model.seen[1] == ["lookup_customer"]


async def test_model_retry_exhaustion_ends_the_run(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    bad = [("divide", {"a": "x", "b": 2})]
    tools = await _adapter(gantry).select("divide integers", limit=6)
    with pytest.raises(UnexpectedModelBehavior, match="retries"):
        await Agent(_script(bad, bad, bad), tools=tools).run("divide")


async def test_per_run_selection_cache_is_bounded(gantry, monkeypatch):
    _pydantic_ai()
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    import agent_gantry.integrations.frameworks.pydantic_ai_live as live

    monkeypatch.setattr(live, "_MAX_CACHED_SELECTIONS", 2)
    run = await _adapter(gantry).toolset(limit=1).for_run(_ctx())
    for prompt in ("weather forecast", "divide integers", "send an email", "who am I"):
        ctx = _ctx()
        ctx.messages = [ModelRequest(parts=[UserPromptPart(content=prompt)])]
        await run.get_tools(ctx)
    assert len(run._cache) == 2


async def test_unavailable_tool_asks_the_model_to_choose_again(gantry):
    _pydantic_ai()
    from pydantic_ai.exceptions import ModelRetry

    toolset = _adapter(gantry).toolset(limit=1).set_query("weather forecast for a city")
    with pytest.raises(ModelRetry, match="not available"):
        await toolset.call_tool("divide", {"a": 1, "b": 2}, _ctx(), None)

    raising = _adapter(gantry).toolset(limit=1, error_mode="raise")
    raising.set_query("weather forecast for a city")
    with pytest.raises(ToolExecutionError, match="not available") as info:
        await raising.call_tool("divide", {"a": 1, "b": 2}, _ctx(), None)
    assert info.value.error_type == "ToolNotFound"


async def test_error_mode_raise_through_search_mode(gantry):
    _pydantic_ai()
    from pydantic_ai import Agent

    model = _script(
        [("search_tools", {"queries": ["divide integer a by integer b"]})],
        [("divide", {"a": 1, "b": 0})],
    )
    capability = _adapter(gantry).capability(mode="search", limit=1, error_mode="raise")
    with pytest.raises(ToolExecutionError, match="division by zero"):
        await Agent(model, capabilities=[capability]).run("divide 1 by 0")


def test_dict_tool_results_render_as_json():
    from agent_gantry.integrations.frameworks.pydantic_ai_live import _content_text

    assert _content_text({"to": "jo@example.com", "ok": True}) == (
        '{"to": "jo@example.com", "ok": true}'
    )


async def test_terminal_failure_without_tool_failed_is_returned_not_retried(gantry, monkeypatch):
    _pydantic_ai()
    import pydantic_ai.exceptions as pai_exceptions
    from pydantic_ai import Agent

    # Simulate a 2.x release from before ToolFailed existed.
    monkeypatch.delattr(pai_exceptions, "ToolFailed", raising=False)
    tools = await _adapter(gantry).select("divide integers", limit=6)
    # The model repeats the failing call; with ModelRetry that would exhaust
    # the retry budget and fail the run.
    bad = [("divide", {"a": 1, "b": 0})]
    result = await Agent(_script(bad, bad, bad), tools=tools).run("1/0")
    assert result.output == "Error: ZeroDivisionError: division by zero"


async def test_failure_text_shown_to_the_model_is_capped(gantry):
    from agent_gantry.integrations.frameworks.pydantic_ai import failure_text

    text = failure_text(ToolExecutionError("t", "failure", "x" * 5000, "ValueError"))
    assert len(text) < 2100
    assert text.startswith("ValueError: xxx")
    assert text.endswith("[truncated]")


async def test_search_keeps_another_registrys_tools_discoverable(gantry):
    _pydantic_ai()
    other = AgentGantry(embedder=SimpleEmbedder(dimension=256))

    @other.register
    def translate_text(text: str) -> str:
        """Translate text into French."""
        return text

    await other.sync()
    try:
        own = [
            t.tool_def
            for t in (await _adapter(gantry).deferred_toolset().get_tools(_ctx())).values()
        ]
        foreign = [
            t.tool_def
            for t in (await _adapter(other).deferred_toolset().get_tools(_ctx())).values()
        ]
        # This adapter's strategy cannot retrieve from `other`; its tools must
        # fall through to keyword matching rather than being claimed and lost.
        names = await _adapter(gantry).search_strategy(limit=1)(
            _ctx(), ["translate text into french", "weather forecast"], [*own, *foreign]
        )
        assert "translate_text" in names
        assert "get_weather" in names
    finally:
        await other.close()
