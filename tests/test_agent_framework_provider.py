"""
Tests for :class:`agent_gantry.integrations.GantryContextProvider`.

The provider is the AF-native way to attach Gantry to an
``agent_framework.Agent`` and must:

* Inject the top-k semantically relevant tools into ``SessionContext`` for
  each ``agent.run`` call.
* Coexist with other ``ContextProvider`` instances (e.g.
  ``SkillsProvider``) without interfering — verified here by checking
  ``source_id`` attribution and that we only ever *append* to
  ``context.tools``.
* Honour the ``skills=True`` flag for always-on skill-bound tools.
* Honour explicit ``always_include`` pins.
* Fail safe on retrieval errors and on empty input.
"""

from __future__ import annotations

import pytest

# ``agent_framework`` is required to construct the provider; skip the entire
# module when the optional dependency is missing.
af = pytest.importorskip("agent_framework")

from agent_gantry import AgentGantry, GantryContextProvider  # noqa: E402
from agent_gantry.integrations.anthropic_skills import SkillRegistry  # noqa: E402


def _user_msg(text: str) -> af.Message:
    return af.Message(role="user", contents=[text])


def _tool_names(tools: list) -> list[str]:
    out: list[str] = []
    for t in tools:
        out.append(getattr(t, "name", None) or getattr(t, "__name__", "?"))
    return out


@pytest.fixture
async def gantry_with_tools() -> AgentGantry:
    g = AgentGantry()

    @g.register
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Weather: {city}"

    @g.register
    def book_flight(origin: str, destination: str) -> str:
        """Book a flight between two cities."""
        return f"flight {origin}->{destination}"

    @g.register
    def lookup_user(user_id: str) -> str:
        """Look up a user by ID."""
        return f"user {user_id}"

    @g.register
    def issue_refund(amount: float) -> str:
        """Issue a refund to a customer account."""
        return f"refund {amount}"

    await g.sync()
    return g


class TestGantryContextProvider:
    @pytest.mark.asyncio
    async def test_injects_top_k_tools(self, gantry_with_tools: AgentGantry) -> None:
        """before_run extracts the latest user message and injects top-k tools."""
        provider = GantryContextProvider(
            gantry_with_tools, top_k=2, score_threshold=0.0
        )
        ctx = af.SessionContext(input_messages=[_user_msg("weather in Paris")])

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        assert len(ctx.tools) <= 2
        assert "get_weather" in _tool_names(ctx.tools)

    @pytest.mark.asyncio
    async def test_default_source_id(self, gantry_with_tools: AgentGantry) -> None:
        """Provider defaults to a stable source_id for AF attribution."""
        provider = GantryContextProvider(gantry_with_tools)
        assert provider.source_id == "agent_gantry"

    @pytest.mark.asyncio
    async def test_custom_source_id(self, gantry_with_tools: AgentGantry) -> None:
        """source_id is overridable so multiple gantries can run side-by-side."""
        provider = GantryContextProvider(gantry_with_tools, source_id="gantry_a")
        assert provider.source_id == "gantry_a"

    @pytest.mark.asyncio
    async def test_empty_input_messages_does_not_crash(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """A run with no input messages must not raise — provider stays passive."""
        provider = GantryContextProvider(gantry_with_tools, score_threshold=0.0)
        ctx = af.SessionContext(input_messages=[])

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        assert ctx.tools == []

    @pytest.mark.asyncio
    async def test_picks_latest_user_message(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """When several messages are present, the latest user turn drives retrieval."""
        provider = GantryContextProvider(
            gantry_with_tools, top_k=1, score_threshold=0.0
        )
        ctx = af.SessionContext(
            input_messages=[
                _user_msg("totally unrelated topic"),
                _user_msg("book me a flight from LHR to NRT"),
            ]
        )

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        assert "book_flight" in _tool_names(ctx.tools)

    @pytest.mark.asyncio
    async def test_always_include_pins_tools(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """always_include adds pinned tools regardless of semantic match."""
        provider = GantryContextProvider(
            gantry_with_tools,
            top_k=1,
            score_threshold=0.0,
            always_include=["issue_refund"],
        )
        ctx = af.SessionContext(input_messages=[_user_msg("weather in Paris")])

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        names = _tool_names(ctx.tools)
        assert "issue_refund" in names

    @pytest.mark.asyncio
    async def test_skills_flag_injects_registered_skill_tools(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """skills=True always injects the union of skill-bound tool names."""
        skills = SkillRegistry()
        skills.register(
            name="customer_support",
            description="Customer support utilities",
            instructions="Use when handling customer issues.",
            tools=["lookup_user", "issue_refund"],
        )
        provider = GantryContextProvider(
            gantry_with_tools,
            top_k=1,
            score_threshold=0.0,
            skills=True,
            skill_registry=skills,
        )
        ctx = af.SessionContext(input_messages=[_user_msg("weather in Paris")])

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        names = _tool_names(ctx.tools)
        assert "lookup_user" in names
        assert "issue_refund" in names

    @pytest.mark.asyncio
    async def test_skills_flag_without_registry_is_safe(
        self, gantry_with_tools: AgentGantry, caplog: pytest.LogCaptureFixture
    ) -> None:
        """skills=True without a registry logs a debug note and keeps going."""
        provider = GantryContextProvider(
            gantry_with_tools, top_k=1, score_threshold=0.0, skills=True
        )
        ctx = af.SessionContext(input_messages=[_user_msg("weather")])

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        # Still injects the dynamic top-1 result; just no skill tools added.
        assert len(ctx.tools) >= 1

    @pytest.mark.asyncio
    async def test_unknown_pinned_tool_logs_warning(
        self, gantry_with_tools: AgentGantry, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown names in always_include warn but don't break the run."""
        import logging

        provider = GantryContextProvider(
            gantry_with_tools,
            top_k=1,
            score_threshold=0.0,
            always_include=["does_not_exist"],
        )
        ctx = af.SessionContext(input_messages=[_user_msg("weather")])

        with caplog.at_level(logging.WARNING):
            await provider.before_run(
                agent=None, session=None, context=ctx, state={}
            )
        assert "does_not_exist" in caplog.text

    @pytest.mark.asyncio
    async def test_does_not_overwrite_other_provider_tools(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """The provider must only *append* to context.tools, never replace.

        This is what guarantees coexistence with ``SkillsProvider`` and any
        other ``ContextProvider`` upstream in the pipeline.
        """
        provider = GantryContextProvider(
            gantry_with_tools, top_k=1, score_threshold=0.0
        )
        sentinel = object()
        ctx = af.SessionContext(
            input_messages=[_user_msg("weather")],
            tools=[sentinel],  # pretend another provider already added one
        )

        await provider.before_run(agent=None, session=None, context=ctx, state={})

        assert sentinel in ctx.tools
        assert len(ctx.tools) > 1

    @pytest.mark.asyncio
    async def test_per_call_refresh_preserves_foreign_skill_tools(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """The per-call refresh must keep tools from other providers.

        This is the load-bearing guarantee for coexisting with AF
        ``SkillsProvider``: every chat round the middleware rebuilds the tool
        list, dropping *gantry-known* stale tools and re-adding the fresh
        top-k — but anything foreign (skill tools, another provider's tools)
        must survive untouched so skills keep working.
        """
        provider = GantryContextProvider(
            gantry_with_tools, top_k=2, query_strategy="per_call"
        )

        class _SkillTool:
            # Foreign: a name that is NOT in the gantry registry, standing in
            # for a tool injected by AF SkillsProvider.
            name = "summarise_skill"

        class _StaleGantryTool:
            # Gantry-known: simulates a previous round's injected tool that
            # should be refreshed (not duplicated), never the skill.
            name = "get_weather"

        skill_tool = _SkillTool()

        class _ChatCtx:
            def __init__(self) -> None:
                self.options = {"tools": [skill_tool, _StaleGantryTool()]}
                self.messages = [_user_msg("weather in Paris")]

        ctx = _ChatCtx()
        await provider._refresh_tools_on_chat_context(ctx)

        names = [
            getattr(t, "name", None) or getattr(t, "__name__", "?")
            for t in ctx.options["tools"]
        ]
        # The skill tool survived the per-round rebuild — skills still work.
        assert "summarise_skill" in names
        # The stale gantry tool was not duplicated (dropped or refreshed to one).
        assert names.count("get_weather") <= 1
        # A fresh gantry selection was injected alongside the skill.
        gantry_tool_names = {"get_weather", "book_flight", "lookup_user", "issue_refund"}
        assert any(n in gantry_tool_names for n in names)

    @pytest.mark.asyncio
    async def test_retrieval_failure_is_swallowed(
        self, gantry_with_tools: AgentGantry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A retrieval exception must not break ``agent.run``."""
        provider = GantryContextProvider(
            gantry_with_tools, top_k=1, score_threshold=0.0
        )

        async def _boom(*_a: object, **_k: object) -> list:
            raise RuntimeError("retrieval down")

        # Patch the bridge's decision-returning retrieval (the provider's
        # actual call path) so the provider's own try/except is exercised.
        monkeypatch.setattr(
            provider._bridge, "get_tools_with_decision", _boom
        )

        ctx = af.SessionContext(input_messages=[_user_msg("weather")])
        await provider.before_run(agent=None, session=None, context=ctx, state={})

        # No exception, no tools injected — agent.run continues.
        assert ctx.tools == []

    # ------------------------------------------------------------------
    # New surface area: last_selection / dry_run_retrieve / static_tools
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_last_selection_exposes_decision_after_before_run(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """After each retrieval the provider exposes the structured decision."""
        provider = GantryContextProvider(
            gantry_with_tools, top_k=2, score_threshold=0.0
        )
        ctx = af.SessionContext(input_messages=[_user_msg("weather")])
        await provider.before_run(agent=None, session=None, context=ctx, state={})

        decision = provider.last_selection
        assert decision is not None
        assert decision.query == "weather"
        assert decision.injected
        # Candidates is the ranked list and is non-empty.
        assert decision.candidates
        # Every injected tool was kept.
        assert all(c.kept for c in decision.candidates if c.name in decision.injected)

    @pytest.mark.asyncio
    async def test_dry_run_retrieve_uses_same_code_path(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """dry_run_retrieve must mirror the live retrieval (same threshold,
        same kwargs) so users can validate "would the LLM see X?" offline."""
        provider = GantryContextProvider(
            gantry_with_tools, top_k=2, score_threshold=0.0
        )
        decision = await provider.dry_run_retrieve("book me a flight")
        names = {c.name for c in decision.candidates}
        # The fixture's book_flight should rank for a "flight" query.
        assert "book_flight" in names
        # Same path => same threshold mode reported.
        assert decision.threshold_mode == "absolute"

    @pytest.mark.asyncio
    async def test_static_tools_injected_on_every_run(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """Tools that live outside the gantry registry must still be
        injected when supplied via static_tools."""

        async def af_native_tool(x: str) -> str:
            """A tool not registered with gantry."""
            return x

        af_native_tool.__name__ = "af_native_tool"
        provider = GantryContextProvider(
            gantry_with_tools,
            top_k=1,
            score_threshold=0.0,
            static_tools=[af_native_tool],
        )
        ctx = af.SessionContext(input_messages=[_user_msg("weather")])
        await provider.before_run(agent=None, session=None, context=ctx, state={})
        names = _tool_names(ctx.tools)
        assert "af_native_tool" in names

    @pytest.mark.asyncio
    async def test_per_call_default_generator_is_fallback_chain(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """per_call mode must pick a generator that actually adapts each
        round; the per_run default (last_user_text) would silently
        disable the very thing per_call enables."""
        from agent_gantry.query import last_user_text

        provider = GantryContextProvider(
            gantry_with_tools, top_k=1, query_strategy="per_call"
        )
        # The selected generator is *not* the bare last_user_text default.
        assert provider._query_generator is not last_user_text  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_per_call_with_last_user_text_warns(
        self,
        gantry_with_tools: AgentGantry,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Explicitly pairing per_call with last_user_text — the
        misconfiguration the issue calls out — must warn."""
        import logging

        from agent_gantry.query import last_user_text

        with caplog.at_level(logging.WARNING):
            GantryContextProvider(
                gantry_with_tools,
                query_strategy="per_call",
                query_generator=last_user_text,
            )
        assert "per_call" in caplog.text and "last_user_text" in caplog.text

    @pytest.mark.asyncio
    async def test_per_call_warns_when_chat_middleware_not_attached(
        self,
        gantry_with_tools: AgentGantry,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """per_call mode without the chat middleware on the agent must
        warn so the user knows retrieval is silently per_run-only."""
        import logging

        class FakeAgent:
            middleware: list = []

        provider = GantryContextProvider(
            gantry_with_tools, top_k=1, query_strategy="per_call"
        )
        ctx = af.SessionContext(input_messages=[_user_msg("weather")])
        with caplog.at_level(logging.WARNING):
            await provider.before_run(
                agent=FakeAgent(), session=None, context=ctx, state={}
            )
        assert "as_chat_middleware" in caplog.text

    @pytest.mark.asyncio
    async def test_attach_to_appends_provider_and_middleware(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """attach_to is the one-call setup helper for the per_call flow."""

        class FakeAgent:
            context_providers: list = []
            middleware: list = []

        agent = FakeAgent()
        provider = GantryContextProvider(
            gantry_with_tools, query_strategy="per_call"
        )
        provider.attach_to(agent)
        assert provider in agent.context_providers
        assert len(agent.middleware) == 1

    def test_relative_threshold_accepted_at_construction(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """A 'relative:<frac>' string is accepted and not rejected as a float."""
        provider = GantryContextProvider(
            gantry_with_tools, score_threshold="relative:0.8"
        )
        assert provider.score_threshold == "relative:0.8"


class TestSelectionsAndTrace:
    """Per-round selection history and the built-in console trace middleware."""

    @pytest.mark.asyncio
    async def test_selections_accumulate_across_rounds(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """Every dynamic retrieval appends to the bounded selections history."""
        provider = GantryContextProvider(gantry_with_tools, top_k=2)

        assert provider.selections == ()
        for query in ("weather in Paris", "issue a refund"):
            ctx = af.SessionContext(input_messages=[_user_msg(query)])
            await provider.before_run(
                agent=None, session=None, context=ctx, state={}
            )

        assert len(provider.selections) == 2
        # last_selection is always the most recent entry in the history.
        assert provider.selections[-1] is provider.last_selection

    @pytest.mark.asyncio
    async def test_selections_is_immutable_snapshot(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """The property returns a tuple copy, not the internal deque."""
        provider = GantryContextProvider(gantry_with_tools, top_k=2)
        ctx = af.SessionContext(input_messages=[_user_msg("weather in Paris")])
        await provider.before_run(agent=None, session=None, context=ctx, state={})
        assert isinstance(provider.selections, tuple)

    @pytest.mark.asyncio
    async def test_trace_middleware_prints_call_and_result(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """trace() prints a before/after line and the router's surfaced set."""
        provider = GantryContextProvider(gantry_with_tools, top_k=2)
        # Seed a selection so the surfaced list is populated.
        ctx = af.SessionContext(input_messages=[_user_msg("weather in Paris")])
        await provider.before_run(agent=None, session=None, context=ctx, state={})

        lines: list[str] = []
        middleware = provider.trace(printer=lines.append)

        class _FakeFn:
            name = "get_weather"

        class _FakeCtx:
            function = _FakeFn()
            arguments = {"city": "Paris"}
            result = "Weather: Paris"

        calls = {"n": 0}

        async def _call_next() -> None:
            calls["n"] += 1

        await middleware(_FakeCtx(), _call_next)

        assert calls["n"] == 1
        joined = "\n".join(lines)
        assert "round 1" in joined
        assert "get_weather" in joined
        assert "Weather: Paris" in joined
        assert "surfaced:" in joined

    @pytest.mark.asyncio
    async def test_trace_render_false_skips_result_line(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        provider = GantryContextProvider(gantry_with_tools, top_k=2)
        lines: list[str] = []
        middleware = provider.trace(render=False, printer=lines.append)

        class _FakeFn:
            name = "get_weather"

        class _FakeCtx:
            function = _FakeFn()
            arguments = {"city": "Paris"}
            result = "Weather: Paris"

        async def _call_next() -> None:
            return None

        await middleware(_FakeCtx(), _call_next)
        # Only the pre-call ">>>" line, no post-call "<<<" preview.
        assert any(line.startswith(">>>") for line in lines)
        assert not any(line.startswith("<<<") for line in lines)

    @pytest.mark.asyncio
    async def test_attach_to_trace_adds_trace_middleware(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """attach_to(trace=True) wires the trace middleware alongside the rest."""

        class FakeAgent:
            context_providers: list = []
            middleware: list = []

        # per_run: no chat middleware, so trace is the only one attached.
        per_run = GantryContextProvider(gantry_with_tools)
        per_run.attach_to(FakeAgent(), trace=True)

        agent = FakeAgent()
        per_call = GantryContextProvider(
            gantry_with_tools, query_strategy="per_call"
        )
        per_call.attach_to(agent, trace=True)
        # per_call attaches both the chat refresher and the trace middleware.
        assert len(agent.middleware) == 2

    @pytest.mark.asyncio
    async def test_selections_isolated_across_concurrent_tasks(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """A provider shared across concurrent runs must not interleave history.

        ContextVar-backed state means each task (asyncio.gather wraps each
        coroutine in a Task, which copies the context) accumulates only its own
        selections rather than appending into one shared list.
        """
        import asyncio

        provider = GantryContextProvider(gantry_with_tools, top_k=2)

        async def run_once(query: str) -> tuple:
            ctx = af.SessionContext(input_messages=[_user_msg(query)])
            await provider.before_run(
                agent=None, session=None, context=ctx, state={}
            )
            # Read within the same task: should see exactly this run's history.
            return provider.selections

        a, b = await asyncio.gather(
            run_once("weather in Paris"),
            run_once("issue a refund"),
        )
        assert len(a) == 1
        assert len(b) == 1

    @pytest.mark.asyncio
    async def test_trace_surfaced_shows_only_kept_capped(
        self, gantry_with_tools: AgentGantry
    ) -> None:
        """The trace line lists kept candidates only, capped, not the full list."""
        from agent_gantry.integrations.agent_framework_bridge import (
            RetrievalCandidate,
            RetrievalDecision,
        )

        provider = GantryContextProvider(gantry_with_tools, top_k=2)
        # Hand-build a decision: 7 kept + 1 dropped.
        kept = [
            RetrievalCandidate(
                name=f"tool_{i}",
                qualified_name=f"default.tool_{i}",
                score=0.9 - i * 0.01,
                kept=True,
            )
            for i in range(7)
        ]
        dropped = RetrievalCandidate(
            name="dropped_tool",
            qualified_name="default.dropped_tool",
            score=0.01,
            kept=False,
        )
        decision = RetrievalDecision(query="q", candidates=[*kept, dropped])
        provider._last_selection_var.set(decision)

        lines: list[str] = []
        middleware = provider.trace(render=False, printer=lines.append)

        class _FakeFn:
            name = "tool_0"

        class _FakeCtx:
            function = _FakeFn()
            arguments: dict = {}
            result = "ok"

        async def _call_next() -> None:
            return None

        await middleware(_FakeCtx(), _call_next)
        line = "\n".join(lines)
        assert "dropped_tool" not in line          # dropped candidate excluded
        assert "+2 more" in line                    # 7 kept, capped at 5 -> +2
        assert line.count(":0.") <= 6               # 5 shown scores (+ guard)
