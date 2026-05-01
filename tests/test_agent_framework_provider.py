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


def _user_msg(text: str) -> "af.Message":
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
    async def test_retrieval_failure_is_swallowed(
        self, gantry_with_tools: AgentGantry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A retrieval exception must not break ``agent.run``."""
        provider = GantryContextProvider(
            gantry_with_tools, top_k=1, score_threshold=0.0
        )

        async def _boom(*_a: object, **_k: object) -> list:
            raise RuntimeError("retrieval down")

        # Patch the bridge's get_tools rather than gantry.retrieve so the
        # provider's own try/except path is exercised.
        monkeypatch.setattr(provider._bridge, "get_tools", _boom)

        ctx = af.SessionContext(input_messages=[_user_msg("weather")])
        await provider.before_run(agent=None, session=None, context=ctx, state={})

        # No exception, no tools injected — agent.run continues.
        assert ctx.tools == []
