"""
A selector that cannot answer fails open to semantic routing -- but never silently.

Failing open is the right default: an agent should not lose its tools because
a decision endpoint is unreachable. The fallback used to stop at the gantry
boundary, though, so a caller could not tell a result the selector produced
from one the semantic router produced in its place, and an eval that
configured a selector could pass on a different backend without knowing.

Two words are kept apart here, as in the library: a selector *declines* when it
answers with nothing (a result), and *falls back* when it does not answer.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from agent_gantry import AgentGantry, SelectorFallbackError
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.schema.config import AgentGantryConfig, SelectorConfig
from agent_gantry.schema.query import ConversationContext, RetrievalResult, ToolQuery
from agent_gantry.schema.selection import SelectionResult
from agent_gantry.schema.skill import Skill

GANTRY_LOG = "agent_gantry.core.gantry"


class _Selector:
    """A selector whose next answers are scripted: scores, or a fallback reason."""

    def __init__(self, *answers: dict[str, float] | str) -> None:
        self.answers = list(answers)
        self.calls = 0

    async def select(
        self, query: str, candidates: Any, limit: int, *, kind: str = "tool"
    ) -> SelectionResult:
        self.calls += 1
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, str):
            return SelectionResult(fallback=True, reason=answer)
        # Candidate names are namespaced ("default.add_numbers"); the script
        # names entries by their bare name.
        scores = {c.id: answer.get(c.name.rsplit(".", 1)[-1], 0.0) for c in candidates}
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return SelectionResult(
            selected=[cid for cid, score in ranked if score > 0][:limit], scores=scores
        )


def _gantry(selector: Any | None, **kwargs: Any) -> AgentGantry:
    gantry = AgentGantry(embedder=SimpleEmbedder(), selector=selector, **kwargs)

    @gantry.register(tags=["math"])
    async def add_numbers(a: int, b: int) -> int:
        """Add two numbers together for the caller."""
        return a + b

    @gantry.register(tags=["text"])
    async def shout(text: str) -> str:
        """Uppercase a string of text for the caller."""
        return text.upper()

    return gantry


def _query(text: str = "add two numbers") -> ToolQuery:
    return ToolQuery(context=ConversationContext(query=text), limit=2, score_threshold=0.0)


def _selector_logs(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
    return [
        (record.levelname, record.getMessage())
        for record in caplog.records
        if record.name == GANTRY_LOG and "elector" in record.getMessage()
    ]


# ---------------------------------------------------------------------------
# 1. Which backend answered is on the result
# ---------------------------------------------------------------------------


async def test_a_selector_answer_is_marked_as_the_selectors() -> None:
    gantry = _gantry(_Selector({"add_numbers": 0.9}))
    try:
        result = await gantry.retrieve(_query())
        assert [t.tool.name for t in result.tools] == ["add_numbers"]
        assert result.backend == "selector"
        assert result.selector_fallback is False
        assert result.selector_fallback_reason is None
    finally:
        await gantry.close()


async def test_a_fallback_is_marked_with_its_reason() -> None:
    """The case that produced a false pass: semantic tools, selector configured."""
    gantry = _gantry(_Selector("ConnectError: endpoint unreachable"))
    try:
        result = await gantry.retrieve(_query())
        assert result.tools, "failing open still returns the semantic router's tools"
        assert result.backend == "semantic"
        assert result.selector_fallback is True
        assert result.selector_fallback_reason == "ConnectError: endpoint unreachable"
        assert result.selection_time_ms is None
    finally:
        await gantry.close()


async def test_a_decline_is_a_selector_result_not_a_fallback() -> None:
    """Looking at the catalogue and choosing nothing is the selector answering."""
    gantry = _gantry(_Selector({}))
    try:
        result = await gantry.retrieve(_query("what is the capital of France?"))
        assert result.tools == []
        assert result.backend == "selector"
        assert result.selector_fallback is False
    finally:
        await gantry.close()


async def test_no_selector_is_semantic_without_a_fallback() -> None:
    gantry = _gantry(None)
    try:
        result = await gantry.retrieve(_query())
        assert result.backend == "semantic"
        assert result.selector_fallback is False
        assert result.selector_fallback_reason is None
    finally:
        await gantry.close()


async def test_a_selector_that_was_never_consulted_is_still_reported() -> None:
    """An empty query never costs a selection request; the result says the
    router answered, so the provenance holds for every call."""
    selector = _Selector({"add_numbers": 0.9})
    gantry = _gantry(selector)
    try:
        result = await gantry.retrieve(_query("   "))
        assert selector.calls == 0
        assert result.backend == "semantic"
        assert result.selector_fallback is True
        assert result.selector_fallback_reason == "the query is empty"
    finally:
        await gantry.close()


def test_the_new_fields_default_for_results_built_elsewhere() -> None:
    """Additive: a result constructed without them reads as plain semantic."""
    result = RetrievalResult(
        tools=[],
        query_embedding_time_ms=0.0,
        vector_search_time_ms=0.0,
        total_time_ms=0.0,
        candidate_count=0,
        filtered_count=0,
        trace_id="t",
    )
    assert (result.backend, result.selector_fallback, result.selector_fallback_reason) == (
        "semantic",
        False,
        None,
    )


# ---------------------------------------------------------------------------
# 2. A fallback is logged at WARNING, once per reason, and called a failure
# ---------------------------------------------------------------------------


async def test_a_fallback_warns_once_per_reason(caplog: pytest.LogCaptureFixture) -> None:
    """A catalogue over the selector's ceiling falls back on every call, for
    good. One WARNING makes that visible; one per call would bury the log."""
    reason = "catalogue of 600 exceeds max_candidates=512"
    gantry = _gantry(_Selector(reason))
    try:
        with caplog.at_level(logging.DEBUG, logger=GANTRY_LOG):
            for _ in range(3):
                await gantry.retrieve(_query())
        logs = _selector_logs(caplog)
        assert [level for level, _ in logs] == ["WARNING", "DEBUG", "DEBUG"]
        assert all(reason in message for _, message in logs)
    finally:
        await gantry.close()


async def test_a_fallback_is_called_a_failure_not_a_decline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gantry = _gantry(_Selector("TimeoutError: no answer"))
    try:
        with caplog.at_level(logging.DEBUG, logger=GANTRY_LOG):
            await gantry.retrieve(_query())
        (level, message), = _selector_logs(caplog)
        assert level == "WARNING"
        assert message.startswith("Selector failed for tool selection (TimeoutError: no answer)")
        assert "declined" not in message
    finally:
        await gantry.close()


async def test_a_new_reason_and_a_recurrence_after_recovery_warn_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    selector = _Selector("first fault", "first fault", "second fault", {"add_numbers": 0.9}, "second fault")
    gantry = _gantry(selector)
    try:
        with caplog.at_level(logging.DEBUG, logger=GANTRY_LOG):
            for _ in range(5):
                await gantry.retrieve(_query())
        levels = [level for level, _ in _selector_logs(caplog)]
        # first fault (new), first fault (repeat), second fault (new),
        # then an answer (nothing logged), then second fault again (news again).
        assert levels == ["WARNING", "DEBUG", "WARNING", "WARNING"]
    finally:
        await gantry.close()


async def test_nothing_to_select_on_is_not_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    gantry = _gantry(_Selector({"add_numbers": 0.9}))
    try:
        with caplog.at_level(logging.DEBUG, logger=GANTRY_LOG):
            await gantry.retrieve(_query(""))
        (level, message), = _selector_logs(caplog)
        assert level == "DEBUG"
        assert message.startswith("Selector not consulted for tool selection")
    finally:
        await gantry.close()


async def test_the_selectors_own_ceiling_warning_is_once_per_size(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from agent_gantry.adapters.selectors.system_one import SystemOneSelector
    from agent_gantry.schema.selection import SelectionCandidate

    class NeverAsked:
        async def score(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise AssertionError("over the ceiling: the model must not be asked")

    selector = SystemOneSelector(client=NeverAsked(), max_candidates=2)
    three = [SelectionCandidate(id=f"t{i}", name=f"t{i}") for i in range(3)]
    four = [*three, SelectionCandidate(id="t3", name="t3")]
    log = "agent_gantry.adapters.selectors.system_one"
    with caplog.at_level(logging.DEBUG, logger=log):
        for candidates in (three, three, four):
            result = await selector.select("q", candidates, limit=2)
            assert result.fallback and "exceeds max_candidates=2" in (result.reason or "")
    levels = [r.levelname for r in caplog.records if r.name == log and "ceiling" in r.getMessage()]
    assert levels == ["WARNING", "DEBUG", "WARNING"]


async def test_the_shipped_client_warns_once_per_outage_not_once_per_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The gantry's once-per-reason line is no use if the client underneath it
    warns on every request for as long as the provider stays down."""
    pytest.importorskip("typesafe_sdk")
    from agent_gantry.adapters.selectors.system_one import SystemOneSelector
    from agent_gantry.adapters.system_one_client import SystemOneClient

    class Sdk:
        """A stand-in TypeSafe client that is down, then up, then down again."""

        def __init__(self) -> None:
            self.down = True

        async def system_one(self, *, state: Any, questions: dict[str, Any]) -> Any:
            if self.down:
                raise ConnectionError("endpoint unreachable")
            answers = {key: type("Answer", (), {"noul": 0.9})() for key in questions}
            return type("Response", (), {"answers": answers, "usage": None})()

    sdk = Sdk()
    gantry = _gantry(SystemOneSelector(client=SystemOneClient(client=sdk)))
    try:
        with caplog.at_level(logging.DEBUG, logger="agent_gantry"):
            for _ in range(3):
                assert (await gantry.retrieve(_query())).selector_fallback is True
            sdk.down = False
            assert (await gantry.retrieve(_query())).backend == "selector"
            sdk.down = True
            assert (await gantry.retrieve(_query())).selector_fallback is True

        def warnings_from(logger_name: str, text: str) -> int:
            return sum(
                record.levelno == logging.WARNING
                for record in caplog.records
                if record.name == logger_name and text in record.getMessage()
            )

        # One when the outage starts and one when it recurs after an answer,
        # from each layer; five failed-or-not requests would have been four.
        assert warnings_from("agent_gantry.adapters.system_one_client", "selection failed") == 2
        assert warnings_from(GANTRY_LOG, "Selector failed") == 2
    finally:
        await gantry.close()


# ---------------------------------------------------------------------------
# 3. Strict mode: fail loudly instead of failing open
# ---------------------------------------------------------------------------


async def test_strict_mode_raises_instead_of_falling_back() -> None:
    gantry = _gantry(_Selector("ConnectError: unreachable"), on_selector_fallback="raise")
    try:
        with pytest.raises(SelectorFallbackError, match="ConnectError: unreachable") as raised:
            await gantry.retrieve(_query())
        assert raised.value.kind == "tool"
        assert raised.value.reason == "ConnectError: unreachable"
    finally:
        await gantry.close()


async def test_strict_mode_can_be_set_in_config() -> None:
    config = AgentGantryConfig(selector=SelectorConfig(on_fallback="raise"))
    gantry = AgentGantry(
        config=config, embedder=SimpleEmbedder(), selector=_Selector("boom")
    )
    try:

        @gantry.register
        def ping() -> str:
            """Reply with pong."""
            return "pong"

        with pytest.raises(SelectorFallbackError):
            await gantry.retrieve(_query("ping"))
    finally:
        await gantry.close()


async def test_strict_mode_leaves_answers_and_declines_alone() -> None:
    gantry = _gantry(_Selector({"add_numbers": 0.9}, {}), on_selector_fallback="raise")
    try:
        answered = await gantry.retrieve(_query())
        declined = await gantry.retrieve(_query("what is the capital of France?"))
        assert answered.backend == declined.backend == "selector"
        assert declined.tools == []
    finally:
        await gantry.close()


async def test_strict_mode_does_not_raise_when_there_is_nothing_to_select_on() -> None:
    """An empty query is not a selector failure; it is reported, not raised."""
    gantry = _gantry(_Selector({"add_numbers": 0.9}), on_selector_fallback="raise")
    try:
        result = await gantry.retrieve(_query(""))
        assert result.selector_fallback is True
    finally:
        await gantry.close()


async def test_strict_mode_covers_skills_and_mcp_servers() -> None:
    gantry = AgentGantry(
        embedder=SimpleEmbedder(), selector=_Selector("boom"), on_selector_fallback="raise"
    )
    try:
        await gantry.add_skill(
            Skill(
                name="deploying",
                description="How to deploy the service to production safely.",
                content="Run the deploy script.",
            )
        )
        with pytest.raises(SelectorFallbackError) as raised:
            await gantry.retrieve_skills("how do I ship this", limit=1)
        assert raised.value.kind == "skill"

        if gantry._mcp_registry is not None:
            gantry.register_mcp_server(
                "files", ["echo", "files"], description="Reads and writes local files."
            )
            with pytest.raises(SelectorFallbackError) as raised:
                await gantry.retrieve_mcp_servers("open a file", limit=1)
            assert raised.value.kind == "mcp_server"
    finally:
        await gantry.close()


async def test_skill_and_mcp_fallbacks_warn_too(caplog: pytest.LogCaptureFixture) -> None:
    gantry = AgentGantry(embedder=SimpleEmbedder(), selector=_Selector("boom"))
    try:
        await gantry.add_skill(
            Skill(
                name="deploying",
                description="How to deploy the service to production safely.",
                content="Run the deploy script.",
            )
        )
        with caplog.at_level(logging.DEBUG, logger=GANTRY_LOG):
            await gantry.retrieve_skills("how do I ship this", limit=1)
            await gantry.retrieve_skills("how do I ship this", limit=1)
        logs = _selector_logs(caplog)
        assert [level for level, _ in logs] == ["WARNING", "DEBUG"]
        assert logs[0][1].startswith("Selector failed for skill selection (boom)")
    finally:
        await gantry.close()


def test_an_unknown_strict_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="'semantic' or 'raise'"):
        AgentGantry(embedder=SimpleEmbedder(), on_selector_fallback="explode")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_the_selector_surface_is_importable_from_the_package_root() -> None:
    import agent_gantry
    from agent_gantry.adapters import selectors
    from agent_gantry.adapters.system_one_client import SystemOneClient

    assert agent_gantry.SystemOneClient is SystemOneClient
    assert agent_gantry.SelectorFallbackError is selectors.SelectorFallbackError
    assert {"SystemOneClient", "SystemOneSelector", "SystemOneReranker", "SelectorFallbackError"} <= set(
        agent_gantry.__all__
    )


# ---------------------------------------------------------------------------
# The decision record a provider keeps says which backend it came from
# ---------------------------------------------------------------------------


async def test_the_retrieval_decision_carries_the_backend() -> None:
    """``provider.selections`` / ``last_selection`` / ``dry_run_retrieve`` all
    hand back a ``RetrievalDecision``; an eval reads the backend off that."""
    pytest.importorskip("agent_framework")
    from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

    selector = _Selector(
        {"add_numbers": 0.9}, "ConnectError: unreachable", "ConnectError: unreachable"
    )
    gantry = _gantry(selector)
    try:
        bridge = GantryToolBridge(gantry)
        _, served = await bridge.get_tools_with_decision("add two numbers", limit=2)
        assert served.backend == "selector"
        assert served.selector_fallback is False
        assert served.selector_fallback_reason is None

        _, degraded = await bridge.get_tools_with_decision("add two numbers", limit=2)
        assert degraded.backend == "semantic"
        assert degraded.selector_fallback is True
        assert degraded.selector_fallback_reason == "ConnectError: unreachable"
        attributes = degraded.as_span_attributes()
        assert attributes["backend"] == "semantic"
        assert attributes["selector_fallback"] is True
        assert attributes["selector_fallback_reason"] == "ConnectError: unreachable"
    finally:
        await gantry.close()


async def test_a_provider_dry_run_reports_the_fallback() -> None:
    pytest.importorskip("agent_framework")
    from agent_gantry import GantryContextProvider

    gantry = _gantry(_Selector("catalogue of 600 exceeds max_candidates=512"))
    try:
        provider = GantryContextProvider(gantry, top_k=2)
        decision = await provider.dry_run_retrieve("add two numbers")
        assert decision.selector_fallback is True
        assert decision.selector_fallback_reason == "catalogue of 600 exceeds max_candidates=512"
    finally:
        await gantry.close()
