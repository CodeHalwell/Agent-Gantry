"""
A selector that cannot answer fails open to semantic routing -- but never silently.

Failing open is the right default: an agent should not lose its tools because
a decision endpoint is unreachable. The fallback used to stop at the fetcher
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

from dotfetch import DotFetch, SelectorFallbackError
from dotfetch.adapters.embedders.simple import SimpleEmbedder
from dotfetch.schema.config import DotFetchConfig, SelectorConfig
from dotfetch.schema.query import ConversationContext, RetrievalResult, ToolQuery
from dotfetch.schema.selection import SelectionResult
from dotfetch.schema.skill import Skill

DOTFETCH_LOG = "dotfetch.core.fetcher"


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


def _fetcher(selector: Any | None, **kwargs: Any) -> DotFetch:
    fetcher = DotFetch(embedder=SimpleEmbedder(), selector=selector, **kwargs)

    @fetcher.register(tags=["math"])
    async def add_numbers(a: int, b: int) -> int:
        """Add two numbers together for the caller."""
        return a + b

    @fetcher.register(tags=["text"])
    async def shout(text: str) -> str:
        """Uppercase a string of text for the caller."""
        return text.upper()

    return fetcher


def _query(text: str = "add two numbers") -> ToolQuery:
    return ToolQuery(context=ConversationContext(query=text), limit=2, score_threshold=0.0)


def _selector_logs(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
    return [
        (record.levelname, record.getMessage())
        for record in caplog.records
        if record.name == DOTFETCH_LOG and "elector" in record.getMessage()
    ]


# ---------------------------------------------------------------------------
# 1. Which backend answered is on the result
# ---------------------------------------------------------------------------


async def test_a_selector_answer_is_marked_as_the_selectors() -> None:
    fetcher = _fetcher(_Selector({"add_numbers": 0.9}))
    try:
        result = await fetcher.retrieve(_query())
        assert [t.tool.name for t in result.tools] == ["add_numbers"]
        assert result.backend == "selector"
        assert result.selector_fallback is False
        assert result.selector_fallback_reason is None
    finally:
        await fetcher.close()


async def test_a_fallback_is_marked_with_its_reason() -> None:
    """The case that produced a false pass: semantic tools, selector configured."""
    fetcher = _fetcher(_Selector("ConnectError: endpoint unreachable"))
    try:
        result = await fetcher.retrieve(_query())
        assert result.tools, "failing open still returns the semantic router's tools"
        assert result.backend == "semantic"
        assert result.selector_fallback is True
        assert result.selector_fallback_reason == "ConnectError: endpoint unreachable"
        assert result.selection_time_ms is None
    finally:
        await fetcher.close()


async def test_a_decline_is_a_selector_result_not_a_fallback() -> None:
    """Looking at the catalogue and choosing nothing is the selector answering."""
    fetcher = _fetcher(_Selector({}))
    try:
        result = await fetcher.retrieve(_query("what is the capital of France?"))
        assert result.tools == []
        assert result.backend == "selector"
        assert result.selector_fallback is False
    finally:
        await fetcher.close()


async def test_no_selector_is_semantic_without_a_fallback() -> None:
    fetcher = _fetcher(None)
    try:
        result = await fetcher.retrieve(_query())
        assert result.backend == "semantic"
        assert result.selector_fallback is False
        assert result.selector_fallback_reason is None
    finally:
        await fetcher.close()


async def test_a_selector_that_was_never_consulted_is_still_reported() -> None:
    """An empty query never costs a selection request; the result says the
    router answered, so the provenance holds for every call."""
    selector = _Selector({"add_numbers": 0.9})
    fetcher = _fetcher(selector)
    try:
        result = await fetcher.retrieve(_query("   "))
        assert selector.calls == 0
        assert result.backend == "semantic"
        assert result.selector_fallback is True
        assert result.selector_fallback_reason == "the query is empty"
    finally:
        await fetcher.close()


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
    fetcher = _fetcher(_Selector(reason))
    try:
        with caplog.at_level(logging.DEBUG, logger=DOTFETCH_LOG):
            for _ in range(3):
                await fetcher.retrieve(_query())
        logs = _selector_logs(caplog)
        assert [level for level, _ in logs] == ["WARNING", "DEBUG", "DEBUG"]
        assert all(reason in message for _, message in logs)
    finally:
        await fetcher.close()


async def test_a_fallback_is_called_a_failure_not_a_decline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fetcher = _fetcher(_Selector("TimeoutError: no answer"))
    try:
        with caplog.at_level(logging.DEBUG, logger=DOTFETCH_LOG):
            await fetcher.retrieve(_query())
        (level, message), = _selector_logs(caplog)
        assert level == "WARNING"
        assert message.startswith("Selector failed for tool selection (TimeoutError: no answer)")
        assert "declined" not in message
    finally:
        await fetcher.close()


async def test_a_new_reason_and_a_recurrence_after_recovery_warn_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    selector = _Selector("first fault", "first fault", "second fault", {"add_numbers": 0.9}, "second fault")
    fetcher = _fetcher(selector)
    try:
        with caplog.at_level(logging.DEBUG, logger=DOTFETCH_LOG):
            for _ in range(5):
                await fetcher.retrieve(_query())
        levels = [level for level, _ in _selector_logs(caplog)]
        # first fault (new), first fault (repeat), second fault (new),
        # then an answer (nothing logged), then second fault again (news again).
        assert levels == ["WARNING", "DEBUG", "WARNING", "WARNING"]
    finally:
        await fetcher.close()


async def test_nothing_to_select_on_is_not_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    fetcher = _fetcher(_Selector({"add_numbers": 0.9}))
    try:
        with caplog.at_level(logging.DEBUG, logger=DOTFETCH_LOG):
            await fetcher.retrieve(_query(""))
        (level, message), = _selector_logs(caplog)
        assert level == "DEBUG"
        assert message.startswith("Selector not consulted for tool selection")
    finally:
        await fetcher.close()


async def test_the_selectors_own_ceiling_warning_is_once_per_size(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dotfetch.adapters.selectors.system_one import SystemOneSelector
    from dotfetch.schema.selection import SelectionCandidate

    class NeverAsked:
        async def score(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise AssertionError("over the ceiling: the model must not be asked")

    selector = SystemOneSelector(client=NeverAsked(), max_candidates=2)
    three = [SelectionCandidate(id=f"t{i}", name=f"t{i}") for i in range(3)]
    four = [*three, SelectionCandidate(id="t3", name="t3")]
    log = "dotfetch.adapters.selectors.system_one"
    with caplog.at_level(logging.DEBUG, logger=log):
        for candidates in (three, three, four):
            result = await selector.select("q", candidates, limit=2)
            assert result.fallback and "exceeds max_candidates=2" in (result.reason or "")
    levels = [r.levelname for r in caplog.records if r.name == log and "ceiling" in r.getMessage()]
    assert levels == ["WARNING", "DEBUG", "WARNING"]


async def test_the_shipped_client_warns_once_per_outage_not_once_per_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fetcher's once-per-reason line is no use if the client underneath it
    warns on every request for as long as the provider stays down."""
    pytest.importorskip("typesafe_sdk")
    from dotfetch.adapters.selectors.system_one import SystemOneSelector
    from dotfetch.adapters.system_one_client import SystemOneClient

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
    fetcher = _fetcher(SystemOneSelector(client=SystemOneClient(client=sdk)))
    try:
        with caplog.at_level(logging.DEBUG, logger="dotfetch"):
            for _ in range(3):
                assert (await fetcher.retrieve(_query())).selector_fallback is True
            sdk.down = False
            assert (await fetcher.retrieve(_query())).backend == "selector"
            sdk.down = True
            assert (await fetcher.retrieve(_query())).selector_fallback is True

        def warnings_from(logger_name: str, text: str) -> int:
            return sum(
                record.levelno == logging.WARNING
                for record in caplog.records
                if record.name == logger_name and text in record.getMessage()
            )

        # One when the outage starts and one when it recurs after an answer,
        # from each layer; five failed-or-not requests would have been four.
        assert warnings_from("dotfetch.adapters.system_one_client", "selection failed") == 2
        assert warnings_from(DOTFETCH_LOG, "Selector failed") == 2
    finally:
        await fetcher.close()


# ---------------------------------------------------------------------------
# 3. Strict mode: fail loudly instead of failing open
# ---------------------------------------------------------------------------


async def test_strict_mode_raises_instead_of_falling_back() -> None:
    fetcher = _fetcher(_Selector("ConnectError: unreachable"), on_selector_fallback="raise")
    try:
        with pytest.raises(SelectorFallbackError, match="ConnectError: unreachable") as raised:
            await fetcher.retrieve(_query())
        assert raised.value.kind == "tool"
        assert raised.value.reason == "ConnectError: unreachable"
    finally:
        await fetcher.close()


async def test_strict_mode_can_be_set_in_config() -> None:
    config = DotFetchConfig(selector=SelectorConfig(on_fallback="raise"))
    fetcher = DotFetch(
        config=config, embedder=SimpleEmbedder(), selector=_Selector("boom")
    )
    try:

        @fetcher.register
        def ping() -> str:
            """Reply with pong."""
            return "pong"

        with pytest.raises(SelectorFallbackError):
            await fetcher.retrieve(_query("ping"))
    finally:
        await fetcher.close()


async def test_strict_mode_leaves_answers_and_declines_alone() -> None:
    fetcher = _fetcher(_Selector({"add_numbers": 0.9}, {}), on_selector_fallback="raise")
    try:
        answered = await fetcher.retrieve(_query())
        declined = await fetcher.retrieve(_query("what is the capital of France?"))
        assert answered.backend == declined.backend == "selector"
        assert declined.tools == []
    finally:
        await fetcher.close()


async def test_strict_mode_does_not_raise_when_there_is_nothing_to_select_on() -> None:
    """An empty query is not a selector failure; it is reported, not raised."""
    fetcher = _fetcher(_Selector({"add_numbers": 0.9}), on_selector_fallback="raise")
    try:
        result = await fetcher.retrieve(_query(""))
        assert result.selector_fallback is True
    finally:
        await fetcher.close()


async def test_strict_mode_covers_skills_and_mcp_servers() -> None:
    fetcher = DotFetch(
        embedder=SimpleEmbedder(), selector=_Selector("boom"), on_selector_fallback="raise"
    )
    try:
        await fetcher.add_skill(
            Skill(
                name="deploying",
                description="How to deploy the service to production safely.",
                content="Run the deploy script.",
            )
        )
        with pytest.raises(SelectorFallbackError) as raised:
            await fetcher.retrieve_skills("how do I ship this", limit=1)
        assert raised.value.kind == "skill"

        if fetcher._mcp_registry is not None:
            fetcher.register_mcp_server(
                "files", ["echo", "files"], description="Reads and writes local files."
            )
            with pytest.raises(SelectorFallbackError) as raised:
                await fetcher.retrieve_mcp_servers("open a file", limit=1)
            assert raised.value.kind == "mcp_server"
    finally:
        await fetcher.close()


async def test_skill_and_mcp_fallbacks_warn_too(caplog: pytest.LogCaptureFixture) -> None:
    fetcher = DotFetch(embedder=SimpleEmbedder(), selector=_Selector("boom"))
    try:
        await fetcher.add_skill(
            Skill(
                name="deploying",
                description="How to deploy the service to production safely.",
                content="Run the deploy script.",
            )
        )
        with caplog.at_level(logging.DEBUG, logger=DOTFETCH_LOG):
            await fetcher.retrieve_skills("how do I ship this", limit=1)
            await fetcher.retrieve_skills("how do I ship this", limit=1)
        logs = _selector_logs(caplog)
        assert [level for level, _ in logs] == ["WARNING", "DEBUG"]
        assert logs[0][1].startswith("Selector failed for skill selection (boom)")
    finally:
        await fetcher.close()


def test_an_unknown_strict_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="'semantic' or 'raise'"):
        DotFetch(embedder=SimpleEmbedder(), on_selector_fallback="explode")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_the_selector_surface_is_importable_from_the_package_root() -> None:
    import dotfetch
    from dotfetch.adapters import selectors
    from dotfetch.adapters.system_one_client import SystemOneClient

    assert dotfetch.SystemOneClient is SystemOneClient
    assert dotfetch.SelectorFallbackError is selectors.SelectorFallbackError
    assert {"SystemOneClient", "SystemOneSelector", "SystemOneReranker", "SelectorFallbackError"} <= set(
        dotfetch.__all__
    )


# ---------------------------------------------------------------------------
# The decision record a provider keeps says which backend it came from
# ---------------------------------------------------------------------------


async def test_the_retrieval_decision_carries_the_backend() -> None:
    """``provider.selections`` / ``last_selection`` / ``dry_run_retrieve`` all
    hand back a ``RetrievalDecision``; an eval reads the backend off that."""
    pytest.importorskip("agent_framework")
    from dotfetch.integrations.agent_framework_bridge import DotFetchToolBridge

    selector = _Selector(
        {"add_numbers": 0.9}, "ConnectError: unreachable", "ConnectError: unreachable"
    )
    fetcher = _fetcher(selector)
    try:
        bridge = DotFetchToolBridge(fetcher)
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
        await fetcher.close()


async def test_a_provider_dry_run_reports_the_fallback() -> None:
    pytest.importorskip("agent_framework")
    from dotfetch import DotFetchContextProvider

    fetcher = _fetcher(_Selector("catalogue of 600 exceeds max_candidates=512"))
    try:
        provider = DotFetchContextProvider(fetcher, top_k=2)
        decision = await provider.dry_run_retrieve("add two numbers")
        assert decision.selector_fallback is True
        assert decision.selector_fallback_reason == "catalogue of 600 exceeds max_candidates=512"
    finally:
        await fetcher.close()


# ---------------------------------------------------------------------------
# Strict mode holds through the wrappers that otherwise survive a failure
# ---------------------------------------------------------------------------
#
# Every integration catches a retrieval failure and carries on without (or
# with the previous) tools, so that a broken vector store cannot end a run.
# ``on_selector_fallback="raise"`` is the caller asking for the opposite. An
# integration that swallowed it would leave strict mode holding for direct
# ``retrieve()`` calls only, and an eval run through an agent would pass with
# no tools at all.


async def test_strict_mode_reaches_the_caller_of_with_semantic_tools() -> None:
    from dotfetch import with_semantic_tools

    fetcher = _fetcher(_Selector("ConnectError: unreachable"), on_selector_fallback="raise")
    called: list[Any] = []
    try:

        @with_semantic_tools(fetcher, limit=2)
        async def generate(prompt: str, *, tools: list[Any] | None = None) -> str:
            called.append(tools)
            return "ok"

        with pytest.raises(SelectorFallbackError, match="ConnectError: unreachable"):
            await generate("add two numbers")
        assert called == []  # the model was never called without the selector
    finally:
        await fetcher.close()


def test_strict_mode_reaches_the_caller_of_a_sync_with_semantic_tools() -> None:
    from dotfetch import with_semantic_tools

    fetcher = _fetcher(_Selector("ConnectError: unreachable"), on_selector_fallback="raise")
    called: list[Any] = []

    @with_semantic_tools(fetcher, limit=2)
    def generate(prompt: str, *, tools: list[Any] | None = None) -> str:
        called.append(tools)
        return "ok"

    with pytest.raises(SelectorFallbackError, match="ConnectError: unreachable"):
        generate("add two numbers")
    assert called == []


async def test_a_fail_open_fetcher_still_survives_through_with_semantic_tools() -> None:
    """The default is untouched: the fallback serves the call, and says so."""
    from dotfetch import with_semantic_tools

    fetcher = _fetcher(_Selector("ConnectError: unreachable"))
    called: list[Any] = []
    try:

        @with_semantic_tools(fetcher, limit=2)
        async def generate(prompt: str, *, tools: list[Any] | None = None) -> str:
            called.append(tools)
            return "ok"

        assert await generate("add two numbers") == "ok"
        assert called and called[0]  # the semantic router's tools were injected
    finally:
        await fetcher.close()


async def test_strict_mode_ends_a_context_provider_run() -> None:
    af = pytest.importorskip("agent_framework")
    from dotfetch import DotFetchContextProvider

    fetcher = _fetcher(_Selector("ConnectError: unreachable"), on_selector_fallback="raise")
    try:
        provider = DotFetchContextProvider(fetcher, top_k=2, score_threshold=0.0)
        context = af.SessionContext(
            input_messages=[af.Message(role="user", contents=["add two numbers"])]
        )
        with pytest.raises(SelectorFallbackError):
            await provider.before_run(agent=None, session=None, context=context, state={})
        assert provider.last_selection is None
    finally:
        await fetcher.close()


async def test_strict_mode_reaches_the_per_turn_hooks() -> None:
    """Two hooks that need no framework installed; the guard below covers the rest."""
    import types

    from dotfetch.integrations.frameworks.google_adk_live import _inject_selected_tools
    from dotfetch.integrations.frameworks.langgraph_live import _select_tools_for_state

    fetcher = _fetcher(_Selector("ConnectError: unreachable"), on_selector_fallback="raise")
    try:
        state = {"messages": [{"role": "user", "content": "add two numbers"}]}
        with pytest.raises(SelectorFallbackError):
            await _select_tools_for_state(fetcher, state, limit=2, score_threshold=0.0)
        with pytest.raises(SelectorFallbackError):
            await _inject_selected_tools(
                fetcher, "add two numbers", types.SimpleNamespace(), limit=2, score_threshold=0.0
            )
    finally:
        await fetcher.close()


#: Calls that reach ``DotFetch.retrieve`` (or its skill twin) from an
#: integration. A ``try`` around one of these that also catches ``Exception``
#: is a place where strict mode can be swallowed.
_RETRIEVAL_CALLS = {
    "select",
    "select_or_empty",
    "_select",
    "retrieve",
    "retrieve_tools",
    "_retrieve_tools",
    "retrieve_skills",
    "retrieve_skills_as_prompt",
    "get_tools_with_decision",
    "_refresh_tools_on_chat_context",
}


def _swallowing_handlers() -> list[str]:
    """``file:line`` of each broad handler around retrieval that strict mode cannot pass."""
    import ast
    from pathlib import Path

    import dotfetch

    def names(node: ast.expr | None) -> set[str]:
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} if node else set()

    offenders: list[str] = []
    root = Path(dotfetch.__file__).parent / "integrations"
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Try):
                continue
            retrieves = any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in _RETRIEVAL_CALLS
                for statement in node.body
                for call in ast.walk(statement)
            )
            if not retrieves:
                continue
            let_through = False
            for handler in node.handlers:
                caught = names(handler.type)
                if "SelectorFallbackError" in caught:
                    let_through = len(handler.body) == 1 and isinstance(handler.body[0], ast.Raise)
                elif handler.type is None or caught & {"Exception", "BaseException"}:
                    if not let_through:
                        offenders.append(f"{path.relative_to(root).as_posix()}:{handler.lineno}")
                    break
    return offenders


def test_no_integration_swallows_strict_mode() -> None:
    assert _swallowing_handlers() == []
