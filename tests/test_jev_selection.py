"""
Selection through TypeSafe's Jev model, as an alternative to semantic matching.

Runs against the real ``typesafe_sdk`` question and response models — the SDK
is days old and explicitly early access, so building genuine ``Noul`` objects
and genuine ``SystemOneResponse`` objects is what would catch a signature
change — but never against the network: a stand-in client answers in process.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_gantry.adapters.jev_client import (
    JevClient,
    JevVerdict,
    candidate_payload,
    estimate_tokens,
)
from agent_gantry.adapters.rerankers.jev import JevReranker
from agent_gantry.adapters.selectors.jev import JevSelector
from agent_gantry.schema.selection import SelectionCandidate, SelectionResult
from agent_gantry.schema.tool import ToolDefinition

typesafe_sdk = pytest.importorskip("typesafe_sdk")


# ----------------------------------------------------------------------
# Stand-ins
# ----------------------------------------------------------------------


class _StubTypeSafe:
    """Stands in for ``AsyncTypeSafeClient``, answering from a score table.

    Scores are looked up by the candidate *name* carried in each question's
    instructions, which also pins the payload shape: if the candidate stopped
    being sent under ``instructions["candidate"]["name"]``, every lookup here
    would miss.
    """

    def __init__(
        self,
        scores: dict[str, float],
        *,
        fail_on_call: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.scores = scores
        self.calls: list[dict[str, Any]] = []
        self._fail_on_call = fail_on_call
        self._error = error or RuntimeError("boom")

    async def system_one(
        self, state: Any, questions: dict[str, Any], **_: Any
    ) -> Any:
        self.calls.append({"state": state, "questions": questions})
        if self._fail_on_call is not None and len(self.calls) == self._fail_on_call:
            raise self._error
        answers = {}
        for key, question in questions.items():
            name = question.instructions["candidate"]["name"]
            if name in self.scores:
                answers[key] = typesafe_sdk.NoulAnswer(
                    type="noul", noul=self.scores[name]
                )
        return typesafe_sdk.SystemOneResponse(
            model="jev-test",
            usage=typesafe_sdk.Usage(input_tokens=11, output_tokens=0),
            answers=answers,
        )

    async def aclose(self) -> None:  # pragma: no cover - never owned by us
        raise AssertionError("a supplied client must not be closed by the wrapper")


class _StubJevClient:
    """Stands in for :class:`JevClient` at the verdict level."""

    def __init__(self, *verdicts: JevVerdict) -> None:
        self._verdicts = list(verdicts)
        self.scored: list[list[str]] = []

    async def score(self, state: Any, candidates: Any, question: str) -> JevVerdict:
        self.scored.append([c.id for c in candidates])
        return self._verdicts.pop(0) if self._verdicts else JevVerdict()


def _client(scores: dict[str, float], **kwargs: Any) -> tuple[JevClient, _StubTypeSafe]:
    stub = _StubTypeSafe(scores, **kwargs)
    return JevClient(client=stub), stub


def _tool(name: str, description: str | None = None) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description or f"The {name} tool does something useful.",
        parameters_schema={"type": "object", "properties": {}},
    )


def _candidate(name: str, group: str | None = None) -> SelectionCandidate:
    return SelectionCandidate(
        id=name, name=name, description=f"{name} does a thing.", group=group
    )


# ----------------------------------------------------------------------
# Payload shaping
# ----------------------------------------------------------------------


def test_a_long_description_is_cut_on_a_word_boundary() -> None:
    candidate = SelectionCandidate(
        id="x", name="x", description="alpha beta gamma delta epsilon", tags=["t"]
    )
    payload = candidate_payload(candidate, 20)
    assert payload["description"].endswith("...")
    assert "delt" not in payload["description"], "cut mid-word"
    assert payload["name"] == "x"
    assert payload["tags"] == ["t"]


def test_a_short_description_is_sent_whole_and_empty_ones_are_omitted() -> None:
    assert candidate_payload(_candidate("a"), 500)["description"] == "a does a thing."
    bare = SelectionCandidate(id="b", name="b")
    assert "description" not in candidate_payload(bare, 500)


def test_token_estimates_grow_with_payload_size() -> None:
    assert estimate_tokens("x" * 350) > estimate_tokens("x" * 35)
    assert estimate_tokens({"a": "x" * 100}) > estimate_tokens({"a": "x"})


# ----------------------------------------------------------------------
# Batching against the per-request token budget
# ----------------------------------------------------------------------


def test_candidates_are_split_into_requests_that_fit_the_budget() -> None:
    client = JevClient(client=_StubTypeSafe({}), token_budget=120)
    candidates = [
        SelectionCandidate(id=str(i), name=f"tool_{i}", description="d " * 40)
        for i in range(10)
    ]
    batches = client.batch("find something", candidates, "does it help?")

    assert len(batches) > 1, "one batch means the budget was ignored"
    assert [c.id for batch in batches for c in batch] == [c.id for c in candidates]
    assert all(batch for batch in batches), "an empty batch would waste a request"


def test_a_candidate_too_large_for_the_budget_still_gets_sent() -> None:
    """Dropping it would silently remove a tool from the catalogue.

    Truncation already bounds how large one candidate can be, so the remedy is
    a batch of its own rather than an exclusion the caller never hears about.
    """
    client = JevClient(client=_StubTypeSafe({}), token_budget=1)
    batches = client.batch("q", [_candidate("a"), _candidate("b")], "?")
    assert [c.id for batch in batches for c in batch] == ["a", "b"]


# ----------------------------------------------------------------------
# Scoring and failing open
# ----------------------------------------------------------------------


async def test_scores_come_back_keyed_by_candidate_id() -> None:
    client, stub = _client({"a": 0.9, "b": 0.1})
    verdict = await client.score("q", [_candidate("a"), _candidate("b")], "help?")

    assert verdict.scores == {"a": 0.9, "b": 0.1}
    assert verdict.fallback is False
    assert verdict.requests == 1
    assert verdict.input_tokens == 11
    # The query is the state; the candidates are not. Jev's accuracy falls as
    # the state fills with content unrelated to the decision.
    assert stub.calls[0]["state"] == "q"


async def test_an_api_failure_falls_back_instead_of_raising() -> None:
    client, _stub = _client({"a": 0.9}, fail_on_call=1)
    verdict = await client.score("q", [_candidate("a")], "help?")

    assert verdict.fallback is True
    assert verdict.scores == {}
    assert "RuntimeError" in (verdict.reason or "")


async def test_one_failed_batch_discards_the_whole_pass() -> None:
    """A partial ranking is worse than none.

    A candidate that was never scored is indistinguishable from one scored
    zero, so keeping the batches that succeeded would silently bury everything
    in the batch that did not.
    """
    client = JevClient(
        client=_StubTypeSafe({"a": 0.9, "b": 0.8}, fail_on_call=2), token_budget=1
    )
    verdict = await client.score("q", [_candidate("a"), _candidate("b")], "help?")

    assert verdict.fallback is True
    assert verdict.scores == {}


async def test_an_unanswered_candidate_is_absent_rather_than_zero() -> None:
    """Scoring it zero would rank it below things the model actually rejected."""
    client, _stub = _client({"a": 0.9})
    verdict = await client.score("q", [_candidate("a"), _candidate("b")], "help?")

    assert verdict.scores == {"a": 0.9}
    assert verdict.fallback is False


async def test_a_supplied_client_is_not_closed_by_the_wrapper() -> None:
    client, _stub = _client({})
    await client.aclose()  # the stub raises if it is closed


# ----------------------------------------------------------------------
# Reranker
# ----------------------------------------------------------------------


async def test_the_reranker_reorders_by_probability() -> None:
    client, _stub = _client({"default.beta": 0.95, "default.alpha": 0.05})
    # The stub keys off the name the model is shown, which deliberately has no
    # version in it even though the id does.
    reranker = JevReranker(client=client)
    tools = [(_tool("alpha"), 0.9), (_tool("beta"), 0.1)]

    ranked = await reranker.rerank("do the beta thing", tools, top_k=2)

    assert [tool.name for tool, _ in ranked] == ["beta", "alpha"]
    assert ranked[0][1] == 0.95


async def test_the_reranker_keeps_search_order_when_the_api_fails() -> None:
    client, _stub = _client({}, fail_on_call=1)
    reranker = JevReranker(client=client)
    tools = [(_tool("alpha"), 0.9), (_tool("beta"), 0.1)]

    ranked = await reranker.rerank("q", tools, top_k=2)

    assert [tool.name for tool, _ in ranked] == ["alpha", "beta"]
    assert [score for _, score in ranked] == [0.9, 0.1]


async def test_unscored_tools_rank_below_scored_ones_but_are_not_dropped() -> None:
    client, _stub = _client({"default.beta": 0.4})
    reranker = JevReranker(client=client)
    tools = [(_tool("alpha"), 0.9), (_tool("beta"), 0.1)]

    ranked = await reranker.rerank("q", tools, top_k=2)

    assert [tool.name for tool, _ in ranked] == ["beta", "alpha"]


async def test_the_reranker_returns_nothing_for_nothing() -> None:
    client, stub = _client({})
    assert await JevReranker(client=client).rerank("q", [], top_k=5) == []
    assert stub.calls == [], "an empty shortlist must not cost a request"


# ----------------------------------------------------------------------
# Selector
# ----------------------------------------------------------------------


async def test_the_selector_keeps_only_candidates_over_the_threshold() -> None:
    client, _stub = _client({"a": 0.91, "b": 0.42, "c": 0.02})
    selector = JevSelector(client=client, threshold=0.3)

    result = await selector.select(
        "q", [_candidate("a"), _candidate("b"), _candidate("c")], limit=10
    )

    assert result.selected == ["a", "b"]
    # Every score is reported, not just the winners, so a caller can see what
    # the threshold cut.
    assert result.scores == {"a": 0.91, "b": 0.42, "c": 0.02}
    assert result.fallback is False


async def test_the_selector_respects_the_limit() -> None:
    client, _stub = _client({"a": 0.9, "b": 0.8, "c": 0.7})
    selector = JevSelector(client=client, threshold=0.0)

    result = await selector.select(
        "q", [_candidate("a"), _candidate("b"), _candidate("c")], limit=2
    )

    assert result.selected == ["a", "b"]


async def test_the_selector_falls_back_rather_than_raising() -> None:
    client, _stub = _client({}, fail_on_call=1)
    result = await JevSelector(client=client).select("q", [_candidate("a")], limit=5)

    assert result.fallback is True
    assert result.selected == []


async def test_a_large_grouped_catalogue_narrows_by_group_first() -> None:
    """Two requests, not one per tool: the group pass is the budget device."""
    members = [_candidate(f"t{i}", group="alpha") for i in range(5)]
    members += [_candidate(f"u{i}", group="beta") for i in range(5)]

    stub = _StubJevClient(
        JevVerdict(scores={"alpha": 0.9, "beta": 0.1}),
        JevVerdict(scores={f"t{i}": 0.8 for i in range(5)}),
    )
    selector = JevSelector(client=stub, group_after=5, max_groups=1)

    result = await selector.select("q", members, limit=10)

    assert stub.scored[0] == ["alpha", "beta"], "first pass scores groups"
    assert stub.scored[1] == [f"t{i}" for i in range(5)], "second pass scores members"
    assert result.selected == [f"t{i}" for i in range(5)]


async def test_the_group_pass_is_not_gated_by_the_threshold() -> None:
    """It exists to fit the budget, not to decide relevance.

    Gating it as well would let one cautious answer zero out the catalogue
    before any individual tool had been looked at.
    """
    members = [_candidate(f"t{i}", group="alpha") for i in range(3)]
    members += [_candidate(f"u{i}", group="beta") for i in range(3)]

    stub = _StubJevClient(
        # Both groups sit well under the 0.3 threshold.
        JevVerdict(scores={"alpha": 0.05, "beta": 0.01}),
        JevVerdict(scores={"t0": 0.99}),
    )
    selector = JevSelector(client=stub, threshold=0.3, group_after=2, max_groups=1)

    result = await selector.select("q", members, limit=10)

    assert stub.scored[1] == ["t0", "t1", "t2"], "the best group was still opened"
    assert result.selected == ["t0"]


async def test_a_failed_group_pass_falls_back_rather_than_scoring_everything() -> None:
    """Scoring every member instead would be the budget blow-out this avoids."""
    members = [_candidate(f"t{i}", group=f"g{i % 3}") for i in range(9)]
    stub = _StubJevClient(JevVerdict(fallback=True, reason="429"))
    selector = JevSelector(client=stub, group_after=2)

    result = await selector.select("q", members, limit=10)

    assert result.fallback is True
    assert len(stub.scored) == 1, "no second pass after the first failed"


async def test_an_ungrouped_catalogue_stays_single_stage() -> None:
    client, stub = _client({f"t{i}": 0.9 for i in range(6)})
    selector = JevSelector(client=client, group_after=2)

    result = await selector.select(
        "q", [_candidate(f"t{i}") for i in range(6)], limit=10
    )

    assert len(result.selected) == 6
    assert len(stub.calls) == 1, "nothing to group by, so no group pass"


async def test_the_selector_returns_nothing_for_nothing() -> None:
    client, stub = _client({})
    assert (await JevSelector(client=client).select("q", [], limit=5)).selected == []
    assert (await JevSelector(client=client).select("q", [_candidate("a")], 0)).selected == []
    assert stub.calls == []


# ----------------------------------------------------------------------
# Wiring into the facade
# ----------------------------------------------------------------------


class _StubSelector:
    """Stands in for a selector adapter, answering from a fixed table."""

    def __init__(
        self, scores: dict[str, float] | None = None, *, fallback: bool = False
    ) -> None:
        self.scores = scores or {}
        self.fallback = fallback
        self.seen: list[list[str]] = []

    async def select(self, query: str, candidates: Any, limit: int) -> SelectionResult:
        self.seen.append([c.id for c in candidates])
        if self.fallback:
            return SelectionResult(fallback=True, reason="stub declined")
        scores = {c.id: self.scores.get(c.name, 0.0) for c in candidates}
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return SelectionResult(
            selected=[cid for cid, score in ranked if score > 0][:limit], scores=scores
        )


async def _gantry_with(selector: Any) -> Any:
    from agent_gantry import AgentGantry

    gantry = AgentGantry(selector=selector)

    @gantry.register(tags=["math"])
    async def add_numbers(a: int, b: int) -> int:
        """Add two numbers together for the caller."""
        return a + b

    @gantry.register(tags=["text"])
    async def shout(text: str) -> str:
        """Uppercase a string of text for the caller."""
        return text.upper()

    return gantry


async def test_tool_retrieval_uses_the_selector_and_never_embeds() -> None:
    selector = _StubSelector({"default.shout": 0.9})
    gantry = await _gantry_with(selector)
    try:
        result = await gantry.retrieve_tools("make this loud", limit=3)

        assert [schema["function"]["name"] for schema in result] == ["shout"]
        assert selector.seen, "the selector was never consulted"
    finally:
        await gantry.close()


async def test_the_selector_path_reports_no_embedding_or_search_time() -> None:
    from agent_gantry.schema.query import ConversationContext, ToolQuery

    selector = _StubSelector({"default.shout": 0.9})
    gantry = await _gantry_with(selector)
    try:
        retrieval = await gantry.retrieve(
            ToolQuery(context=ConversationContext(query="be loud"), limit=3, score_threshold=0.0)
        )

        assert retrieval.query_embedding_time_ms == 0.0
        assert retrieval.vector_search_time_ms == 0.0
        assert retrieval.selection_time_ms is not None
        assert [scored.tool.name for scored in retrieval.tools] == ["shout"]
    finally:
        await gantry.close()


async def test_a_declining_selector_falls_back_to_semantic_routing() -> None:
    selector = _StubSelector(fallback=True)
    gantry = await _gantry_with(selector)
    try:
        result = await gantry.retrieve_tools("add two numbers", limit=2)

        assert selector.seen, "the selector should have been tried first"
        # The semantic path ran instead, so tools still came back.
        assert result, "falling back must not cost the catalogue"
    finally:
        await gantry.close()


async def test_a_deprecated_tool_never_reaches_the_selector() -> None:
    """The query's hard constraints are applied before selection.

    Otherwise a selector would surface tools the semantic path hides, which is
    a silent widening of what the agent can reach rather than a ranking change.
    """
    from agent_gantry import AgentGantry

    selector = _StubSelector({"default.old_tool": 0.99})
    gantry = AgentGantry(selector=selector)
    try:
        await gantry.add_tool(
            ToolDefinition(
                name="old_tool",
                description="A tool that has since been deprecated.",
                parameters_schema={"type": "object", "properties": {}},
                deprecated=True,
            ),
            lambda: None,
        )
        result = await gantry.retrieve_tools("use the old tool", limit=3)

        assert result == []
        assert selector.seen == [], "no allowed tools, so nothing to select from"
    finally:
        await gantry.close()


async def test_a_blank_query_never_costs_a_selection_request() -> None:
    selector = _StubSelector({"default.shout": 0.9})
    gantry = await _gantry_with(selector)
    try:
        await gantry.retrieve_tools("   ", limit=3)
        assert selector.seen == []
    finally:
        await gantry.close()


async def test_skill_retrieval_uses_the_selector() -> None:
    from agent_gantry import AgentGantry
    from agent_gantry.schema.skill import Skill

    selector = _StubSelector({"default.deploying": 0.8})
    gantry = AgentGantry(selector=selector)
    try:
        await gantry.add_skill(
            Skill(
                name="deploying",
                description="How to deploy the service to production safely.",
                content="Run the deploy script.",
            )
        )
        await gantry.add_skill(
            Skill(
                name="testing",
                description="How to run the test suite and read its output.",
                content="Run pytest.",
            )
        )
        found = await gantry.retrieve_skills("how do I ship this", limit=2)

        assert [result.skill.name for result in found] == ["deploying"]
        assert found[0].score == pytest.approx(0.8)
    finally:
        await gantry.close()


async def test_mcp_server_retrieval_uses_the_selector() -> None:
    from agent_gantry import AgentGantry

    selector = _StubSelector({"default.files": 0.7})
    gantry = AgentGantry(selector=selector)
    if gantry._mcp_registry is None:  # pragma: no cover - mcp extra absent
        await gantry.close()
        pytest.skip("MCP support is not installed")
    try:
        gantry.register_mcp_server(
            "files",
            ["echo", "files"],
            description="Reads and writes files on the local disk.",
        )
        gantry.register_mcp_server(
            "search",
            ["echo", "search"],
            description="Searches the web for pages matching a query.",
        )
        servers = await gantry.retrieve_mcp_servers("open a file", limit=2)

        assert [server.name for server in servers] == ["files"]
    finally:
        await gantry.close()


# ----------------------------------------------------------------------
# Findings from review on #419
# ----------------------------------------------------------------------


async def test_a_tool_added_without_a_handler_still_reaches_the_selector() -> None:
    """``add_tool(tool)`` with no handler lands only in the pending buffer.

    That is how MCP and A2A discovery add theirs. Building the catalogue from
    the registry alone made them invisible to selection — and because the other
    tools still let the selector answer, the semantic fallback never ran, so
    they stayed unreachable rather than merely unranked.
    """
    from agent_gantry import AgentGantry
    from agent_gantry.schema.config import AgentGantryConfig

    selector = _StubSelector({"default.discovered": 0.9})
    gantry = AgentGantry(config=AgentGantryConfig(auto_sync=False), selector=selector)
    try:
        await gantry.add_tool(_tool("registered"), lambda: None)
        await gantry.add_tool(_tool("discovered"))  # no handler -> pending only

        await gantry.retrieve_tools("anything", limit=5)

        assert selector.seen, "the selector was never consulted"
        assert any("discovered" in candidate_id for candidate_id in selector.seen[0])
    finally:
        await gantry.close()


async def test_a_catalogue_over_the_ceiling_falls_back_rather_than_truncating() -> None:
    """Scoring an insertion-order prefix would strand everything after it.

    Reporting success while silently ignoring the tail means semantic routing —
    which has no such ceiling — never gets a look in, so a tool registered past
    the ceiling could never be retrieved however relevant it was.
    """
    selector = JevSelector(client=_StubJevClient(), max_candidates=3)
    candidates = [_candidate(f"t{i}") for i in range(10)]

    result = await selector.select("q", candidates, limit=5)

    assert result.fallback is True
    assert "max_candidates" in (result.reason or "")
    assert result.selected == []


async def test_an_ungrouped_candidate_does_not_break_the_two_stage_path() -> None:
    """``group=None`` bucketed to ``""``, which ``SelectionCandidate`` rejects.

    The ``ValidationError`` escaped ``select()`` instead of failing open, so a
    single ungrouped entry in a large catalogue took the caller's tools away.
    """
    stub = _StubJevClient(
        JevVerdict(scores={"g1": 0.9, "g2": 0.4, "(ungrouped)": 0.2}),
        JevVerdict(scores={"a": 0.8}),
    )
    selector = JevSelector(client=stub, group_after=2, max_groups=1)
    candidates = [
        _candidate("a", group="g1"),
        _candidate("b", group="g2"),
        _candidate("c"),  # no group at all
    ]

    result = await selector.select("q", candidates, limit=5)

    assert result.fallback is False
    assert result.selected == ["a"]
    assert "(ungrouped)" in stub.scored[0], "the ungrouped bucket must still be offered"


async def test_an_adapter_defect_falls_open_like_a_provider_failure() -> None:
    """Fail-open is the contract, and a bug in here must not break it."""

    class _Exploding:
        async def score(self, state: Any, candidates: Any, question: str) -> Any:
            raise ValueError("adapter defect")

    result = await JevSelector(client=_Exploding()).select("q", [_candidate("a")], limit=5)

    assert result.fallback is True
    assert "ValueError" in (result.reason or "")


async def test_tool_selection_never_opens_the_vector_store() -> None:
    """A selector-only deployment should not need a store to be reachable.

    Initialising eagerly meant an unavailable or deliberately unused store took
    down the one path that never reads it.
    """
    from agent_gantry import AgentGantry

    class _ExplodingStore:
        async def initialize(self) -> None:
            raise AssertionError("the selector path must not open the vector store")

    selector = _StubSelector({"default.alpha": 0.9})
    gantry = AgentGantry(selector=selector)
    await gantry.add_tool(_tool("alpha"), lambda: None)
    gantry._vector_store = _ExplodingStore()
    gantry._initialized = False

    result = await gantry.retrieve_tools("find alpha", limit=2)
    assert [schema["function"]["name"] for schema in result] == ["alpha"]


async def test_mcp_selection_never_syncs_server_vectors() -> None:
    """``register_server`` already populates the registry selection reads.

    Syncing first embedded every server for a path that never reads those
    vectors, and broke the embedding-free route whenever the embedder or store
    was unavailable.
    """
    from agent_gantry import AgentGantry

    selector = _StubSelector({"default.files": 0.9})
    gantry = AgentGantry(selector=selector)
    if gantry._mcp_registry is None:  # pragma: no cover - mcp extra absent
        await gantry.close()
        pytest.skip("MCP support is not installed")

    gantry.register_mcp_server(
        "files", ["echo", "files"], description="Reads and writes files on disk."
    )

    async def _explode(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the selector path must not sync MCP vectors")

    gantry.sync_mcp_servers = _explode  # type: ignore[method-assign]

    servers = await gantry.retrieve_mcp_servers("open a file", limit=2)
    assert [server.name for server in servers] == ["files"]
