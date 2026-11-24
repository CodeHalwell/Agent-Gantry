"""
Measure retrieval quality against a labelled set of queries.

The registry linters (``analyze_registry``, ``pairwise_similarity``) measure the
*catalogue*: overlapping descriptions, weak names. This module measures
*retrieval*: given a query whose right answer is known, did it come back, and
when nothing in the catalogue fits, was nothing returned.

It is deliberately small and has no dependencies. A case names a query and the
ids that should come back; a retriever is any coroutine that turns a query into
ranked ids. The same runner therefore serves tools, skills and MCP servers, and
any backend (semantic routing, a selector, a reranker), because the backend is
whatever the gantry behind the retriever was built with.

Example:
    >>> from agent_gantry.evaluation import RetrievalCase, evaluate_retrieval, tool_retriever
    >>> cases = [
    ...     RetrievalCase("What is 17 plus 25?", {"default.add_numbers"}),
    ...     RetrievalCase("Book me a flight to Tokyo", expect_decline=True),
    ... ]
    >>> report = await evaluate_retrieval(tool_retriever(gantry), cases, k=5)  # doctest: +SKIP
    >>> report.recall_at_1, report.decline_correctness  # doctest: +SKIP
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

__all__ = [
    "CaseOutcome",
    "RetrievalCase",
    "RetrievalReport",
    "Retriever",
    "compare_to_baseline",
    "evaluate_retrieval",
    "format_reports",
    "mcp_server_retriever",
    "skill_retriever",
    "tool_retriever",
]


@dataclass(frozen=True)
class RetrievalCase:
    """One labelled query.

    Attributes:
        query: The text to retrieve for.
        expected: Ids that count as a right answer, in any order. For tools,
            skills and MCP servers the id is ``"<namespace>.<name>"``.
        expect_decline: ``True`` when nothing in the catalogue fits, so the
            right answer is an empty result. ``expected`` must then be empty.
        kind: A free label for slicing the report, e.g. ``"paraphrase"`` (the
            query shares meaning but not words with its answer) or
            ``"identifier"`` (the query names a token the answer carries).
    """

    query: str
    expected: frozenset[str] = frozenset()
    expect_decline: bool = False
    kind: str = "paraphrase"

    def __init__(
        self,
        query: str,
        expected: Iterable[str] = (),
        expect_decline: bool = False,
        kind: str = "paraphrase",
    ) -> None:
        expected = frozenset(expected)
        if expect_decline and expected:
            raise ValueError("A case that expects a decline cannot also expect ids.")
        if not expect_decline and not expected:
            raise ValueError(
                "A case needs expected ids, or expect_decline=True if nothing should come back."
            )
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "expected", expected)
        object.__setattr__(self, "expect_decline", expect_decline)
        object.__setattr__(self, "kind", kind)


#: Turns ``(query, limit)`` into ranked ids, best first. It may also return
#: ``(ids, backend)`` to say which backend answered, e.g. ``"selector"``.
Retriever = Callable[[str, int], Awaitable["Sequence[str] | tuple[Sequence[str], str]"]]


@dataclass(frozen=True)
class CaseOutcome:
    """What came back for one case."""

    case: RetrievalCase
    returned: tuple[str, ...]
    latency_ms: float
    backend: str | None = None

    @property
    def hit_at_1(self) -> bool:
        """The top result is one of the expected ids."""
        return bool(self.returned) and self.returned[0] in self.case.expected

    @property
    def recall(self) -> float:
        """Fraction of the expected ids found anywhere in the returned list."""
        if not self.case.expected:
            return 0.0
        return len(self.case.expected.intersection(self.returned)) / len(self.case.expected)

    @property
    def declined(self) -> bool:
        """Nothing was returned."""
        return not self.returned


@dataclass(frozen=True)
class RetrievalReport:
    """Aggregate retrieval quality over a set of cases.

    Attributes:
        modality: What was retrieved, e.g. ``"tools"``.
        k: The ``limit`` every query was run with.
        recall_at_1: Fraction of answerable cases whose top result was an
            expected id. ``None`` when there were no answerable cases.
        recall_at_k: Mean fraction of each answerable case's expected ids found
            in its top ``k``. ``None`` when there were no answerable cases.
        decline_correctness: Fraction of unanswerable cases that returned
            nothing. ``None`` when there were no unanswerable cases.
        mean_latency_ms: Mean wall-clock time per query.
        backends: How many cases each backend answered, when the retriever
            reported one.
        outcomes: The per-case results the aggregates were computed from.
    """

    modality: str
    k: int
    recall_at_1: float | None
    recall_at_k: float | None
    decline_correctness: float | None
    mean_latency_ms: float
    backends: dict[str, int] = field(default_factory=dict)
    outcomes: tuple[CaseOutcome, ...] = ()

    @property
    def answerable(self) -> int:
        """Number of cases that expected ids back."""
        return sum(1 for o in self.outcomes if not o.case.expect_decline)

    @property
    def unanswerable(self) -> int:
        """Number of cases that expected a decline."""
        return sum(1 for o in self.outcomes if o.case.expect_decline)

    def by_kind(self) -> dict[str, RetrievalReport]:
        """The same report, split by each case's ``kind`` label."""
        kinds: dict[str, list[CaseOutcome]] = {}
        for outcome in self.outcomes:
            kinds.setdefault(outcome.case.kind, []).append(outcome)
        return {
            kind: _aggregate(f"{self.modality}/{kind}", self.k, outcomes)
            for kind, outcomes in sorted(kinds.items())
        }

    def misses(self) -> list[CaseOutcome]:
        """Cases that went wrong: a wrong top result, or a decline not made."""
        return [
            o
            for o in self.outcomes
            if (o.case.expect_decline and not o.declined)
            or (not o.case.expect_decline and not o.hit_at_1)
        ]

    def to_dict(self) -> dict[str, Any]:
        """The aggregate metrics as plain data, for a committed baseline."""
        return {
            "k": self.k,
            "cases": len(self.outcomes),
            "recall_at_1": _rounded(self.recall_at_1),
            "recall_at_k": _rounded(self.recall_at_k),
            "decline_correctness": _rounded(self.decline_correctness),
        }


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _aggregate(modality: str, k: int, outcomes: Sequence[CaseOutcome]) -> RetrievalReport:
    answerable = [o for o in outcomes if not o.case.expect_decline]
    unanswerable = [o for o in outcomes if o.case.expect_decline]
    backends: dict[str, int] = {}
    for outcome in outcomes:
        if outcome.backend:
            backends[outcome.backend] = backends.get(outcome.backend, 0) + 1
    return RetrievalReport(
        modality=modality,
        k=k,
        recall_at_1=(
            sum(o.hit_at_1 for o in answerable) / len(answerable) if answerable else None
        ),
        recall_at_k=(sum(o.recall for o in answerable) / len(answerable) if answerable else None),
        decline_correctness=(
            sum(o.declined for o in unanswerable) / len(unanswerable) if unanswerable else None
        ),
        mean_latency_ms=(sum(o.latency_ms for o in outcomes) / len(outcomes) if outcomes else 0.0),
        backends=backends,
        outcomes=tuple(outcomes),
    )


async def evaluate_retrieval(
    retriever: Retriever,
    cases: Iterable[RetrievalCase],
    *,
    k: int = 5,
    modality: str = "tools",
) -> RetrievalReport:
    """Run every case through ``retriever`` and score what comes back.

    Args:
        retriever: Coroutine taking ``(query, limit)`` and returning ranked
            ids, optionally with the backend that answered. See
            :func:`tool_retriever`, :func:`skill_retriever` and
            :func:`mcp_server_retriever`.
        cases: The labelled queries.
        k: The ``limit`` to retrieve with, and the ``k`` in recall@k.
        modality: A label for the report.

    Returns:
        A :class:`RetrievalReport`. An exception from the retriever propagates:
        a harness that swallowed one would report a failure as a decline.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    outcomes: list[CaseOutcome] = []
    for case in cases:
        started = time.perf_counter()
        answer = await retriever(case.query, k)
        latency_ms = (time.perf_counter() - started) * 1000
        backend: str | None = None
        if isinstance(answer, tuple) and len(answer) == 2 and isinstance(answer[1], str):
            ids, backend = answer
        else:
            ids = answer
        outcomes.append(
            CaseOutcome(
                case=case, returned=tuple(ids)[:k], latency_ms=latency_ms, backend=backend
            )
        )
    return _aggregate(modality, k, outcomes)


def tool_retriever(
    gantry: AgentGantry, *, score_threshold: float = 0.0, **query_fields: Any
) -> Retriever:
    """A retriever over ``gantry.retrieve()``, returning ``namespace.name`` ids.

    Args:
        gantry: The gantry to query. Its embedder, reranker and selector decide
            which backend is being measured.
        score_threshold: Passed to :class:`~agent_gantry.schema.query.ToolQuery`.
            The library's convenience layers default to ``0.0``, at which the
            semantic path never declines; pass the threshold you deploy with.
        **query_fields: Any other ``ToolQuery`` field, e.g. ``namespaces`` or
            ``enable_reranking``.
    """
    from agent_gantry.schema.query import ConversationContext, ToolQuery

    async def retrieve(query: str, limit: int) -> tuple[list[str], str]:
        result = await gantry.retrieve(
            ToolQuery(
                context=ConversationContext(query=query),
                limit=limit,
                score_threshold=score_threshold,
                **query_fields,
            )
        )
        return [f"{s.tool.namespace}.{s.tool.name}" for s in result.tools], result.backend

    return retrieve


def skill_retriever(
    gantry: AgentGantry, *, score_threshold: float | None = None, **kwargs: Any
) -> Retriever:
    """A retriever over ``gantry.retrieve_skills()``, returning ``namespace.name`` ids."""

    async def retrieve(query: str, limit: int) -> list[str]:
        results = await gantry.retrieve_skills(
            query, limit=limit, score_threshold=score_threshold, **kwargs
        )
        return [f"{r.skill.namespace}.{r.skill.name}" for r in results]

    return retrieve


def mcp_server_retriever(
    gantry: AgentGantry, *, score_threshold: float | None = None, **kwargs: Any
) -> Retriever:
    """A retriever over ``gantry.retrieve_mcp_servers()``, returning ``namespace.name`` ids."""

    async def retrieve(query: str, limit: int) -> list[str]:
        servers = await gantry.retrieve_mcp_servers(
            query, limit=limit, score_threshold=score_threshold, **kwargs
        )
        return [f"{s.namespace}.{s.name}" for s in servers]

    return retrieve


_METRICS = ("recall_at_1", "recall_at_k", "decline_correctness")


def compare_to_baseline(
    current: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    *,
    tolerance: float = 0.0,
) -> list[str]:
    """List the metrics in ``current`` that are worse than ``baseline``.

    Both arguments map a modality to :meth:`RetrievalReport.to_dict` output.

    Args:
        current: The metrics just measured.
        baseline: The committed metrics to hold the line against.
        tolerance: How far a metric may fall before it counts as a regression.

    Returns:
        One human-readable line per regression; empty when nothing got worse.
        A modality missing from ``current`` is a regression, because a
        measurement that silently stopped running protects nothing. A modality
        missing from ``baseline`` is not: it is new.
    """
    regressions: list[str] = []
    for modality, expected in sorted(baseline.items()):
        measured = current.get(modality)
        if measured is None:
            regressions.append(f"{modality}: in the baseline but not measured")
            continue
        for metric in _METRICS:
            before, after = expected.get(metric), measured.get(metric)
            if before is None:
                continue
            if after is None:
                regressions.append(f"{modality}: {metric} was {before}, now not measured")
            elif after < before - tolerance:
                regressions.append(f"{modality}: {metric} fell from {before} to {after}")
    return regressions


def _cell(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:6.3f}"


def format_reports(reports: Iterable[RetrievalReport], *, by_kind: bool = False) -> str:
    """Render reports as a plain-text table, one row per modality.

    Args:
        reports: The reports to show.
        by_kind: Also show a row per case ``kind`` under each modality.
    """
    header = (
        f"{'modality':<28} {'cases':>5} {'k':>3} {'recall@1':>9} {'recall@k':>9} "
        f"{'decline':>8} {'ms/query':>9}  backend"
    )
    lines = [header, "-" * len(header)]

    def row(report: RetrievalReport, label: str) -> str:
        backends = ", ".join(f"{name}={count}" for name, count in sorted(report.backends.items()))
        return (
            f"{label:<28} {len(report.outcomes):>5} {report.k:>3} "
            f"{_cell(report.recall_at_1):>9} {_cell(report.recall_at_k):>9} "
            f"{_cell(report.decline_correctness):>8} {report.mean_latency_ms:>9.2f}  {backends}"
        )

    for report in reports:
        lines.append(row(report, report.modality))
        if by_kind:
            for kind, part in report.by_kind().items():
                lines.append(row(part, f"  {kind}"))
    return "\n".join(lines)
