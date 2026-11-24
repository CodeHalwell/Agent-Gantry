"""The retrieval eval: its arithmetic, its fixture, and the baseline it holds.

Nine open relevance changes are waiting on a way to tell a fix from a
regression. This is that way. The last test here is the gate: it measures the
labelled catalogue with the deterministic hash embedder and fails if any metric
is worse than ``baseline.json``. A change that is meant to move the numbers
updates the baseline in the same commit
(``python scripts/eval_retrieval.py --update-baseline``), where a reviewer can
see what moved.
"""

from __future__ import annotations

import pytest

from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.evaluation import (
    RetrievalCase,
    compare_to_baseline,
    evaluate_retrieval,
    format_reports,
)
from agent_gantry.schema.tool import RESERVED_NAMESPACES
from tests.retrieval import catalogue
from tests.retrieval.harness import flatten, load_baseline, measure


def _retriever(answers: dict[str, list[str]]):
    async def retrieve(query: str, limit: int) -> list[str]:
        return answers.get(query, [])

    return retrieve


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #


async def test_the_metrics_are_what_their_names_say() -> None:
    cases = [
        RetrievalCase("first", {"a"}),
        RetrievalCase("second", {"b", "c"}),
        RetrievalCase("third", {"d"}),
        RetrievalCase("nothing fits", expect_decline=True, kind="decline"),
        RetrievalCase("nothing fits either", expect_decline=True, kind="decline"),
    ]
    report = await evaluate_retrieval(
        _retriever(
            {
                "first": ["a", "x"],  # right at the top
                "second": ["x", "b"],  # one of two, and not first
                "third": ["x", "y"],  # missed
                "nothing fits": [],  # declined, correctly
                "nothing fits either": ["x"],  # should have declined
            }
        ),
        cases,
        k=5,
    )

    assert report.recall_at_1 == pytest.approx(1 / 3)
    assert report.recall_at_k == pytest.approx((1 + 0.5 + 0) / 3)
    assert report.decline_correctness == pytest.approx(0.5)
    assert (report.answerable, report.unanswerable) == (3, 2)
    assert [o.case.query for o in report.misses()] == ["second", "third", "nothing fits either"]


async def test_only_the_top_k_count() -> None:
    report = await evaluate_retrieval(
        _retriever({"q": ["x", "y", "a"]}), [RetrievalCase("q", {"a"})], k=2
    )
    assert report.recall_at_k == 0.0
    assert report.outcomes[0].returned == ("x", "y")


async def test_a_metric_with_no_cases_is_absent_not_zero() -> None:
    """Zero would read as "always wrong" in a baseline comparison."""
    report = await evaluate_retrieval(_retriever({"q": ["a"]}), [RetrievalCase("q", {"a"})])
    assert report.decline_correctness is None
    assert report.to_dict()["decline_correctness"] is None


async def test_the_report_splits_by_kind_and_names_the_backend() -> None:
    async def retrieve(query: str, limit: int) -> tuple[list[str], str]:
        return ["a"], "selector"

    report = await evaluate_retrieval(
        retrieve,
        [RetrievalCase("q1", {"a"}), RetrievalCase("q2", {"b"}, kind="identifier")],
        modality="tools",
    )
    parts = report.by_kind()
    assert parts["paraphrase"].recall_at_1 == 1.0
    assert parts["identifier"].recall_at_1 == 0.0
    assert parts["identifier"].modality == "tools/identifier"
    assert report.backends == {"selector": 2}
    assert "selector=2" in format_reports([report], by_kind=True)


async def test_two_ids_in_a_tuple_are_two_ids_not_ids_and_a_backend() -> None:
    async def retrieve(query: str, limit: int) -> tuple[str, ...]:
        return ("default.add_numbers", "default.subtract_numbers")

    report = await evaluate_retrieval(
        retrieve, [RetrievalCase("add", {"default.add_numbers"})], k=2
    )

    assert report.outcomes[0].returned == ("default.add_numbers", "default.subtract_numbers")
    assert report.recall_at_1 == 1.0
    assert report.backends == {}


async def test_a_retriever_that_raises_is_not_scored_as_a_decline() -> None:
    async def broken(query: str, limit: int) -> list[str]:
        raise RuntimeError("the store is down")

    with pytest.raises(RuntimeError, match="the store is down"):
        await evaluate_retrieval(broken, [RetrievalCase("q", expect_decline=True)])


def test_a_case_must_say_what_it_expects() -> None:
    with pytest.raises(ValueError, match="expect_decline"):
        RetrievalCase("no answer given")
    with pytest.raises(ValueError, match="cannot also expect"):
        RetrievalCase("both", {"a"}, expect_decline=True)


def test_a_regression_is_named_and_an_improvement_is_not() -> None:
    baseline = {
        "tools": {"recall_at_1": 0.5, "recall_at_k": 0.8, "decline_correctness": None},
        "skills": {"recall_at_1": 0.4, "recall_at_k": 0.6, "decline_correctness": 0.5},
    }
    better = {
        "tools": {"recall_at_1": 0.6, "recall_at_k": 0.8, "decline_correctness": None},
        "skills": {"recall_at_1": 0.4, "recall_at_k": 0.6, "decline_correctness": 0.5},
        "mcp_servers": {"recall_at_1": 0.1, "recall_at_k": 0.1, "decline_correctness": 0.0},
    }
    assert compare_to_baseline(better, baseline) == []

    worse = {"tools": {"recall_at_1": 0.5, "recall_at_k": 0.7, "decline_correctness": None}}
    assert compare_to_baseline(worse, baseline) == [
        "skills: in the baseline but not measured",
        "tools: recall_at_k fell from 0.8 to 0.7",
    ]
    assert compare_to_baseline(worse, baseline, tolerance=0.2) == [
        "skills: in the baseline but not measured"
    ]


# --------------------------------------------------------------------------- #
# The fixture
# --------------------------------------------------------------------------- #


def test_the_catalogue_is_large_and_every_label_points_at_something_in_it() -> None:
    pytest.importorskip("mcp")
    tools = [*catalogue.native_tools(), *catalogue.mcp_tools()]
    tool_ids = {f"{t.namespace}.{t.name}" for t in tools}
    # Under ~50 entries the failures this eval exists to show do not appear.
    assert len(tools) == len(tool_ids) >= 300
    assert not any(t.examples for t in tools), "an example a query paraphrases is a leak"

    skill_ids = {f"{s.namespace}.{s.name}" for s in catalogue.skills()}
    server_ids = {f"default.{name}" for name in catalogue.MCP_SERVERS}
    for cases, ids in (
        (catalogue.TOOL_CASES, tool_ids),
        (catalogue.SKILL_CASES, skill_ids),
        (catalogue.MCP_SERVER_CASES, server_ids),
    ):
        for case in cases:
            assert case.expected <= ids, f"{case.query!r} expects an id that does not exist"
        assert any(case.expect_decline for case in cases)


def test_the_tail_cases_really_are_past_the_description_ceiling() -> None:
    """Otherwise the ``tail`` kind measures nothing chunked indexing would fix."""
    pytest.importorskip("mcp")
    by_name = {t.name: t for t in catalogue.mcp_tools()}
    for name, tail in (
        ("search_logs", "payment gateway timeouts"),
        ("get_workspace_hierarchy", "which team owns a folder"),
    ):
        tool = by_name[name]
        assert tail not in tool.description
        assert tail in (tool.extended_description or "")


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


async def test_retrieval_is_no_worse_than_the_committed_baseline() -> None:
    pytest.importorskip("mcp")
    baseline = load_baseline()
    embedder = SimpleEmbedder()
    assert baseline["embedder"] == embedder.model_name

    reports = await measure(k=baseline["k"], embedder=embedder)

    assert compare_to_baseline(flatten(reports), baseline["metrics"]) == []
    for report in reports:
        for outcome in report.outcomes:
            leaked = [i for i in outcome.returned if i.split(".", 1)[0] in RESERVED_NAMESPACES]
            assert not leaked, f"{outcome.case.query!r} was handed index bookkeeping: {leaked}"
