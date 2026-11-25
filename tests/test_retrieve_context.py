"""One call for tools, skills and MCP servers, merged by rank."""

from __future__ import annotations

from typing import Any

import pytest

from dotfetch import ContextResult, DotFetch
from dotfetch.adapters.embedders.simple import SimpleEmbedder
from dotfetch.schema.selection import SelectionResult
from dotfetch.schema.skill import Skill
from dotfetch.schema.tool import ToolDefinition


def _tool(name: str, description: str, **fields: Any) -> ToolDefinition:
    fields.setdefault("parameters_schema", {"type": "object", "properties": {}})
    return ToolDefinition(name=name, description=description, **fields)


async def _fetcher(*, with_servers: bool = True, **kwargs: Any) -> DotFetch:
    fetcher = DotFetch(embedder=SimpleEmbedder(), **kwargs)
    for name, description in (
        ("issue_refund", "Issue a refund to a customer."),
        ("send_email", "Send an email to a customer."),
        ("list_orders", "List the orders a customer has placed."),
    ):
        await fetcher.add_tool(_tool(name, description))
    await fetcher.add_skills(
        [
            Skill(
                name="refund_policy",
                description="Rules for when a customer refund is due.",
                content="Refunds over 500 pounds need a manager.",
            ),
            Skill(
                name="tone_of_voice",
                description="How to write to a customer politely.",
                content="Be brief and kind.",
            ),
        ]
    )
    if with_servers:
        pytest.importorskip("mcp")
        fetcher.register_mcp_server(
            "paydesk", url="https://paydesk.example/mcp", description="Refund a customer card."
        )
        fetcher.register_mcp_server(
            "mapkit", url="https://mapkit.example/mcp", description="Directions and distances."
        )
    return fetcher


async def test_one_call_returns_all_three_kinds_merged_by_rank() -> None:
    fetcher = await _fetcher()
    try:
        result = await fetcher.retrieve_context("refund a customer", tools=2, skills=2, servers=1)

        assert isinstance(result, ContextResult)
        # Every kind's best comes before any kind's second.
        assert [(e.kind, e.rank) for e in result.entries] == [
            ("tool", 1),
            ("skill", 1),
            ("mcp_server", 1),
            ("tool", 2),
            ("skill", 2),
        ]
        assert result.entries[0].id == "default.issue_refund"
        assert [t.name for t in result.tools][0] == "issue_refund"
        assert [s.name for s in result.skills][0] == "refund_policy"
        assert [s.name for s in result.mcp_servers] == ["paydesk"]
        assert result.skipped == {}
        assert result.total_time_ms > 0
    finally:
        await fetcher.close()


async def test_each_entry_keeps_its_own_score_and_the_tool_search_its_provenance() -> None:
    fetcher = await _fetcher()
    try:
        result = await fetcher.retrieve_context("refund a customer", tools=2, skills=1, servers=1)
        by_kind = {entry.kind: entry for entry in result.entries if entry.rank == 1}

        assert 0 < by_kind["tool"].score <= 1
        assert 0 < by_kind["skill"].score <= 1
        assert by_kind["mcp_server"].score is None  # the server path reports none
        assert result.tool_retrieval.backend == "semantic"
        assert result.tool_retrieval.embedder == "simple-hash-embedder"
    finally:
        await fetcher.close()


async def test_a_kind_asked_for_as_zero_is_not_searched() -> None:
    fetcher = await _fetcher()
    try:
        result = await fetcher.retrieve_context("refund a customer", tools=0, skills=1, servers=0)
        assert [(e.kind, e.rank) for e in result.entries] == [("skill", 1)]
        assert result.tool_retrieval is None
    finally:
        await fetcher.close()


async def test_a_kind_that_cannot_be_searched_is_named_and_the_rest_still_answer() -> None:
    fetcher = await _fetcher(with_servers=False)
    try:
        fetcher._mcp_manager = None  # as when the ``mcp`` package is not installed
        result = await fetcher.retrieve_context("refund a customer", tools=1, skills=1, servers=1)

        assert {e.kind for e in result.entries} == {"tool", "skill"}
        assert result.skipped == {"mcp_server": "MCP support is not installed"}
    finally:
        await fetcher.close()


async def test_no_registered_servers_is_an_empty_answer_not_a_skip() -> None:
    fetcher = await _fetcher(with_servers=False)
    try:
        result = await fetcher.retrieve_context("refund a customer", servers=2)
        assert result.mcp_servers == []
        assert "mcp_server" not in result.skipped
    finally:
        await fetcher.close()


async def test_a_selector_answer_is_reported_through_the_tool_search() -> None:
    class _Selector:
        async def select(self, query: str, candidates: Any, limit: int, *, kind: str = "tool"):
            chosen = [c.id for c in candidates][:limit]
            return SelectionResult(selected=chosen, scores={cid: 0.8 for cid in chosen})

    fetcher = await _fetcher(selector=_Selector())
    try:
        result = await fetcher.retrieve_context("refund a customer", tools=2, skills=1, servers=1)
        assert result.tool_retrieval.backend == "selector"
        assert {e.kind for e in result.entries} == {"tool", "skill", "mcp_server"}
    finally:
        await fetcher.close()
