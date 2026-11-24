"""A reranker written for tools also serves skills and MCP servers.

``RerankerAdapter.rerank`` takes ``(ToolDefinition, score)`` pairs, so the
library's best relevance machinery was unavailable to two of its three
catalogues. ``rerank_texts`` puts any text where a tool reranker looks, and
``retrieve_skills`` / ``retrieve_mcp_servers`` take ``rerank=True``.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.simple import SimpleEmbedder
from agent_gantry.adapters.rerankers.base import TextDocument, rerank_texts
from agent_gantry.adapters.rerankers.cross_encoder import CrossEncoderReranker
from agent_gantry.schema.skill import Skill
from agent_gantry.schema.tool import ToolDefinition


class _KeywordReranker:
    """A tool-shaped reranker: scores by how often ``keyword`` appears in a tool's text."""

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        self.seen: list[ToolDefinition] = []

    async def rerank(
        self, query: str, tools: list[tuple[ToolDefinition, float]], top_k: int
    ) -> list[tuple[ToolDefinition, float]]:
        self.seen = [tool for tool, _ in tools]
        scored = [
            (tool, min(1.0, tool.to_searchable_text().lower().count(self.keyword) / 3))
            for tool, _ in tools
        ]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:top_k]


async def test_a_tool_reranker_scores_plain_texts() -> None:
    reranker = _KeywordReranker("refund")
    texts = ["Charge a card.", "Refund a refund of a refund.", "Refund once.", "x"]

    ranked = await rerank_texts(
        reranker, "money back", texts, 2, names=["Pay Desk", "refunds", "refunds", "9lives"]
    )

    assert ranked == [(1, 1.0), (2, pytest.approx(1 / 3))]
    documents = reranker.seen
    assert all(isinstance(d, TextDocument) for d in documents)
    # The text reaches the reranker untouched, whichever field it reads.
    assert [d.to_searchable_text() for d in documents] == texts
    assert documents[1].description == texts[1]
    # Names are made valid and kept apart even when two items share one.
    assert [d.name for d in documents] == ["pay_desk_0", "refunds_1", "refunds_2", "item_9lives_3"]
    assert await rerank_texts(reranker, "q", [], 3) == []


async def test_a_reranker_with_its_own_text_method_is_used_directly() -> None:
    class _Native(_KeywordReranker):
        async def rerank_texts(self, query: str, texts: list[str], top_k: int) -> Any:
            return [(len(texts) - 1, 0.75)]

    reranker = _Native("refund")
    assert await rerank_texts(reranker, "q", ["a", "b", "c"], 2) == [(2, 0.75)]
    assert reranker.seen == []


async def test_the_cross_encoder_ranks_texts_without_a_tool_wrapper() -> None:
    class _Model:
        def predict(self, pairs: list[list[str]]) -> list[float]:
            assert all(query == "money back" for query, _ in pairs)
            return [float(text.count("refund")) for _, text in pairs]

    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker._model = _Model()

    async def ready() -> None:
        return None

    reranker._aensure_initialized = ready  # type: ignore[method-assign]

    ranked = await reranker.rerank_texts("money back", ["pay", "refund refund", "refund"], 2)
    assert [index for index, _ in ranked] == [1, 2]
    assert ranked[0][1] > ranked[1][1]


def _skills() -> list[Skill]:
    def skill(name: str, description: str) -> Skill:
        return Skill(name=name, description=description, content=description)

    return [
        skill("refund_form", "The customer refund form and where it is kept."),
        skill("refund_policy", "Refund rules: when a refund or a part refund is due."),
        skill("refund_letter", "A letter to a customer about a refund decision."),
        skill("card_charge", "How to charge a customer card again."),
    ]


async def test_skills_can_be_reranked() -> None:
    gantry = AgentGantry(embedder=SimpleEmbedder(), reranker=_KeywordReranker("refund"))
    try:
        await gantry.add_skills(_skills())

        plain = await gantry.retrieve_skills("customer refund form", limit=2)
        assert plain[0].skill.name == "refund_form"

        reranked = await gantry.retrieve_skills("customer refund form", limit=2, rerank=True)
        assert [r.skill.name for r in reranked] == ["refund_policy", "refund_form"]
        assert reranked[0].score == 1.0  # the reranker's score, not the cosine
    finally:
        await gantry.close()


async def test_mcp_servers_can_be_reranked() -> None:
    pytest.importorskip("mcp")
    gantry = AgentGantry(embedder=SimpleEmbedder(), reranker=_KeywordReranker("refund"))
    try:
        for name, description in (
            ("forms", "The customer refund form and where it is kept."),
            ("policy", "Refund rules: when a refund or a part refund is due."),
            ("cards", "Charge a customer card again."),
        ):
            gantry.register_mcp_server(name, url=f"https://{name}.example/mcp", description=description)

        plain = await gantry.retrieve_mcp_servers("customer refund form", limit=2)
        assert plain[0].name == "forms"

        reranked = await gantry.retrieve_mcp_servers("customer refund form", limit=2, rerank=True)
        assert [s.name for s in reranked] == ["policy", "forms"]
    finally:
        await gantry.close()


async def test_reranking_is_off_unless_asked_for_and_needs_a_reranker() -> None:
    reranker = _KeywordReranker("refund")
    with_one = AgentGantry(embedder=SimpleEmbedder(), reranker=reranker)
    without = AgentGantry(embedder=SimpleEmbedder())
    try:
        await with_one.add_skills(_skills())
        await without.add_skills(_skills())

        await with_one.retrieve_skills("customer refund form", limit=2)
        assert reranker.seen == []  # configured for tools; skills are opt-in

        with pytest.raises(ValueError, match="needs a reranker"):
            await without.retrieve_skills("customer refund form", rerank=True)
    finally:
        await with_one.close()
        await without.close()
