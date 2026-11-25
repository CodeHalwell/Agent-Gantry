"""A reranker written for tools also serves skills and MCP servers.

``RerankerAdapter.rerank`` takes ``(ToolDefinition, score)`` pairs, so the
library's best relevance machinery was unavailable to two of its three
catalogues. ``rerank_texts`` puts any text where a tool reranker looks, and
``retrieve_skills`` / ``retrieve_mcp_servers`` take ``rerank=True``.
"""

from __future__ import annotations

from typing import Any

import pytest

from dotfetch import DotFetch
from dotfetch.adapters.embedders.simple import SimpleEmbedder
from dotfetch.adapters.rerankers.base import TextDocument, rerank_texts
from dotfetch.adapters.rerankers.cross_encoder import CrossEncoderReranker
from dotfetch.schema.skill import Skill
from dotfetch.schema.tool import ToolDefinition


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
    fetcher = DotFetch(embedder=SimpleEmbedder(), reranker=_KeywordReranker("refund"))
    try:
        await fetcher.add_skills(_skills())

        plain = await fetcher.retrieve_skills("customer refund form", limit=2)
        assert plain[0].skill.name == "refund_form"

        reranked = await fetcher.retrieve_skills("customer refund form", limit=2, rerank=True)
        assert [r.skill.name for r in reranked] == ["refund_policy", "refund_form"]
        assert reranked[0].score == 1.0  # the reranker's score, not the cosine
    finally:
        await fetcher.close()


async def test_mcp_servers_can_be_reranked() -> None:
    pytest.importorskip("mcp")
    fetcher = DotFetch(embedder=SimpleEmbedder(), reranker=_KeywordReranker("refund"))
    try:
        for name, description in (
            ("forms", "The customer refund form and where it is kept."),
            ("policy", "Refund rules: when a refund or a part refund is due."),
            ("cards", "Charge a customer card again."),
        ):
            fetcher.register_mcp_server(name, url=f"https://{name}.example/mcp", description=description)

        plain = await fetcher.retrieve_mcp_servers("customer refund form", limit=2)
        assert plain[0].name == "forms"

        reranked = await fetcher.retrieve_mcp_servers("customer refund form", limit=2, rerank=True)
        assert [s.name for s in reranked] == ["policy", "forms"]
    finally:
        await fetcher.close()


async def test_reranking_is_off_unless_asked_for_and_needs_a_reranker() -> None:
    reranker = _KeywordReranker("refund")
    with_one = DotFetch(embedder=SimpleEmbedder(), reranker=reranker)
    without = DotFetch(embedder=SimpleEmbedder())
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


async def test_a_skill_matched_by_its_body_is_reranked_on_that_passage() -> None:
    """Two skills with the same description; only one's body is about zebras.

    A skill's own text leaves its body out, so the reranker was shown two
    identical texts and could not keep the one the search had found.
    """
    reranker = _KeywordReranker("zebra")
    fetcher = DotFetch(embedder=SimpleEmbedder(), reranker=reranker)
    try:
        await fetcher.add_skills(
            [
                Skill(
                    name="keeper_notes_a",
                    description="Notes for the keepers on duty.",
                    content="Check the heating in the reptile house before opening.",
                ),
                Skill(
                    name="keeper_notes_b",
                    description="Notes for the keepers on duty.",
                    content="Zebra feed is in the second barn. A zebra kicks; approach a zebra side on.",
                ),
            ]
        )

        plain = await fetcher.retrieve_skills("zebra feed barn", limit=2)
        assert plain[0].skill.name == "keeper_notes_b"

        reranked = await fetcher.retrieve_skills("zebra feed barn", limit=2, rerank=True)
        assert reranked[0].skill.name == "keeper_notes_b"
        shown = {d.name.rsplit("_", 1)[0]: d.to_searchable_text() for d in reranker.seen}
        assert "second barn" in shown["keeper_notes_b"]
        assert shown["keeper_notes_b"].startswith("keeper_notes_b")  # its own text still leads
    finally:
        await fetcher.close()


async def test_a_tool_matched_by_a_piece_is_reranked_on_that_piece() -> None:
    """The matching sentence is past what ``description`` holds, so show it to the reranker."""
    from dotfetch.schema.query import ConversationContext, ToolQuery

    tail = "Zebra quagga okapi: use this for striped ungulate questions."
    filler = "Results are paged and times are in UTC unless a zone is given. "
    reranker = _KeywordReranker("zebra")
    fetcher = DotFetch(embedder=SimpleEmbedder(), reranker=reranker)
    try:
        await fetcher.add_tool(
            ToolDefinition(
                name="search_archive",
                description="Search the archive.",
                extended_description=filler * 40 + tail,
                parameters_schema={"type": "object", "properties": {}},
            )
        )
        await fetcher.add_tool(
            ToolDefinition(
                name="list_animals",
                description="List the animals in the collection.",
                parameters_schema={"type": "object", "properties": {}},
            )
        )
        await fetcher.sync()

        result = await fetcher.retrieve(
            ToolQuery(
                context=ConversationContext(query="zebra quagga okapi striped ungulate"),
                limit=2,
                score_threshold=0.0,
                enable_reranking=True,
            )
        )

        assert result.tools[0].tool.name == "search_archive"
        shown = {tool.name: tool for tool in reranker.seen}
        assert tail in shown["search_archive"].description  # the piece that matched
        assert "Search the archive." in shown["search_archive"].description  # and its own text
        # What comes back is the tool itself, not the copy the reranker read.
        assert result.tools[0].tool.description == "Search the archive."
        assert shown["list_animals"].description == "List the animals in the collection."
    finally:
        await fetcher.close()


async def test_a_server_matched_by_its_catalogue_is_reranked_on_that_passage() -> None:
    """A vaguely described server found by a tool it lists; the reranker must see the tool."""
    pytest.importorskip("mcp")
    reranker = _KeywordReranker("backup")
    fetcher = DotFetch(embedder=SimpleEmbedder(), reranker=reranker)
    try:
        fetcher.register_mcp_server(
            "skyhost", url="https://skyhost.example/mcp", description="Platform services."
        )
        fetcher.register_mcp_server(
            "mailroom", url="https://mailroom.example/mcp", description="Send and search email."
        )

        async def list_tools() -> list[ToolDefinition]:
            return [
                ToolDefinition(
                    name="restore_backup",
                    description="Restore a database backup by its number.",
                    parameters_schema={"type": "object", "properties": {}},
                )
            ]

        fetcher._mcp_registry.get_client("skyhost").list_tools = list_tools  # type: ignore[method-assign]
        await fetcher.list_server_tools("skyhost")

        servers = await fetcher.retrieve_mcp_servers("restore the backup", limit=1, rerank=True)

        assert [s.name for s in servers] == ["skyhost"]
        shown = {d.name.rsplit("_", 1)[0]: d.to_searchable_text() for d in reranker.seen}
        assert "restore backup" in shown["skyhost"]  # from its catalogue
        assert "Platform services." in shown["skyhost"]  # after its own text
    finally:
        await fetcher.close()
