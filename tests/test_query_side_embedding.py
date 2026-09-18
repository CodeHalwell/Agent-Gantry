"""Queries go through ``embed_query``, not ``embed_text``.

Several modern embedders are asymmetric: Nomic's v1.5 wants ``search_document:``
on a stored tool and ``search_query:`` on the prompt, and applying the document
instruction to a query is off-label use of the model. ``NomicEmbedder`` has
carried a correct ``embed_query`` since it was written — the retrieval path just
never called it, so the prefix it exists to apply was never applied.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_gantry import AgentGantry, ConversationContext, Skill, ToolQuery
from agent_gantry.adapters.embedders.base import EmbeddingAdapter
from agent_gantry.schema.tool import ToolDefinition


class _RecordingEmbedder(EmbeddingAdapter):
    """Records which side of the asymmetry each call came through."""

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.queries: list[str] = []

    @property
    def dimension(self) -> int:
        return 4

    @property
    def model_name(self) -> str:
        return "recording"

    async def embed_text(self, text: str) -> list[float]:
        self.documents.append(text)
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_text(t) for t in texts]

    async def embed_batch(self, texts: list[str], batch_size: int | None = None) -> Any:
        return await self.embed_texts(texts)

    async def health_check(self) -> bool:
        return True


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"The {name} tool does something useful for the caller.",
        parameters_schema={"type": "object", "properties": {}},
    )


async def test_tool_retrieval_embeds_the_query_as_a_query() -> None:
    embedder = _RecordingEmbedder()
    gantry = AgentGantry(embedder=embedder)
    try:
        await gantry.add_tool(_tool("alpha"), lambda: None)
        await gantry.retrieve(
            ToolQuery(context=ConversationContext(query="find alpha"), score_threshold=0.0)
        )
        assert embedder.queries == ["find alpha"]
        assert "find alpha" not in embedder.documents
    finally:
        await gantry.close()


async def test_skill_retrieval_embeds_the_query_as_a_query() -> None:
    embedder = _RecordingEmbedder()
    gantry = AgentGantry(embedder=embedder)
    try:
        await gantry.add_skill(
            Skill(name="deploying", description="How to deploy safely.", content="Body.")
        )
        await gantry.retrieve_skills("how do I ship this", limit=1)
        assert embedder.queries == ["how do I ship this"]
    finally:
        await gantry.close()


async def test_mcp_server_retrieval_embeds_the_query_as_a_query() -> None:
    embedder = _RecordingEmbedder()
    gantry = AgentGantry(embedder=embedder)
    if gantry._mcp_registry is None:  # pragma: no cover - mcp extra absent
        await gantry.close()
        pytest.skip("MCP support is not installed")
    try:
        gantry.register_mcp_server(
            "files", ["echo", "files"], description="Reads and writes files on disk."
        )
        await gantry.retrieve_mcp_servers("open a file", limit=1)
        assert "open a file" in embedder.queries
    finally:
        await gantry.close()


async def test_a_symmetric_embedder_needs_no_embed_query_of_its_own() -> None:
    """The protocol default forwards, so third-party adapters keep working."""

    class _Symmetric(_RecordingEmbedder):
        embed_query = EmbeddingAdapter.embed_query  # type: ignore[assignment]

    embedder = _Symmetric()
    gantry = AgentGantry(embedder=embedder)
    try:
        await gantry.add_tool(_tool("alpha"), lambda: None)
        await gantry.retrieve(
            ToolQuery(context=ConversationContext(query="find alpha"), score_threshold=0.0)
        )
        assert "find alpha" in embedder.documents, "the default must fall through"
    finally:
        await gantry.close()


def test_nomic_applies_the_query_prefix_rather_than_the_document_one() -> None:
    """Pinned against the adapter's own prefix table, without loading the model."""
    from agent_gantry.adapters.embedders.nomic import NomicEmbedder

    assert NomicEmbedder.TASK_PREFIXES["search_query"] == "search_query: "
    assert NomicEmbedder.TASK_PREFIXES["search_document"] == "search_document: "
    source = NomicEmbedder.embed_query.__doc__ or ""
    assert "search_query" in source


async def test_a_duck_typed_adapter_without_embed_query_still_works() -> None:
    """A Protocol's default method does not reach a class that merely satisfies it.

    ``EmbeddingAdapter`` is a Protocol, so an adapter written against it
    structurally — which is the point of a protocol, and what a third-party
    integration is most likely to do — has no ``embed_query`` at all. Assuming
    the method were there would have turned every retrieval into an
    ``AttributeError`` for those adapters the day this landed.
    """

    class _DuckTyped:
        """Satisfies the protocol structurally, inherits nothing from it."""

        def __init__(self) -> None:
            self.seen: list[str] = []

        @property
        def dimension(self) -> int:
            return 4

        @property
        def model_name(self) -> str:
            return "duck"

        def get_embedder_id(self) -> str:
            return "duck:4"

        async def embed_text(self, text: str) -> list[float]:
            self.seen.append(text)
            return [1.0, 0.0, 0.0, 0.0]

        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [await self.embed_text(t) for t in texts]

        async def embed_batch(self, texts: list[str], batch_size: int | None = None) -> Any:
            return await self.embed_texts(texts)

        async def health_check(self) -> bool:
            return True

    embedder = _DuckTyped()
    assert not hasattr(embedder, "embed_query"), "the premise of this test"

    gantry = AgentGantry(embedder=embedder)
    try:
        await gantry.add_tool(_tool("alpha"), lambda: None)
        await gantry.retrieve(
            ToolQuery(context=ConversationContext(query="find alpha"), score_threshold=0.0)
        )
        assert "find alpha" in embedder.seen
    finally:
        await gantry.close()


async def test_a_non_awaitable_embed_query_is_not_called() -> None:
    """Test doubles routinely auto-create attributes; a MagicMock ``embed_query``
    is not a coroutine function and awaiting it raises ``TypeError``. Falling
    back keeps every such stand-in working without it having to grow a method."""
    from unittest.mock import AsyncMock, MagicMock

    from agent_gantry.adapters.embedders.base import embed_query

    embedder = MagicMock()
    embedder.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])

    assert await embed_query(embedder, "q") == [0.1, 0.2, 0.3, 0.4]
    embedder.embed_text.assert_awaited_once_with("q")


async def test_a_cache_wrapper_preserves_the_query_side() -> None:
    """``CachedEmbedder`` advertises that it wraps any embedder safely.

    Without its own ``embed_query`` the retrieval helper falls back to
    ``embed_text``, which reaches the *document* side of whatever is wrapped —
    so ``CachedEmbedder(NomicEmbedder(...))`` would go on embedding prompts
    with ``search_document:``, silently undoing the asymmetry for anyone who
    added caching.
    """
    import tempfile
    from pathlib import Path as _Path

    from agent_gantry.adapters.embedders.cached import CachedEmbedder

    inner = _RecordingEmbedder()
    with tempfile.TemporaryDirectory() as tmp:
        cached = CachedEmbedder(inner, cache_path=_Path(tmp) / "c.sqlite")

        vector = await cached.embed_query("find alpha")

        assert vector == [1.0, 0.0, 0.0, 0.0]
        assert inner.queries == ["find alpha"], "the wrapper must use the query side"
        assert "find alpha" not in inner.documents

        # Second call is served from the cache, not the embedder.
        await cached.embed_query("find alpha")
        assert inner.queries == ["find alpha"], "a cache hit must not re-embed"


async def test_the_query_cache_does_not_collide_with_the_document_cache() -> None:
    """An asymmetric model returns different vectors for the two sides.

    Sharing one key would serve a document vector for a query, and poison the
    other direction on the way back.
    """
    import tempfile
    from pathlib import Path as _Path

    from agent_gantry.adapters.embedders.cached import CachedEmbedder

    class _Asymmetric(_RecordingEmbedder):
        async def embed_text(self, text: str) -> list[float]:
            self.documents.append(text)
            return [1.0, 0.0, 0.0, 0.0]

        async def embed_query(self, query: str) -> list[float]:
            self.queries.append(query)
            return [0.0, 1.0, 0.0, 0.0]

    inner = _Asymmetric()
    with tempfile.TemporaryDirectory() as tmp:
        cached = CachedEmbedder(inner, cache_path=_Path(tmp) / "c.sqlite")

        document_vector = await cached.embed_text("same text")
        query_vector = await cached.embed_query("same text")

        assert document_vector == [1.0, 0.0, 0.0, 0.0]
        assert query_vector == [0.0, 1.0, 0.0, 0.0], "the query side was served a document vector"


async def test_skill_selection_pages_past_the_stores_default_limit() -> None:
    """``list_all_skills()`` defaults to ``limit=1000`` on both stores.

    One call reads the first page, and the selector would then answer from a
    truncated catalogue and report success — keeping the semantic fallback from
    ever running, so every skill past the first page stayed unreachable.
    """
    from typing import Any as _Any

    from agent_gantry import AgentGantry
    from agent_gantry.schema.selection import SelectionResult

    seen: list[int] = []

    class _CountingSelector:
        async def select(
            self, query: str, candidates: _Any, limit: int, *, kind: str = "tool"
        ) -> SelectionResult:
            seen.append(len(candidates))
            return SelectionResult(fallback=True, reason="counted")

    gantry = AgentGantry(selector=_CountingSelector())
    try:
        from agent_gantry.schema.skill import Skill

        def _skills(start: int, count: int) -> list[Skill]:
            return [
                Skill(
                    name=f"skill_{start + i}",
                    description=f"Skill number {start + i} does a thing.",
                    content="Body.",
                )
                for i in range(count)
            ]

        store = gantry._vector_store
        pages = [_skills(0, 1000), _skills(1000, 1000), _skills(2000, 7)]
        calls: list[tuple[int, int]] = []

        async def _paged(limit: int = 1000, offset: int = 0, **_: _Any) -> list[_Any]:
            calls.append((limit, offset))
            index = offset // 1000
            return pages[index] if index < len(pages) else []

        store.list_all_skills = _paged  # type: ignore[method-assign]
        await gantry._select_skills("q", 3, None, None, None)

        assert len(calls) == 3, f"must page until a short page arrives, got {calls}"
        assert seen == [2007], "the whole catalogue must reach the selector"
    finally:
        await gantry.close()
