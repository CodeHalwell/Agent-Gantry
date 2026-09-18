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
