"""
Base reranker adapter protocol.
"""

from __future__ import annotations

import re
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Protocol

from dotfetch.schema.tool import ToolDefinition


class RerankerAdapter(Protocol):
    """
    Rerank tools after vector search for higher precision.

    Implementations: CohereReranker, CrossEncoderReranker, SystemOneReranker.

    A reranker may also define
    ``async def rerank_texts(query, texts, top_k) -> list[tuple[int, float]]``,
    returning ``(index into texts, score)`` best first. It is optional:
    :func:`rerank_texts` uses it when it is there and otherwise goes through
    :meth:`rerank`, so a reranker written for tools serves skills and MCP
    servers as it stands.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        tools: list[tuple[ToolDefinition, float]],
        top_k: int,
    ) -> list[tuple[ToolDefinition, float]]:
        """
        Rerank tools based on the query.

        Args:
            query: The user's query
            tools: List of (tool, score) tuples from initial retrieval
            top_k: Number of top results to return

        Returns:
            Reranked list of (tool, score) tuples
        """
        ...


class TextDocument(ToolDefinition):
    """A piece of text dressed as a tool, so a tool reranker can score it.

    Rerankers were written against ``ToolDefinition``: they read a tool's
    searchable text, or its name, description and tags. A skill or an MCP
    server has those too, in other fields. This puts them where a reranker
    looks, with :meth:`to_searchable_text` returning the text exactly.
    """

    text: str = ""

    def to_searchable_text(self) -> str:
        """The text itself, not a tool's flattening of it."""
        return self.text


def _document(index: int, name: str, text: str) -> TextDocument:
    stem = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")[:100] or "item"
    if not stem[0].isalpha():
        stem = f"item_{stem}"
    return TextDocument(
        # The index keeps two items with one name apart, and maps the result back.
        name=f"{stem}_{index}",
        description=(text if len(text) >= 10 else f"{text} (no further description)")[:2000],
        parameters_schema={"type": "object", "properties": {}},
        text=text,
    )


async def rerank_texts(
    reranker: Any,
    query: str,
    texts: Sequence[str],
    top_k: int,
    *,
    names: Sequence[str] | None = None,
    scores: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """Rerank arbitrary texts with a reranker written for tools.

    Args:
        reranker: Any :class:`RerankerAdapter`.
        query: The user's query.
        texts: What to rank: a skill's or a server's searchable text.
        top_k: How many to return.
        names: A name per text, for rerankers that show the model one.
        scores: The first-stage score per text, which a reranker falls back
            to when it cannot answer. Defaults to the incoming order.

    Returns:
        ``(index into texts, score)``, best first, at most ``top_k``.
    """
    if not texts or top_k <= 0:
        return []
    native = getattr(reranker, "rerank_texts", None)
    if callable(native):
        return [(int(i), float(s)) for i, s in await native(query, list(texts), top_k)]

    documents = [
        _document(index, names[index] if names else "item", text)
        for index, text in enumerate(texts)
    ]
    first_stage = (
        list(scores) if scores is not None else [1.0 - i / len(texts) for i in range(len(texts))]
    )
    by_name = {document.name: index for index, document in enumerate(documents)}
    ranked = await reranker.rerank(query, list(zip(documents, first_stage)), top_k)
    return [(by_name[tool.name], float(score)) for tool, score in ranked if tool.name in by_name]
