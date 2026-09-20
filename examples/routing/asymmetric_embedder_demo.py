"""
Writing an asymmetric embedder: ``embed_query`` vs ``embed_text``.

Several modern embedding models are *asymmetric*. They are trained with one
instruction for the thing being stored and a different one for the thing being
searched for, and using the wrong one is off-label use of the model. Nomic's
v1.5 wants ``search_document:`` on a stored tool and ``search_query:`` on the
prompt; E5 wants ``passage:`` and ``query:``; BGE wants a bare passage and an
instruction-prefixed query.

Gantry uses one embedder instance for both sides, so the adapter decides:

* ``embed_text(text)``  — the document side. Used when tools are synced.
* ``embed_query(query)`` — the query side. Used when a prompt is routed.

``embed_query`` is optional. The protocol provides a default that forwards to
``embed_text``, and the retrieval path falls back to ``embed_text`` for any
adapter that does not define one — including an adapter that satisfies the
protocol structurally without subclassing it, which is the usual way to write
one. So a symmetric embedder needs no extra code at all.

This demo needs no API key and no model download.

Run with::

    python examples/routing/asymmetric_embedder_demo.py
"""

from __future__ import annotations

import asyncio
import hashlib

from agent_gantry import AgentGantry, ConversationContext, ToolQuery
from agent_gantry.adapters.embedders.base import EmbeddingAdapter
from agent_gantry.schema.tool import ToolDefinition


def _fake_vector(text: str, dimension: int) -> list[float]:
    """A deterministic stand-in for a real model, so the demo needs nothing."""
    digest = hashlib.sha256(text.encode()).digest()
    return [digest[i % len(digest)] / 255.0 for i in range(dimension)]


class PrefixedEmbedder(EmbeddingAdapter):
    """An asymmetric adapter, in the shape a real one takes.

    The only thing that makes it asymmetric is that ``embed_query`` applies a
    different instruction from ``embed_text``. Swap ``_fake_vector`` for a real
    ``model.encode`` call and this is a working adapter.
    """

    DOCUMENT_PREFIX = "search_document: "
    QUERY_PREFIX = "search_query: "

    def __init__(self, dimension: int = 16) -> None:
        self._dimension = dimension
        self.calls: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "demo-asymmetric"

    def get_embedder_id(self) -> str:
        # Include anything that changes the vectors, so stored embeddings are
        # invalidated when you change it. The prefixes qualify.
        return f"{self.model_name}:{self._dimension}:asymmetric"

    async def embed_text(self, text: str) -> list[float]:
        """The document side — what a tool is stored as."""
        self.calls.append(f"document <- {text[:40]}")
        return _fake_vector(self.DOCUMENT_PREFIX + text, self._dimension)

    async def embed_query(self, query: str) -> list[float]:
        """The query side — what a prompt is searched as."""
        self.calls.append(f"query    <- {query[:40]}")
        return _fake_vector(self.QUERY_PREFIX + query, self._dimension)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_text(text) for text in texts]

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        return await self.embed_texts(texts)

    async def health_check(self) -> bool:
        return True


class SymmetricEmbedder(PrefixedEmbedder):
    """A symmetric adapter: no ``embed_query``, and none needed.

    Deleting the override is the whole change. Retrieval falls back to
    ``embed_text``, so nothing breaks and nothing has to be configured.
    """

    embed_query = None  # type: ignore[assignment]


async def show(label: str, embedder: PrefixedEmbedder) -> None:
    gantry = AgentGantry(embedder=embedder)
    try:
        await gantry.add_tool(
            ToolDefinition(
                name="send_email",
                description="Send an email message to a named recipient.",
                examples=["email Priya about the report", "send a note to the team"],
                parameters_schema={"type": "object", "properties": {}},
            ),
            lambda: None,
        )
        await gantry.retrieve(
            ToolQuery(
                context=ConversationContext(query="email Priya about the report"),
                score_threshold=0.0,
            )
        )
        print(f"\n{label}")
        for call in embedder.calls:
            print(f"    {call}")
    finally:
        await gantry.close()


async def main() -> None:
    print("Which side of the model each step goes through:")
    await show("asymmetric — the prompt takes the query instruction", PrefixedEmbedder())
    await show("symmetric — everything takes one instruction", SymmetricEmbedder())
    print(
        "\nNote the asymmetric run routes the prompt through 'query' while the"
        "\ntool went through 'document'. Before this existed, both went through"
        "\n'document', so an asymmetric model was being asked the wrong question."
    )


if __name__ == "__main__":
    asyncio.run(main())
