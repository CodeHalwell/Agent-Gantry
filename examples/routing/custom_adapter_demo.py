"""
Plugging a custom embedder into the router.

Implements the ``EmbeddingAdapter`` protocol with a deterministic dummy
embedder and hands it to ``DotFetch(embedder=...)``. The vectors carry no
meaning, so the ranking is arbitrary; the point is that the whole
register -> sync -> retrieve pipeline runs on your own adapter. No API key,
no model download.

Run with::

    python examples/routing/custom_adapter_demo.py
"""

import asyncio
import random

from dotfetch import DotFetch
from dotfetch.adapters.embedders.base import EmbeddingAdapter


# 1. Define a Custom Embedder
class RandomEmbedder(EmbeddingAdapter):
    """
    A dummy embedder that returns random vectors.
    Useful for testing or when no embedding model is available.
    """

    def __init__(self, dimension: int = 10):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "random-embedder"

    async def embed_text(self, text: str) -> list[float]:
        # Deterministic seed based on text length for consistency in demo
        random.seed(len(text))
        return [random.random() for _ in range(self._dimension)]

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        return [await self.embed_text(t) for t in texts]

    async def health_check(self) -> bool:
        return True


async def main():
    # 2. Initialize dotfetch with the custom embedder
    embedder = RandomEmbedder(dimension=4)
    fetcher = DotFetch(embedder=embedder)
    try:

        @fetcher.register(examples=["run tool A", "use the first tool"])
        def tool_a() -> str:
            """Tool A description."""
            return "A"

        @fetcher.register(examples=["run tool B", "use the second tool"])
        def tool_b() -> str:
            """Tool B description."""
            return "B"

        await fetcher.sync()

        print("--- Custom Adapter Demo ---")
        print(f"Using Embedder: {embedder.model_name}")

        # 3. Retrieve tools
        # Since embeddings are random/length-based, semantic relevance is meaningless here,
        # but it proves the pipeline works with custom components.
        tools = await fetcher.retrieve_tools("query")
        print(f"Retrieved {len(tools)} tools using custom embedder.")
        for t in tools:
            print(f" - {t['function']['name']}")
    finally:
        await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
