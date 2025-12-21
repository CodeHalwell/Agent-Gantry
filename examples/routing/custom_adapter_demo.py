import asyncio
import random

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.base import EmbeddingAdapter


# 1. Define a Custom Embedder
class RandomEmbedder(EmbeddingAdapter):
    """
    A dummy embedder that returns random vectors.
    Useful for testing or when no LLM is available.
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

    async def embed_batch(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        return [await self.embed_text(t) for t in texts]

    async def health_check(self) -> bool:
        return True

async def main():
    # 2. Initialize Gantry with the custom embedder
    embedder = RandomEmbedder(dimension=4)
    gantry = AgentGantry(embedder=embedder)

    @gantry.register
    def tool_a() -> str:
        """Tool A description."""
        return "A"

    @gantry.register
    def tool_b() -> str:
        """Tool B description."""
        return "B"

    await gantry.sync()

    print("--- Custom Adapter Demo ---")
    print(f"Using Embedder: {embedder.model_name}")

    # 3. Retrieve tools
    # Since embeddings are random/length-based, semantic relevance is meaningless here,
    # but it proves the pipeline works with custom components.
    tools = await gantry.retrieve_tools("query")
    print(f"Retrieved {len(tools)} tools using custom embedder.")
    for t in tools:
        print(f" - {t['function']['name']}")

if __name__ == "__main__":
    asyncio.run(main())
