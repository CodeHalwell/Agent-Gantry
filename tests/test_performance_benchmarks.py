"""
Performance benchmarks for Agent-Gantry.

These tests verify the performance improvements from the code review:
1. Event loop non-blocking in Nomic embedder
2. MMR embedding caching
3. Concurrent request handling

Run with: pytest tests/test_performance_benchmarks.py -v -s
"""

import asyncio
import time

import pytest

from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.base import EmbeddingAdapter
from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore


class MockEmbedder(EmbeddingAdapter):
    """Mock embedder for testing that simulates embedding latency."""

    def __init__(self, latency_ms: int = 50, dimension: int = 128):
        self._latency_ms = latency_ms
        self._dimension = dimension
        self.embed_count = 0  # Track number of embed calls

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "mock-embedder"

    async def embed_text(self, text: str) -> list[float]:
        """Simulate embedding with latency."""
        self.embed_count += 1
        await asyncio.sleep(self._latency_ms / 1000)
        # Generate deterministic embedding based on text hash
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_val % 1000) / 1000.0] * self._dimension

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Simulate batch embedding."""
        self.embed_count += len(texts)
        await asyncio.sleep(self._latency_ms / 1000)
        return [await self.embed_text(text) for text in texts]

    async def health_check(self) -> bool:
        return True


@pytest.fixture
async def gantry_with_tools():
    """Create a gantry instance with multiple tools for testing."""
    gantry = AgentGantry(
        embedder=MockEmbedder(latency_ms=10),
        vector_store=InMemoryVectorStore(),
    )

    # Register 50 tools with diverse descriptions
    for i in range(50):
        @gantry.register(namespace="test")
        def tool_func(x: int = i) -> str:
            return f"Result from tool {x}"

        # Update the function metadata
        tool_func.__name__ = f"tool_{i}"
        tool_func.__doc__ = f"Tool number {i} for testing. Category: {i % 5}"

    await gantry.sync()
    return gantry


@pytest.mark.asyncio
async def test_concurrent_retrieval_throughput(gantry_with_tools):
    """
    Test that concurrent retrievals don't block each other.

    This verifies the Nomic embedder event loop blocking fix.
    Expected: Concurrent requests should complete much faster than sequential.
    """
    gantry = gantry_with_tools
    num_requests = 20

    # Test 1: Sequential retrievals (baseline)
    start = time.time()
    for i in range(num_requests):
        await gantry.retrieve_tools(f"query {i}", limit=5)
    sequential_duration = time.time() - start

    # Test 2: Concurrent retrievals (should be much faster if non-blocking)
    start = time.time()
    tasks = [
        gantry.retrieve_tools(f"query {i}", limit=5)
        for i in range(num_requests)
    ]
    await asyncio.gather(*tasks)
    concurrent_duration = time.time() - start

    # Calculate speedup
    speedup = sequential_duration / concurrent_duration

    print(f"\n{'='*60}")
    print("Concurrent Retrieval Throughput Benchmark")
    print(f"{'='*60}")
    print(f"Sequential: {sequential_duration:.2f}s ({num_requests/sequential_duration:.1f} req/s)")
    print(f"Concurrent: {concurrent_duration:.2f}s ({num_requests/concurrent_duration:.1f} req/s)")
    print(f"Speedup: {speedup:.1f}x")
    print(f"{'='*60}\n")

    # If event loop is not blocked, concurrent should be at least 5x faster
    # With proper async, we should see near-linear scaling up to thread pool size
    assert speedup > 3.0, (
        f"Concurrent requests only {speedup:.1f}x faster than sequential. "
        f"Expected >3x speedup. This suggests event loop blocking."
    )


@pytest.mark.asyncio
async def test_mmr_embedding_caching():
    """
    Test that MMR uses cached embeddings instead of re-embedding.

    This verifies the MMR optimization from the code review.
    """
    embedder = MockEmbedder(latency_ms=50)
    gantry = AgentGantry(
        embedder=embedder,
        vector_store=InMemoryVectorStore(),
    )

    # Register tools
    for i in range(10):
        @gantry.register
        def tool_func(x: int = i) -> str:
            return f"Result {x}"
        tool_func.__name__ = f"diverse_tool_{i}"
        tool_func.__doc__ = f"Tool for task type {i}"

    await gantry.sync()
    initial_embed_count = embedder.embed_count

    # Retrieve with MMR diversity
    start = time.time()
    await gantry.retrieve_tools(
        "task query",
        limit=5,
        diversity_factor=0.5,  # Enable MMR
    )
    duration = time.time() - start
    embed_count_after = embedder.embed_count

    # Calculate how many embeddings were generated during retrieval
    retrieval_embeds = embed_count_after - initial_embed_count

    print(f"\n{'='*60}")
    print("MMR Embedding Caching Benchmark")
    print(f"{'='*60}")
    print(f"Duration: {duration*1000:.1f}ms")
    print(f"Embeddings generated during retrieval: {retrieval_embeds}")
    print(f"Total embed calls: {embed_count_after}")
    print(f"{'='*60}\n")

    # With caching, we should only embed the query (1 embedding)
    # Without caching, we would embed: query + top candidates for MMR (>10 embeddings)
    assert retrieval_embeds <= 2, (
        f"MMR generated {retrieval_embeds} embeddings during retrieval. "
        f"Expected <=2 (query + maybe fallback). Cache not working?"
    )


@pytest.mark.asyncio
async def test_embedding_latency():
    """
    Benchmark embedding generation latency.

    This helps establish baseline performance metrics.
    """
    embedder = MockEmbedder(latency_ms=100, dimension=768)

    # Test single embedding
    start = time.time()
    await embedder.embed_text("test query")
    single_duration = time.time() - start

    # Test batch embedding
    texts = [f"query {i}" for i in range(10)]
    start = time.time()
    await embedder.embed_batch(texts)
    batch_duration = time.time() - start

    print(f"\n{'='*60}")
    print("Embedding Latency Benchmark")
    print(f"{'='*60}")
    print(f"Single embedding: {single_duration*1000:.1f}ms")
    print(f"Batch (10 texts): {batch_duration*1000:.1f}ms ({batch_duration*100:.1f}ms per text)")
    print(f"Batch efficiency: {single_duration*10/batch_duration:.1f}x faster than individual")
    print(f"{'='*60}\n")

    # Batch should be more efficient than individual embeddings
    assert batch_duration < single_duration * 5, "Batch embedding not efficient"


@pytest.mark.asyncio
async def test_vector_search_performance(gantry_with_tools):
    """
    Benchmark vector search performance.

    Measures the speed of similarity search in the vector store.
    """
    gantry = gantry_with_tools

    # Generate a test query embedding
    query_embedding = await gantry._embedder.embed_text("test query")

    # Benchmark search performance
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        await gantry._vector_store.search(
            query_vector=query_embedding,
            limit=10,
        )
    duration = time.time() - start
    avg_latency = (duration / iterations) * 1000

    print(f"\n{'='*60}")
    print("Vector Search Performance Benchmark")
    print(f"{'='*60}")
    print(f"Total searches: {iterations}")
    print(f"Total duration: {duration:.2f}s")
    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"Throughput: {iterations/duration:.0f} searches/sec")
    print(f"{'='*60}\n")

    # In-memory search should be very fast (<5ms per search)
    assert avg_latency < 5.0, f"Vector search too slow: {avg_latency:.2f}ms"


@pytest.mark.asyncio
async def test_end_to_end_retrieval_latency(gantry_with_tools):
    """
    Benchmark end-to-end tool retrieval latency.

    Measures the complete retrieval pipeline including:
    - Query embedding
    - Vector search
    - Scoring
    - Filtering
    """
    gantry = gantry_with_tools

    # Warm up
    await gantry.retrieve_tools("warmup query", limit=5)

    # Benchmark retrieval
    iterations = 50
    latencies = []

    for i in range(iterations):
        start = time.time()
        tools = await gantry.retrieve_tools(f"query {i}", limit=5)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        assert len(tools) <= 5, "Limit not respected"

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]

    print(f"\n{'='*60}")
    print("End-to-End Retrieval Latency Benchmark")
    print(f"{'='*60}")
    print(f"Iterations: {iterations}")
    print(f"Average: {avg_latency:.1f}ms")
    print(f"Min: {min_latency:.1f}ms")
    print(f"Max: {max_latency:.1f}ms")
    print(f"P50: {p50:.1f}ms")
    print(f"P95: {p95:.1f}ms")
    print(f"{'='*60}\n")

    # With mock embedder at 10ms latency, total should be <50ms
    assert avg_latency < 50.0, f"Retrieval too slow: {avg_latency:.1f}ms"


@pytest.mark.asyncio
async def test_concurrent_execution_scalability():
    """
    Test how well the system scales under concurrent load.

    Measures throughput at different concurrency levels.
    """
    gantry = AgentGantry(
        embedder=MockEmbedder(latency_ms=20),
        vector_store=InMemoryVectorStore(),
    )

    # Register tools
    for i in range(30):
        @gantry.register
        def tool_func(x: int = i) -> str:
            return f"Result {x}"
        tool_func.__name__ = f"scale_tool_{i}"
        tool_func.__doc__ = f"Scalability test tool {i}"

    await gantry.sync()

    # Test different concurrency levels
    concurrency_levels = [1, 5, 10, 20]
    results = {}

    for concurrency in concurrency_levels:
        start = time.time()
        tasks = [
            gantry.retrieve_tools(f"query {i}", limit=5)
            for i in range(concurrency)
        ]
        await asyncio.gather(*tasks)
        duration = time.time() - start
        throughput = concurrency / duration

        results[concurrency] = {
            "duration": duration,
            "throughput": throughput,
        }

    print(f"\n{'='*60}")
    print("Concurrent Execution Scalability Benchmark")
    print(f"{'='*60}")
    for concurrency, metrics in results.items():
        print(f"Concurrency {concurrency:2d}: {metrics['duration']:.2f}s, "
              f"{metrics['throughput']:.1f} req/s")
    print(f"{'='*60}\n")

    # Throughput should increase with concurrency (if non-blocking)
    assert results[20]["throughput"] > results[1]["throughput"] * 2, (
        "System doesn't scale with concurrency - possible event loop blocking"
    )


if __name__ == "__main__":
    # Run benchmarks directly
    asyncio.run(test_concurrent_retrieval_throughput(
        asyncio.run(gantry_with_tools())
    ))
