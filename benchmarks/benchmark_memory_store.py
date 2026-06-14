"""Micro-benchmark for InMemoryVectorStore.search throughput.

Measures cosine-search latency across registry sizes. The store now computes
scores with a single cached, NumPy-vectorized matmul (``query · M.T``) instead
of a per-tool pure-Python dot-product loop. This script reports per-search
latency so the speedup is visible as the registry grows.

Run::

    python benchmarks/benchmark_memory_store.py
"""

from __future__ import annotations

import asyncio
import random
import time

from agent_gantry.adapters.vector_stores.memory import InMemoryVectorStore
from agent_gantry.schema.tool import ToolDefinition

DIM = 384  # all-MiniLM-L6-v2 dimensionality


def _rand_vec(d: int) -> list[float]:
    return [random.gauss(0.0, 1.0) for _ in range(d)]


async def bench_size(n: int, searches: int = 200) -> dict[str, float]:
    store = InMemoryVectorStore(dimension=DIM)
    tools = [
        ToolDefinition(
            name=f"tool_{i}",
            namespace="bench",
            description=f"benchmark tool number {i}",
            parameters_schema={"type": "object", "properties": {}},
        )
        for i in range(n)
    ]
    embeddings = [_rand_vec(DIM) for _ in range(n)]
    await store.add_tools(tools, embeddings)

    queries = [_rand_vec(DIM) for _ in range(searches)]
    # Warm the cached matrix (first search pays the one-time build).
    await store.search(queries[0], limit=3)

    t0 = time.perf_counter()
    for q in queries:
        await store.search(q, limit=3)
    elapsed = time.perf_counter() - t0
    per_search_ms = (elapsed / searches) * 1000.0
    return {"n": n, "searches": searches, "per_search_ms": round(per_search_ms, 4)}


async def main() -> None:
    print(f"InMemoryVectorStore.search latency (dim={DIM})\n")
    print(f"{'tools':>8} {'searches':>9} {'per_search_ms':>14}")
    for n in (10, 50, 100, 500, 1000, 5000):
        r = await bench_size(n)
        print(f"{r['n']:>8} {r['searches']:>9} {r['per_search_ms']:>14}")


if __name__ == "__main__":
    asyncio.run(main())
