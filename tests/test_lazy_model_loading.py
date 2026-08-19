"""Model construction must not stall the event loop.

The embedders and the cross-encoder reranker load their model on first use.
``encode``/``predict`` were already offloaded with ``asyncio.to_thread``, but
construction itself ran inline in the coroutine — and that is the expensive
part, downloading weights on a cold cache. A single first call therefore froze
every other task on the loop: concurrent requests, health checks, cancellation.

These tests stub ``sentence_transformers`` with a deliberately slow constructor
so they neither download anything nor require the real package.
"""

from __future__ import annotations

import asyncio
import sys
import time
import types

import numpy as np
import pytest

from agent_gantry.schema.tool import ToolDefinition

#: How long the fake model takes to "load".
_LOAD_SECONDS = 0.4

#: A blocked loop shows a gap of roughly ``_LOAD_SECONDS``; a healthy one shows
#: the heartbeat interval. This threshold sits well between the two.
_MAX_ACCEPTABLE_GAP = _LOAD_SECONDS / 2


@pytest.fixture
def slow_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Install a stub whose constructor sleeps, and count constructions."""
    constructions = {"count": 0}

    class SlowModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            constructions["count"] += 1
            time.sleep(_LOAD_SECONDS)

        def predict(self, pairs: list[list[str]]) -> list[float]:
            return [0.5] * len(pairs)

        def encode(self, texts: list[str], **kwargs: object) -> object:
            # The real model returns a numpy array; the adapters call .tolist().
            return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=np.float32)

        def get_sentence_embedding_dimension(self) -> int:
            return 3

    stub = types.ModuleType("sentence_transformers")
    stub.CrossEncoder = SlowModel  # type: ignore[attr-defined]
    stub.SentenceTransformer = SlowModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)
    return constructions


@pytest.fixture
def tool() -> ToolDefinition:
    return ToolDefinition(
        name="sample",
        description="A sample tool used for reranking tests.",
        parameters_schema={"type": "object", "properties": {}},
    )


async def _largest_heartbeat_gap(coro: object) -> float:
    """Run ``coro`` while ticking a heartbeat; return the largest tick gap."""
    ticks: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        while not stop.is_set():
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.01)

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.03)  # let a few ticks land before the load
    await coro  # type: ignore[misc]
    stop.set()
    await hb

    assert len(ticks) > 2, "heartbeat never ran"
    return max(b - a for a, b in zip(ticks, ticks[1:]))


async def test_reranker_load_does_not_block_the_loop(
    slow_sentence_transformers: dict[str, int], tool: ToolDefinition
) -> None:
    from agent_gantry.adapters.rerankers.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    gap = await _largest_heartbeat_gap(reranker.rerank("q", [(tool, 1.0)], top_k=1))

    assert gap < _MAX_ACCEPTABLE_GAP, (
        f"event loop stalled {gap * 1000:.0f}ms during model construction "
        f"(model load is {_LOAD_SECONDS * 1000:.0f}ms)"
    )


async def test_embedder_load_does_not_block_the_loop(
    slow_sentence_transformers: dict[str, int],
) -> None:
    from agent_gantry.adapters.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )

    embedder = SentenceTransformersEmbedder()
    gap = await _largest_heartbeat_gap(embedder.embed_text("hello"))

    assert gap < _MAX_ACCEPTABLE_GAP, (
        f"event loop stalled {gap * 1000:.0f}ms during model construction"
    )


async def test_concurrent_first_calls_load_the_model_once(
    slow_sentence_transformers: dict[str, int], tool: ToolDefinition
) -> None:
    """Offloading must not turn one load into a thundering herd."""
    from agent_gantry.adapters.rerankers.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    await asyncio.gather(
        *(reranker.rerank("q", [(tool, 1.0)], top_k=1) for _ in range(5))
    )

    assert slow_sentence_transformers["count"] == 1


async def test_model_is_reused_across_calls(
    slow_sentence_transformers: dict[str, int],
) -> None:
    from agent_gantry.adapters.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )

    embedder = SentenceTransformersEmbedder()
    await embedder.embed_text("one")
    await embedder.embed_text("two")

    assert slow_sentence_transformers["count"] == 1


def test_sync_property_still_loads(slow_sentence_transformers: dict[str, int]) -> None:
    """``dimension`` is a sync property and must keep working."""
    from agent_gantry.adapters.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )

    embedder = SentenceTransformersEmbedder()
    assert embedder.dimension == 3
    assert slow_sentence_transformers["count"] == 1
