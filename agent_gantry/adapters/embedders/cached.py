"""
Disk-backed cache wrapper for embedders.

The default :class:`InMemoryVectorStore` discards embeddings between
processes, so any non-trivial registry re-embeds every cold start. With
paid embedders (OpenAI, Azure OpenAI, Cohere) that's real spend repeated
on every restart. ``CachedEmbedder`` wraps any
:class:`~agent_gantry.adapters.embedders.base.EmbeddingAdapter` with a
SQLite cache keyed by ``(embedder_id, sha256(text))``. Lookups that hit
return instantly; misses are forwarded to the underlying embedder and
the result is persisted before being returned.

Storage is a single SQLite file. Default location is
``~/.cache/agent_gantry/embeddings.sqlite`` (overridable). The cache
is keyed by the underlying embedder's :meth:`get_embedder_id` so two
embedders with different models / dimensions never collide.

Usage::

    from agent_gantry.adapters.embedders.openai import OpenAIEmbedder
    from agent_gantry.adapters.embedders.cached import CachedEmbedder

    base = OpenAIEmbedder(config)
    embedder = CachedEmbedder(base)
    gantry = AgentGantry(embedder=embedder)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter

logger = logging.getLogger(__name__)


DEFAULT_CACHE_PATH = Path.home() / ".cache" / "agent_gantry" / "embeddings.sqlite"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedEmbedder:
    """Wrap any :class:`EmbeddingAdapter` with a persistent on-disk cache.

    Args:
        embedder: The backing embedder. All non-cache operations are
            delegated to it.
        cache_path: Path to the SQLite cache file. Defaults to
            ``~/.cache/agent_gantry/embeddings.sqlite``.
    """

    def __init__(
        self,
        embedder: EmbeddingAdapter,
        *,
        cache_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._embedder = embedder
        self._cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        # A connection-per-instance is fine — SQLite handles per-connection
        # serialisation internally. We disable the same-thread check because
        # this is read from anyio-driven async code that may bounce calls
        # between event-loop threads.
        self._conn = sqlite3.connect(
            str(self._cache_path), check_same_thread=False
        )
        self._lock = asyncio.Lock()
        self._init_schema()
        self.hits = 0
        self.misses = 0

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    embedder_id TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    PRIMARY KEY (embedder_id, text_hash)
                )
                """
            )

    @property
    def dimension(self) -> int:
        return self._embedder.dimension

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    def get_embedder_id(self) -> str:
        get_id = getattr(self._embedder, "get_embedder_id", None)
        if callable(get_id):
            return get_id()
        return f"{self.model_name}:{self.dimension}"

    async def embed_text(self, text: str) -> list[float]:
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        embedder_id = self.get_embedder_id()
        # Hash once per text and reuse for both read and write.
        hashes = [_hash(t) for t in texts]

        async with self._lock:
            cached = self._lookup_batch(embedder_id, hashes)

        outputs: list[list[float] | None] = [cached.get(h) for h in hashes]
        miss_indices = [i for i, v in enumerate(outputs) if v is None]
        self.hits += len(texts) - len(miss_indices)
        self.misses += len(miss_indices)

        if miss_indices:
            miss_texts = [texts[i] for i in miss_indices]
            # Delegate the actual embedding work outside the lock so
            # cache reads from other coroutines can proceed in parallel.
            new_embeddings = await self._embedder.embed_batch(
                miss_texts, batch_size=batch_size
            )
            async with self._lock:
                self._store_batch(
                    embedder_id,
                    [(hashes[i], emb) for i, emb in zip(miss_indices, new_embeddings)],
                )
            for i, emb in zip(miss_indices, new_embeddings):
                outputs[i] = emb

        # outputs has no Nones at this point — every slot was either a
        # hit or filled by the embedding call.
        return [o for o in outputs if o is not None]

    def _lookup_batch(
        self, embedder_id: str, hashes: list[str]
    ) -> dict[str, list[float]]:
        if not hashes:
            return {}
        # SQLite parameter limit is 999 by default; chunk to stay safe.
        out: dict[str, list[float]] = {}
        cur = self._conn.cursor()
        for i in range(0, len(hashes), 500):
            chunk = hashes[i : i + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = cur.execute(
                f"SELECT text_hash, embedding FROM embeddings "
                f"WHERE embedder_id = ? AND text_hash IN ({placeholders})",
                (embedder_id, *chunk),
            ).fetchall()
            for h, blob in rows:
                out[h] = json.loads(blob)
        return out

    def _store_batch(
        self,
        embedder_id: str,
        items: list[tuple[str, list[float]]],
    ) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings "
                "(embedder_id, text_hash, embedding) VALUES (?, ?, ?)",
                [
                    (embedder_id, h, json.dumps(emb))
                    for h, emb in items
                ],
            )

    async def health_check(self) -> bool:
        return await self._embedder.health_check()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # pragma: no cover - shutdown best-effort
            logger.debug("CachedEmbedder: error closing SQLite connection.", exc_info=True)


__all__ = ["CachedEmbedder", "DEFAULT_CACHE_PATH"]
