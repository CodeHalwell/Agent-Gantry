"""
Lightweight, dependency-free embedder for local development and tests.

Uses a deterministic token hashing scheme to produce fixed-length vectors.
This keeps retrieval fast and avoids heavyweight model downloads.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

from agent_gantry.adapters.embedders.base import EmbeddingAdapter


class SimpleEmbedder(EmbeddingAdapter):
    """A tiny, deterministic embedder suitable for unit tests."""

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "simple-hash-embedder"

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text into a fixed-length vector."""
        return self._vectorise(self._tokenise(text))

    async def embed_batch(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Embed multiple texts."""
        return [await self.embed_text(text) for text in texts]

    async def health_check(self) -> bool:
        """Always healthy for the in-memory embedder."""
        return True

    def _tokenise(self, text: str) -> Iterable[str]:
        """Lower-case alphanumeric tokenisation."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _vectorise(self, tokens: Iterable[str]) -> list[float]:
        """Hash tokens into a normalised vector."""
        vec = [0.0] * self._dimension
        for token in tokens:
            idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self._dimension
            vec[idx] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]
