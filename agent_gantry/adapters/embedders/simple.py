"""
Lightweight, dependency-free embedder for local development and tests.

**For testing only — produces near-uniform similarity scores.** Uses a
deterministic token hashing scheme to produce fixed-length vectors. This keeps
retrieval fast and avoids heavyweight model downloads, but has no semantic
understanding: similarity reflects token overlap, not meaning. For production
use install one of the real embedders (Nomic, SentenceTransformers, OpenAI,
Cohere) and pass a non-zero ``score_threshold`` only with those.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import warnings
from collections.abc import Iterable

from agent_gantry.adapters.embedders.base import EmbeddingAdapter

logger = logging.getLogger(__name__)


class SimpleEmbedder(EmbeddingAdapter):
    """**For testing only — produces near-uniform similarity scores.**

    Hash-based deterministic embedder suitable for unit tests and local
    development without downloading models. Not for production retrieval;
    its scores cluster tightly around 0.0–0.3 even for unrelated text, so
    pairing it with a non-zero ``score_threshold`` typically results in
    "0 tools surfaced" silently.

    For meaningful retrieval, use one of the real embedders:

    - :class:`agent_gantry.adapters.embedders.NomicEmbedder` (local,
      ``pip install agent-gantry[nomic]``)
    - :class:`agent_gantry.adapters.embedders.SentenceTransformersEmbedder`
      (local, ``pip install agent-gantry[sentence-transformers]``)
    - :class:`agent_gantry.adapters.embedders.OpenAIEmbedder` (remote)
    """

    _warned_about_threshold: bool = False

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

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
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
