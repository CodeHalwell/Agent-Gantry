"""
SentenceTransformers embedder adapter.

Generic adapter for any sentence-transformers model, supporting
configurable models and dimensions.
"""

from __future__ import annotations

import asyncio
import threading
import warnings
from typing import Any

import numpy as np

from agent_gantry.adapters.embedders.base import EmbeddingAdapter, LazyModelMixin


def _l2_normalize(vector: list[float]) -> list[float]:
    """Rescale ``vector`` to unit length; a zero vector is returned unchanged."""
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return list(vector)
    return (arr / norm).tolist()


class SentenceTransformersEmbedder(LazyModelMixin, EmbeddingAdapter):
    """
    Generic sentence-transformers embedder.

    Wraps any sentence-transformers model for use as an embedding adapter.
    The model is loaded on first use, in a worker thread.

    Example:
        >>> embedder = SentenceTransformersEmbedder(model="all-MiniLM-L6-v2")
        >>> vector = await embedder.embed_text("Hello world")
        >>> assert len(vector) == 384
    """

    #: Extra keyword arguments passed to ``SentenceTransformer(...)``.
    _model_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        dimension: int | None = None,
        device: str | None = None,
    ) -> None:
        """
        Args:
            model: Hugging Face model identifier or path
            dimension: Output dimension (truncates if smaller than the model's
                native dimension). None uses the model's full dimension.
            device: Device to run on (e.g. "cpu", "cuda"). Auto-detected if None.
        """
        self._model_name = model
        self._requested_dimension = dimension
        # Kept as given: _requested_dimension is clamped to the native size on
        # load, and the embedder id must not change when the model loads.
        self._configured_dimension = dimension
        self._device = device
        self._model: Any = None
        self._native_dimension: int | None = None
        self._load_lock = threading.Lock()

    def _load_model(self) -> None:
        """Construct the model. Caller holds ``_load_lock``."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                f"sentence-transformers is required for {type(self).__name__}. "
                "Install it with: pip install sentence-transformers (or uv add sentence-transformers)"
            ) from exc

        kwargs: dict[str, Any] = dict(self._model_kwargs)
        if self._device:
            kwargs["device"] = self._device

        self._model = SentenceTransformer(self._model_name, **kwargs)
        # ``get_sentence_embedding_dimension`` was renamed to
        # ``get_embedding_dimension`` in newer sentence-transformers releases.
        get_dim = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        self._native_dimension = get_dim()

        if self._requested_dimension and self._requested_dimension > self._native_dimension:
            warnings.warn(
                f"Requested dimension {self._requested_dimension} exceeds model's "
                f"native dimension {self._native_dimension}. "
                f"Using native dimension instead.",
                UserWarning,
                stacklevel=2,
            )
            self._requested_dimension = self._native_dimension

    @property
    def dimension(self) -> int:
        """Return the output embedding dimension."""
        if self._requested_dimension:
            return self._requested_dimension
        self._ensure_initialized()
        return self._native_dimension  # type: ignore[return-value]

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    def _truncate(self, embeddings: list[list[float]]) -> list[list[float]]:
        """Slice to the requested dimension and re-normalise.

        A sliced prefix of a unit vector is no longer unit length, and cosine
        scoring downstream (LanceDB's ``1 - d/2`` conversion, dot products)
        is only exact for unit vectors.
        """
        dim = self._requested_dimension
        if not dim:
            return embeddings
        return [_l2_normalize(emb[:dim]) if len(emb) > dim else emb for emb in embeddings]

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return (await self.embed_batch([text]))[0]

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts (``batch_size`` defaults to the model's)."""
        if not texts:
            return []

        await self._aensure_initialized()

        kwargs: dict[str, Any] = {"normalize_embeddings": True}
        if batch_size:
            kwargs["batch_size"] = batch_size

        # Encode in a thread to avoid blocking the event loop
        embeddings = await asyncio.to_thread(self._model.encode, texts, **kwargs)
        return self._truncate(embeddings.tolist())

    async def health_check(self) -> bool:
        """Check if the embedder is operational."""
        try:
            await self._aensure_initialized()
            return self._model is not None
        except Exception:
            return False

    def get_embedder_id(self) -> str:
        """Return a unique identifier for this embedder configuration.

        Must not depend on whether the model has loaded: the sync manager
        compares it with the store's recorded id before the first embed call,
        so it uses the configured dimension (or "auto") rather than the
        native one, which would block on a model load.
        """
        dim = self._configured_dimension or "auto"
        return f"SentenceTransformersEmbedder-{self._model_name}-dim{dim}"
