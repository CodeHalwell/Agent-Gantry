"""
SentenceTransformers embedder adapter.

Generic adapter for any sentence-transformers model, supporting
configurable models and dimensions.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any

from agent_gantry.adapters.embedders.base import EmbeddingAdapter

logger = logging.getLogger(__name__)


class SentenceTransformersEmbedder(EmbeddingAdapter):
    """
    Generic sentence-transformers embedder.

    Wraps any sentence-transformers model for use as an embedding adapter.
    Unlike the NomicEmbedder, this supports arbitrary models without
    task-specific prefixing.

    Example:
        >>> embedder = SentenceTransformersEmbedder(model="all-MiniLM-L6-v2")
        >>> vector = await embedder.embed_text("Hello world")
        >>> assert len(vector) == 384
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        dimension: int | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialize the SentenceTransformers embedder.

        Args:
            model: Hugging Face model identifier or path
            dimension: Output dimension (truncates if smaller than model's native dim).
                       If None, uses the model's full dimension.
            device: Device to run on (e.g., "cpu", "cuda"). Auto-detected if None.
        """
        self._model_name = model
        self._requested_dimension = dimension
        self._device = device
        self._model: Any = None
        self._native_dimension: int | None = None

    def _ensure_initialized(self) -> None:
        """Lazily load the model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformersEmbedder. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self._device:
            kwargs["device"] = self._device

        self._model = SentenceTransformer(self._model_name, **kwargs)
        self._native_dimension = self._model.get_sentence_embedding_dimension()

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

    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as a list of floats
        """
        self._ensure_initialized()

        # Run in thread pool to avoid blocking the event loop
        embedding = await asyncio.to_thread(
            self._model.encode,
            text,
            normalize_embeddings=True,
        )

        result = embedding.tolist()

        # Truncate if a smaller dimension was requested
        if self._requested_dimension and len(result) > self._requested_dimension:
            result = result[:self._requested_dimension]

        return result

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding (default: model's default)

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self._ensure_initialized()

        kwargs: dict[str, Any] = {"normalize_embeddings": True}
        if batch_size:
            kwargs["batch_size"] = batch_size

        embeddings = await asyncio.to_thread(
            self._model.encode,
            texts,
            **kwargs,
        )

        results = embeddings.tolist()

        # Truncate if a smaller dimension was requested
        if self._requested_dimension:
            results = [emb[:self._requested_dimension] for emb in results]

        return results

    async def health_check(self) -> bool:
        """Check if the embedder is operational."""
        try:
            self._ensure_initialized()
            return self._model is not None
        except Exception:
            return False

    def get_embedder_id(self) -> str:
        """Return a unique identifier for this embedder configuration."""
        dim = self._requested_dimension or self._native_dimension or "auto"
        return f"SentenceTransformersEmbedder-{self._model_name}-dim{dim}"
