"""
Cross-encoder reranker adapter using sentence-transformers.

Uses a cross-encoder model to rerank tool candidates by computing
pairwise relevance scores between the query and each tool description.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranker using sentence-transformers cross-encoder models.

    Cross-encoders provide higher accuracy than bi-encoders for reranking
    because they jointly attend to both the query and document, at the
    cost of being slower (not parallelizable for encoding).

    Example:
        >>> reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        >>> reranked = await reranker.rerank("find weather", tools_with_scores, top_k=3)
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        """
        Initialize the cross-encoder reranker.

        Args:
            model: Hugging Face model identifier for the cross-encoder
            device: Device to run on ("cpu", "cuda"). Auto-detected if None.
        """
        self._model_name = model
        self._device = device
        self._model: Any = None
        self._load_lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """Load the model on first use, blocking the caller.

        Prefer :meth:`_aensure_initialized` from async code. This stays sync
        because it is also reached from sync properties.
        """
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._load_model()

    async def _aensure_initialized(self) -> None:
        """Load the model without stalling the event loop.

        Construction downloads weights on first use and takes seconds (minutes
        on a cold cache). Running it inline in a coroutine freezes every other
        task on the loop. The ``encode``/``predict`` calls were already
        offloaded; only construction was not. The guard is a
        ``threading.Lock`` rather than an ``asyncio.Lock`` because the work
        runs in a worker thread and the adapter may outlive one event loop.
        """
        if self._model is not None:
            return
        await asyncio.to_thread(self._ensure_initialized)

    def _load_model(self) -> None:
        """Construct the cross-encoder. Caller holds ``_load_lock``."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        kwargs: dict[str, Any] = {}
        if self._device:
            kwargs["device"] = self._device

        self._model = CrossEncoder(self._model_name, **kwargs)

    async def rerank(
        self,
        query: str,
        tools: list[tuple[ToolDefinition, float]],
        top_k: int,
    ) -> list[tuple[ToolDefinition, float]]:
        """
        Rerank tools using cross-encoder relevance scores.

        Args:
            query: The user's query
            tools: List of (tool, initial_score) tuples from vector search
            top_k: Number of top results to return

        Returns:
            Reranked list of (tool, cross_encoder_score) tuples
        """
        if not tools:
            return []

        await self._aensure_initialized()

        # Build query-document pairs for cross-encoder
        pairs = [
            [query, tool.to_searchable_text()]
            for tool, _ in tools
        ]

        # Score pairs in a thread to avoid blocking the event loop
        scores = await asyncio.to_thread(
            self._model.predict,
            pairs,
        )

        # Combine tools with their new scores
        scored_tools = [
            (tool, float(score))
            for (tool, _), score in zip(tools, scores)
        ]

        # Sort by cross-encoder score (descending) and take top_k
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return scored_tools[:top_k]
