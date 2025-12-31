"""
Production Cohere reranker implementation.

Uses the Cohere Rerank API to improve tool retrieval precision.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from agent_gantry.adapters.rerankers.base import RerankerAdapter

if TYPE_CHECKING:
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)


class CohereReranker(RerankerAdapter):
    """
    Production Cohere reranker using the official Cohere Python client.

    Supports models: rerank-english-v3.0, rerank-multilingual-v3.0, rerank-english-v2.0
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_chunks_per_doc: int | None = None,
    ) -> None:
        """
        Initialize the Cohere reranker.

        Args:
            api_key: Optional API key (defaults to COHERE_API_KEY env var)
            model: Model name (default: rerank-english-v3.0)
            max_chunks_per_doc: Max chunks per document for reranking

        Raises:
            ImportError: If cohere package is not installed
        """
        try:
            from cohere import AsyncClient
        except ImportError as exc:
            raise ImportError(
                "Cohere package is not installed. Install it with:\n"
                "  pip install agent-gantry[cohere]"
            ) from exc

        api_key = api_key or os.getenv("COHERE_API_KEY")
        self._model = model or "rerank-english-v3.0"
        self._max_chunks_per_doc = max_chunks_per_doc
        self._client = AsyncClient(api_key=api_key) if api_key else None

        logger.info(f"Initialized CohereReranker with model={self._model}")

    def _tool_to_document(self, tool: ToolDefinition) -> str:
        """
        Convert a ToolDefinition to a searchable document string.

        Args:
            tool: Tool definition

        Returns:
            Document string containing tool metadata
        """
        parts = [
            f"Name: {tool.name}",
            f"Description: {tool.description}",
        ]

        if tool.tags:
            parts.append(f"Tags: {', '.join(tool.tags)}")

        if tool.examples:
            # ToolDefinition schema enforces examples: list[str]
            examples_str = " | ".join(tool.examples)
            parts.append(f"Examples: {examples_str}")

        return " | ".join(parts)

    async def rerank(
        self,
        query: str,
        tools: list[tuple[ToolDefinition, float]],
        top_k: int,
    ) -> list[tuple[ToolDefinition, float]]:
        """
        Rerank tools using Cohere Rerank API.

        Args:
            query: The user's query
            tools: List of (tool, score) tuples from initial retrieval
            top_k: Number of top results to return

        Returns:
            Reranked list of (tool, score) tuples

        Falls back to original scores if API call fails.
        """
        if not tools:
            return []

        # If no client configured, fall back to original scores
        if self._client is None:
            logger.warning("Cohere client not configured, using original scores")
            return sorted(tools, key=lambda x: x[1], reverse=True)[:top_k]

        try:
            # Convert tools to documents
            documents = [self._tool_to_document(tool) for tool, _ in tools]

            # Call Cohere Rerank API
            params = {
                "query": query,
                "documents": documents,
                "top_n": min(top_k, len(documents)),
                "model": self._model,
            }

            if self._max_chunks_per_doc is not None:
                params["max_chunks_per_doc"] = self._max_chunks_per_doc

            response = await self._client.rerank(**params)

            # Map results back to tools using index
            reranked: list[tuple[ToolDefinition, float]] = []
            for result in response.results:
                tool, _ = tools[result.index]
                # Use relevance_score from Cohere (0-1 scale)
                reranked.append((tool, result.relevance_score))

            return reranked

        except Exception as e:
            # Fall back to original scores on error
            logger.warning(f"Cohere rerank failed: {e}, using original scores")
            return sorted(tools, key=lambda x: x[1], reverse=True)[:top_k]

