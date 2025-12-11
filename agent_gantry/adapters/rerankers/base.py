"""
Base reranker adapter protocol.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from agent_gantry.schema.tool import ToolDefinition


class RerankerAdapter(Protocol):
    """
    Rerank tools after vector search for higher precision.

    Implementations: CohereReranker, CrossEncoderReranker, LLMReranker.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        tools: list[tuple[ToolDefinition, float]],
        top_k: int,
    ) -> list[tuple[ToolDefinition, float]]:
        """
        Rerank tools based on the query.

        Args:
            query: The user's query
            tools: List of (tool, score) tuples from initial retrieval
            top_k: Number of top results to return

        Returns:
            Reranked list of (tool, score) tuples
        """
        ...
