"""
Lightweight Cohere reranker stub.

Implements the reranker protocol without external dependencies by
re-scoring candidates using simple keyword overlap.
"""

from __future__ import annotations

from typing import Iterable

from agent_gantry.adapters.rerankers.base import RerankerAdapter
from agent_gantry.schema.tool import ToolDefinition


class CohereReranker(RerankerAdapter):
    """Deterministic reranker for tests and config switching."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or "rerank-english-v3.0"

    async def rerank(
        self,
        query: str,
        tools: list[tuple[ToolDefinition, float]],
        top_k: int,
    ) -> list[tuple[ToolDefinition, float]]:
        keywords = set(self._tokenize(query))
        rescored: list[tuple[ToolDefinition, float]] = []

        for tool, base_score in tools:
            text = f"{tool.name} {tool.description} {' '.join(tool.tags)}"
            overlap = len(keywords.intersection(self._tokenize(text)))
            rescored.append((tool, base_score + 0.05 * overlap))

        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[:top_k]

    def _tokenize(self, text: str) -> Iterable[str]:
        return text.lower().split()
