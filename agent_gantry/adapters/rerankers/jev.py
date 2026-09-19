"""
TypeSafe Jev reranker.

Reranks the vector-search shortlist by asking Jev one ``Noul`` per candidate —
"would this tool help?" — and ordering by the returned probability. This is the
recipe TypeSafe publish for retrieval: on their CLERC benchmark, reranking a
BM25 top-30 shortlist moved top-1 accuracy from 5% to 18% and top-10 from 38%
to 62%.

Unlike the selector in ``adapters/selectors/jev.py``, this runs *after*
semantic search, so it works at any catalogue size: the shortlist is what gets
sent, not the registry.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any

from agent_gantry.adapters.jev_client import JevClient
from agent_gantry.adapters.rerankers.base import RerankerAdapter
from agent_gantry.schema.selection import SelectionCandidate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_gantry.schema.tool import ToolDefinition

logger = logging.getLogger(__name__)

DEFAULT_QUESTION = "Would this tool help accomplish the request? Answer about this tool only."


def _accepts_require_all(score: Any) -> bool:
    """Whether ``score`` takes the ``require_all`` keyword.

    ``client`` is a documented extension point ("or a compatible stub"), and
    ``require_all`` was added to :meth:`JevClient.score` after that contract
    was published. Passing it blindly turns a stub written against the older
    three-argument signature into a ``TypeError`` raised straight through
    ``rerank()`` — in a code path whose entire promise is that a failing
    provider costs precision and never the catalogue. Ask first.

    An unreadable signature (a C callable, an exotic mock) is treated as
    accepting it: the keyword is the behaviour we want, so only decline it
    where we can see it would break.
    """
    try:
        parameters = inspect.signature(score).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return True
    if "require_all" in parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


class JevReranker(RerankerAdapter):
    """Rerank retrieved tools with TypeSafe's Jev model.

    Args:
        api_key: TypeSafe API key. Defaults to ``TYPESAFE_API_KEY``.
        model: Model id. Defaults to the SDK's default (``jev-latest``).
        question: The yes/no question asked of each tool. Worth tuning to the
            deployment: Jev "answers the question you wrote, not the one you
            meant", so scoping words and negations are read literally.
        timeout: Per-request timeout in seconds.
        max_candidates: Ceiling on how many tools are sent. The router hands
            over ``limit * 4`` candidates, so this is a cost guard rather than
            a correctness one; anything beyond it keeps its incoming order and
            ranks below everything scored.
        client: Optional pre-built :class:`JevClient` (or a compatible stub).
            A stub whose ``score`` predates the ``require_all`` keyword is
            still supported: it is called without it, which matches the
            leniency reranking wants anyway.

    Raises:
        ImportError: If the ``typesafe-sdk`` package is not installed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        question: str = DEFAULT_QUESTION,
        timeout: float | None = None,
        max_candidates: int = 100,
        client: Any | None = None,
    ) -> None:
        self._question = question
        self._max_candidates = max(1, int(max_candidates))
        self._client = (
            client
            if client is not None
            else JevClient(api_key=api_key, model=model, timeout=timeout)
        )
        self._client_takes_require_all = _accepts_require_all(self._client.score)
        logger.info(f"Initialized JevReranker with model={model or 'jev-latest'}")

    async def rerank(
        self,
        query: str,
        tools: list[tuple[ToolDefinition, float]],
        top_k: int,
    ) -> list[tuple[ToolDefinition, float]]:
        """Rerank tools by Jev's relevance probability.

        Args:
            query: The user's query.
            tools: ``(tool, score)`` pairs from vector search.
            top_k: How many to return.

        Returns:
            Reranked ``(tool, probability)`` pairs. On any API failure the
            incoming order is returned untouched, truncated to ``top_k``:
            reranking is a precision boost over an already-usable ranking, so
            losing it must cost precision, not the tools themselves.
        """
        if not tools:
            return []

        considered = tools[: self._max_candidates]
        overflow = tools[self._max_candidates :]

        # Qualified names are unique per registry, but a caller can hand the
        # same tool over twice; keep the first of each so the scores mapping
        # stays one-to-one.
        candidates: list[SelectionCandidate] = []
        seen: set[str] = set()
        for tool, _score in considered:
            candidate = SelectionCandidate.from_tool(tool)
            if candidate.id in seen:
                continue
            seen.add(candidate.id)
            candidates.append(candidate)

        # require_all=False: reranking returns the whole shortlist either way,
        # so an unscored tool keeps its search rank below the scored ones
        # rather than costing the pass. Selection cannot do that — there an
        # unanswered candidate would simply vanish.
        if self._client_takes_require_all:
            verdict = await self._client.score(
                query, candidates, self._question, require_all=False
            )
        else:
            # A stub predating the keyword. Its partial-response behaviour is
            # the old lenient one, which is exactly what reranking wants, so
            # nothing is lost by omitting it.
            verdict = await self._client.score(query, candidates, self._question)
        if verdict.fallback:
            logger.debug(f"Jev rerank fell back ({verdict.reason}); keeping search order")
            return tools[:top_k]

        scored: list[tuple[ToolDefinition, float]] = []
        unscored: list[tuple[ToolDefinition, float]] = []
        for tool, score in considered:
            probability = verdict.scores.get(tool.qualified_name)
            if probability is None:
                unscored.append((tool, score))
            else:
                scored.append((tool, probability))

        # Stable within equal probabilities, so the vector store's own ordering
        # breaks ties rather than dict insertion order.
        scored.sort(key=lambda pair: pair[1], reverse=True)
        # Anything the model did not answer for keeps its search rank, below
        # everything it did answer for — the model saw them and said nothing,
        # which is weaker evidence than a low probability, not stronger.
        return (scored + unscored + overflow)[:top_k]

    async def aclose(self) -> None:
        """Release the underlying HTTP client."""
        aclose = getattr(self._client, "aclose", None)
        if aclose is not None:
            await aclose()
