"""
Base selector adapter protocol.

A selector is the alternative to semantic routing: rather than embedding the
query and searching a vector store, it asks a model directly which entries of a
catalogue are relevant. No embeddings, no vector store, no sync step.

The trade is that every candidate is sent as input on every query, so cost and
latency grow with the catalogue rather than staying flat. That is affordable
with a decision model priced two orders of magnitude below a frontier LLM, and
it is why selection suits catalogues of tens to a few hundred entries while
semantic retrieval (optionally with a reranker over its shortlist) stays the
right answer above that.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from agent_gantry.schema.selection import SelectionCandidate, SelectionResult


class SelectorAdapter(Protocol):
    """Choose relevant entries from a catalogue without embedding it.

    Implementations: JevSelector.
    """

    @abstractmethod
    async def select(
        self,
        query: str,
        candidates: Sequence[SelectionCandidate],
        limit: int,
    ) -> SelectionResult:
        """Select the entries relevant to *query*.

        Args:
            query: The user's request.
            candidates: The catalogue to choose from.
            limit: Maximum number of entries to return.

        Returns:
            A :class:`SelectionResult`. Implementations must not raise for a
            provider failure — they set ``fallback`` instead, so the caller can
            fall back to its own ordering rather than losing the catalogue.
        """
        ...
