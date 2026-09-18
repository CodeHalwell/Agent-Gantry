"""
TypeSafe Jev selector.

Replaces embed-then-search with a direct question per candidate: "would this
help?", answered as a probability. One ``Noul`` per entry rather than a single
``Choice``, because a request routinely needs several tools and ``Choice``
returns exactly one option (and caps out at 255 anyway).

For catalogues too large to send in one pass the selector narrows in two
stages, the shape TypeSafe use for their own 182-entry skill catalogue: score
the *groups* first (a namespace, or the MCP server a tool came from), then
score only the members of the groups that came top. The group stage is a budget
device and always keeps the best few groups; the relevance gate is the
threshold applied to members.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_gantry.adapters.jev_client import JevClient
from agent_gantry.adapters.selectors.base import SelectorAdapter
from agent_gantry.schema.selection import SelectionCandidate, SelectionResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: One question per catalogue, because jev-1.13 "answers the question you
#: wrote, not the one you meant". Asking whether a *tool* would help is the
#: wrong question about an Agent Skill, which is procedural knowledge to read
#: rather than something to call, and about an MCP server, which is a source of
#: tools rather than one itself.
DEFAULT_QUESTIONS = {
    "tool": "Would this tool help accomplish the request? Answer about this tool only.",
    "skill": (
        "Would this skill's instructions help someone carry out the request? "
        "Answer about this skill only."
    ),
    "mcp_server": (
        "Would this server provide tools useful for the request? "
        "Answer about this server only."
    ),
}

DEFAULT_QUESTION = DEFAULT_QUESTIONS["tool"]

DEFAULT_GROUP_QUESTIONS = {
    "tool": (
        "Could any tool in this group help accomplish the request? "
        "Answer about this group only."
    ),
    "skill": (
        "Could any skill in this group help someone carry out the request? "
        "Answer about this group only."
    ),
    "mcp_server": (
        "Could any server in this group provide tools useful for the request? "
        "Answer about this group only."
    ),
}

DEFAULT_GROUP_QUESTION = DEFAULT_GROUP_QUESTIONS["tool"]

#: Below this probability a candidate is not selected. TypeSafe's own skill
#: recipe gates at 0.30; the same value is used here, and it is the one knob
#: most worth tuning per deployment.
DEFAULT_THRESHOLD = 0.3

#: Bucket key for candidates carrying no group. Bracketed so it cannot collide
#: with a real namespace or MCP server name.
_UNGROUPED = "(ungrouped)"


class JevSelector(SelectorAdapter):
    """Select tools, skills or MCP servers with TypeSafe's Jev model.

    Args:
        api_key: TypeSafe API key. Defaults to ``TYPESAFE_API_KEY``.
        model: Model id. Defaults to the SDK's default (``jev-latest``).
        question: The yes/no question asked of each candidate, overriding the
            per-catalogue defaults. Jev "answers the question you wrote, not
            the one you meant" — scoping words and negations are read
            literally — so this is worth tuning.
        group_question: The question asked of each group in the two-stage path,
            overriding the per-catalogue defaults.
        threshold: Minimum probability for a candidate to be selected.
        max_candidates: Hard ceiling on candidates scored in one call, before
            grouping. Guards cost on an unexpectedly large registry.
        group_after: Catalogue size above which the two-stage path is used,
            when the candidates carry groups.
        max_groups: How many groups the first stage keeps.
        timeout: Per-request timeout in seconds.
        client: Optional pre-built :class:`JevClient` (or a compatible stub).

    Raises:
        ImportError: If the ``typesafe-sdk`` package is not installed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        question: str | None = None,
        group_question: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        max_candidates: int = 512,
        group_after: int = 120,
        max_groups: int = 5,
        timeout: float | None = None,
        client: Any | None = None,
    ) -> None:
        # ``None`` means "use the question for whatever catalogue is being
        # selected"; an explicit string overrides all of them, which is what a
        # caller tuning one deployment wants.
        self._question = question
        self._group_question = group_question
        self._threshold = float(threshold)
        self._max_candidates = max(1, int(max_candidates))
        self._group_after = max(1, int(group_after))
        self._max_groups = max(1, int(max_groups))
        self._client = (
            client
            if client is not None
            else JevClient(api_key=api_key, model=model, timeout=timeout)
        )
        logger.info(
            f"Initialized JevSelector with model={model or 'jev-latest'}, "
            f"threshold={self._threshold}"
        )

    def _question_for(self, kind: str) -> str:
        """The per-candidate question for *kind*, unless one was configured."""
        return self._question or DEFAULT_QUESTIONS.get(kind, DEFAULT_QUESTION)

    def _group_question_for(self, kind: str) -> str:
        """The group-pass question for *kind*, unless one was configured."""
        return self._group_question or DEFAULT_GROUP_QUESTIONS.get(kind, DEFAULT_GROUP_QUESTION)

    async def select(
        self,
        query: str,
        candidates: Sequence[SelectionCandidate],
        limit: int,
        *,
        kind: str = "tool",
    ) -> SelectionResult:
        """Select the candidates relevant to *query*.

        Args:
            query: The user's request. Sent as the shared state.
            candidates: The catalogue to choose from.
            limit: Maximum number of ids to return.
            kind: Which catalogue this is — ``"tool"``, ``"skill"`` or
                ``"mcp_server"`` — which picks the question asked about each
                entry.

        Returns:
            A :class:`SelectionResult`, best first. Never raises for a provider
            failure: ``fallback`` is set instead.
        """
        if not candidates or limit <= 0:
            return SelectionResult()

        if len(candidates) > self._max_candidates:
            # Falling back, not truncating. Scoring an insertion-order prefix
            # would make every later entry permanently unreachable while
            # reporting success, so a tool registered after the ceiling could
            # never be retrieved however relevant it was — and semantic
            # routing, which has no such ceiling, would never get a look in.
            logger.warning(
                f"Jev selection saw {len(candidates)} candidates, over the "
                f"{self._max_candidates} ceiling; falling back to semantic routing. Raise "
                "max_candidates, or use the reranker over semantic search at this size."
            )
            return SelectionResult(
                fallback=True,
                reason=f"catalogue of {len(candidates)} exceeds max_candidates={self._max_candidates}",
            )
        considered = list(candidates)

        groups = {c.group for c in considered if c.group}
        try:
            if len(considered) > self._group_after and len(groups) > 1:
                return await self._select_two_stage(query, considered, limit, kind)
            return await self._select_one_stage(query, considered, limit, kind)
        except Exception as exc:  # noqa: BLE001 - fail open is the contract
            # The client already turns provider failures into a fallback
            # verdict, so reaching here means a defect in this adapter. Even
            # then the caller must get its catalogue back: raising out of the
            # selection layer costs the agent every tool it has, which is a
            # far worse failure than an unselected list.
            logger.warning(f"Jev selection raised; falling back: {exc}", exc_info=True)
            return SelectionResult(fallback=True, reason=f"{type(exc).__name__}: {exc}")

    async def _select_one_stage(
        self,
        query: str,
        candidates: Sequence[SelectionCandidate],
        limit: int,
        kind: str,
    ) -> SelectionResult:
        """Score every candidate and keep those over the threshold."""
        verdict = await self._client.score(query, candidates, self._question_for(kind))
        if verdict.fallback:
            return SelectionResult(
                fallback=True,
                reason=verdict.reason,
                requests=verdict.requests,
            )
        return self._finalize(verdict.scores, limit, verdict)

    async def _select_two_stage(
        self,
        query: str,
        candidates: Sequence[SelectionCandidate],
        limit: int,
        kind: str,
    ) -> SelectionResult:
        """Narrow to the best groups, then score only their members."""
        by_group: dict[str, list[SelectionCandidate]] = {}
        for candidate in candidates:
            # A real group name can never collide with the sentinel, because
            # ``group`` is a namespace or server name and neither is bracketed.
            by_group.setdefault(candidate.group or _UNGROUPED, []).append(candidate)

        group_candidates = [
            SelectionCandidate(
                id=group,
                name=group,
                # The members are the only thing that says what a namespace is
                # for, so the group's description is a roll-call of its tools.
                description="Contains: " + ", ".join(c.name for c in members),
            )
            for group, members in by_group.items()
        ]

        group_verdict = await self._client.score(
            query, group_candidates, self._group_question_for(kind)
        )
        if group_verdict.fallback:
            # Narrowing failed, and scoring everything instead would be the
            # budget blow-out the two-stage path exists to avoid.
            return SelectionResult(
                fallback=True,
                reason=group_verdict.reason,
                requests=group_verdict.requests,
            )

        # Always keep the best few groups rather than applying the threshold
        # here: this stage exists to fit the budget, not to decide relevance.
        # Gating it too would let one cautious answer zero out the catalogue
        # before any tool had been looked at individually.
        ranked_groups = sorted(
            by_group, key=lambda g: group_verdict.scores.get(g, 0.0), reverse=True
        )
        kept = ranked_groups[: self._max_groups]
        members = [c for group in kept for c in by_group[group]]

        verdict = await self._client.score(query, members, self._question_for(kind))
        if verdict.fallback:
            return SelectionResult(
                fallback=True,
                reason=verdict.reason,
                requests=group_verdict.requests + verdict.requests,
            )

        result = self._finalize(verdict.scores, limit, verdict)
        result.requests += group_verdict.requests
        result.input_tokens += group_verdict.input_tokens
        result.output_tokens += group_verdict.output_tokens
        return result

    def _finalize(
        self,
        scores: dict[str, float],
        limit: int,
        verdict: Any,
    ) -> SelectionResult:
        """Threshold, order and truncate a scored pass."""
        selected = [
            candidate_id
            for candidate_id, score in sorted(
                scores.items(), key=lambda pair: pair[1], reverse=True
            )
            if score >= self._threshold
        ]
        return SelectionResult(
            selected=selected[:limit],
            # Every score, not just the selected ones: a caller tuning the
            # threshold needs to see the near-misses it is cutting.
            scores=scores,
            input_tokens=verdict.input_tokens,
            output_tokens=verdict.output_tokens,
            requests=verdict.requests,
        )

    async def aclose(self) -> None:
        """Release the underlying HTTP client."""
        aclose = getattr(self._client, "aclose", None)
        if aclose is not None:
            await aclose()
