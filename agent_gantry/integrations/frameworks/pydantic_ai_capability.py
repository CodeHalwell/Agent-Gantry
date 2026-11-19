"""Gantry as a Pydantic AI ``Capability`` and ``ToolSearch`` strategy.

``gantry_search_strategy``
    A ``ToolSearchFunc`` -- ``(ctx, queries, tools) -> list[str]`` -- for
    ``pydantic_ai.capabilities.ToolSearch(strategy=...)``. Gantry-managed
    deferred tools (recognised by the ``"agent_gantry"`` entry the Gantry
    toolsets put in ``ToolDefinition.metadata``, which survives ``prefixed`` /
    ``renamed`` wrappers) are ranked by Gantry's semantic router; any other
    deferred tools on the agent fall back to keyword matching so they stay
    discoverable.

``GantryCapability``
    An ``AbstractCapability`` contributing a Gantry toolset (per-turn or
    deferred) and, optionally, the Gantry skills relevant to the current turn as
    dynamic instructions. In search mode it is combined with the Gantry-backed
    ``ToolSearch`` so a single object wires the whole thing:
    ``Agent(model, capabilities=[adapter.capability(mode="search")])``.

Built lazily (the base classes only exist once ``pydantic_ai`` is importable).
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import DEFAULT_TOOL_LIMIT
from agent_gantry.integrations.frameworks.base import GantryToolset as _BaseToolset
from agent_gantry.integrations.frameworks.pydantic_ai import (
    _INSTALL_HINT,
    METADATA_KEY,
    registry_token,
)
from agent_gantry.integrations.frameworks.pydantic_ai_live import _query_from_ctx

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")

#: ``ToolQuery.limit`` ceiling: the most candidates one retrieval can return.
_MAX_RETRIEVAL_LIMIT = 50


def _gantry_name(tool_def: Any, registry: str) -> str | None:
    """The tool's qualified name if it belongs to ``registry``, else ``None``.

    A deferred tool from another ``AgentGantry`` on the same agent is not
    this strategy's to rank: it falls through to keyword matching instead.
    """
    meta = getattr(tool_def, "metadata", None) or {}
    entry = meta.get(METADATA_KEY) if isinstance(meta, dict) else None
    if not isinstance(entry, dict) or entry.get("registry") != registry:
        return None
    return entry.get("qualified_name")


def _keyword_rank(ctx: Any, queries: Sequence[str], tools: Sequence[Any]) -> list[str]:
    """Pydantic AI's own keyword search where available, else a local equivalent."""
    if not tools:
        return []
    try:
        from pydantic_ai.capabilities import keywords_search
    except ImportError:
        keywords_search = None
    if keywords_search is not None:
        return list(keywords_search(ctx, queries, tools))
    terms = set(_WORD.findall(" ".join(queries).lower()))
    scored = []
    for td in tools:
        words = set(_WORD.findall(f"{td.name} {td.description or ''}".lower()))
        hits = len(terms & words)
        if hits:
            scored.append((-hits, td.name))
    return [name for _, name in sorted(scored)]


def gantry_search_strategy(
    gantry: AgentGantry,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
) -> Any:
    """Build a ``ToolSearch`` strategy backed by Gantry's semantic router.

    ``required``/``always_include`` are accepted for signature symmetry with the
    other builders and ignored: pinned tools are made visible by the deferred
    toolset rather than found through search.
    """
    selector = _BaseToolset(gantry)
    registry = registry_token(gantry)

    def _fetch_limit(eligible: int) -> int:
        """Candidates to retrieve so ``limit`` eligible ones survive the filter.

        The registry can hold tools the search corpus does not -- pinned-visible
        ones, or ones a ``filtered``/namespaced toolset leaves out. They can
        outrank eligible tools, so retrieve enough to cover them before cutting
        to ``limit`` (bounded by the retrieval ceiling).
        """
        in_scope = sum(
            1 for tool in gantry.list_tools_sync() if not namespaces or tool.namespace in namespaces
        )
        return min(_MAX_RETRIEVAL_LIMIT, limit + max(0, in_scope - eligible))

    async def search(ctx: Any, queries: Sequence[str], tools: Sequence[Any]) -> list[str]:
        by_qualified: dict[str, str] = {}
        others = []
        for td in tools:
            qualified = _gantry_name(td, registry)
            if qualified:
                by_qualified[qualified] = td.name
            else:
                others.append(td)

        ranked: list[str] = []
        if by_qualified:
            fetch = _fetch_limit(len(by_qualified))
            per_query = []
            for query in queries:
                if not (query or "").strip():
                    continue
                try:
                    per_query.append(
                        await selector.select(
                            query,
                            limit=fetch,
                            score_threshold=score_threshold,
                            namespaces=namespaces,
                        )
                    )
                except Exception:
                    logger.warning(
                        "Gantry tool search: semantic retrieval failed for %r; "
                        "falling back to keyword matching.",
                        query,
                        exc_info=True,
                    )
                    others.extend(td for td in tools if _gantry_name(td, registry))
                    by_qualified = {}
                    per_query = []
                    break
            # Interleave per-query rankings so each query's best hit comes first.
            seen: set[str] = set()
            for rank in range(max((len(r) for r in per_query), default=0)):
                for results in per_query:
                    if rank < len(results):
                        name = by_qualified.get(results[rank].qualified_name)
                        if name and name not in seen:
                            seen.add(name)
                            ranked.append(name)
            ranked = ranked[:limit]
        for name in _keyword_rank(ctx, queries, others):
            if name not in ranked:
                ranked.append(name)
        return ranked

    search.__name__ = "gantry_semantic_search"
    return search


def _build_capability_class() -> type:
    try:
        from pydantic_ai.capabilities import AbstractCapability
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return _capability_class(AbstractCapability)


@functools.cache
def _capability_class(abstract_capability: type) -> type:
    class GantryCapability(abstract_capability):  # type: ignore[misc, valid-type]
        """Contributes a Gantry toolset and, optionally, per-turn Gantry skills."""

        def __init__(
            self,
            gantry: AgentGantry,
            toolset: Any,
            *,
            skills: int = 0,
            skill_namespaces: list[str] | None = None,
            id: str | None = None,  # noqa: A002 - Pydantic AI's capability ``id``
            query_from_results: bool = True,
        ) -> None:
            super().__init__()
            if id is not None:
                self.id = id
            self._gantry = gantry
            self._toolset = toolset
            self._skills = skills
            self._skill_namespaces = skill_namespaces
            self._query_from_results = query_from_results

        def __repr__(self) -> str:
            return f"GantryCapability(toolset={self._toolset!r}, skills={self._skills})"

        def get_toolset(self) -> Any:
            return self._toolset

        def get_instructions(self) -> Any:
            if self._skills <= 0:
                return None

            async def gantry_skills(ctx: Any) -> str:
                query = _query_from_ctx(ctx, include_results=self._query_from_results)
                if not query.strip():
                    return ""
                try:
                    return await self._gantry.retrieve_skills_as_prompt(
                        query, limit=self._skills, namespace=self._skill_namespaces
                    )
                except Exception:
                    logger.warning(
                        "GantryCapability: skill retrieval failed; continuing without "
                        "Gantry skills this step.",
                        exc_info=True,
                    )
                    return ""

            return gantry_skills

    return GantryCapability


def _gantry_capability(
    gantry: AgentGantry,
    toolset: Any,
    *,
    tool_search: Any = None,
    skills: int = 0,
    skill_namespaces: list[str] | None = None,
    id: str | None = None,  # noqa: A002
    query_from_results: bool = True,
) -> Any:
    """A ``GantryCapability``, combined with ``tool_search`` when given."""
    capability = _build_capability_class()(
        gantry,
        toolset,
        skills=skills,
        skill_namespaces=skill_namespaces,
        id=id,
        query_from_results=query_from_results,
    )
    if tool_search is None:
        return capability
    try:
        from pydantic_ai.capabilities import CombinedCapability
    except ImportError as exc:  # pragma: no cover - guarded by the class build above
        raise ImportError(_INSTALL_HINT) from exc

    return CombinedCapability([capability, tool_search])


def __getattr__(name: str) -> Any:
    """Expose the dynamically-built ``GantryCapability`` lazily (for ``isinstance``)."""
    if name == "GantryCapability":
        return _build_capability_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
