"""dotfetch as a Pydantic AI ``Capability`` and ``ToolSearch`` strategy.

``fetcher_search_strategy``
    A ``ToolSearchFunc`` -- ``(ctx, queries, tools) -> list[str]`` -- for
    ``pydantic_ai.capabilities.ToolSearch(strategy=...)``. dotfetch-managed
    deferred tools (recognised by the ``"dotfetch"`` entry the dotfetch
    toolsets put in ``ToolDefinition.metadata``, which survives ``prefixed`` /
    ``renamed`` wrappers) are ranked by dotfetch's semantic router; any other
    deferred tools on the agent fall back to keyword matching so they stay
    discoverable.

``DotFetchCapability``
    An ``AbstractCapability`` contributing a dotfetch toolset (per-turn or
    deferred) and, optionally, the dotfetch skills relevant to the current turn as
    dynamic instructions. In search mode it is combined with the dotfetch-backed
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

from dotfetch.adapters.selectors.base import SelectorFallbackError
from dotfetch.integrations.frameworks.base import DEFAULT_TOOL_LIMIT
from dotfetch.integrations.frameworks.base import DotFetchToolset as _BaseToolset
from dotfetch.integrations.frameworks.pydantic_ai import (
    _INSTALL_HINT,
    METADATA_KEY,
    registry_token,
)
from dotfetch.integrations.frameworks.pydantic_ai_live import _query_from_ctx

if TYPE_CHECKING:
    from dotfetch.core.fetcher import DotFetch

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")

#: ``ToolQuery.limit`` ceiling: the most candidates one retrieval can return.
_MAX_RETRIEVAL_LIMIT = 50

#: Reciprocal-rank-fusion constant (the conventional 60): damps the gap between
#: adjacent ranks so neither ranking dominates through its top hit alone.
_RRF_K = 60


def _fuse(semantic: list[str], keyword: list[str]) -> list[str]:
    """Merge two rankings with reciprocal rank fusion.

    Used when dotfetch-ranked tools share one search budget with keyword-ranked
    tools the agent owns natively, so neither kind is pushed past the
    ``max_results`` cut just by being listed second. Ties go to keyword
    matches: an exact word match is stronger evidence than a semantic hit
    taken at a zero score threshold.
    """
    scores: dict[str, float] = {}
    for ranking in (semantic, keyword):
        for rank, name in enumerate(ranking, start=1):
            scores[name] = scores.get(name, 0.0) + 1.0 / (_RRF_K + rank)
    keyword_hits = set(keyword)
    order = {name: i for i, name in enumerate(dict.fromkeys([*semantic, *keyword]))}
    return sorted(scores, key=lambda n: (-scores[n], n not in keyword_hits, order[n]))


def _fetcher_name(tool_def: Any, registry: str) -> str | None:
    """The tool's qualified name if it belongs to ``registry``, else ``None``.

    A deferred tool from another ``DotFetch`` on the same agent is not
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


def fetcher_search_strategy(
    fetcher: DotFetch,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
) -> Any:
    """Build a ``ToolSearch`` strategy backed by dotfetch's semantic router.

    ``required``/``always_include`` are accepted for signature symmetry with the
    other builders and ignored: pinned tools are made visible by the deferred
    toolset rather than found through search.
    """
    selector = _BaseToolset(fetcher)
    registry = registry_token(fetcher)

    def _fetch_limit(eligible: int) -> int:
        """Candidates to retrieve so ``limit`` eligible ones survive the filter.

        The registry can hold tools the search corpus does not -- pinned-visible
        ones, or ones a ``filtered``/namespaced toolset leaves out. They can
        outrank eligible tools, so retrieve enough to cover them before cutting
        to ``limit`` (bounded by the retrieval ceiling).
        """
        in_scope = sum(
            1 for tool in fetcher.list_tools_sync() if not namespaces or tool.namespace in namespaces
        )
        return min(_MAX_RETRIEVAL_LIMIT, limit + max(0, in_scope - eligible))

    def _allowed(tool: Any) -> bool:
        """The router's hard filters, as ``select`` applies them by default."""
        if getattr(tool, "deprecated", False):
            return False
        if namespaces and tool.namespace not in namespaces:
            return False
        health = getattr(tool, "health", None)
        return not getattr(health, "circuit_breaker_open", False)

    def _eligible_own(tools: Sequence[Any]) -> list[Any]:
        """This registry's tools from ``tools`` that pass the hard filters."""
        known = {tool.qualified_name: tool for tool in fetcher.list_tools_sync()}
        eligible = []
        for td in tools:
            tool = known.get(_fetcher_name(td, registry) or "")
            if tool is not None and _allowed(tool):
                eligible.append(td)
        return eligible

    async def search(ctx: Any, queries: Sequence[str], tools: Sequence[Any]) -> list[str]:
        by_qualified: dict[str, str] = {}
        others = []
        for td in tools:
            qualified = _fetcher_name(td, registry)
            if qualified:
                by_qualified[qualified] = td.name
            else:
                others.append(td)

        ranked: list[str] = []
        if by_qualified:
            fetch = _fetch_limit(len(by_qualified))
            per_query = []
            failed = False
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
                except SelectorFallbackError:
                    raise  # strict mode: never search on a different backend
                except Exception:
                    # Keep the queries that did succeed; only if every one fails
                    # do dotfetch's tools fall back to keyword matching (below).
                    logger.warning(
                        "dotfetch tool search: semantic retrieval failed for %r; skipping it.",
                        query,
                        exc_info=True,
                    )
                    failed = True
            if failed and not per_query:
                # Total outage: rank this registry's tools by keyword instead,
                # but only those the router itself would have allowed.
                others.extend(_eligible_own(tools))
                by_qualified = {}
            # Interleave per-query rankings so each query's best hit comes first.
            seen: set[str] = set()
            for rank in range(max((len(r) for r in per_query), default=0)):
                for results in per_query:
                    if rank < len(results):
                        name = by_qualified.get(results[rank].qualified_name)
                        if name and name not in seen:
                            seen.add(name)
                            ranked.append(name)
        if not others:
            return ranked[:limit]  # dotfetch tools only: pure semantic ranking.
        # Shared budget: fuse dotfetch's semantic ranking with a keyword ranking of
        # the *other* tools, so a native tool's exact match can outrank an
        # unrelated dotfetch hit, then apply ``limit`` once to the merged list.
        # Keyword ranking never sees this registry's tools while semantic
        # retrieval works: that would reintroduce tools its namespace,
        # deprecation and health filters excluded.
        return _fuse(ranked, _keyword_rank(ctx, queries, others))[:limit]

    search.__name__ = "fetcher_semantic_search"
    return search


def _build_capability_class() -> type:
    try:
        from pydantic_ai.capabilities import AbstractCapability
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return _capability_class(AbstractCapability)


@functools.cache
def _capability_class(abstract_capability: type) -> type:
    class DotFetchCapability(abstract_capability):  # type: ignore[misc, valid-type]
        """Contributes a dotfetch toolset and, optionally, per-turn dotfetch skills."""

        def __init__(
            self,
            fetcher: DotFetch,
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
            self._fetcher = fetcher
            self._toolset = toolset
            self._skills = skills
            self._skill_namespaces = skill_namespaces
            self._query_from_results = query_from_results

        def __repr__(self) -> str:
            return f"DotFetchCapability(toolset={self._toolset!r}, skills={self._skills})"

        def get_toolset(self) -> Any:
            return self._toolset

        def get_instructions(self) -> Any:
            if self._skills <= 0:
                return None

            async def fetcher_skills(ctx: Any) -> str:
                query = _query_from_ctx(ctx, include_results=self._query_from_results)
                if not query.strip():
                    return ""
                try:
                    return await self._fetcher.retrieve_skills_as_prompt(
                        query, limit=self._skills, namespace=self._skill_namespaces
                    )
                except SelectorFallbackError:
                    raise  # strict mode: never retrieve on a different backend
                except Exception:
                    logger.warning(
                        "DotFetchCapability: skill retrieval failed; continuing without "
                        "dotfetch skills this step.",
                        exc_info=True,
                    )
                    return ""

            return fetcher_skills

    return DotFetchCapability


def _fetcher_capability(
    fetcher: DotFetch,
    toolset: Any,
    *,
    tool_search: Any = None,
    skills: int = 0,
    skill_namespaces: list[str] | None = None,
    id: str | None = None,  # noqa: A002
    query_from_results: bool = True,
) -> Any:
    """A ``DotFetchCapability``, combined with ``tool_search`` when given."""
    capability = _build_capability_class()(
        fetcher,
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
    """Expose the dynamically-built ``DotFetchCapability`` lazily (for ``isinstance``)."""
    if name == "DotFetchCapability":
        return _build_capability_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
