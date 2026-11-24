"""
Semantic router for Agent-Gantry.

Intelligent tool selection using semantic search, intent classification, and context.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from agent_gantry.adapters.embedders.base import embed_query
from agent_gantry.core.chunks import CHUNK_SCORE_WEIGHT, chunk_parent
from agent_gantry.schema.a2a import a2a_agent_tag
from agent_gantry.schema.mcp import mcp_server_tag
from agent_gantry.schema.tool import ToolDefinition, chunk_namespace, is_reserved_namespace

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.llm_client import LLMClient
    from agent_gantry.adapters.rerankers.base import RerankerAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.schema.query import ToolQuery


@lru_cache(maxsize=256)
def _get_token_pattern(token: str) -> re.Pattern[str]:
    """Cache compiled regex patterns for token matching."""
    return re.compile(rf"(?<!\w){re.escape(token)}(?!\w)")


class TaskIntent(str, Enum):
    """High-level task intent classification."""

    DATA_QUERY = "data_query"
    DATA_MUTATION = "data_mutation"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    FILE_OPERATIONS = "file_operations"
    CUSTOMER_SUPPORT = "customer_support"
    ADMIN = "admin"
    UNKNOWN = "unknown"


INTENT_TAG_MAPPING: dict[TaskIntent, list[str]] = {
    TaskIntent.DATA_QUERY: ["query", "search", "get", "list", "fetch", "read"],
    TaskIntent.DATA_MUTATION: ["create", "update", "delete", "write", "modify"],
    TaskIntent.ANALYSIS: ["analyze", "compute", "aggregate", "calculate", "report"],
    TaskIntent.COMMUNICATION: [
        "email",
        "message",
        "notify",
        "send",
        "chat",
        "dm",
        "teams",
        "discord",
        "communicate",
    ],
    TaskIntent.FILE_OPERATIONS: ["file", "upload", "download", "convert", "export"],
    TaskIntent.CUSTOMER_SUPPORT: ["ticket", "refund", "support", "customer"],
    TaskIntent.ADMIN: ["user", "permission", "setting", "config", "admin"],
}


# An intent keyword matches at the start of a token — ``send`` matches
# ``send``, ``sending`` and ``send_email`` — but never inside one. Plain
# substring matching counted ``budget`` and ``target`` as ``get``, ``thread``
# and ``spread`` as ``read`` and ``admin`` as ``dm``, and at 0.15 of the final
# score a spurious intent match is larger than the semantic gap between
# neighbouring tools, so mismatched tools filled top-k slots. An underscore is
# a token separator here (tool names are snake_case), which is why this is not
# the regex ``\b``. One pattern builder serves both query classification and
# tool-text matching so the two cannot disagree about what a keyword is.
_TOKEN_START = r"(?<![A-Za-z0-9])"


def _keyword_pattern(keywords: list[str]) -> re.Pattern[str]:
    """Compile a pattern matching any of ``keywords`` at the start of a token."""
    alternatives = "|".join(re.escape(kw) for kw in keywords)
    return re.compile(rf"{_TOKEN_START}(?:{alternatives})")


# Pre-compiled per intent for the routing inner loop: one ``search`` per tool
# rather than a Python-level scan over every keyword.
INTENT_TAG_PATTERNS: dict[TaskIntent, re.Pattern[str]] = {
    intent: _keyword_pattern(keywords) for intent, keywords in INTENT_TAG_MAPPING.items()
}

#: What a tool loses for having been called already in this conversation, for
#: having failed in it, and for being deprecated. Subtracted from a score in
#: ``[0, 1]`` on every path that ranks tools.
ALREADY_USED_PENALTY = 0.1
ALREADY_FAILED_PENALTY = 0.3
DEPRECATED_PENALTY = 0.5

#: What an MCP server loses while its last connection attempt failed: the
#: same as a tool that failed in this conversation.
UNAVAILABLE_SERVER_PENALTY = ALREADY_FAILED_PENALTY


def conversation_penalty(
    tool: ToolDefinition, tools_used: Collection[str], tools_failed: Collection[str]
) -> float:
    """How much to take off ``tool``'s score for what the conversation knows.

    Relevance says whether a tool fits the request. It cannot say that the
    tool was just tried and failed, or has already been called: that is the
    conversation's state, and it is orthogonal to relevance, so the penalty
    is applied on top of whichever score a path ranks by (a blended score on
    the semantic path, a probability on the selector path) and is the same
    on both.

    Args:
        tool: The tool being ranked.
        tools_used: Names (bare or ``namespace.name``) already called.
        tools_failed: Names (bare or ``namespace.name``) that failed.

    Returns:
        The total penalty, ``0.0`` for a tool the conversation says nothing
        about.
    """
    qualified = f"{tool.namespace}.{tool.name}"
    penalty = DEPRECATED_PENALTY if tool.deprecated else 0.0
    if tool.name in tools_used or qualified in tools_used:
        penalty += ALREADY_USED_PENALTY
    if tool.name in tools_failed or qualified in tools_failed:
        penalty += ALREADY_FAILED_PENALTY
    return penalty


def mmr_order(
    relevance: Sequence[float],
    embeddings: Sequence[Sequence[float]],
    diversity_factor: float,
    limit: int,
) -> list[int]:
    """Order candidates by maximal marginal relevance.

    Each pick maximises ``(1 - d) * relevance - d * similarity to what is
    already picked``, so with ``d = 0`` this is plain ranking and with a
    larger ``d`` a near-duplicate of something already chosen gives way to
    something different. It works on anything with a score and a vector:
    tools, skills and MCP servers all use it.

    Args:
        relevance: One score per candidate.
        embeddings: One vector per candidate, in the same order.
        diversity_factor: ``d``, in ``[0, 1]``.
        limit: How many to pick.

    Returns:
        Indices into the inputs, in pick order, at most ``limit`` of them.
    """
    if not relevance or limit <= 0:
        return []
    lambda_param = 1.0 - diversity_factor

    # Convert to numpy matrix for vectorized cosine similarity
    emb_matrix = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(emb_matrix, axis=1)

    # Pre-normalize embeddings to simplify cosine similarity to dot product
    with np.errstate(divide="ignore", invalid="ignore"):
        normed_emb = emb_matrix / norms[:, np.newaxis]
        normed_emb[np.isnan(normed_emb)] = 0.0

    selected: list[int] = []
    candidates = list(range(len(relevance)))

    first_idx = max(candidates, key=lambda i: relevance[i])
    selected.append(first_idx)
    candidates.remove(first_idx)

    # Track maximum similarity to any selected item for each candidate
    max_sims = {idx: 0.0 for idx in candidates}

    while candidates and len(selected) < limit:
        last_emb = normed_emb[selected[-1]]

        # Vectorized dot product of all remaining candidates with the last selected item
        all_sims = np.dot(normed_emb, last_emb)

        best_idx = -1
        best_score = -float("inf")

        for idx in candidates:
            sim = float(all_sims[idx])
            if sim > max_sims[idx]:
                max_sims[idx] = sim

            score = lambda_param * relevance[idx] - (1.0 - lambda_param) * max_sims[idx]
            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected


#: A keyword found only in a tool's description counts for this much at most,
#: against 1.0 for one in its name or tags.
_INTENT_DESCRIPTION_WEIGHT = 0.5
#: ...and for less the longer the description is, in proportion, past this
#: many characters.
_INTENT_DESCRIPTION_CHARS = 200


def intent_match_score(tool: ToolDefinition, pattern: re.Pattern[str]) -> float:
    """How strongly ``tool`` matches an intent's keywords, in ``[0, 1]``.

    A keyword in the name or tags says what the tool *is* (``search_logs``,
    ``list_users``) and scores 1.0. A keyword that appears only in the
    description scores at most half that, scaled down as the description
    grows, because prose mentions verbs in passing: any one of ``get``,
    ``list``, ``read``, ``search``, ``query`` or ``fetch`` turns up somewhere
    in most long descriptions.

    This used to be 1.0 for a keyword anywhere in name, description or tags.
    Measured over the library's own docstrings, 14% of those under 200
    characters matched the data-query keywords and 83% of those over 800 did,
    so the signal stopped discriminating on exactly the catalogues with long
    descriptions, and handed verbose tools a flat 0.15 of the final score
    over concise ones for no reason of relevance.
    """
    if pattern.search(f"{tool.name.lower()} {' '.join(tool.tags).lower()}"):
        return 1.0
    description = tool.description
    if not pattern.search(description.lower()):
        return 0.0
    return _INTENT_DESCRIPTION_WEIGHT * min(
        1.0, _INTENT_DESCRIPTION_CHARS / max(len(description), 1)
    )


# Per-keyword patterns for classification, which counts how many of an
# intent's keywords a query mentions rather than whether it mentions any.
_INTENT_KEYWORD_PATTERNS: dict[TaskIntent, list[re.Pattern[str]]] = {
    intent: [_keyword_pattern([kw]) for kw in keywords]
    for intent, keywords in INTENT_TAG_MAPPING.items()
}


@dataclass
class _QueryContextCache:
    """Per-query normalized conversation context, computed once per route call."""

    summary_lower: str
    messages_lower: list[str]
    tools_used: set[str]
    tools_failed: set[str]


@dataclass
class RoutingSignals:
    """Signals used for tool scoring."""

    semantic_similarity: float
    intent_match: float
    conversation_relevance: float
    health_score: float
    cost_score: float
    already_used_penalty: float
    already_failed_penalty: float
    deprecated_penalty: float


#: Fewest candidates fetched from the vector store per query, whatever the
#: query's ``limit``. Candidates come back ranked by raw cosine and are then
#: re-scored with intent, conversation, health and cost, and intent alone is
#: worth 0.15 against a semantic weight of 0.6 (see :class:`RoutingWeights`):
#: a tool a quarter of a cosine point behind the leader can still finish first
#: — but only if it was fetched. ``limit * 4`` alone gave a ``limit=3`` query a
#: pool of twelve, and the project's own 300-tool demo lost ``calculate_mean``
#: at cosine rank 13 to three random-number tools that embedded closer. Thirty-
#: two covers the cosine spread a few-hundred-tool catalogue shows in practice;
#: fetching them costs a top-k over the same vectors either way.
MIN_CANDIDATE_POOL = 32

#: How many times :meth:`SemanticRouter.route` may search again for a wider
#: pool when rows that are not tools of their own (server and agent entries,
#: pieces of long descriptions) crowded tools out of the first one. It stops
#: sooner, as soon as the pool is full or the store has no more rows.
_MAX_POOL_WIDENINGS = 6

#: The widest that search may get. Every row short of it that is not a tool is
#: an MCP server's or A2A agent's entry or a piece of a long description, so
#: reaching it takes thousands of those ranked ahead of the tools asked for.
_MAX_POOL_ROWS = 4096


def _distinct_tools(candidates: Sequence[Any]) -> int:
    """How many tools a page of search results amounts to.

    A piece of a long description counts as the tool it was cut from, once
    however many of its pieces are there; an MCP server's or A2A agent's entry
    counts as none.
    """
    tools: set[tuple[str, str]] = set()
    for candidate in candidates:
        entry = candidate[0]
        parent = chunk_parent(entry)
        if parent is not None:
            tools.add(parent)
        elif not is_reserved_namespace(entry.namespace):
            tools.add((entry.namespace, entry.name))
    return len(tools)


@dataclass
class RoutingWeights:
    """Weights for combining routing signals."""

    semantic: float = 0.6
    intent: float = 0.15
    conversation: float = 0.1
    health: float = 0.1
    cost: float = 0.05


@dataclass
class RoutingResult:
    """Routing outcome with timing metadata."""

    tools: list[tuple[ToolDefinition, float]]
    query_embedding_time_ms: float
    vector_search_time_ms: float
    rerank_time_ms: float | None
    candidate_count: int
    filtered_count: int


def compute_final_score(signals: RoutingSignals, weights: RoutingWeights) -> float:
    """
    Compute the final score for a tool.

    Args:
        signals: The routing signals for the tool
        weights: The weights for each signal

    Returns:
        The final composite score
    """
    base_score = (
        signals.semantic_similarity * weights.semantic
        + signals.intent_match * weights.intent
        + signals.conversation_relevance * weights.conversation
        + signals.health_score * weights.health
        + signals.cost_score * weights.cost
    )
    penalties = (
        signals.already_used_penalty + signals.already_failed_penalty + signals.deprecated_penalty
    )
    return max(0.0, base_score - penalties)


async def classify_intent(
    query: str,
    conversation_summary: str | None = None,
    use_llm: bool = False,
    llm_client: LLMClient | None = None,
) -> TaskIntent:
    """
    Classify the intent of a query.

    Args:
        query: The user's query
        conversation_summary: Optional conversation context
        use_llm: Whether to use LLM for classification
        llm_client: Optional LLM client for classification

    Returns:
        The classified intent
    """
    query_lower = query.lower()

    enriched_query = query_lower
    if conversation_summary:
        enriched_query = f"{enriched_query} {conversation_summary.lower()}"

    # First try keyword-based classification
    best_intent = None
    best_score = 0

    for intent, patterns in _INTENT_KEYWORD_PATTERNS.items():
        count = sum(1 for pattern in patterns if pattern.search(enriched_query))

        if count > best_score:
            best_score = count
            best_intent = intent

    if best_score > 0 and best_intent is not None:
        return best_intent

    # Fall back to LLM-based classification if enabled and available
    if use_llm and llm_client:
        available_intents = [intent.value for intent in TaskIntent]
        try:
            result = await llm_client.classify_intent(
                query=query,
                conversation_summary=conversation_summary,
                available_intents=available_intents,
            )
            # Convert string result to TaskIntent enum
            for intent in TaskIntent:
                if intent.value == result:
                    return intent
        except Exception as e:
            # Fall back to UNKNOWN if LLM classification fails
            _logger.warning(f"LLM intent classification failed, falling back to UNKNOWN: {e}")

    return TaskIntent.UNKNOWN


class SemanticRouter:
    """
    Semantic router for intelligent tool selection.

    Uses:
    - Vector similarity search
    - Intent classification
    - Conversation context
    - Tool health metrics
    - MMR diversity
    """

    def __init__(
        self,
        vector_store: VectorStoreAdapter,
        embedder: EmbeddingAdapter,
        reranker: RerankerAdapter | None = None,
        weights: RoutingWeights | None = None,
        llm_client: LLMClient | None = None,
        use_llm_for_intent: bool = False,
    ) -> None:
        """
        Initialize the semantic router.

        Args:
            vector_store: Vector store for tool embeddings
            embedder: Embedding model for queries
            reranker: Optional reranker for precision
            weights: Routing signal weights
            llm_client: Optional LLM client for intent classification
            use_llm_for_intent: Whether to use LLM for intent classification
        """
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._weights = weights or RoutingWeights()
        self._llm_client = llm_client
        self._use_llm_for_intent = use_llm_for_intent

    async def route(
        self,
        query: ToolQuery,
    ) -> RoutingResult:
        """
        Route a query to the most relevant tools.

        Args:
            query: The tool query with context

        Returns:
            Routing result with scores and timings
        """
        scoped = query.mcp_servers is not None or query.a2a_agents is not None
        if scoped and not query.mcp_servers and not query.a2a_agents:
            # A scope that names nothing matches nothing. Stores read an empty
            # tag filter as no filter at all, so the search would have run in
            # full only for every candidate to be rejected afterwards.
            return RoutingResult(
                tools=[],
                query_embedding_time_ms=0.0,
                vector_search_time_ms=0.0,
                rerank_time_ms=None,
                candidate_count=0,
                filtered_count=0,
            )

        # Intent resolution is independent of embedding + vector search, so run
        # it concurrently. With keyword classification this is near-free; with
        # LLM classification it removes an entire LLM round-trip from the
        # critical path.
        intent_task = asyncio.ensure_future(self._resolve_intent(query))

        try:
            embed_start = perf_counter()
            # embed_query, not embed_text: an asymmetric embedder wants a
            # different instruction on the thing being searched for than on
            # the thing stored. The protocol default forwards to embed_text,
            # so a symmetric embedder is unaffected.
            query_embedding = await embed_query(self._embedder, query.context.query)
            query_embedding_time_ms = (perf_counter() - embed_start) * 1000

            filters: dict[str, list[str]] | None = None
            if query.namespaces:
                # The pieces of a long description live in a namespace of
                # their own per parent namespace; without those here a
                # namespace filter would hide them, and the tool would again
                # be findable only by the opening of its text.
                filters = {
                    "namespace": [
                        *query.namespaces,
                        *(chunk_namespace(namespace) for namespace in query.namespaces),
                    ]
                }
            if query.mcp_servers is not None or query.a2a_agents is not None:
                # Every discovered tool carries its server or agent as a tag,
                # and every store filters on tags (any-of), so the scope costs
                # no protocol change.
                filters = {
                    **(filters or {}),
                    "tags": [
                        *(mcp_server_tag(server) for server in query.mcp_servers or ()),
                        *(a2a_agent_tag(agent) for agent in query.a2a_agents or ()),
                    ],
                }

            # Request embeddings if MMR will be used (optimization to avoid re-embedding)
            include_embeddings = query.diversity_factor > 0

            search_start = perf_counter()
            # Wide enough for re-scoring to change the answer; see
            # MIN_CANDIDATE_POOL for why ``limit * 4`` alone was not.
            pool = requested = max(query.limit * 4, MIN_CANDIDATE_POOL)
            for _ in range(_MAX_POOL_WIDENINGS + 1):
                candidates = await self._vector_store.search(
                    query_vector=query_embedding,
                    limit=requested,
                    filters=filters,
                    score_threshold=query.score_threshold,
                    include_embeddings=include_embeddings,
                )
                # Not every row is a tool of its own. MCP servers and A2A
                # agents are indexed beside the tools as entries that can
                # never be returned, and a long description is indexed in
                # pieces, several of which may be one tool's. What the pool
                # needs is that many tools, so count those.
                found = _distinct_tools(candidates)
                if (
                    found >= pool
                    or len(candidates) < requested  # the store has no more to give
                    or requested >= _MAX_POOL_ROWS
                ):
                    break
                # Ask again for as many rows as the pool would take at the
                # rate tools have turned up so far, and four times as many
                # when none has yet. No more than eight times wider in one
                # step: one tool among thirty-one other rows says little about
                # the rows after them.
                wanted = math.ceil(requested * pool / found) if found else requested * 4
                requested = min(wanted, requested * 8, _MAX_POOL_ROWS)
            candidates = await self._fold_chunks(candidates, query)
            vector_search_time_ms = (perf_counter() - search_start) * 1000
        except BaseException:
            # Cancel AND await the intent task so it never outlives route()
            # as a pending task ("Task was destroyed but it is pending").
            intent_task.cancel()
            try:
                await intent_task
            except (asyncio.CancelledError, Exception):
                pass
            raise

        intent = await intent_task

        scored_tools, tool_embeddings = self._score_and_filter_candidates(
            candidates=list(candidates),
            query=query,
            intent=intent,
            include_embeddings=include_embeddings,
        )

        scored_tools.sort(key=lambda x: x[1], reverse=True)

        rerank_time_ms: float | None = None
        if self._reranker and query.enable_reranking:
            rerank_start = perf_counter()
            scored_tools = await self._reranker.rerank(
                query.context.query,
                scored_tools,
                query.limit,
            )
            rerank_time_ms = (perf_counter() - rerank_start) * 1000

        if query.diversity_factor > 0 and len(scored_tools) > 1:
            scored_tools = await self._apply_mmr(
                scored_tools, query.diversity_factor, query.limit, tool_embeddings
            )

        final_tools = scored_tools[: query.limit]
        return RoutingResult(
            tools=final_tools,
            query_embedding_time_ms=query_embedding_time_ms,
            vector_search_time_ms=vector_search_time_ms,
            rerank_time_ms=rerank_time_ms,
            candidate_count=len(candidates),
            filtered_count=len(scored_tools),
        )

    async def _fold_chunks(self, candidates: Sequence[Any], query: ToolQuery) -> list[Any]:
        """Credit each chunk hit to the tool it was cut from.

        A long description is indexed in pieces beside the tool's own entry
        (see :mod:`agent_gantry.core.chunks`). A tool's semantic score becomes
        the best of its own and its pieces', a piece counting for
        ``CHUNK_SCORE_WEIGHT`` of its score, and a tool that was found only
        through a piece, because the matching sentence sits past what one
        embedding carries, is fetched and joins the candidates.
        """
        best: dict[tuple[str, str], float] = {}
        folded: list[Any] = []
        for candidate in candidates:
            parent = chunk_parent(candidate[0])
            if parent is None:
                folded.append(candidate)
            elif not query.namespaces or parent[0] in query.namespaces:
                best[parent] = max(best.get(parent, 0.0), float(candidate[1]) * CHUNK_SCORE_WEIGHT)
        if not best:
            return folded

        for index, candidate in enumerate(folded):
            tool = candidate[0]
            score = best.pop((tool.namespace, tool.name), None)
            if score is not None and score > candidate[1]:
                folded[index] = (tool, score, *candidate[2:])
        # Together, not one after another: on a remote store each is a round trip.
        parents = list(best.items())
        fetched = await asyncio.gather(
            *(self._vector_store.get_by_name(name, namespace) for (namespace, name), _ in parents),
            return_exceptions=True,
        )
        for ((namespace, name), score), tool in zip(parents, fetched):
            if isinstance(tool, BaseException):
                # The search itself worked, so this is an anomaly worth seeing:
                # the tool is left out, and silently so would be a quiet loss.
                _logger.warning(
                    f"Could not fetch tool '{namespace}.{name}', matched through a chunk "
                    f"of its description; leaving it out of this result: {tool}"
                )
            elif isinstance(tool, ToolDefinition):
                # No stored embedding to hand: MMR embeds it again if it needs to.
                folded.append((tool, score))
        return folded

    def _score_and_filter_candidates(
        self,
        candidates: list[Any],
        query: ToolQuery,
        intent: TaskIntent,
        include_embeddings: bool,
    ) -> tuple[list[tuple[ToolDefinition, float]], dict[str, list[float]]]:
        """Score and filter candidates from vector search."""
        tool_embeddings: dict[str, list[float]] = {}
        scored_tools: list[tuple[ToolDefinition, float]] = []

        req_caps = set(query.required_capabilities) if query.required_capabilities else None
        exc_caps = set(query.excluded_capabilities) if query.excluded_capabilities else None

        # Normalize conversation context once per query instead of once per
        # candidate — lowering the summary and every recent message inside the
        # scoring loop is O(candidates × context size).
        ctx = _QueryContextCache(
            summary_lower=(query.context.conversation_summary or "").lower(),
            messages_lower=[m.get("content", "").lower() for m in query.context.recent_messages],
            tools_used=set(query.context.tools_already_used),
            tools_failed=set(query.context.tools_failed),
        )

        for candidate in candidates:
            # ``(tool, score)`` or ``(tool, score, embedding)``. A tool brought
            # in through one of its chunks has no stored embedding even when
            # embeddings were asked for.
            tool, semantic_score, *rest = candidate
            if include_embeddings and rest:
                tool_embeddings[f"{tool.namespace}.{tool.name}"] = rest[0]

            if not self._is_tool_allowed(tool, query, req_caps, exc_caps):
                continue

            signals = self._compute_signals(
                tool=tool,
                semantic_score=semantic_score,
                intent=intent,
                ctx=ctx,
            )
            final_score = compute_final_score(signals, self._weights)
            scored_tools.append((tool, final_score))

        return scored_tools, tool_embeddings

    def filter_tools(
        self,
        tools: Sequence[ToolDefinition],
        query: ToolQuery,
    ) -> list[ToolDefinition]:
        """Apply a query's hard constraints to a list of tools.

        The same rules :meth:`route` applies to vector-search candidates —
        deprecation, namespaces, capabilities, sources and circuit-breaker
        health. Public so a selector, which never sees a vector-search
        candidate, filters by exactly these rules rather than a second copy of
        them that could drift: a selector surfacing a deprecated or unhealthy
        tool the semantic path hides would be a silent widening of what the
        agent can reach.

        Args:
            tools: Tools to filter.
            query: The query whose constraints apply.

        Returns:
            The tools the query permits, in their incoming order.
        """
        req_caps = set(query.required_capabilities) if query.required_capabilities else None
        exc_caps = set(query.excluded_capabilities) if query.excluded_capabilities else None
        return [
            tool for tool in tools if self._is_tool_allowed(tool, query, req_caps, exc_caps)
        ]

    def _is_tool_allowed(
        self,
        tool: ToolDefinition,
        query: ToolQuery,
        req_caps: set[str] | None = None,
        exc_caps: set[str] | None = None,
    ) -> bool:
        """Filter candidates based on query constraints."""
        # Index bookkeeping, not a tool: the pseudo-entry an MCP server is
        # embedded as shares this index, and used to be handed back as a tool
        # that no executor can run whenever it out-scored the real ones.
        if is_reserved_namespace(tool.namespace):
            return False
        if query.exclude_deprecated and tool.deprecated:
            return False
        if query.namespaces and tool.namespace not in query.namespaces:
            return False

        if req_caps or exc_caps:
            tool_caps = set(tool.capabilities)
            if req_caps and not req_caps.issubset(tool_caps):
                return False
            if exc_caps and not exc_caps.isdisjoint(tool_caps):
                return False

        if query.sources and tool.source not in query.sources:
            return False
        # The tag does the pushdown; the metadata is the authority. A tag is
        # free text anyone can put on a tool, and this filter must not be how a
        # hand-tagged native tool gets returned as a server's.
        if query.mcp_servers is not None or query.a2a_agents is not None:
            from_server = tool.metadata.get("mcp_server") in (query.mcp_servers or ())
            from_agent = tool.metadata.get("a2a_agent") in (query.a2a_agents or ())
            if not (from_server or from_agent):
                return False
        if query.exclude_unhealthy and tool.health.circuit_breaker_open:
            return False
        return True

    async def _resolve_intent(self, query: ToolQuery) -> TaskIntent:
        """Resolve intent using context override or classification."""
        if query.context.inferred_intent:
            try:
                return TaskIntent(query.context.inferred_intent)
            except ValueError:
                pass
        return await classify_intent(
            query.context.query,
            query.context.conversation_summary,
            use_llm=self._use_llm_for_intent,
            llm_client=self._llm_client,
        )

    def _compute_signals(
        self,
        tool: ToolDefinition,
        semantic_score: float,
        intent: TaskIntent,
        ctx: _QueryContextCache,
    ) -> RoutingSignals:
        """Compute routing signals for a tool."""
        # Intent match
        intent_match = 0.0
        if intent != TaskIntent.UNKNOWN:
            pattern = INTENT_TAG_PATTERNS.get(intent)
            if pattern:
                intent_match = intent_match_score(tool, pattern)

        # Conversation relevance
        tool_name_lower = tool.name.lower()
        qualified_name = f"{tool.namespace}.{tool.name}"
        was_used = tool.name in ctx.tools_used or qualified_name in ctx.tools_used
        previously_failed = tool.name in ctx.tools_failed or qualified_name in ctx.tools_failed
        conversation_relevance = 0.0
        if was_used:
            conversation_relevance = 0.5
        if ctx.summary_lower and self._contains_token(ctx.summary_lower, tool_name_lower):
            conversation_relevance = min(1.0, conversation_relevance + 0.2)
        for content in ctx.messages_lower:
            if content and self._contains_token(content, tool_name_lower):
                conversation_relevance = min(1.0, conversation_relevance + 0.2)

        # Health score
        health_score = 0.0 if tool.health.circuit_breaker_open else tool.health.success_rate

        # Cost score (inverse - lower cost is better)
        cost_score = 1.0 - min(tool.cost.estimated_latency_ms / 10000, 1.0)

        # Penalties
        already_used_penalty = ALREADY_USED_PENALTY if was_used else 0.0
        already_failed_penalty = ALREADY_FAILED_PENALTY if previously_failed else 0.0
        deprecated_penalty = DEPRECATED_PENALTY if tool.deprecated else 0.0

        return RoutingSignals(
            semantic_similarity=semantic_score,
            intent_match=intent_match,
            conversation_relevance=conversation_relevance,
            health_score=health_score,
            cost_score=cost_score,
            already_used_penalty=already_used_penalty,
            already_failed_penalty=already_failed_penalty,
            deprecated_penalty=deprecated_penalty,
        )

    async def _apply_mmr(
        self,
        scored_tools: list[tuple[ToolDefinition, float]],
        diversity_factor: float,
        limit: int,
        cached_embeddings: dict[str, list[float]] | None = None,
    ) -> list[tuple[ToolDefinition, float]]:
        """Apply MMR to encourage diversity in selected tools."""
        if not scored_tools:
            return []

        relevance_scores = [score for _, score in scored_tools]

        # Embeddings the store returned with the candidates are reused; only
        # tools missing from that cache are embedded again, in one batch.
        cached = cached_embeddings or {}
        embeddings: list[list[float]] = [
            cached.get(f"{tool.namespace}.{tool.name}", []) for tool, _ in scored_tools
        ]
        missing = [i for i, emb in enumerate(embeddings) if len(emb) == 0]
        if missing:
            fresh = await self._embedder.embed_batch(
                [scored_tools[i][0].to_searchable_text() for i in missing]
            )
            for idx, emb in zip(missing, fresh):
                embeddings[idx] = emb

        selected = mmr_order(relevance_scores, embeddings, diversity_factor, limit)
        return [scored_tools[i] for i in selected]

    def _contains_token(self, text: str, token: str) -> bool:
        """Return True if token appears as a standalone word in text."""
        if not token:
            return False
        if token not in text:
            return False
        return _get_token_pattern(token).search(text) is not None
