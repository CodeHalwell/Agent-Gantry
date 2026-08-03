"""
Semantic router for Agent-Gantry.

Intelligent tool selection using semantic search, intent classification, and context.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.llm_client import LLMClient
    from agent_gantry.adapters.rerankers.base import RerankerAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.schema.query import ToolQuery
    from agent_gantry.schema.tool import ToolDefinition


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


# Pre-compile regex for intent matching to optimize routing inner loop
# Benchmark: ~40% faster than generator expression (any(kw in text))
# Note: we omit word boundaries to exactly preserve original substring matching behavior
INTENT_TAG_PATTERNS: dict[TaskIntent, re.Pattern[str]] = {
    intent: re.compile("|".join(re.escape(kw) for kw in keywords))
    for intent, keywords in INTENT_TAG_MAPPING.items()
}


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

    for intent, keywords in INTENT_TAG_MAPPING.items():
        count = 0
        for kw in keywords:
            if kw in enriched_query:
                count += 1

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
        embed_start = perf_counter()
        query_embedding = await self._embedder.embed_text(query.context.query)
        query_embedding_time_ms = (perf_counter() - embed_start) * 1000

        filters: dict[str, list[str]] | None = None
        if query.namespaces:
            filters = {"namespace": query.namespaces}

        # Request embeddings if MMR will be used (optimization to avoid re-embedding)
        include_embeddings = query.diversity_factor > 0

        search_start = perf_counter()
        candidates = await self._vector_store.search(
            query_vector=query_embedding,
            limit=query.limit * 4,
            filters=filters,
            score_threshold=query.score_threshold,
            include_embeddings=include_embeddings,
        )
        vector_search_time_ms = (perf_counter() - search_start) * 1000

        intent = await self._resolve_intent(query)

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
                scored_tools,
                query_embedding,
                query.diversity_factor,
                query.limit,
                tool_embeddings,
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

        for candidate in candidates:
            if include_embeddings:
                tool, semantic_score, embedding = candidate  # type: ignore
                tool_key = f"{tool.namespace}.{tool.name}"
                tool_embeddings[tool_key] = embedding
            else:
                tool, semantic_score = candidate  # type: ignore

            if not self._is_tool_allowed(tool, query, req_caps, exc_caps):
                continue

            signals = self._compute_signals(
                tool=tool,
                semantic_score=semantic_score,
                intent=intent,
                query=query,
            )
            final_score = compute_final_score(signals, self._weights)
            scored_tools.append((tool, final_score))

        return scored_tools, tool_embeddings

    def _is_tool_allowed(
        self,
        tool: ToolDefinition,
        query: ToolQuery,
        req_caps: set[str] | None = None,
        exc_caps: set[str] | None = None,
    ) -> bool:
        """Filter candidates based on query constraints."""
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
        query: ToolQuery,
    ) -> RoutingSignals:
        """Compute routing signals for a tool."""
        # Intent match
        intent_match = 0.0
        if intent != TaskIntent.UNKNOWN:
            pattern = INTENT_TAG_PATTERNS.get(intent)
            if pattern:
                tool_text = (
                    f"{tool.name.lower()} {tool.description.lower()} {' '.join(tool.tags).lower()}"
                )
                if pattern.search(tool_text):
                    intent_match = 1.0

        # Conversation relevance
        conversation_relevance = 0.0
        if tool.name in query.context.tools_already_used:
            conversation_relevance = 0.5
        summary = (query.context.conversation_summary or "").lower()
        if summary and self._contains_token(summary, tool.name.lower()):
            conversation_relevance = min(1.0, conversation_relevance + 0.2)
        for message in query.context.recent_messages:
            content = message.get("content", "").lower()
            if content and self._contains_token(content, tool.name.lower()):
                conversation_relevance = min(1.0, conversation_relevance + 0.2)

        # Health score
        health_score = 0.0 if tool.health.circuit_breaker_open else tool.health.success_rate

        # Cost score (inverse - lower cost is better)
        cost_score = 1.0 - min(tool.cost.estimated_latency_ms / 10000, 1.0)

        # Penalties
        already_used_penalty = 0.1 if tool.name in query.context.tools_already_used else 0.0
        already_failed_penalty = 0.3 if tool.name in query.context.tools_failed else 0.0
        deprecated_penalty = 0.5 if tool.deprecated else 0.0

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
        query_embedding: list[float],
        diversity_factor: float,
        limit: int,
        cached_embeddings: dict[str, list[float]] | None = None,
    ) -> list[tuple[ToolDefinition, float]]:
        """Apply MMR to encourage diversity in selected tools."""
        if not scored_tools:
            return []

        _ = query_embedding  # kept for signature consistency; not used in relevance scoring
        lambda_param = 1.0 - diversity_factor
        relevance_scores = [score for _, score in scored_tools]

        # Use cached embeddings if available, otherwise re-embed (fallback)
        embeddings: list[list[float]] = []
        if cached_embeddings:
            # Need to collect embeddings in order of scored_tools
            missing_indices: list[int] = []
            missing_texts: list[str] = []

            # Initialize with placeholders
            embeddings = [[] for _ in scored_tools]

            for i, (tool, _) in enumerate(scored_tools):
                tool_key = f"{tool.namespace}.{tool.name}"
                embedding = cached_embeddings.get(tool_key)
                if embedding is not None:
                    embeddings[i] = embedding
                else:
                    missing_indices.append(i)
                    missing_texts.append(tool.to_searchable_text())

            # Batch embed missing
            if missing_texts:
                new_embeddings = await self._embedder.embed_batch(missing_texts)
                for idx, emb in zip(missing_indices, new_embeddings):
                    embeddings[idx] = emb
        else:
            # Fallback: re-embed all tools (old behavior for backward compatibility)
            tool_texts = [tool.to_searchable_text() for tool, _ in scored_tools]
            embeddings = await self._embedder.embed_batch(tool_texts)

        # Convert to numpy matrix for vectorized cosine similarity
        emb_matrix = np.array(embeddings, dtype=np.float64)
        norms = np.linalg.norm(emb_matrix, axis=1)

        # Pre-normalize embeddings to simplify cosine similarity to dot product
        with np.errstate(divide="ignore", invalid="ignore"):
            normed_emb = emb_matrix / norms[:, np.newaxis]
            normed_emb[np.isnan(normed_emb)] = 0.0

        selected: list[int] = []
        candidates = list(range(len(scored_tools)))

        # ⚡ Bolt: Use __getitem__ instead of lambda in max() to avoid Python function
        # overhead for every item. This runs nearly 2x faster while remaining clean.
        first_idx = max(candidates, key=relevance_scores.__getitem__)
        selected.append(first_idx)
        candidates.remove(first_idx)

        # Track maximum similarity to any selected item for each candidate
        max_sims = {idx: 0.0 for idx in candidates}

        while candidates and len(selected) < limit:
            last_selected = selected[-1]
            last_emb = normed_emb[last_selected]

            # Vectorized dot product of all remaining candidates with the last selected item
            all_sims = np.dot(normed_emb, last_emb)

            best_idx = -1
            best_score = -float("inf")

            for idx in candidates:
                sim = float(all_sims[idx])
                if sim > max_sims[idx]:
                    max_sims[idx] = sim

                score = lambda_param * relevance_scores[idx] - (1.0 - lambda_param) * max_sims[idx]
                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected.append(best_idx)
            candidates.remove(best_idx)

        return [scored_tools[i] for i in selected]

    def _contains_token(self, text: str, token: str) -> bool:
        """Return True if token appears as a standalone word in text."""
        if not token:
            return False
        if token not in text:
            return False
        return _get_token_pattern(token).search(text) is not None
