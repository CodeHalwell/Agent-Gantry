"""
Semantic router for Agent-Gantry.

Intelligent tool selection using semantic search, intent classification, and context.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_gantry.adapters.embedders.base import EmbeddingAdapter
    from agent_gantry.adapters.rerankers.base import RerankerAdapter
    from agent_gantry.adapters.vector_stores.base import VectorStoreAdapter
    from agent_gantry.schema.query import ToolQuery
    from agent_gantry.schema.tool import ToolDefinition


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
    TaskIntent.COMMUNICATION: ["email", "message", "notify", "slack", "send"],
    TaskIntent.FILE_OPERATIONS: ["file", "upload", "download", "convert", "export"],
    TaskIntent.CUSTOMER_SUPPORT: ["ticket", "refund", "support", "customer"],
    TaskIntent.ADMIN: ["user", "permission", "setting", "config", "admin"],
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
        signals.already_used_penalty
        + signals.already_failed_penalty
        + signals.deprecated_penalty
    )
    return max(0.0, base_score - penalties)


async def classify_intent(
    query: str,
    conversation_summary: str | None = None,
    use_llm: bool = False,
) -> TaskIntent:
    """
    Classify the intent of a query.

    Args:
        query: The user's query
        conversation_summary: Optional conversation context
        use_llm: Whether to use LLM for classification

    Returns:
        The classified intent
    """
    query_lower = query.lower()
    scores: dict[TaskIntent, int] = {}

    for intent, keywords in INTENT_TAG_MAPPING.items():
        scores[intent] = sum(1 for kw in keywords if kw in query_lower)

    if max(scores.values()) > 0:
        return max(scores, key=lambda k: scores[k])

    if use_llm and conversation_summary:
        # TODO: Implement LLM-based classification
        raise NotImplementedError("LLM-based intent classification is not implemented yet.")

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
    ) -> None:
        """
        Initialize the semantic router.

        Args:
            vector_store: Vector store for tool embeddings
            embedder: Embedding model for queries
            reranker: Optional reranker for precision
            weights: Routing signal weights
        """
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._weights = weights or RoutingWeights()

    async def route(
        self,
        query: ToolQuery,
    ) -> list[tuple[ToolDefinition, float]]:
        """
        Route a query to the most relevant tools.

        Args:
            query: The tool query with context

        Returns:
            List of (tool, score) tuples
        """
        # 1. Embed the query
        query_embedding = await self._embedder.embed_text(query.context.query)

        # 2. Vector search
        candidates = await self._vector_store.search(
            query_vector=query_embedding,
            limit=query.limit * 4,  # Get more for filtering
            score_threshold=query.score_threshold,
        )

        # 3. Classify intent
        intent = await classify_intent(
            query.context.query,
            query.context.conversation_summary,
        )

        # 4. Score each candidate
        scored_tools: list[tuple[ToolDefinition, float]] = []
        for tool, semantic_score in candidates:
            signals = self._compute_signals(
                tool=tool,
                semantic_score=semantic_score,
                intent=intent,
                query=query,
            )
            final_score = compute_final_score(signals, self._weights)
            scored_tools.append((tool, final_score))

        # 5. Sort by score
        scored_tools.sort(key=lambda x: x[1], reverse=True)

        # 6. Optional reranking
        if self._reranker and query.enable_reranking:
            scored_tools = await self._reranker.rerank(
                query.context.query,
                scored_tools,
                query.limit,
            )

        # 7. Apply diversity (MMR) if requested
        if query.diversity_factor > 0:
            # TODO: Implement MMR
            pass

        return scored_tools[: query.limit]

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
            intent_keywords = INTENT_TAG_MAPPING.get(intent, [])
            if any(kw in tool.name.lower() or kw in tool.description.lower() for kw in intent_keywords):
                intent_match = 1.0

        # Conversation relevance
        conversation_relevance = 0.0
        if tool.name in query.context.tools_already_used:
            conversation_relevance = 0.5

        # Health score
        health_score = tool.health.success_rate if tool.health else 1.0

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
