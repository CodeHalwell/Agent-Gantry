"""
Query and context models for dotfetch.

Models for conversation context, tool queries, and retrieval results.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dotfetch.schema.base import reject_newlines
from dotfetch.schema.tool import ToolCapability, ToolDefinition, ToolSource


class ConversationContext(BaseModel):
    """
    Conversation state for context-aware routing.

    Key idea: routing isn't just about the current query; we use:
    - conversation summary
    - recent messages
    - which tools have been tried / failed
    - user capabilities
    """

    # Immediate request
    query: str = Field(...)

    # Coarse history
    conversation_summary: str | None = Field(default=None)
    recent_messages: list[dict[str, str]] = Field(default_factory=list, max_length=10)

    # Tool usage in this conversation
    tools_already_used: list[str] = Field(default_factory=list)
    tools_failed: list[str] = Field(default_factory=list)

    # Optional high-level intent classification
    inferred_intent: str | None = Field(default=None)

    # Permission context
    user_capabilities: list[ToolCapability] = Field(
        default_factory=lambda: [cap for cap in ToolCapability]
    )


class ToolQuery(BaseModel):
    """Request to find relevant tools."""

    context: ConversationContext

    limit: int = Field(default=5, ge=1, le=50)
    # NOTE: this 0.5 default is intentionally *not* mirrored by the
    # higher-level convenience APIs (``dotfetch.integrations.frameworks``
    # adapters, ``with_semantic_tools`` / ``SemanticToolSelector`` in
    # ``dotfetch.integrations.semantic_tools``). Those all default to
    # ``0.0`` (no filtering) because a flat absolute-similarity cutoff is a
    # silent-drop trap once queries get longer or more specific — long queries
    # dilute absolute similarity scores, so a non-zero default can silently
    # return zero tools with no error. Callers constructing a ``ToolQuery``
    # directly should set ``score_threshold`` explicitly rather than relying on
    # this default; the convenience layers set it to ``0.0`` for exactly this
    # reason. Kept at 0.5 here for backward compatibility with existing direct
    # ``ToolQuery`` callers — do not change this without a migration plan.
    #
    # This field is a *cosine* cutoff and applies only on the semantic path.
    # When a selector is configured (``DotFetchConfig.selector``) retrieval
    # is a probability, not a cosine, and this field is ignored there; use
    # ``selection_threshold`` below for that path. The two used to share this
    # one field, which silently reinterpreted the 0.5 default as a minimum
    # probability for direct callers (#428).
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Minimum selector probability for a tool to be returned when a selector
    # is configured. ``None`` (the default) applies no per-query cutoff beyond
    # the selector's own ``SelectorConfig.threshold``, which the selector has
    # already enforced. Set it to tighten a single query on top of that. It
    # has no effect on the semantic path, where ``score_threshold`` applies.
    selection_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    # Filters
    namespaces: list[str] | None = None
    required_capabilities: list[ToolCapability] | None = None
    excluded_capabilities: list[ToolCapability] | None = None
    sources: list[ToolSource] | None = None
    #: Search only within the tools discovered from these MCP servers, by
    #: server name. Unlike ``sources``, which can only say "any MCP server"
    #: and is applied after the candidate pool is drawn, this is pushed down
    #: to the vector store, so a server holding a few of a large registry's
    #: tools is searched in full rather than through whatever part of it made
    #: the pool. ``None`` is no scope; an empty list names no server, so it
    #: matches nothing and no search is made.
    mcp_servers: list[str] | None = None
    #: Search only within the tools converted from these A2A agents' skills,
    #: by agent name. Pushed down like ``mcp_servers``, with the same meaning
    #: for ``None`` and an empty list; naming both searches the tools of
    #: either.
    a2a_agents: list[str] | None = None
    exclude_deprecated: bool = True
    exclude_unhealthy: bool = True

    # Advanced
    # ``None`` means "defer to the fetcher's reranker config": ``retrieve()``
    # turns it on when a reranker is configured. Set it explicitly to
    # ``True``/``False`` to force the behaviour regardless of config.
    enable_reranking: bool | None = None
    diversity_factor: float = Field(default=0.0, ge=0.0, le=1.0)


class ScoredTool(BaseModel):
    """A tool with its relevance scores."""

    tool: ToolDefinition
    semantic_score: float = Field(ge=0.0, le=1.0)
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        """The rerank score when one was computed, else the semantic score."""
        return self.rerank_score if self.rerank_score is not None else self.semantic_score


class RetrievalResult(BaseModel):
    """Result of a tool retrieval operation."""

    tools: list[ScoredTool]

    query_embedding_time_ms: float
    vector_search_time_ms: float
    rerank_time_ms: float | None = None
    #: Time spent in a selector, when one answered instead of the semantic
    #: router. ``None`` on the semantic path; the two embedding/search timings
    #: are then 0.0, because neither step ran.
    selection_time_ms: float | None = None
    total_time_ms: float

    candidate_count: int
    filtered_count: int

    trace_id: str

    #: Which backend produced ``tools``: ``"selector"`` when the configured
    #: selector answered, ``"semantic"`` when the router did. With a selector
    #: configured, ``"semantic"`` means it was not the one that answered, and
    #: the two fields below say why.
    backend: Literal["semantic", "selector"] = "semantic"
    #: ``True`` when a selector is configured but this result came from the
    #: semantic router instead: the selector failed, was over its candidate
    #: ceiling, or had nothing to work with. Always ``False`` when no selector
    #: is configured, and when the selector answered -- including when it
    #: answered with no tools, which is a decline, not a fallback.
    selector_fallback: bool = False
    #: Why the selector did not answer, when ``selector_fallback`` is set.
    selector_fallback_reason: str | None = None
    #: ``model_name`` of the embedder that searched for ``tools``, e.g.
    #: ``"all-MiniLM-L6-v2"``. ``"simple-hash-embedder"`` means the hash-based
    #: test embedder answered, which matches tokens rather than meaning: the
    #: usual cause is a deployment without sentence-transformers installed.
    #: ``None`` when a selector answered, because no embedding was made.
    embedder: str | None = None

    model_config = ConfigDict(validate_assignment=True)

    _reject_newline_identifiers = field_validator("trace_id")(reject_newlines)

    def to_dialect(self, dialect: str = "auto", **options: Any) -> list[dict[str, Any]]:
        """
        Convert retrieved tools to provider-specific format.

        Uses the dialect registry for extensible provider support.

        Args:
            dialect: Target dialect/provider name (default: 'auto')
            **options: Provider-specific options (e.g., strict mode for OpenAI)

        Returns:
            List of provider-specific tool schemas
        """
        return [t.tool.to_dialect(dialect, **options) for t in self.tools]


class ContextEntry(BaseModel):
    """One entry of a :class:`ContextResult`: a tool, a skill, an MCP server or an A2A agent.

    Attributes:
        kind: Which catalogue the entry came from.
        id: ``namespace.name``.
        rank: Its place among the entries of its own kind, from 1.
        score: Its score on that kind's own scale, when the path that
            retrieved it reports one. Scores of different kinds are not
            comparable: a tool's blends several signals or is a selector's
            probability, a skill's is a cosine.
        entry: The ``ToolDefinition``, ``Skill``, ``MCPServerDefinition`` or
            ``A2AAgentDefinition``.
    """

    kind: Literal["tool", "skill", "mcp_server", "a2a_agent"]
    id: str
    rank: int = Field(ge=1)
    score: float | None = None
    entry: Any


class ContextResult(BaseModel):
    """Tools, skills and MCP servers for one query, in one ranked view.

    ``entries`` is ordered by rank within each kind, not by score: every
    kind's best match comes before any kind's second, and so on. The three
    catalogues are scored on different scales, so their scores cannot be
    merged without calibration; their ranks can. Within one rank the order is
    tools, then skills, then MCP servers, then A2A agents.

    Attributes:
        entries: The merged view.
        tool_retrieval: The full result of the tool search, with its backend,
            embedder and selector provenance. ``None`` if no tools were asked
            for.
        skipped: Kinds that were asked for and could not be searched, with
            the reason: for example MCP support not being installed. A kind
            that was searched and matched nothing is not here.
        total_time_ms: Wall-clock time for the whole call; the searches run
            concurrently.
    """

    entries: list[ContextEntry]
    tool_retrieval: RetrievalResult | None = None
    skipped: dict[str, str] = Field(default_factory=dict)
    total_time_ms: float = 0.0

    def of_kind(self, kind: str) -> list[Any]:
        """The underlying objects of one kind, best first."""
        return [e.entry for e in sorted(self.entries, key=lambda e: e.rank) if e.kind == kind]

    @property
    def tools(self) -> list[ToolDefinition]:
        """The tools, best first."""
        return self.of_kind("tool")

    @property
    def skills(self) -> list[Any]:
        """The skills, best first."""
        return self.of_kind("skill")

    @property
    def mcp_servers(self) -> list[Any]:
        """The MCP servers, best first."""
        return self.of_kind("mcp_server")

    @property
    def a2a_agents(self) -> list[Any]:
        """The A2A agents, best first."""
        return self.of_kind("a2a_agent")
