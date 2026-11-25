# API reference quick-card

> Part of the `dotfetch` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

```python
from dotfetch import (
    DotFetch,                  # main facade
    DotFetchContextProvider,        # AF native provider (per-run / per-turn)
    MissingRequiredToolError,     # raised when required=[...] tool absent
    RetrievalDecision,            # introspection: what the bridge surfaced
    RetrievalCandidate,           # one row in RetrievalDecision.candidates
    with_semantic_tools,          # decorator for plain LLM SDK calls
    set_default_fetcher,           # bind a fetcher to with_semantic_tools
    create_default_fetcher,        # quick factory (auto-picks Nomic if available)
    render_result,                # stringify any tool result (incl. AF Content lists)
    enable_console_logging,       # opt into console output (lib configures no logging)
    ToolCall, ToolResult,
    ToolCallEvent,                # delivered to fetcher.on_tool_call(callback)
    current_tool_invocation,      # framework run state inside a tool handler (or None)
    ToolInvocationContext,        # .framework / .deps / .run_id / .tool_call_id / .native_context
    ToolQuery, ConversationContext,
    ToolCapability, ToolCost, ToolDefinition, ToolHealth, ToolSource,
    Skill, SkillCategory, SkillSearchResult,   # procedural memory (see "Skills")
)
# Skills: fetcher.add_skill / add_skills / add_skills_from_directory(path)
#         fetcher.retrieve_skills(query) / retrieve_skills_as_prompt(query)
# Selection (0.16.0+, needs the `system-one` extra; Jev* names are deprecated aliases):
#   DotFetch(selector=SystemOneSelector(threshold=0.4))  -> no embedder/vector store/sync
#   DotFetch(reranker=SystemOneReranker())               -> refine a semantic shortlist
#   SelectionResult.scores                            -> every candidate, not just winners
#   (await fetcher.retrieve(ToolQuery(...))).backend   -> "selector" | "semantic"
#       .selector_fallback / .selector_fallback_reason -> the router answered for the selector, and why
#   DotFetch(selector=..., on_selector_fallback="raise") -> SelectorFallbackError instead of failing open
#   also importable from dotfetch: SystemOneClient, SelectorFallbackError
#   (await fetcher.retrieve(...)).embedder -> model_name of the embedder that searched; None on the
#       selector path. "simple-hash-embedder" = sentence-transformers is missing and retrieval is
#       matching tokens, not meaning (logged once at WARNING). EmbedderConfig(on_unavailable="raise")
#       raises ImportError instead of falling back.
# A2A:    fetcher.register_a2a_agent(name, url, description=...)   -> routable without contacting it
#         await fetcher.retrieve_a2a_agents(query, limit=3) / fetcher.list_a2a_agents()
#         await fetcher.discover_skills_from_agent(name)  (or add_a2a_agent(A2AAgentConfig(...)) in one step)
#         converted skills are tagged "a2a:<agent>"; ToolQuery(a2a_agents=[...]) scopes a search to them
#   ctx = await fetcher.retrieve_context(query, tools=5, skills=2, servers=1, agents=1) -> ContextResult
#       ctx.entries (merged by rank; .kind/.id/.rank/.score/.entry), ctx.tools / .skills / .mcp_servers,
#       ctx.tool_retrieval (backend, embedder, selector_fallback), ctx.skipped
#   fetcher.retrieve_skills(q, rerank=True, diversity_factor=0.4) / retrieve_mcp_servers(q, rerank=True, ...)
#   long tool descriptions and skill bodies are indexed in pieces (RoutingConfig(chunk_long_text=False) to stop)
# Measuring retrieval: from dotfetch.evaluation import RetrievalCase, evaluate_retrieval,
#       tool_retriever, skill_retriever, mcp_server_retriever
#   report = await evaluate_retrieval(tool_retriever(fetcher), [RetrievalCase("q", {"ns.tool"}),
#       RetrievalCase("nothing fits", expect_decline=True)], k=5)
#   report.recall_at_1 / .recall_at_k / .decline_correctness / .by_kind() / .misses()
#   found = await calibrate_threshold(lambda t: tool_retriever(fetcher, score_threshold=t), cases)
#       found.threshold, found.window (None = no threshold separates answers from non-answers)
# MCP:    fetcher.serve_mcp(transport="stdio"|"http"|"sse", mode=..., expose=[...])
#         reset_sse_shutdown_latch()  -> release sse_starlette's process-global
#           shutdown latch between servers in one process; without it a second
#           EventSourceResponse is cancelled between headers and body (hangs).
#           Needs the `mcp` extra; raises AttributeError without it.
#         fetcher.add_mcp_server(MCPServerConfig(command=[...] | url="https://...", headers=...))
#         tools discovered from a server are tagged "mcp:<server>" (mcp_server_tag(name) in
#           dotfetch.schema.mcp): the server name is searchable, and a selector groups by it
#         fetcher.retrieve_tools_from_server("datadog", query, limit=5)  -> search within one server
#         big server? catalogue first, commit a subset (no embeddings/handlers until adopted):
#           specs = await fetcher.list_server_tools("datadog")          # after register_mcp_server(...)
#           await fetcher.adopt_server_tools("datadog", [t.name for t in specs if "log" in t.name])
#           discover_tools_from_server(name) == adopt_server_tools(name, None, refresh=True)
#         ToolQuery(..., mcp_servers=["datadog", "azure"]) / retrieve_tools(q, mcp_servers=[...])
#         fetcher.register_mcp_server(name, command=None, *, url=None, description=...)
#         fetcher.sync(prune=True) drops stored tools that are no longer registered
# Observability built-ins (0.8.0+):
#   fetcher.on_tool_call(cb)            -> framework-agnostic ToolCallEvent stream
#   provider.trace() / attach_to(..., trace=True)  -> console trace middleware
#   provider.selections                -> per-round RetrievalDecision history
from dotfetch.integrations import (
    DotFetchToolBridge,             # static AF bridge + workflow builders
    DotFetchApprovalMiddleware,
    DotFetchObservabilityMiddleware,
    DotFetchToolChoiceMiddleware,   # round-by-round tool_choice modulation
    ToolRefresher,                # framework-agnostic multi-turn re-selection
    fetch_framework_tools,        # legacy schema-only adapter (OpenAI-shape)
)
from dotfetch.integrations.frameworks import (
    DotFetchToolset,                # selection core behind every <Framework>Adapter
    ToolSpec,                     # framework-neutral tool handle (.ainvoke/.invoke)
    spec_from_tool,
)
# One <Framework>Adapter class per framework, from the clean per-framework
# namespace. Each exposes .select(query, limit=...) (async) and
# .convert(spec) (staticmethod):
from dotfetch.langchain import LangChainAdapter
from dotfetch.langgraph import LangGraphAdapter
from dotfetch.llamaindex import LlamaIndexAdapter
from dotfetch.crewai import CrewAIAdapter
from dotfetch.pydantic_ai import PydanticAIAdapter
from dotfetch.openai_agents import OpenAIAgentsAdapter
from dotfetch.haystack import HaystackAdapter
from dotfetch.agno import AgnoAdapter
from dotfetch.strands import StrandsAdapter
from dotfetch.dspy import DSPyAdapter
from dotfetch.google_adk import GoogleADKAdapter
from dotfetch.agent_framework import AgentFrameworkAdapter   # Microsoft Agent Framework
# Per-SDK LLM adapters (dialect baked in): .tools(query, limit=...)
from dotfetch.openai import OpenAIAdapter
from dotfetch.anthropic import AnthropicAdapter
from dotfetch.gemini import GeminiAdapter
from dotfetch.groq import GroqAdapter
from dotfetch.vertexai import VertexAIAdapter
from dotfetch.mistral import MistralAdapter
from dotfetch.query import (
    last_user_text, last_assistant_text, last_tool_result,
    concatenate_recent, fallback_chain,
    keyword_focused, truncated, latest_activity,
)
from dotfetch.adapters.embedders.cached import CachedEmbedder
from dotfetch.skills import load_skills_from_directory, load_skill   # Agent Skills (SKILL.md) loaders
from dotfetch.servers.mcp_server import create_mcp_server          # .streamable_http_app() / .sse_app()
from dotfetch.utils.registry_linter import (
    analyze_registry, pairwise_similarity, RegistryAnalysis,
)
```

For detailed reference on individual modules, see `references/` next to this file.
