# API reference quick-card

> Part of the `agent-gantry` skill. Core concepts, debugging and common
> pitfalls are in [`SKILL.md`](../SKILL.md); read those first.

```python
from agent_gantry import (
    AgentGantry,                  # main facade
    GantryContextProvider,        # AF native provider (per-run / per-turn)
    MissingRequiredToolError,     # raised when required=[...] tool absent
    RetrievalDecision,            # introspection: what the bridge surfaced
    RetrievalCandidate,           # one row in RetrievalDecision.candidates
    with_semantic_tools,          # decorator for plain LLM SDK calls
    set_default_gantry,           # bind a gantry to with_semantic_tools
    create_default_gantry,        # quick factory (auto-picks Nomic if available)
    render_result,                # stringify any tool result (incl. AF Content lists)
    enable_console_logging,       # opt into console output (lib configures no logging)
    ToolCall, ToolResult,
    ToolCallEvent,                # delivered to gantry.on_tool_call(callback)
    current_tool_invocation,      # framework run state inside a tool handler (or None)
    ToolInvocationContext,        # .framework / .deps / .run_id / .tool_call_id / .native_context
    ToolQuery, ConversationContext,
    ToolCapability, ToolCost, ToolDefinition, ToolHealth, ToolSource,
    Skill, SkillCategory, SkillSearchResult,   # procedural memory (see "Skills")
)
# Skills: gantry.add_skill / add_skills / add_skills_from_directory(path)
#         gantry.retrieve_skills(query) / retrieve_skills_as_prompt(query)
# Selection (0.16.0+, needs the `system-one` extra; Jev* names are deprecated aliases):
#   AgentGantry(selector=SystemOneSelector(threshold=0.4))  -> no embedder/vector store/sync
#   AgentGantry(reranker=SystemOneReranker())               -> refine a semantic shortlist
#   SelectionResult.scores                            -> every candidate, not just winners
#   (await gantry.retrieve(ToolQuery(...))).backend   -> "selector" | "semantic"
#       .selector_fallback / .selector_fallback_reason -> the router answered for the selector, and why
#   AgentGantry(selector=..., on_selector_fallback="raise") -> SelectorFallbackError instead of failing open
#   also importable from agent_gantry: SystemOneClient, SelectorFallbackError
#   (await gantry.retrieve(...)).embedder -> model_name of the embedder that searched; None on the
#       selector path. "simple-hash-embedder" = sentence-transformers is missing and retrieval is
#       matching tokens, not meaning (logged once at WARNING). EmbedderConfig(on_unavailable="raise")
#       raises ImportError instead of falling back.
# Measuring retrieval: from agent_gantry.evaluation import RetrievalCase, evaluate_retrieval,
#       tool_retriever, skill_retriever, mcp_server_retriever
#   report = await evaluate_retrieval(tool_retriever(gantry), [RetrievalCase("q", {"ns.tool"}),
#       RetrievalCase("nothing fits", expect_decline=True)], k=5)
#   report.recall_at_1 / .recall_at_k / .decline_correctness / .by_kind() / .misses()
# MCP:    gantry.serve_mcp(transport="stdio"|"http"|"sse", mode=..., expose=[...])
#         reset_sse_shutdown_latch()  -> release sse_starlette's process-global
#           shutdown latch between servers in one process; without it a second
#           EventSourceResponse is cancelled between headers and body (hangs).
#           Needs the `mcp` extra; raises AttributeError without it.
#         gantry.add_mcp_server(MCPServerConfig(command=[...] | url="https://...", headers=...))
#         tools discovered from a server are tagged "mcp:<server>" (mcp_server_tag(name) in
#           agent_gantry.schema.mcp): the server name is searchable, and a selector groups by it
#         gantry.register_mcp_server(name, command=None, *, url=None, description=...)
#         gantry.sync(prune=True) drops stored tools that are no longer registered
# Observability built-ins (0.8.0+):
#   gantry.on_tool_call(cb)            -> framework-agnostic ToolCallEvent stream
#   provider.trace() / attach_to(..., trace=True)  -> console trace middleware
#   provider.selections                -> per-round RetrievalDecision history
from agent_gantry.integrations import (
    GantryToolBridge,             # static AF bridge + workflow builders
    GantryApprovalMiddleware,
    GantryObservabilityMiddleware,
    GantryToolChoiceMiddleware,   # round-by-round tool_choice modulation
    ToolRefresher,                # framework-agnostic multi-turn re-selection
    fetch_framework_tools,        # legacy schema-only adapter (OpenAI-shape)
)
from agent_gantry.integrations.frameworks import (
    GantryToolset,                # selection core behind every <Framework>Adapter
    ToolSpec,                     # framework-neutral tool handle (.ainvoke/.invoke)
    spec_from_tool,
)
# One <Framework>Adapter class per framework, from the clean per-framework
# namespace. Each exposes .select(query, limit=...) (async) and
# .convert(spec) (staticmethod):
from agent_gantry.langchain import LangChainAdapter
from agent_gantry.langgraph import LangGraphAdapter
from agent_gantry.llamaindex import LlamaIndexAdapter
from agent_gantry.crewai import CrewAIAdapter
from agent_gantry.pydantic_ai import PydanticAIAdapter
from agent_gantry.openai_agents import OpenAIAgentsAdapter
from agent_gantry.haystack import HaystackAdapter
from agent_gantry.agno import AgnoAdapter
from agent_gantry.strands import StrandsAdapter
from agent_gantry.dspy import DSPyAdapter
from agent_gantry.google_adk import GoogleADKAdapter
from agent_gantry.agent_framework import AgentFrameworkAdapter   # Microsoft Agent Framework
# Per-SDK LLM adapters (dialect baked in): .tools(query, limit=...)
from agent_gantry.openai import OpenAIAdapter
from agent_gantry.anthropic import AnthropicAdapter
from agent_gantry.gemini import GeminiAdapter
from agent_gantry.groq import GroqAdapter
from agent_gantry.vertexai import VertexAIAdapter
from agent_gantry.mistral import MistralAdapter
from agent_gantry.query import (
    last_user_text, last_assistant_text, last_tool_result,
    concatenate_recent, fallback_chain,
    keyword_focused, truncated, latest_activity,
)
from agent_gantry.adapters.embedders.cached import CachedEmbedder
from agent_gantry.skills import load_skills_from_directory, load_skill   # Agent Skills (SKILL.md) loaders
from agent_gantry.servers.mcp_server import create_mcp_server          # .streamable_http_app() / .sse_app()
from agent_gantry.utils.registry_linter import (
    analyze_registry, pairwise_similarity, RegistryAnalysis,
)
```

For detailed reference on individual modules, see `references/` next to this file.
