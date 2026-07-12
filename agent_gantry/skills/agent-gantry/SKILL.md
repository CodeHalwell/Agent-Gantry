---
name: agent-gantry
description: Use this skill when the user is writing or debugging code that uses the `agent-gantry` Python library (semantic tool routing for LLM agents). Triggers on `import agent_gantry`, mentions of `AgentGantry`, `GantryContextProvider`, `GantryToolBridge`, `gantry.register`, `with_semantic_tools`, `LangChainAdapter`/`CrewAIAdapter`/`StrandsAdapter`/other `<Framework>Adapter` classes (or the per-SDK `OpenAIAdapter`/`AnthropicAdapter`/`GeminiAdapter`/`GroqAdapter`/`VertexAIAdapter`/`MistralAdapter`), `ToolRefresher`, or tool retrieval / surfacing problems with Microsoft Agent Framework, LangChain, LangGraph, AutoGen, CrewAI, LlamaIndex, Pydantic AI, OpenAI Agents SDK, Smolagents, Haystack, Agno, Strands Agents, Semantic Kernel, Google ADK, A2A, or MCP. Use proactively when the code imports `agent_gantry` even if the user did not explicitly invoke the skill.
---

# Agent-Gantry

Universal semantic tool router for LLM agents. Register tools once with `@gantry.register`, sync them into a vector store, then have the router surface only the top-K relevant tools per query — typically cutting prompt token spend by ~80% versus listing every tool.

This skill is the canonical reference for using the library. Read the section that matches the user's framework before writing code. When a user reports "the LLM doesn't see tool X" or "my surface is empty", jump straight to **Debugging routing** below — that's the most common failure mode and the library ships first-class introspection for it.

## When to use which integration path

| User's framework | API to use | Section |
|---|---|---|
| Microsoft Agent Framework (AF 1.x) | `AgentFrameworkAdapter(gantry).context_provider(top_k=...)` (dynamic, per-turn or per-call) or `.tool_bridge()` (static) | [Microsoft Agent Framework](#microsoft-agent-framework) |
| LangChain | `LangChainAdapter(gantry).select(query, limit=...)` → `bind_tools` | [Native framework adapters](#native-framework-adapters) |
| LangGraph | `LangGraphAdapter(gantry).select(...)` (static) or `.react_agent(model)` (live) | [Native framework adapters](#native-framework-adapters) |
| LlamaIndex | `LlamaIndexAdapter(gantry).select(...)` (static) or `.function_agent(llm)` (live) | [Native framework adapters](#native-framework-adapters) |
| CrewAI | `CrewAIAdapter(gantry).select(...)` | [Native framework adapters](#native-framework-adapters) |
| Pydantic AI | `PydanticAIAdapter(gantry).select(...)` (static) or `.toolset()` (live) | [Native framework adapters](#native-framework-adapters) |
| OpenAI Agents SDK | `OpenAIAgentsAdapter(gantry).select(...)` (static) or `.run(agent, run_input)` (live) | [Native framework adapters](#native-framework-adapters) |
| Smolagents | `SmolagentsAdapter(gantry).select(...)` | [Native framework adapters](#native-framework-adapters) |
| Haystack | `HaystackAdapter(gantry).select(...)` | [Native framework adapters](#native-framework-adapters) |
| Agno | `AgnoAdapter(gantry).select(...)` | [Native framework adapters](#native-framework-adapters) |
| Strands Agents | `StrandsAdapter(gantry).select(...)` (static) or `.agent(...)` / `.tool_hook(...)` (live, per-turn) | [Native framework adapters](#native-framework-adapters) |
| AutoGen / AG2 | `AutoGenAdapter(gantry).select(...)` / `.register(...)` (static) or `.workbench()` (live) | [Native framework adapters](#native-framework-adapters) |
| Semantic Kernel | `SemanticKernelAdapter(gantry).select(...)` / `.plugin(query)` | [Native framework adapters](#native-framework-adapters) |
| Google ADK | `GoogleADKAdapter(gantry).select(...)` (static) or `.agent(model=, name=)` (live) | [Native framework adapters](#native-framework-adapters) |
| Any framework, re-select every turn | `ToolRefresher(gantry).refresh(messages)` | [Multi-turn re-selection](#multi-turn-re-selection-toolrefresher) |
| Plain OpenAI / Anthropic / Gemini SDK | `OpenAIAdapter(gantry).tools(query)` (or `@with_semantic_tools(dialect=...)`) | [LLM-SDK direct](#llm-sdk-direct) |
| MCP server (expose Gantry as MCP) | `gantry.serve_mcp(mode="dynamic")` | [MCP](#mcp) |
| A2A (expose Gantry as an A2A agent) | `gantry.serve_a2a(...)` | [A2A](#a2a) |
| CLI inspection / linting | `agent-gantry lint / sim / sync --dry-run` | [CLI](#cli) |

> **Native adapters vs. `fetch_framework_tools`.** The per-framework
> `<Framework>Adapter(gantry).select(...)` methods return the framework's **native
> tool objects** (e.g. a LangChain `StructuredTool`, a CrewAI `BaseTool`), with
> execution routed through `gantry.execute` (retries, timeouts, circuit breakers,
> security policy). Prefer these. `fetch_framework_tools` still exists but only
> emits OpenAI-shape JSON schemas and supports a smaller name set — see
> [its note](#schema-only-adapter-fetch_framework_tools).

## Core concepts (read first)

1. **Tools are functions decorated with `@gantry.register`.** Type hints + docstring become the schema and the embedding text. Tags and `examples=[...]` improve recall.

2. **`await gantry.sync()` embeds every tool.** Fingerprint-based change detection means only modified tools re-embed on subsequent calls. With paid embedders, wrap the embedder in `CachedEmbedder` to persist across cold starts.

3. **`gantry.retrieve(...)` returns a `RetrievalResult`.** That's the universal API. The high-level integrations (`AgentFrameworkAdapter`/`GantryContextProvider`, `GantryToolBridge`, the `<Framework>Adapter` native adapters, the per-SDK LLM adapters, `ToolRefresher`, `@with_semantic_tools`) are thin wrappers around it that emit framework-specific shapes.

   Every native adapter exposes `<Framework>Adapter(gantry).select(query, *, limit=3, **select_kwargs)` (async) and shares the same selection knobs (`score_threshold`, `namespaces`, `tools_already_used`). Import the class from the clean per-framework namespace — `from agent_gantry.langchain import LangChainAdapter`. Importing `agent_gantry` never pulls in any third-party framework; the import is lazy and raises a `pip install ...` hint only when you call an adapter method.

4. **`score_threshold` is opt-in filtering.** Default is `0.0` (no filtering). Use `score_threshold="relative:0.8"` for length-robust filtering — that keeps anything within 80% of the top score. Fixed absolute cutoffs degrade badly on long, instruction-style queries because the embedding gets diluted.

5. **`query_strategy="per_call"` re-runs retrieval every chat-completion round.** This is the right choice for multi-step agents. Pair it with `provider.attach_to(agent)` or `middleware=[provider.as_chat_middleware()]`. Without the middleware, `per_call` silently degrades to `per_run` and the provider will warn you once.

6. **Observability ships built-in — don't hand-roll it.** `provider.trace()` (or `provider.attach_to(agent, trace=True)`) prints a readable per-round line of what was called and surfaced; `provider.selections` keeps the per-round retrieval history; `gantry.on_tool_call(cb)` fires a `ToolCallEvent` after every `gantry.execute` for framework-agnostic logging/metrics; `render_result(...)` stringifies any tool result. See [Observability & tracing](#observability--tracing).

7. **Importing `agent_gantry` configures no logging.** As of 0.8.0 the library attaches a `NullHandler` and never sets handlers or levels on your behalf, so a default `AgentGantry()` is silent. Opt into console output with `enable_console_logging()` (or your own logging config).

## Installing

This project uses **uv** as its package manager; `pip` works as a fallback. Inside a uv project use `uv add`; for an ad-hoc environment use `uv pip install`.

```bash
uv add agent-gantry                     # core (in a uv project)
uv add "agent-gantry[nomic]"            # local embeddings (recommended)
uv add "agent-gantry[openai]"           # OpenAI/Azure embeddings + custom OpenAI-compatible base_url
uv add "agent-gantry[agent-frameworks]" # Microsoft AF, LangChain, AutoGen, CrewAI, LlamaIndex, Semantic Kernel, Google ADK
uv add "agent-gantry[lancedb]"          # disk-persistent vector store
uv add "agent-gantry[mcp]"              # MCP client/server
uv add "agent-gantry[a2a]"              # A2A agent
uv add "agent-gantry[all]"              # everything
```

Not in a uv project? Swap `uv add` for `uv pip install`. No uv at all? Fall back to `pip install "agent-gantry[...]"` — the extras are identical.

The `[nomic]` extra is the recommended default for getting started — local, free, and accurate enough for production. `SimpleEmbedder` (the fallback when no embedder extra is installed) is hash-based and only useful for tests.

## Minimum-viable code

Every Agent-Gantry program has this shape:

```python
import asyncio
from agent_gantry import AgentGantry

gantry = AgentGantry()

@gantry.register
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: sunny"

@gantry.register
def book_flight(origin: str, destination: str) -> str:
    """Book a flight between two cities."""
    return f"Booked {origin} -> {destination}"

async def main():
    await gantry.sync()                                          # embed
    tools = await gantry.retrieve_tools("weather in Paris", limit=3)
    print(tools)  # OpenAI-shape schemas for the top 3 matches

asyncio.run(main())
```

`retrieve_tools(...)` returns OpenAI-shape tool schemas by default. Pass `dialect="anthropic"`, `"gemini"`, `"agent_framework"`, etc. to convert.

## Microsoft Agent Framework

The native, idiomatic integration. The one-class entry point is
`AgentFrameworkAdapter`; `AgentFrameworkAdapter(gantry).context_provider(top_k=...)`
returns a `GantryContextProvider` — an AF `ContextProvider` that runs at
`before_run` (per-run mode) or on every chat-completion round (per-call mode, via
a paired middleware). The adapter's other methods build the tool bridge and the
approval / observability / tool-choice middleware:
`.tool_bridge()`, `.approval_middleware(policy)`, `.observability_middleware()`,
`.tool_choice_middleware(decider)`. The returned types (`GantryContextProvider`,
`GantryToolBridge`, the middleware classes) are still importable directly for type
annotations.

### Per-run mode (default — fixed tool set for one `agent.run(...)` call)

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_gantry import AgentGantry
from agent_gantry.agent_framework import AgentFrameworkAdapter

gantry = AgentGantry()
# ... register tools, await gantry.sync() ...

provider = AgentFrameworkAdapter(gantry).context_provider(top_k=5)  # -> GantryContextProvider

agent = Agent(
    OpenAIChatClient(),
    "You are a helpful assistant.",
    context_providers=[provider],
)

result = await agent.run("Book me a flight to Tokyo")
```

### Per-call mode (re-runs retrieval every chat round)

Use when the agent reasons in multiple steps and needs different tools at different stages. The chat middleware is **required** — without it the per-call mode silently degrades to per-run.

```python
from agent_gantry.agent_framework import AgentFrameworkAdapter
from agent_gantry.query import fallback_chain, last_tool_result, last_user_text

provider = AgentFrameworkAdapter(gantry).context_provider(
    top_k=3,
    query_strategy="per_call",
    # Default for per_call is already fallback_chain(last_tool_result, last_user_text).
    # Pass query_generator=... only if you want a different rotation.
)

agent = Agent(
    OpenAIChatClient(),
    "...",
    context_providers=[provider],
    middleware=[provider.as_chat_middleware()],  # REQUIRED for per_call
)
```

Or use the one-call helper:

```python
agent = Agent(OpenAIChatClient(), "...")
provider.attach_to(agent)                 # appends provider + middleware in one shot
provider.attach_to(agent, trace=True)     # ...and a console trace of every tool call
```

`trace=True` also installs the built-in trace middleware (see
[Observability & tracing](#observability--tracing)) — the library-owned
replacement for hand-rolled `@function_middleware` logging.

### Pinning tools that must always be visible

```python
provider = AgentFrameworkAdapter(gantry).context_provider(
    top_k=5,
    required=["validate_input"],     # MissingRequiredToolError at construction if absent
    always_include=["log_event"],    # warning + skip if absent
    static_tools=[some_af_native_tool],  # tool not registered with gantry
)
```

`required` and `always_include` reference *gantry-registered* tool names. `static_tools` is for AF-native `@tool` callables that live outside the gantry registry — they're appended every round and never filtered.

### Static (no per-turn retrieval — bake once)

Use `GantryToolBridge` directly when the tool set is fixed at construction:

```python
from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

bridge = GantryToolBridge(gantry)
tools = await bridge.get_tools("customer support tasks", limit=5)
agent = Agent(OpenAIChatClient(), "...", tools=tools)
```

The bridge also exposes one-call agent constructors: `bridge.as_agent(...)`, `bridge.build_handoff_workflow(...)`, `bridge.build_sequential_workflow(...)`, `bridge.build_workflow(...)`.

### Approval middleware + observability

```python
from agent_gantry.core.security import SecurityPolicy
from agent_gantry.integrations.agent_framework_middleware import (
    GantryApprovalMiddleware,
    GantryObservabilityMiddleware,
    GantryToolChoiceMiddleware,
)

policy = SecurityPolicy(require_confirmation=["delete_*", "refund_*"])

# tool_choice modulation: force tool calls for the first N rounds, then auto.
rounds = {"n": 0}
def choose(_ctx):
    rounds["n"] += 1
    return "required" if rounds["n"] <= 5 else "auto"

agent = Agent(
    OpenAIChatClient(),
    "...",
    context_providers=[provider],
    middleware=[
        provider.as_chat_middleware(),
        GantryApprovalMiddleware(policy),
        GantryObservabilityMiddleware(gantry),
        GantryToolChoiceMiddleware(choose),
    ],
)
```

## Native framework adapters

For LangChain, LangGraph, LlamaIndex, CrewAI, Pydantic AI, OpenAI Agents SDK, Smolagents, Haystack, Agno, Strands Agents, AutoGen/AG2, Semantic Kernel, and Google ADK, use the native `<Framework>Adapter` class. Its `.select(query, limit=...)` method selects the top-K relevant tools and returns the framework's **native tool objects**, with every call still routed through `gantry.execute`.

```python
from agent_gantry import AgentGantry
from agent_gantry.langchain import LangChainAdapter   # clean per-framework namespace

gantry = AgentGantry()
# ... register tools, await gantry.sync() ...

tools = await LangChainAdapter(gantry).select("email the quarterly report to finance", limit=3)
# hand `tools` straight to a LangChain agent / ChatModel.bind_tools(tools)
```

Every adapter shares the identical signature `<Framework>Adapter(gantry).select(query, *, limit=3, **select_kwargs)` and the same per-framework namespace (`agent_gantry.<framework>`):

| Framework | Adapter class | Native object | Namespace import |
|---|---|---|---|
| LangChain | `LangChainAdapter` | `StructuredTool` | `from agent_gantry.langchain import LangChainAdapter` |
| LangGraph | `LangGraphAdapter` | LangChain `StructuredTool` | `from agent_gantry.langgraph import LangGraphAdapter` |
| LlamaIndex | `LlamaIndexAdapter` | `FunctionTool` | `from agent_gantry.llamaindex import LlamaIndexAdapter` |
| CrewAI | `CrewAIAdapter` | `crewai.tools.BaseTool` | `from agent_gantry.crewai import CrewAIAdapter` |
| Pydantic AI | `PydanticAIAdapter` | `pydantic_ai.tools.Tool` | `from agent_gantry.pydantic_ai import PydanticAIAdapter` |
| OpenAI Agents SDK | `OpenAIAgentsAdapter` | `agents.FunctionTool` | `from agent_gantry.openai_agents import OpenAIAgentsAdapter` |
| Smolagents | `SmolagentsAdapter` | `smolagents.Tool` | `from agent_gantry.smolagents import SmolagentsAdapter` |
| Haystack | `HaystackAdapter` | `haystack.tools.Tool` | `from agent_gantry.haystack import HaystackAdapter` |
| Agno | `AgnoAdapter` | `agno.tools.function.Function` | `from agent_gantry.agno import AgnoAdapter` |
| Strands Agents | `StrandsAdapter` | `strands.tools.decorator.DecoratedFunctionTool` | `from agent_gantry.strands import StrandsAdapter` |
| AutoGen / AG2 | `AutoGenAdapter` (`.select` / `.register`) | callables + `register_function` | `from agent_gantry.autogen import AutoGenAdapter` |
| Semantic Kernel | `SemanticKernelAdapter` (`.select` / `.plugin`) | `@kernel_function` `KernelFunction` | `from agent_gantry.semantic_kernel import SemanticKernelAdapter` |
| Google ADK | `GoogleADKAdapter` | `google.adk.tools.FunctionTool` | `from agent_gantry.google_adk import GoogleADKAdapter` |

Adapters with secondary methods (`AutoGenAdapter.register`, `SemanticKernelAdapter.plugin`) expose them on the same class — there is nothing extra to import.

Need one conversion at a time (you already hold the selected specs)? Use `<Adapter>.convert(spec)` (a staticmethod) with specs from `GantryToolset(gantry).select(query, limit=...)`.

```python
from agent_gantry.integrations.frameworks import GantryToolset
from agent_gantry.crewai import CrewAIAdapter

specs = await GantryToolset(gantry).select("research and writing", limit=4)
crew_tools = [CrewAIAdapter.convert(s) for s in specs]
```

### Deep per-turn "live" providers

The `.select(...)` method is *static* — select once, hand over a fixed list. The **live** adapter methods hook each framework's own per-turn lifecycle so Gantry re-selects tools on **every turn**, matching `GantryContextProvider` depth for Microsoft Agent Framework. Construct the adapter from the per-framework namespace, then call the live method.

```python
from agent_gantry.llamaindex import LlamaIndexAdapter
agent = LlamaIndexAdapter(gantry).function_agent(llm)   # re-selects tools each step
```

| Framework | Live adapter method | Native hook |
|---|---|---|
| LlamaIndex | `LlamaIndexAdapter(gantry).tool_retriever()` / `.function_agent(llm)` | `FunctionAgent(tool_retriever=…)` |
| Pydantic AI | `PydanticAIAdapter(gantry).toolset()` | `AbstractToolset.get_tools()` |
| AutoGen | `AutoGenAdapter(gantry).workbench()` | `Workbench.list_tools()` |
| Google ADK | `GoogleADKAdapter(gantry).before_model_callback()` / `.agent(model=, name=)` | `Agent(before_model_callback=…)` |
| Strands Agents | `StrandsAdapter(gantry).tool_hook()` / `.agent(...)` | `Agent(hooks=[…])` — `BeforeModelCallEvent` |
| LangGraph | `LangGraphAdapter(gantry).react_agent(model)` / `.areact_agent(model)` | dynamic `model` callable (re-binds tools per turn) |
| Semantic Kernel | `SemanticKernelAdapter(gantry).function_provider(kernel)` / `.refresh(kernel, query)` | per-invocation plugin refresh |
| OpenAI Agents SDK | `OpenAIAgentsAdapter(gantry).run(agent, run_input)` / `.session(agent)` / `.run_hooks(agent)` | `RunHooks.on_llm_start` + per-run refresh |

The returned live objects keep their classes (`GantryToolRetriever`, live `GantryToolset`, `GantryWorkbench`, `GantryFunctionProvider`, `GantryAgentSession`) — still importable from each framework's `*_live` module for `isinstance` checks.

Frameworks whose tool list is **fixed at agent construction** (CrewAI, Agno, Haystack, Smolagents) can't re-advertise tools mid-run. Build a self-rebuilding agent with `<Adapter>(gantry).agent_builder(...)` (Haystack: `HaystackAdapter(gantry).tool_invoker_builder(...)`); it re-selects and rebuilds on each top-level call. For a one-shot fresh slice of native tools, call `<Adapter>(gantry).live_tools(query)` (async).

## Multi-turn re-selection (ToolRefresher)

`ToolRefresher` generalizes the per-call retrieval pattern to **any** framework without a native live hook: re-rank the whole registry on every turn so the agent can pivot tools as the task changes.

```python
from agent_gantry.integrations import ToolRefresher

refresher = ToolRefresher(gantry, limit=3, dialect="openai")

# Each turn, pass the running message list. Selection is recomputed fresh and
# tools already used are deprioritized so the agent keeps moving forward.
tools = await refresher.refresh(messages)        # dialect schemas
specs = await refresher.refresh_specs(messages)  # ToolSpec objects
```

The default query generator (`agent_gantry.query.latest_activity`) is **recency-aware**, so one refresher serves both styles with no config:

- **Autonomous agents / tool pipelines** — newest message is a tool result, so that result's *content* selects the next tool (`fetch → clean → train → evaluate → report`).
- **Conversational agents** — newest message is the user's, so their new request drives selection (`weather → flights → hotel → email`).

Force one behaviour with `query_generator=last_user_text` or `last_tool_result`.

## Schema-only adapter (`fetch_framework_tools`)

The legacy adapter returns OpenAI-shape JSON schemas (not native objects) and supports a smaller, hyphen/underscore-sensitive name set: `"langgraph"`, `"semantic-kernel"`, `"crew_ai"`, `"google_adk"`, `"strands"`, `"agent_framework"`. Prefer the `<Framework>Adapter` classes for the frameworks listed above; reach for this only when you want raw schemas (e.g. a framework that just wants OpenAI tool dicts).

```python
from agent_gantry.integrations import fetch_framework_tools

schemas = await fetch_framework_tools(
    gantry, "code execution", framework="google_adk", limit=3
)
```

## LLM-SDK direct

For a minimal stack — no agent framework — there are two paths.

**Per-SDK adapter classes** (explicit, no default-gantry binding). One class per provider, each baking in the right schema dialect:

```python
from openai import AsyncOpenAI
from agent_gantry import AgentGantry
from agent_gantry.openai import OpenAIAdapter

gantry = AgentGantry()
# ... register tools, await gantry.sync() ...

client = AsyncOpenAI()
tools = await OpenAIAdapter(gantry).tools("weather in Paris", limit=3)  # OpenAI chat-completions schemas
resp = await client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "weather in Paris?"}],
    tools=tools,
)
```

| Provider | Adapter class | Namespace import | Method |
|---|---|---|---|
| OpenAI | `OpenAIAdapter` | `from agent_gantry.openai import OpenAIAdapter` | `.tools(query, limit=n)` (`.responses_tools(...)` for the Responses API) |
| Anthropic | `AnthropicAdapter` | `from agent_gantry.anthropic import AnthropicAdapter` | `.tools(query, limit=n)` |
| Gemini | `GeminiAdapter` | `from agent_gantry.gemini import GeminiAdapter` | `.tools(query, limit=n)` |
| Groq | `GroqAdapter` | `from agent_gantry.groq import GroqAdapter` | `.tools(query, limit=n)` |
| Vertex AI | `VertexAIAdapter` | `from agent_gantry.vertexai import VertexAIAdapter` | `.tools(query, limit=n)` |
| Mistral | `MistralAdapter` | `from agent_gantry.mistral import MistralAdapter` | `.tools(query, limit=n)` |

`<Provider>Adapter(gantry).tools(query, limit=n)` is equivalent to `await gantry.retrieve_tools(query, limit=n, dialect="<provider>")` — the adapter just bakes the dialect in.

**Decorator** (`@with_semantic_tools`) when you'd rather bind a default gantry and have tools injected automatically:

```python
from openai import AsyncOpenAI
from agent_gantry import AgentGantry, set_default_gantry, with_semantic_tools

gantry = AgentGantry()
set_default_gantry(gantry)

@gantry.register
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return "..."

client = AsyncOpenAI()

@with_semantic_tools(limit=3)   # default dialect="openai"
async def chat(prompt: str, *, tools=None):
    return await client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
    )

await chat("weather in Paris?")
```

Dialect support: `dialect="anthropic"`, `"gemini"`, `"mistral"`, `"groq"`, `"openai_responses"`. Each emits the right schema for that provider; the call shape stays identical.

Mistral note: the official `mistralai` SDK is quarantined on PyPI. Use `AsyncOpenAI` pointed at `https://api.mistral.ai/v1` — Mistral's API is OpenAI-compatible. With `agent-gantry`, the OpenAI dialect (or `MistralAdapter`) works directly.

OpenAI-compatible custom endpoints (Requesty, OpenRouter, Together, vLLM, …) are first-class: pass `api_base` in the `EmbedderConfig` or set `OPENAI_BASE_URL`, and `OpenAIEmbedder` forwards it to the client.

## MCP

Expose Gantry as an MCP server (Claude Desktop, Cline, etc.):

```python
gantry = AgentGantry()
# ... register tools, await gantry.sync() ...
await gantry.serve_mcp(transport="stdio", mode="dynamic")
```

`mode="dynamic"` exposes only two meta-tools (`find_relevant_tools`, `execute_tool`) — the MCP client semantic-searches Gantry on demand. ~90% smaller tool list footprint than `mode="static"`.

Consume external MCP servers:

```python
from agent_gantry.schema.config import MCPServerConfig

await gantry.add_mcp_server(MCPServerConfig(
    name="filesystem",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
    args=["--path", "/tmp"],
    namespace="fs",
))
```

## A2A

```python
gantry.serve_a2a(host="0.0.0.0", port=8080)
# Agent Card at: http://localhost:8080/.well-known/agent.json
```

Skills exposed: `tool_discovery` (semantic search) and `tool_execution` (run a tool).

## CLI

The `agent-gantry` command ships with the package. Inside a uv project, prefix with `uv run` (e.g. `uv run agent-gantry list`); the bare `agent-gantry` form works when the package is on `PATH` (pip install or an activated venv).

```bash
uv run agent-gantry list                              # list registered demo tools
uv run agent-gantry search "refund order" --limit 3   # semantic search
uv run agent-gantry lint                              # detect tool-description authoring mistakes
uv run agent-gantry sim toolA toolB                   # cosine similarity between two tools
uv run agent-gantry sync --dry-run                    # which tools would (re-)embed and why
uv run agent-gantry install-skill --claude            # install THIS skill into ~/.claude/skills
```

`lint` flags three patterns that silently degrade routing quality:

1. Descriptions that mention other registered tools (embedding pulls them toward each other).
2. Pairs of tools with >0.85 cosine similarity (probably should be one tool, or differentiated).
3. Tags that appear on more than half the registry (low discriminative value).

## Observability & tracing

As of 0.8.0 the library ships the tracing/logging glue you'd otherwise hand-roll. Reach for these before writing custom middleware.

### Console trace — one readable line per tool call

`provider.trace()` returns an AF *function* middleware; `attach_to(agent, trace=True)` installs it (plus the per-call retrieval middleware) in one call. For every tool the model invokes it prints the round, the call, the router's surfaced set, and a preview of the result:

```python
provider = AgentFrameworkAdapter(gantry).context_provider(top_k=3, query_strategy="per_call")
agent = Agent(OpenAIChatClient(), "...")
provider.attach_to(agent, trace=True)
# >>> round 1: generate_password({'length': 20})  [surfaced: generate_password:0.62, ...]
# <<< round 1: generate_password -> Xk9$mP2v...
```

`provider.trace(render=False, printer=logger.info, limit=200)` tunes it: skip the result line, redirect the sink (e.g. to a logger), or change the preview length.

### Per-round selection history

`provider.last_selection` is a single mutable slot. `provider.selections` is the per-round history for the **current run** (oldest first, bounded; reset at the start of each `agent.run`), so you can audit what was surfaced at *each* step of a run, not just the last:

```python
for i, decision in enumerate(provider.selections, start=1):
    print(f"round {i}: {decision.summary()}")
```

### Framework-agnostic tool-call events

`gantry.on_tool_call(callback)` fires after **every** `gantry.execute` (and once per call in `execute_batch`) with a `ToolCallEvent`. Because `execute` is the single choke point every framework adapter routes through, this gives logging/metrics across LangChain, CrewAI, AF, direct calls — with no per-framework middleware:

```python
from agent_gantry import ToolCallEvent

def log_call(event: ToolCallEvent) -> None:
    status = "ok" if event.ok else f"FAILED ({event.result.error})"
    print(f"{event.tool_name} -> {status} ({event.latency_ms:.0f} ms)")

unsubscribe = gantry.on_tool_call(log_call)   # sync or async callbacks; returns an unsubscribe fn
```

Callbacks are error-isolated — a raising listener never breaks the tool run — and failed tools still emit an event (`event.ok is False`). `ToolCallEvent` carries `.call`, `.result`, and the convenience accessors `.tool_name`, `.status`, `.ok`, `.latency_ms`.

> **Batch timing:** events fire immediately for each `gantry.execute`, but for `gantry.execute_batch` they're emitted *after the whole batch finishes* (one per call, paired by index). `latency_ms` stays accurate; only delivery is batched — relevant if you timestamp events for a latency dashboard.

### Rendering tool results

`render_result(result, *, limit=None, collapse_whitespace=False)` turns any result — including AF `Content`-block lists, bytes, dicts, or arbitrary objects — into readable text for logs/UIs. The trace middleware uses it internally.

### Logging

The library configures no logging (a `NullHandler` is attached at import). To see Gantry's own INFO lines on the console, opt in once:

```python
from agent_gantry import enable_console_logging
enable_console_logging()   # attaches a console handler + sets the agent_gantry level
```

Telemetry (`ConsoleTelemetryAdapter`, the default) still emits structured records; they flow to whatever handlers your app configured. There's nothing to silence anymore — a default `AgentGantry()` is quiet.

## Debugging routing

When a user says "the LLM doesn't see my tool" / "my surface is empty" / "wrong tools are being selected", use these in order:

### 1. `provider.last_selection` — what just happened

```python
provider = GantryContextProvider(gantry, top_k=5)
result = await agent.run("...")

decision = provider.last_selection
print(decision.summary())           # one-line: query + top scores
print(decision.injected)            # tool names that made it to the LLM
for c in decision.candidates:
    print(c.name, c.score, c.kept)  # full ranked list, kept/dropped flag
```

`RetrievalDecision` carries: the query, every candidate the gantry returned, the threshold mode used, the effective numeric cutoff, and the final injected list. For the **per-round history** (not just the latest), use `provider.selections` — see [Observability & tracing](#observability--tracing).

### 2. `provider.dry_run_retrieve(query)` — same code path as live, no agent

```python
decision = await provider.dry_run_retrieve("find boundaries in OCR text")
for c in decision.candidates:
    print(f"{c.score:.3f}  {c.qualified_name}  kept={c.kept}")
```

This uses the exact same threshold, `top_k`, and `query_kwargs` as the live middleware — so the answer is "what the LLM would see for that query".

### 3. `verbose=True` — one-line INFO log per round

```python
provider = GantryContextProvider(gantry, top_k=5, verbose=True)
# logs: gantry: query="..." → top5: [tool_a:0.61, tool_b:0.58, ...]
```

### 4. `score_threshold` filtered everything?

When the threshold drops all candidates, the bridge logs a WARNING with the top scores so you can tell "threshold issue" from "relevance issue". Default is `0.0`. If a user has set `score_threshold=0.3` (the legacy default) on a long pipeline query, **lower it or switch to relative**:

```python
provider = GantryContextProvider(
    gantry,
    score_threshold="relative:0.8",   # keep anything within 80% of top score
)
```

### 5. Long queries silently degrade routing

Long imperative scaffolding ("Please run this five-step pipeline. Use a different tool for each step…") dilutes the embedding. Strip it with `keyword_focused`:

```python
from agent_gantry.query import keyword_focused, truncated, last_user_text

provider = GantryContextProvider(
    gantry,
    query_strategy="per_call",
    query_generator=keyword_focused,       # drops scaffolding tokens
)

# Or cap the query length, biasing toward the latest tool output:
provider = GantryContextProvider(
    gantry,
    query_strategy="per_call",
    query_generator=truncated(last_user_text, max_chars=200, keep="tail"),
)
```

### 6. Lint the registry — author-side bugs

```bash
agent-gantry lint
```

Or programmatically:

```python
analysis = await gantry.analyze_registry()
print(analysis.format_text())
```

This catches the headline mistakes: a tool description that names another tool (which pulls it toward the wrong queries), pairs of tools that are too similar to disambiguate, and tags that are too generic.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Empty tool surface | `score_threshold` too aggressive for query length | Lower to `0.0` or use `"relative:0.8"` |
| `per_call` not adapting | `as_chat_middleware()` not attached | Use `provider.attach_to(agent)` or add to `middleware=[...]` |
| `per_call` set but identical surface each round | Default `query_generator=last_user_text` doesn't change | The new default is `fallback_chain(last_tool_result, last_user_text)`; if you overrode it, switch back |
| Wrong tools selected | Description names another tool ("unrelated to factorial…") | Run `agent-gantry lint`; remove cross-references |
| `top_k=6` but I see 8 tools | Skills / `always_include` / `static_tools` add on top of dynamic top_k | Expected; subtract those |
| Cold-start re-embeds every time | Default `InMemoryVectorStore` is ephemeral | Wrap embedder in `CachedEmbedder` or use `[lancedb]` |
| OpenAI embedder can't reach my proxy | Custom base_url not configured | Pass `api_base=...` in `EmbedderConfig` or set `OPENAI_BASE_URL` |
| No telemetry / INFO logs appear (0.8.0+) | Library no longer configures logging by default | Call `enable_console_logging()` or configure your own handler on the `agent_gantry` logger |

## Persistent embedding cache

Re-embedding costs API spend at every cold start. Wrap any embedder:

```python
from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.openai import OpenAIEmbedder
from agent_gantry.adapters.embedders.cached import CachedEmbedder
from agent_gantry.schema.config import EmbedderConfig

base = OpenAIEmbedder(EmbedderConfig(type="openai", model="text-embedding-3-large"))
embedder = CachedEmbedder(base)   # default: ~/.cache/agent_gantry/embeddings.sqlite

gantry = AgentGantry(embedder=embedder)
```

Cache is keyed by `(embedder_id, sha256(text))` — different models / dimensions never collide.

## Query generators reference

| Generator | Use when |
|---|---|
| `last_user_text` (default for `per_run`) | Tool selection driven by the user's most recent message |
| `last_tool_result` | Next tool should match the *content* of the last tool's output |
| `last_assistant_text` | Tool selection driven by the model's most recent reasoning |
| `concatenate_recent(n=3)` | Multi-message context window matters |
| `fallback_chain(*gens)` | Try each in order until one returns non-empty |
| `keyword_focused` | Long instructional queries dilute the signal |
| `truncated(gen, max_chars=200)` | Cap the query length, defaults to keeping the tail |
| `latest_activity` (default for `ToolRefresher`) | Recency-aware: picks user text *or* last tool result, whichever is newest |

Default for `query_strategy="per_call"` is `fallback_chain(last_tool_result, last_user_text)` — it adapts as the agent reasons. `ToolRefresher` defaults to `latest_activity`, which serves autonomous and conversational agents alike.

## API reference quick-card

```python
from agent_gantry import (
    AgentGantry,                  # main facade
    GantryContextProvider,        # AF native provider (per-run / per-call)
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
    ToolQuery, ConversationContext,
    ToolCapability, ToolCost, ToolDefinition, ToolHealth, ToolSource,
)
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
from agent_gantry.smolagents import SmolagentsAdapter
from agent_gantry.haystack import HaystackAdapter
from agent_gantry.agno import AgnoAdapter
from agent_gantry.strands import StrandsAdapter
from agent_gantry.autogen import AutoGenAdapter
from agent_gantry.semantic_kernel import SemanticKernelAdapter
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
from agent_gantry.utils.registry_linter import (
    analyze_registry, pairwise_similarity, RegistryAnalysis,
)
```

For detailed reference on individual modules, see `references/` next to this file.
