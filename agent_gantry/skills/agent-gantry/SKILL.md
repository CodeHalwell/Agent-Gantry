---
name: agent-gantry
description: Use this skill when the user is writing or debugging code that uses the `agent-gantry` Python library (semantic tool routing for LLM agents). Triggers on `import agent_gantry`, mentions of `AgentGantry`, `GantryContextProvider`, `GantryToolBridge`, `gantry.register`, `with_semantic_tools`, `for_langchain`/`for_crewai`/other `for_<framework>` adapters, `ToolRefresher`, or tool retrieval / surfacing problems with Microsoft Agent Framework, LangChain, LangGraph, AutoGen, CrewAI, LlamaIndex, Pydantic AI, OpenAI Agents SDK, Smolagents, Haystack, Agno, Semantic Kernel, Google ADK, A2A, or MCP. Use proactively when the code imports `agent_gantry` even if the user did not explicitly invoke the skill.
---

# Agent-Gantry

Universal semantic tool router for LLM agents. Register tools once with `@gantry.register`, sync them into a vector store, then have the router surface only the top-K relevant tools per query — typically cutting prompt token spend by ~80% versus listing every tool.

This skill is the canonical reference for using the library. Read the section that matches the user's framework before writing code. When a user reports "the LLM doesn't see tool X" or "my surface is empty", jump straight to **Debugging routing** below — that's the most common failure mode and the library ships first-class introspection for it.

## When to use which integration path

| User's framework | API to use | Section |
|---|---|---|
| Microsoft Agent Framework (AF 1.x) | `GantryContextProvider` (dynamic, per-turn or per-call) or `GantryToolBridge` (static) | [Microsoft Agent Framework](#microsoft-agent-framework) |
| LangChain | `for_langchain(gantry, query, limit=...)` → `bind_tools` | [Native framework adapters](#native-framework-adapters) |
| LangGraph | `for_langgraph(...)` (static) or `create_gantry_react_agent(...)` (live) | [Native framework adapters](#native-framework-adapters) |
| LlamaIndex | `for_llamaindex(...)` (static) or `gantry_function_agent(...)` (live) | [Native framework adapters](#native-framework-adapters) |
| CrewAI | `for_crewai(...)` | [Native framework adapters](#native-framework-adapters) |
| Pydantic AI | `for_pydantic_ai(...)` (static) or `gantry_toolset(...)` (live) | [Native framework adapters](#native-framework-adapters) |
| OpenAI Agents SDK | `for_openai_agents(...)` (static) or `run_with_gantry(...)` (live) | [Native framework adapters](#native-framework-adapters) |
| Smolagents | `for_smolagents(...)` | [Native framework adapters](#native-framework-adapters) |
| Haystack | `for_haystack(...)` | [Native framework adapters](#native-framework-adapters) |
| Agno | `for_agno(...)` | [Native framework adapters](#native-framework-adapters) |
| AutoGen / AG2 | `for_autogen(...)` / `register_with_autogen(...)` (static) or `gantry_workbench(...)` (live) | [Native framework adapters](#native-framework-adapters) |
| Semantic Kernel | `for_semantic_kernel(...)` / `gantry_plugin(...)` | [Native framework adapters](#native-framework-adapters) |
| Google ADK | `for_google_adk(...)` (static) or `gantry_adk_agent(...)` (live) | [Native framework adapters](#native-framework-adapters) |
| Any framework, re-select every turn | `ToolRefresher(gantry).refresh(messages)` | [Multi-turn re-selection](#multi-turn-re-selection-toolrefresher) |
| Plain OpenAI / Anthropic / Gemini SDK | `@with_semantic_tools(dialect=...)` decorator | [LLM-SDK direct](#llm-sdk-direct) |
| MCP server (expose Gantry as MCP) | `gantry.serve_mcp(mode="dynamic")` | [MCP](#mcp) |
| A2A (expose Gantry as an A2A agent) | `gantry.serve_a2a(...)` | [A2A](#a2a) |
| CLI inspection / linting | `agent-gantry lint / sim / sync --dry-run` | [CLI](#cli) |

> **Native adapters vs. `fetch_framework_tools`.** As of v0.5.0 the per-framework
> `for_<fw>()` helpers return the framework's **native tool objects** (e.g. a
> LangChain `StructuredTool`, a CrewAI `BaseTool`), with execution routed through
> `gantry.execute` (retries, timeouts, circuit breakers, security policy). Prefer
> these. `fetch_framework_tools` still exists but only emits OpenAI-shape JSON
> schemas and supports a smaller name set — see [its note](#schema-only-adapter-fetch_framework_tools).

## Core concepts (read first)

1. **Tools are functions decorated with `@gantry.register`.** Type hints + docstring become the schema and the embedding text. Tags and `examples=[...]` improve recall.

2. **`await gantry.sync()` embeds every tool.** Fingerprint-based change detection means only modified tools re-embed on subsequent calls. With paid embedders, wrap the embedder in `CachedEmbedder` to persist across cold starts.

3. **`gantry.retrieve(...)` returns a `RetrievalResult`.** That's the universal API. The high-level integrations (`GantryContextProvider`, `GantryToolBridge`, the `for_<fw>` native adapters, `ToolRefresher`, `@with_semantic_tools`) are thin wrappers around it that emit framework-specific shapes.

   Every native adapter is `for_<fw>(gantry, query, *, limit=3, **select_kwargs)` and shares the same selection knobs (`score_threshold`, `namespaces`, `tools_already_used`). Import it from the clean per-framework namespace — `from agent_gantry.langchain import for_langchain`. Importing `agent_gantry` never pulls in any third-party framework; the import is lazy and raises a `pip install ...` hint only when you call the adapter.

4. **`score_threshold` is opt-in filtering.** Default is `0.0` (no filtering). Use `score_threshold="relative:0.8"` for length-robust filtering — that keeps anything within 80% of the top score. Fixed absolute cutoffs degrade badly on long, instruction-style queries because the embedding gets diluted.

5. **`query_strategy="per_call"` re-runs retrieval every chat-completion round.** This is the right choice for multi-step agents. Pair it with `provider.attach_to(agent)` or `middleware=[provider.as_chat_middleware()]`. Without the middleware, `per_call` silently degrades to `per_run` and the provider will warn you once.

## Installing

```bash
pip install agent-gantry                # core
pip install "agent-gantry[nomic]"       # local embeddings (recommended)
pip install "agent-gantry[openai]"      # OpenAI/Azure embeddings + custom OpenAI-compatible base_url
pip install "agent-gantry[agent-frameworks]"  # Microsoft AF, LangChain, AutoGen, CrewAI, LlamaIndex, Semantic Kernel, Google ADK
pip install "agent-gantry[lancedb]"     # disk-persistent vector store
pip install "agent-gantry[mcp]"         # MCP client/server
pip install "agent-gantry[a2a]"         # A2A agent
pip install "agent-gantry[all]"         # everything
```

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

The native, idiomatic integration. `GantryContextProvider` is an AF `ContextProvider` that runs at `before_run` (per-run mode) or on every chat-completion round (per-call mode, via a paired middleware).

### Per-run mode (default — fixed tool set for one `agent.run(...)` call)

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_gantry import AgentGantry, GantryContextProvider

gantry = AgentGantry()
# ... register tools, await gantry.sync() ...

provider = GantryContextProvider(gantry, top_k=5)

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
from agent_gantry import GantryContextProvider
from agent_gantry.query import fallback_chain, last_tool_result, last_user_text

provider = GantryContextProvider(
    gantry,
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
provider.attach_to(agent)        # appends provider + middleware in one shot
```

### Pinning tools that must always be visible

```python
provider = GantryContextProvider(
    gantry,
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

For LangChain, LangGraph, LlamaIndex, CrewAI, Pydantic AI, OpenAI Agents SDK, Smolagents, Haystack, Agno, AutoGen/AG2, Semantic Kernel, and Google ADK, use the native `for_<fw>` adapter. It selects the top-K relevant tools and returns the framework's **native tool objects**, with every call still routed through `gantry.execute`.

```python
from agent_gantry import AgentGantry
from agent_gantry.langchain import for_langchain     # clean per-framework namespace

gantry = AgentGantry()
# ... register tools, await gantry.sync() ...

tools = await for_langchain(gantry, "email the quarterly report to finance", limit=3)
# hand `tools` straight to a LangChain agent / ChatModel.bind_tools(tools)
```

Every adapter has the identical signature `for_<fw>(gantry, query, *, limit=3, **select_kwargs)` and the same per-framework namespace (`agent_gantry.<framework>`):

| Framework | Static adapter | Native object | Namespace import |
|---|---|---|---|
| LangChain | `for_langchain` | `StructuredTool` | `from agent_gantry.langchain import for_langchain` |
| LangGraph | `for_langgraph` | LangChain `StructuredTool` | `from agent_gantry.langgraph import for_langgraph` |
| LlamaIndex | `for_llamaindex` | `FunctionTool` | `from agent_gantry.llamaindex import for_llamaindex` |
| CrewAI | `for_crewai` | `crewai.tools.BaseTool` | `from agent_gantry.crewai import for_crewai` |
| Pydantic AI | `for_pydantic_ai` | `pydantic_ai.tools.Tool` | `from agent_gantry.pydantic_ai import for_pydantic_ai` |
| OpenAI Agents SDK | `for_openai_agents` | `agents.FunctionTool` | `from agent_gantry.openai_agents import for_openai_agents` |
| Smolagents | `for_smolagents` | `smolagents.Tool` | `from agent_gantry.smolagents import for_smolagents` |
| Haystack | `for_haystack` | `haystack.tools.Tool` | `from agent_gantry.haystack import for_haystack` |
| Agno | `for_agno` | `agno.tools.function.Function` | `from agent_gantry.agno import for_agno` |
| AutoGen / AG2 | `for_autogen`, `register_with_autogen` | callables + `register_function` | `from agent_gantry.autogen import for_autogen` |
| Semantic Kernel | `for_semantic_kernel`, `gantry_plugin` | `@kernel_function` `KernelFunction` | `from agent_gantry.semantic_kernel import for_semantic_kernel` |
| Google ADK | `for_google_adk` | `google.adk.tools.FunctionTool` | `from agent_gantry.google_adk import for_google_adk` |

Need one conversion at a time (you already hold the selected specs)? Use `spec_to_<fw>(spec)` with specs from `GantryToolset(gantry).select(query, limit=...)`.

```python
from agent_gantry.integrations.frameworks import GantryToolset, spec_to_crewai

specs = await GantryToolset(gantry).select("research and writing", limit=4)
crew_tools = [spec_to_crewai(s) for s in specs]
```

### Deep per-turn "live" providers

The `for_<fw>` helpers are *static* — select once, hand over a fixed list. The **live** providers hook each framework's own per-turn lifecycle so Gantry re-selects tools on **every turn**, matching `GantryContextProvider` depth for Microsoft Agent Framework. Import them lazily from the per-framework namespace.

```python
from agent_gantry.llamaindex import gantry_function_agent
agent = gantry_function_agent(gantry, llm)   # re-selects tools each step
```

| Framework | Live entry point | Native hook |
|---|---|---|
| LlamaIndex | `gantry_tool_retriever` / `gantry_function_agent` | `FunctionAgent(tool_retriever=…)` |
| Pydantic AI | `gantry_toolset` | `AbstractToolset.get_tools()` |
| AutoGen | `gantry_workbench` | `Workbench.list_tools()` |
| Google ADK | `gantry_before_model_callback` / `gantry_adk_agent` | `Agent(before_model_callback=…)` |
| LangGraph | `create_gantry_react_agent` / `acreate_gantry_react_agent` | dynamic `model` callable (re-binds tools per turn) |
| Semantic Kernel | `GantryFunctionProvider` / `refresh_kernel_tools` | per-invocation plugin refresh |
| OpenAI Agents SDK | `run_with_gantry` / `GantryAgentSession` / `gantry_run_hooks` | `RunHooks.on_llm_start` + per-run refresh |

Frameworks whose tool list is **fixed at agent construction** (CrewAI, Agno, Haystack, Smolagents) can't re-advertise tools mid-run; their live wrappers (`GantryLiveCrewAgent`, `GantryLiveAgnoAgent`, `gantry_haystack_tools`, `GantryLiveSmolAgent`, all under `agent_gantry.integrations.frameworks`) re-select and rebuild the agent on each top-level call.

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

The legacy adapter returns OpenAI-shape JSON schemas (not native objects) and supports a smaller, hyphen/underscore-sensitive name set: `"langgraph"`, `"semantic-kernel"`, `"crew_ai"`, `"google_adk"`, `"strands"`, `"agent_framework"`. Prefer `for_<fw>` for the frameworks listed above; reach for this only when you want raw schemas (e.g. a framework that just wants OpenAI tool dicts).

```python
from agent_gantry.integrations import fetch_framework_tools

schemas = await fetch_framework_tools(
    gantry, "code execution", framework="google_adk", limit=3
)
```

## LLM-SDK direct

For minimal stack — no agent framework — use the `@with_semantic_tools` decorator:

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

Mistral note: the official `mistralai` SDK is quarantined on PyPI. Use `AsyncOpenAI` pointed at `https://api.mistral.ai/v1` — Mistral's API is OpenAI-compatible. With `agent-gantry`, the OpenAI dialect works directly.

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

The `agent-gantry` command ships with the package:

```bash
agent-gantry list                              # list registered demo tools
agent-gantry search "refund order" --limit 3   # semantic search
agent-gantry lint                              # detect tool-description authoring mistakes
agent-gantry sim toolA toolB                   # cosine similarity between two tools
agent-gantry sync --dry-run                    # which tools would (re-)embed and why
agent-gantry install-skill --claude            # install THIS skill into ~/.claude/skills
```

`lint` flags three patterns that silently degrade routing quality:

1. Descriptions that mention other registered tools (embedding pulls them toward each other).
2. Pairs of tools with >0.85 cosine similarity (probably should be one tool, or differentiated).
3. Tags that appear on more than half the registry (low discriminative value).

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

`RetrievalDecision` carries: the query, every candidate the gantry returned, the threshold mode used, the effective numeric cutoff, and the final injected list.

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
    ToolCall, ToolResult,
    ToolQuery, ConversationContext,
    ToolCapability, ToolCost, ToolDefinition, ToolHealth, ToolSource,
)
from agent_gantry.integrations import (
    GantryToolBridge,             # static AF bridge + workflow builders
    GantryApprovalMiddleware,
    GantryObservabilityMiddleware,
    GantryToolChoiceMiddleware,   # round-by-round tool_choice modulation
    ToolRefresher,                # framework-agnostic multi-turn re-selection
    fetch_framework_tools,        # legacy schema-only adapter (OpenAI-shape)
)
from agent_gantry.integrations.frameworks import (
    GantryToolset,                # selection core behind every for_<fw>
    ToolSpec,                     # framework-neutral tool handle (.ainvoke/.invoke)
    spec_from_tool,
    for_langchain, for_langgraph, for_llamaindex, for_crewai,
    for_pydantic_ai, for_openai_agents, for_smolagents, for_haystack,
    for_agno, for_autogen, for_semantic_kernel, for_google_adk,
    # ...and matching spec_to_<fw> converters.
)
# Or the clean per-framework namespace: from agent_gantry.langchain import for_langchain
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
