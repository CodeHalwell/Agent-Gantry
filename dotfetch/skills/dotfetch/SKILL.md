---
name: dotfetch
description: Use this skill when the user is writing or debugging code that uses the `dotfetch` Python library (semantic tool routing for LLM agents). Triggers on `import dotfetch`, mentions of `DotFetch`, `DotFetchContextProvider`, `DotFetchToolBridge`, `fetcher.register`, `with_semantic_tools`, `LangChainAdapter`/`CrewAIAdapter`/`StrandsAdapter`/`DSPyAdapter`/other `<Framework>Adapter` classes (or the per-SDK `OpenAIAdapter`/`AnthropicAdapter`/`GeminiAdapter`/`GroqAdapter`/`VertexAIAdapter`/`MistralAdapter`), `ToolRefresher`, or tool retrieval / surfacing problems with Microsoft Agent Framework, LangChain, LangGraph, CrewAI, LlamaIndex, Pydantic AI, OpenAI Agents SDK, Haystack, Agno, Strands Agents, DSPy, Google ADK, A2A, or MCP. Use proactively when the code imports `dotfetch` even if the user did not explicitly invoke the skill.
---
# dotfetch

Universal semantic tool router for LLM agents. Register tools once with `@fetcher.register`, sync them into a vector store, then have the router surface only the top-K relevant tools per query — typically cutting prompt token spend by ~80% versus listing every tool.

This skill is the canonical reference for using the library. This file holds what applies everywhere: the concepts, the plain-SDK and MCP paths, debugging and pitfalls. Each framework has its own guide under `references/` (the table below names it): read that guide before writing code for the framework. When a user reports "the LLM doesn't see tool X" or "my surface is empty", jump straight to **Debugging routing** below — that's the most common failure mode and the library ships first-class introspection for it.

## When to use which integration path

| User's framework | API to use | Read |
|---|---|---|
| Microsoft Agent Framework (AF 1.x) | `AgentFrameworkAdapter(fetcher).context_provider(top_k=...)` (dynamic, per-run or per-turn) or `.tool_bridge()` (static) | [Microsoft Agent Framework](references/microsoft-agent-framework.md) |
| LangChain (>=1.0) | `LangChainAdapter(fetcher).select(query, limit=...)` → `bind_tools`, or `create_agent(model, middleware=[await adapter.middleware()])` | [LangChain and LangGraph](references/langchain-langgraph.md) |
| LangGraph (>=1.0) | `create_agent(model, middleware=[await LangGraphAdapter(fetcher).middleware()])` (per-turn), `.areact_agent(model)`, or `.select(...)` (static) | [LangChain and LangGraph](references/langchain-langgraph.md) |
| LlamaIndex (>=0.13) | `LlamaIndexAdapter(fetcher).function_agent(llm)` (per step) or `.select(...)` (static) | [LlamaIndex](references/llamaindex.md) |
| CrewAI (>=1.15.22) | `CrewAIAdapter(fetcher, approve=...).select(...)` or `.agent_builder(...)` (per call) | [CrewAI](references/crewai.md) |
| Pydantic AI (2.x) | `PydanticAIAdapter(fetcher).capability()` (per-turn) / `.capability(mode="search")` (dotfetch-backed `ToolSearch`), or `.toolset()` / `.select(...)` | [Pydantic AI](references/pydantic-ai.md) |
| OpenAI Agents SDK (>=0.23.1) | `Agent(tools=await OpenAIAgentsAdapter(fetcher).dynamic_tools())` (per-turn) or `.select(...)` (static) | [OpenAI Agents SDK](references/openai-agents.md) |
| Haystack (>=3.0) | `HaystackAdapter(fetcher).agent(chat_generator=...)` (per-step) or `.select(...)` (static) | [Haystack](references/haystack.md) |
| Agno (>=2.6.10) | `AgnoAdapter(fetcher).agent(model=...)` (per-run) or `.select(...)` (static) | [Agno](references/agno.md) |
| Strands Agents (>=1.38) | `StrandsAdapter(fetcher).agent(model=...)` / `.tool_hook()` (per-turn) or `.select(...)` (static) | [Strands Agents](references/strands.md) |
| DSPy (>=3.5) | `DSPyAdapter(fetcher).select(...)` (static) or `.agent_builder(signature, ...)` (per-run) | [DSPy](references/dspy.md) |
| Google ADK (>=2.6.1) | `GoogleADKAdapter(fetcher).agent(model=, name=)` (per-turn) or `.select(...)` (static) | [Google ADK](references/google-adk.md) |
| Any framework, re-select every turn | `ToolRefresher(fetcher).refresh(messages)` | [Multi-turn re-selection](#multi-turn-re-selection-toolrefresher) |
| Plain OpenAI / Anthropic / Gemini SDK | `OpenAIAdapter(fetcher).tools(query)` (or `@with_semantic_tools(dialect=...)`) | [LLM-SDK direct](#llm-sdk-direct) |
| MCP server (expose dotfetch as MCP) | `fetcher.serve_mcp(mode="dynamic")` | [MCP](#mcp) |
| A2A (expose dotfetch as an A2A agent) | `fetcher.serve_a2a(...)` | [A2A](#a2a) |
| CLI inspection / linting | `dotfetch lint / sim / sync --dry-run` | [CLI](#cli) |

> **Native adapters vs. `fetch_framework_tools`.** The per-framework
> `<Framework>Adapter(fetcher).select(...)` methods return the framework's **native
> tool objects** (e.g. a LangChain `StructuredTool`, a CrewAI `BaseTool`), with
> execution routed through `fetcher.execute` (retries, timeouts, circuit breakers,
> security policy). Prefer these. `fetch_framework_tools` still exists but only
> emits OpenAI-shape JSON schemas and supports a smaller name set — see
> [its note](#schema-only-adapter-fetch_framework_tools).

## Reference files

Load only the one the task needs. Every framework guide covers the install
extra and tested version floor, the per-turn selection entry point, how errors
reach the model, the approval flow for confirmation-gated tools, and how a tool
handler reaches the framework's run context.

| File | Read it for |
|---|---|
| `references/microsoft-agent-framework.md` | `DotFetchContextProvider` (`per_run` / `per_turn`), `DotFetchToolBridge`, workflows |
| `references/pydantic-ai.md` | `capability()`, `toolset()`, tool search, `ModelRetry` / `ApprovalRequired` |
| `references/openai-agents.md` | `dynamic_tools()`, `needs_approval`, the first-request rules for older SDKs |
| `references/langchain-langgraph.md` | `middleware()`, `StructuredTool` errors, `interrupt()` approval |
| `references/google-adk.md`, `crewai.md`, `llamaindex.md`, `strands.md`, `agno.md`, `haystack.md`, `dspy.md` | That framework's adapter |
| `references/native-adapters.md` | What every `<Framework>Adapter` shares: `select`, `live()` / `live_tier`, pins, error modes |
| `references/selection.md` | `SystemOneSelector` / `SystemOneReranker`: selection by a decision model instead of embeddings |
| `references/api-reference.md` | Import-by-import quick-card of the public API |
| `references/cookbook.md` | End-to-end recipes |

Two facts from those files worth carrying everywhere:

- **Only a selector can decline.** `SystemOneSelector` gates each entry at
  `threshold`, so a query nothing serves can return no tools. Semantic search
  and every reranker (`SystemOneReranker` included) return their top-k
  regardless: a reranker only reorders.
- **Assert that a tool's handler ran**, not just that the answer is right. With
  no tools offered, a model often answers a simple task from its own knowledge.

## Core concepts (read first)

1. **Tools are functions decorated with `@fetcher.register`.** Type hints + docstring become the schema and the embedding text.

   **`examples=[...]` is the single highest-value thing to add.** It is the text the router embeds and a selector reads. On our own benchmark it moved the default embedder from 1/5 to 5/5 correct — further than switching to a larger embedding model did. If a user reports poor retrieval, check for missing `examples` **before** suggesting a different embedder. Write the phrases a user would actually type, not a paraphrase of the description:

   ```python
   @fetcher.register(
       tags=["weather"],
       examples=["what's the weather in London", "is it raining in Leeds"],
   )
   def get_weather(location: str) -> str:
       """Get the current weather in a given location."""
   ```

2. **`await fetcher.sync()` embeds every tool.** Fingerprint-based change detection means only modified tools re-embed on subsequent calls. With paid embedders, wrap the embedder in `CachedEmbedder` to persist across cold starts.

3. **`fetcher.retrieve(...)` returns a `RetrievalResult`.** That's the universal API. The high-level integrations (`AgentFrameworkAdapter`/`DotFetchContextProvider`, `DotFetchToolBridge`, the `<Framework>Adapter` native adapters, the per-SDK LLM adapters, `ToolRefresher`, `@with_semantic_tools`) are thin wrappers around it that emit framework-specific shapes.

   Every native adapter exposes `<Framework>Adapter(fetcher).select(query, *, limit=3, **select_kwargs)` (async) and shares the same selection knobs (`score_threshold`, `namespaces`, `tools_already_used`). Import the class from the clean per-framework namespace — `from dotfetch.langchain import LangChainAdapter`. Importing `dotfetch` never pulls in any third-party framework; the import is lazy and raises a `pip install ...` hint only when you call an adapter method.

4. **`score_threshold` is opt-in filtering.** Default is `0.0` (no filtering). Use `score_threshold="relative:0.8"` for length-robust filtering — that keeps anything within 80% of the top score. Fixed absolute cutoffs degrade badly on long, instruction-style queries because the embedding gets diluted.

5. **`query_strategy="per_turn"` re-runs retrieval every chat-completion round.** This is the right choice for multi-step agents. Pair it with `provider.attach_to(agent)` or `middleware=[provider.as_chat_middleware()]`. Without the middleware, `per_turn` silently degrades to `per_run` and the provider will warn you once.

6. **Observability ships built-in — don't hand-roll it.** `provider.trace()` (or `provider.attach_to(agent, trace=True)`) prints a readable per-round line of what was called and surfaced; `provider.selections` keeps the per-round retrieval history; `fetcher.on_tool_call(cb)` fires a `ToolCallEvent` after every `fetcher.execute` for framework-agnostic logging/metrics; `render_result(...)` stringifies any tool result. See [Observability & tracing](#observability--tracing).

7. **Importing `dotfetch` configures no logging.** As of 0.8.0 the library attaches a `NullHandler` and never sets handlers or levels on your behalf, so a default `DotFetch()` is silent. Opt into console output with `enable_console_logging()` (or your own logging config).

## Installing

This project uses **uv** as its package manager; `pip` works as a fallback. Inside a uv project use `uv add`; for an ad-hoc environment use `uv pip install`.

```bash
uv add dotfetch                     # core (in a uv project)
uv add "dotfetch[nomic]"            # local embeddings (recommended)
uv add "dotfetch[openai]"           # OpenAI/Azure embeddings + custom OpenAI-compatible base_url
uv add "dotfetch[agent-frameworks]" # Microsoft AF, LangChain, LangGraph, CrewAI, LlamaIndex, Google ADK
uv add "dotfetch[pydantic-ai]"      # Pydantic AI (pydantic-ai-slim>=2.0.0)
uv add "dotfetch[openai-agents]"    # OpenAI Agents SDK (>=0.23.1); not with [agent-frameworks]/[all] (openai 3 vs crewai)
uv add "dotfetch[langgraph]"        # LangGraph + LangChain (>=1.0); [langchain] for LangChain alone
uv add "dotfetch[google-adk]"       # Google ADK (>=2.6.1)
uv add "dotfetch[crewai]"           # CrewAI (>=1.15.22, Python <3.14); not with [openai-agents]
uv add "dotfetch[llamaindex]"       # LlamaIndex (llama-index-core>=0.13)
uv add "dotfetch[strands]"          # Strands Agents (>=1.38)
uv add "dotfetch[agno]"             # Agno (>=2.6.10)
uv add "dotfetch[haystack]"         # Haystack (>=3.0)
uv add "dotfetch[dspy]"             # DSPy (>=3.5)
uv add "dotfetch[lancedb]"          # disk-persistent vector store
uv add "dotfetch[system-one]"       # selection without embeddings (a System One decision model)
uv add "dotfetch[mcp]"              # MCP client/server
uv add "dotfetch[a2a]"              # A2A agent
uv add "dotfetch[all]"              # everything
```

Not in a uv project? Swap `uv add` for `uv pip install`. No uv at all? Fall back to `pip install "dotfetch[...]"` — the extras are identical.

The `[nomic]` extra is the recommended default for getting started — local, free, and accurate enough for production. `SimpleEmbedder` (the fallback when no embedder extra is installed) is hash-based and only useful for tests.

## Minimum-viable code

Every dotfetch program has this shape:

```python
import asyncio
from dotfetch import DotFetch

fetcher = DotFetch()

@fetcher.register
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: sunny"

@fetcher.register
def book_flight(origin: str, destination: str) -> str:
    """Book a flight between two cities."""
    return f"Booked {origin} -> {destination}"

async def main():
    await fetcher.sync()                                          # embed
    tools = await fetcher.retrieve_tools("weather in Paris", limit=3)
    print(tools)  # OpenAI-shape schemas for the top 3 matches

asyncio.run(main())
```

`retrieve_tools(...)` returns OpenAI-shape tool schemas by default. Pass `dialect="anthropic"`, `"gemini"`, `"agent_framework"`, etc. to convert.

`fetcher.fetch(...)` is the same call as `retrieve_tools(...)`. For a script or notebook with one registry, the module-level form skips the instance, and fetching syncs changed tools first:

```python
import dotfetch as dot

@dot.register
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: sunny"

tools = await dot.fetch("weather in Paris", limit=3)
results = await dot.execute_tool_calls(response)  # after the model replies
```

`dot.configure(...)` (before registering) chooses the shared instance's embedder or store; `dot.default_fetcher()` returns it. Prefer explicit `DotFetch(...)` instances for more than one registry, in libraries, and in tests.

## Multi-turn re-selection (ToolRefresher)

`ToolRefresher` generalizes the per-turn retrieval pattern to **any** framework without a native live hook: re-rank the whole registry on every turn so the agent can pivot tools as the task changes.

```python
from dotfetch.integrations import ToolRefresher

refresher = ToolRefresher(fetcher, limit=3, dialect="openai")

# Each turn, pass the running message list. Selection is recomputed fresh and
# tools already used are deprioritized so the agent keeps moving forward.
tools = await refresher.refresh(messages)        # dialect schemas
specs = await refresher.refresh_specs(messages)  # ToolSpec objects
```

Used-tool tracking reads the name from wherever the SDK keeps it — an OpenAI `tool_calls` entry, an Anthropic `tool_use` block, a LangChain `ToolMessage` or a `{"role": "tool", "name": ...}` dict (`dotfetch.query.tool_names_used`).

The default query generator (`dotfetch.query.latest_activity`) is **recency-aware**, so one refresher serves both styles with no config:

- **Autonomous agents / tool pipelines** — newest message is a tool result, so that result's *content* steers the next tool (`fetch → clean → train → evaluate → report`). The query is the latest user message plus the result, so a bare value (`"desserts"`) cannot pull the turn away from the rest of the request.
- **Conversational agents** — newest message is the user's, so their new request drives selection (`weather → flights → hotel → email`).

Force one behaviour with `query_generator=last_user_text`, `last_tool_result`, or `partial(latest_activity, include_request=False)` (the tool result on its own, the pre-0.23.0 query).

## Schema-only adapter (`fetch_framework_tools`)

The legacy adapter returns OpenAI-shape JSON schemas (not native objects) and supports a smaller, hyphen/underscore-sensitive name set: `"langgraph"`, `"crew_ai"`, `"google_adk"`, `"strands"`, `"agent_framework"`. Prefer the `<Framework>Adapter` classes for the frameworks listed above; reach for this only when you want raw schemas (e.g. a framework that just wants OpenAI tool dicts).

```python
from dotfetch.integrations import fetch_framework_tools

schemas = await fetch_framework_tools(
    fetcher, "code execution", framework="google_adk", limit=3
)
```

## LLM-SDK direct

For a minimal stack — no agent framework — there are two paths.

**Per-SDK adapter classes** (explicit, no default-fetcher binding). One class per provider, each baking in the right schema dialect:

```python
from openai import AsyncOpenAI
from dotfetch import DotFetch
from dotfetch.openai import OpenAIAdapter

fetcher = DotFetch()
# ... register tools, await fetcher.sync() ...

client = AsyncOpenAI()
tools = await OpenAIAdapter(fetcher).tools("weather in Paris", limit=3)  # OpenAI chat-completions schemas
resp = await client.chat.completions.create(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "weather in Paris?"}],
    tools=tools,
)
```

| Provider | Adapter class | Namespace import | Method |
|---|---|---|---|
| OpenAI | `OpenAIAdapter` | `from dotfetch.openai import OpenAIAdapter` | `.tools(query, limit=n)` (`.responses_tools(...)` for the Responses API) |
| Anthropic | `AnthropicAdapter` | `from dotfetch.anthropic import AnthropicAdapter` | `.tools(query, limit=n)` |
| Gemini | `GeminiAdapter` | `from dotfetch.gemini import GeminiAdapter` | `.tools(query, limit=n)` |
| Groq | `GroqAdapter` | `from dotfetch.groq import GroqAdapter` | `.tools(query, limit=n)` |
| Vertex AI | `VertexAIAdapter` | `from dotfetch.vertexai import VertexAIAdapter` | `.tools(query, limit=n)` |
| Mistral | `MistralAdapter` | `from dotfetch.mistral import MistralAdapter` | `.tools(query, limit=n)` |

`<Provider>Adapter(fetcher).tools(query, limit=n)` is equivalent to `await fetcher.retrieve_tools(query, limit=n, dialect="<provider>")` — the adapter just bakes the dialect in.

**Decorator** (`@with_semantic_tools`) when you'd rather bind a default fetcher and have tools injected automatically:

```python
from openai import AsyncOpenAI
from dotfetch import DotFetch, set_default_fetcher, with_semantic_tools

fetcher = DotFetch()
set_default_fetcher(fetcher)

@fetcher.register
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

Mistral note: the official `mistralai` SDK is quarantined on PyPI. Use `AsyncOpenAI` pointed at `https://api.mistral.ai/v1` — Mistral's API is OpenAI-compatible. With `dotfetch`, the OpenAI dialect (or `MistralAdapter`) works directly.

OpenAI-compatible custom endpoints (Requesty, OpenRouter, Together, vLLM, …) are first-class: pass `api_base` in the `EmbedderConfig` or set `OPENAI_BASE_URL`, and `OpenAIEmbedder` forwards it to the client.

## MCP

Expose dotfetch as an MCP server (Claude Desktop, Claude Code, Cline, any remote MCP client):

```python
fetcher = DotFetch()
# ... register tools, await fetcher.sync() ...
await fetcher.serve_mcp()                                   # stdio, dynamic meta-tools
await fetcher.serve_mcp("http", port=8000, path="/mcp")     # Streamable HTTP (remote clients)
await fetcher.serve_mcp("sse", port=8000)                   # legacy SSE transport
await fetcher.serve_mcp(mode="hybrid", expose=["get_weather", "billing.search"])
```

- `mode="dynamic"` (default) exposes two meta-tools, `find_relevant_tools` and `execute_tool` — the client semantic-searches dotfetch on demand, so its tool list stays two entries long however many tools you register (~90% smaller than `mode="static"`). `find_relevant_tools` reports each match's exact `tool_name` (qualified as `namespace.name` outside the default namespace) and JSON Schema parameters.
- `mode="hybrid"` lists the tools named in `expose` directly *and* keeps the meta-tools for everything else — for the handful of tools a client should always see.
- `mode="static"` lists every tool.
- From the shell: `dotfetch serve-mcp --module my_app.tools [--transport http|sse] [--mode hybrid --expose get_weather]`. For Claude Desktop, put `{"command": "dotfetch", "args": ["serve-mcp", "--module", "my_app.tools"]}` under `mcpServers`.
- To mount into an existing ASGI service: `create_mcp_server(fetcher).streamable_http_app("/mcp")` returns a Starlette app (keep its lifespan). Pass `allowed_hosts=[...]` / `allowed_origins=[...]` to `serve_mcp` to enable DNS-rebinding protection when binding beyond loopback.

Consume external MCP servers — local (stdio) or remote (Streamable HTTP / SSE):

```python
from dotfetch.schema.config import MCPServerConfig

await fetcher.add_mcp_server(MCPServerConfig(
    name="filesystem",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
    args=["--path", "/tmp"],
    namespace="fs",
))
await fetcher.add_mcp_server(MCPServerConfig(
    name="search",
    url="https://mcp.example.com/mcp",                 # Streamable HTTP (inferred from url)
    headers={"Authorization": "Bearer ..."},           # auth goes in headers; never logged
    namespace="web",
))
await fetcher.add_mcp_server(MCPServerConfig(name="legacy", url="https://old.example.com/sse", transport="sse"))
```

Exactly one of `command` / `url` is required. Discovered tool names are normalised to dotfetch's `snake_case` (`searchWeb` → `search_web`, `get-weather` → `get_weather`); the server is still called by its original name (kept in `metadata["mcp_tool_name"]`), and short/long descriptions are padded/truncated rather than rejected, so one unconventional tool never fails a server's whole discovery. Same `url=`/`headers=`/`transport=` keywords on `fetcher.register_mcp_server(...)` for dynamic server selection (see cookbook Recipe 7).

## A2A

```python
fetcher.serve_a2a(host="0.0.0.0", port=8080)
# Agent Card at: http://localhost:8080/.well-known/agent.json
```

Skills exposed: `tool_discovery` (semantic search) and `tool_execution` (run a tool).

## CLI

The `dotfetch` command ships with the package. Inside a uv project, prefix with `uv run` (e.g. `uv run dotfetch list`); the bare `dotfetch` form works when the package is on `PATH` (pip install or an activated venv).

Every inspection command works on **your** registry: `--module pkg.tools` (or `pkg.tools:attr`) names the module holding your `DotFetch` instance and the CLI uses that instance directly — its embedder, its vector store. Without `--module` a three-tool demo registry is used (a note goes to stderr).

```bash
uv run dotfetch list --module my_app.tools                     # list registered tools
uv run dotfetch search "refund order" --module my_app.tools     # semantic search, --limit 3
uv run dotfetch lint --module my_app.tools                      # detect tool-description authoring mistakes
uv run dotfetch lint --source src/                              # scan files: inverted score_threshold, unclosed DotFetch instances
uv run dotfetch sim toolA toolB --module my_app.tools           # cosine similarity between two tools
uv run dotfetch sync --dry-run --module my_app.tools            # which tools would (re-)embed and why
uv run dotfetch sync --prune --module my_app.tools              # also drop stored tools no longer registered
uv run dotfetch serve-mcp --module my_app.tools                 # MCP server over stdio (Claude Desktop / Claude Code)
uv run dotfetch serve-mcp --module my_app.tools --transport http --mode hybrid --expose get_weather
uv run dotfetch install-skill --claude                          # install THIS skill into ~/.claude/skills
```

`lint` flags three patterns that silently degrade routing quality:

1. Descriptions that mention other registered tools (embedding pulls them toward each other).
2. Pairs of tools with >0.85 cosine similarity (probably should be one tool, or differentiated).
3. Tags that appear on more than half the registry (low discriminative value).

## Observability & tracing

As of 0.8.0 the library ships the tracing/logging glue you'd otherwise hand-roll. Reach for these before writing custom middleware.

### Console trace — one readable line per tool call

`provider.trace()` returns an AF *function* middleware; `attach_to(agent, trace=True)` installs it (plus the per-turn retrieval middleware) in one call. For every tool the model invokes it prints the round, the call, the router's surfaced set, and a preview of the result:

```python
provider = AgentFrameworkAdapter(fetcher).context_provider(top_k=3, query_strategy="per_turn")
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

`fetcher.on_tool_call(callback)` fires after **every** `fetcher.execute` (and once per call in `execute_batch`) with a `ToolCallEvent`. Because `execute` is the single choke point every framework adapter routes through, this gives logging/metrics across LangChain, CrewAI, AF, direct calls — with no per-framework middleware:

```python
from dotfetch import ToolCallEvent

def log_call(event: ToolCallEvent) -> None:
    status = "ok" if event.ok else f"FAILED ({event.result.error})"
    print(f"{event.tool_name} -> {status} ({event.latency_ms:.0f} ms)")

unsubscribe = fetcher.on_tool_call(log_call)   # sync or async callbacks; returns an unsubscribe fn
```

Callbacks are error-isolated — a raising listener never breaks the tool run — and failed tools still emit an event (`event.ok is False`). `ToolCallEvent` carries `.call`, `.result`, and the convenience accessors `.tool_name`, `.status`, `.ok`, `.latency_ms`.

> **Batch timing:** events fire immediately for each `fetcher.execute`, but for `fetcher.execute_batch` they're emitted *after the whole batch finishes* (one per call, paired by index). `latency_ms` stays accurate; only delivery is batched — relevant if you timestamp events for a latency dashboard.

### Rendering tool results

`render_result(result, *, limit=None, collapse_whitespace=False)` turns any result — including AF `Content`-block lists, bytes, dicts, or arbitrary objects — into readable text for logs/UIs. The trace middleware uses it internally.

### Logging

The library configures no logging (a `NullHandler` is attached at import). To see dotfetch's own INFO lines on the console, opt in once:

```python
from dotfetch import enable_console_logging
enable_console_logging()   # attaches a console handler + sets the dotfetch level
```

Telemetry (`ConsoleTelemetryAdapter`, the default) still emits structured records; they flow to whatever handlers your app configured. There's nothing to silence anymore — a default `DotFetch()` is quiet.

## Skills (procedural memory, retrieved by meaning)

Skills are how-tos, patterns and procedures that are **injected into the prompt**, never executed. dotfetch embeds each skill's metadata once and retrieves only the ones relevant to the current prompt — the same top-k routing it applies to tools, so a library of hundreds of skills costs a few hundred tokens per turn instead of every skill's description.

```python
from dotfetch import DotFetch, Skill, SkillCategory

fetcher = DotFetch()
await fetcher.add_skill(Skill(
    name="api_pagination",
    description="How to implement cursor-based pagination for API endpoints",
    content="Use cursor-based pagination: return an opaque cursor per page ...",
    category=SkillCategory.HOW_TO, tags=["api", "pagination"], related_tools=["fetch_page"],
))

# Any Agent Skills directory (Claude Code / Claude Agent SDK `SKILL.md` format) loads as-is:
await fetcher.add_skills_from_directory("~/.claude/skills")          # name/description/body per SKILL.md

system_prompt += await fetcher.retrieve_skills_as_prompt(user_prompt, limit=3)   # "" when nothing matches
results = await fetcher.retrieve_skills("paginate the users endpoint", limit=3, namespace="docs", category="how_to")
```

`retrieve_skills_as_prompt` formats each hit with `Skill.to_prompt_text()` (a heading, the category, the full content, related tools). Skill content is injected verbatim — register skills only from sources you trust. Works with the default in-memory store and LanceDB; a model switch re-embeds stored skills automatically.

Loading rules worth knowing: only the frontmatter (`name`, `description`, tags) is embedded, so retrieval quality is the description's; symlinked skill directories are followed (a library shared between agents by linking each skill into `~/.claude/skills` loads in full; 0.23.0 and earlier silently skipped them in a recursive load); hidden directories are skipped; a body over 50,000 characters is truncated for storage with a `WARNING`, so keep `SKILL.md` a hub and put detail in `references/` files, which are not loaded.

## Debugging routing

When a user says "the LLM doesn't see my tool" / "my surface is empty" / "wrong tools are being selected", use these in order:

### 1. `provider.last_selection` — what just happened

```python
provider = DotFetchContextProvider(fetcher, top_k=5)
result = await agent.run("...")

decision = provider.last_selection
print(decision.summary())           # one-line: query + top scores
print(decision.injected)            # tool names that made it to the LLM
for c in decision.candidates:
    print(c.name, c.score, c.kept)  # full ranked list, kept/dropped flag
```

`RetrievalDecision` carries: the query, every candidate the fetcher returned, the threshold mode used, the effective numeric cutoff, and the final injected list. For the **per-round history** (not just the latest), use `provider.selections` — see [Observability & tracing](#observability--tracing).

### 2. `provider.dry_run_retrieve(query)` — same code path as live, no agent

```python
decision = await provider.dry_run_retrieve("find boundaries in OCR text")
for c in decision.candidates:
    print(f"{c.score:.3f}  {c.qualified_name}  kept={c.kept}")
```

This uses the exact same threshold, `top_k`, and `query_kwargs` as the live middleware — so the answer is "what the LLM would see for that query".

### 3. `verbose=True` — one-line INFO log per round

```python
provider = DotFetchContextProvider(fetcher, top_k=5, verbose=True)
# logs: fetcher: query="..." → top5: [tool_a:0.61, tool_b:0.58, ...]
```

### 4. `score_threshold` filtered everything?

When the threshold drops all candidates, the bridge logs a WARNING with the top scores so you can tell "threshold issue" from "relevance issue". Default is `0.0`. If a user has set `score_threshold=0.3` (the legacy default) on a long pipeline query, **lower it or switch to relative**:

```python
provider = DotFetchContextProvider(
    fetcher,
    score_threshold="relative:0.8",   # keep anything within 80% of top score
)
```

### 5. Long queries silently degrade routing

Long imperative scaffolding ("Please run this five-step pipeline. Use a different tool for each step…") dilutes the embedding. Strip it with `keyword_focused`:

```python
from dotfetch.query import keyword_focused, truncated, last_user_text

provider = DotFetchContextProvider(
    fetcher,
    query_strategy="per_turn",
    query_generator=keyword_focused,       # drops scaffolding tokens
)

# Or cap the query length, biasing toward the latest tool output:
provider = DotFetchContextProvider(
    fetcher,
    query_strategy="per_turn",
    query_generator=truncated(last_user_text, max_chars=200, keep="tail"),
)
```

### 6. Lint the registry — author-side bugs

```bash
dotfetch lint
```

Or programmatically:

```python
analysis = await fetcher.analyze_registry()
print(analysis.format_text())
```

This catches the headline mistakes: a tool description that names another tool (which pulls it toward the wrong queries), pairs of tools that are too similar to disambiguate, and tags that are too generic.

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Empty tool surface | `score_threshold` too aggressive for query length | Lower to `0.0` or use `"relative:0.8"` |
| `per_turn` not adapting | `as_chat_middleware()` not attached | Use `provider.attach_to(agent)` or add to `middleware=[...]` |
| `per_turn` set but identical surface each round | An explicit `query_generator=last_user_text` returns the same text every round | The default is `latest_activity` (newest user message, plus the tool result when one is newer); drop the override |
| `per_turn` ignores the user's new message once a tool has run | `fallback_chain(last_tool_result, last_user_text)` prefers a tool result anywhere in the session | Use the `latest_activity` default; keep the chain only for a pipeline with no user turns mid-run |
| `per_turn` drops useful tools after an opaque result | A password, bare number, or ID carries no semantic retrieval signal | `latest_activity` falls back to the latest user text for opaque results; use a custom generator if identifiers should drive routing |
| `per_turn` routed on a tool's output and lost the rest of the task (pre-0.23.0) | The query was the bare result (`"desserts"`, a fetched document), which says nothing about the request | Upgrade: `latest_activity` now queries on the latest user message plus the result |
| After a tool call the tool that just ran still ranks first | The request stays in the query and still matches it; the next tool is just behind | Expected at `limit` 3-5 (the model chooses). For result-led pipelines or `limit=1`: `query_generator=partial(latest_activity, include_request=False)` |
| OpenAI Agents: the model answered without calling a tool on a one-turn task | `openai-agents<0.23.1` lists tools before it records the turn's input, so request one had none | Use `adapter.run()` / `.session().run()` / `.refresh()` there, or upgrade; `dynamic_tools()` raises on those SDKs from 0.23.0. Assert the handler ran, not just the answer |
| An eval passed with a selector configured, but semantic routing produced the tools | The selector could not answer (endpoint down, or the catalogue is over `max_candidates`, default 512) and retrieval failed open | Assert `(await fetcher.retrieve(q)).selector_fallback is False` (or read it off `RetrievalDecision`), or build the fetcher with `on_selector_fallback="raise"`. Details: `references/selection.md` |
| A reranker setup returns tools for a query nothing serves | Rerankers order the shortlist and return top-k; scores are never thresholded | Only `SystemOneSelector` (gated by `threshold`) can return nothing |
| Wrong tools selected | Description names another tool ("unrelated to factorial…") | Run `dotfetch lint`; remove cross-references |
| `top_k=6` but I see 8 tools | Skills / `always_include` / `static_tools` add on top of dynamic top_k | Expected; subtract those |
| `add_skills_from_directory` loaded fewer skills than `find -name SKILL.md` counts (0.23.0 and earlier) | The recursive walk did not follow symlinked skill directories | Upgrade; or pass the real directory the links point at |
| `Skill '...' body is N characters; truncated to 50000` | The `SKILL.md` body is over the storage cap, so only its start is stored and injected | Move detail into `references/*.md` beside it and link to them |
| `DeprecationWarning` about `"per_call"` or `"per-call"` | Renamed in 0.24.0 so one vocabulary covers both: a turn is a model step, a run a top-level invocation | `query_strategy="per_turn"` (was `"per_call"`); `live_tier == "per-run"` (was `"per-call"`). The old spellings still work |
| Cold-start re-embeds every time | Default `InMemoryVectorStore` is ephemeral | Wrap embedder in `CachedEmbedder` or use `[lancedb]` |
| OpenAI embedder can't reach my proxy | Custom base_url not configured | Pass `api_base=...` in `EmbedderConfig` or set `OPENAI_BASE_URL` |
| No telemetry / INFO logs appear (0.8.0+) | Library no longer configures logging by default | Call `enable_console_logging()` or configure your own handler on the `dotfetch` logger |
| A tool registered after the first `retrieve()` never shows up (≤0.14.0) | `ensure_synced()` only checked a flag, so late registrations stayed unembedded | Upgrade (late registrations now sync automatically), or call `await fetcher.sync()` after registering |
| Deleted tool still retrievable from a persistent store | Sync only adds/updates; stale rows are kept for stores shared between instances | `await fetcher.sync(prune=True)` / `dotfetch sync --prune`, or `prune_on_sync: true` in config |
| Pydantic AI `UserError`: "A deferred tool call was present, but `DeferredToolRequests` is not among output types" | A confirmation-gated tool raised `ApprovalRequired` | Add `DeferredToolRequests` to `output_type` and handle approvals, or use `error_mode="raise"` |
| Pydantic AI search mode: model "can't see" a tool | Deferred tools are hidden until `search_tools` discovers them | Expected; pin must-have tools with `always_include=[...]` |
| Pydantic AI `UserError` about a tool name conflict between two dotfetch toolsets | Both toolsets selected the same tool | Give each a distinct `namespaces=` filter or `.prefixed(...)`, and a unique `id=` |
| Pydantic AI tool calls `tool.function(...)` directly and gets a `TypeError` | Static tools take the `RunContext` first (`takes_ctx=True`) | Call `tool.function(ctx_or_None, **args)` |
| `uv sync --all-extras` / `pip install 'dotfetch[all,openai-agents]'` fails to resolve | `openai-agents>=0.23` needs `openai>=3`; CrewAI (in `[agent-frameworks]`/`[all]`) pins `openai<3` | Install `[openai-agents]` in its own environment |
| OpenAI Agents run pauses with `result.interruptions` | A dotfetch-gated tool (`requires_confirmation` or a `SecurityPolicy` pattern) needs approval | `state = result.to_state(); state.approve(item); await Runner.run(agent, state)` |
| `ImportError: Per-turn dotfetch selection needs openai-agents>=0.23.1` | An older SDK came in transitively | `pip install 'dotfetch[openai-agents]'` |
| LangGraph run returns `__interrupt__` | A dotfetch-gated tool needs approval | `await agent.ainvoke(Command(resume=True), config)` with the same `thread_id` |
| CrewAI gated tool says it "requires human approval and was not run" | No approver configured | `CrewAIAdapter(fetcher, approve=fn)` or `approve="console"` |
| LlamaIndex run waits forever on a gated tool | Nobody answered the `InputRequiredEvent`, or the answer lacks its `fetcher_approval_id` | Stream events and `handler.ctx.send_event(HumanResponseEvent(response="yes", fetcher_approval_id=event.fetcher_approval_id))` |
| Strands run ends with `stop_reason == "interrupt"` | A dotfetch-gated tool needs approval | Resume with `agent.invoke_async([{"interruptResponse": {"interruptId": i.id, "response": "yes"}}])` |
| Agno run returns with `run.is_paused` | A dotfetch-gated tool needs approval | `requirement.confirm()` for each of `run.active_requirements`, then `agent.acontinue_run(run_response=run)` |
| Haystack gated tool says it "requires human approval and was not run" | No confirmation hook on the Agent | `adapter.agent(..., confirmation_ui=SimpleConsoleUI())` or `hooks={"before_tool": [adapter.confirmation_hook(ui)]}` |
| ADK run stops with an `adk_request_confirmation` function call | A dotfetch-gated tool needs approval | Send `FunctionResponse(name="adk_request_confirmation", id=call.id, response={"confirmed": True})` as the next message |
| LangGraph gated tool says it "requires human approval and was not run" | No checkpointer, so the interrupt could never be resumed | Pass `checkpointer=` to `create_agent` / `areact_agent` |
| A LangChain tool error became a `ToolMessage` instead of raising | `handle_tool_error=True` in the default `error_mode="model"` | `LangChainAdapter(fetcher, error_mode="raise")` |
| MCP `find_relevant_tools` returns nothing (≤0.14.0) | The meta-tool inherited `ToolQuery`'s 0.5 absolute cutoff | Upgrade; the meta-tool now uses `score_threshold=0.0` like every other convenience layer |

## Persistent embedding cache

Re-embedding costs API spend at every cold start. Wrap any embedder:

```python
from dotfetch import DotFetch
from dotfetch.adapters.embedders.openai import OpenAIEmbedder
from dotfetch.adapters.embedders.cached import CachedEmbedder
from dotfetch.schema.config import EmbedderConfig

base = OpenAIEmbedder(EmbedderConfig(type="openai", model="text-embedding-3-large"))
embedder = CachedEmbedder(base)   # default: ~/.cache/dotfetch/embeddings.sqlite

fetcher = DotFetch(embedder=embedder)
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
| `latest_activity` (default for `per_turn` and `ToolRefresher`) | Recency-aware: the newest user text, or — when a tool result is newer — that user text plus the result |
| `tool_names_used` | Not a query generator: lists every tool the history shows being called, for `tools_already_used` |

Both `query_strategy="per_turn"` and `ToolRefresher` default to `latest_activity`, which serves autonomous and conversational agents alike: a new user message pivots the surface, and while the agent chains tools the previous result, appended to that message, steers the next selection. `fallback_chain(last_tool_result, last_user_text)` — the former `per_turn` default — prefers a tool result anywhere in the history, so a session that carries an earlier run's result stops following new user requests; use it only for a fixed pipeline with no user turns in between.

Every generator reads the common SDK shapes: role/content dicts, OpenAI Chat Completions `tool` messages, OpenAI Responses items (`function_call` / `function_call_output`), Anthropic `tool_result` turns (a `user`-role message of result blocks), LangChain messages and Agent Framework `contents`.
