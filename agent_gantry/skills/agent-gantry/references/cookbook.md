# Agent-Gantry Cookbook

Recipes for the patterns that come up most often when building real agents with Agent-Gantry. Each section is self-contained — jump to the one matching the user's question.

## Recipe 1: Multi-step pipeline with per-call retrieval

Goal: an agent that runs a fixed N-step pipeline (e.g. fetch → transform → validate → write), where each step needs a different tool, and the user wants the LLM to see only the relevant tool at each step.

```python
from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from agent_gantry import AgentGantry, GantryContextProvider
from agent_gantry.query import fallback_chain, last_tool_result, last_user_text
from agent_gantry.integrations.agent_framework_middleware import GantryToolChoiceMiddleware

gantry = AgentGantry()
# ... register fetch_*, transform_*, validate_*, write_* tools, await gantry.sync() ...

provider = GantryContextProvider(
    gantry,
    top_k=3,
    query_strategy="per_call",
    # Default for per_call already adapts to tool output; spelled out for clarity:
    query_generator=fallback_chain(last_tool_result, last_user_text),
    score_threshold="relative:0.8",
)

# Force tool calls for the first 4 rounds (the pipeline) then allow text on the
# 5th (summarisation) — keeps the model from bailing to text mid-pipeline.
rounds = {"n": 0}
def choose_tool_choice(_ctx):
    rounds["n"] += 1
    return "required" if rounds["n"] <= 4 else "auto"

agent = Agent(
    OpenAIChatClient(),
    "You run a 4-step pipeline. Use the suggested tool at each step.",
    context_providers=[provider],
    middleware=[
        provider.as_chat_middleware(),
        GantryToolChoiceMiddleware(choose_tool_choice),
    ],
)
```

## Recipe 1b: Native adapters + multi-turn re-selection (any framework)

Goal: route a large registry into a non-AF framework (LangChain, CrewAI, Pydantic AI, …) and re-select tools every turn — without hand-rolling schema plumbing.

Static slice (select once, native tool objects):

```python
from agent_gantry.langchain import LangChainAdapter   # clean per-framework namespace

tools = await LangChainAdapter(gantry).select("email the quarterly report", limit=3)
llm = ChatOpenAI(model="gpt-5.5").bind_tools(tools)   # native StructuredTools
```

Multi-turn (re-rank the whole registry each turn, deprioritising tools already used):

```python
from agent_gantry.integrations import ToolRefresher

refresher = ToolRefresher(gantry, limit=3, dialect="openai")

messages = [...]                      # the running conversation / tool-result log
while not done:
    tools = await refresher.refresh(messages)     # fresh selection this turn
    # ...call the model with `tools`, append its output to `messages`...
```

The default `latest_activity` query generator is recency-aware: a tool result drives the next selection in an autonomous pipeline (`fetch → clean → train → report`), while a new user message drives it in a chat agent (`weather → flights → hotel`). Pin one with `query_generator=last_user_text` or `last_tool_result`.

For frameworks with a native per-turn hook (LlamaIndex, Pydantic AI, AutoGen, Google ADK, LangGraph, Semantic Kernel, OpenAI Agents SDK), prefer the deep **live** adapter method instead of `ToolRefresher` — e.g. `from agent_gantry.llamaindex import LlamaIndexAdapter` then `LlamaIndexAdapter(gantry).function_agent(llm)`.

## Recipe 2: Pipe a custom embedding endpoint (Requesty / OpenRouter / vLLM)

```python
from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.openai import OpenAIEmbedder
from agent_gantry.schema.config import EmbedderConfig

embedder = OpenAIEmbedder(EmbedderConfig(
    type="openai",
    model="text-embedding-3-large",
    api_key="...",
    api_base="https://router.requesty.ai/v1",
))

gantry = AgentGantry(embedder=embedder)
```

Or via env var: `export OPENAI_BASE_URL=https://router.requesty.ai/v1`.

## Recipe 3: Persistent embedding cache (avoid cold-start cost)

```python
from agent_gantry import AgentGantry
from agent_gantry.adapters.embedders.openai import OpenAIEmbedder
from agent_gantry.adapters.embedders.cached import CachedEmbedder
from agent_gantry.schema.config import EmbedderConfig

base = OpenAIEmbedder(EmbedderConfig(type="openai", model="text-embedding-3-large"))
embedder = CachedEmbedder(base)  # default cache: ~/.cache/agent_gantry/embeddings.sqlite
gantry = AgentGantry(embedder=embedder)
```

Inspect cache effectiveness: `print(embedder.hits, embedder.misses)`.

## Recipe 4: Pinning AF-native tools alongside dynamic Gantry tools

```python
from agent_framework import tool

@tool
def log_event(message: str) -> str:
    """Write an audit event."""
    return "ok"

provider = GantryContextProvider(
    gantry,
    top_k=4,
    static_tools=[log_event],  # always visible to the LLM, never filtered
)
```

`static_tools` is the right slot for AF-native `@tool` callables that don't live in the gantry registry. For gantry-registered tools, use `always_include=["log_event"]` (by name) instead.

## Recipe 5: Diagnosing "the LLM doesn't see my tool"

The five-step debugging path the issue tracker keeps surfacing. Run them in order.

```python
# Step 1: Is the tool even registered?
print([t.name for t in gantry.list_tools_sync()])

# Step 2: Would semantic search find it for this query?
decision = await provider.dry_run_retrieve("the user's actual query")
for c in decision.candidates:
    print(f"  {c.score:.3f}  {c.qualified_name}  kept={c.kept}")

# Step 3: After a real run, what did the LLM actually see?
result = await agent.run("...")
print(provider.last_selection.summary())
print(provider.last_selection.injected)

# Step 4: Threshold issue?
# If "filtered out all N candidates" WARNING appears in logs, lower or switch
# to relative mode.

# Step 5: Author-side bug? Run the registry linter.
analysis = await gantry.analyze_registry()
print(analysis.format_text())
```

## Recipe 6: A2A — expose Gantry as an agent network endpoint

```python
gantry.serve_a2a(host="0.0.0.0", port=8080)
# Agent Card lives at http://localhost:8080/.well-known/agent.json
# Skills exposed: tool_discovery (semantic search), tool_execution (run a tool)
```

Consume another A2A agent's skills as gantry-managed tools:

```python
from agent_gantry.schema.config import A2AAgentConfig

await gantry.add_a2a_agent(A2AAgentConfig(
    name="translator",
    url="https://translator.example.com",
    namespace="ext",
))
# Translator's skills are now registered as gantry tools in the "ext" namespace.
```

## Recipe 7: MCP — dynamic server selection

```python
# Register servers with metadata so the router can pick which one(s) to connect to.
gantry.register_mcp_server(
    name="filesystem",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
    description="Read and write files on the local filesystem",
    tags=["filesystem", "files", "io"],
)
gantry.register_mcp_server(
    name="db",
    command=["python", "-m", "mcp_postgresql"],
    description="Query and mutate PostgreSQL databases",
    tags=["database", "sql"],
)

await gantry.sync_mcp_servers()

# Semantic search to pick servers, then connect only those:
servers = await gantry.retrieve_mcp_servers("read config.yaml", limit=1)
for s in servers:
    await gantry.discover_tools_from_server(s.name)

tools = await gantry.retrieve_tools("read my config.yaml")
```

## Recipe 8: Many-tool registry — keep recall high with the linter + verbose logging

For registries >50 tools, two practices keep routing quality high:

1. **CI lint**: add `agent-gantry lint` to your test pipeline.
2. **Verbose provider in staging**: `GantryContextProvider(gantry, verbose=True)` prints one line per round so accuracy regressions are caught before production.

## Recipe 9: Token-savings benchmarking

```python
from agent_gantry.integrations.agent_framework_bridge import GantryToolBridge

bridge = GantryToolBridge(gantry)
tools_full = bridge.wrap_tools(gantry.list_tools_sync())          # all tools
tools_top5 = await bridge.get_tools("the query", limit=5)         # gantry top-5

# Call your LLM with each list and compare prompt_tokens from the usage block.
```

`tests/test_token_savings_and_accuracy.py` in the repo is a full reference benchmark.
