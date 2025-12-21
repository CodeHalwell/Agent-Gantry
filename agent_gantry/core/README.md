# agent_gantry/core

The **core** package is the heart of Agent-Gantry. It owns the public facade, semantic routing,
execution engine, registry, and security enforcement. Every high-level feature (CLI, servers,
framework integrations) ultimately flows through these building blocks.

## Modules

- `gantry.py`: The `AgentGantry` facade. It wires configuration into adapters, registers tools, syncs
  embeddings to the vector store, and exposes helpers to retrieve and execute tools. It also hosts
  server helpers (`serve_mcp`, `serve_a2a`) so you can expose the same registry over different
  protocols.
- `router.py`: Home of the `SemanticRouter` and `RoutingWeights`. It performs vector search,
  re-ranking, namespace filtering, and health-aware scoring to return the top-k `ScoredTool`
  candidates for a query. The weights mirror Phase 3 controls (semantic similarity vs. health).
- `executor.py`: Defines the `ExecutionEngine`, which validates arguments, enforces timeouts,
  performs retries with backoff, and tracks circuit breaker state per tool. Execution is telemetry
  aware, emitting spans/events via the configured telemetry adapter.
- `registry.py`: A simple registry that maps tool names to callables and `ToolDefinition` metadata.
  It is kept separate from the router to allow dynamic providers (e.g., MCP/A2A) to populate tools
  without reconfiguring routing logic.
- `context.py`: Utilities for building a `ToolQuery` out of conversation state and summarizing past
  tool calls. These helpers keep retrieval deterministic for the LLM integration decorators.
- `security.py`: Implements the capability-based `SecurityPolicy` and enforcement hooks (`require` /
  `confirm`). The executor checks these policies before invoking tools to implement zero-trust
  semantics.

## Control flow at a glance

```
AgentGantry.register(...) ──▶ ToolRegistry
                         └──▶ SemanticRouter (embeddings + vector store)
ToolQuery(...) ──▶ SemanticRouter.search(...) ──▶ list[ScoredTool]
ToolCall(...)  ──▶ ExecutionEngine.execute(...) ──▶ result / CircuitBreaker
```

## Common patterns

### Register, sync, retrieve, execute

```python
from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall

gantry = AgentGantry()

@gantry.register(capability="files:read")
def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()

await gantry.sync()  # embeds tools and loads vector store

results = await gantry.retrieve_tools("open and read a markdown file", limit=2)
best_tool = results[0].tool

output = await gantry.execute(ToolCall(tool_name=best_tool.name, arguments={"path": "README.md"}))
print(output.output)
```

### Custom adapters and weights

```python
from agent_gantry import AgentGantry, AgentGantryConfig
from agent_gantry.core.router import RoutingWeights

config = AgentGantryConfig()
config.routing.weights = RoutingWeights(semantic=0.8, health=0.2)

# Plug in your own vector store or embedder
# gantry = AgentGantry(config=config, vector_store=my_store, embedder=my_embedder)
gantry = AgentGantry(config=config)
```

The core package is intentionally small but highly composable. If you are debugging routing scores,
start in `router.py`. If a tool is being blocked or timing out, look at `executor.py` and the
telemetry emitted there.*** End Patch"|()
