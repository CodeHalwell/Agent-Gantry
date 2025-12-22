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

## How the Decorator Works Behind the Scenes

When you use the `@with_semantic_tools` decorator from `agent_gantry.integrations.decorator`, here's what happens:

```
User's LLM Function Call
    ↓
@with_semantic_tools intercepts
    ↓
Extract prompt from function arguments
    ↓
Build ToolQuery with ConversationContext
    ↓
SemanticRouter.search()
    ├──▶ Embedder: Convert query to vector
    ├──▶ VectorStore: Semantic similarity search
    ├──▶ Reranker (optional): Re-rank by relevance
    └──▶ RoutingWeights: Combine semantic + health scores
    ↓
Return list[ScoredTool]
    ↓
Convert to target dialect (OpenAI/Anthropic/Gemini)
    ↓
Inject tools into function's 'tools' parameter
    ↓
User's LLM Function executes with relevant tools
    ↓
LLM returns tool_calls
    ↓
ExecutionEngine.execute()
    ├──▶ Registry: Lookup tool callable
    ├──▶ SecurityPolicy: Check capabilities
    ├──▶ Validator: Validate arguments
    ├──▶ CircuitBreaker: Check tool health
    └──▶ Execute with retries + telemetry
    ↓
Return ExecutionResult
```

**Key Components:**

1. **Registry**: Stores tool metadata and callables
   ```python
   @gantry.register(tags=["weather"])
   def get_weather(city: str) -> str:
       ...
   # Registry now maps "get_weather" → callable + ToolDefinition
   ```

2. **Router**: Performs semantic search
   ```python
   # Behind the scenes in decorator:
   query = ToolQuery(context=ConversationContext(query="What's the weather?"))
   scored_tools = await router.search(query, limit=3)
   # Returns tools ranked by relevance
   ```

3. **Executor**: Runs tools safely
   ```python
   # When LLM calls a tool:
   result = await executor.execute(ToolCall(
       tool_name="get_weather",
       arguments={"city": "Paris"}
   ))
   # Handles retries, timeouts, circuit breakers, telemetry
   ```

## Control flow at a glance

```
AgentGantry.register(...) ──▶ ToolRegistry
                         └──▶ SemanticRouter (embeddings + vector store)
ToolQuery(...) ──▶ SemanticRouter.search(...) ──▶ list[ScoredTool]
ToolCall(...)  ──▶ ExecutionEngine.execute(...) ──▶ result / CircuitBreaker
```

## Common patterns

### Pattern 1: Automatic Injection (with decorator)

```python
from agent_gantry import AgentGantry, with_semantic_tools
from openai import AsyncOpenAI
from pathlib import Path

gantry = AgentGantry()
client = AsyncOpenAI()

# Define allowed directory for file access (security best practice)
ALLOWED_DIR = Path("/app/data").resolve()

@gantry.register(capability="files:read")
def read_file(path: str) -> str:
    """Safely read a file with path validation to prevent directory traversal attacks."""
    # Resolve the absolute path and validate it's within allowed directory
    file_path = (ALLOWED_DIR / path).resolve()
    
    # Security check: ensure the resolved path is within ALLOWED_DIR
    if not str(file_path).startswith(str(ALLOWED_DIR)):
        raise ValueError(f"Access denied: path must be within {ALLOWED_DIR}")
    
    # Additional security: check file exists and is a file (not a directory)
    if not file_path.is_file():
        raise ValueError(f"Invalid path: {path} is not a valid file")
    
    with open(file_path) as f:
        return f.read()

# The decorator handles: sync, retrieve, inject
@with_semantic_tools(gantry, limit=2)
async def chat(prompt: str, *, tools=None):
    return await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=tools  # Auto-populated by decorator
    )

# Router + Registry work behind the scenes
response = await chat("read the data.json file")
```

### Pattern 2: Manual Control (explicit retrieval and execution)

```python
from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall
from pathlib import Path

gantry = AgentGantry()

# Define allowed directory for file access (security best practice)
ALLOWED_DIR = Path("/app/data").resolve()

@gantry.register(capability="files:read")
def read_file(path: str) -> str:
    """Safely read a file with path validation to prevent directory traversal attacks."""
    # Resolve the absolute path and validate it's within allowed directory
    file_path = (ALLOWED_DIR / path).resolve()
    
    # Security check: ensure the resolved path is within ALLOWED_DIR
    if not str(file_path).startswith(str(ALLOWED_DIR)):
        raise ValueError(f"Access denied: path must be within {ALLOWED_DIR}")
    
    # Additional security: check file exists and is a file (not a directory)
    if not file_path.is_file():
        raise ValueError(f"Invalid path: {path} is not a valid file")
    
    with open(file_path) as f:
        return f.read()

await gantry.sync()  # embeds tools and loads vector store

# Manually query the router
results = await gantry.retrieve_tools("open and read a markdown file", limit=2)
best_tool = results[0].tool

# Manually execute via executor (will only work for files in /app/data)
output = await gantry.execute(ToolCall(tool_name=best_tool.name, arguments={"path": "data.md"}))
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

## Debugging Tips

- **Routing Scores:** If tools aren't being selected correctly, check `router.py` and the embedder quality. Use `score_threshold` to filter low-relevance tools.
- **Execution Failures:** Check `executor.py` for timeout/retry settings. Review telemetry for execution spans.
- **Circuit Breakers:** Tools with high failure rates trigger circuit breakers. Check health metrics with `gantry.get_tool_health(tool_name)`.
- **Security Blocks:** If tools are blocked, verify the `SecurityPolicy` and capability requirements.

The core package is intentionally small but highly composable. If you are debugging routing scores,
start in `router.py`. If a tool is being blocked or timing out, look at `executor.py` and the
telemetry emitted there.
