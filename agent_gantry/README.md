# agent_gantry package

The `agent_gantry` package is the library surface for Agent-Gantry. It wires together the semantic
router, execution engine, telemetry, protocol adapters, and configuration schema into a single
importable module. Everything exported in `__init__.py` is considered public API and is used
throughout the examples and tests.

## What lives here

- `__init__.py`: Re-exports `AgentGantry`, schema classes, and common adapter types so that users can
  import from `agent_gantry` without knowing the exact module layout.
- `adapters/`: Concrete integrations for embedders, rerankers, vector stores, and execution
  backends. These are the plug points that let you swap provider SDKs or storage engines.
- `cli/`: Entrypoint and helpers for the `agent-gantry` command-line tool.
- `core/`: The orchestration layer (facade, registry, router, executor, and security policy).
- `integrations/`: Helpers for plugging Agent-Gantry into higher-level agent frameworks (LangChain,
  AutoGen, LlamaIndex, CrewAI). The decorator helper in this folder powers the semantic tool
  injection demos.
- `observability/`: Telemetry adapters (console, OpenTelemetry, Prometheus) and tracing hooks.
- `providers/`: Clients that pull external tools/skills into the registry (e.g., A2A agents).
- `schema/`: Pydantic models that define tools, configs, queries, events, and execution payloads.
- `servers/`: Implementations for serving Agent-Gantry over MCP and A2A protocols.
- `metrics/`: Utility code for normalizing and reporting token usage.

## Two Ways to Use Agent-Gantry

Agent-Gantry offers two complementary APIs to fit different use cases:

### 1. Automatic Injection API (Recommended for Most Users)

The **"Plug and Play"** approach uses decorators to automatically inject relevant tools into your LLM calls:

```python
from agent_gantry import AgentGantry, with_semantic_tools, set_default_gantry
from openai import AsyncOpenAI

# One-time setup
gantry = AgentGantry()
set_default_gantry(gantry)
client = AsyncOpenAI()

# Register tools
@gantry.register(tags=["math"])
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# Decorate your LLM function - tools are automatically selected and injected
@with_semantic_tools(limit=3)
async def ask_llm(prompt: str, *, tools=None):
    return await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        tools=tools  # Automatically populated with relevant tools
    )

# Just call it - semantic routing happens behind the scenes
response = await ask_llm("What is 5 + 3?")
```

**Benefits:** Minimal code changes, works with any LLM provider, automatic schema conversion.

### 2. Manual Control API (For Power Users)

For fine-grained control over tool retrieval and execution:

```python
from agent_gantry import AgentGantry, AgentGantryConfig
from agent_gantry.schema.execution import ToolCall

# Configure with custom adapters
config = AgentGantryConfig.from_yaml("gantry.yaml")
gantry = AgentGantry(config=config)

# Register tools with detailed metadata
@gantry.register(tags=["math"], capability="calculator")
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# Sync to index tools in vector store
await gantry.sync()

# Manually retrieve relevant tools
tools = await gantry.retrieve_tools("sum two integers", limit=3)

# Manually execute tools with security policies
result = await gantry.execute(ToolCall(
    tool_name="add",
    arguments={"a": 3, "b": 4}
))
```

**Benefits:** Full control over retrieval parameters, custom scoring, batch operations, advanced error handling.

## Project Structure

Most directories under `agent_gantry` have their own README with details, configuration knobs, and
code snippets. Start with `core/README.md` if you want to understand how routing and execution fit
together, or `integrations/README.md` for the decorator patterns.
