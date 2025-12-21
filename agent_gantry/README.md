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

## Typical usage

```python
from agent_gantry import AgentGantry, AgentGantryConfig

config = AgentGantryConfig.from_yaml("gantry.yaml")  # or build programmatically
gantry = AgentGantry(config=config)

@gantry.register(tags=["math"], capability="calculator")
def add(a: int, b: int) -> int:
    return a + b

await gantry.sync()                   # push tool embeddings to the vector store
tools = await gantry.retrieve_tools("sum two integers", limit=3)
result = await gantry.execute_from_llm(tools[0], {"a": 3, "b": 4})  # helper around ToolCall
```

Most directories under `agent_gantry` have their own README with details, configuration knobs, and
code snippets. Start with `core/README.md` if you want to understand how routing and execution fit
together.*** End Patch"|()
