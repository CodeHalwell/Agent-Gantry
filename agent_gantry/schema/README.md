# agent_gantry/schema

Pydantic models that define the public contracts for Agent-Gantry. Everything that crosses a
boundary—config files, tool definitions, telemetry events, execution payloads—is defined here. These
schemas are stable and are the safest place to integrate with external systems.

## Modules

- `config.py`: Central configuration schema (`AgentGantryConfig`, `EmbedderConfig`, `VectorStoreConfig`,
  `TelemetryConfig`, etc.). Supports YAML loading via `from_yaml`.
- `tool.py`: Canonical `ToolDefinition`, `ToolCapability`, and schema transcoding helpers for OpenAI,
  Anthropic, and Google tool formats. Also performs argument validation.
- `query.py`: `ToolQuery`, `ConversationContext`, `ScoredTool` and `RetrievalResult`, the models the
  router consumes and produces. (`RoutingWeights` lives with the router, in `core/router.py`.)
- `execution.py`: `ToolCall`, `ToolResult`, `ToolCallEvent` and the batch execution models, along with
  failure metadata.
- `mcp.py`, `skill.py`, `selection.py`: MCP server definitions, Agent Skills, and the neutral
  candidate/result models a selector works on.
- `introspection.py`: Builds a tool's JSON parameter schema from a Python signature.
- `a2a.py`: Agent-to-Agent protocol models (agent cards, skill definitions, skill execution).

## Example: loading config from YAML

```python
from agent_gantry.schema.config import AgentGantryConfig

config = AgentGantryConfig.from_yaml("gantry.yaml")
print(config.embedder.type)  # e.g., "openai"
```

If you are integrating Agent-Gantry with another runtime or protocol, prefer importing these models
instead of re-declaring equivalents—they capture validation rules and backward compatibility logic.***
