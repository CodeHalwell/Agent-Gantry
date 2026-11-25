# dotfetch/schema

Pydantic models that define the public contracts for dotfetch. Everything that crosses a
boundary—config files, tool definitions, telemetry events, execution payloads—is defined here. These
schemas are stable and are the safest place to integrate with external systems.

## Modules

- `config.py`: Central configuration schema (`DotFetchConfig`, `EmbedderConfig`, `VectorStoreConfig`,
  `TelemetryConfig`, etc.). Supports YAML loading via `from_yaml`.
- `tool.py`: Canonical `ToolDefinition`, `ToolCapability`, and schema transcoding helpers for OpenAI,
  Anthropic, and Google tool formats. Also performs argument validation.
- `query.py`: `ToolQuery`, `ScoredTool`, `RetrievalResult`, and `RoutingWeights` models used by the
  router.
- `execution.py`: `ToolCall`, `ToolResult`, and batch execution models along with failure metadata.
- `events.py`: Telemetry event payloads emitted during retrieval/execution.
- `a2a.py`: Agent-to-Agent protocol models (agent cards, skill definitions, skill execution).

## Example: loading config from YAML

```python
from dotfetch.schema.config import DotFetchConfig

config = DotFetchConfig.from_yaml("fetcher.yaml")
print(config.embedder.type)  # e.g., "openai"
```

If you are integrating dotfetch with another runtime or protocol, prefer importing these models
instead of re-declaring equivalents—they capture validation rules and backward compatibility logic.***
