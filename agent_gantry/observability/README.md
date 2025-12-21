# agent_gantry/observability

Telemetry adapters that instrument routing and execution. Each adapter implements
`TelemetryAdapter` (`telemetry.py`) and emits spans, counters, and events during tool retrieval and
execution.

## Modules

- `telemetry.py`: Base interface plus context managers for spans and event emission.
- `console.py`: Console and noop adapters. The console adapter prints human-friendly spans and is
  used in most examples; the noop adapter keeps overhead near zero.
- `opentelemetry_adapter.py`: Bridges to OpenTelemetry and Prometheus, exporting traces and metrics
  for production observability pipelines.

## Enabling telemetry

```python
from agent_gantry import AgentGantry
from agent_gantry.observability.console import ConsoleTelemetryAdapter

gantry = AgentGantry(telemetry=ConsoleTelemetryAdapter())
await gantry.retrieve_tools("search for refunds")  # emits spans to stdout
```

For OpenTelemetry/Prometheus, configure exporters in `TelemetryConfig` (see `schema.config`) or
instantiate `OpenTelemetryAdapter` directly with the desired resource and exporters. The execution
engine and router automatically emit spans, so no additional code changes are required.***
