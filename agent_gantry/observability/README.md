# agent_gantry/observability

Telemetry and logging adapters for instrumenting Agent-Gantry.

- `__init__.py`: Exposes telemetry adapter interfaces.
- `console.py`: Console and noop telemetry adapters for local use.
- `opentelemetry_adapter.py`: OpenTelemetry and Prometheus exporters for tracing and metrics.
- `telemetry.py`: Base telemetry adapter interface and span helpers.
