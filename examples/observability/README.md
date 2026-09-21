# examples/observability

Telemetry- and metrics-focused examples that show how to instrument Agent-Gantry.

## Files
- `telemetry_demo.py`: Uses the console telemetry adapter to emit spans and events during retrieval
  and execution. Calls `enable_console_logging()` first — the package never configures logging on
  its own, so without that opt-in the adapter's records go nowhere.
- `multi_provider_metrics_demo.py`: Feeds Anthropic-, Google GenAI- and OpenAI-shaped `usage` dicts
  through `calculate_token_savings` to show they normalise to one `ProviderUsage`. Hard-coded
  numbers; no provider is called.
- `token_savings_demo.py`: Measures the saving on a real OpenAI call — one prompt with all 30 tool
  schemas attached versus only the two Gantry retrieved. Needs `OPENAI_API_KEY` and the `openai`
  extra; two completions are billed.

## Run

```bash
python examples/observability/telemetry_demo.py
python examples/observability/multi_provider_metrics_demo.py
python examples/observability/token_savings_demo.py          # needs OPENAI_API_KEY
```

`telemetry_demo.py` prints `Span started` / `Span completed` records for `tool_retrieval` and
`tool_execution`, plus a `Tool retrieval` and a `Tool execution` event with timings. Swap in a
different telemetry adapter (`agent_gantry.observability.opentelemetry_adapter`) to forward the same
events to OpenTelemetry or Prometheus.
