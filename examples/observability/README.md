# examples/observability

Telemetry-focused examples that show how to instrument Agent-Gantry.

## Files
- `telemetry_demo.py`: Uses the console telemetry adapter to emit spans during retrieval and execution.

## Run

```bash
python examples/observability/telemetry_demo.py
```

Expect span-style output describing router scores, execution timings, and any retries triggered by
the execution engine. Swap in a different telemetry adapter in the script to forward data to
OpenTelemetry or Prometheus.
