# examples/execution

Reliability-focused demos that exercise the execution engine: retries, timeouts, circuit breakers,
and security policies.

## Files
- `circuit_breaker_demo.py`: Forces a failing tool to trip the circuit breaker and shows the next
  call being refused with `CIRCUIT_OPEN`.
- `batch_execution_demo.py`: Uses `execute_batch` to run many tool calls concurrently with per-call timeouts.
- `security_demo.py`: Illustrates capability- and confirmation-based security policies that gate tool execution.

## Run commands

```bash
python examples/execution/circuit_breaker_demo.py
python examples/execution/batch_execution_demo.py
python examples/execution/security_demo.py
```

Each script prints the `ToolResult` status it gets back (`success`, `failure`, `circuit_open`,
`pending_confirmation`). All run with the in-memory adapters by default and need no API key; see
`../observability/telemetry_demo.py` for the console telemetry adapter.
