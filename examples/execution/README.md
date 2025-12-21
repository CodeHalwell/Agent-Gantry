# examples/execution

Reliability-focused demos that exercise the execution engine: retries, timeouts, circuit breakers,
and security policies.

## Files
- `circuit_breaker_demo.py`: Forces a failing tool to trip the circuit breaker and shows recovery.
- `batch_execution_demo.py`: Uses `execute_batch` to run many tool calls concurrently with per-call timeouts.
- `security_demo.py`: Illustrates capability- and confirmation-based security policies that gate tool execution.

## Run commands

```bash
python examples/execution/circuit_breaker_demo.py
python examples/execution/batch_execution_demo.py
python examples/execution/security_demo.py
```

These scripts log execution decisions to the console (via the console telemetry adapter) so you can
see retries, denials, and circuit state transitions. All run with the in-memory adapters by default.
