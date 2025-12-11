# Phase 2 Implementation: Robust Execution, Security, and Health

This document describes the Phase 2 implementation of Agent-Gantry, focusing on robust execution, security, and health tracking.

## Overview

Phase 2 delivers production-grade execution capabilities with:
- **ExecutionEngine**: Retries, timeouts, circuit breaker, and health tracking
- **SecurityPolicy**: Pattern-based tool access control and confirmation requirements
- **PermissionChecker**: Capability-based access control
- **Argument Validation**: Defensive validation against tool schemas
- **Telemetry**: Structured logging and observability
- **Async-Native**: Full async support for tools and execution

## Features Implemented

### 1. Execution Engine

The `ExecutionEngine` provides robust tool execution with:

#### Circuit Breaker
Automatically opens after consecutive failures to prevent cascade failures:
```python
from agent_gantry import AgentGantry
from agent_gantry.schema.execution import ToolCall, ExecutionStatus

gantry = AgentGantry()

@gantry.register
def flaky_service(data: str) -> str:
    """A service that might fail."""
    # ... implementation
    
# After 5 consecutive failures, circuit opens
# Prevents further attempts until timeout expires
result = await gantry.execute(ToolCall(
    tool_name="flaky_service",
    arguments={"data": "test"}
))
if result.status == ExecutionStatus.CIRCUIT_OPEN:
    print("Circuit breaker is open - service unavailable")
```

Default settings:
- Threshold: 5 consecutive failures
- Timeout: 60 seconds before attempting recovery

#### Retries with Exponential Backoff
Automatically retries failed executions:
```python
result = await gantry.execute(ToolCall(
    tool_name="unstable_tool",
    arguments={"x": 10},
    retry_count=3  # Will retry up to 3 times
))
```

Backoff formula: `2^attempt * 0.1` seconds

#### Timeouts
Prevents long-running tools from blocking:
```python
result = await gantry.execute(ToolCall(
    tool_name="slow_operation",
    arguments={"data": "large_dataset"},
    timeout_ms=5000  # 5 second timeout
))
if result.status == ExecutionStatus.TIMEOUT:
    print("Operation timed out")
```

Default timeout: 30,000ms (30 seconds)

#### Health Metrics
Tracks per-tool health metrics:
- Success rate
- Average latency
- Total calls
- Consecutive failures
- Last success/failure timestamps
- Circuit breaker state

```python
tool = await gantry.get_tool("my_tool")
print(f"Success rate: {tool.health.success_rate:.2%}")
print(f"Avg latency: {tool.health.avg_latency_ms:.2f}ms")
print(f"Circuit open: {tool.health.circuit_breaker_open}")
```

### 2. Security Policy

Pattern-based security controls for tool access:

```python
from agent_gantry.core.security import SecurityPolicy

policy = SecurityPolicy(
    require_confirmation=["delete_*", "payment_*", "drop_*"],
    allowed_domains=["internal-api.company.com"],
    max_requests_per_minute=60
)

gantry = AgentGantry(security_policy=policy)
```

#### Confirmation Requirements
Tools matching patterns require human approval:
```python
@gantry.register(requires_confirmation=True)
def delete_user(user_id: str) -> str:
    """Delete a user from the system."""
    return f"Deleted user {user_id}"

# Execution requires explicit confirmation
result = await gantry.execute(ToolCall(
    tool_name="delete_user",
    arguments={"user_id": "123"}
))
assert result.status == ExecutionStatus.PENDING_CONFIRMATION
```

Override confirmation on a per-call basis:
```python
result = await gantry.execute(ToolCall(
    tool_name="safe_tool",
    arguments={"x": 5},
    require_confirmation=True  # Force confirmation
))
```

### 3. Permission Checker

Capability-based access control:

```python
from agent_gantry.core.security import PermissionChecker
from agent_gantry.schema.tool import ToolCapability

# Define user capabilities
checker = PermissionChecker([
    ToolCapability.READ_DATA,
    ToolCapability.WRITE_DATA
])

# Check if user can use a tool
can_use, error = checker.can_use(tool)
if not can_use:
    print(f"Permission denied: {error}")

# Filter tools by capabilities
allowed_tools = checker.filter_tools(all_tools)
```

Available capabilities:
- `READ_DATA`: Read operations
- `WRITE_DATA`: Write/update operations
- `DELETE_DATA`: Delete operations
- `EXECUTE_CODE`: Code execution
- `NETWORK_ACCESS`: External network access
- `FILE_SYSTEM`: File system access
- `FINANCIAL`: Financial operations
- `PII_ACCESS`: Personal information access
- `EXTERNAL_API`: External API calls

Define tool capabilities:
```python
@gantry.register(capabilities=[ToolCapability.DELETE_DATA, ToolCapability.WRITE_DATA])
def remove_record(record_id: str) -> bool:
    """Remove a record from the database."""
    # ... implementation
```

### 4. Argument Validation

Defensive validation of tool arguments against schemas:

```python
@gantry.register
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Missing required parameter
result = await gantry.execute(ToolCall(
    tool_name="add_numbers",
    arguments={"a": 5}  # Missing 'b'
))
assert result.status == ExecutionStatus.FAILURE
assert "Missing required parameter" in result.error

# Wrong type
result = await gantry.execute(ToolCall(
    tool_name="add_numbers",
    arguments={"a": "not_a_number", "b": 5}
))
assert result.status == ExecutionStatus.FAILURE
assert "must be an integer" in result.error
```

Supported types:
- `integer`: Python `int`
- `number`: Python `int` or `float`
- `string`: Python `str`
- `boolean`: Python `bool`

### 5. Telemetry and Observability

Structured logging for all operations:

```python
from agent_gantry.observability.console import ConsoleTelemetryAdapter
import logging

telemetry = ConsoleTelemetryAdapter(log_level=logging.INFO)
gantry = AgentGantry(telemetry=telemetry)
```

Logged events:
- Tool retrieval (query, results, timing)
- Tool execution (status, latency, attempts)
- Health changes (success rate, circuit breaker state)
- Spans for tracing execution flow

Use `NoOpTelemetryAdapter` to disable telemetry:
```python
from agent_gantry.observability.console import NoOpTelemetryAdapter

gantry = AgentGantry(telemetry=NoOpTelemetryAdapter())
```

### 6. Async Operations

Full async support for both sync and async tools:

```python
# Async tool
@gantry.register
async def async_operation(data: str) -> str:
    """An async operation."""
    await some_async_call()
    return f"Processed: {data}"

# Sync tool (automatically wrapped)
@gantry.register
def sync_operation(value: int) -> int:
    """A sync operation."""
    return value * 2

# Both execute seamlessly
async_result = await gantry.execute(ToolCall(
    tool_name="async_operation",
    arguments={"data": "test"}
))

sync_result = await gantry.execute(ToolCall(
    tool_name="sync_operation",
    arguments={"value": 5}
))
```

#### Batch Execution
Execute multiple tools in parallel or sequence:

```python
from agent_gantry.schema.execution import BatchToolCall

batch = BatchToolCall(
    calls=[
        ToolCall(tool_name="tool_one", arguments={"x": 5}),
        ToolCall(tool_name="tool_two", arguments={"y": 10}),
    ],
    execution_strategy="parallel",  # or "sequential"
    fail_fast=True  # Stop on first failure
)

result = await gantry.execute_batch(batch)
print(f"Successful: {result.successful_count}")
print(f"Failed: {result.failed_count}")
```

## Testing

Phase 2 includes comprehensive tests covering:

- Circuit breaker functionality
- Retry and timeout behavior
- Security policy enforcement
- Permission checking
- Argument validation
- Async operations
- Health tracking

Run tests:
```bash
pytest tests/test_phase2.py -v
```

## Best Practices

### 1. Define Capabilities
Always declare tool capabilities:
```python
@gantry.register(
    capabilities=[ToolCapability.DELETE_DATA, ToolCapability.WRITE_DATA],
    requires_confirmation=True
)
def delete_records(filter: str) -> int:
    """Delete records matching filter."""
    # ... implementation
```

### 2. Use Appropriate Timeouts
Set timeouts based on expected execution time:
```python
# Quick operation
await gantry.execute(ToolCall(
    tool_name="cache_lookup",
    arguments={"key": "user:123"},
    timeout_ms=1000  # 1 second
))

# Long operation
await gantry.execute(ToolCall(
    tool_name="generate_report",
    arguments={"month": "2025-01"},
    timeout_ms=300000  # 5 minutes
))
```

### 3. Handle Failures Gracefully
Check execution status and handle errors:
```python
result = await gantry.execute(call)

if result.status == ExecutionStatus.SUCCESS:
    return result.result
elif result.status == ExecutionStatus.CIRCUIT_OPEN:
    # Service unavailable, use fallback
    return fallback_value
elif result.status == ExecutionStatus.TIMEOUT:
    # Operation too slow, try later
    return schedule_retry()
elif result.status == ExecutionStatus.PENDING_CONFIRMATION:
    # Request human approval
    return request_confirmation()
else:
    # Handle other failures
    log.error(f"Tool failed: {result.error}")
    raise ToolExecutionError(result.error)
```

### 4. Monitor Health Metrics
Regularly check tool health:
```python
tools = await gantry.list_tools()
unhealthy = [
    t for t in tools
    if t.health.circuit_breaker_open or t.health.success_rate < 0.9
]

for tool in unhealthy:
    alert_ops(f"Tool {tool.name} is unhealthy: {tool.health}")
```

## What's Next: Phase 3

Phase 3 will add:
- Intent classification
- Conversation context tracking
- Multi-signal scoring
- MMR diversity
- Reranker support

Stay tuned for more intelligent routing capabilities!
