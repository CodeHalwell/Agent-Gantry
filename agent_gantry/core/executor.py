"""
Execution engine for Agent-Gantry.

Handles tool execution with retries, timeouts, circuit breakers, and health tracking.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from agent_gantry.schema.execution import (
    BatchToolCall,
    BatchToolResult,
    ExecutionStatus,
    ToolCall,
    ToolResult,
)

if TYPE_CHECKING:
    from agent_gantry.core.rate_limiter import RateLimiter
    from agent_gantry.core.registry import ToolRegistry
    from agent_gantry.core.security import SecurityPolicy
    from agent_gantry.observability.telemetry import TelemetryAdapter
    from agent_gantry.schema.tool import ToolDefinition


class ExecutionEngine:
    """
    Execution engine for tool calls.

    Handles:
    - Permission checks (policy + capabilities)
    - Argument validation
    - Circuit breaker logic
    - Retries, back-off, and timeouts
    - Health metric updates
    - Telemetry emission
    """

    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout_ms: int = 30000,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout_s: int = 60,
        security_policy: SecurityPolicy | None = None,
        telemetry: TelemetryAdapter | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """
        Initialize the execution engine.

        Args:
            registry: Tool registry for looking up handlers
            default_timeout_ms: Default timeout for tool execution
            max_retries: Maximum number of retry attempts
            circuit_breaker_threshold: Failures before opening circuit
            circuit_breaker_timeout_s: Seconds before attempting recovery
            security_policy: Security policy for permission checks
            telemetry: Telemetry adapter for observability
            rate_limiter: Rate limiter for controlling execution throughput
        """
        self._registry = registry
        self._default_timeout = default_timeout_ms
        self._max_retries = max_retries
        self._cb_threshold = circuit_breaker_threshold
        self._cb_timeout = circuit_breaker_timeout_s
        self._security_policy = security_policy
        self._telemetry = telemetry
        self._rate_limiter = rate_limiter

    async def execute(self, call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Args:
            call: The tool call to execute

        Returns:
            Result of the execution
        """
        trace_id = call.trace_id or self._generate_trace_id()
        span_id = self._generate_span_id()
        queued_at = datetime.now(timezone.utc)

        # Look up tool from registry (search across all namespaces)
        tool = self._registry.get_tool_by_name(call.tool_name)
        if not tool:
            return ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.FAILURE,
                error=f"Tool '{call.tool_name}' not found",
                error_type="ToolNotFound",
                queued_at=queued_at,
                completed_at=datetime.now(timezone.utc),
                trace_id=trace_id,
                span_id=span_id,
            )

        # Check circuit breaker
        cb_result = await self._check_circuit_breaker(
            tool, call, queued_at, trace_id, span_id
        )
        if cb_result:
            return cb_result

        # Security policy check
        sp_result = await self._check_security_policy(
            call, queued_at, trace_id, span_id
        )
        if sp_result:
            return sp_result

        # Rate limiting check
        rl_result = await self._check_rate_limit(
            tool, call, queued_at, trace_id, span_id
        )
        if rl_result:
            return rl_result

        # Argument validation
        val_result = await self._validate_call_arguments(
            tool, call, queued_at, trace_id, span_id
        )
        if val_result:
            return val_result

        # Check if this requires special execution (A2A, MCP, etc.)
        from agent_gantry.schema.tool import ToolSource

        if tool.source == ToolSource.A2A_AGENT:
            # Use A2A executor
            from agent_gantry.adapters.executors.a2a_executor import A2AExecutor

            a2a_executor = A2AExecutor()
            return await a2a_executor.execute(tool, call, None)

        # Get handler for Python functions
        handler = self._registry.get_handler(f"{tool.namespace}.{call.tool_name}")
        if not handler:
            return ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.FAILURE,
                error=f"No handler found for tool '{call.tool_name}'",
                error_type="HandlerNotFound",
                queued_at=queued_at,
                completed_at=datetime.now(timezone.utc),
                trace_id=trace_id,
                span_id=span_id,
            )

        # Check confirmation requirement
        confirm_result = await self._check_confirmation_required(
            tool, call, queued_at, trace_id, span_id
        )
        if confirm_result:
            return confirm_result

        # Execute with retries (release rate limiter slot when done)
        try:
            return await self._execute_handler_with_retries(
                tool, call, handler, queued_at, trace_id, span_id
            )
        finally:
            if self._rate_limiter:
                await self._rate_limiter.release(call.tool_name, tool.namespace)

    async def execute_batch(self, batch: BatchToolCall) -> BatchToolResult:
        """
        Execute multiple tool calls.

        Args:
            batch: The batch of tool calls

        Returns:
            Results of all executions
        """
        start_time = datetime.now(timezone.utc)
        results: list[ToolResult] = []

        if batch.execution_strategy == "sequential":
            for call in batch.calls:
                result = await self.execute(call)
                results.append(result)
                if batch.fail_fast and result.status != ExecutionStatus.SUCCESS:
                    break
        else:
            # Parallel execution
            tasks = [self.execute(call) for call in batch.calls]
            results = list(await asyncio.gather(*tasks))

        end_time = datetime.now(timezone.utc)
        total_time_ms = (end_time - start_time).total_seconds() * 1000

        successful = sum(1 for r in results if r.status == ExecutionStatus.SUCCESS)
        failed = len(results) - successful

        return BatchToolResult(
            results=results,
            total_time_ms=total_time_ms,
            successful_count=successful,
            failed_count=failed,
        )

    async def _execute_with_timeout(
        self,
        handler: Callable[..., Any],
        arguments: dict[str, Any],
        timeout_ms: int,
    ) -> Any:
        """Execute a handler with a timeout.

        Sync handlers are dispatched via :func:`asyncio.to_thread`, which
        binds correctly to the *running* loop. ``asyncio.get_event_loop()``
        is deprecated in 3.10+ when there is no running loop, and in worker-
        thread contexts (e.g. ``DurableAIAgentWorker``) it can return a
        different loop from the one currently running, surfacing as cross-
        loop runtime errors when the executor is driven from a loop other
        than the one the gantry was constructed on.
        """
        timeout_s = timeout_ms / 1000

        if asyncio.iscoroutinefunction(handler):
            return await asyncio.wait_for(handler(**arguments), timeout=timeout_s)
        return await asyncio.wait_for(
            asyncio.to_thread(handler, **arguments),
            timeout=timeout_s,
        )

    async def _execute_handler_with_retries(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        handler: Callable[..., Any],
        queued_at: datetime,
        trace_id: str,
        span_id: str,
    ) -> ToolResult:
        """Execute tool handler with retries."""
        max_attempts = (call.retry_count or self._max_retries) + 1
        last_error: str | None = None
        last_error_type: str | None = None

        for attempt in range(1, max_attempts + 1):
            started_at = datetime.now(timezone.utc)
            try:
                result_value = await self._execute_with_timeout(
                    handler,
                    call.arguments,
                    call.timeout_ms or self._default_timeout,
                )
                completed_at = datetime.now(timezone.utc)
                await self._record_success(tool, (completed_at - started_at).total_seconds() * 1000)
                result = ToolResult(
                    tool_name=call.tool_name,
                    status=ExecutionStatus.SUCCESS,
                    result=result_value,
                    queued_at=queued_at,
                    started_at=started_at,
                    completed_at=completed_at,
                    attempt_number=attempt,
                    trace_id=trace_id,
                    span_id=span_id,
                )
                if self._telemetry:
                    await self._telemetry.record_execution(call, result)
                return result
            except asyncio.TimeoutError:
                last_error = "Execution timed out"
                last_error_type = "TimeoutError"
            except Exception as e:
                last_error = str(e)
                last_error_type = type(e).__name__

            if attempt < max_attempts:
                await asyncio.sleep(2**attempt * 0.1)

        completed_at = datetime.now(timezone.utc)
        await self._record_failure(tool)

        status = (
            ExecutionStatus.TIMEOUT
            if last_error_type == "TimeoutError"
            else ExecutionStatus.FAILURE
        )

        result = ToolResult(
            tool_name=call.tool_name,
            status=status,
            error=last_error,
            error_type=last_error_type,
            queued_at=queued_at,
            completed_at=completed_at,
            attempt_number=max_attempts,
            trace_id=trace_id,
            span_id=span_id,
        )
        if self._telemetry:
            await self._telemetry.record_execution(call, result)
        return result

    def _should_attempt_recovery(self, tool: ToolDefinition) -> bool:
        """Check if we should attempt circuit breaker recovery."""
        if not tool.health.last_failure:
            return True
        elapsed = (datetime.now(timezone.utc) - tool.health.last_failure).total_seconds()
        return elapsed >= self._cb_timeout

    async def _check_circuit_breaker(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        queued_at: datetime,
        trace_id: str,
        span_id: str,
    ) -> ToolResult | None:
        """Check if circuit breaker is open."""
        if tool.health.circuit_breaker_open and not self._should_attempt_recovery(tool):
            result = ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.CIRCUIT_OPEN,
                error="Circuit breaker is open due to repeated failures",
                queued_at=queued_at,
                completed_at=datetime.now(timezone.utc),
                trace_id=trace_id,
                span_id=span_id,
            )
            if self._telemetry:
                await self._telemetry.record_execution(call, result)
            return result
        return None

    async def _check_security_policy(
        self,
        call: ToolCall,
        queued_at: datetime,
        trace_id: str,
        span_id: str,
    ) -> ToolResult | None:
        """Check security policy permissions."""
        if self._security_policy:
            from agent_gantry.core.security import ConfirmationRequiredError, PermissionDeniedError

            try:
                self._security_policy.check_permission(call.tool_name, call.arguments)
            except ConfirmationRequiredError:
                result = ToolResult(
                    tool_name=call.tool_name,
                    status=ExecutionStatus.PENDING_CONFIRMATION,
                    error="Tool requires human confirmation",
                    queued_at=queued_at,
                    completed_at=datetime.now(timezone.utc),
                    trace_id=trace_id,
                    span_id=span_id,
                )
                if self._telemetry:
                    await self._telemetry.record_execution(call, result)
                return result
            except PermissionDeniedError as e:
                result = ToolResult(
                    tool_name=call.tool_name,
                    status=ExecutionStatus.FAILURE,
                    error=str(e),
                    error_type="PermissionDeniedError",
                    queued_at=queued_at,
                    completed_at=datetime.now(timezone.utc),
                    trace_id=trace_id,
                    span_id=span_id,
                )
                if self._telemetry:
                    await self._telemetry.record_execution(call, result)
                return result
        return None

    async def _check_rate_limit(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        queued_at: datetime,
        trace_id: str,
        span_id: str,
    ) -> ToolResult | None:
        """Check rate limit and acquire execution slot."""
        if self._rate_limiter:
            from agent_gantry.core.rate_limiter import RateLimitExceeded

            try:
                await self._rate_limiter.acquire(call.tool_name, tool.namespace)
            except RateLimitExceeded as e:
                result = ToolResult(
                    tool_name=call.tool_name,
                    status=ExecutionStatus.FAILURE,
                    error=str(e),
                    error_type="RateLimitExceeded",
                    queued_at=queued_at,
                    completed_at=datetime.now(timezone.utc),
                    trace_id=trace_id,
                    span_id=span_id,
                )
                if self._telemetry:
                    await self._telemetry.record_execution(call, result)
                return result
        return None

    async def _validate_call_arguments(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        queued_at: datetime,
        trace_id: str,
        span_id: str,
    ) -> ToolResult | None:
        """Validate arguments and return ToolResult if invalid."""
        is_valid, validation_error = await self._validate_arguments(tool, call.arguments)
        if not is_valid:
            result = ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.FAILURE,
                error=validation_error,
                error_type="ValidationError",
                queued_at=queued_at,
                completed_at=datetime.now(timezone.utc),
                trace_id=trace_id,
                span_id=span_id,
            )
            if self._telemetry:
                await self._telemetry.record_execution(call, result)
            return result
        return None

    async def _check_confirmation_required(
        self,
        tool: ToolDefinition,
        call: ToolCall,
        queued_at: datetime,
        trace_id: str,
        span_id: str,
    ) -> ToolResult | None:
        """Check if tool requires confirmation."""
        needs_confirm = call.require_confirmation
        if needs_confirm is None:
            needs_confirm = tool.requires_confirmation
        if needs_confirm:
            result = ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.PENDING_CONFIRMATION,
                queued_at=queued_at,
                completed_at=datetime.now(timezone.utc),
                trace_id=trace_id,
                span_id=span_id,
            )
            if self._telemetry:
                await self._telemetry.record_execution(call, result)
            return result
        return None

    async def _validate_arguments(
        self, tool: ToolDefinition, arguments: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """
        Validate arguments against tool schema.

        Args:
            tool: Tool definition with parameter schema
            arguments: Arguments to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        schema = tool.parameters_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        def _validate_value(value: Any, val_schema: dict[str, Any], path: str) -> tuple[bool, str | None]:
            expected_type = val_schema.get("type")

            if expected_type == "boolean":
                if not isinstance(value, bool):
                    return False, f"Parameter '{path}' must be a boolean"
            elif expected_type == "integer":
                if not isinstance(value, int) or isinstance(value, bool):
                    return False, f"Parameter '{path}' must be an integer"
            elif expected_type == "number":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return False, f"Parameter '{path}' must be a number"
            elif expected_type == "string":
                if not isinstance(value, str):
                    return False, f"Parameter '{path}' must be a string"
            elif expected_type == "array":
                if not isinstance(value, list):
                    return False, f"Parameter '{path}' must be an array"

                item_schema = val_schema.get("items")
                if item_schema:
                    for i, item in enumerate(value):
                        is_valid, err = _validate_value(item, item_schema, f"{path}[{i}]")
                        if not is_valid:
                            return False, err
            elif expected_type == "object":
                if not isinstance(value, dict):
                    return False, f"Parameter '{path}' must be an object"

                obj_properties = val_schema.get("properties", {})
                obj_required = val_schema.get("required", [])

                for req_prop in obj_required:
                    if req_prop not in value:
                        return False, f"Missing required parameter: {path}.{req_prop}"

                for prop_name, prop_value in value.items():
                    if prop_name not in obj_properties:
                        return False, f"Unknown parameter: {path}.{prop_name}"

                    is_valid, err = _validate_value(prop_value, obj_properties[prop_name], f"{path}.{prop_name}")
                    if not is_valid:
                        return False, err

            return True, None

        # Check top-level required parameters
        for param in required:
            if param not in arguments:
                return False, f"Missing required parameter: {param}"

        # Check parameter types
        for param_name, param_value in arguments.items():
            if param_name not in properties:
                return False, f"Unknown parameter: {param_name}"

            is_valid, err = _validate_value(param_value, properties[param_name], param_name)
            if not is_valid:
                return False, err

        return True, None

    async def _record_success(self, tool: ToolDefinition, latency_ms: float) -> None:
        """Record a successful execution."""
        old_health = tool.health.model_copy() if self._telemetry else None

        tool.health.total_calls += 1
        tool.health.last_success = datetime.now(timezone.utc)
        tool.health.consecutive_failures = 0
        tool.health.circuit_breaker_open = False

        # Update average latency
        n = tool.health.total_calls
        tool.health.avg_latency_ms = (tool.health.avg_latency_ms * (n - 1) + latency_ms) / n

        # Update success rate
        tool.health.success_rate = (tool.health.success_rate * (n - 1) + 1) / n

        if self._telemetry and old_health:
            await self._telemetry.record_health_change(
                f"{tool.namespace}.{tool.name}", old_health, tool.health
            )

    async def _record_failure(self, tool: ToolDefinition) -> None:
        """Record a failed execution."""
        old_health = tool.health.model_copy() if self._telemetry else None

        tool.health.total_calls += 1
        tool.health.last_failure = datetime.now(timezone.utc)
        tool.health.consecutive_failures += 1

        # Update success rate
        n = tool.health.total_calls
        tool.health.success_rate = (tool.health.success_rate * (n - 1)) / n

        # Check circuit breaker
        if tool.health.consecutive_failures >= self._cb_threshold:
            tool.health.circuit_breaker_open = True

        if self._telemetry and old_health:
            await self._telemetry.record_health_change(
                f"{tool.namespace}.{tool.name}", old_health, tool.health
            )

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return str(uuid.uuid4())

    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return str(uuid.uuid4())[:16]
