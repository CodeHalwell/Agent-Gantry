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
    from agent_gantry.core.registry import ToolRegistry
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
    ) -> None:
        """
        Initialize the execution engine.

        Args:
            registry: Tool registry for looking up handlers
            default_timeout_ms: Default timeout for tool execution
            max_retries: Maximum number of retry attempts
            circuit_breaker_threshold: Failures before opening circuit
            circuit_breaker_timeout_s: Seconds before attempting recovery
        """
        self._registry = registry
        self._default_timeout = default_timeout_ms
        self._max_retries = max_retries
        self._cb_threshold = circuit_breaker_threshold
        self._cb_timeout = circuit_breaker_timeout_s

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

        # Look up tool
        tool = self._registry.get_tool(call.tool_name)
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
        if tool.health.circuit_breaker_open and not self._should_attempt_recovery(tool):
            return ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.CIRCUIT_OPEN,
                error="Circuit breaker is open due to repeated failures",
                queued_at=queued_at,
                completed_at=datetime.now(timezone.utc),
                trace_id=trace_id,
                span_id=span_id,
            )

        # Get handler
        handler = self._registry.get_handler(call.tool_name)
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
        needs_confirm = call.require_confirmation
        if needs_confirm is None:
            needs_confirm = tool.requires_confirmation
        if needs_confirm:
            return ToolResult(
                tool_name=call.tool_name,
                status=ExecutionStatus.PENDING_CONFIRMATION,
                queued_at=queued_at,
                completed_at=datetime.now(timezone.utc),
                trace_id=trace_id,
                span_id=span_id,
            )

        # Execute with retries
        max_attempts = (call.retry_count or self._max_retries) + 1
        last_error: str | None = None
        last_error_type: str | None = None

        for attempt in range(1, max_attempts + 1):
            started_at = datetime.now(timezone.utc)
            try:
                result = await self._execute_with_timeout(
                    handler,
                    call.arguments,
                    call.timeout_ms or self._default_timeout,
                )
                completed_at = datetime.now(timezone.utc)
                await self._record_success(tool, (completed_at - started_at).total_seconds() * 1000)
                return ToolResult(
                    tool_name=call.tool_name,
                    status=ExecutionStatus.SUCCESS,
                    result=result,
                    queued_at=queued_at,
                    started_at=started_at,
                    completed_at=completed_at,
                    attempt_number=attempt,
                    trace_id=trace_id,
                    span_id=span_id,
                )
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

        status = ExecutionStatus.TIMEOUT if "timeout" in (last_error or "").lower() else ExecutionStatus.FAILURE

        return ToolResult(
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
        """Execute a handler with a timeout."""
        timeout_s = timeout_ms / 1000

        if asyncio.iscoroutinefunction(handler):
            return await asyncio.wait_for(handler(**arguments), timeout=timeout_s)
        else:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: handler(**arguments)),
                timeout=timeout_s,
            )

    def _should_attempt_recovery(self, tool: ToolDefinition) -> bool:
        """Check if we should attempt circuit breaker recovery."""
        if not tool.health.last_failure:
            return True
        elapsed = (datetime.now(timezone.utc) - tool.health.last_failure).total_seconds()
        return elapsed >= self._cb_timeout

    async def _record_success(self, tool: ToolDefinition, latency_ms: float) -> None:
        """Record a successful execution."""
        tool.health.total_calls += 1
        tool.health.last_success = datetime.now(timezone.utc)
        tool.health.consecutive_failures = 0
        tool.health.circuit_breaker_open = False

        # Update average latency
        n = tool.health.total_calls
        tool.health.avg_latency_ms = (
            (tool.health.avg_latency_ms * (n - 1) + latency_ms) / n
        )

        # Update success rate
        tool.health.success_rate = (tool.health.success_rate * (n - 1) + 1) / n

    async def _record_failure(self, tool: ToolDefinition) -> None:
        """Record a failed execution."""
        tool.health.total_calls += 1
        tool.health.last_failure = datetime.now(timezone.utc)
        tool.health.consecutive_failures += 1

        # Update success rate
        n = tool.health.total_calls
        tool.health.success_rate = (tool.health.success_rate * (n - 1)) / n

        # Check circuit breaker
        if tool.health.consecutive_failures >= self._cb_threshold:
            tool.health.circuit_breaker_open = True

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return str(uuid.uuid4())

    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return str(uuid.uuid4())[:16]
