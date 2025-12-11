"""
Console and NoOp telemetry adapter implementations.

Provides simple telemetry adapters for development and production.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from agent_gantry.schema.execution import ToolCall, ToolResult
    from agent_gantry.schema.query import RetrievalResult, ToolQuery
    from agent_gantry.schema.tool import ToolHealth


logger = logging.getLogger("agent_gantry")


class ConsoleTelemetryAdapter:
    """
    Console-based telemetry adapter for development.

    Logs all telemetry events to console using structured logging.
    """

    def __init__(self, log_level: int = logging.INFO) -> None:
        """
        Initialize console telemetry adapter.

        Args:
            log_level: Logging level to use
        """
        self.log_level = log_level
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)
            logger.setLevel(log_level)

    @asynccontextmanager
    async def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> AsyncIterator[None]:
        """
        Create a tracing span.

        Args:
            name: Span name
            attributes: Optional span attributes
        """
        start = datetime.now(timezone.utc)
        logger.log(
            self.log_level,
            "Span started",
            extra={
                "span_name": name,
                "attributes": attributes or {},
                "timestamp": start.isoformat(),
            },
        )
        try:
            yield
        finally:
            end = datetime.now(timezone.utc)
            duration_ms = (end - start).total_seconds() * 1000
            logger.log(
                self.log_level,
                "Span completed",
                extra={
                    "span_name": name,
                    "duration_ms": duration_ms,
                    "timestamp": end.isoformat(),
                },
            )

    async def record_retrieval(self, query: ToolQuery, result: RetrievalResult) -> None:
        """
        Record a retrieval event.

        Args:
            query: The tool query
            result: The retrieval result
        """
        logger.log(
            self.log_level,
            "Tool retrieval",
            extra={
                "event_type": "retrieval",
                "query": query.context.query,
                "limit": query.limit,
                "tools_found": len(result.tools),
                "total_time_ms": result.total_time_ms,
                "trace_id": result.trace_id,
            },
        )

    async def record_execution(self, call: ToolCall, result: ToolResult) -> None:
        """
        Record an execution event.

        Args:
            call: The tool call
            result: The execution result
        """
        log_data = {
            "event_type": "execution",
            "tool_name": call.tool_name,
            "status": result.status.value,
            "latency_ms": result.latency_ms,
            "attempt_number": result.attempt_number,
            "trace_id": result.trace_id,
        }
        if result.error:
            log_data["error"] = result.error
            log_data["error_type"] = result.error_type

        log_level = logging.ERROR if result.error else self.log_level
        logger.log(log_level, "Tool execution", extra=log_data)

    async def record_health_change(
        self,
        tool_name: str,
        old_health: ToolHealth,
        new_health: ToolHealth,
    ) -> None:
        """
        Record a health change event.

        Args:
            tool_name: Name of the tool
            old_health: Previous health state
            new_health: New health state
        """
        logger.log(
            logging.WARNING if new_health.circuit_breaker_open else self.log_level,
            "Tool health changed",
            extra={
                "event_type": "health_change",
                "tool_name": tool_name,
                "old_success_rate": old_health.success_rate,
                "new_success_rate": new_health.success_rate,
                "consecutive_failures": new_health.consecutive_failures,
                "circuit_breaker_open": new_health.circuit_breaker_open,
            },
        )

    async def health_check(self) -> bool:
        """Check if telemetry is healthy."""
        return True


class NoOpTelemetryAdapter:
    """
    No-op telemetry adapter.

    Discards all telemetry events. Useful for testing or when telemetry is disabled.
    """

    @asynccontextmanager
    async def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> AsyncIterator[None]:
        """
        Create a no-op span.

        Args:
            name: Span name (ignored)
            attributes: Optional span attributes (ignored)
        """
        yield

    async def record_retrieval(self, query: ToolQuery, result: RetrievalResult) -> None:
        """Record a retrieval event (no-op)."""
        pass

    async def record_execution(self, call: ToolCall, result: ToolResult) -> None:
        """Record an execution event (no-op)."""
        pass

    async def record_health_change(
        self,
        tool_name: str,
        old_health: ToolHealth,
        new_health: ToolHealth,
    ) -> None:
        """Record a health change event (no-op)."""
        pass

    async def health_check(self) -> bool:
        """Check if telemetry is healthy."""
        return True
