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

    from agent_gantry.metrics.token_usage import ProviderUsage, TokenSavings
    from agent_gantry.schema.execution import ToolCall, ToolResult
    from agent_gantry.schema.query import RetrievalResult, ToolQuery
    from agent_gantry.schema.tool import ToolHealth


logger = logging.getLogger("agent_gantry")

_CONSOLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Marker attribute set on the handler this helper installs, so we can detect
# "did *we* already add a console handler?" without subclass-sniffing. A plain
# isinstance(StreamHandler) check would also match an app's FileHandler
# (FileHandler subclasses StreamHandler), causing us to skip attaching real
# console output when only a file handler is configured.
_CONSOLE_HANDLER_FLAG = "_agent_gantry_console_handler"


def enable_console_logging(level: int = logging.INFO) -> None:
    """Opt in to human-readable console output for the ``agent_gantry`` logger.

    Attaches a console :class:`~logging.StreamHandler` (at most once) and sets
    the logger level. This is the explicit, consumer-driven replacement for the
    handler/level side effect that :class:`ConsoleTelemetryAdapter` used to
    perform on construction.

    A library must never configure logging on the application's behalf, so
    importing ``agent_gantry`` only attaches a :class:`~logging.NullHandler`
    (see ``agent_gantry/__init__.py``). Call this helper from a script, demo,
    or CLI when you actually want Gantry's telemetry/INFO lines on the console;
    an embedding application should configure its own logging instead.

    Idempotent: a second call won't stack another console handler. It only
    considers handlers *this* helper installed, so an app that already has a
    :class:`~logging.FileHandler` still gets a console handler here.
    """
    already_attached = any(
        getattr(h, _CONSOLE_HANDLER_FLAG, False) for h in logger.handlers
    )
    if not already_attached:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        setattr(handler, _CONSOLE_HANDLER_FLAG, True)
        logger.addHandler(handler)
    logger.setLevel(level)


class ConsoleTelemetryAdapter:
    """
    Console-based telemetry adapter for development.

    Emits every telemetry event as a structured log record on the
    ``agent_gantry`` logger. By default it does **not** attach a handler or
    change the logger level — that is the application's responsibility (or
    opt in via ``attach_handler=True`` / :func:`enable_console_logging`).
    Records simply propagate to whatever handlers the consumer configured,
    and are swallowed by the package ``NullHandler`` if none are.
    """

    def __init__(
        self,
        log_level: int = logging.INFO,
        *,
        attach_handler: bool = False,
    ) -> None:
        """
        Initialize console telemetry adapter.

        Args:
            log_level: Logging level used when emitting records.
            attach_handler: When ``True``, attach a console
                :class:`~logging.StreamHandler` and raise the ``agent_gantry``
                logger to ``log_level`` (the legacy convenience behaviour).
                Defaults to ``False`` so the adapter never mutates the shared
                logger as a side effect of construction.
        """
        self.log_level = log_level
        if attach_handler:
            enable_console_logging(log_level)

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
            if result.error_type:
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

    async def record_token_usage(
        self,
        usage: ProviderUsage,
        model_name: str,
        savings: TokenSavings | None = None,
        trace_id: str | None = None,
    ) -> None:
        """
        Record token usage and optional savings.

        Args:
            usage: The actual usage reported by the provider
            model_name: Name of the model used
            savings: Optional savings calculation
            trace_id: Optional trace ID
        """
        log_data = {
            "event_type": "token_usage",
            "model_name": model_name,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "trace_id": trace_id,
        }
        if savings:
            log_data.update(
                {
                    "saved_prompt_tokens": savings.saved_prompt_tokens,
                    "prompt_savings_pct": f"{savings.prompt_savings_pct:.1f}%",
                }
            )

        logger.log(self.log_level, "Token usage", extra=log_data)

    async def health_check(self) -> bool:
        """Check if telemetry is healthy."""
        return True


class NoopTelemetryAdapter:
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

    async def record_token_usage(
        self,
        usage: ProviderUsage,
        model_name: str,
        savings: TokenSavings | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Record token usage (no-op)."""
        pass

    async def health_check(self) -> bool:
        """Check if telemetry is healthy."""
        return True
