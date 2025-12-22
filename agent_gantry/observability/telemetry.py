"""
Telemetry adapter protocol.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agent_gantry.metrics.token_usage import ProviderUsage, TokenSavings
    from agent_gantry.schema.execution import ToolCall, ToolResult
    from agent_gantry.schema.query import RetrievalResult, ToolQuery
    from agent_gantry.schema.tool import ToolHealth


class TelemetryAdapter(Protocol):
    """
    Observability backend for Agent-Gantry.

    Implementations: OpenTelemetryAdapter, DatadogAdapter, PrometheusAdapter,
                     ConsoleAdapter, NoOpAdapter.
    """

    @abstractmethod
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
        yield

    @abstractmethod
    async def record_retrieval(
        self, query: ToolQuery, result: RetrievalResult
    ) -> None:
        """
        Record a retrieval event.

        Args:
            query: The tool query
            result: The retrieval result
        """
        ...

    @abstractmethod
    async def record_execution(
        self, call: ToolCall, result: ToolResult
    ) -> None:
        """
        Record an execution event.

        Args:
            call: The tool call
            result: The execution result
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...
