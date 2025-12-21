"""
OpenTelemetry-compatible telemetry adapters.

These adapters provide tracing spans and in-memory metrics suitable for tests
without requiring external collectors.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from agent_gantry.observability.telemetry import TelemetryAdapter

if TYPE_CHECKING:
    from agent_gantry.metrics.token_usage import ProviderUsage, TokenSavings
    from agent_gantry.schema.execution import ToolCall, ToolResult
    from agent_gantry.schema.query import RetrievalResult, ToolQuery
    from agent_gantry.schema.tool import ToolHealth


class OpenTelemetryAdapter(TelemetryAdapter):
    """Minimal OpenTelemetry-style adapter."""

    def __init__(self, service_name: str, otlp_endpoint: str | None = None) -> None:
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.spans: list[dict[str, Any]] = []
        self.metrics: dict[str, float] = {}

    @asynccontextmanager
    async def span(
        self, name: str, attributes: dict[str, Any] | None = None
    ) -> AsyncIterator[None]:
        start = datetime.now(timezone.utc)
        span = {
            "name": name,
            "attributes": attributes or {},
            "start": start,
        }
        self.spans.append(span)
        try:
            yield
        finally:
            end = datetime.now(timezone.utc)
            span["end"] = end
            span["duration_ms"] = (end - start).total_seconds() * 1000

    async def record_retrieval(self, query: ToolQuery, result: RetrievalResult) -> None:
        self.metrics["retrievals_total"] = self.metrics.get("retrievals_total", 0) + 1
        self.metrics["retrieved_tools"] = self.metrics.get("retrieved_tools", 0) + len(result.tools)
        self.metrics["retrieval_latency_ms"] = result.total_time_ms

    async def record_execution(self, call: ToolCall, result: ToolResult) -> None:
        key = "executions_total"
        self.metrics[key] = self.metrics.get(key, 0) + 1
        status_key = f"executions_status_{result.status.value}"
        self.metrics[status_key] = self.metrics.get(status_key, 0) + 1

    async def record_health_change(
        self,
        tool_name: str,
        old_health: ToolHealth,
        new_health: ToolHealth,
    ) -> None:
        key = f"health_changes_{tool_name}"
        self.metrics[key] = self.metrics.get(key, 0) + 1

    async def record_token_usage(
        self,
        usage: ProviderUsage,
        model_name: str,
        savings: TokenSavings | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.metrics["tokens_prompt_total"] = (
            self.metrics.get("tokens_prompt_total", 0) + usage.prompt_tokens
        )
        self.metrics["tokens_completion_total"] = (
            self.metrics.get("tokens_completion_total", 0) + usage.completion_tokens
        )
        self.metrics["tokens_total"] = self.metrics.get("tokens_total", 0) + usage.total_tokens

        if savings:
            self.metrics["tokens_saved_prompt_total"] = (
                self.metrics.get("tokens_saved_prompt_total", 0) + savings.saved_prompt_tokens
            )
            # We track the running average prompt savings percentage.
            # Handle the first (or any invalid non-positive) count explicitly to avoid
            # division-by-zero and keep the logic easy to follow.
            count = int(self.metrics.get("savings_count", 0))
            if count <= 0:
                # First sample: the average is just this savings value.
                self.metrics["avg_prompt_savings_pct"] = savings.prompt_savings_pct
                self.metrics["savings_count"] = 1
            else:
                current_avg = self.metrics.get("avg_prompt_savings_pct", 0.0)
                new_count = count + 1
                self.metrics["avg_prompt_savings_pct"] = (
                    (current_avg * count) + savings.prompt_savings_pct
                ) / new_count
                self.metrics["savings_count"] = new_count

    async def health_check(self) -> bool:
        if self.otlp_endpoint:
            return self.otlp_endpoint.startswith("http")
        return bool(self.service_name)


class PrometheusTelemetryAdapter(OpenTelemetryAdapter):
    """Prometheus-flavoured adapter that can export metrics text."""

    def __init__(self, service_name: str, prometheus_port: int = 9100) -> None:
        super().__init__(service_name, otlp_endpoint=None)
        self.port = prometheus_port

    def export_metrics(self) -> str:
        lines = [f"# Metrics for {self.service_name}", f"# Port: {self.port}"]
        for key, value in sorted(self.metrics.items()):
            metric_name = re.sub(r"[^a-zA-Z0-9_]", "_", key.replace(" ", "_"))
            lines.append(f"agent_gantry_{metric_name} {value}")
        return "\n".join(lines)
