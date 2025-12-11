"""
Event models for Agent-Gantry observability.

Structured events for retrieval, execution, and health changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agent_gantry.schema.tool import ToolHealth


class RetrievalEvent(BaseModel):
    """Event emitted after a tool retrieval operation."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str
    query: str
    candidate_count: int
    returned_count: int
    total_time_ms: float
    tool_names: list[str]
    scores: list[float]
    filters_applied: dict[str, Any] = Field(default_factory=dict)


class ExecutionEvent(BaseModel):
    """Event emitted after a tool execution."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str
    span_id: str
    tool_name: str
    status: str
    latency_ms: float
    attempt_number: int
    error: str | None = None
    error_type: str | None = None


class HealthChangeEvent(BaseModel):
    """Event emitted when a tool's health status changes."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tool_name: str
    old_health: ToolHealth
    new_health: ToolHealth
    reason: str
