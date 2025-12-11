"""
Execution models for Agent-Gantry.

Models for tool calls, results, and batch operations.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    """Status of a tool execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    CIRCUIT_OPEN = "circuit_open"
    PENDING_CONFIRMATION = "pending_confirmation"
    CANCELLED = "cancelled"


class ToolCall(BaseModel):
    """Request to execute a tool."""

    tool_name: str
    arguments: dict[str, Any]

    timeout_ms: int = Field(default=30000, ge=100, le=300000)
    retry_count: int = Field(default=0, ge=0, le=5)
    require_confirmation: bool | None = None

    trace_id: str | None = None
    parent_span_id: str | None = None


class ToolResult(BaseModel):
    """Result of a tool execution."""

    tool_name: str
    status: ExecutionStatus

    result: Any | None = None
    error: str | None = None
    error_type: str | None = None

    queued_at: datetime
    started_at: datetime | None = None
    completed_at: datetime

    attempt_number: int = Field(default=1)

    trace_id: str
    span_id: str

    @property
    def latency_ms(self) -> float:
        """Calculate execution latency in milliseconds."""
        if self.started_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0


class BatchToolCall(BaseModel):
    """Request to execute multiple tools."""

    calls: list[ToolCall]
    execution_strategy: Literal["parallel", "sequential", "adaptive"] = "adaptive"
    fail_fast: bool = False


class BatchToolResult(BaseModel):
    """Result of a batch tool execution."""

    results: list[ToolResult]
    total_time_ms: float
    successful_count: int
    failed_count: int
