"""
Execution models for Agent-Gantry.

Models for tool calls, results, and batch operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_gantry.schema.base import reject_newlines


class ExecutionStatus(str, Enum):
    """Status of a tool execution."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    CIRCUIT_OPEN = "circuit_open"
    PENDING_CONFIRMATION = "pending_confirmation"


class ToolCall(BaseModel):
    """Request to execute a tool."""

    tool_name: str
    #: Namespace the call targets. ``None`` means "resolve by bare name",
    #: which is what a provider tool-call payload can express -- the model only
    #: ever sees the bare name. Callers that already know which tool was
    #: selected (every framework adapter, since selection is namespace-aware)
    #: should set this, otherwise a same-named tool in another namespace can
    #: be executed instead. A qualified ``tool_name`` ("billing.search") is
    #: also accepted and takes effect when this field is unset.
    namespace: str | None = None
    arguments: dict[str, Any]

    #: Wall-clock limit for *each attempt*, so a call that is retried can take
    #: up to ``timeout_ms * (retry_count + 1)`` plus the back-off between
    #: attempts. ``None`` takes the engine's ``default_timeout_ms``
    #: (``ExecutionConfig.default_timeout_ms``), which a fixed 30000 here used
    #: to make impossible to configure.
    timeout_ms: int | None = Field(default=None, ge=100, le=300000)
    #: Retries after a failed or timed-out attempt. ``0`` means none; ``None``
    #: takes the engine's ``max_retries`` (``ExecutionConfig.max_retries``).
    #: ``0`` used to be the field default *and* read as "unset", so asking for
    #: no retries was impossible and a non-idempotent tool was run again.
    retry_count: int | None = Field(default=None, ge=0, le=5)
    require_confirmation: bool | None = None

    trace_id: str | None = None
    parent_span_id: str | None = None

    model_config = ConfigDict(validate_assignment=True)

    _reject_newline_identifiers = field_validator("tool_name", "trace_id", "parent_span_id")(
        reject_newlines
    )


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

    model_config = ConfigDict(validate_assignment=True)

    _reject_newline_identifiers = field_validator("tool_name", "trace_id", "span_id")(
        reject_newlines
    )

    @property
    def latency_ms(self) -> float:
        """Calculate execution latency in milliseconds."""
        if self.started_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0


class BatchToolCall(BaseModel):
    """Request to execute multiple tools."""

    calls: list[ToolCall]
    #: ``adaptive`` runs the calls one at a time when ``fail_fast`` is set (it
    #: cannot stop a call that is already running) and concurrently otherwise.
    execution_strategy: Literal["parallel", "sequential", "adaptive"] = "adaptive"
    #: Stop at the first call that does not succeed. Honoured by ``sequential``
    #: and by ``adaptive``; ``parallel`` starts every call at once.
    fail_fast: bool = False


class BatchToolResult(BaseModel):
    """Result of a batch tool execution."""

    results: list[ToolResult]
    total_time_ms: float
    successful_count: int
    failed_count: int


@dataclass(frozen=True)
class ToolCallEvent:
    """A completed tool execution, delivered to ``gantry.on_tool_call`` callbacks.

    Intentionally a ``@dataclass`` rather than a Pydantic ``BaseModel`` like the
    rest of this module: it is an ephemeral, in-process event (never serialised
    or validated), so a frozen dataclass keeps it allocation-cheap on the
    ``execute`` hot path. Don't "promote" it to a BaseModel without reason.

    Framework-agnostic: every call routed through
    :meth:`~agent_gantry.core.gantry.AgentGantry.execute` (and each call in a
    batch) emits one of these once execution finishes — successfully or not —
    regardless of which agent framework (if any) drove the call. This is the
    single seam for cross-framework logging and metrics; the convenience
    accessors mirror the most-used fields of the underlying result.
    """

    call: ToolCall
    result: ToolResult

    @property
    def tool_name(self) -> str:
        """Name of the tool that was executed.

        Prefers ``result.tool_name`` and falls back to ``call.tool_name`` only
        defensively (the result name is normally always populated; the fallback
        covers any error path that produced a result without one).
        """
        return self.result.tool_name or self.call.tool_name

    @property
    def status(self) -> ExecutionStatus:
        """Terminal status of the execution."""
        return self.result.status

    @property
    def ok(self) -> bool:
        """``True`` when the call completed successfully."""
        return self.result.status == ExecutionStatus.SUCCESS

    @property
    def latency_ms(self) -> float:
        """Execution latency in milliseconds (0.0 if not started)."""
        return self.result.latency_ms
