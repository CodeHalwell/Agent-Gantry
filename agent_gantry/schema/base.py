"""Shared building blocks for Agent-Gantry schema models.

Houses the small pieces that several schema models would otherwise duplicate:
the identifier newline-rejection validator (a security invariant) and the
health-metric fields common to tools and MCP servers.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


def reject_newlines(value: str | None) -> str | None:
    """Reject newline characters in identifier fields.

    Pydantic v2's Rust regex engine treats ``$`` as end-of-line rather than
    end-of-string, so a ``pattern=r"^...$"`` would accept ``"valid_name\\n"``.
    Attaching this as a reusable ``field_validator`` to every identifier field
    closes that bypass with a single, shared definition.
    """
    if isinstance(value, str) and ("\n" in value or "\r" in value):
        raise ValueError("Value cannot contain newline characters")
    return value


class HealthMetrics(BaseModel):
    """Runtime health fields shared by tools and MCP servers.

    Subclasses add the metrics specific to what they track (per-call latency
    and circuit-breaker state for tools; connection counts and availability for
    MCP servers).
    """

    success_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    consecutive_failures: int = Field(default=0)
    last_success: datetime | None = None
    last_failure: datetime | None = None
