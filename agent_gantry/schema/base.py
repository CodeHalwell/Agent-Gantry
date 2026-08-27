"""Shared building blocks for Agent-Gantry schema models.

Houses the small pieces that several schema models would otherwise duplicate:
the identifier newline-rejection validator (a security invariant) and the
health-metric fields common to tools and MCP servers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

__all__ = ["HealthMetrics", "json_identity_key", "reject_newlines"]


def json_identity_key(value: Any) -> Any:
    """A hashable key equal for exactly the values JSON Schema calls equal.

    Used to enforce ``uniqueItems``. Plain Python equality gets this wrong in
    one specific, reachable way: ``True == 1``, so ``[1, true]`` — two
    distinct instance values in JSON Schema, which compares types before
    values — looked like a duplicate and was rejected. Tagging each value with
    its JSON type separates the two while leaving ``1`` and ``1.0`` equal, as
    JSON Schema requires of numbers.

    Containers are keyed recursively so the same distinction holds inside them
    (``[[1]]`` and ``[[true]]`` are different), and the result is hashable, so
    callers can use a ``set`` rather than a quadratic scan.

    Lives here rather than beside either caller because the executor's
    validator and the framework args-model bridge must agree on it: a
    disagreement means a framework accepts a payload the engine then rejects.
    """
    if isinstance(value, bool):
        # Checked before int — ``bool`` subclasses it.
        return ("boolean", value)
    if isinstance(value, (int, float)):
        return ("number", value)
    if isinstance(value, str):
        return ("string", value)
    if value is None:
        return ("null", None)
    if isinstance(value, (list, tuple)):
        return ("array", tuple(json_identity_key(item) for item in value))
    if isinstance(value, dict):
        return (
            "object",
            tuple(sorted(((k, json_identity_key(v)) for k, v in value.items()), key=lambda kv: kv[0])),
        )
    # Not a JSON value at all (a handler default, say). Fall back to the value
    # itself; an unhashable one makes the caller drop to an equality scan.
    return ("other", value)


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
