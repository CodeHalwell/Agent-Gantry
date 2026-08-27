"""Shared building blocks for Agent-Gantry schema models.

Houses the small pieces that several schema models would otherwise duplicate:
the identifier newline-rejection validator (a security invariant) and the
health-metric fields common to tools and MCP servers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "HealthMetrics",
    "json_identity_key",
    "reject_newlines",
    "resolve_numeric_bounds",
    "schema_declares_null",
]


def schema_declares_null(prop: Any) -> bool:
    """Whether a property schema gives ``null`` a declared, meaningful value.

    ``null`` reaches a schema two ways: a ``type`` that names it (``"null"``
    or a list containing it) and a combinator branch —
    ``{"anyOf": [{"type": "string"}, {"type": "null"}]}``, which is what
    Pydantic and OpenAPI emit for ``str | None``. Checking only ``type`` would
    miss the far more common of the two.

    ``allOf`` intersects its branches, so the combined schema admits null only
    when *every* branch does; ``any()`` there would call
    ``[{"type": ["string", "null"]}, {"type": "string"}]`` nullable and
    preserve a null the schema actually forbids.

    Shared between the executor's argument normalization and the framework
    ``ToolSpec`` path: both decide whether an explicit ``null`` is a value the
    caller meant or strict mode's "not provided" placeholder, and they must
    decide it identically.
    """
    if not isinstance(prop, dict):
        return False
    # ``enum``/``const`` apply independently of ``type``, so a schema can name
    # null in its type list and still forbid it: an optional ``Literal`` that
    # strict mode pre-widened arrives as ``{"type": ["string", "null"],
    # "enum": ["fast", "slow"]}``. Treating that as nullable preserved a
    # strict-mode placeholder the canonical schema then rejected, instead of
    # dropping it so the handler's default applies.
    enum_values = prop.get("enum")
    if isinstance(enum_values, list) and enum_values and None not in enum_values:
        return False
    if "const" in prop and prop["const"] is not None:
        return False
    prop_type = prop.get("type")
    if prop_type == "null" or (isinstance(prop_type, list) and "null" in prop_type):
        return True
    for key in ("anyOf", "oneOf"):
        branches = prop.get(key)
        if isinstance(branches, list) and any(schema_declares_null(b) for b in branches):
            return True
    all_of = prop.get("allOf")
    if isinstance(all_of, list):
        usable = [b for b in all_of if isinstance(b, dict) and b]
        if usable and all(schema_declares_null(b) for b in usable):
            return True
    return False


def _numeric(value: Any) -> float | int | None:
    """``value`` when it is a JSON number, else ``None`` (booleans excluded)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def resolve_numeric_bounds(schema: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """A schema's bounds as ``(min, max, exclusiveMin, exclusiveMax)``.

    Two JSON Schema dialects spell exclusivity differently and both reach this
    library. Modern drafts give ``exclusiveMinimum`` a *number*; draft-04 —
    which is what OpenAPI 3.0 emits, and OpenAPI/MCP import is a supported way
    to register a tool — gives it a *boolean* that promotes ``minimum`` to an
    exclusive bound. Reading only the modern form left the boolean ignored and
    the bound applied inclusively, so ``{"minimum": 5, "exclusiveMinimum":
    true}`` accepted ``5``.

    Lives here, beside :func:`json_identity_key`, for the same reason: the
    executor's validator and the framework args-model bridge must read a bound
    identically, or a framework accepts what the engine rejects.
    """
    lower = _numeric(schema.get("minimum"))
    upper = _numeric(schema.get("maximum"))
    excl_lower = schema.get("exclusiveMinimum")
    excl_upper = schema.get("exclusiveMaximum")

    if isinstance(excl_lower, bool):
        excl_lower, lower = (lower, None) if excl_lower else (None, lower)
    else:
        excl_lower = _numeric(excl_lower)
    if isinstance(excl_upper, bool):
        excl_upper, upper = (upper, None) if excl_upper else (None, upper)
    else:
        excl_upper = _numeric(excl_upper)

    return lower, upper, excl_lower, excl_upper


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
