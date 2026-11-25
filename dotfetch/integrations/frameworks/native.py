"""Framework-neutral building blocks for first-class native integrations.

Every first-class adapter (Pydantic AI, OpenAI Agents SDK, ...) needs the same
few things around ``fetcher.execute``; they live here so each framework module
only maps them onto its own mechanisms:

- :data:`ErrorMode` / :data:`ErrorFormatter` / :func:`failure_text` -- how a
  failed execution is shown to the model (capped, optionally redacted, failing
  closed if the formatter itself breaks);
- :func:`execute_spec` -- run a selected tool through ``fetcher.execute`` with the
  framework's run state visible to the handler via
  :func:`dotfetch.current_tool_invocation` and the caller's trace attached;
- :func:`otel_ids` -- the active OpenTelemetry span, for trace propagation;
- :func:`carry_over` -- the sticky, capped tool surface a per-turn provider
  offers across the steps of one run.

Nothing here imports a third-party framework.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Any, Literal

from dotfetch.core.invocation import tool_invocation
from dotfetch.integrations.frameworks.base import (
    ToolExecutionError,
    ToolPermissionDeniedError,
    ToolSpec,
    _coerce_arguments,
)
from dotfetch.schema.execution import ToolInvocationContext

logger = logging.getLogger(__name__)

#: How a failed dotfetch execution surfaces inside a framework's run.
#:
#: - ``"model"`` -- the framework's own "tell the model" mechanism: the model
#:   sees the failure text (see :func:`failure_text`) and can correct or adapt;
#:   confirmation-gated calls use the framework's human-approval flow.
#: - ``"raise"`` -- the cross-framework default contract: the
#:   :class:`~dotfetch.integrations.frameworks.base.ToolExecutionError`
#:   family propagates.
ErrorMode = Literal["model", "raise"]

#: Turns a failed execution into the text the model sees, e.g. to redact
#: paths, hostnames or connection strings from handler exception messages. The
#: result is still capped; a formatter that raises is replaced by a generic
#: message, so a bug in it cannot leak the original text.
ErrorFormatter = Callable[[ToolExecutionError], str]

#: Executor ``error_type`` values the model can fix by changing its call.
MODEL_CORRECTABLE_ERRORS = frozenset({"ValidationError", "ToolNotFound"})

#: Longest failure text shown to the model.
MAX_ERROR_CHARS = 2000

#: Only the head of a tool result is a useful retrieval signal, and embedders
#: have input limits.
MAX_TOOL_RESULT_QUERY_CHARS = 2000


def check_error_mode(error_mode: str) -> ErrorMode:
    """Validate an ``error_mode`` value early, at construction time."""
    if error_mode not in ("model", "raise"):
        raise ValueError(f"error_mode must be 'model' or 'raise', not {error_mode!r}")
    return error_mode  # type: ignore[return-value]


def capped(message: str) -> str:
    """``message`` cut to :data:`MAX_ERROR_CHARS`, marked when truncated."""
    if len(message) <= MAX_ERROR_CHARS:
        return message
    return message[:MAX_ERROR_CHARS] + " ... [truncated]"


def _formatted(exc: ToolExecutionError, formatter: ErrorFormatter) -> str:
    try:
        return capped(str(formatter(exc)))
    except Exception:
        logger.warning(
            "error_formatter raised; sending a generic failure message to the model instead.",
            exc_info=True,
        )
        return f"{exc.error_type or 'ToolError'}: the tool call failed."


def failure_text(exc: ToolExecutionError, formatter: ErrorFormatter | None = None) -> str:
    """The model-facing text for a failed execution (``error_type: message``, capped).

    With ``formatter``, its output replaces the default text (see :data:`ErrorFormatter`).
    """
    if formatter is not None:
        return _formatted(exc, formatter)
    message = exc.error or "tool execution failed (no error message)"
    if isinstance(exc, ToolPermissionDeniedError):
        message = f"Permission denied: {message}"
    elif (
        exc.error_type
        and exc.error_type not in message
        and exc.error_type not in MODEL_CORRECTABLE_ERRORS
    ):
        message = f"{exc.error_type}: {message}"
    return capped(message)


def otel_ids() -> tuple[str | None, str | None]:
    """The active OpenTelemetry span's trace/span ids, when one is recording."""
    try:
        from opentelemetry import trace
    except ImportError:
        return None, None
    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None, None
    return format(span_context.trace_id, "032x"), format(span_context.span_id, "016x")


async def execute_spec(
    spec: ToolSpec,
    arguments: Any,
    *,
    framework: str,
    native_context: Any = None,
    deps: Any = None,
    run_id: str | None = None,
    tool_call_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    approved: bool = False,
    trace_ids: tuple[str | None, str | None] | None = None,
) -> Any:
    """Run ``spec`` through ``fetcher.execute`` on behalf of a framework tool call.

    Raises the :class:`ToolExecutionError` family on failure; each framework
    maps that onto its own control flow.

    Args:
        spec: The selected tool.
        arguments: The model's arguments (a mapping).
        framework: Canonical framework name for :class:`ToolInvocationContext`.
        native_context: The framework's per-call context object, exposed to the
            handler through :func:`dotfetch.current_tool_invocation`.
        deps: The framework's dependency-injection payload.
        run_id: The framework's run identifier, when it has one.
        tool_call_id: The model-issued id of this call.
        metadata: Framework-specific extras for the invocation context.
        approved: The framework's human-approval flow approved this exact call;
            clears dotfetch's confirmation gate (every denial check still runs).
        trace_ids: ``(trace_id, parent_span_id)``; defaults to the active
            OpenTelemetry span.
    """
    trace_id, parent_span_id = trace_ids if trace_ids is not None else otel_ids()
    args = _coerce_arguments((), dict(arguments or {}))
    scope = (
        tool_invocation(
            ToolInvocationContext(
                framework=framework,
                native_context=native_context,
                deps=deps,
                run_id=run_id,
                tool_call_id=tool_call_id,
                metadata=dict(metadata or {}),
            )
        )
        if native_context is not None
        else nullcontext()
    )
    with scope:
        return await spec._ainvoke(
            args,
            confirmation_approved=approved,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
        )


def carry_over(
    offered: dict[str, ToolSpec],
    last_selected: dict[str, int],
    step: int,
    selected: list[ToolSpec],
    max_offered: int,
) -> list[ToolSpec]:
    """This step's selection plus recent earlier ones, up to ``max_offered``.

    ``offered`` (insertion-ordered, so the surface stays stable) and
    ``last_selected`` (name -> step last selected) are the caller's per-run
    state, updated in place. The current selection -- pins included, since they
    are in every selection -- is always offered in full and refreshes its specs;
    carried-over tools are dropped least recently selected first.
    """
    current = set()
    for spec in selected:
        offered[spec.native_name] = spec
        last_selected[spec.native_name] = step
        current.add(spec.native_name)
    evictable = sorted(
        (name for name in offered if name not in current),
        key=lambda name: last_selected.get(name, 0),
    )
    while len(offered) > max_offered and evictable:
        name = evictable.pop(0)
        del offered[name]
        last_selected.pop(name, None)
    return list(offered.values())


__all__ = [
    "MAX_ERROR_CHARS",
    "MAX_TOOL_RESULT_QUERY_CHARS",
    "MODEL_CORRECTABLE_ERRORS",
    "ErrorFormatter",
    "ErrorMode",
    "capped",
    "carry_over",
    "check_error_mode",
    "execute_spec",
    "failure_text",
    "otel_ids",
]
