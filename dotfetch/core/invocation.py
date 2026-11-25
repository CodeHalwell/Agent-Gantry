"""Ambient access to the calling framework's run state from inside a tool.

Framework adapters wrap each tool execution in :func:`tool_invocation`, and a
dotfetch tool handler reads it back with :func:`current_tool_invocation`::

    from dotfetch import current_tool_invocation

    @fetcher.register
    async def lookup_order(order_id: str) -> str:
        \"\"\"Look up an order for the current user.\"\"\"
        invocation = current_tool_invocation()
        db = invocation.deps.db if invocation else default_db
        return await db.fetch_order(order_id)

The tool's JSON schema is unchanged (the model never sees the context), and
the same handler still works when called with no framework at all --
:func:`current_tool_invocation` simply returns ``None``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from dotfetch.schema.execution import ToolInvocationContext

_CURRENT: ContextVar[ToolInvocationContext | None] = ContextVar(
    "dotfetch_tool_invocation", default=None
)


def current_tool_invocation() -> ToolInvocationContext | None:
    """Return the framework run state for the tool call executing now, if any."""
    return _CURRENT.get()


@contextmanager
def tool_invocation(context: ToolInvocationContext) -> Iterator[ToolInvocationContext]:
    """Expose ``context`` to tool handlers executed inside this block."""
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


__all__ = ["ToolInvocationContext", "current_tool_invocation", "tool_invocation"]
