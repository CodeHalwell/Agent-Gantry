"""Framework-agnostic rendering of tool-call results to readable text.

Tool results arrive in many shapes depending on the framework that executed
them: a bare string, a Pydantic/JSON-able object, or — for the Microsoft Agent
Framework and similar — a *list of content blocks* where each block exposes a
``.text`` attribute. :func:`render_result` collapses any of these into a single
human-readable string so callers (trace middleware, logs, dashboards) don't
have to special-case each dialect.

This is intentionally dependency-light and import-safe: it never imports a
framework package and only duck-types on ``.text``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["render_result"]


def _block_text(block: Any) -> str:
    """Best-effort text for a single content block."""
    if isinstance(block, str):
        return block
    # AF Content objects (TextContent, FunctionResultContent, ...) expose .text.
    text = getattr(block, "text", None)
    if isinstance(text, str) and text:
        return text
    if isinstance(block, dict):
        for key in ("text", "content", "output"):
            value = block.get(key)
            if isinstance(value, str) and value:
                return value
    return str(block)


def render_result(
    result: Any,
    *,
    limit: int | None = None,
    collapse_whitespace: bool = False,
    placeholder: str = "…",
) -> str:
    """Render a tool-call result as readable text.

    Handles the common result shapes uniformly:

    - ``str`` is returned as-is.
    - ``bytes`` is decoded as UTF-8 (errors replaced).
    - a ``list``/``tuple`` (e.g. AF content blocks) has each item rendered via
      its ``.text`` attribute when present, else ``str(item)``, then joined
      with single spaces (empty parts dropped).
    - anything else falls back to ``str(result)``.

    Args:
        result: The value returned by a tool / function invocation.
        limit: If set, truncate the rendered string to this many characters,
            appending ``placeholder`` when truncation occurs.
        collapse_whitespace: When ``True``, collapse all runs of whitespace
            (including newlines) to single spaces — handy for one-line trace
            output. Defaults to ``False`` so structured text (JSON, tables)
            stays faithful.
        placeholder: Suffix appended when ``limit`` truncates the output.

    Returns:
        A single rendered string (never ``None``).
    """
    if result is None:
        text = ""
    elif isinstance(result, str):
        text = result
    elif isinstance(result, bytes):
        text = result.decode("utf-8", errors="replace")
    elif isinstance(result, (list, tuple)):
        parts = [_block_text(item) for item in result]
        text = " ".join(p for p in parts if p)
    else:
        # Single content-block-like object (has .text) or an arbitrary value.
        text = _block_text(result)

    if collapse_whitespace:
        text = " ".join(text.split())

    if limit is not None and limit >= 0 and len(text) > limit:
        text = text[:limit] + placeholder

    return text
