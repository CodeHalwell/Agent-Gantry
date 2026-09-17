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

import json
from typing import Any

__all__ = ["render_result"]


#: The ``type`` discriminators of every MCP content block, from the SDK's own
#: ``ContentBlock`` union (TextContent, ImageContent, AudioContent,
#: ResourceLink, EmbeddedResource). Spelled out rather than imported: this
#: module is deliberately dependency-light and import-safe. A test pins the
#: set against the installed SDK so a protocol addition cannot drift past it.
_MCP_BLOCK_TYPES = frozenset({"text", "image", "audio", "resource", "resource_link"})


def _is_content_block(item: Any) -> bool:
    """Whether ``item`` is an MCP content block.

    Accepting *any* string ``type`` was too weak to be identity: typed lists
    are ordinary outside MCP, so a record like ``{"title": ..., "score": ...,
    "content": [{"type": "paragraph", "text": ...}]}`` was taken for a result
    and rendered as its blocks, dropping every sibling field. The protocol
    names its block types, so membership is the check.
    """
    kind = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
    return isinstance(kind, str) and kind in _MCP_BLOCK_TYPES


def _result_content(value: Any) -> Any:
    """A result's ``content``, read through a mapping as well as an object.

    A proxied or JSON-decoded ``CallToolResult`` arrives as a plain dict,
    where ``getattr`` finds nothing: such a result was never recognised and
    rendered as its own repr instead of its text.
    """
    if isinstance(value, dict):
        return value.get("content")
    return getattr(value, "content", None)


def _is_mcp_result(value: Any) -> bool:
    """Whether ``value`` wraps content blocks, or merely has a ``content`` field.

    Duck-typing on ``content`` alone swept up ordinary records —
    ``Article(title="...", content=["body"], score=0.9)`` is the obvious case
    — and unwrapping those emitted the content items and dropped the title
    and the score, the same defect the ``.text`` path has to guard against.
    """
    content = _result_content(value)
    if not isinstance(content, (list, tuple)) or isinstance(value, type):
        return False
    if content:
        # Something to unwrap: the items decide, and nothing else can. Letting
        # a marker field vouch for them classified
        # ``Result(content=["body"], is_error=False, score=0.9)`` as a result
        # — ``False is not None`` — and dropped the score. An attribute name
        # is not protocol identity when the payload itself can be checked.
        return all(_is_content_block(item) for item in content)
    # Empty content: a result answering entirely through ``structuredContent``
    # is indistinguishable from a plain object by payload alone, so here its
    # own protocol fields are all there is to go on. Both spellings of each:
    # mcp 2.x renamed ``isError`` to ``is_error`` and ``structuredContent`` to
    # ``structured_content``, and reading only the 1.x names sent a 2.x error
    # result to ``str()``. Nothing is dropped by a false positive here, since
    # there are no content items to emit in the first place.
    names = ("structuredContent", "structured_content", "isError", "is_error")
    if isinstance(value, dict):
        return any(value.get(name) is not None for name in names)
    return any(getattr(value, name, None) is not None for name in names)


def _block_text(block: Any) -> str:
    """Best-effort text for a single content block."""
    if isinstance(block, str):
        return block
    # AF Content objects (TextContent, FunctionResultContent, ...) expose .text.
    # A content block declaring ``type`` has said what it is, so its text is
    # taken at face value — ``""`` included. Requiring a truthy string treated
    # a tool that legitimately returned nothing as having no text at all and
    # emitted the block's repr instead, and disagreed with
    # ``mcp_server._is_text_block``, which accepts an empty one.
    declared = _is_content_block(block)
    text = getattr(block, "text", None)
    if isinstance(text, str) and (text or declared):
        return text
    if isinstance(block, dict):
        for key in ("text", "content", "output"):
            value = block.get(key)
            if isinstance(value, str) and (value or (declared and key == "text")):
                return value
    return str(block)


def _structured_text(result: Any) -> str:
    """JSON for a result's structured content, or ``""`` if it carries none.

    An MCP ``CallToolResult`` may answer entirely through
    ``structuredContent``, leaving ``content`` empty. Rendering the blocks
    alone then produced ``""`` — a successful call reported as no output,
    with the actual result silently dropped.
    """
    for attribute in ("structuredContent", "structured_content"):
        value = getattr(result, attribute, None)
        if value is None and isinstance(result, dict):
            value = result.get(attribute)
        if value in (None, {}, []):
            continue
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)
    return ""


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
        if _is_mcp_result(result):
            # A result object wrapping content blocks — an MCP CallToolResult
            # proxied from an upstream server, say — renders as its blocks,
            # falling back to its structured content when those yield nothing.
            # Identity is required: a plain record whose ``content`` happens to
            # hold a list is not one, and unwrapping it dropped its siblings.
            # The ``.text`` path below stays duck-typed, as this helper's
            # callers rely on.
            parts = [_block_text(item) for item in _result_content(result)]
            text = " ".join(p for p in parts if p) or _structured_text(result)
        else:
            # Single content-block-like object (has .text) or an arbitrary value.
            text = _block_text(result)

    if collapse_whitespace:
        text = " ".join(text.split())

    if limit is not None and limit >= 0 and len(text) > limit:
        text = text[:limit] + placeholder

    return text
