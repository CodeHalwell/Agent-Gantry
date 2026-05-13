"""
Built-in deterministic query-generation strategies.

These work over any iterable of message-like objects. Each message is
expected to expose at minimum ``role`` and ``text`` attributes; alternative
content fields (``content``) are tried as a fallback. This matches both the
Microsoft Agent Framework message type and the lighter-weight dict-shaped
messages used by other frameworks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def _msg_text(msg: Any) -> str:
    """Pull plain text out of a message-like object.

    Tries ``.text`` first (AF native), then ``.content`` (OpenAI dialect),
    and for dict-shaped messages checks ``"text"`` and ``"content"``.

    If none of those surface text, walks the structured ``contents`` list
    (AF native) looking for text-bearing Content variants:

    - ``type="text"``: use ``.text``
    - ``type="function_result"``: prefer joined ``items[].text``; fall
      back to ``str(.result)`` when the result is a primitive.

    This is important for tool-role messages whose result text is nested
    inside a ``function_result`` Content rather than exposed at the
    Message level — without it, ``last_tool_result`` would always miss.

    Returns the empty string if no non-empty string content is found.
    """
    text = getattr(msg, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(msg, dict):
        for key in ("text", "content"):
            value = msg.get(key)
            if isinstance(value, str) and value.strip():
                return value

    contents = getattr(msg, "contents", None)
    if contents:
        parts: list[str] = []
        for c in contents:
            ctype = getattr(c, "type", None)
            if ctype == "text":
                t = getattr(c, "text", None)
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            elif ctype == "function_result":
                items = getattr(c, "items", None) or []
                for item in items:
                    if getattr(item, "type", None) == "text":
                        t = getattr(item, "text", None)
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
                if not parts:
                    result = getattr(c, "result", None)
                    if isinstance(result, (str, int, float, bool)):
                        s = str(result).strip()
                        if s:
                            parts.append(s)
        if parts:
            return " ".join(parts)

    return ""


def _msg_role(msg: Any) -> str:
    """Get the role of a message as a lowercase string."""
    role = getattr(msg, "role", None)
    if role is None and isinstance(msg, dict):
        role = msg.get("role")
    return str(role).lower() if role is not None else ""


def _msg_name(msg: Any) -> str:
    """Best-effort tool/author name for a tool-role message."""
    for attr in ("author_name", "name", "tool_name"):
        value = getattr(msg, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(msg, dict):
        for key in ("author_name", "name", "tool_name"):
            value = msg.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def last_user_text(messages: Iterable[Any] | None) -> str:
    """Return the most recent user-role message text (or empty string)."""
    if not messages:
        return ""
    for msg in reversed(list(messages)):
        if _msg_role(msg) in ("user", ""):
            text = _msg_text(msg)
            if text.strip():
                return text.strip()
    return ""


def last_assistant_text(messages: Iterable[Any] | None) -> str:
    """Return the most recent assistant-role message text (or empty string).

    Useful when the next retrieval should be driven by the model's most
    recent reasoning rather than the original user input.
    """
    if not messages:
        return ""
    for msg in reversed(list(messages)):
        if _msg_role(msg) == "assistant":
            text = _msg_text(msg)
            if text.strip():
                return text.strip()
    return ""


def last_tool_result(
    messages: Iterable[Any] | None,
    *,
    max_chars: int = 500,
) -> str:
    """Return a description of the most recent tool/function result.

    For fixed multi-step workflows this often produces the best retrieval
    signal: after one tool returns, the next tool to surface should match
    the *content* of that result, not the original user prompt.

    Args:
        messages: Conversation history (most recent last).
        max_chars: Cap on characters of the tool result included in the
            generated query, to keep embeddings focused. Defaults to 500.

    Returns:
        A string of the form ``"result of <tool>: <text>"`` (or just
        ``"tool result: <text>"`` if the tool name is not available), or
        the empty string if no tool message was found.
    """
    if not messages:
        return ""
    for msg in reversed(list(messages)):
        if _msg_role(msg) in ("tool", "function"):
            text = _msg_text(msg)
            if not text.strip():
                continue
            snippet = text.strip()[:max_chars]
            tool_name = _msg_name(msg)
            if tool_name:
                return f"result of {tool_name}: {snippet}"
            return f"tool result: {snippet}"
    return ""


def concatenate_recent(
    messages: Iterable[Any] | None,
    *,
    n: int = 3,
    separator: str = "\n",
) -> str:
    """Concatenate the text of the last ``n`` messages.

    Args:
        messages: Conversation history.
        n: Number of trailing messages to include. Defaults to 3.
        separator: String used to join the messages. Defaults to newline.

    Returns:
        The joined non-empty texts, or the empty string if none were found.
    """
    if not messages:
        return ""
    msgs = list(messages)[-n:]
    parts = [t for t in (_msg_text(m) for m in msgs) if t.strip()]
    return separator.join(p.strip() for p in parts)


def fallback_chain(
    *generators: Callable[[Iterable[Any] | None], str],
) -> Callable[[Iterable[Any] | None], str]:
    """Compose multiple generators, returning the first non-empty result.

    .. code-block:: python

        from agent_gantry.query import (
            fallback_chain, last_tool_result, last_assistant_text, last_user_text,
        )

        gen = fallback_chain(last_tool_result, last_assistant_text, last_user_text)
        query = gen(messages)

    Returns:
        A callable with the same shape as the inputs.
    """

    def _composed(messages: Iterable[Any] | None) -> str:
        for gen in generators:
            text = gen(messages)
            if text and text.strip():
                return text
        return ""

    return _composed
