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
    if contents is None and isinstance(msg, dict):
        contents = msg.get("contents")
    if contents:
        parts: list[str] = []
        for c in contents:
            ctype = getattr(c, "type", None)
            if ctype == "text":
                t = getattr(c, "text", None)
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            elif ctype == "function_result":
                # Track per-content contribution: the `.result` fallback
                # must fire when *this* function_result had no item text,
                # independently of whether earlier contents in the same
                # message produced any text.
                contributed = False
                items = getattr(c, "items", None) or []
                for item in items:
                    if getattr(item, "type", None) == "text":
                        t = getattr(item, "text", None)
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
                            contributed = True
                if not contributed:
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


# Imperative scaffolding tokens that crowd out content nouns/verbs in
# instruction-style queries. Lowered for case-insensitive matching.
_SCAFFOLD_TOKENS: frozenset[str] = frozenset(
    {
        "please", "kindly", "thanks", "thank", "you", "your", "yours",
        "must", "should", "shall", "will", "would", "could", "can",
        "do", "don't", "dont", "not", "no", "never", "always",
        "i", "me", "my", "we", "us", "our", "ours",
        "the", "a", "an", "this", "that", "these", "those",
        "is", "are", "was", "were", "be", "been", "being", "am",
        "have", "has", "had", "having",
        "to", "from", "of", "in", "on", "at", "by", "for", "with",
        "and", "or", "but", "if", "then", "else", "so", "as",
        "step", "steps", "first", "next", "last", "final", "finally",
        "use", "using", "run", "make", "ensure", "remember",
        "different", "each", "every", "any", "some", "one", "two",
        "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "pipeline", "task", "instruction", "instructions",
        "it", "its", "they", "them", "their",
    }
)


_KEYWORD_TOKEN_RE = None  # type: ignore[var-annotated]


def _tokenize_for_keywords(text: str) -> list[str]:
    """Split into alpha/numeric tokens, lowercase."""
    import re

    global _KEYWORD_TOKEN_RE
    if _KEYWORD_TOKEN_RE is None:
        _KEYWORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
    return [m.group(0).lower() for m in _KEYWORD_TOKEN_RE.finditer(text)]


def keyword_focused(
    messages: Iterable[Any] | None,
    *,
    max_tokens: int = 32,
    base: Callable[[Iterable[Any] | None], str] | None = None,
) -> str:
    """Strip imperative scaffolding from the conversation tail.

    Long instructional queries ("Please run this five-step pipeline. Do
    not skip steps...") dilute the embedding signal because most of the
    text is verbs of obligation and connective tissue rather than the
    nouns/verbs that describe what tool you actually want. This
    generator pulls a base text out of the messages (default:
    :func:`last_user_text`), drops obvious scaffolding tokens, and keeps
    the residual content words.

    Args:
        messages: Conversation history (most recent last).
        max_tokens: Cap on the number of retained tokens. Defaults to 32.
        base: Generator used to extract the source text. Defaults to
            :func:`last_user_text`.

    Returns:
        A space-joined string of retained tokens (lowercased). Returns
        the empty string when the base generator does.
    """
    base_fn = base or last_user_text
    raw = base_fn(messages)
    if not raw or not raw.strip():
        return ""
    tokens = _tokenize_for_keywords(raw)
    kept: list[str] = []
    for tok in tokens:
        if len(tok) < 2:
            continue
        if tok in _SCAFFOLD_TOKENS:
            continue
        kept.append(tok)
        if len(kept) >= max_tokens:
            break
    return " ".join(kept)


def truncated(
    generator: Callable[..., Any],
    *,
    max_chars: int = 200,
    keep: str = "tail",
) -> Callable[[Iterable[Any] | None], Any]:
    """Wrap a generator to cap the produced query length.

    Sync and async generators are both supported; the returned wrapper
    mirrors the underlying coroutine-ness so it plugs into
    ``GantryContextProvider`` without further adaptation.

    Args:
        generator: The generator to wrap.
        max_chars: Maximum number of characters in the output. Defaults
            to 200.
        keep: ``"tail"`` (default) keeps the final ``max_chars``
            characters — typically the latest tool output. ``"head"``
            keeps the leading ``max_chars`` characters.

    Returns:
        A new generator with the same call shape as ``generator``.
    """
    if keep not in ("head", "tail"):
        raise ValueError(f"keep must be 'head' or 'tail', got {keep!r}")

    def _cap(text: str) -> str:
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        if keep == "tail":
            return text[-max_chars:]
        return text[:max_chars]

    import inspect

    if inspect.iscoroutinefunction(generator):

        async def _async_wrapper(messages: Iterable[Any] | None) -> str:
            value = await generator(messages)
            return _cap(value or "")

        return _async_wrapper

    def _wrapper(messages: Iterable[Any] | None) -> str:
        value = generator(messages)
        return _cap(value or "")

    return _wrapper


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
