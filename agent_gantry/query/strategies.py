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


def _blocks_text(value: Any) -> str:
    """Join the text of a list of content blocks.

    Handles the structured ``content`` lists used by the OpenAI Responses API /
    Agents SDK (``[{"type": "input_text", "text": "..."}]``) and multimodal
    messages, where each item is a string, a dict with a ``text``/``input_text``
    field, or an object exposing ``.text``. Returns ``""`` for non-lists.
    """
    if not isinstance(value, (list, tuple)):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                parts.append(item.strip())
        elif isinstance(item, dict):
            t = item.get("text") or item.get("input_text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
            elif item.get("type") == "tool_result":
                # Anthropic Messages API: a tool's output lives in a
                # ``tool_result`` block's ``content`` (a string or a list of
                # text blocks), never under a ``text`` key.
                t = _tool_result_text(item.get("content"))
                if t:
                    parts.append(t)
        elif getattr(item, "type", None) == "tool_result":
            # Anthropic SDK object form of a ``tool_result`` block (as
            # opposed to the dict form the API/dict-based callers use).
            t = _tool_result_text(getattr(item, "content", None))
            if t:
                parts.append(t)
        else:
            t = getattr(item, "text", None)
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
    return " ".join(parts)


def _tool_result_text(content: Any) -> str:
    """Text of an Anthropic ``tool_result`` block's ``content``."""
    if isinstance(content, str):
        return content.strip()
    return _blocks_text(content)


def _is_tool_result_message(msg: Any) -> bool:
    """Whether ``msg`` is an Anthropic-style user turn made only of tool results.

    The Messages API returns tool output to the model as a ``user``-role
    message whose content is a list of ``tool_result`` blocks. For routing
    that is a tool message: it carries no new request from the user, and its
    text is what the agent just learned.
    """
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if not isinstance(content, (list, tuple)) or not content:
        return False
    return all(_block_type(block) == "tool_result" for block in content)


def _block_type(block: Any) -> Any:
    """A content block's ``type``, whether it is a dict or an SDK object."""
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


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

    kind = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", None)
    if kind == "function_call_output":
        # OpenAI Responses API / Agents SDK: a tool's output is a top-level
        # input item carrying ``output`` rather than ``content`` — a plain
        # dict from a caller building input by hand, or an SDK response
        # object (e.g. ``ResponseFunctionToolCallOutputItem``) exposing the
        # same field as an attribute.
        output = msg.get("output") if isinstance(msg, dict) else getattr(msg, "output", None)
        if isinstance(output, str) and output.strip():
            return output
        block_text = _blocks_text(output)
        if block_text:
            return block_text

    # ``content`` as a list of structured blocks (OpenAI Responses input parts,
    # multimodal messages) — pull the text parts.
    block_text = _blocks_text(content)
    if block_text:
        return block_text
    if isinstance(msg, dict):
        block_text = _blocks_text(msg.get("content"))
        if block_text:
            return block_text

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


# LangChain messages expose the role as ``.type`` ("human"/"ai"/...) rather
# than ``.role``. Only these known values are treated as roles so an unrelated
# ``type`` attribute is never mistaken for one.
_LANGCHAIN_ROLES = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
}


def _msg_role(msg: Any) -> str:
    """Get the role of a message as a lowercase string.

    Tool output is reported as ``"tool"`` whichever way the format spells it:
    a ``tool``/``function`` role, an Anthropic user turn made only of
    ``tool_result`` blocks, or an OpenAI Responses ``function_call_output``
    item (which has no role at all).
    """
    role = getattr(msg, "role", None)
    if role is None and isinstance(msg, dict):
        role = msg.get("role")
    if role is not None:
        role = str(role).lower()
        if role == "user" and _is_tool_result_message(msg):
            return "tool"
        return role

    kind = getattr(msg, "type", None)
    if kind is None and isinstance(msg, dict):
        kind = msg.get("type")
    if isinstance(kind, str):
        kind = kind.lower()
        # OpenAI Responses API input items carry a ``type`` and no role.
        if kind == "function_call_output":
            return "tool"
        if kind == "function_call":
            return "assistant"
        # LangChain ``BaseMessage`` and its dict form carry the role in ``type``.
        return _LANGCHAIN_ROLES.get(kind, "")

    return ""


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


def _called_tool_names(msg: Any) -> list[str]:
    """Names of the tools ``msg`` *calls*, in the shapes agent SDKs emit."""
    found: list[Any] = []

    # OpenAI Chat Completions assistant ``tool_calls`` (dicts or SDK objects
    # with ``.function.name``) and LangChain ``AIMessage.tool_calls``
    # (``{"name": ..., "args": ...}`` dicts).
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls is None and isinstance(msg, dict):
        tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, (list, tuple)):
        for call in tool_calls:
            function = (
                call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
            )
            name = None
            if function is not None:
                name = (
                    function.get("name")
                    if isinstance(function, dict)
                    else getattr(function, "name", None)
                )
            if name is None:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            found.append(name)

    # OpenAI Responses API / Agents SDK ``function_call`` item.
    kind = getattr(msg, "type", None)
    if kind is None and isinstance(msg, dict):
        kind = msg.get("type")
    if kind == "function_call":
        found.append(msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", None))

    # Anthropic ``tool_use`` blocks and Strands ``toolUse`` blocks in a content list.
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if isinstance(content, (list, tuple)):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "tool_use":
                    found.append(block.get("name"))
                tool_use = block.get("toolUse")
                if isinstance(tool_use, dict):
                    found.append(tool_use.get("name"))
            elif getattr(block, "type", None) == "tool_use":
                found.append(getattr(block, "name", None))

    # Microsoft Agent Framework ``function_call`` contents.
    contents = getattr(msg, "contents", None)
    if contents is None and isinstance(msg, dict):
        contents = msg.get("contents")
    if isinstance(contents, (list, tuple)):
        for c in contents:
            if getattr(c, "type", None) == "function_call":
                found.append(getattr(c, "name", None))

    # Pydantic AI ``ModelMessage`` parts.
    parts = getattr(msg, "parts", None)
    if isinstance(parts, (list, tuple)):
        for part in parts:
            if getattr(part, "part_kind", None) in ("tool-call", "tool-return"):
                found.append(getattr(part, "tool_name", None))

    return [name for name in found if isinstance(name, str) and name]


def tool_names_used(messages: Iterable[Any] | None) -> list[str]:
    """Return the name of every tool invoked so far, first-seen order, once each.

    Feed the result to ``ConversationContext.tools_already_used`` (or the
    ``tools_already_used=`` keyword of ``GantryToolset.select``) so the
    router's already-used penalty nudges the next selection toward tools the
    agent has not tried yet. :class:`~agent_gantry.integrations.refresh.ToolRefresher`
    does this on every refresh.

    A call is recorded wherever the message format keeps it. Most formats put
    the name on the *assistant's call*, not on the result — an OpenAI ``tool``
    message carries only ``tool_call_id`` — so reading tool-role messages
    alone misses every tool the agent used:

    - ``tool``/``function``-role messages carrying ``name``/``tool_name``
      (LangChain ``ToolMessage``, the plain dict form).
    - OpenAI Chat Completions ``tool_calls`` on an assistant message.
    - OpenAI Responses API / Agents SDK ``function_call`` items.
    - Anthropic ``tool_use`` content blocks.
    - LangChain ``AIMessage.tool_calls``.
    - Microsoft Agent Framework ``function_call`` contents.
    - Pydantic AI ``tool-call`` / ``tool-return`` parts.
    - Strands ``toolUse`` content blocks.
    """
    names: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name and name not in seen:
            seen.add(name)
            names.append(name)

    for msg in messages or ():
        if _msg_role(msg) in ("tool", "function"):
            _add(_msg_name(msg))
        for name in _called_tool_names(msg):
            _add(name)
    return names


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


def latest_activity(
    messages: Iterable[Any] | None,
    *,
    max_chars: int = 500,
) -> str:
    """Return text from the single most recent *activity* message.

    This is the recency-aware generator: it drives retrieval from whatever just
    happened, regardless of role — so it serves **both** modes of agent
    operation without a fixed precedence:

    - **Conversational agents.** When the user has just spoken, the newest
      message is their request, so retrieval follows the new sub-task. (User
      pivots → tools pivot.)
    - **Autonomous agents / tool pipelines.** When the agent is chaining tools
      with no fresh user input, the newest message is the last tool's result,
      so the *next* tool is selected from what the previous one produced.

    It walks from the end of the conversation and returns the text of the first
    ``user`` or ``tool``/``function`` message it finds (skipping the empty
    tool-call stub ``assistant`` messages many frameworks emit). User text is
    returned verbatim; a tool result is returned as its raw content snippet
    (capped at ``max_chars``) so the *content* — not the tool's name — drives
    the next selection. Falls back to the latest assistant text, then empty.

    Args:
        messages: Conversation history (most recent message last).
        max_chars: Cap on characters taken from a tool result. Defaults to 500.

    Returns:
        The driving query text for the current turn, or the empty string.
    """
    if not messages:
        return ""
    for msg in reversed(list(messages)):
        role = _msg_role(msg)
        if role in ("user", ""):
            text = _msg_text(msg)
            if text.strip():
                return text.strip()
        elif role in ("tool", "function"):
            text = _msg_text(msg)
            if text.strip():
                return text.strip()[:max_chars]
    # Nothing concrete in the tail — fall back to the model's latest planning.
    return last_assistant_text(messages)


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
