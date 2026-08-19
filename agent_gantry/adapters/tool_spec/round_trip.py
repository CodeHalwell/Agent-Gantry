"""Turn a provider response into tool calls, and stream deltas into the same.

Every dialect adapter already implements ``from_provider_payload`` /
``to_tool_call`` / ``format_tool_result``, but each of those takes *one*
already-extracted payload. Nothing bridged the gap between "the SDK handed me
a response object" and "here are the calls to run", so every caller hand-rolled
``json.loads(tc.function.arguments)`` and the parallel-call case was left as an
exercise. These helpers close that gap:

- :func:`extract_tool_calls` pulls every tool call out of a whole response.
- :class:`StreamingToolCallAccumulator` reassembles calls from streamed
  fragments, which no part of the library previously handled at all.

Both accept SDK objects and the equivalent plain dicts, since the SDKs are
optional dependencies and responses are frequently replayed from fixtures.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_gantry.adapters.tool_spec.base import ToolCallPayload

__all__ = [
    "StreamingToolCallAccumulator",
    "extract_tool_calls",
]

logger = logging.getLogger(__name__)

#: Dialects that share OpenAI's Chat Completions response shape.
_OPENAI_CHAT_DIALECTS = frozenset({"openai", "groq", "mistral", "agent_framework"})


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping or an attribute holder."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_payload_dict(obj: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    """Project ``keys`` off an SDK object into the dict the adapters expect."""
    if isinstance(obj, dict):
        return obj
    return {key: _get(obj, key) for key in keys}


def _openai_chat_calls(response: Any) -> list[dict[str, Any]]:
    choices = _get(response, "choices") or []
    if not choices:
        return []
    message = _get(choices[0], "message")
    raw_calls = _get(message, "tool_calls") or [] if message is not None else []

    calls: list[dict[str, Any]] = []
    for call in raw_calls:
        function = _get(call, "function")
        calls.append(
            {
                "id": _get(call, "id"),
                "function": {
                    "name": _get(function, "name", ""),
                    "arguments": _get(function, "arguments", "{}"),
                },
            }
        )
    return calls


def _openai_responses_calls(response: Any) -> list[dict[str, Any]]:
    output = _get(response, "output") or []
    return [
        _as_payload_dict(item, ("call_id", "name", "arguments"))
        for item in output
        if _get(item, "type") == "function_call"
    ]


def _anthropic_calls(response: Any) -> list[dict[str, Any]]:
    content = _get(response, "content") or []
    return [
        _as_payload_dict(block, ("id", "name", "input"))
        for block in content
        if _get(block, "type") == "tool_use"
    ]


def _gemini_calls(response: Any) -> list[dict[str, Any]]:
    candidates = _get(response, "candidates") or []
    if not candidates:
        return []
    content = _get(candidates[0], "content")
    parts = _get(content, "parts") or [] if content is not None else []

    calls: list[dict[str, Any]] = []
    for part in parts:
        function_call = _get(part, "function_call") or _get(part, "functionCall")
        if function_call is None:
            continue
        calls.append(_as_payload_dict(function_call, ("name", "args", "id")))
    return calls


def extract_tool_calls(response: Any, dialect: str = "openai") -> list[ToolCallPayload]:
    """Pull every tool call out of a provider response.

    Handles the parallel-call case, which is the normal case for current
    models and the one a hand-rolled ``response.choices[0].message.tool_calls[0]``
    silently truncates.

    Args:
        response: The provider's response object, or an equivalent dict.
        dialect: Which response shape to read. Defaults to ``"openai"``.

    Returns:
        One :class:`ToolCallPayload` per tool call, in the order the model
        emitted them. Empty when the model called no tools.

    Raises:
        ValueError: If ``dialect`` has no known response shape.
    """
    from agent_gantry.adapters.tool_spec.registry import get_adapter

    if dialect in _OPENAI_CHAT_DIALECTS:
        raw_calls = _openai_chat_calls(response)
    elif dialect == "openai_responses":
        raw_calls = _openai_responses_calls(response)
    elif dialect == "anthropic":
        raw_calls = _anthropic_calls(response)
    elif dialect == "gemini":
        raw_calls = _gemini_calls(response)
    else:
        raise ValueError(
            f"No known response shape for dialect {dialect!r}. "
            f"Supported: {', '.join(sorted({*_OPENAI_CHAT_DIALECTS, 'openai_responses', 'anthropic', 'gemini'}))}"
        )

    adapter = get_adapter(dialect)
    return [adapter.from_provider_payload(call) for call in raw_calls]


class StreamingToolCallAccumulator:
    """Reassemble tool calls from a streamed response.

    Providers split a tool call across many chunks: OpenAI sends
    ``delta.tool_calls`` entries whose ``arguments`` are string fragments to be
    concatenated per ``index``, and Anthropic sends a ``content_block_start``
    naming the tool followed by ``input_json_delta`` fragments. Nothing in the
    library reassembled either, so streaming callers — the norm in production
    chat — got no help from the round-trip layer at all.

    Feed every chunk to :meth:`add_chunk`, then read :meth:`tool_calls`.

    Example:
        >>> acc = StreamingToolCallAccumulator(dialect="openai")
        >>> async for chunk in stream:  # doctest: +SKIP
        ...     acc.add_chunk(chunk)
        >>> calls = acc.tool_calls()  # doctest: +SKIP
    """

    def __init__(self, dialect: str = "openai") -> None:
        known = {*_OPENAI_CHAT_DIALECTS, "openai_responses", "anthropic"}
        if dialect not in known:
            raise ValueError(
                f"Streaming accumulation is not implemented for dialect {dialect!r}. "
                f"Supported: {', '.join(sorted(known))}"
            )
        self._dialect = dialect
        # Keyed by the provider's own ordinal so out-of-order chunks still land
        # in the right call; insertion order preserves emission order.
        self._calls: dict[Any, dict[str, Any]] = {}

    def add_chunk(self, chunk: Any) -> None:
        """Fold one streamed chunk into the accumulated state."""
        if self._dialect == "anthropic":
            self._add_anthropic_event(chunk)
        elif self._dialect == "openai_responses":
            self._add_responses_event(chunk)
        else:
            self._add_openai_chunk(chunk)

    def _slot(self, key: Any) -> dict[str, Any]:
        return self._calls.setdefault(key, {"id": None, "name": "", "arguments": ""})

    def _add_openai_chunk(self, chunk: Any) -> None:
        choices = _get(chunk, "choices") or []
        if not choices:
            return
        delta = _get(choices[0], "delta")
        if delta is None:
            return
        for entry in _get(delta, "tool_calls") or []:
            # ``index`` ties fragments of the same call together; it is absent
            # only on providers that never parallelize, so fall back to 0.
            index = _get(entry, "index", 0) or 0
            slot = self._slot(index)
            if (call_id := _get(entry, "id")) is not None:
                slot["id"] = call_id
            function = _get(entry, "function")
            if function is not None:
                if name := _get(function, "name"):
                    slot["name"] = name
                fragment = _get(function, "arguments")
                if fragment:
                    slot["arguments"] += fragment

    def _add_responses_event(self, event: Any) -> None:
        event_type = _get(event, "type") or ""
        item = _get(event, "item")
        if item is not None and _get(item, "type") == "function_call":
            slot = self._slot(_get(event, "output_index", len(self._calls)))
            slot["id"] = _get(item, "call_id") or slot["id"]
            slot["name"] = _get(item, "name") or slot["name"]
        elif "function_call_arguments.delta" in event_type:
            slot = self._slot(_get(event, "output_index", 0) or 0)
            if fragment := _get(event, "delta"):
                slot["arguments"] += fragment

    def _add_anthropic_event(self, event: Any) -> None:
        event_type = _get(event, "type") or ""
        index = _get(event, "index", 0) or 0

        if event_type == "content_block_start":
            block = _get(event, "content_block")
            if block is not None and _get(block, "type") == "tool_use":
                slot = self._slot(index)
                slot["id"] = _get(block, "id")
                slot["name"] = _get(block, "name", "")
        elif event_type == "content_block_delta":
            delta = _get(event, "delta")
            if delta is not None and _get(delta, "type") == "input_json_delta":
                if index in self._calls and (fragment := _get(delta, "partial_json")):
                    self._calls[index]["arguments"] += fragment

    def tool_calls(self) -> list[ToolCallPayload]:
        """Return the calls accumulated so far, in emission order.

        A call whose arguments never parsed yields empty arguments rather than
        raising: a truncated stream should surface as a tool failing on missing
        input, not as an exception swallowing the rest of the turn.
        """
        payloads: list[ToolCallPayload] = []
        for slot in self._calls.values():
            if not slot["name"]:
                continue
            raw = slot["arguments"].strip()
            arguments: dict[str, Any] = {}
            if raw:
                try:
                    parsed = json.loads(raw)
                    arguments = parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    logger.warning(
                        "Incomplete or malformed streamed arguments for '%s'; "
                        "treating as empty: %s",
                        slot["name"],
                        raw[:200],
                    )
            payloads.append(
                ToolCallPayload(
                    tool_name=slot["name"],
                    tool_call_id=slot["id"],
                    arguments=arguments,
                    raw_payload={"streamed": True, **slot},
                )
            )
        return payloads

    def reset(self) -> None:
        """Drop accumulated state so the instance can serve the next turn."""
        self._calls.clear()
