"""
Query generation strategies for semantic tool retrieval.

When Agent-Gantry retrieves tools per chat round (via the
``query_strategy="per_call"`` mode of
:class:`agent_gantry.GantryContextProvider`), it needs a way to derive a
retrieval query from the *current* conversation state — not just the original
user message. This module ships the common deterministic strategies; an
opt-in :mod:`agent_gantry.query.llm` submodule adds LLM-based rewriters.

A "query generator" is any sync or async callable that takes the conversation
messages list and returns a string (empty string falls back to the conversation
tail handled by the caller). The signature is:

.. code-block:: python

    def my_generator(messages: list[Any]) -> str: ...
    async def my_generator(messages: list[Any]) -> str: ...

Built-in strategies:

- :func:`last_user_text` — most recent user-role message (default).
- :func:`last_assistant_text` — most recent assistant-role message
  (the model's latest planning).
- :func:`last_tool_result` — most recent tool/function-role message
  (best for fixed multi-step workflows where the next decision is driven by
  what the previous tool produced).
- :func:`concatenate_recent` — last N turns concatenated.
- :func:`fallback_chain` — try each generator in order until one returns
  non-empty text. Useful for "tool-result then assistant then user".

Example:

.. code-block:: python

    from agent_gantry.query import last_tool_result, fallback_chain, last_user_text

    provider = GantryContextProvider(
        gantry,
        query_strategy="per_call",
        query_generator=fallback_chain(last_tool_result, last_user_text),
    )
"""

from __future__ import annotations

from agent_gantry.query.strategies import (
    concatenate_recent,
    fallback_chain,
    keyword_focused,
    last_assistant_text,
    last_tool_result,
    last_user_text,
    latest_activity,
    truncated,
)

__all__ = [
    "concatenate_recent",
    "fallback_chain",
    "keyword_focused",
    "last_assistant_text",
    "last_tool_result",
    "last_user_text",
    "latest_activity",
    "truncated",
]
