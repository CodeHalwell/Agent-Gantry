"""Per-turn dynamic-tool hook for AWS Strands Agents (the *deep* integration).

Where :mod:`agent_gantry.integrations.frameworks.strands` exposes the *static*
path — ``_for_strands`` selects a fixed slice of tools once and you hand them
to ``Agent(tools=[...])`` — this module wires Agent-Gantry into Strands' native
per-turn hook so the tool surface is re-selected **before every model call**.

Strands fires a ``BeforeModelCallEvent`` immediately before each model
invocation, and only reads ``agent.tool_registry.get_all_tool_specs()``
*afterward* (``strands.event_loop.event_loop.stream_messages`` invokes the
``BeforeModelCallEvent`` hooks, then — several lines later in the same
iteration — calls ``agent.tool_registry.get_all_tool_specs()`` to build the
request sent to the model). A hook callback that mutates the registry during
``BeforeModelCallEvent`` therefore changes what the model sees on that very
call — genuine per-turn re-selection, matching the depth of Google ADK's
``before_model_callback`` (see ``google_adk_live``), and deeper than the
per-top-level-call rebuild used for CrewAI/Agno/Haystack/Smolagents (see
``live_wrappers``), which fix their tool list at agent construction.

:class:`GantryStrandsToolHook` implements Strands' ``HookProvider`` protocol
structurally — ``HookProvider`` is ``@runtime_checkable``, so a plain
``register_hooks(self, registry)`` method is sufficient, no subclassing
required. The hook instance is retained across turns so it can diff each
turn's newly-selected tool names against the previous turn's and *retract*
tools that fell out of the top-k, not just add new ones.

.. code-block:: python

    from agent_gantry import AgentGantry
    from agent_gantry.integrations.frameworks.strands_live import (
        _gantry_strands_agent,
    )

    gantry = AgentGantry()
    # ... register tools, await gantry.sync() ...

    agent = _gantry_strands_agent(gantry, limit=5)

Or attach the hook to a hand-built agent:

.. code-block:: python

    from strands import Agent
    from agent_gantry.integrations.frameworks.strands_live import (
        GantryStrandsToolHook,
    )

    agent = Agent(tools=[], hooks=[GantryStrandsToolHook(gantry, limit=5)])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.strands import _for_strands

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Strands Agents support requires `strands-agents`. "
    "Install it with `pip install strands-agents`."
)


def _import_strands_agent() -> Any:
    """Lazily import the ``strands.Agent`` class with a helpful error.

    Deferred so ``import agent_gantry`` (and importing this module) never
    requires ``strands-agents``.
    """
    try:
        from strands import Agent
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(_INSTALL_HINT) from exc
    return Agent


def _content_text(content: Any) -> str:
    """Extract and join the ``text`` fields of a Strands content-block list."""
    chunks: list[str] = []
    for block in content or []:
        if isinstance(block, dict):
            text = block.get("text")
            if text:
                chunks.append(text)
    return " ".join(chunks).strip()


def _query_from_messages(messages: Any) -> str:
    """Derive this turn's retrieval query from the agent's message history.

    Walks the recent tail of ``agent.messages`` (a list of Strands
    ``{"role": ..., "content": [...]}`` dicts) backwards for the latest
    user-role message's text content — the per-turn signal that makes
    re-selection actually adapt across turns. Only scans the tail so a long
    conversation doesn't pay an O(n) scan before every model call.
    """
    for message in reversed(list(messages or [])[-20:]):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _content_text(message.get("content"))
        if text:
            return text
    return ""


class GantryStrandsToolHook:
    """``HookProvider`` that re-selects Gantry tools before every model call.

    Registers a ``BeforeModelCallEvent`` callback that: derives this turn's
    query from the agent's latest user message, re-runs Gantry's semantic
    router, converts the fresh slice to ``DecoratedFunctionTool``s, and swaps
    them into ``agent.tool_registry`` — registering newly-selected tools and
    retracting ones that fell out of the top-k since the previous turn.
    Because Strands reads the registry only *after* ``BeforeModelCallEvent``
    fires, the swap takes effect for the very model call about to happen.

    Obtain one via ``StrandsAdapter(gantry).tool_hook(...)``.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` to select from.
        limit: Max tools to surface per turn. Defaults to ``5``.
        score_threshold: Minimum semantic relevance score. Defaults to ``0.0``.
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> None:
        self._gantry = gantry
        self._limit = limit
        self._score_threshold = score_threshold
        self._active_names: set[str] = set()

    def register_hooks(self, registry: Any, **kwargs: Any) -> None:
        """Register :meth:`_on_before_model_call` for ``BeforeModelCallEvent``.

        Implements the ``HookProvider`` protocol (structurally — Strands'
        ``HookProvider`` is ``@runtime_checkable``). Called automatically when
        this hook is passed to ``Agent(hooks=[...])`` or ``agent.add_hook(hook)``.
        The event type is passed explicitly, so no callback type-hint inference
        is required.
        """
        from strands.hooks import BeforeModelCallEvent

        registry.add_callback(BeforeModelCallEvent, self._on_before_model_call)

    async def _on_before_model_call(self, event: Any) -> None:
        """Re-select and swap this turn's tools into ``event.agent.tool_registry``.

        Never raises on selection failure — a broken retrieval must not break
        the agent's model call; the previous turn's tools (if any) are left in
        place when that happens.
        """
        query = _query_from_messages(getattr(event.agent, "messages", None))
        if not query:
            return
        try:
            tools = await _for_strands(
                self._gantry,
                query,
                limit=self._limit,
                score_threshold=self._score_threshold,
            )
        except Exception:
            logger.exception(
                "GantryStrandsToolHook: semantic retrieval failed; "
                "continuing with the previous turn's tools."
            )
            return

        tool_registry = event.agent.tool_registry
        new_names = {t.tool_name for t in tools}

        # Retract tools that were selected on a previous turn but not this one.
        for stale_name in self._active_names - new_names:
            tool_registry.registry.pop(stale_name, None)
            tool_registry.dynamic_tools.pop(stale_name, None)

        # Register (or hot-reload-replace) this turn's tools.
        for native_tool in tools:
            native_tool.mark_dynamic()
            tool_registry.register_tool(native_tool)

        self._active_names = new_names


def _gantry_strands_agent(
    gantry: AgentGantry,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    **agent_kwargs: Any,
) -> Any:
    """Build a ``strands.Agent`` wired for per-turn dynamic tool selection.

    Constructs ``Agent(tools=[], hooks=[GantryStrandsToolHook(...)],
    **agent_kwargs)``. The agent ships with *no* statically registered tools —
    the hook injects the relevant slice before every model call (see
    :class:`GantryStrandsToolHook`).

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` instance.
        limit: Maximum number of tools selected/injected per turn.
        score_threshold: Minimum semantic relevance score for selection.
        **agent_kwargs: Extra keyword arguments forwarded to ``Agent(...)``
            (``model``, ``system_prompt``, ``callback_handler``, ...).

    Returns:
        A configured ``strands.Agent``.

    Raises:
        ImportError: If ``strands-agents`` is not installed.
    """
    agent_cls = _import_strands_agent()
    hook = GantryStrandsToolHook(gantry, limit=limit, score_threshold=score_threshold)
    return agent_cls(tools=[], hooks=[hook], **agent_kwargs)


__all__ = ["GantryStrandsToolHook"]
