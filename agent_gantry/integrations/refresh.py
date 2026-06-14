"""Framework-agnostic multi-turn tool refresher.

Agent-Gantry's core value is surfacing a *small, relevant* slice of tools
instead of dumping the whole registry into the prompt. The Microsoft Agent
Framework provider already does this *per call* (``query_strategy="per_call"``
in :mod:`agent_gantry.integrations.agent_framework_provider`), re-selecting
tools on every chat-completion round so the agent's tool surface tracks where
its reasoning is heading rather than staying frozen on the first user message.

:class:`ToolRefresher` generalises that idea to *any* framework. It has no
dependency on Agent Framework (or LangChain, LlamaIndex, CrewAI, …) — it
operates purely on a list of conversation messages (dicts or message objects)
and a gantry instance. Call :meth:`ToolRefresher.refresh` (or
:meth:`refresh_specs`) once per turn with the conversation-so-far and it returns
a fresh top-k selection appropriate to the *latest* sub-task.

Two behaviours make this genuinely multi-turn / direction-changing:

- **Query follows the conversation tail.** The default query generator is
  ``fallback_chain(last_user_text, last_tool_result)``, so the next turn's
  retrieval is driven by the most recent user message (falling back to the
  latest tool result). When the user pivots (weather → email → currency), the
  surfaced tools pivot with them. For autonomous tool *pipelines* where the
  previous tool's output should drive the next selection, pass
  ``query_generator=fallback_chain(last_tool_result, last_user_text)``.
- **Used tools nudge the agent forward.** When ``track_used`` is on, every
  tool/function-role message seen in the history accumulates into a
  ``tools_already_used`` set. The router applies its ``already_used_penalty``
  to those names, gently steering each turn toward *new* tools instead of
  re-suggesting ones the agent already invoked.

Typical usage in a hand-rolled agent loop::

    refresher = ToolRefresher(gantry, limit=3, dialect="openai")
    messages = [{"role": "user", "content": "what's the weather in Paris?"}]
    while not done:
        tools = await refresher.refresh(messages)   # dialect schemas for this turn
        response = await my_llm(messages, tools=tools)
        messages.append(...)                         # assistant / tool messages
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, ToolSpec
from agent_gantry.query import fallback_chain, last_tool_result, last_user_text
from agent_gantry.schema.query import ConversationContext, ToolQuery

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def _default_query_generator() -> Callable[[Iterable[Any] | None], str]:
    """Recommended default: latest user text, falling back to tool result.

    Built fresh on each construction so the composed callable is never a
    shared global. User text first makes the *conversational* case (the user
    pivots to a new sub-task each turn) intuitive: the surfaced tools follow
    what the user just asked for. For autonomous tool *pipelines* — where the
    previous tool's output should pick the next tool — pass
    ``query_generator=fallback_chain(last_tool_result, last_user_text)``
    explicitly (this is what the Agent Framework provider's ``per_call`` mode
    uses).
    """
    return fallback_chain(last_user_text, last_tool_result)


def _msg_text(msg: Any) -> str:
    """Best-effort plain text from a message dict or object (last-resort)."""
    text = getattr(msg, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(msg, dict):
        for key in ("text", "content"):
            value = msg.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _msg_role(msg: Any) -> str:
    """Lowercased role of a message dict or object."""
    role = getattr(msg, "role", None)
    if role is None and isinstance(msg, dict):
        role = msg.get("role")
    return str(role).lower() if role is not None else ""


def _msg_tool_name(msg: Any) -> str:
    """Best-effort tool/function name for a tool-role message."""
    for attr in ("name", "tool_name", "author_name"):
        value = getattr(msg, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(msg, dict):
        for key in ("name", "tool_name", "author_name"):
            value = msg.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` when it is awaitable, otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


class ToolRefresher:
    """Re-select tools every turn of a single agent run, for any framework.

    The refresher keeps no per-turn conversation state of its own beyond the
    accumulated set of used tool names (when ``track_used`` is enabled) and the
    most recent selection. Each call to :meth:`refresh` / :meth:`refresh_specs`
    recomputes the selection *fresh* from the messages you pass, so it is safe
    to drive from any loop that appends to a growing message list.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` providing
            semantic retrieval.
        limit: Maximum number of tools to surface per turn. Defaults to ``3``.
        dialect: Provider dialect for :meth:`refresh`'s schema output
            (``"openai"``, ``"anthropic"``, …). Defaults to ``"openai"``.
        score_threshold: Minimum semantic relevance score for retained tools.
            Defaults to ``0.0`` (no filtering) — long queries dilute absolute
            similarities, so a non-zero default silently drops relevant tools.
        query_generator: Sync or async callable mapping the messages list to a
            retrieval query string. Defaults to
            ``fallback_chain(last_user_text, last_tool_result)`` so retrieval is
            driven by the latest user message (falling back to the latest tool
            result) — the intuitive choice for conversational pivots. For tool
            pipelines, pass ``fallback_chain(last_tool_result, last_user_text)``.
            See :mod:`agent_gantry.query` for built-in alternatives.
        track_used: When ``True`` (default), scan the messages on every refresh
            for tool/function-role entries and accumulate their tool names into
            the ``tools_already_used`` set passed to selection. The router's
            ``already_used_penalty`` then nudges each turn toward *new* tools.
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        limit: int = 3,
        dialect: str = "openai",
        score_threshold: float = 0.0,
        query_generator: Callable[[Iterable[Any] | None], Any] | None = None,
        track_used: bool = True,
    ) -> None:
        self._gantry = gantry
        self._toolset = GantryToolset(gantry, default_limit=limit)
        self._limit = limit
        self._dialect = dialect
        self._score_threshold = score_threshold
        self._query_generator = query_generator or _default_query_generator()
        self._track_used = track_used
        self._tools_used: list[str] = []
        self._last_selection: list[ToolSpec] = []

    # -- public accessors -------------------------------------------------- #
    @property
    def last_selection(self) -> list[ToolSpec]:
        """The :class:`ToolSpec` list produced by the most recent refresh."""
        return list(self._last_selection)

    @property
    def tools_used(self) -> list[str]:
        """Tool names accumulated across turns (when ``track_used``).

        Returns an empty list when ``track_used`` is disabled. The order is
        first-seen; each name appears once.
        """
        return list(self._tools_used)

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def dialect(self) -> str:
        return self._dialect

    # -- core refresh ------------------------------------------------------ #
    async def refresh_specs(self, messages: Iterable[Any] | None) -> list[ToolSpec]:
        """Re-select tools for the current turn and return ranked specs.

        Derives the retrieval query from ``messages`` via the configured
        generator (falling back to the last message's text when the generator
        yields an empty string), honours the accumulated ``tools_already_used``
        set when ``track_used`` is on, runs a fresh selection, stores it on
        :attr:`last_selection`, and returns the ranked :class:`ToolSpec` list.

        Args:
            messages: The conversation so far (most recent message last).

        Returns:
            Ranked :class:`ToolSpec` handles for this turn (at most ``limit``).
        """
        msg_list = list(messages) if messages else []

        if self._track_used:
            self._accumulate_used(msg_list)

        query = await self._build_query(msg_list)
        if not query:
            # Nothing to retrieve against; nothing to surface this turn.
            self._last_selection = []
            return []

        specs = await self._toolset.select(
            query,
            limit=self._limit,
            score_threshold=self._score_threshold,
            tools_already_used=self._tools_used if self._track_used else None,
        )
        self._last_selection = specs
        return specs

    async def refresh(self, messages: Iterable[Any] | None) -> list[dict[str, Any]]:
        """Re-select tools for the current turn and return dialect schemas.

        Same selection path as :meth:`refresh_specs` (so ``tools_already_used``
        is honoured), then converts each selected tool to ``self.dialect``
        schema via the underlying :class:`~agent_gantry.schema.tool.ToolDefinition`.

        Args:
            messages: The conversation so far (most recent message last).

        Returns:
            A list of provider-specific tool-schema dicts for this turn.
        """
        msg_list = list(messages) if messages else []

        if self._track_used:
            self._accumulate_used(msg_list)

        query = await self._build_query(msg_list)
        if not query:
            self._last_selection = []
            return []

        # Build the ToolQuery ourselves so tools_already_used (and thus the
        # router's already_used_penalty) is honoured, then transcode each
        # ToolDefinition to the requested dialect. We also reconstruct the
        # ToolSpec list so last_selection / refresh_specs stay consistent.
        context = ConversationContext(
            query=query,
            tools_already_used=(
                list(self._tools_used) if self._track_used else []
            ),
        )
        result = await self._gantry.retrieve(
            ToolQuery(
                context=context,
                limit=self._limit,
                score_threshold=self._score_threshold,
            )
        )

        from agent_gantry.integrations.frameworks.base import spec_from_tool

        self._last_selection = [
            spec_from_tool(self._gantry, st.tool, st.semantic_score)
            for st in result.tools
        ]
        return [st.tool.to_dialect(self._dialect) for st in result.tools]

    # -- internals --------------------------------------------------------- #
    async def _build_query(self, messages: list[Any]) -> str:
        """Run the query generator, falling back to the last message's text."""
        result = await _maybe_await(self._query_generator(messages))
        query = (result or "").strip()
        if query:
            return query
        if messages:
            return _msg_text(messages[-1])
        return ""

    def _accumulate_used(self, messages: list[Any]) -> None:
        """Add tool names from tool/function-role messages to the used set."""
        seen = set(self._tools_used)
        for msg in messages:
            if _msg_role(msg) in ("tool", "function"):
                name = _msg_tool_name(msg)
                if name and name not in seen:
                    seen.add(name)
                    self._tools_used.append(name)


__all__ = ["ToolRefresher"]
