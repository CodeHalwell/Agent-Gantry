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

Where this sits next to ``adapter.live()``
-------------------------------------------
Every ``<Framework>Adapter`` in :mod:`agent_gantry.integrations.frameworks`
exposes a uniform :meth:`~agent_gantry.integrations.frameworks.base.BaseFrameworkAdapter.live`
method that delegates to that framework's *native* per-turn/per-call hook
(LangGraph middleware, a Pydantic AI ``AbstractToolset``, a Strands
``HookProvider``, …) — see ``integrations/frameworks/README.md`` for the full
table. Each of those hooks derives its retrieval query from that framework's
own state object.

``ToolRefresher`` is deliberately **not** one of those hooks: it is the
*standalone* utility for callers who are not using one of the supported
frameworks at all — a hand-rolled agent loop that owns its own message list
and its own calls to an LLM SDK. If your framework has a ``live()``, prefer
it. Both sit on the same underlying selection primitive
(:class:`~agent_gantry.integrations.frameworks.base.GantryToolset`, plus its
:meth:`~agent_gantry.integrations.frameworks.base.GantryToolset.select_or_empty`
guard against selecting on an empty query), so score thresholds, namespace
filtering and pinned tools behave the same whichever path you use. The
already-used penalty below is ``ToolRefresher``'s own addition: it is the
path that hands the router ``tools_already_used``; the framework hooks
select on the turn's query alone.

Two behaviours make this genuinely multi-turn / direction-changing:

- **Query follows the conversation tail (recency-aware).** The default query
  generator is :func:`~agent_gantry.query.latest_activity`, which drives
  retrieval from whatever happened *most recently* — the newest user message
  **or** the newest tool result. Autonomous agents chaining tools select the
  next tool from the previous tool's *result*; conversational agents pivot
  with each new user message.
- **Used tools nudge the agent forward.** When ``track_used`` is on, every
  tool call the history shows — read with
  :func:`~agent_gantry.query.tool_names_used`, so an OpenAI ``tool_calls``
  entry, an Anthropic ``tool_use`` block, a LangChain ``ToolMessage`` and a
  plain ``{"role": "tool", "name": ...}`` dict all count — accumulates into a
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

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    GantryToolset,
    ToolSpec,
    _maybe_await,
    check_query_bounds,
)
from agent_gantry.query import latest_activity, tool_names_used
from agent_gantry.query.strategies import _msg_text

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


class ToolRefresher:
    """Re-select tools every turn of a single agent run, for any framework.

    **Standalone utility, not a framework hook.** If you're using one of the
    supported frameworks (see ``integrations/frameworks/README.md``), prefer
    ``<Framework>Adapter(gantry).live(...)`` instead — it wires Gantry into
    that framework's own per-turn/per-call lifecycle so re-selection happens
    automatically. Reach for ``ToolRefresher`` when you're driving a
    hand-rolled agent loop (a raw LLM SDK call in a ``while`` loop) with no
    framework underneath to hook into.

    The refresher keeps no per-turn conversation state of its own beyond the
    accumulated set of used tool names (when ``track_used`` is enabled) and the
    most recent selection. Each call to :meth:`refresh` / :meth:`refresh_specs`
    recomputes the selection *fresh* from the messages you pass, so it is safe
    to drive from any loop that appends to a growing message list.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` providing
            semantic retrieval.
        limit: Maximum number of tools to surface per turn. Defaults to
            ``DEFAULT_TOOL_LIMIT`` (5).
        dialect: Provider dialect for :meth:`refresh`'s schema output
            (``"openai"``, ``"anthropic"``, …). Defaults to ``"openai"``.
        score_threshold: Minimum semantic relevance score for retained tools.
            Defaults to ``0.0`` (no filtering) — long queries dilute absolute
            similarities, so a non-zero default silently drops relevant tools.
        query_generator: Sync or async callable mapping the messages list to a
            retrieval query string. Defaults to
            :func:`~agent_gantry.query.latest_activity` (recency-aware: the
            newest user message *or* tool result drives selection). Pass
            ``last_user_text`` / ``last_tool_result`` / ``fallback_chain(...)``
            to force a specific behaviour. See :mod:`agent_gantry.query`.
        track_used: When ``True`` (default), scan the messages on every refresh
            for tool calls — via :func:`~agent_gantry.query.tool_names_used` —
            and pass their names as ``tools_already_used`` to selection. The
            router's ``already_used_penalty`` then nudges each turn toward
            *new* tools.
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        limit: int = DEFAULT_TOOL_LIMIT,
        dialect: str = "openai",
        score_threshold: float = 0.0,
        query_generator: Callable[[Iterable[Any] | None], Any] | None = None,
        track_used: bool = True,
    ) -> None:
        check_query_bounds(limit=limit, score_threshold=score_threshold, owner="ToolRefresher")
        self._toolset = GantryToolset(gantry)
        self._limit = limit
        self._dialect = dialect
        self._score_threshold = score_threshold
        self._query_generator = query_generator or latest_activity
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
        """Tool names the most recent refresh's message history shows being called.

        Recomputed fresh on every refresh (no cross-conversation leakage).
        Returns an empty list when ``track_used`` is disabled or before the
        first refresh. Order is first-seen; each name appears once.
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
            self._tools_used = tool_names_used(msg_list)

        query = await self._build_query(msg_list)
        # select_or_empty: nothing to retrieve against means nothing to
        # surface this turn, same empty-query guard every live per-framework
        # provider uses (see GantryToolset.select_or_empty).
        specs = await self._toolset.select_or_empty(
            query,
            limit=self._limit,
            score_threshold=self._score_threshold,
            tools_already_used=self._tools_used if self._track_used else None,
        )
        self._last_selection = specs
        return specs

    async def refresh(self, messages: Iterable[Any] | None) -> list[dict[str, Any]]:
        """Re-select tools for the current turn and return dialect schemas.

        Same selection as :meth:`refresh_specs`, with each selected tool
        transcoded to ``self.dialect`` schema.

        Args:
            messages: The conversation so far (most recent message last).

        Returns:
            A list of provider-specific tool-schema dicts for this turn.
        """
        specs = await self.refresh_specs(messages)
        return [spec.tool.to_dialect(self._dialect) for spec in specs if spec.tool is not None]

    # -- internals --------------------------------------------------------- #
    async def _build_query(self, messages: list[Any]) -> str:
        """Run the query generator, falling back to the last message's text."""
        result = await _maybe_await(self._query_generator(messages))
        query = (result or "").strip()
        if query:
            return query
        if messages:
            return _msg_text(messages[-1]).strip()
        return ""


__all__ = ["ToolRefresher"]
