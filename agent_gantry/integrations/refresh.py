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
Every ``<Framework>Adapter`` in :mod:`agent_gantry.integrations.frameworks` now
exposes a uniform :meth:`~agent_gantry.integrations.frameworks.base.BaseFrameworkAdapter.live`
method that delegates to that framework's *native* per-turn/per-call hook
(LangGraph middleware, a Pydantic AI ``AbstractToolset``, a Strands
``HookProvider``, …) — see ``integrations/frameworks/README.md`` for the full
table. Each of those hooks derives its retrieval query from the shape of
*that framework's own* state object (a LangGraph ``state["messages"]``, a
Pydantic AI ``RunContext``, an ADK ``callback_context``, …), because that is
what the framework hands the hook.

``ToolRefresher`` is deliberately **not** one of those hooks and is not called
by any adapter's ``live()``: it is the *standalone* utility for callers who are
not using one of the 14 supported frameworks at all — a hand-rolled agent loop
that owns its own message list and its own calls to an LLM SDK. If your
framework has a ``live()``, prefer it (it wires into the framework's actual
lifecycle instead of you calling ``refresh()`` by hand between turns). Both
sit on the same underlying selection primitive
(:class:`~agent_gantry.integrations.frameworks.base.GantryToolset`, plus its
:meth:`~agent_gantry.integrations.frameworks.base.GantryToolset.select_or_empty`
guard against selecting on an empty query), so behaviour — score thresholds,
namespace filtering, already-used-tool penalties — is consistent whichever
path you use.

Two behaviours make this genuinely multi-turn / direction-changing:

- **Query follows the conversation tail (recency-aware).** The default query
  generator is :func:`~agent_gantry.query.latest_activity`, which drives
  retrieval from whatever happened *most recently* — the newest user message
  **or** the newest tool result. This serves both modes with no config:
  autonomous agents chaining tools select the next tool from the previous
  tool's *result*; conversational agents pivot with each new user message. The
  two are not exclusive — a tool chain *inside* a conversational turn is driven
  by its results, then the next user message takes over.
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

from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    GantryToolset,
    ToolSpec,
)
from agent_gantry.query import latest_activity
from agent_gantry.query.strategies import _msg_role, _msg_text
from agent_gantry.schema.query import ConversationContext, ToolQuery

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def _default_query_generator() -> Callable[[Iterable[Any] | None], str]:
    """Recommended default: :func:`~agent_gantry.query.latest_activity`.

    Recency-aware, so it serves **both** autonomous and conversational agents
    with no configuration:

    - **Autonomous / tool pipelines:** when the newest message is a tool result
      (the agent is chaining tools with no new user input), the *content of
      that result* drives the next selection.
    - **Conversational:** when the newest message is the user's, their new
      request drives selection.

    To force a single behaviour, pass ``query_generator`` explicitly — e.g.
    ``last_user_text`` (always the user), ``last_tool_result`` (always the last
    result), or ``fallback_chain(...)`` for a custom precedence.
    """
    return latest_activity


# ``_msg_text``/``_msg_role`` live in agent_gantry.query.strategies; this module
# reuses them instead of keeping a second copy that drifts (the canonical pair
# also understands Responses-API ``input_text`` parts, Agent Framework
# ``contents``/``function_result`` blocks, and LangChain's ``.type`` role).


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

    **Standalone utility, not a framework hook.** If you're using one of the
    14 supported frameworks (see ``integrations/frameworks/README.md``),
    prefer ``<Framework>Adapter(gantry).live(...)`` instead — it wires Gantry
    into that framework's own per-turn/per-call lifecycle so re-selection
    happens automatically. Reach for ``ToolRefresher`` when you're driving a
    hand-rolled agent loop (a raw LLM SDK call in a ``while`` loop) with no
    framework underneath to hook into — see the module docstring for the
    full comparison.

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
            :func:`~agent_gantry.query.latest_activity` (recency-aware: the
            newest user message *or* tool result drives selection), which works
            for both autonomous tool pipelines and conversational agents. Pass
            ``last_user_text`` / ``last_tool_result`` / ``fallback_chain(...)``
            to force a specific behaviour. See :mod:`agent_gantry.query`.
        track_used: When ``True`` (default), scan the messages on every refresh
            for tool/function-role entries and accumulate their tool names into
            the ``tools_already_used`` set passed to selection. The router's
            ``already_used_penalty`` then nudges each turn toward *new* tools.
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
        """Tool names seen in the most recent refresh's message history.

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
            self._accumulate_used(msg_list)

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
            tools_already_used=(list(self._tools_used) if self._track_used else []),
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
            spec_from_tool(self._gantry, st.tool, st.semantic_score) for st in result.tools
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
            # The canonical ``_msg_text`` returns text unstripped; this
            # fallback has always yielded a stripped query.
            return _msg_text(messages[-1]).strip()
        return ""

    def _accumulate_used(self, messages: list[Any]) -> None:
        """Recompute the used-tool list *fresh* from the current message history.

        Recomputing (rather than appending) makes a single ``ToolRefresher``
        safe to reuse across conversations and robust to truncated/reset
        histories: ``tools_used`` always reflects exactly the tools present in
        the messages passed to *this* call, never leaking state from a prior
        run. Order is first-seen; each name appears once.
        """
        used: list[str] = []
        seen: set[str] = set()
        for msg in messages:
            if _msg_role(msg) in ("tool", "function"):
                name = _msg_tool_name(msg)
                if name and name not in seen:
                    seen.add(name)
                    used.append(name)
        self._tools_used = used


__all__ = ["ToolRefresher"]
