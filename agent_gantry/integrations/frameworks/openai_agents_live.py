"""DEEP per-turn dynamic-tool provider for the OpenAI Agents SDK.

Where :mod:`agent_gantry.integrations.frameworks.openai_agents` exposes the
*static* slice (:class:`OpenAIAgentsAdapter.select` /
:meth:`OpenAIAgentsAdapter.convert`) — you select a slice of tools once and hand
the resulting ``FunctionTool`` list to an :class:`agents.Agent` — this module
wires Agent-Gantry into the SDK as a
**live** tool source so the agent's tool set is **re-selected by Gantry as the
conversation progresses**, matching the depth of the Microsoft Agent Framework
``GantryContextProvider`` as closely as this SDK allows.

How deep can it go on this SDK?
-------------------------------
The OpenAI Agents SDK reads the agent's tool surface from ``agent.tools`` afresh
**on every turn**: the run loop calls ``Agent.get_all_tools`` once per turn to
snapshot the callable functions before invoking the model. There is no
documented "return the tools for this turn" callback, but there are lifecycle
hooks on ``RunHooks``: ``on_agent_start`` fires as the agent starts and
``on_llm_start`` immediately before each model request. Because ``agent.tools``
is a plain mutable list, two complementary mechanisms give genuine per-turn
re-selection:

1. **Mid-run (hook-driven), via** :func:`_gantry_run_hooks`. The returned
   :class:`agents.RunHooks` rewrites ``agent.tools`` in place at two points.
   ``on_agent_start`` selects from the run's input; from ``openai-agents``
   0.23.1 the SDK fires it *before* the turn's snapshot, so the **first** model
   request already carries the selection. ``on_llm_start`` re-selects from the
   running conversation; it fires *within* a turn, after the snapshot, so its
   rewrite is picked up by the **next** turn — within a single multi-turn
   ``Runner.run`` the tool set tracks the evolving conversation turn by turn,
   with no application code between turns. This is the deepest the SDK natively
   supports. (Before 0.23.1 the snapshot precedes every hook, so the first
   request uses ``agent.tools`` as the caller left it; mechanism 2 covers that.)

2. **Per-run (loop-driven), via** :class:`GantryAgentSession` /
   :func:`_run_with_gantry`. Re-selects tools from the run input and updates
   ``agent.tools`` **before** each ``Runner.run`` call, so the new selection is
   in force on the very *first* turn of that run on every SDK version. Driving
   a multi-turn conversation as a sequence of runs (the common chat pattern)
   yields immediate, query-accurate re-selection each turn and composes with
   the hook for intra-run dynamism.

Both update ``agent.tools`` in place (an ``agent.clone(tools=...)`` variant is
also offered) and route every tool call back through ``gantry.execute`` via the
:class:`~agents.FunctionTool` objects built by ``_spec_to_openai_agents`` — so
retries, timeouts, circuit breakers, and the security policy still apply.

The ``agents`` import is lazy (only inside the builders / hook), so ``import
agent_gantry`` never requires the OpenAI Agents SDK to be installed; the helpful
``pip install openai-agents`` (or ``uv add openai-agents``) hint is raised only when the live provider is used.

Usage
-----
.. code-block:: python

    from agents import Agent, Runner
    from agent_gantry import AgentGantry
    from agent_gantry.openai_agents import OpenAIAgentsAdapter

    adapter = OpenAIAgentsAdapter(gantry)
    agent = Agent(name="assistant", tools=[])

    # Per-run (immediate first-turn re-selection) — primary entry point:
    result = await adapter.run(agent, "what's the weather in Paris?")

    # ...or hold a session across a multi-turn chat:
    session = adapter.session(agent, limit=5)
    r1 = await session.run("what's the weather in Paris?")   # weather tools
    r2 = await session.run("now email that to my boss")      # email tools

    # ...or hand the hook to a single Runner.run for intra-run dynamism:
    await Runner.run(agent, input, hooks=adapter.run_hooks(agent))
"""

from __future__ import annotations

import asyncio
import logging
import warnings
import weakref
from typing import TYPE_CHECKING, Any

from agent_gantry.adapters.selectors.base import SelectorFallbackError
from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    GantryToolset,
    QueryBoundsError,
    ToolSpec,
    _resolve_pins,
    _with_native_names,
    check_query_bounds,
    spec_from_tool,
)
from agent_gantry.integrations.frameworks.errors import MissingRequiredToolError
from agent_gantry.integrations.frameworks.native import (
    MAX_TOOL_RESULT_QUERY_CHARS,
    ErrorFormatter,
    ErrorMode,
    carry_over,
    check_error_mode,
)
from agent_gantry.integrations.frameworks.openai_agents import (
    _INSTALL_HINT,
    _first_request_sees_input,
    _require_per_turn_sdk,
    _spec_to_openai_agents,
)
from agent_gantry.query import latest_activity

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

logger = logging.getLogger(__name__)

_PIP_HINT = _INSTALL_HINT


def _require_agents() -> Any:
    """Import the ``agents`` package lazily, with a helpful error if missing.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    try:
        import agents
    except ImportError as exc:  # pragma: no cover - exercised via importorskip
        raise ImportError(_PIP_HINT) from exc
    return agents


def _query_from(query_or_input: Any) -> str:
    """Coerce a query string or a run input into a retrieval query.

    A plain ``str`` is used verbatim (the Agents SDK accepts a bare string as
    ``Runner.run`` input). Anything else is treated as a list of input items /
    messages and run through :func:`~agent_gantry.query.latest_activity`, which
    derives the driving text from the most recent activity message.
    """
    if query_or_input is None:
        return ""
    if isinstance(query_or_input, str):
        return query_or_input
    return latest_activity(query_or_input) or ""


async def _select_function_tools(
    gantry: AgentGantry,
    query_or_input: Any,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> list[Any]:
    """Re-select tools for the current activity and return ``FunctionTool``s.

    ``query_or_input`` may be a query string (used verbatim) or a run input /
    message list (the query is derived from the latest activity). Each selected
    :class:`~agent_gantry.integrations.frameworks.base.ToolSpec` is wrapped as a
    native :class:`agents.FunctionTool` whose ``on_invoke_tool`` routes back
    through ``gantry.execute``.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    _require_agents()  # fail fast with the pip hint before doing any work
    query = _query_from(query_or_input)
    # select_or_empty: no extractable text (empty/None input, or only non-text
    # items) exposes no tools rather than selecting on an empty embedding,
    # which yields an arbitrary top-k for some embedders. _refresh_agent_tools
    # then clears agent.tools instead of surfacing unrelated functions.
    specs = await GantryToolset(gantry).select_or_empty(
        query,
        limit=limit,
        score_threshold=score_threshold,
        namespaces=namespaces,
        required=required,
        always_include=always_include,
    )
    return [
        _spec_to_openai_agents(spec, error_mode=error_mode, error_formatter=error_formatter)
        for spec in specs
    ]


async def _apply_selection(
    agent: Any,
    gantry: AgentGantry,
    query_or_input: Any,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> list[Any] | None:
    """Select for ``query_or_input`` and rewrite ``agent.tools`` in place.

    Returns the new tools, or ``None`` when retrieval failed and
    ``agent.tools`` was left as it was (logged at WARNING).

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    try:
        tools = await _select_function_tools(
            gantry,
            query_or_input,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
            error_mode=error_mode,
            error_formatter=error_formatter,
        )
    except (ImportError, MissingRequiredToolError, QueryBoundsError):
        # Install-time and configuration errors, not transient retrieval
        # failures — keep failing fast/loudly.
        raise
    except Exception:
        logger.warning(
            "_refresh_agent_tools: semantic retrieval failed; "
            "continuing with the previous turn's tools.",
            exc_info=True,
        )
        return None
    # Mutate in place: the run loop reads ``agent.tools`` each turn, and other
    # references to this agent must see the same updated list.
    agent.tools[:] = tools
    return tools


async def _refresh_agent_tools(
    agent: Any,
    gantry: AgentGantry,
    query_or_input: Any,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
) -> list[Any]:
    """Re-select tools and rewrite ``agent.tools`` in place; return the new tools.

    This is the single per-turn primitive both the hook and the session build
    on. ``agent.tools`` is mutated in place (not reassigned) so any reference the
    SDK already holds to the agent observes the updated surface on its next
    ``get_all_tools`` snapshot.

    Never raises on selection failure — a broken retrieval must not break the
    agent's turn/run. ``agent.tools`` persists across turns, so on failure
    this logs a WARNING and leaves it untouched (skips the mutation) rather
    than wiping it to empty — see "Per-turn selection-failure policy" in
    ``integrations/frameworks/README.md``.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    tools = await _apply_selection(
        agent,
        gantry,
        query_or_input,
        limit=limit,
        score_threshold=score_threshold,
        namespaces=namespaces,
        required=required,
        always_include=always_include,
        error_mode=error_mode,
        error_formatter=error_formatter,
    )
    return list(agent.tools) if tools is None else tools


def _gantry_run_hooks(
    gantry: AgentGantry,
    agent: Any,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
    applied_query: str | None = None,
) -> Any:
    """Build a :class:`agents.RunHooks` that keeps ``agent.tools`` selected.

    Pass the returned hooks to ``Runner.run(agent, input, hooks=...)``. They
    re-select at two points, both by rewriting ``agent.tools`` in place:

    - ``on_agent_start`` -- from the run's input, as the agent starts (and
      again when a handoff returns to it). From ``openai-agents`` 0.23.1 the
      SDK fires this before it lists the turn's tools, so the **first** model
      request already carries the selection.
    - ``on_llm_start`` -- from the running conversation, before each model
      request. The turn's tools are already listed by then, so this takes
      effect from the **next** turn: the tool surface follows the run as it
      chains tools, with no application code between turns.

    Before 0.23.1 the SDK lists a turn's tools before either hook fires, so
    the first request uses ``agent.tools`` as the caller left it. Select first
    with :func:`_refresh_agent_tools`, or use :func:`_run_with_gantry` /
    :class:`GantryAgentSession`, which do. A run that starts there with tools
    other than the ones its input selects gets a ``RuntimeWarning``.

    Args:
        applied_query: A query the caller has already selected ``agent.tools``
            for (the session's pre-run selection), so the hooks do not repeat
            that retrieval.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    agents = _require_agents()
    check_query_bounds(limit=limit, score_threshold=score_threshold, owner="OpenAIAgentsAdapter")
    first_request_covered = _first_request_sees_input()

    class _GantryRunHooks(agents.RunHooks):  # type: ignore[misc, valid-type]
        """Re-selects ``agent.tools`` at agent start and before each model call."""

        def __init__(self) -> None:
            super().__init__()
            # The query ``agent.tools`` currently reflects, so an unchanged
            # conversation (agent start, then the first model call) costs one
            # retrieval rather than one per hook. Paired with the tool objects
            # that selection put there: if anything else has rewritten the
            # list since (a ``refresh`` for another query, a second hooks
            # object, a reuse of this one across runs), the record is stale
            # and the selection is made again.
            self._applied: str | None = applied_query
            self._applied_tools = list(agent.tools) if applied_query is not None else None

        def _still_applied(self, query: str) -> bool:
            held = self._applied_tools
            return (
                held is not None
                and query == self._applied
                and len(held) == len(agent.tools)
                and all(mine is theirs for mine, theirs in zip(held, agent.tools, strict=True))
            )

        async def _reselect(self, query_or_input: Any) -> None:
            query = _query_from(query_or_input)
            if self._still_applied(query):
                return
            tools = await _apply_selection(
                agent,
                gantry,
                query,
                limit=limit,
                score_threshold=score_threshold,
                namespaces=namespaces,
                required=required,
                always_include=always_include,
                error_mode=error_mode,
                error_formatter=error_formatter,
            )
            # A failed retrieval is not recorded, so the next hook tries again.
            if tools is not None:
                self._applied = query
                self._applied_tools = list(tools)

        async def on_agent_start(self, context: Any, hook_agent: Any) -> None:
            # Run hooks fire for every agent in the run (handoff targets too);
            # only the bound agent's tools are Gantry's to rewrite.
            if hook_agent is not None and hook_agent is not agent:
                return
            turn_input = getattr(context, "turn_input", None)
            if not _query_from(turn_input):
                # No input on the context (SDKs before ``turn_input``, or a
                # resumed run): leave the tools alone rather than clear them.
                return
            offered = [getattr(tool, "name", None) for tool in agent.tools]
            await self._reselect(turn_input)
            if first_request_covered:
                return
            selected = [getattr(tool, "name", None) for tool in agent.tools]
            if sorted(map(str, offered)) != sorted(map(str, selected)):
                warnings.warn(
                    "openai-agents<0.23.1 lists an agent's tools before any run hook fires, "
                    f"so this run's first model request was built with {offered} rather "
                    f"than the selection for its input, {selected}. Call "
                    "adapter.refresh(agent, input) before Runner.run, or use adapter.run() "
                    "/ adapter.session(), which do; openai-agents>=0.23.1 needs neither.",
                    RuntimeWarning,
                    # The SDK calls this hook, so no frame above is the user's:
                    # attribute the warning to this module, not SDK internals.
                    stacklevel=1,
                )

        async def on_llm_start(
            self,
            context: Any,
            hook_agent: Any,
            system_prompt: str | None,
            input_items: list[Any],
        ) -> None:
            if hook_agent is not None and hook_agent is not agent:
                return
            # ``input_items`` is the model input for the upcoming call: the full
            # running conversation. Derive the query from its latest activity.
            await self._reselect(input_items)

    return _GantryRunHooks()


class GantryAgentSession:
    """Drive an :class:`agents.Agent` whose tools Gantry re-selects each run.

    Holds an ``agent`` and a ``gantry`` and exposes :meth:`run` — call it once
    per conversational turn. Each :meth:`run`:

    1. derives a retrieval query from the supplied input (latest activity),
    2. re-selects the top-k relevant tools and rewrites ``agent.tools`` in place
       **before** the run, so the new selection is in force on the first turn,
    3. installs :func:`_gantry_run_hooks` for the run so any *additional* turns
       inside that run (tool-call loops) also track the conversation,
    4. delegates to ``agents.Runner.run`` and returns its ``RunResult``.

    This is the run-loop-level deep mechanism: between turns the tool set is
    re-selected and applied immediately, and within a multi-turn run the hook
    keeps it fresh.

    Args:
        agent: The :class:`agents.Agent` whose ``tools`` are refreshed in place.
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` providing
            semantic retrieval and execution.
        limit: Maximum number of tools re-selected per turn. Defaults to ``5``.
        score_threshold: Minimum semantic relevance score. Defaults to ``0.0``.
        namespaces: Optional namespace filter for selection.
    """

    def __init__(
        self,
        agent: Any,
        gantry: AgentGantry,
        *,
        limit: int = DEFAULT_TOOL_LIMIT,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode = "model",
        error_formatter: ErrorFormatter | None = None,
    ) -> None:
        _require_agents()  # fail fast with the pip hint at construction
        check_query_bounds(limit=limit, score_threshold=score_threshold, owner="GantryAgentSession")
        self._agent = agent
        self._gantry = gantry
        self._limit = limit
        self._score_threshold = score_threshold
        self._namespaces = namespaces
        self._required = required
        self._always_include = always_include
        self._error_mode = check_error_mode(error_mode)
        self._error_formatter = error_formatter

    @property
    def agent(self) -> Any:
        return self._agent

    @property
    def gantry(self) -> AgentGantry:
        return self._gantry

    async def refresh(self, query_or_input: Any) -> list[Any]:
        """Re-select tools for ``query_or_input`` and rewrite ``agent.tools``.

        The per-turn primitive, exposed for callers that want to re-select
        without immediately running (e.g. to inspect the chosen tools). Returns
        the freshly selected :class:`agents.FunctionTool` list.
        """
        tools = await self._apply(query_or_input)
        return list(self._agent.tools) if tools is None else tools

    async def _apply(self, query_or_input: Any) -> list[Any] | None:
        return await _apply_selection(
            self._agent,
            self._gantry,
            query_or_input,
            limit=self._limit,
            score_threshold=self._score_threshold,
            namespaces=self._namespaces,
            required=self._required,
            always_include=self._always_include,
            error_mode=self._error_mode,
            error_formatter=self._error_formatter,
        )

    async def run(self, run_input: Any, **run_kwargs: Any) -> Any:
        """Re-select tools for ``run_input``, run the agent, return the result.

        Re-selects and applies the tool set before the run (immediate first-turn
        accuracy) and installs :func:`_gantry_run_hooks` for the run (intra-run
        per-turn dynamism). Extra ``run_kwargs`` are forwarded to
        ``agents.Runner.run`` (``context``, ``max_turns``, ``session``, …). If
        the caller supplies their own ``hooks`` it is respected and the Gantry
        hook is not installed.
        """
        agents = _require_agents()
        applied = await self._apply(run_input)
        if "hooks" not in run_kwargs:
            # The hooks start from this selection instead of repeating it.
            run_kwargs["hooks"] = self._gantry_hooks(
                None if applied is None else _query_from(run_input)
            )
        return await agents.Runner.run(self._agent, run_input, **run_kwargs)

    def _gantry_hooks(self, applied_query: str | None = None) -> Any:
        return _gantry_run_hooks(
            self._gantry,
            self._agent,
            limit=self._limit,
            score_threshold=self._score_threshold,
            namespaces=self._namespaces,
            required=self._required,
            always_include=self._always_include,
            error_mode=self._error_mode,
            error_formatter=self._error_formatter,
            applied_query=applied_query,
        )


async def _run_with_gantry(
    agent: Any,
    gantry: AgentGantry,
    run_input: Any,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
    required: list[str] | None = None,
    always_include: list[str] | None = None,
    error_mode: ErrorMode = "model",
    error_formatter: ErrorFormatter | None = None,
    **run_kwargs: Any,
) -> Any:
    """Re-select ``agent``'s tools for ``run_input`` and run it once via Gantry.

    The primary one-shot entry point: equivalent to constructing a
    :class:`GantryAgentSession` and calling :meth:`~GantryAgentSession.run` a
    single time. Re-selects tools from the input's latest activity and updates
    ``agent.tools`` **before** the run (so the first turn already sees the right
    tools), installs :func:`_gantry_run_hooks` for intra-run dynamism, then
    delegates to ``agents.Runner.run``. Extra ``run_kwargs`` are forwarded.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    session = GantryAgentSession(
        agent,
        gantry,
        limit=limit,
        score_threshold=score_threshold,
        namespaces=namespaces,
        required=required,
        always_include=always_include,
        error_mode=error_mode,
        error_formatter=error_formatter,
    )
    return await session.run(run_input, **run_kwargs)


# --------------------------------------------------------------------------- #
# Per-turn selection through ``FunctionTool.is_enabled``
# --------------------------------------------------------------------------- #


def _text_of(content: Any) -> str:
    """Plain text of a Responses-style message ``content`` (str or parts list)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(part, str) and part.strip():
                parts.append(part.strip())
        return " ".join(parts)
    return ""


def _latest_user_text(turn_input: Any) -> str:
    """The newest user message in a run's ``turn_input``."""
    if isinstance(turn_input, str):
        return turn_input.strip()
    try:
        items = list(turn_input or [])
    except TypeError:
        return ""
    for item in reversed(items):
        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
        if role != "user":
            continue
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        text = _text_of(content)
        if text:
            return text
    return ""


def _called_tools(turn_input: Any) -> list[str]:
    """Names of function calls already present in the run's input history."""
    names: dict[str, None] = {}
    try:
        items = list(turn_input or [])
    except TypeError:
        return []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            name = item.get("name")
            if isinstance(name, str):
                names[name] = None
    return list(names)


class _RunState:
    """One run's selection state (cache, sticky surface, newest tool output)."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.cache: dict[tuple[str, tuple[str, ...]], list[ToolSpec]] = {}
        self.offered: dict[str, ToolSpec] = {}
        self.last_selected: dict[str, int] = {}
        self.step = 0
        self.turn_key: tuple[Any, ...] | None = None
        self.enabled: frozenset[str] = frozenset()
        self.last_output = ""
        self.outputs = 0
        self.called: list[str] = []
        # Model-request count at a failed retrieval; None when the last one worked.
        self.failed_at: int | None = None


def _requests(ctx: Any) -> int | None:
    """The run's model-request count: a per-turn counter the SDK keeps on ``usage``."""
    requests = getattr(getattr(ctx, "usage", None), "requests", None)
    return requests if isinstance(requests, int) else None


_MAX_CACHED_SELECTIONS = 32


class GantryDynamicTools:
    """Per-turn Gantry selection for the OpenAI Agents SDK via ``is_enabled``.

    Built by :meth:`OpenAIAgentsAdapter.dynamic_tools`. :meth:`build` returns
    the catalogue as ``FunctionTool`` objects whose ``is_enabled`` callbacks all
    consult this selector. The SDK evaluates them at the start of every turn
    with the run's ``RunContextWrapper``; the first evaluation in a turn runs one
    Gantry selection (cached per query within the run) and the rest reuse it.

    The selection query is the newest user message in ``turn_input``, or --
    after a Gantry tool call in this run -- that message plus the tool's output
    (first 2,000 characters), recorded by the tool itself. Run state is keyed
    to the run's ``usage`` object, which the SDK shares between a run's
    ``RunContextWrapper`` and every ``ToolContext`` in it, and is released with
    the run, so concurrent runs of one agent never see each other's state.
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        limit: int = DEFAULT_TOOL_LIMIT,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode = "model",
        error_formatter: ErrorFormatter | None = None,
        sticky: bool = True,
        max_offered: int | None = None,
        query_from_results: bool = True,
    ) -> None:
        check_query_bounds(limit=limit, score_threshold=score_threshold, owner="GantryDynamicTools")
        if max_offered is not None and max_offered < 1:
            raise ValueError("max_offered must be >= 1")
        self._gantry = gantry
        self._selector = GantryToolset(gantry)
        self._limit = limit
        self._score_threshold = score_threshold
        self._namespaces = namespaces
        self._required = required
        self._always_include = always_include
        self._error_mode = error_mode
        self._error_formatter = error_formatter
        self._sticky = sticky
        self._max_offered = 3 * limit if max_offered is None else max_offered
        self._query_from_results = query_from_results
        self._states: dict[int, tuple[weakref.ref[Any], _RunState]] = {}
        self._pinned: frozenset[str] = frozenset()

    async def build(self) -> list[Any]:
        """The catalogue (in-scope registry tools) as gated ``FunctionTool``s.

        Raises:
            ImportError: If ``openai-agents`` is missing or older than 0.23.1
                (see :func:`~agent_gantry.integrations.frameworks.openai_agents._require_per_turn_sdk`).
        """
        _require_agents()
        _require_per_turn_sdk()
        try:
            await self._gantry.ensure_synced()
        except Exception:
            logger.warning(
                "GantryDynamicTools: sync failed; building from the tools already registered.",
                exc_info=True,
            )
        tools = [
            t
            for t in self._gantry.list_tools_sync()
            if not self._namespaces or t.namespace in self._namespaces
        ]
        specs = [spec_from_tool(self._gantry, t) for t in tools]
        # Pins can sit outside ``namespaces``; the catalogue must still hold them
        # or the selector would enable a name with no matching tool.
        specs += _resolve_pins(
            self._gantry,
            specs,
            required=self._required,
            always_include=self._always_include,
            namespaces=self._namespaces,
        )
        specs = _with_native_names(self._gantry, specs)
        pins = _resolve_pins(
            self._gantry,
            [],
            required=self._required,
            always_include=self._always_include,
            namespaces=self._namespaces,
        )
        pin_keys = {(pin._namespace, pin.name) for pin in pins}
        self._pinned = frozenset(
            spec.native_name for spec in specs if (spec._namespace, spec.name) in pin_keys
        )
        return [
            _spec_to_openai_agents(
                spec,
                error_mode=self._error_mode,
                error_formatter=self._error_formatter,
                is_enabled=self._gate(spec.native_name),
                on_output=self._record_output,
            )
            for spec in specs
        ]

    # -- per-run state ---------------------------------------------------- #
    def _state(self, ctx: Any) -> _RunState:
        anchor = getattr(ctx, "usage", None)
        if anchor is None:  # pragma: no cover - every SDK context has usage
            anchor = ctx
        key = id(anchor)
        entry = self._states.get(key)
        if entry is not None and entry[0]() is anchor:
            return entry[1]
        state = _RunState()
        states = self._states

        def _drop(_ref: Any, key: int = key) -> None:
            states.pop(key, None)

        self._states[key] = (weakref.ref(anchor, _drop), state)
        return state

    def _record_output(self, ctx: Any, name: str, text: str) -> None:
        state = self._state(ctx)
        if text:
            state.last_output = text[:MAX_TOOL_RESULT_QUERY_CHARS]
        state.outputs += 1
        if name not in state.called:
            state.called.append(name)

    # -- selection -------------------------------------------------------- #
    def _gate(self, name: str) -> Any:
        async def is_enabled(ctx: Any, agent: Any) -> bool:
            return name in await self._enabled(ctx)

        is_enabled.__name__ = f"gantry_enabled_{name}"
        return is_enabled

    async def _enabled(self, ctx: Any) -> frozenset[str]:
        state = self._state(ctx)
        turn_input = getattr(ctx, "turn_input", None)
        prompt = _latest_user_text(turn_input)
        query = prompt
        if self._query_from_results and state.last_output:
            query = f"{prompt}\n{state.last_output}".strip()
        used = tuple(dict.fromkeys([*_called_tools(turn_input), *state.called]))
        turn_key = (query, used, state.outputs)
        requests = _requests(ctx)

        def current() -> bool:
            # A failed retrieval holds only for the turn it happened in, even
            # when the key is unchanged (e.g. only non-Gantry tools ran since).
            return state.turn_key == turn_key and (
                state.failed_at is None or state.failed_at == requests
            )

        if current():
            return state.enabled
        async with state.lock:
            if current():
                return state.enabled
            specs = await self._select(state, query, used)
            if specs is None:
                # Retrieval failed: every gate this turn reuses the last tools
                # plus the pins (which never depend on retrieval).
                state.enabled = state.enabled | self._pinned
                state.turn_key = turn_key
                state.failed_at = requests
                return state.enabled
            state.failed_at = None
            if self._sticky:
                state.step += 1
                specs = carry_over(
                    state.offered, state.last_selected, state.step, specs, self._max_offered
                )
            state.enabled = frozenset(spec.native_name for spec in specs)
            state.turn_key = turn_key
            return state.enabled

    async def _select(
        self, state: _RunState, query: str, used: tuple[str, ...]
    ) -> list[ToolSpec] | None:
        key = (query, used)
        if key in state.cache:
            return state.cache[key]
        try:
            specs = await self._selector.select_or_empty(
                query,
                limit=self._limit,
                score_threshold=self._score_threshold,
                namespaces=self._namespaces,
                tools_already_used=list(used),
                required=self._required,
                always_include=self._always_include,
            )
        except (MissingRequiredToolError, QueryBoundsError, SelectorFallbackError):
            raise
        except Exception:
            logger.warning(
                "GantryDynamicTools: semantic retrieval failed; offering this run's "
                "previously offered tools (if any) this turn.",
                exc_info=True,
            )
            return None
        if len(state.cache) >= _MAX_CACHED_SELECTIONS:
            state.cache.pop(next(iter(state.cache)))
        state.cache[key] = specs
        return specs


__all__ = ["GantryAgentSession", "GantryDynamicTools"]
