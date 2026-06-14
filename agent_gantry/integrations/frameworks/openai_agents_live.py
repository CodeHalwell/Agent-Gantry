"""DEEP per-turn dynamic-tool provider for the OpenAI Agents SDK.

Where :mod:`agent_gantry.integrations.frameworks.openai_agents` exposes the
*static* helpers (``for_openai_agents`` / ``spec_to_openai_agents``) — you
select a slice of tools once and hand the resulting ``FunctionTool`` list to an
:class:`agents.Agent` — this module wires Agent-Gantry into the SDK as a
**live** tool source so the agent's tool set is **re-selected by Gantry as the
conversation progresses**, matching the depth of the Microsoft Agent Framework
``GantryContextProvider`` as closely as this SDK allows.

How deep can it go on this SDK?
-------------------------------
The OpenAI Agents SDK reads the agent's tool surface from ``agent.tools`` afresh
**on every turn**: the run loop calls ``Agent.get_all_tools`` once per turn to
snapshot the callable functions before invoking the model. There is no
documented "return the tools for this turn" callback, but there is a per-model
lifecycle hook, ``RunHooks.on_llm_start`` (and the per-agent
``AgentHooks.on_llm_start``), that fires immediately before each model request.
Because ``agent.tools`` is a plain mutable list, two complementary mechanisms
give genuine per-turn re-selection:

1. **Mid-run (hook-driven), via** :func:`gantry_run_hooks`. The returned
   :class:`agents.RunHooks` re-selects tools from the run's current input and
   rewrites ``agent.tools`` in place on every ``on_llm_start``. The SDK snapshots
   tools at the *start* of each turn and fires ``on_llm_start`` *within* that
   turn, so a rewrite here is picked up by the **next** turn's snapshot — i.e.
   within a single multi-turn ``Runner.run`` the tool set tracks the evolving
   conversation turn by turn, with no application code between turns. This is the
   deepest the SDK natively supports.

2. **Per-run (loop-driven), via** :class:`GantryAgentSession` /
   :func:`run_with_gantry`. Re-selects tools from the run input and updates
   ``agent.tools`` **before** each ``Runner.run`` call, so the new selection is
   in force on the very *first* turn of that run. Driving a multi-turn
   conversation as a sequence of runs (the common chat pattern) yields immediate,
   query-accurate re-selection each turn and composes with the hook for
   intra-run dynamism.

Both update ``agent.tools`` in place (an ``agent.clone(tools=...)`` variant is
also offered) and route every tool call back through ``gantry.execute`` via the
:class:`~agents.FunctionTool` objects built by ``spec_to_openai_agents`` — so
retries, timeouts, circuit breakers, and the security policy still apply.

The ``agents`` import is lazy (only inside the builders / hook), so ``import
agent_gantry`` never requires the OpenAI Agents SDK to be installed; the helpful
``pip install openai-agents`` hint is raised only when the live provider is used.

Usage
-----
.. code-block:: python

    from agents import Agent, Runner
    from agent_gantry import AgentGantry
    from agent_gantry.integrations.frameworks.openai_agents_live import (
        gantry_run_hooks, run_with_gantry, GantryAgentSession,
    )

    agent = Agent(name="assistant", tools=[])

    # Per-run (immediate first-turn re-selection) — primary entry point:
    result = await run_with_gantry(agent, gantry, "what's the weather in Paris?")

    # ...or hold a session across a multi-turn chat:
    session = GantryAgentSession(agent, gantry, limit=5)
    r1 = await session.run("what's the weather in Paris?")   # weather tools
    r2 = await session.run("now email that to my boss")      # email tools

    # ...or hand the hook to a single Runner.run for intra-run dynamism:
    await Runner.run(agent, input, hooks=gantry_run_hooks(gantry, agent))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset
from agent_gantry.integrations.frameworks.openai_agents import spec_to_openai_agents
from agent_gantry.query import latest_activity

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


_PIP_HINT = (
    "The OpenAI Agents SDK is required for the live (per-turn) Gantry provider; "
    "install it with `pip install openai-agents`."
)


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


async def select_function_tools(
    gantry: AgentGantry,
    query_or_input: Any,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
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
    specs = await GantryToolset(gantry).select(
        query,
        limit=limit,
        score_threshold=score_threshold,
        namespaces=namespaces,
    )
    return [spec_to_openai_agents(spec) for spec in specs]


async def refresh_agent_tools(
    agent: Any,
    gantry: AgentGantry,
    query_or_input: Any,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
) -> list[Any]:
    """Re-select tools and rewrite ``agent.tools`` in place; return the new tools.

    This is the single per-turn primitive both the hook and the session build
    on. ``agent.tools`` is mutated in place (not reassigned) so any reference the
    SDK already holds to the agent observes the updated surface on its next
    ``get_all_tools`` snapshot.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    tools = await select_function_tools(
        gantry,
        query_or_input,
        limit=limit,
        score_threshold=score_threshold,
        namespaces=namespaces,
    )
    # Mutate in place: the run loop reads ``agent.tools`` each turn, and other
    # references to this agent must see the same updated list.
    agent.tools[:] = tools
    return tools


def gantry_run_hooks(
    gantry: AgentGantry,
    agent: Any,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
) -> Any:
    """Build a :class:`agents.RunHooks` that re-selects tools every model call.

    Pass the returned hooks to ``Runner.run(agent, input, hooks=...)``. Before
    each model request (``on_llm_start``) Gantry re-selects the relevant tools
    from the turn's current input items and rewrites ``agent.tools`` in place.
    The SDK snapshots tools at the start of each turn, so the rewrite takes
    effect from the **next** turn within the same run — giving intra-run,
    per-turn dynamism with no application code between turns.

    For immediate first-turn accuracy, combine this with
    :func:`run_with_gantry` / :class:`GantryAgentSession`, which re-select
    *before* the run starts.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
    """
    agents = _require_agents()

    class _GantryRunHooks(agents.RunHooks):  # type: ignore[misc, valid-type]
        """Re-selects ``agent.tools`` from the live input before each model call."""

        async def on_llm_start(
            self,
            context: Any,
            hook_agent: Any,
            system_prompt: str | None,
            input_items: list[Any],
        ) -> None:
            # ``input_items`` is the model input for the upcoming call: the full
            # running conversation. Derive the query from its latest activity.
            await refresh_agent_tools(
                hook_agent if hook_agent is not None else agent,
                gantry,
                input_items,
                limit=limit,
                score_threshold=score_threshold,
                namespaces=namespaces,
            )

    return _GantryRunHooks()


class GantryAgentSession:
    """Drive an :class:`agents.Agent` whose tools Gantry re-selects each run.

    Holds an ``agent`` and a ``gantry`` and exposes :meth:`run` — call it once
    per conversational turn. Each :meth:`run`:

    1. derives a retrieval query from the supplied input (latest activity),
    2. re-selects the top-k relevant tools and rewrites ``agent.tools`` in place
       **before** the run, so the new selection is in force on the first turn,
    3. installs :func:`gantry_run_hooks` for the run so any *additional* turns
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
        limit: int = 5,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
    ) -> None:
        _require_agents()  # fail fast with the pip hint at construction
        self._agent = agent
        self._gantry = gantry
        self._limit = limit
        self._score_threshold = score_threshold
        self._namespaces = namespaces

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
        return await refresh_agent_tools(
            self._agent,
            self._gantry,
            query_or_input,
            limit=self._limit,
            score_threshold=self._score_threshold,
            namespaces=self._namespaces,
        )

    async def run(self, input: Any, **run_kwargs: Any) -> Any:
        """Re-select tools for ``input``, then run the agent and return the result.

        Re-selects and applies the tool set before the run (immediate first-turn
        accuracy) and installs :func:`gantry_run_hooks` for the run (intra-run
        per-turn dynamism). Extra ``run_kwargs`` are forwarded to
        ``agents.Runner.run`` (``context``, ``max_turns``, ``session``, …). If
        the caller supplies their own ``hooks`` it is respected and the Gantry
        hook is not installed.
        """
        agents = _require_agents()
        await self.refresh(input)
        run_kwargs.setdefault("hooks", self._gantry_hooks())
        return await agents.Runner.run(self._agent, input, **run_kwargs)

    def _gantry_hooks(self) -> Any:
        return gantry_run_hooks(
            self._gantry,
            self._agent,
            limit=self._limit,
            score_threshold=self._score_threshold,
            namespaces=self._namespaces,
        )


async def run_with_gantry(
    agent: Any,
    gantry: AgentGantry,
    input: Any,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    namespaces: list[str] | None = None,
    **run_kwargs: Any,
) -> Any:
    """Re-select ``agent``'s tools for ``input`` and run it once via Gantry.

    The primary one-shot entry point: equivalent to constructing a
    :class:`GantryAgentSession` and calling :meth:`~GantryAgentSession.run` a
    single time. Re-selects tools from the input's latest activity and updates
    ``agent.tools`` **before** the run (so the first turn already sees the right
    tools), installs :func:`gantry_run_hooks` for intra-run dynamism, then
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
    )
    return await session.run(input, **run_kwargs)


__all__ = [
    "GantryAgentSession",
    "gantry_run_hooks",
    "refresh_agent_tools",
    "run_with_gantry",
    "select_function_tools",
]
