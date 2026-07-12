"""DEEP per-turn dynamic-tool provider for LangGraph.

Where :mod:`agent_gantry.integrations.frameworks.langgraph` exposes the *static*
helpers (``for_langgraph`` / ``spec_to_langgraph``) — you select a slice of tools
once and hand the resulting LangChain ``StructuredTool`` list to a graph at
construction time — this module wires Agent-Gantry into LangGraph as a **live,
per-turn** tool source, matching the depth of the Microsoft Agent Framework
``GantryContextProvider``: the set of tools the model can call is **re-selected by
Gantry on every model turn**.

The native hook
---------------
This module builds the agent with :func:`langchain.agents.create_agent`, the
LangChain/LangGraph-recommended agent constructor (``langchain>=1.0``) that
replaces the deprecated :func:`langgraph.prebuilt.create_react_agent` — the
latter is removed outright in LangGraph 2.0, and ``langchain>=1.3.4`` /
``langgraph>=1.2.4`` (this project's floors, pinned together in the
``agent-frameworks`` extra) both post-date the migration, so ``create_agent``
is unconditionally available and no runtime fallback to the deprecated symbol
is kept (see :func:`_import_create_agent`).

``create_agent`` has no dynamic-``model`` callable like the old
``create_react_agent`` did. Its per-turn hook is **middleware**: an
``AgentMiddleware`` subclass implementing ``awrap_model_call(request, handler)``
is invoked on *every* model turn with a ``ModelRequest`` carrying the current
``state`` and ``tools``, and can call ``handler(request.override(tools=...))``
to change the tools advertised to the model for that turn only. This is the
same mechanism ``langchain.agents.middleware.LLMToolSelectorMiddleware`` (an
LLM-based tool selector shipped in ``langchain``) uses, so it is the documented,
supported way to reproduce ``create_react_agent``'s dynamic-model tool rebinding
under the new API.

LangGraph still imposes one rule: tools placed in ``request.tools`` **must be a
subset (by name) of the static ``tools=`` argument** passed to ``create_agent``,
because the agent's ``ToolNode`` can only execute tools it was constructed with.
This module therefore:

1. builds the **full superset** of Gantry tools once (every registered tool,
   wrapped as a LangChain ``StructuredTool`` that routes execution through
   ``gantry.execute``) and passes it as ``tools=`` so the ``ToolNode`` can run
   *any* tool Gantry might pick;
2. installs a ``wrap_model_call`` middleware that, on each turn, derives a
   retrieval query from the current ``state["messages"]`` via
   :func:`~agent_gantry.query.latest_activity`, runs Gantry semantic selection,
   and overrides ``request.tools`` with just the selected subset.

Because tool names (not identity) are what the ``ToolNode`` subset check and
dispatch rely on, the per-turn subset is always valid and executes correctly.
The LLM on each turn sees exactly the tools Gantry chose for the *current*
conversation — stale selections never accumulate.

Usage
-----
.. code-block:: python

    from agent_gantry import AgentGantry
    from agent_gantry.integrations.frameworks.langgraph_live import (
        _create_gantry_react_agent,
    )

    agent = _create_gantry_react_agent(chat_model, gantry, limit=5)
    result = await agent.ainvoke({"messages": [("user", "what's the weather?")]})

The ``langchain`` / ``langgraph`` / ``langchain-core`` imports are lazy so
``import agent_gantry`` never requires them; the helpful
``pip install langchain langgraph langchain-core`` hint is raised only when the
live agent is actually built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, spec_from_tool
from agent_gantry.integrations.frameworks.langchain import _spec_to_langchain
from agent_gantry.query import latest_activity

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def _import_create_agent() -> tuple[Any, Any]:
    """Lazily import ``langchain.agents.create_agent`` and ``AgentMiddleware``.

    Deferred to call time so the module stays importable without LangChain /
    LangGraph installed.

    No fallback to the deprecated :func:`langgraph.prebuilt.create_react_agent`
    is provided. This project's floors — ``langchain>=1.3.4`` and
    ``langgraph>=1.2.4`` — are pinned together in the ``agent-frameworks``
    extra (see ``pyproject.toml``), and ``langchain.agents.create_agent`` has
    existed since ``langchain`` 1.0.0, so every supported install already has
    the new API. A fallback would just be dead code that silently regresses to
    an API LangGraph 2.0 removes outright.

    Raises:
        ImportError: If ``langchain`` / ``langgraph`` are not installed.
    """
    try:
        from langchain.agents import create_agent
        from langchain.agents.middleware import AgentMiddleware
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(
            "LangGraph live support requires `langchain` (>=1.0, for "
            "`langchain.agents.create_agent`) and `langgraph`. Install them with "
            "`pip install langchain langgraph langchain-core`."
        ) from exc
    return create_agent, AgentMiddleware


def _query_from_state(state: Any) -> str:
    """Derive the per-turn retrieval query from a LangGraph agent state.

    LangGraph's ReAct state is a mapping carrying a ``"messages"`` list. The
    query is the latest *activity* in that history (new user request, or the
    last tool result when the agent is chaining tools) via
    :func:`~agent_gantry.query.latest_activity`. LangChain ``BaseMessage``
    objects expose ``.content`` (and ``.type`` for the role), which
    ``latest_activity`` already understands.
    """
    if isinstance(state, dict):
        messages = state.get("messages")
    else:  # pragma: no cover - defensive for non-dict state schemas
        messages = getattr(state, "messages", None)
    return latest_activity(messages) or ""


async def _select_tools_for_state(
    gantry: AgentGantry,
    state: Any,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
) -> list[Any]:
    """Re-select Gantry tools for the current turn and wrap them for LangChain.

    Derives the query from ``state["messages"]``, runs Gantry semantic
    selection, and returns the chosen tools as LangChain ``StructuredTool``
    objects (each routing execution through ``gantry.execute``). Exposed
    separately from the model callable so it can be unit-tested directly.
    """
    query = _query_from_state(state)
    if not (query or "").strip():
        # No retrieval signal this turn: bind no tools rather than selecting on
        # an empty embedding (consistent with the other live providers).
        return []
    specs = await GantryToolset(gantry).select(query, limit=limit, score_threshold=score_threshold)
    return [_spec_to_langchain(s) for s in specs]


async def _all_tools(gantry: AgentGantry) -> list[Any]:
    """Wrap every registered Gantry tool as a LangChain ``StructuredTool``.

    This is the static superset handed to ``create_agent(tools=...)`` so the
    agent's ``ToolNode`` can execute *any* tool the per-turn selection might
    pick (LangGraph requires the tools in each turn's ``ModelRequest`` to be a
    subset of this set, by name).
    """
    definitions = await gantry.list_tools()
    return [_spec_to_langchain(spec_from_tool(gantry, d)) for d in definitions]


def _create_gantry_react_agent(
    model: Any,
    gantry: AgentGantry,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    **agent_kwargs: Any,
) -> Any:
    """Build a LangGraph agent whose tools are re-selected every turn.

    Unlike the static :func:`~agent_gantry.integrations.frameworks.langgraph.for_langgraph`
    (tools fixed at construction), this wires Gantry in as a **live per-turn**
    provider via a ``create_agent`` ``wrap_model_call`` middleware hook. On each
    model turn the agent:

    1. derives a retrieval query from the current ``state["messages"]``,
    2. runs Gantry semantic selection (top-``limit`` tools), and
    3. overrides the model request's tools with *only* that selection for the turn.

    The full superset of registered tools is passed to ``tools=`` so the agent's
    ``ToolNode`` can execute whatever was selected (LangGraph requires the
    per-turn tools to be a subset, by name, of the static set).

    Args:
        model: A LangChain ``BaseChatModel`` (or model identifier string) for
            ``create_agent`` to bind Gantry-selected tools to each turn.
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` providing
            semantic retrieval and execution.
        limit: Maximum number of tools re-selected per turn. Defaults to ``5``.
        score_threshold: Minimum semantic relevance score for selected tools.
            Defaults to ``0.0`` (no filtering).
        **agent_kwargs: Forwarded verbatim to ``create_agent`` (e.g.
            ``system_prompt``, ``checkpointer``, ``state_schema``,
            ``middleware`` — note the system-prompt kwarg is ``system_prompt``,
            not the old ``create_react_agent``'s ``prompt``). Any ``middleware``
            passed here is appended *after* Gantry's tool-selection middleware.

    Note:
        This is a **synchronous** factory; it resolves the (async) tool superset
        on a worker thread via Gantry's sync bridge, so it is safe to call from a
        running event loop but will block the calling thread until enumeration
        completes. In already-async contexts (Jupyter, FastAPI startup) prefer
        :func:`_acreate_gantry_react_agent`, which awaits the enumeration directly.

    Returns:
        The compiled LangGraph agent (a ``Pregel`` graph) ready for
        ``.ainvoke`` / ``.invoke``.
    """
    from agent_gantry.integrations.frameworks.base import _run_coroutine_sync

    superset = _run_coroutine_sync(_all_tools(gantry))
    return _build_react_agent(
        model, gantry, superset, limit=limit, score_threshold=score_threshold, **agent_kwargs
    )


async def _acreate_gantry_react_agent(
    model: Any,
    gantry: AgentGantry,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    **agent_kwargs: Any,
) -> Any:
    """Async-native variant of :func:`_create_gantry_react_agent`.

    Awaits the tool-superset enumeration directly (no sync bridge), so it is the
    right choice inside an already-running event loop (Jupyter, FastAPI startup).
    Behaviour is otherwise identical — tools are re-selected by Gantry on every
    model turn via the ``create_agent`` ``wrap_model_call`` middleware hook.
    """
    superset = await _all_tools(gantry)
    return _build_react_agent(
        model, gantry, superset, limit=limit, score_threshold=score_threshold, **agent_kwargs
    )


def _build_react_agent(
    model: Any,
    gantry: AgentGantry,
    superset: list[Any],
    *,
    limit: int,
    score_threshold: float,
    **agent_kwargs: Any,
) -> Any:
    """Compile the per-turn tool-selecting agent given a pre-resolved superset."""
    create_agent, AgentMiddleware = _import_create_agent()  # noqa: N806

    class _GantryToolSelectionMiddleware(AgentMiddleware):  # type: ignore[misc, valid-type]
        """Re-selects Gantry tools into ``request.tools`` on every model turn.

        This is the ``create_agent`` equivalent of the dynamic-``model``
        callable the deprecated ``create_react_agent`` used for per-turn tool
        rebinding: ``awrap_model_call`` is invoked on *every* model turn before
        the model is called, so overriding ``request.tools`` here reproduces
        the exact per-turn re-selection semantics — the LLM sees exactly the
        tools Gantry chose for the *current* turn, and stale selections never
        accumulate.
        """

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            """Derive a query from the turn's state, re-select, and re-invoke."""
            selected = await _select_tools_for_state(
                gantry, request.state, limit=limit, score_threshold=score_threshold
            )
            # ``selected`` is `[]` when there's no retrieval signal this turn;
            # create_agent binds the model with no tools in that case (mirrors
            # the old dynamic-model callable's "return model unbound" branch).
            return await handler(request.override(tools=selected))

    middleware = [_GantryToolSelectionMiddleware(), *agent_kwargs.pop("middleware", ())]
    return create_agent(model=model, tools=superset, middleware=middleware, **agent_kwargs)
