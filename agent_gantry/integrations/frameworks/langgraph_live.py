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
LangGraph 1.x's :func:`langgraph.prebuilt.create_react_agent` accepts ``model``
as a **dynamic callable** with signature ``(state, runtime) -> BaseChatModel``
(sync *or* async — an ``Awaitable[BaseChatModel]`` is supported). The agent's
``call_model`` node resolves this callable on *every* model turn, so it is the
correct per-turn hook for rebinding tools (``pre_model_hook`` / ``post_model_hook``
exist but operate on messages and cannot change the tools advertised to the LLM).

LangGraph imposes one rule on the dynamic model: the tools bound via
``model.bind_tools(...)`` **must be a subset of the static ``tools=`` argument**,
because the agent's ``ToolNode`` can only execute tools it was constructed with.
This module therefore:

1. builds the **full superset** of Gantry tools once (every registered tool,
   wrapped as a LangChain ``StructuredTool`` that routes execution through
   ``gantry.execute``) and passes it as ``tools=`` so the ``ToolNode`` can run
   *any* tool Gantry might pick;
2. installs an **async dynamic-model callable** that, on each turn, derives a
   retrieval query from the current ``state["messages"]`` via
   :func:`~agent_gantry.query.latest_activity`, runs Gantry semantic selection,
   and returns ``model.bind_tools(<selected subset>)``.

Because the same ``StructuredTool`` identity is shared between the superset and
the per-turn subset, the bound subset is always a valid subset and the ``ToolNode``
executes the selected tools correctly. The LLM on each turn sees exactly the tools
Gantry chose for the *current* conversation — stale selections never accumulate.

Usage
-----
.. code-block:: python

    from agent_gantry import AgentGantry
    from agent_gantry.integrations.frameworks.langgraph_live import (
        create_gantry_react_agent,
    )

    agent = create_gantry_react_agent(chat_model, gantry, limit=5)
    result = await agent.ainvoke({"messages": [("user", "what's the weather?")]})

The ``langgraph`` / ``langchain-core`` imports are lazy so ``import agent_gantry``
never requires them; the helpful ``pip install langgraph langchain-core`` hint is
raised only when the live agent is actually built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, spec_from_tool
from agent_gantry.integrations.frameworks.langchain import spec_to_langchain
from agent_gantry.query import latest_activity

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def _import_create_react_agent() -> Any:
    """Lazily import :func:`langgraph.prebuilt.create_react_agent`.

    Deferred to call time so the module stays importable without LangGraph
    installed.

    Raises:
        ImportError: If ``langgraph`` / ``langchain-core`` are not installed.
    """
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(
            "LangGraph live support requires `langgraph` and `langchain-core`. "
            "Install them with `pip install langgraph langchain-core`."
        ) from exc
    return create_react_agent


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


async def select_tools_for_state(
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
    specs = await GantryToolset(gantry).select(
        query, limit=limit, score_threshold=score_threshold
    )
    return [spec_to_langchain(s) for s in specs]


async def _all_tools(gantry: AgentGantry) -> list[Any]:
    """Wrap every registered Gantry tool as a LangChain ``StructuredTool``.

    This is the static superset handed to ``create_react_agent(tools=...)`` so
    the agent's ``ToolNode`` can execute *any* tool the per-turn selection might
    pick (LangGraph requires bound tools to be a subset of this set).
    """
    definitions = await gantry.list_tools()
    return [spec_to_langchain(spec_from_tool(gantry, d)) for d in definitions]


def create_gantry_react_agent(
    model: Any,
    gantry: AgentGantry,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    **agent_kwargs: Any,
) -> Any:
    """Build a LangGraph ReAct agent whose tools are re-selected every turn.

    Unlike the static :func:`~agent_gantry.integrations.frameworks.langgraph.for_langgraph`
    (tools fixed at construction), this wires Gantry in as a **live per-turn**
    provider via LangGraph's dynamic-``model`` callable. On each model turn the
    agent:

    1. derives a retrieval query from the current ``state["messages"]``,
    2. runs Gantry semantic selection (top-``limit`` tools), and
    3. binds *only* those tools to ``model`` for that turn.

    The full superset of registered tools is passed to ``tools=`` so the agent's
    ``ToolNode`` can execute whatever was selected (LangGraph requires bound
    tools to be a subset of the static set).

    Args:
        model: A LangChain ``BaseChatModel`` to bind Gantry-selected tools to
            each turn.
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` providing
            semantic retrieval and execution.
        limit: Maximum number of tools re-selected per turn. Defaults to ``5``.
        score_threshold: Minimum semantic relevance score for selected tools.
            Defaults to ``0.0`` (no filtering).
        **agent_kwargs: Forwarded verbatim to ``create_react_agent`` (e.g.
            ``prompt``, ``checkpointer``, ``state_schema``).

    Note:
        This is a **synchronous** factory; it resolves the (async) tool superset
        on a worker thread via Gantry's sync bridge, so it is safe to call from a
        running event loop but will block the calling thread until enumeration
        completes. In already-async contexts (Jupyter, FastAPI startup) prefer
        :func:`acreate_gantry_react_agent`, which awaits the enumeration directly.

    Returns:
        The compiled LangGraph agent (a ``Pregel`` graph) ready for
        ``.ainvoke`` / ``.invoke``.
    """
    from agent_gantry.integrations.frameworks.base import _run_coroutine_sync

    superset = _run_coroutine_sync(_all_tools(gantry))
    return _build_react_agent(
        model, gantry, superset, limit=limit, score_threshold=score_threshold, **agent_kwargs
    )


async def acreate_gantry_react_agent(
    model: Any,
    gantry: AgentGantry,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    **agent_kwargs: Any,
) -> Any:
    """Async-native variant of :func:`create_gantry_react_agent`.

    Awaits the tool-superset enumeration directly (no sync bridge), so it is the
    right choice inside an already-running event loop (Jupyter, FastAPI startup).
    Behaviour is otherwise identical — tools are re-selected by Gantry on every
    model turn via LangGraph's dynamic-``model`` callable.
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
    """Compile the dynamic-``model`` ReAct agent given a pre-resolved superset."""
    create_react_agent = _import_create_react_agent()

    async def _dynamic_model(state: Any, runtime: Any) -> Any:
        """Per-turn hook: re-select Gantry tools and bind them to ``model``."""
        selected = await select_tools_for_state(
            gantry, state, limit=limit, score_threshold=score_threshold
        )
        if not selected:
            return model
        return model.bind_tools(selected)

    return create_react_agent(model=_dynamic_model, tools=superset, **agent_kwargs)


__all__ = [
    "create_gantry_react_agent",
    "acreate_gantry_react_agent",
    "select_tools_for_state",
]
