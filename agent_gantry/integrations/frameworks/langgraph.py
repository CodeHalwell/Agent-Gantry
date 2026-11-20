"""LangGraph native tool adapter for Agent-Gantry.

LangGraph has no tool object of its own: graphs (``ToolNode``, ``create_agent``)
run LangChain ``BaseTool`` objects, so this adapter builds on the LangChain one
(errors as error ``ToolMessage``s, approval through ``interrupt()``, run context
through :func:`~agent_gantry.integrations.frameworks.langchain.current_tool_runtime`)
and adds per-turn selection: a ``create_agent`` middleware that re-selects the
Gantry tools offered on every model turn.

Public entry point: :class:`LangGraphAdapter`.
"""

from __future__ import annotations

from typing import Any

from agent_gantry.integrations.frameworks.langchain import LangChainAdapter
from agent_gantry.integrations.frameworks.langgraph_live import LANGGRAPH_FRAMEWORK_NAME
from agent_gantry.integrations.frameworks.native import ErrorMode


class LangGraphAdapter(LangChainAdapter):
    """Route Gantry tools into LangGraph, re-selected every model turn.

    ::

        from langchain.agents import create_agent
        from agent_gantry.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(gantry)
        agent = create_agent(model, middleware=[await adapter.middleware(limit=5)])

        # or, the same thing in one call
        agent = await adapter.areact_agent(model, limit=5, checkpointer=InMemorySaver())

        # or a static slice for your own graph's ToolNode
        tools = await adapter.select("summarise the incident", limit=3)

    :meth:`live` requires ``model=`` and returns :meth:`react_agent`'s agent;
    other keywords (``system_prompt``, ``checkpointer``, ``middleware``, ...)
    are forwarded to ``create_agent``.
    """

    live_tier = "per-turn"
    _live_delegate = "react_agent"
    _live_required_kwargs = ("model",)
    _framework_name = LANGGRAPH_FRAMEWORK_NAME

    def _agent_kwargs(
        self,
        limit: int | None,
        score_threshold: float,
        namespaces: list[str] | None,
        required: list[str] | None,
        always_include: list[str] | None,
        error_mode: ErrorMode | None,
    ) -> dict[str, Any]:
        return {
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            "error_mode": self._mode(error_mode),
            "error_formatter": self._error_formatter,
            "framework": self._framework_name,
        }

    def react_agent(
        self,
        model: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode | None = None,
        **agent_kwargs: Any,
    ) -> Any:
        """Build a ``create_agent`` agent that re-selects tools every model turn (sync).

        Blocks while it resolves the catalogue; in async code prefer
        :meth:`areact_agent`. ``sticky``, ``max_offered`` and
        ``query_from_results`` are accepted as in :meth:`middleware`; other
        keywords go to ``create_agent``.
        """
        from agent_gantry.integrations.frameworks.langgraph_live import (
            _create_gantry_react_agent,
        )

        return _create_gantry_react_agent(
            model,
            self._gantry,
            **self._agent_kwargs(
                limit, score_threshold, namespaces, required, always_include, error_mode
            ),
            **agent_kwargs,
        )

    async def areact_agent(
        self,
        model: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode | None = None,
        **agent_kwargs: Any,
    ) -> Any:
        """Async :meth:`react_agent`."""
        from agent_gantry.integrations.frameworks.langgraph_live import (
            _acreate_gantry_react_agent,
        )

        return await _acreate_gantry_react_agent(
            model,
            self._gantry,
            **self._agent_kwargs(
                limit, score_threshold, namespaces, required, always_include, error_mode
            ),
            **agent_kwargs,
        )

    async def select_for_state(
        self,
        state: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        error_mode: ErrorMode | None = None,
    ) -> list[Any]:
        """Select tools for a LangGraph agent ``state`` once (no per-run state)."""
        from agent_gantry.integrations.frameworks.langgraph_live import (
            _select_tools_for_state,
        )

        return await _select_tools_for_state(
            self._gantry,
            state,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
            error_mode=self._mode(error_mode),
            error_formatter=self._error_formatter,
        )
