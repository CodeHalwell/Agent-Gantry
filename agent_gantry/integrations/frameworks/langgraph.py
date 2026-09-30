"""LangGraph native tool adapter for Agent-Gantry.

LangGraph does not define its own tool object: a LangGraph graph (e.g. a
``ToolNode`` or a prebuilt ReAct agent) consumes plain LangChain ``BaseTool``
objects — the same ``StructuredTool`` instances produced by the LangChain
adapter, so this module reuses that wrapper.

Public entry point: :class:`LangGraphAdapter` (static slice + a deep per-turn
live ReAct agent that re-selects tools every model turn).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import BaseFrameworkAdapter
from agent_gantry.integrations.frameworks.langchain import _spec_to_langchain

if TYPE_CHECKING:
    from agent_gantry.integrations.frameworks.base import ToolSpec


class LangGraphAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into LangGraph.

    Static slice (LangChain ``BaseTool`` objects for a ``ToolNode`` / prebuilt
    graph) plus a deep per-turn live ReAct agent that re-selects tools on every
    model turn::

        from agent_gantry.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(gantry)
        tools = await adapter.select("summarise the incident", limit=3)   # static
        agent = await adapter.areact_agent(chat_model, limit=5)           # live

    :meth:`live` requires ``model=<BaseChatModel>`` and returns the compiled
    agent from :meth:`react_agent`; any other keyword (``system_prompt``,
    ``checkpointer``, ``middleware``, …) is forwarded to ``create_agent``.
    """

    live_tier = "per-turn"
    _live_delegate = "react_agent"
    _live_required_kwargs = ("model",)

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a LangChain ``StructuredTool``."""
        return _spec_to_langchain(spec)

    def react_agent(
        self,
        model: Any,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **agent_kwargs: Any,
    ) -> Any:
        """Build a ReAct agent that re-selects tools every model turn (sync).

        Resolves the tool superset on the sync bridge; safe from a running loop
        but blocking. In an already-async context prefer :meth:`areact_agent`.
        """
        from agent_gantry.integrations.frameworks.langgraph_live import (
            _create_gantry_react_agent,
        )

        return _create_gantry_react_agent(
            model,
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
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
        **agent_kwargs: Any,
    ) -> Any:
        """Async-native :meth:`react_agent` (awaits the tool-superset enumeration)."""
        from agent_gantry.integrations.frameworks.langgraph_live import (
            _acreate_gantry_react_agent,
        )

        return await _acreate_gantry_react_agent(
            model,
            self._gantry,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
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
    ) -> list[Any]:
        """Re-select tools for a LangGraph agent ``state`` (per-turn primitive)."""
        from agent_gantry.integrations.frameworks.langgraph_live import (
            _select_tools_for_state,
        )

        return await _select_tools_for_state(
            self._gantry,
            state,
            **self._selection_kwargs(limit, score_threshold, namespaces, required, always_include),
        )
