"""LangGraph native tool adapter for Agent-Gantry.

LangGraph does not define its own tool object: a LangGraph graph (e.g. a
``ToolNode`` or a prebuilt ReAct agent) consumes plain LangChain ``BaseTool``
objects — the same ``StructuredTool`` instances produced by the LangChain
adapter. So this module reuses the LangChain wrappers verbatim rather than
duplicating them.

Public entry point: :class:`LangGraphAdapter` (static slice + a deep per-turn
live ReAct agent that re-selects tools every model turn).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import DEFAULT_TOOL_LIMIT, BaseFrameworkAdapter
from agent_gantry.integrations.frameworks.langchain import (
    _for_langchain,
    _spec_to_langchain,
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.integrations.frameworks.base import ToolSpec

# A LangGraph node accepts LangChain BaseTool objects, so the per-spec wrapper
# is identical to the LangChain one.
_spec_to_langgraph = _spec_to_langchain


async def _for_langgraph(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` as LangChain ``BaseTool``s for a LangGraph node."""
    return await _for_langchain(gantry, query, limit=limit, **select_kwargs)


class LangGraphAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into LangGraph.

    Static slice (LangChain ``BaseTool`` objects for a ``ToolNode`` / prebuilt
    graph) plus a deep per-turn live ReAct agent that re-selects tools on every
    model turn::

        from agent_gantry.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(gantry)
        tools = await adapter.select("summarise the incident", limit=3)   # static
        agent = await adapter.areact_agent(chat_model, limit=5)           # live
    """

    live_tier = "per-turn"

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a LangChain ``StructuredTool``."""
        return _spec_to_langgraph(spec)

    def live(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **framework_kwargs: Any,
    ) -> Any:
        """Per-turn uniform entry point: delegates to :meth:`react_agent`.

        Requires ``model=<langchain_core.language_models.BaseChatModel>`` in
        ``framework_kwargs`` (LangGraph's per-turn hook is a middleware bound
        into the compiled agent, so there is no lower-level standalone hook to
        return — the compiled agent *is* the live object here). Returns the
        compiled LangGraph agent (a ``Pregel`` graph); call ``.ainvoke`` /
        ``.invoke`` on it directly — no further plumbing needed.
        ``required``/``always_include`` are re-applied on every model turn
        (see
        :meth:`~agent_gantry.integrations.frameworks.base.GantryToolset.select`).
        Any other ``framework_kwargs`` (``system_prompt``, ``checkpointer``,
        ``state_schema``, ``middleware``, …) are forwarded to
        ``create_agent``.
        """
        model = framework_kwargs.pop("model")
        return self.react_agent(
            model,
            limit=limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
            **framework_kwargs,
        )

    # -- deep per-turn (live) -------------------------------------------- #
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

        Resolves the tool superset on a worker thread; safe from a running loop
        but blocking. In an already-async context prefer :meth:`areact_agent`.
        """
        from agent_gantry.integrations.frameworks.langgraph_live import (
            _create_gantry_react_agent,
        )

        return _create_gantry_react_agent(
            model,
            self._gantry,
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
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
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
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
            limit=self._default_limit if limit is None else limit,
            score_threshold=score_threshold,
            namespaces=namespaces,
            required=required,
            always_include=always_include,
        )
