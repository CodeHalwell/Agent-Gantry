"""Best-effort "live" tool wrappers for fixed-tool-set frameworks.

CrewAI, Agno, Haystack and Smolagents all fix an agent's tool list **at agent
construction time**: none of them exposes a native per-turn / per-reasoning-step
hook to re-advertise tools mid-run (the way LlamaIndex's ``tool_retriever`` or
AutoGen's ``Workbench`` do — see ``llamaindex_live`` / ``autogen_live`` for those
genuinely per-turn integrations). Once you hand one of these frameworks a list of
tools and build the agent, that list is frozen for the whole run.

So "as deep as the framework allows" here is **per top-level call**, not
per-intra-run turn. The builders below re-run Gantry selection for the query of
*each* new call (each CrewAI task, each Agno run, each Haystack invocation, each
Smolagents run), convert the freshly selected tools to the framework's native
objects, and (re)build the agent / tool set for that call. Between calls the tool
surface tracks the new query; *within* a single agent run it stays fixed — that
is the deepest these frameworks permit.

These builders are returned by the per-call ``agent_builder`` /
``tool_invoker_builder`` methods on the framework adapters
(:class:`~agent_gantry.crewai.CrewAIAdapter`,
:class:`~agent_gantry.agno.AgnoAdapter`,
:class:`~agent_gantry.haystack.HaystackAdapter`,
:class:`~agent_gantry.smolagents.SmolagentsAdapter`). All selection and
native-tool conversion is delegated to the existing ``_for_crewai`` / ``_for_agno``
/ ``_for_haystack`` / ``_for_smolagents`` adapters; nothing here re-implements
either. Each framework import is lazy (performed inside the helper/class), so
``import agent_gantry`` never requires any of these frameworks to be installed; a
missing one raises ``ImportError`` with the right ``pip install`` hint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.agno import _for_agno
from agent_gantry.integrations.frameworks.base import DEFAULT_TOOL_LIMIT
from agent_gantry.integrations.frameworks.crewai import _for_crewai
from agent_gantry.integrations.frameworks.haystack import _for_haystack
from agent_gantry.integrations.frameworks.smolagents import _for_smolagents

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


# --------------------------------------------------------------------------- #
# CrewAI
# --------------------------------------------------------------------------- #
class GantryLiveCrewAgent:
    """Rebuild a fresh ``crewai.Agent`` per call, with tools re-selected by Gantry.

    CrewAI freezes an agent's tools at construction, so this builder constructs a
    *new* ``crewai.Agent`` for every call via :meth:`build`, each time wiring in
    the tools Gantry selects for that call's query. The role / goal / backstory /
    llm (and any extra ``crewai.Agent`` kwargs) are configured once on the
    constructor and reused for every rebuild.

    Obtain one via ``CrewAIAdapter(gantry).agent_builder(...)``.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` to select from.
        role/goal/backstory: Standard CrewAI agent identity fields.
        llm: Optional LLM passed straight to ``crewai.Agent``.
        limit: Max tools to surface per call. Defaults to ``5``.
        score_threshold: Minimum semantic relevance score. Defaults to ``0.0``.
        **agent_kwargs: Extra kwargs forwarded to ``crewai.Agent``.
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        role: str = "Gantry Agent",
        goal: str = "Help the user by selecting and using the right tools.",
        backstory: str = "An agent whose tools are chosen by Agent-Gantry per task.",
        llm: Any | None = None,
        limit: int = DEFAULT_TOOL_LIMIT,
        score_threshold: float = 0.0,
        **agent_kwargs: Any,
    ) -> None:
        self._gantry = gantry
        self._role = role
        self._goal = goal
        self._backstory = backstory
        self._llm = llm
        self._limit = limit
        self._score_threshold = score_threshold
        self._agent_kwargs = agent_kwargs

    async def select_tools(self, query: str) -> list[Any]:
        """Re-select this call's CrewAI tools for ``query``."""
        return await _for_crewai(
            self._gantry,
            query,
            limit=self._limit,
            score_threshold=self._score_threshold,
        )

    async def build(self, query: str) -> Any:
        """Build a fresh ``crewai.Agent`` whose tools are selected for ``query``.

        Raises:
            ImportError: If ``crewai`` is not installed.
        """
        try:
            from crewai import Agent
        except ImportError as exc:  # pragma: no cover - exercised via importorskip
            raise ImportError(
                "CrewAI support requires `crewai`. "
                "Install it with `pip install crewai`."
            ) from exc

        tools = await self.select_tools(query)
        kwargs: dict[str, Any] = {
            "role": self._role,
            "goal": self._goal,
            "backstory": self._backstory,
            "tools": tools,
            **self._agent_kwargs,
        }
        if self._llm is not None:
            kwargs["llm"] = self._llm
        return Agent(**kwargs)


# --------------------------------------------------------------------------- #
# Agno
# --------------------------------------------------------------------------- #
class GantryLiveAgnoAgent:
    """Rebuild a fresh ``agno.agent.Agent`` per call, tools re-selected by Gantry.

    Agno fixes an agent's tools at construction, so this builder constructs a new
    ``agno.agent.Agent`` for every call via :meth:`build`, each time wiring in the
    tools Gantry selects for that call's query. The model (and any extra Agno
    ``Agent`` kwargs) are configured once on the constructor.

    Obtain one via ``AgnoAdapter(gantry).agent_builder(...)``.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` to select from.
        model: Optional Agno model passed straight to ``Agent``.
        limit: Max tools to surface per call. Defaults to ``5``.
        score_threshold: Minimum semantic relevance score. Defaults to ``0.0``.
        **agent_kwargs: Extra kwargs forwarded to ``agno.agent.Agent``.
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        model: Any | None = None,
        limit: int = DEFAULT_TOOL_LIMIT,
        score_threshold: float = 0.0,
        **agent_kwargs: Any,
    ) -> None:
        self._gantry = gantry
        self._model = model
        self._limit = limit
        self._score_threshold = score_threshold
        self._agent_kwargs = agent_kwargs

    async def select_tools(self, query: str) -> list[Any]:
        """Re-select this call's Agno ``Function`` tools for ``query``."""
        return await _for_agno(
            self._gantry,
            query,
            limit=self._limit,
            score_threshold=self._score_threshold,
        )

    async def build(self, query: str) -> Any:
        """Build a fresh ``agno.agent.Agent`` whose tools are selected for ``query``.

        Raises:
            ImportError: If ``agno`` is not installed.
        """
        try:
            from agno.agent import Agent
        except ImportError as exc:  # pragma: no cover - exercised via importorskip
            raise ImportError(
                "Agno support requires `agno`. Install it with `pip install agno`."
            ) from exc

        tools = await self.select_tools(query)
        kwargs: dict[str, Any] = {"tools": tools, **self._agent_kwargs}
        if self._model is not None:
            kwargs["model"] = self._model
        return Agent(**kwargs)


# --------------------------------------------------------------------------- #
# Haystack
# --------------------------------------------------------------------------- #
class GantryLiveHaystackToolInvoker:
    """Rebuild a fresh Haystack ``ToolInvoker`` per call with re-selected tools.

    Haystack fixes a ``ToolInvoker``'s tools at construction, so this builder
    constructs a new ``ToolInvoker`` for every call via :meth:`build`, each time
    wiring in the tools Gantry selects for that call's query.

    Obtain one via ``HaystackAdapter(gantry).tool_invoker_builder(...)``.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` to select from.
        limit: Max tools to surface per call. Defaults to ``5``.
        score_threshold: Minimum semantic relevance score. Defaults to ``0.0``.
        **invoker_kwargs: Extra kwargs forwarded to ``ToolInvoker``.
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        limit: int = DEFAULT_TOOL_LIMIT,
        score_threshold: float = 0.0,
        **invoker_kwargs: Any,
    ) -> None:
        self._gantry = gantry
        self._limit = limit
        self._score_threshold = score_threshold
        self._invoker_kwargs = invoker_kwargs

    async def select_tools(self, query: str) -> list[Any]:
        """Re-select this call's Haystack ``Tool`` list for ``query``."""
        return await _for_haystack(
            self._gantry,
            query,
            limit=self._limit,
            score_threshold=self._score_threshold,
        )

    async def build(self, query: str) -> Any:
        """Build a fresh ``ToolInvoker`` whose tools are selected for ``query``.

        Raises:
            ImportError: If ``haystack-ai`` is not installed.
        """
        try:
            from haystack.components.tools import ToolInvoker
        except ImportError as exc:  # pragma: no cover - exercised via importorskip
            raise ImportError(
                "Haystack support requires `haystack-ai`. "
                "Install it with `pip install haystack-ai`."
            ) from exc

        tools = await self.select_tools(query)
        return ToolInvoker(tools=tools, **self._invoker_kwargs)


# --------------------------------------------------------------------------- #
# Smolagents
# --------------------------------------------------------------------------- #
class GantryLiveSmolAgent:
    """Rebuild a fresh smolagents agent per call, tools re-selected by Gantry.

    Smolagents fixes an agent's tools at construction, so this builder constructs
    a new agent (``ToolCallingAgent`` by default) for every call via
    :meth:`build`, each time wiring in the tools Gantry selects for that call's
    query. The model (and any extra agent kwargs) are configured once on the
    constructor.

    Obtain one via ``SmolagentsAdapter(gantry).agent_builder(...)``.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` to select from.
        model: The smolagents model passed straight to the agent.
        agent_cls: The smolagents agent class to build. Defaults to
            ``smolagents.ToolCallingAgent``.
        limit: Max tools to surface per call. Defaults to ``5``.
        score_threshold: Minimum semantic relevance score. Defaults to ``0.0``.
        **agent_kwargs: Extra kwargs forwarded to the agent class.
    """

    def __init__(
        self,
        gantry: AgentGantry,
        *,
        model: Any | None = None,
        agent_cls: Any | None = None,
        limit: int = DEFAULT_TOOL_LIMIT,
        score_threshold: float = 0.0,
        **agent_kwargs: Any,
    ) -> None:
        self._gantry = gantry
        self._model = model
        self._agent_cls = agent_cls
        self._limit = limit
        self._score_threshold = score_threshold
        self._agent_kwargs = agent_kwargs

    async def select_tools(self, query: str) -> list[Any]:
        """Re-select this call's smolagents ``Tool`` objects for ``query``."""
        return await _for_smolagents(
            self._gantry,
            query,
            limit=self._limit,
            score_threshold=self._score_threshold,
        )

    async def build(self, query: str) -> Any:
        """Build a fresh smolagents agent whose tools are selected for ``query``.

        Raises:
            ImportError: If ``smolagents`` is not installed.
        """
        try:
            from smolagents import ToolCallingAgent
        except ImportError as exc:  # pragma: no cover - exercised via importorskip
            raise ImportError(
                "Smolagents support requires `smolagents`. "
                "Install it with `pip install smolagents`."
            ) from exc

        agent_cls = self._agent_cls or ToolCallingAgent
        tools = await self.select_tools(query)
        kwargs: dict[str, Any] = {"tools": tools, **self._agent_kwargs}
        if self._model is not None:
            kwargs["model"] = self._model
        return agent_cls(**kwargs)


__all__ = [
    "GantryLiveCrewAgent",
    "GantryLiveAgnoAgent",
    "GantryLiveHaystackToolInvoker",
    "GantryLiveSmolAgent",
]
