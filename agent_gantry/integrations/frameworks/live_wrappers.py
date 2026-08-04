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
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        self._gantry = gantry
        self._role = role
        self._goal = goal
        self._backstory = backstory
        self._llm = llm
        self._limit = limit
        self._score_threshold = score_threshold
        self._namespaces = namespaces
        self._required = required
        self._always_include = always_include
        self._agent_kwargs = agent_kwargs

    async def select_tools(self, query: str) -> list[Any]:
        """Re-select this call's CrewAI tools for ``query``."""
        return await _for_crewai(
            self._gantry,
            query,
            limit=self._limit,
            score_threshold=self._score_threshold,
            namespaces=self._namespaces,
            required=self._required,
            always_include=self._always_include,
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
                "CrewAI support requires `crewai`. Install it with `pip install crewai`."
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
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        self._gantry = gantry
        self._model = model
        self._limit = limit
        self._score_threshold = score_threshold
        self._namespaces = namespaces
        self._required = required
        self._always_include = always_include
        self._agent_kwargs = agent_kwargs

    async def select_tools(self, query: str) -> list[Any]:
        """Re-select this call's Agno ``Function`` tools for ``query``."""
        return await _for_agno(
            self._gantry,
            query,
            limit=self._limit,
            score_threshold=self._score_threshold,
            namespaces=self._namespaces,
            required=self._required,
            always_include=self._always_include,
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
    """Rebuild a fresh Haystack tool-execution component per call.

    Haystack fixes a component's tools at construction, so this builder
    constructs a new one for every call via :meth:`build`, each time wiring in
    the tools Gantry selects for that call's query.

    What :meth:`build` returns depends on the installed haystack-ai version:

    * **haystack 2.x** — a ``ToolInvoker``, as before.
    * **haystack >= 3.0** — ``ToolInvoker`` was removed (the ``Agent``
      component owns tool execution). If a ``chat_generator`` was supplied in
      the builder kwargs, :meth:`build` constructs a per-call
      ``haystack.components.agents.Agent`` with the selected tools; without
      one it raises a clear error pointing at the alternatives.

    Obtain one via ``HaystackAdapter(gantry).tool_invoker_builder(...)``.

    Args:
        gantry: The :class:`~agent_gantry.core.gantry.AgentGantry` to select from.
        limit: Max tools to surface per call. Defaults to ``5``.
        score_threshold: Minimum semantic relevance score. Defaults to ``0.0``.
        **invoker_kwargs: Extra kwargs forwarded to ``ToolInvoker``
            (haystack 2.x) or ``Agent`` (haystack >= 3, where a
            ``chat_generator=...`` entry is required).
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
        **invoker_kwargs: Any,
    ) -> None:
        self._gantry = gantry
        self._limit = limit
        self._score_threshold = score_threshold
        self._namespaces = namespaces
        self._required = required
        self._always_include = always_include
        self._invoker_kwargs = invoker_kwargs

    async def select_tools(self, query: str) -> list[Any]:
        """Re-select this call's Haystack ``Tool`` list for ``query``."""
        return await _for_haystack(
            self._gantry,
            query,
            limit=self._limit,
            score_threshold=self._score_threshold,
            namespaces=self._namespaces,
            required=self._required,
            always_include=self._always_include,
        )

    async def build(self, query: str) -> Any:
        """Build a fresh tool-execution component for ``query``.

        Returns a ``ToolInvoker`` on haystack 2.x, or an ``Agent`` on
        haystack >= 3.0 (which removed ``ToolInvoker``) when the builder was
        given a ``chat_generator``.

        Raises:
            ImportError: If ``haystack-ai`` is not installed.
            RuntimeError: On haystack >= 3.0 when no ``chat_generator`` was
                supplied to the builder.
        """
        try:
            from haystack.components.tools import ToolInvoker
        except ImportError as exc:
            # Distinguish "haystack absent" from "haystack >= 3.0, where
            # ToolInvoker was removed" — otherwise the error tells users to
            # install a package they already have.
            try:
                import haystack  # noqa: F401
            except ImportError:  # pragma: no cover - exercised via importorskip
                raise ImportError(
                    "Haystack support requires `haystack-ai`. "
                    "Install it with `pip install haystack-ai`."
                ) from exc
            return await self._build_haystack3_agent(query, exc)

        tools = await self.select_tools(query)
        return ToolInvoker(tools=tools, **self._invoker_kwargs)

    async def _build_haystack3_agent(self, query: str, cause: ImportError) -> Any:
        """haystack >= 3.0 path: build a per-call ``Agent`` with fresh tools."""
        from haystack.components.agents import Agent

        if "chat_generator" not in self._invoker_kwargs:
            raise RuntimeError(
                "haystack-ai >= 3.0 removed ToolInvoker; the Agent component "
                "now owns tool execution. Pass chat_generator=... to "
                "tool_invoker_builder(...) so build() can construct a "
                "per-call haystack Agent with the selected tools, or use "
                "HaystackAdapter.live_tools() and wire the tools into your "
                "own haystack.components.agents.Agent."
            ) from cause

        tools = await self.select_tools(query)
        return Agent(tools=tools, **self._invoker_kwargs)


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
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        self._gantry = gantry
        self._model = model
        self._agent_cls = agent_cls
        self._limit = limit
        self._score_threshold = score_threshold
        self._namespaces = namespaces
        self._required = required
        self._always_include = always_include
        self._agent_kwargs = agent_kwargs

    async def select_tools(self, query: str) -> list[Any]:
        """Re-select this call's smolagents ``Tool`` objects for ``query``."""
        return await _for_smolagents(
            self._gantry,
            query,
            limit=self._limit,
            score_threshold=self._score_threshold,
            namespaces=self._namespaces,
            required=self._required,
            always_include=self._always_include,
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
