"""Per-run "live" builders for frameworks that fix an agent's tool list at construction.

CrewAI, Agno and Haystack expose no per-turn hook to re-advertise tools
mid-run (compare LlamaIndex's ``tool_retriever`` in ``llamaindex_live``), so
the deepest re-selection they permit is per top-level call: each builder
re-runs dotfetch selection for the query of *each* new call and builds a fresh
agent / tool component for it. Within a single run the tool surface stays
fixed.

Obtain the builders via the adapters' ``agent_builder`` /
``tool_invoker_builder`` methods (:class:`~dotfetch.crewai.CrewAIAdapter`,
:class:`~dotfetch.agno.AgnoAdapter`,
:class:`~dotfetch.haystack.HaystackAdapter`). Framework imports are lazy,
so ``import dotfetch`` never requires any of these frameworks; a missing
one raises ``ImportError`` naming the package to install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from dotfetch.integrations.frameworks.agno import _INSTALL_HINT as _AGNO_HINT
from dotfetch.integrations.frameworks.agno import AgnoAdapter
from dotfetch.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    check_query_bounds,
)
from dotfetch.integrations.frameworks.crewai import _INSTALL_HINT as _CREWAI_HINT
from dotfetch.integrations.frameworks.crewai import CrewAIAdapter
from dotfetch.integrations.frameworks.haystack import _INSTALL_HINT as _HAYSTACK_HINT
from dotfetch.integrations.frameworks.haystack import HaystackAdapter

if TYPE_CHECKING:
    from dotfetch.core.fetcher import DotFetch


class _PerCallBuilder:
    """Re-select tools for each call's query, then build a fresh framework object.

    Subclasses set :attr:`_adapter_cls` and implement :meth:`_construct`;
    ``framework_kwargs`` are forwarded to the framework object on every build.
    """

    _adapter_cls: ClassVar[type[BaseFrameworkAdapter]]

    def __init__(
        self,
        fetcher: DotFetch,
        *,
        limit: int = DEFAULT_TOOL_LIMIT,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **framework_kwargs: Any,
    ) -> None:
        check_query_bounds(limit=limit, score_threshold=score_threshold, owner=type(self).__name__)
        self._adapter = self._adapter_cls(fetcher)
        self._select_kwargs: dict[str, Any] = {
            "limit": limit,
            "score_threshold": score_threshold,
            "namespaces": namespaces,
            "required": required,
            "always_include": always_include,
        }
        self._framework_kwargs = framework_kwargs

    async def select_tools(self, query: str) -> list[Any]:
        """Re-select this call's native tools for ``query``."""
        return await self._adapter.select(query, **self._select_kwargs)

    async def build(self, query: str) -> Any:
        """Build a fresh framework object whose tools are selected for ``query``.

        Raises:
            ImportError: If the framework is not installed.
        """
        return self._construct(await self.select_tools(query))

    def _construct(self, tools: list[Any]) -> Any:
        raise NotImplementedError


class DotFetchLiveCrewAgent(_PerCallBuilder):
    """Rebuild a fresh ``crewai.Agent`` per call, with tools re-selected by dotfetch.

    Args:
        fetcher: The :class:`~dotfetch.core.fetcher.DotFetch` to select from.
        role/goal/backstory: Standard CrewAI agent identity fields.
        llm: Optional LLM passed straight to ``crewai.Agent``.
        limit: Max tools to surface per call. Defaults to ``DEFAULT_TOOL_LIMIT``.
        score_threshold: Minimum semantic relevance score. Defaults to ``0.0``.
        **agent_kwargs: Extra kwargs forwarded to ``crewai.Agent``.
    """

    _adapter_cls = CrewAIAdapter

    def __init__(
        self,
        fetcher: DotFetch,
        *,
        role: str = "dotfetch Agent",
        goal: str = "Help the user by selecting and using the right tools.",
        backstory: str = "An agent whose tools are chosen by dotfetch per task.",
        llm: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(fetcher, **kwargs)
        self._framework_kwargs.update(role=role, goal=goal, backstory=backstory)
        if llm is not None:
            self._framework_kwargs["llm"] = llm

    def _construct(self, tools: list[Any]) -> Any:
        try:
            from crewai import Agent
        except ImportError as exc:  # pragma: no cover - exercised via importorskip
            raise ImportError(_CREWAI_HINT) from exc
        return Agent(tools=tools, **self._framework_kwargs)


class DotFetchLiveAgnoAgent(_PerCallBuilder):
    """Rebuild a fresh ``agno.agent.Agent`` per call, tools re-selected by dotfetch.

    Args:
        fetcher: The :class:`~dotfetch.core.fetcher.DotFetch` to select from.
        model: Optional Agno model passed straight to ``Agent``.
        limit: Max tools to surface per call. Defaults to ``DEFAULT_TOOL_LIMIT``.
        score_threshold: Minimum semantic relevance score. Defaults to ``0.0``.
        **agent_kwargs: Extra kwargs forwarded to ``agno.agent.Agent``.
    """

    _adapter_cls = AgnoAdapter

    def __init__(self, fetcher: DotFetch, *, model: Any | None = None, **kwargs: Any) -> None:
        super().__init__(fetcher, **kwargs)
        if model is not None:
            self._framework_kwargs["model"] = model

    def _construct(self, tools: list[Any]) -> Any:
        try:
            from agno.agent import Agent
        except ImportError as exc:  # pragma: no cover - exercised via importorskip
            raise ImportError(_AGNO_HINT) from exc
        return Agent(tools=tools, **self._framework_kwargs)


class DotFetchLiveHaystackToolInvoker(_PerCallBuilder):
    """Rebuild a fresh Haystack tool-execution component per call.

    :meth:`build` returns a ``ToolInvoker`` on haystack 2.x. haystack >= 3.0
    removed ``ToolInvoker`` (the ``Agent`` component owns tool execution): there
    it returns a per-run ``haystack.components.agents.Agent`` when a
    ``chat_generator`` was supplied in the builder kwargs, and raises
    ``RuntimeError`` otherwise.

    Args:
        fetcher: The :class:`~dotfetch.core.fetcher.DotFetch` to select from.
        limit: Max tools to surface per call. Defaults to ``DEFAULT_TOOL_LIMIT``.
        score_threshold: Minimum semantic relevance score. Defaults to ``0.0``.
        **invoker_kwargs: Extra kwargs forwarded to ``ToolInvoker`` (haystack
            2.x) or ``Agent`` (haystack >= 3, where ``chat_generator=...`` is
            required).
    """

    _adapter_cls = HaystackAdapter

    def _construct(self, tools: list[Any]) -> Any:
        try:
            from haystack.components.tools import ToolInvoker
        except ImportError as exc:
            try:
                import haystack  # noqa: F401
            except ImportError:  # pragma: no cover - exercised via importorskip
                raise ImportError(_HAYSTACK_HINT) from exc
            from haystack.components.agents import Agent

            if "chat_generator" not in self._framework_kwargs:
                raise RuntimeError(
                    "haystack-ai >= 3.0 removed ToolInvoker; the Agent component "
                    "now owns tool execution. Pass chat_generator=... to "
                    "tool_invoker_builder(...) so build() can construct a "
                    "per-run haystack Agent with the selected tools, or use "
                    "HaystackAdapter.live_tools() and wire the tools into your "
                    "own haystack.components.agents.Agent."
                ) from exc
            return Agent(tools=tools, **self._framework_kwargs)
        return ToolInvoker(tools=tools, **self._framework_kwargs)


__all__ = [
    "DotFetchLiveCrewAgent",
    "DotFetchLiveAgnoAgent",
    "DotFetchLiveHaystackToolInvoker",
]
