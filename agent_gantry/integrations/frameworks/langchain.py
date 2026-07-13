"""LangChain native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a LangChain
``StructuredTool`` — the native tool object LangChain agents introspect
(name / description / args schema) and invoke. The ``langchain-core`` import is
lazy so ``import agent_gantry`` never requires LangChain to be installed.

Public entry point: :class:`LangChainAdapter`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import (
    DEFAULT_TOOL_LIMIT,
    BaseFrameworkAdapter,
    GantryToolset,
    ToolSpec,
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def _spec_to_langchain(spec: ToolSpec) -> Any:
    """Wrap a :class:`ToolSpec` as a LangChain ``StructuredTool``.

    The ``langchain_core`` import happens here, lazily, so callers without
    LangChain installed only hit the error when they actually export a tool.
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(
            "LangChain support requires `langchain-core`. "
            "Install it with `pip install langchain-core`."
        ) from exc

    def _sync(**kwargs: Any) -> Any:
        return spec.invoke(**kwargs)

    return StructuredTool.from_function(
        func=_sync,
        coroutine=spec.callable_for_signature(),
        name=spec.name,
        description=spec.description,
        args_schema=spec.parameters,
    )


async def _for_langchain(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = DEFAULT_TOOL_LIMIT,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as LangChain ``StructuredTool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [_spec_to_langchain(s) for s in specs]


class LangChainAdapter(BaseFrameworkAdapter):
    """Route Gantry-selected tools into LangChain.

    Construct with a gantry, then :meth:`select` a relevant slice of tools as
    native LangChain ``StructuredTool`` objects (each call still routed through
    ``gantry.execute`` so retries, timeouts, circuit breakers, and the security
    policy apply)::

        from agent_gantry.langchain import LangChainAdapter

        adapter = LangChainAdapter(gantry)
        tools = await adapter.select("email the quarterly report", limit=3)
        llm = ChatOpenAI(model="gpt-5.5").bind_tools(tools)
    """

    live_tier = "per-call"

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a LangChain ``StructuredTool``."""
        return _spec_to_langchain(spec)

    def live(
        self,
        *,
        limit: int | None = None,
        score_threshold: float = 0.0,
        namespaces: list[str] | None = None,
        required: list[str] | None = None,
        always_include: list[str] | None = None,
        **framework_kwargs: Any,
    ) -> Callable[[str], Awaitable[list[Any]]]:
        """Per-call uniform entry point: a bound alias of :meth:`select`.

        LangChain's ``AgentExecutor`` / ``.bind_tools()`` fixes its tool list
        at construction with no framework-native per-turn or per-call hook of
        its own — that hook lives one layer up, in :class:`LangGraphAdapter`
        (``live_tier`` = ``"per-turn"``), which plugs Gantry into
        ``langchain.agents.create_agent`` middleware. Without LangGraph, the
        deepest re-selection LangChain permits is: re-run selection for each
        new top-level call's query and rebind the result to a fresh
        ``AgentExecutor`` / ``.bind_tools()`` call.

        Returns a bound async callable ``query -> list[StructuredTool]`` —
        call it with the new call's query before each such rebuild; it *is*
        :meth:`select`, aliased so every adapter's ``live()`` has the same
        uniform ``limit``/``score_threshold``/``namespaces``/``required``/
        ``always_include`` signature. ``framework_kwargs`` are forwarded to
        :meth:`select` verbatim (e.g. ``tools_already_used``).
        """
        eff_limit = self._default_limit if limit is None else limit

        async def _select(query: str) -> list[Any]:
            return await self.select(
                query,
                limit=eff_limit,
                score_threshold=score_threshold,
                namespaces=namespaces,
                required=required,
                always_include=always_include,
                **framework_kwargs,
            )

        return _select
