"""LangChain native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a LangChain
``StructuredTool`` — the native tool object LangChain agents introspect
(name / description / args schema) and invoke. The ``langchain-core`` import is
lazy so ``import agent_gantry`` never requires LangChain to be installed.

Public entry point: :class:`LangChainAdapter`.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Any

from agent_gantry.integrations.frameworks.base import BaseFrameworkAdapter, ToolSpec

_INSTALL_HINT = (
    "LangChain support requires `langchain-core`. "
    "Install it with `pip install langchain-core` (or `uv add langchain-core`)."
)


def _spec_to_langchain(spec: ToolSpec) -> Any:
    """Wrap a :class:`ToolSpec` as a LangChain ``StructuredTool``.

    The ``langchain_core`` import happens here, lazily, so callers without
    LangChain installed only hit the error when they actually export a tool.
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:  # pragma: no cover - exercised via stub
        raise ImportError(_INSTALL_HINT) from exc

    def _sync(**kwargs: Any) -> Any:
        return spec.invoke(**kwargs)

    return StructuredTool.from_function(
        func=_sync,
        coroutine=spec.callable_for_signature(),
        name=spec.name,
        description=spec.description,
        args_schema=spec.parameters,
    )


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

    LangChain's ``AgentExecutor`` / ``.bind_tools()`` fixes its tool list at
    construction and has no mid-run hook (that lives one layer up, in
    :class:`~agent_gantry.langgraph.LangGraphAdapter`), so :meth:`live` returns
    a bound async ``query -> list[StructuredTool]``: re-run it for each new
    top-level call's query and rebind the result.
    """

    live_tier = "per-call"
    _live_delegate = "_bound_select"

    @staticmethod
    def convert(spec: ToolSpec) -> Any:
        """Wrap a single :class:`ToolSpec` as a LangChain ``StructuredTool``."""
        return _spec_to_langchain(spec)

    def _bound_select(self, **select_kwargs: Any) -> Callable[[str], Awaitable[list[Any]]]:
        """The per-call live object: :meth:`select` bound to the selection keywords."""
        return functools.partial(self.select, **select_kwargs)
