"""LangChain native tool adapter for Agent-Gantry.

Selects a relevant slice of Gantry tools and wraps each as a LangChain
``StructuredTool`` — the native tool object LangChain agents introspect
(name / description / args schema) and invoke. The ``langchain-core`` import is
lazy so ``import agent_gantry`` never requires LangChain to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.base import GantryToolset, ToolSpec

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry


def spec_to_langchain(spec: ToolSpec) -> Any:
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


async def for_langchain(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` and return them as LangChain ``StructuredTool``s."""
    specs = await GantryToolset(gantry).select(query, limit=limit, **select_kwargs)
    return [spec_to_langchain(s) for s in specs]
