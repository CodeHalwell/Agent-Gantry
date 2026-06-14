"""LangGraph native tool adapter for Agent-Gantry.

LangGraph does not define its own tool object: a LangGraph graph (e.g. a
``ToolNode`` or a prebuilt ReAct agent) consumes plain LangChain ``BaseTool``
objects — the same ``StructuredTool`` instances produced by the LangChain
adapter. So this module reuses the LangChain wrappers verbatim rather than
duplicating them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_gantry.integrations.frameworks.langchain import (
    for_langchain,
    spec_to_langchain,
)

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry

# A LangGraph node accepts LangChain BaseTool objects, so the per-spec wrapper
# is identical to the LangChain one.
spec_to_langgraph = spec_to_langchain


async def for_langgraph(
    gantry: AgentGantry,
    query: str,
    *,
    limit: int = 3,
    **select_kwargs: Any,
) -> list[Any]:
    """Select tools for ``query`` as LangChain ``BaseTool``s for a LangGraph node."""
    return await for_langchain(gantry, query, limit=limit, **select_kwargs)
