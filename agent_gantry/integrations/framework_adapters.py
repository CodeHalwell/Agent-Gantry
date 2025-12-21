"""
Thin, dependency-free adapters for popular agent frameworks.

These helpers translate Agent-Gantry retrieval results into the schema shapes
expected by common frameworks (LangGraph, Semantic Kernel, CrewAI, Google ADK,
and Strands) while preserving dynamic top-k surfacing.
"""

from __future__ import annotations

from typing import Literal

from agent_gantry.core.gantry import AgentGantry
from agent_gantry.schema.query import ConversationContext, ToolQuery

FrameworkName = Literal["langgraph", "semantic-kernel", "crew_ai", "google_adk", "strands"]


async def fetch_framework_tools(
    gantry: AgentGantry,
    query: str,
    *,
    framework: FrameworkName,
    limit: int = 5,
    score_threshold: float = 0.5,
) -> list[dict]:
    """
    Retrieve top-k tools and emit the schema shape expected by a framework.

    All adapters reuse Agent-Gantry's routing semantics; only the output format
    varies to keep integrations lightweight and optional.
    """
    result = await gantry.retrieve(
        ToolQuery(
            context=ConversationContext(query=query),
            limit=limit,
            score_threshold=score_threshold,
        )
    )

    # LangGraph, Semantic Kernel, CrewAI, Google ADK, and Strands all accept
    # OpenAI-style tool/function schemas today, so default to that shape.
    if framework in {"langgraph", "semantic-kernel", "crew_ai", "google_adk", "strands"}:
        return result.to_openai_tools()

    # Fallback to OpenAI schema for unknown/experimental frameworks.
    return result.to_openai_tools()
