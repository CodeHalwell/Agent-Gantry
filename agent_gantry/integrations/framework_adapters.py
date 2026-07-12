"""
Thin, dependency-free adapters for popular agent frameworks.

These helpers translate Agent-Gantry retrieval results into the schema shapes
expected by common frameworks while preserving dynamic top-k surfacing. They
are schema-only: no third-party framework is ever imported, and the return
value is always a plain list of dicts.

.. note::
    **Prefer the native adapters for anything beyond raw schemas.** This
    module predates ``agent_gantry.integrations.frameworks`` (the per-framework
    ``<Framework>Adapter`` classes exported as ``agent_gantry.<framework>``,
    e.g. ``agent_gantry.langchain.LangChainAdapter``,
    ``agent_gantry.crewai.CrewAIAdapter``), which build real native tool
    objects (LangChain ``StructuredTool``, CrewAI ``BaseTool``, ...) that route
    execution through ``gantry.execute``, plus deep per-turn live selection
    where the framework supports it. :func:`fetch_framework_tools` only emits
    OpenAI-style schema dicts for callers that want to do their own tool
    wiring — reach for the native adapters first.
"""

from __future__ import annotations

from typing import Any, Literal

from agent_gantry.core.gantry import AgentGantry
from agent_gantry.schema.query import ConversationContext, ToolQuery

# Canonical framework names, matching the native adapter module names under
# ``agent_gantry.integrations.frameworks`` / ``agent_gantry.<framework>``.
FrameworkName = Literal[
    "langchain",
    "langgraph",
    "llamaindex",
    "crewai",
    "autogen",
    "semantic_kernel",
    "google_adk",
    "agno",
    "haystack",
    "pydantic_ai",
    "openai_agents",
    "smolagents",
    "agent_framework",
    "strands",
    # Legacy spellings accepted for backwards compatibility (see #101) and
    # normalized internally to the canonical name above.
    "crew_ai",
    "semantic-kernel",
]

_SUPPORTED_FRAMEWORKS: frozenset[str] = frozenset(
    {
        "langchain",
        "langgraph",
        "llamaindex",
        "crewai",
        "autogen",
        "semantic_kernel",
        "google_adk",
        "agno",
        "haystack",
        "pydantic_ai",
        "openai_agents",
        "smolagents",
        "agent_framework",
        "strands",
    }
)

# Legacy / alternate spellings kept for backwards compatibility, mapped to the
# canonical name they normalize to. See GitHub issue #101 — this module's
# ``FrameworkName`` used to cover only a handful of frameworks and disagreed
# with the native per-framework adapters' names.
_LEGACY_ALIASES: dict[str, str] = {
    "crew_ai": "crewai",
    "semantic-kernel": "semantic_kernel",
}


def _canonical_framework(framework: str) -> str:
    """Normalize a legacy framework spelling to its canonical name."""
    return _LEGACY_ALIASES.get(framework, framework)


async def fetch_framework_tools(
    gantry: AgentGantry,
    query: str,
    *,
    framework: FrameworkName,
    limit: int = 5,
    score_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k tools and emit the schema shape expected by a framework.

    All adapters reuse Agent-Gantry's routing semantics; only the output format
    varies to keep integrations lightweight and optional. Today, all supported
    frameworks accept OpenAI-style tool/function schemas, so that shape is
    returned; the framework parameter is validated to fail fast and reserved
    for future per-framework tweaks.

    ``framework`` accepts any canonical native-adapter name (``langchain``,
    ``langgraph``, ``llamaindex``, ``crewai``, ``autogen``, ``semantic_kernel``,
    ``google_adk``, ``agno``, ``haystack``, ``pydantic_ai``, ``openai_agents``,
    ``smolagents``, ``agent_framework``, ``strands``) plus the legacy spellings
    ``crew_ai`` and ``semantic-kernel``, which are normalized internally to
    ``crewai`` and ``semantic_kernel`` respectively.

    For Microsoft Agent Framework, use the ``agent_framework`` dialect which
    produces OpenAI-compatible schemas that AF's tool infrastructure expects.
    For higher-level integration (wrapping tools as Python callables for AF
    agents), see ``agent_gantry.integrations.agent_framework_bridge``.
    """
    canonical = _canonical_framework(framework)
    if canonical not in _SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported framework: {framework}")

    result = await gantry.retrieve(
        ToolQuery(
            context=ConversationContext(query=query),
            limit=limit,
            score_threshold=score_threshold,
        )
    )

    if canonical == "agent_framework":
        return result.to_dialect("agent_framework")

    # Every other supported framework accepts OpenAI-style tool/function
    # schemas today, so default to that shape.
    return result.to_openai_tools()
