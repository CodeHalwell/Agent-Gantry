"""Discover and construct native adapters without importing third-party SDKs."""

from __future__ import annotations

from difflib import get_close_matches
from importlib import import_module
from typing import TYPE_CHECKING

from agent_gantry.integrations.frameworks.base import DEFAULT_TOOL_LIMIT, check_query_bounds

if TYPE_CHECKING:
    from agent_gantry.core.gantry import AgentGantry
    from agent_gantry.integrations.agent_framework_adapter import AgentFrameworkAdapter
    from agent_gantry.integrations.frameworks.base import BaseFrameworkAdapter


# Public namespace modules keep adapter implementation details out of the factory.
_ADAPTERS = {
    "agent_framework": "AgentFrameworkAdapter",
    "agno": "AgnoAdapter",
    "crewai": "CrewAIAdapter",
    "dspy": "DSPyAdapter",
    "google_adk": "GoogleADKAdapter",
    "haystack": "HaystackAdapter",
    "langchain": "LangChainAdapter",
    "langgraph": "LangGraphAdapter",
    "llamaindex": "LlamaIndexAdapter",
    "openai_agents": "OpenAIAgentsAdapter",
    "pydantic_ai": "PydanticAIAdapter",
    "strands": "StrandsAdapter",
}
_ALIASES = {"crew_ai": "crewai", "llama_index": "llamaindex"}


def supported_frameworks() -> tuple[str, ...]:
    """Return canonical names accepted by :meth:`AgentGantry.framework`.

    Discovery does not require the corresponding frameworks to be installed.
    Install only the SDKs your application uses.
    """
    return tuple(_ADAPTERS)


def create_framework_adapter(
    gantry: AgentGantry,
    name: str,
    *,
    default_limit: int = DEFAULT_TOOL_LIMIT,
) -> BaseFrameworkAdapter | AgentFrameworkAdapter:
    """Construct the named adapter, leaving optional SDK imports to its use."""
    canonical = name.strip().lower().replace("-", "_")
    canonical = _ALIASES.get(canonical, canonical)
    if canonical not in _ADAPTERS:
        matches = get_close_matches(canonical, _ADAPTERS, n=1)
        suggestion = f" Did you mean {matches[0]!r}?" if matches else ""
        raise ValueError(
            f"Unsupported framework: {name!r}.{suggestion} "
            f"Supported frameworks: {', '.join(supported_frameworks())}"
        )
    check_query_bounds(limit=default_limit, score_threshold=0.0, owner="AgentGantry.framework")
    adapter_cls = getattr(import_module(f"agent_gantry.{canonical}"), _ADAPTERS[canonical])
    # Microsoft AF uses top_k for the same default selection budget.
    if canonical == "agent_framework":
        return adapter_cls(gantry, default_top_k=default_limit)
    return adapter_cls(gantry, default_limit=default_limit)


__all__ = ["supported_frameworks"]
