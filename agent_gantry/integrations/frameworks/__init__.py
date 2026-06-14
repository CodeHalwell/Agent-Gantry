"""Native per-framework tool adapters built on a shared selection core.

Each module exports ``for_<framework>(gantry, query, ...)`` (and a matching
``spec_to_<framework>`` converter) that turns Gantry-selected tools into that
framework's native tool objects. Imports of the third-party framework are lazy,
so importing this package never requires those frameworks to be installed.

Supported frameworks: LangChain, LangGraph, LlamaIndex, CrewAI, Pydantic AI,
OpenAI Agents SDK, Smolagents, Haystack, Agno, AutoGen/AG2.
"""

from __future__ import annotations

import importlib
from typing import Any

from agent_gantry.integrations.frameworks.agno import for_agno, spec_to_agno
from agent_gantry.integrations.frameworks.autogen import (
    for_autogen,
    register_with_autogen,
    spec_to_autogen,
)
from agent_gantry.integrations.frameworks.base import (
    GantryToolset,
    ToolExecutionError,
    ToolSpec,
    spec_from_tool,
)
from agent_gantry.integrations.frameworks.crewai import for_crewai, spec_to_crewai
from agent_gantry.integrations.frameworks.google_adk import (
    for_google_adk,
    spec_to_google_adk,
)
from agent_gantry.integrations.frameworks.haystack import for_haystack, spec_to_haystack
from agent_gantry.integrations.frameworks.langchain import (
    for_langchain,
    spec_to_langchain,
)
from agent_gantry.integrations.frameworks.langgraph import (
    for_langgraph,
    spec_to_langgraph,
)
from agent_gantry.integrations.frameworks.llamaindex import (
    for_llamaindex,
    spec_to_llamaindex,
)
from agent_gantry.integrations.frameworks.openai_agents import (
    for_openai_agents,
    spec_to_openai_agents,
)
from agent_gantry.integrations.frameworks.pydantic_ai import (
    for_pydantic_ai,
    spec_to_pydantic_ai,
)
from agent_gantry.integrations.frameworks.semantic_kernel import (
    for_semantic_kernel,
    gantry_plugin,
    spec_to_semantic_kernel,
)
from agent_gantry.integrations.frameworks.smolagents import (
    for_smolagents,
    spec_to_smolagents,
)

__all__ = [
    # base
    "GantryToolset",
    "ToolExecutionError",
    "ToolSpec",
    "spec_from_tool",
    # langchain / langgraph
    "for_langchain",
    "spec_to_langchain",
    "for_langgraph",
    "spec_to_langgraph",
    # llamaindex
    "for_llamaindex",
    "spec_to_llamaindex",
    # crewai
    "for_crewai",
    "spec_to_crewai",
    # pydantic-ai
    "for_pydantic_ai",
    "spec_to_pydantic_ai",
    # openai agents sdk
    "for_openai_agents",
    "spec_to_openai_agents",
    # smolagents
    "for_smolagents",
    "spec_to_smolagents",
    # haystack
    "for_haystack",
    "spec_to_haystack",
    # agno
    "for_agno",
    "spec_to_agno",
    # autogen / ag2
    "for_autogen",
    "spec_to_autogen",
    "register_with_autogen",
    # semantic kernel
    "for_semantic_kernel",
    "spec_to_semantic_kernel",
    "gantry_plugin",
    # google adk
    "for_google_adk",
    "spec_to_google_adk",
    # --- deep per-turn "live" providers (lazy; see _LIVE_EXPORTS) ---
    "gantry_tool_retriever",
    "gantry_function_agent",
    "gantry_toolset",
    "gantry_workbench",
    "gantry_before_model_callback",
    "gantry_adk_agent",
    "create_gantry_react_agent",
    "acreate_gantry_react_agent",
    "select_tools_for_state",
    "GantryFunctionProvider",
    "refresh_kernel_tools",
    "gantry_run_hooks",
    "run_with_gantry",
    "GantryAgentSession",
    "refresh_agent_tools",
    "select_function_tools",
    "gantry_crew_tools",
    "GantryLiveCrewAgent",
    "GantryLiveAgnoAgent",
    "gantry_haystack_tools",
    "GantryLiveHaystackToolInvoker",
    "GantryLiveSmolAgent",
]

# Deep per-turn ("live") providers re-select tools on every turn via each
# framework's native dynamic-tool hook. Their modules build framework subclasses
# at import time, so they are loaded LAZILY here — importing `agent_gantry` (or
# this package) never pulls in LlamaIndex / Pydantic AI / AutoGen / etc. Access
# triggers the import (and a clear ImportError if the framework is missing).
_LIVE_EXPORTS: dict[str, str] = {
    "gantry_tool_retriever": "llamaindex_live",
    "gantry_function_agent": "llamaindex_live",
    "gantry_toolset": "pydantic_ai_live",
    "gantry_workbench": "autogen_live",
    "gantry_before_model_callback": "google_adk_live",
    "gantry_adk_agent": "google_adk_live",
    "create_gantry_react_agent": "langgraph_live",
    "acreate_gantry_react_agent": "langgraph_live",
    "select_tools_for_state": "langgraph_live",
    "GantryFunctionProvider": "semantic_kernel_live",
    "refresh_kernel_tools": "semantic_kernel_live",
    "gantry_run_hooks": "openai_agents_live",
    "run_with_gantry": "openai_agents_live",
    "GantryAgentSession": "openai_agents_live",
    "refresh_agent_tools": "openai_agents_live",
    "select_function_tools": "openai_agents_live",
    "gantry_crew_tools": "live_wrappers",
    "GantryLiveCrewAgent": "live_wrappers",
    "GantryLiveAgnoAgent": "live_wrappers",
    "gantry_haystack_tools": "live_wrappers",
    "GantryLiveHaystackToolInvoker": "live_wrappers",
    "GantryLiveSmolAgent": "live_wrappers",
}


def __getattr__(name: str) -> Any:  # noqa: D401 - module-level lazy attribute access
    """Lazily resolve deep per-turn provider symbols from their submodules."""
    module = _LIVE_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod = importlib.import_module(f"{__name__}.{module}")
    return getattr(mod, name)
