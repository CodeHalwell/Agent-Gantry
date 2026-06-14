"""Native per-framework tool adapters built on a shared selection core.

Each module exports ``for_<framework>(gantry, query, ...)`` (and a matching
``spec_to_<framework>`` converter) that turns Gantry-selected tools into that
framework's native tool objects. Imports of the third-party framework are lazy,
so importing this package never requires those frameworks to be installed.

Supported frameworks: LangChain, LangGraph, LlamaIndex, CrewAI, Pydantic AI,
OpenAI Agents SDK, Smolagents, Haystack, Agno, AutoGen/AG2.
"""

from __future__ import annotations

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
]
