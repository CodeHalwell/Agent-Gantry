"""Native per-framework tool adapters built on a shared selection core.

Each ``agent_gantry.<framework>`` module exposes a single ``<Framework>Adapter``
class that selects a relevant slice of Gantry tools and converts them to that
framework's native tool objects — and, where the framework supports it, wires
deep per-turn live re-selection. Imports of the third-party framework are lazy,
so importing this package never requires those frameworks to be installed.

Supported frameworks: LangChain, LangGraph, LlamaIndex, CrewAI, Pydantic AI,
OpenAI Agents SDK, Smolagents, Haystack, Agno, AutoGen/AG2, Semantic Kernel,
Google ADK.
"""

from __future__ import annotations

from agent_gantry.integrations.frameworks.agno import AgnoAdapter
from agent_gantry.integrations.frameworks.autogen import AutoGenAdapter
from agent_gantry.integrations.frameworks.base import (
    GantryToolset,
    ToolExecutionError,
    ToolSpec,
    spec_from_tool,
)
from agent_gantry.integrations.frameworks.crewai import CrewAIAdapter
from agent_gantry.integrations.frameworks.google_adk import GoogleADKAdapter
from agent_gantry.integrations.frameworks.haystack import HaystackAdapter
from agent_gantry.integrations.frameworks.langchain import LangChainAdapter
from agent_gantry.integrations.frameworks.langgraph import LangGraphAdapter
from agent_gantry.integrations.frameworks.llamaindex import LlamaIndexAdapter
from agent_gantry.integrations.frameworks.openai_agents import OpenAIAgentsAdapter
from agent_gantry.integrations.frameworks.pydantic_ai import PydanticAIAdapter
from agent_gantry.integrations.frameworks.semantic_kernel import SemanticKernelAdapter
from agent_gantry.integrations.frameworks.smolagents import SmolagentsAdapter

__all__ = [
    # shared selection core
    "GantryToolset",
    "ToolExecutionError",
    "ToolSpec",
    "spec_from_tool",
    # per-framework adapters
    "LangChainAdapter",
    "LangGraphAdapter",
    "LlamaIndexAdapter",
    "CrewAIAdapter",
    "PydanticAIAdapter",
    "OpenAIAgentsAdapter",
    "SmolagentsAdapter",
    "HaystackAdapter",
    "AgnoAdapter",
    "AutoGenAdapter",
    "SemanticKernelAdapter",
    "GoogleADKAdapter",
]
