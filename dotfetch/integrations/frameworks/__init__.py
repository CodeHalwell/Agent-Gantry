"""Native per-framework tool adapters built on a shared selection core.

Each ``dotfetch.<framework>`` module exposes a single ``<Framework>Adapter``
class that selects a relevant slice of dotfetch tools and converts them to that
framework's native tool objects — and, where the framework supports it, wires
deep per-turn live re-selection. Imports of the third-party framework are lazy,
so importing this package never requires those frameworks to be installed.

Supported frameworks: LangChain, LangGraph, LlamaIndex, CrewAI, Pydantic AI,
OpenAI Agents SDK, Haystack, Agno,
Google ADK, Strands Agents, DSPy.
"""

from __future__ import annotations

from dotfetch.integrations.frameworks.agno import AgnoAdapter
from dotfetch.integrations.frameworks.base import (
    DotFetchToolset,
    QueryBoundsError,
    ToolConfirmationRequiredError,
    ToolExecutionError,
    ToolPermissionDeniedError,
    ToolSpec,
    spec_from_tool,
)
from dotfetch.integrations.frameworks.crewai import CrewAIAdapter
from dotfetch.integrations.frameworks.dspy import DSPyAdapter
from dotfetch.integrations.frameworks.errors import MissingRequiredToolError
from dotfetch.integrations.frameworks.google_adk import GoogleADKAdapter
from dotfetch.integrations.frameworks.haystack import HaystackAdapter
from dotfetch.integrations.frameworks.langchain import LangChainAdapter
from dotfetch.integrations.frameworks.langgraph import LangGraphAdapter
from dotfetch.integrations.frameworks.llamaindex import LlamaIndexAdapter
from dotfetch.integrations.frameworks.openai_agents import OpenAIAgentsAdapter
from dotfetch.integrations.frameworks.pydantic_ai import PydanticAIAdapter
from dotfetch.integrations.frameworks.strands import StrandsAdapter

__all__ = [
    # shared selection core
    "DotFetchToolset",
    "MissingRequiredToolError",
    "QueryBoundsError",
    "ToolConfirmationRequiredError",
    "ToolExecutionError",
    "ToolPermissionDeniedError",
    "ToolSpec",
    "spec_from_tool",
    # per-framework adapters
    "LangChainAdapter",
    "LangGraphAdapter",
    "LlamaIndexAdapter",
    "CrewAIAdapter",
    "PydanticAIAdapter",
    "OpenAIAgentsAdapter",
    "HaystackAdapter",
    "AgnoAdapter",
    "GoogleADKAdapter",
    "StrandsAdapter",
    "DSPyAdapter",
]
