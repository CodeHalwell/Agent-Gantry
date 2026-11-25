"""
Framework integrations for dotfetch.

One ``<Framework>Adapter`` class per agent framework (LangChain, LangGraph,
LlamaIndex, CrewAI, Pydantic AI, OpenAI Agents SDK, Haystack, Agno, Google
ADK), one ``<Provider>Adapter`` per LLM SDK
(OpenAI, Anthropic, Gemini, Groq, Vertex AI, Mistral), the
``AgentFrameworkAdapter`` for Microsoft Agent Framework, plus the shared
selection core and the framework-agnostic ``ToolRefresher`` / semantic-tools
decorator.

The adapters above all **export** dotfetch tools to a framework. For the
reverse direction — registering tools you already built with a framework
*into* dotfetch's own registry — see ``register_langchain_tools`` /
``register_crewai_tools`` / ``register_llamaindex_tools``
(:mod:`dotfetch.integrations.importers`).

Prefer the clean per-framework namespaces (``from dotfetch.langchain import
LangChainAdapter``); these aggregated re-exports are a convenience.
"""

from dotfetch.integrations.agent_framework_adapter import AgentFrameworkAdapter
from dotfetch.integrations.agent_framework_bridge import (
    DotFetchToolBridge,
    RetrievalCandidate,
    RetrievalDecision,
)
from dotfetch.integrations.agent_framework_middleware import (
    DotFetchApprovalMiddleware,
    DotFetchObservabilityMiddleware,
    DotFetchToolChoiceMiddleware,
)
from dotfetch.integrations.agent_framework_provider import (
    DotFetchContextProvider,
    MissingRequiredToolError,
)
from dotfetch.integrations.framework_adapters import fetch_framework_tools
from dotfetch.integrations.frameworks import (
    AgnoAdapter,
    CrewAIAdapter,
    DotFetchToolset,
    GoogleADKAdapter,
    HaystackAdapter,
    LangChainAdapter,
    LangGraphAdapter,
    LlamaIndexAdapter,
    OpenAIAgentsAdapter,
    PydanticAIAdapter,
    ToolExecutionError,
    ToolSpec,
)
from dotfetch.integrations.importers import (
    register_crewai_tools,
    register_langchain_tools,
    register_llamaindex_tools,
)
from dotfetch.integrations.llm_adapters import (
    AnthropicAdapter,
    GeminiAdapter,
    GroqAdapter,
    MistralAdapter,
    OpenAIAdapter,
    VertexAIAdapter,
)
from dotfetch.integrations.refresh import ToolRefresher
from dotfetch.integrations.semantic_tools import (
    SemanticToolsDecorator,
    SemanticToolSelector,
    with_semantic_tools,
)

__all__: list[str] = [
    # Microsoft Agent Framework
    "AgentFrameworkAdapter",
    "DotFetchApprovalMiddleware",
    "DotFetchContextProvider",
    "DotFetchObservabilityMiddleware",
    "DotFetchToolBridge",
    "DotFetchToolChoiceMiddleware",
    "MissingRequiredToolError",
    "RetrievalCandidate",
    "RetrievalDecision",
    # Agent framework adapters
    "LangChainAdapter",
    "LangGraphAdapter",
    "LlamaIndexAdapter",
    "CrewAIAdapter",
    "PydanticAIAdapter",
    "OpenAIAgentsAdapter",
    "HaystackAdapter",
    "AgnoAdapter",
    "GoogleADKAdapter",
    # LLM SDK adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "GroqAdapter",
    "VertexAIAdapter",
    "MistralAdapter",
    # shared selection core + utilities
    "DotFetchToolset",
    "ToolExecutionError",
    "ToolSpec",
    "ToolRefresher",
    "SemanticToolSelector",
    "SemanticToolsDecorator",
    "with_semantic_tools",
    "fetch_framework_tools",
    # reverse-direction importers: framework-native tools -> dotfetch registry
    "register_crewai_tools",
    "register_langchain_tools",
    "register_llamaindex_tools",
]
