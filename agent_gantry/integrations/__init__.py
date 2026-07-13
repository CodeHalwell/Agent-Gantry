"""
Framework integrations for Agent-Gantry.

One ``<Framework>Adapter`` class per agent framework (LangChain, LangGraph,
LlamaIndex, CrewAI, Pydantic AI, OpenAI Agents SDK, Smolagents, Haystack, Agno,
AutoGen, Semantic Kernel, Google ADK), one ``<Provider>Adapter`` per LLM SDK
(OpenAI, Anthropic, Gemini, Groq, Vertex AI, Mistral), the
``AgentFrameworkAdapter`` for Microsoft Agent Framework, plus the shared
selection core and the framework-agnostic ``ToolRefresher`` / semantic-tools
decorator.

The adapters above all **export** Gantry tools to a framework. For the
reverse direction — registering tools you already built with a framework
*into* Gantry's own registry — see ``register_langchain_tools`` /
``register_crewai_tools`` / ``register_llamaindex_tools``
(:mod:`agent_gantry.integrations.importers`).

Prefer the clean per-framework namespaces (``from agent_gantry.langchain import
LangChainAdapter``); these aggregated re-exports are a convenience.
"""

from agent_gantry.integrations.agent_framework_adapter import AgentFrameworkAdapter
from agent_gantry.integrations.agent_framework_bridge import (
    GantryToolBridge,
    RetrievalCandidate,
    RetrievalDecision,
)
from agent_gantry.integrations.agent_framework_middleware import (
    GantryApprovalMiddleware,
    GantryObservabilityMiddleware,
    GantryToolChoiceMiddleware,
)
from agent_gantry.integrations.agent_framework_provider import (
    GantryContextProvider,
    MissingRequiredToolError,
)
from agent_gantry.integrations.framework_adapters import fetch_framework_tools
from agent_gantry.integrations.frameworks import (
    AgnoAdapter,
    AutoGenAdapter,
    CrewAIAdapter,
    GantryToolset,
    GoogleADKAdapter,
    HaystackAdapter,
    LangChainAdapter,
    LangGraphAdapter,
    LlamaIndexAdapter,
    OpenAIAgentsAdapter,
    PydanticAIAdapter,
    SemanticKernelAdapter,
    SmolagentsAdapter,
    ToolExecutionError,
    ToolSpec,
)
from agent_gantry.integrations.importers import (
    register_crewai_tools,
    register_langchain_tools,
    register_llamaindex_tools,
)
from agent_gantry.integrations.llm_adapters import (
    AnthropicAdapter,
    GeminiAdapter,
    GroqAdapter,
    MistralAdapter,
    OpenAIAdapter,
    VertexAIAdapter,
)
from agent_gantry.integrations.refresh import ToolRefresher
from agent_gantry.integrations.semantic_tools import (
    SemanticToolsDecorator,
    SemanticToolSelector,
    with_semantic_tools,
)

__all__: list[str] = [
    # Microsoft Agent Framework
    "AgentFrameworkAdapter",
    "GantryApprovalMiddleware",
    "GantryContextProvider",
    "GantryObservabilityMiddleware",
    "GantryToolBridge",
    "GantryToolChoiceMiddleware",
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
    "SmolagentsAdapter",
    "HaystackAdapter",
    "AgnoAdapter",
    "AutoGenAdapter",
    "SemanticKernelAdapter",
    "GoogleADKAdapter",
    # LLM SDK adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "GroqAdapter",
    "VertexAIAdapter",
    "MistralAdapter",
    # shared selection core + utilities
    "GantryToolset",
    "ToolExecutionError",
    "ToolSpec",
    "ToolRefresher",
    "SemanticToolSelector",
    "SemanticToolsDecorator",
    "with_semantic_tools",
    "fetch_framework_tools",
    # reverse-direction importers: framework-native tools -> Gantry registry
    "register_crewai_tools",
    "register_langchain_tools",
    "register_llamaindex_tools",
]
