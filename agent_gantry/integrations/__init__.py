"""
Framework integrations for Agent-Gantry.

Integrations with LangChain, AutoGen, LlamaIndex, CrewAI,
Microsoft Agent Framework, etc.
"""

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
    GantryToolset,
    ToolExecutionError,
    ToolSpec,
)
from agent_gantry.integrations.refresh import ToolRefresher
from agent_gantry.integrations.semantic_tools import (
    SemanticToolsDecorator,
    SemanticToolSelector,
    with_semantic_tools,
)

__all__: list[str] = [
    "GantryApprovalMiddleware",
    "GantryContextProvider",
    "GantryObservabilityMiddleware",
    "GantryToolBridge",
    "GantryToolChoiceMiddleware",
    "GantryToolset",
    "MissingRequiredToolError",
    "RetrievalCandidate",
    "RetrievalDecision",
    "SemanticToolSelector",
    "SemanticToolsDecorator",
    "ToolExecutionError",
    "ToolRefresher",
    "ToolSpec",
    "with_semantic_tools",
    "fetch_framework_tools",
]
