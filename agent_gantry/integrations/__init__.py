"""
Framework integrations for Agent-Gantry.

Integrations with Microsoft Agent Framework, LangChain, LangGraph,
Google ADK, Pydantic AI, CrewAI, and Strands.
"""

from agent_gantry.integrations.agent_framework_bridge import (
    GantryToolBridge as AgentFrameworkToolBridge,
)
from agent_gantry.integrations.agent_framework_middleware import (
    GantryApprovalMiddleware,
    GantryObservabilityMiddleware,
)
from agent_gantry.integrations.framework_adapters import fetch_framework_tools
from agent_gantry.integrations.langchain_bridge import (
    GantryToolBridge as LangChainToolBridge,
)
from agent_gantry.integrations.semantic_tools import (
    SemanticToolsDecorator,
    SemanticToolSelector,
    with_semantic_tools,
)

# Back-compat: ``GantryToolBridge`` historically referred to the Microsoft
# Agent Framework bridge. Keep the alias so existing imports continue to work
# while new framework-specific bridges land alongside it.
GantryToolBridge = AgentFrameworkToolBridge

__all__: list[str] = [
    "AgentFrameworkToolBridge",
    "GantryApprovalMiddleware",
    "GantryObservabilityMiddleware",
    "GantryToolBridge",
    "LangChainToolBridge",
    "SemanticToolSelector",
    "SemanticToolsDecorator",
    "fetch_framework_tools",
    "with_semantic_tools",
]
